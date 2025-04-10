import asyncio
import json
import random
import os
from datetime import datetime
from pathlib import Path
import aiohttp
import warnings
import pandas as pd
from typing import List, Dict, Any, Optional
from yandex_cloud_ml_sdk import YCloudML
from dotenv import load_dotenv
from os import getenv
import sys
import traceback

load_dotenv()

sys.path.append('.')
try:
    from validate import (
        judge_classification,
        evaluate_answer_news,
        evaluate_contextual_answer_usefulness
    )
except ImportError:
    print("ВНИМАНИЕ: Не удалось импортировать функции валидации из validate.py")
    print("Проверьте, что файл validate.py находится в текущей директории")
    sys.exit(1)

try:
    from dialog_pipeline import classify_question_type, answer_from_news
    from chroma import init_chroma
    from rout_suggestion import get_route_suggestion
    from utils import create_json_chunks
except ImportError:
    print("ВНИМАНИЕ: Не удалось импортировать одну или несколько функций из вашего кода")
    print("Проверьте, что все необходимые файлы находятся в текущей директории")
    sys.exit(1)

# Constants
SOY_TOKEN = getenv('SOY_TOKEN')
RESULTS_DIR = "test_results"
DATASETS_DIR = "test_data/datasets"
TEST_SIZE_DEFAULT = 30  # По умолчанию тестируем 30 диалогов
DETAILED_LOGS = True

def ensure_directories():
    """Создает необходимые директории для хранения результатов"""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_dataset(name):
    """Загружает датасет из JSON-файла"""
    file_path = os.path.join(DATASETS_DIR, f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


async def test_classification(dialog: Dict, llm_model) -> Dict:
    """Тестирует функцию классификации и валидирует результат"""
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "classification": {
            "predicted_type": "",
            "validation": "",
            "is_correct": False
        }
    }

    try:
        # Получаем классификацию от нашей системы
        predicted_type = classify_question_type(dialog["question"], dialog["history"])
        result["classification"]["predicted_type"] = predicted_type

        # Валидируем классификацию
        validation = await judge_classification(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            classifier_label=predicted_type
        )
        result["classification"]["validation"] = validation
        result["classification"]["is_correct"] = validation == "Верно"
    except Exception as e:
        traceback.print_exc()
        result["classification"]["error"] = str(e)

    return result


async def test_news_answer(dialog: Dict, llm_model, news_data: Dict, tickets_data: Dict) -> Dict:
    """Тестирует функцию ответа на основе новостей и оценивает результат"""
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "news_answer": {
            "bot_answer": "",
            "evaluation": "",
            "rating": None
        }
    }

    try:
        # Получаем ответ от функции ответа на основе новостей
        bot_answer = answer_from_news(
            dialog["question"],
            dialog["history"],
            greeting_style="none"  # Без приветствий для более чистой оценки
        )
        result["news_answer"]["bot_answer"] = bot_answer

        # Оцениваем ответ
        evaluation = await evaluate_answer_news(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            bot_answer=bot_answer,
            news_data=news_data,
            tickets_data=tickets_data
        )
        result["news_answer"]["evaluation"] = evaluation

        # Пытаемся извлечь числовую оценку
        try:
            result["news_answer"]["rating"] = int(evaluation) if evaluation.isdigit() else None
        except:
            pass
    except Exception as e:
        traceback.print_exc()
        result["news_answer"]["error"] = str(e)

    return result


async def test_rag_answer(dialog: Dict, llm_model, chroma_collection) -> Dict:
    """Тестирует функцию RAG-ответа и оценивает результат"""
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "rag_answer": {
            "bot_answer": "",
            "context": "",
            "evaluation": "",
            "rating": None
        }
    }

    try:
        # Получаем контекст из Chroma
        query_results = chroma_collection.query(
            query_texts=[dialog["question"]],
            n_results=5
        )
        retrieved_docs = query_results.get('documents', [[]])[0]
        context = "\n\n".join(retrieved_docs)
        result["rag_answer"]["context"] = context

        # Генерируем ответ на основе контекста
        prompt = f"""
        Ты - виртуальный помощник по музею-заповеднику Петергоф.

        Вопрос пользователя: "{dialog["question"]}"

        Релевантный контекст для ответа:
        {context}

        Дай краткий и точный ответ на основе предоставленного контекста.
        """
        llm_result = llm_model.run(prompt)
        bot_answer = llm_result.alternatives[0].text.strip()
        result["rag_answer"]["bot_answer"] = bot_answer

        # Оцениваем ответ
        evaluation = await evaluate_contextual_answer_usefulness(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            relevant_context=context,
            bot_answer=bot_answer
        )
        result["rag_answer"]["evaluation"] = evaluation

        # Пытаемся извлечь числовую оценку
        try:
            result["rag_answer"]["rating"] = int(evaluation) if evaluation.isdigit() else None
        except:
            pass
    except Exception as e:
        traceback.print_exc()
        result["rag_answer"]["error"] = str(e)

    return result


async def test_route_answer(dialog: Dict, llm_model) -> Dict:
    """Тестирует функцию генерации маршрута и оценивает результат"""
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "route_type": dialog.get("route_type", "new"),  # new или modify
        "route_answer": {
            "bot_answer": "",
            "evaluation": "",
            "rating": None
        }
    }

    try:
        # Подготавливаем входные данные для функции маршрута
        data_chunks = create_json_chunks()
        coordinates = ['59.891802', '29.913220']  # Координаты по умолчанию

        # Создаем формат диалогов, ожидаемый функцией route_suggestion
        user_dialogues = []

        # Пытаемся разобрать историю на сообщения пользователя/бота
        if dialog["history"]:
            history_lines = dialog["history"].split('\n')
            for line in history_lines:
                if line.lower().startswith(('user:', 'пользователь:')):
                    user_dialogues.append({"user": line.split(':', 1)[1].strip()})
                elif line.lower().startswith(('bot:', 'бот:')):
                    user_dialogues.append({"bot": line.split(':', 1)[1].strip()})

        # Добавляем текущий вопрос
        user_dialogues.append({"user": dialog["question"]})

        # Получаем предложение маршрута или изменяем существующий
        if result["route_type"] == "new" or not user_dialogues:
            answer_text, route_json = get_route_suggestion(
                user_dialogues,
                data_chunks,
                initial_coordinates=coordinates
            )
        else:
            # Симулируем существующий маршрут для тестирования изменения
            # Используем первый чанк JSON как "существующий маршрут"
            existing_route = data_chunks[0] if data_chunks else {}

            # Используем функцию изменения маршрута
            from rout_suggestion import change_route_by_message
            answer_text, route_json = change_route_by_message(
                dialog["question"],
                existing_route,
                data_chunks,
                user_dialogues,
                initial_coordinates=coordinates
            )

        bot_answer = answer_text.replace('*', '')
        result["route_answer"]["bot_answer"] = bot_answer

        # Конвертируем чанки данных в строку для контекста
        route_context = json.dumps(data_chunks[:2], ensure_ascii=False)  # Ограничиваем размер контекста

        # Оцениваем ответ маршрута
        evaluation = await evaluate_contextual_answer_usefulness(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            relevant_context=route_context,
            bot_answer=bot_answer
        )
        result["route_answer"]["evaluation"] = evaluation

        # Пытаемся извлечь числовую оценку
        try:
            result["route_answer"]["rating"] = int(evaluation) if evaluation.isdigit() else None
        except:
            pass
    except Exception as e:
        traceback.print_exc()
        result["route_answer"]["error"] = str(e)

    return result


async def run_tests(dataset_name="small_test_dataset", test_size=None) -> Dict:
    """Запускает все тесты и возвращает результаты"""
    # Создаем директорию для результатов
    ensure_directories()

    # Инициализируем YandexGPT
    sdk = YCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))
    llm_model = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    llm_model = llm_model.configure(temperature=0.1)

    # Инициализируем Chroma
    print("Инициализация коллекции Chroma...")
    chroma_collection = init_chroma(remote=True)

    # Загружаем данные новостей и билетов
    print("Загрузка данных...")
    try:
        with open('news.json', 'r', encoding='utf-8') as f:
            news_data = json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки данных новостей: {e}")
        news_data = {"news": [], "number": 1}

    try:
        with open('tickets.json', 'r', encoding='utf-8') as f:
            tickets_data = json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки данных билетов: {e}")
        tickets_data = {"data": []}

    # Загружаем тестовый датасет
    dialogs = load_dataset(dataset_name)
    if not dialogs:
        print(f"Ошибка: датасет {dataset_name} не найден")
        return {"error": f"Датасет {dataset_name} не найден"}

    # Ограничиваем размер тестовой выборки, если задано
    if test_size:
        dialogs = dialogs[:test_size]

    print(f"Запуск тестирования на {len(dialogs)} диалогах...")

    # Запускаем тесты
    all_results = []
    for i, dialog in enumerate(dialogs):
        print(f"Тестирование диалога {i + 1}/{len(dialogs)}: {dialog['question'][:50]}...")

        dialog_result = {
            "id": i + 1,
            "question": dialog["question"],
            "history": dialog["history"],
            "expected_type": dialog["expected_type"]
        }

        if "route_type" in dialog:
            dialog_result["route_type"] = dialog["route_type"]

        # Тестируем классификацию
        classification_result = await test_classification(dialog, llm_model)
        dialog_result["classification"] = classification_result["classification"]

        # Тестируем соответствующую функцию ответа в зависимости от ожидаемого типа
        if dialog["expected_type"] == "general":
            news_result = await test_news_answer(dialog, llm_model, news_data, tickets_data)
            dialog_result["answer_test"] = news_result["news_answer"]
            dialog_result["answer_type"] = "news"
        elif dialog["expected_type"] == "museum":
            rag_result = await test_rag_answer(dialog, llm_model, chroma_collection)
            dialog_result["answer_test"] = rag_result["rag_answer"]
            dialog_result["answer_type"] = "rag"
        elif dialog["expected_type"] == "route":
            route_result = await test_route_answer(dialog, llm_model)
            dialog_result["answer_test"] = route_result["route_answer"]
            dialog_result["answer_type"] = "route"

        all_results.append(dialog_result)

        # Сохраняем подробный лог, если включено
        if DETAILED_LOGS:
            log_file = os.path.join(RESULTS_DIR, f"dialog_{i + 1}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(dialog_result, f, ensure_ascii=False, indent=2)

        # Небольшая задержка, чтобы не перегружать API
        await asyncio.sleep(1)

    return all_results


def analyze_results(results: List[Dict]) -> Dict:
    """Анализирует результаты тестов и генерирует сводную статистику"""
    total_tests = len(results)

    # Статистика классификации
    classification_correct = sum(1 for r in results if r.get("classification", {}).get("is_correct", False))
    classification_accuracy = (classification_correct / total_tests) * 100 if total_tests > 0 else 0

    # Статистика качества ответов
    answer_ratings = [r.get("answer_test", {}).get("rating") for r in results]
    valid_ratings = [r for r in answer_ratings if r is not None]

    answer_stats = {
        "rating_0": sum(1 for r in valid_ratings if r == 0),
        "rating_1": sum(1 for r in valid_ratings if r == 1),
        "rating_2": sum(1 for r in valid_ratings if r == 2),
        "average": sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0
    }

    # Статистика по типам
    type_stats = {}
    for q_type in ["museum", "route", "general"]:
        type_results = [r for r in results if r.get("expected_type") == q_type]
        if not type_results:
            continue

        type_correct = sum(1 for r in type_results if r.get("classification", {}).get("is_correct", False))
        type_accuracy = (type_correct / len(type_results)) * 100 if type_results else 0

        type_ratings = [r.get("answer_test", {}).get("rating") for r in type_results]
        valid_type_ratings = [r for r in type_ratings if r is not None]

        type_stats[q_type] = {
            "count": len(type_results),
            "classification_accuracy": type_accuracy,
            "answer_quality": {
                "rating_0": sum(1 for r in valid_type_ratings if r == 0),
                "rating_1": sum(1 for r in valid_type_ratings if r == 1),
                "rating_2": sum(1 for r in valid_type_ratings if r == 2),
                "average": sum(valid_type_ratings) / len(valid_type_ratings) if valid_type_ratings else 0
            }
        }

        # Для маршрутов добавляем статистику по подтипам
        if q_type == "route":
            # Статистика по новым и модифицируемым маршрутам
            route_subtypes = {}
            for subtype in ["new", "modify"]:
                subtype_results = [r for r in type_results if r.get("route_type") == subtype]
                if not subtype_results:
                    continue

                subtype_correct = sum(
                    1 for r in subtype_results if r.get("classification", {}).get("is_correct", False))
                subtype_accuracy = (subtype_correct / len(subtype_results)) * 100 if subtype_results else 0

                subtype_ratings = [r.get("answer_test", {}).get("rating") for r in subtype_results]
                valid_subtype_ratings = [r for r in subtype_ratings if r is not None]

                route_subtypes[subtype] = {
                    "count": len(subtype_results),
                    "classification_accuracy": subtype_accuracy,
                    "answer_quality": {
                        "rating_0": sum(1 for r in valid_subtype_ratings if r == 0),
                        "rating_1": sum(1 for r in valid_subtype_ratings if r == 1),
                        "rating_2": sum(1 for r in valid_subtype_ratings if r == 2),
                        "average": sum(valid_subtype_ratings) / len(
                            valid_subtype_ratings) if valid_subtype_ratings else 0
                    }
                }

            if route_subtypes:
                type_stats[q_type]["subtypes"] = route_subtypes

    # Сводная статистика
    return {
        "total_tests": total_tests,
        "classification": {
            "correct": classification_correct,
            "accuracy": classification_accuracy
        },
        "answer_quality": answer_stats,
        "by_type": type_stats,
        "timestamp": datetime.now().isoformat()
    }


async def main():
    """Основная функция тестирования"""
    import argparse

    parser = argparse.ArgumentParser(description='Тестирование бота Петергофа')
    parser.add_argument('--dataset', type=str, default='full_test_dataset',
                        help='Имя датасета для тестирования (по умолчанию: small_test_dataset)')
    parser.add_argument('--size', type=int, default=None,
                        help=f'Количество тестов для запуска (по умолчанию: все из датасета)')

    args = parser.parse_args()

    print(f"Запуск тестирования на датасете {args.dataset}" +
          (f" с ограничением {args.size} тестов" if args.size else ""))

    start_time = datetime.now()

    try:
        # Запускаем тесты
        results = await run_tests(args.dataset, args.size)

        # Анализируем результаты
        stats = analyze_results(results)

        # Сохраняем полные результаты и статистику
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"all_results_{timestamp}.json")
        stats_file = os.path.join(RESULTS_DIR, f"stats_{timestamp}.json")

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # Создаем CSV для упрощения анализа
        df = pd.DataFrame(results)
        df['classification_correct'] = df['classification'].apply(lambda x: x.get('is_correct', False))
        df['answer_rating'] = df['answer_test'].apply(lambda x: x.get('rating', None))
        df_simple = df[['id', 'question', 'expected_type', 'answer_type', 'classification_correct', 'answer_rating']]
        csv_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
        df_simple.to_csv(csv_file, index=False, encoding='utf-8')

        # Выводим сводку
        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print("СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
        print("=" * 60)
        print(f"Всего тестов: {stats['total_tests']}")
        print(f"Точность классификации: {stats['classification']['accuracy']:.2f}%")
        print(f"Средняя оценка качества ответов: {stats['answer_quality']['average']:.2f}/2")
        print(f"Распределение оценок: 0: {stats['answer_quality']['rating_0']}, " +
              f"1: {stats['answer_quality']['rating_1']}, 2: {stats['answer_quality']['rating_2']}")

        print("\nРезультаты по типам вопросов:")
        for q_type, type_stat in stats["by_type"].items():
            print(f"\n{q_type.upper()} ({type_stat['count']} тестов):")
            print(f"- Точность классификации: {type_stat['classification_accuracy']:.2f}%")
            print(f"- Средняя оценка качества: {type_stat['answer_quality']['average']:.2f}/2")
            print(f"- Оценки: 0: {type_stat['answer_quality']['rating_0']}, " +
                  f"1: {type_stat['answer_quality']['rating_1']}, 2: {type_stat['answer_quality']['rating_2']}")

            # Для маршрутов выводим статистику по подтипам
            if q_type == "route" and "subtypes" in type_stat:
                for subtype, subtype_stat in type_stat["subtypes"].items():
                    print(f"\n  {subtype.upper()} ({subtype_stat['count']} тестов):")
                    print(f"  - Точность классификации: {subtype_stat['classification_accuracy']:.2f}%")
                    print(f"  - Средняя оценка качества: {subtype_stat['answer_quality']['average']:.2f}/2")
                    print(f"  - Оценки: 0: {subtype_stat['answer_quality']['rating_0']}, " +
                          f"1: {subtype_stat['answer_quality']['rating_1']}, 2: {subtype_stat['answer_quality']['rating_2']}")

        print("\n" + "=" * 60)
        print(f"Тестирование завершено за {elapsed:.2f} секунд")
        print(f"Полные результаты сохранены в {results_file}")
        print(f"Статистика сохранена в {stats_file}")
        print(f"CSV-сводка сохранена в {csv_file}")

    except Exception as e:
        print(f"Ошибка при выполнении тестирования: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
