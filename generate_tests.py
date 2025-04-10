import asyncio
import json
import random
import os
from datetime import datetime
from pathlib import Path
import aiohttp
import warnings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from os import getenv
import sys
import traceback

load_dotenv()

SOY_TOKEN = getenv('SOY_TOKEN')
RESULTS_DIR = "test_data"
DATASETS_DIR = os.path.join(RESULTS_DIR, "datasets")


def ensure_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)


def save_dataset(dataset, name):
    file_path = os.path.join(DATASETS_DIR, f"{name}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Датасет сохранён в {file_path}")
    return file_path


def load_dataset(name):
    file_path = os.path.join(DATASETS_DIR, f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


async def generate_museum_questions(count=100):
    """Генерирует вопросы о музейных объектах"""
    prompt = f"""
    Создай {count} разнообразных вопросов на русском языке о конкретных достопримечательностях музея-заповедника "Петергоф".
    Вопросы должны касаться фонтанов, павильонов, дворцов, экспонатов и других объектов.

    НЕ включай вопросы о режиме работы, билетах или расписаниях.
    НЕ используй нумерацию (1., 2. и т.д.), каждый вопрос должен начинаться с новой строки.

    Примеры правильных вопросов:
    - "Что такое Большой каскад в Петергофе?"
    - "Расскажи историю павильона Эрмитаж"
    - "Какие статуи находятся возле фонтана Самсон?"
    - "Опиши скульптуры на Аллее фонтанов"
    - "Какие экспонаты есть в Банном корпусе?"
    - "Что представляет собой фонтан Пирамида?"
    - "Какова история Верхнего сада Петергофа?"

    Вопросы должны быть разными и интересными, охватывать различные аспекты и объекты музея-заповедника.
    """

    messages = [{"role": "user", "content": prompt}]
    response = await get_chatgpt_response(messages)

    if not response:
        print("Не удалось сгенерировать вопросы о музее")
        return []

    questions = [q.strip() for q in response.split('\n') if q.strip() and not q.startswith('-')]

    questions = [q[q.find(' ') + 1:] if q[0].isdigit() and q[1] in '.) ' else q for q in questions]
    questions = [q[2:] if q.startswith('- ') else q for q in questions]

    return questions[:count]


async def generate_general_questions(count=100):
    """Генерирует общие вопросы о музее"""
    prompt = f"""
    Создай {count} разнообразных общих вопросов на русском языке о музее-заповеднике "Петергоф".
    Включи вопросы о режиме работы, билетах, расписании, правилах посещения, инфраструктуре и новостях.

    НЕ используй нумерацию (1., 2. и т.д.), каждый вопрос должен начинаться с новой строки.

    Примеры правильных вопросов:
    - "Когда открываются фонтаны в Петергофе?"
    - "Сколько стоит билет в Нижний парк?"
    - "Можно ли приходить с домашними животными?"
    - "В какое время закрывается музей по выходным?"
    - "Есть ли в музее кафе или рестораны?"
    - "Как добраться до Петергофа из центра Санкт-Петербурга?"
    - "Какие мероприятия проводятся в музее на этой неделе?"
    - "Работает ли музей в понедельник?"
    - "Существуют ли льготы для пенсионеров и детей?"

    Вопросы должны быть разнообразными и охватывать все аспекты функционирования музея.
    """

    messages = [{"role": "user", "content": prompt}]
    response = await get_chatgpt_response(messages)

    if not response:
        print("Не удалось сгенерировать общие вопросы")
        return []

    questions = [q.strip() for q in response.split('\n') if q.strip() and not q.startswith('-')]

    questions = [q[q.find(' ') + 1:] if q[0].isdigit() and q[1] in '.) ' else q for q in questions]
    questions = [q[2:] if q.startswith('- ') else q for q in questions]

    return questions[:count]


async def generate_new_route_questions(count=50):
    prompt = f"""
    Создай {count} разнообразных вопросов на русском языке о создании нового маршрута по музею-заповеднику "Петергоф".
    Это должны быть запросы на ПЕРВИЧНОЕ создание маршрута, а не модификацию существующего.

    НЕ используй нумерацию (1., 2. и т.д.), каждый вопрос должен начинаться с новой строки.

    Примеры правильных вопросов:
    - "Помоги составить маршрут по парку"
    - "Как лучше обойти все основные фонтаны за 3 часа?"
    - "Предложи маршрут для посещения дворцов Петергофа"
    - "Какой маршрут посоветуешь для первого посещения?"
    - "Составь для меня план прогулки по парку с детьми"
    - "У меня есть всего 2 часа, какие объекты стоит посмотреть?"
    - "Составь маршрут посещения тенистых мест парка в жаркий день"
    - "Я интересуюсь скульптурами, предложи маршрут с их осмотром"
    - "Мы с семьей хотим увидеть основные достопримечательности, помоги составить маршрут"

    Используй разнообразие запросов, разные временные ограничения, интересы и предпочтения посетителей.
    """

    messages = [{"role": "user", "content": prompt}]
    response = await get_chatgpt_response(messages)

    if not response:
        print("Не удалось сгенерировать вопросы о новых маршрутах")
        return []

    questions = [q.strip() for q in response.split('\n') if q.strip() and not q.startswith('-')]

    questions = [q[q.find(' ') + 1:] if q[0].isdigit() and q[1] in '.) ' else q for q in questions]
    questions = [q[2:] if q.startswith('- ') else q for q in questions]

    return questions[:count]


async def generate_modify_route_questions(count=50):
    prompt = f"""
    Создай {count} разнообразных вопросов на русском языке для ИЗМЕНЕНИЯ уже существующего маршрута по музею-заповеднику "Петергоф".
    Это должны быть запросы на модификацию ранее составленного маршрута, а не на создание нового.

    НЕ используй нумерацию (1., 2. и т.д.), каждый вопрос должен начинаться с новой строки.

    Примеры правильных вопросов:
    - "Добавь в маршрут фонтан Самсон"
    - "Давай уберем из маршрута Большой дворец, у нас мало времени"
    - "Хочу изменить маршрут, чтобы включить больше фонтанов"
    - "Можно ли исключить из маршрута объекты, находящиеся на большом расстоянии друг от друга?"
    - "Увеличь время на осмотр Монплезира в маршруте"
    - "Добавь в конце маршрута время на отдых в кафе"
    - "Сократи маршрут до 2 часов вместо 3"
    - "Переставь местами в маршруте Эрмитаж и Марли, так удобнее"
    - "Скорректируй маршрут, чтобы начать с западной части парка"
    - "Замени в маршруте музей на прогулку по Верхнему саду"

    Используй разнообразные варианты модификаций: добавление и удаление объектов, изменение порядка посещения, 
    корректировка времени, изменение фокуса маршрута.
    """

    messages = [{"role": "user", "content": prompt}]
    response = await get_chatgpt_response(messages)

    if not response:
        print("Не удалось сгенерировать вопросы об изменении маршрутов")
        return []

    questions = [q.strip() for q in response.split('\n') if q.strip() and not q.startswith('-')]

    questions = [q[q.find(' ') + 1:] if q[0].isdigit() and q[1] in '.) ' else q for q in questions]
    questions = [q[2:] if q.startswith('- ') else q for q in questions]

    return questions[:count]


async def generate_dialog_histories(count=30):
    prompt = f"""
    Создай {count} коротких диалогов между пользователем и ботом музея-заповедника "Петергоф".
    Каждый диалог должен содержать 2-3 реплики (вопрос-ответ) и выглядеть так:

    user: [вопрос пользователя]
    bot: [ответ бота]
    user: [следующий вопрос пользователя]
    bot: [ответ бота]

    Диалоги должны быть разнообразными и охватывать разные темы:
    - Вопросы о конкретных объектах музея
    - Вопросы о режиме работы и билетах
    - Вопросы о маршрутах и навигации

    Например:

    user: Когда работает Большой дворец?
    bot: Большой дворец открыт ежедневно с 10:30 до 19:00, кроме понедельника.
    user: А где он находится?
    bot: Большой дворец расположен в центре архитектурной композиции Верхнего сада и Нижнего парка Петергофа.

    Разделяй диалоги пустой строкой.
    """

    messages = [{"role": "user", "content": prompt}]
    response = await get_chatgpt_response(messages)

    if not response:
        print("Не удалось сгенерировать диалоговые истории")
        return []

    raw_dialogs = response.split('\n\n')

    histories = []
    for dialog in raw_dialogs:
        if dialog.strip():
            histories.append(dialog.strip())

    return histories[:count]


async def generate_all_datasets():
    ensure_directories()

    museum_qs = load_dataset("museum_questions")
    general_qs = load_dataset("general_questions")
    new_route_qs = load_dataset("new_route_questions")
    modify_route_qs = load_dataset("modify_route_questions")
    dialog_histories = load_dataset("dialog_histories")

    tasks = []

    if not museum_qs:
        print("Генерация вопросов о музейных объектах...")
        museum_qs = await generate_museum_questions(100)
        save_dataset(museum_qs, "museum_questions")
    else:
        print("Загружены существующие вопросы о музейных объектах")

    if not general_qs:
        print("Генерация общих вопросов о музее...")
        general_qs = await generate_general_questions(100)
        save_dataset(general_qs, "general_questions")
    else:
        print("Загружены существующие общие вопросы")

    if not new_route_qs:
        print("Генерация вопросов о создании новых маршрутов...")
        new_route_qs = await generate_new_route_questions(50)
        save_dataset(new_route_qs, "new_route_questions")
    else:
        print("Загружены существующие вопросы о создании маршрутов")

    if not modify_route_qs:
        print("Генерация вопросов об изменении маршрутов...")
        modify_route_qs = await generate_modify_route_questions(50)
        save_dataset(modify_route_qs, "modify_route_questions")
    else:
        print("Загружены существующие вопросы об изменении маршрутов")

    if not dialog_histories:
        print("Генерация диалоговых историй...")
        dialog_histories = await generate_dialog_histories(30)
        save_dataset(dialog_histories, "dialog_histories")
    else:
        print("Загружены существующие диалоговые истории")

    full_dataset = []

    for q in museum_qs:
        history = ""
        if random.random() < 0.3 and dialog_histories:
            history = random.choice(dialog_histories)

        full_dataset.append({
            "question": q,
            "history": history,
            "expected_type": "museum"
        })

    for q in general_qs:
        history = ""
        if random.random() < 0.3 and dialog_histories:
            history = random.choice(dialog_histories)

        full_dataset.append({
            "question": q,
            "history": history,
            "expected_type": "general"
        })

    for q in new_route_qs:
        history = ""
        if random.random() < 0.3 and dialog_histories:
            history = random.choice(dialog_histories)

        full_dataset.append({
            "question": q,
            "history": history,
            "expected_type": "route",
            "route_type": "new"
        })

    for q in modify_route_qs:
        history = ""
        if random.random() < 0.3 and dialog_histories:
            history = random.choice(dialog_histories)

        full_dataset.append({
            "question": q,
            "history": history,
            "expected_type": "route",
            "route_type": "modify"
        })

    random.shuffle(full_dataset)

    save_dataset(full_dataset, "full_test_dataset")

    small_dataset = random.sample(full_dataset, min(30, len(full_dataset)))
    save_dataset(small_dataset, "small_test_dataset")

    print(f"Сгенерировано всего {len(full_dataset)} тестовых случаев")
    print(f"Создан малый тестовый набор из {len(small_dataset)} случаев")

    return full_dataset


async def main():
    print("Начинаю генерацию тестовых данных для бота Петергофа...")
    try:
        await generate_all_datasets()
        print("Генерация данных завершена успешно!")
    except Exception as e:
        print(f"Ошибка при генерации данных: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
