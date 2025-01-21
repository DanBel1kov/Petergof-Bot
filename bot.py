import asyncio
import json
from pathlib import Path
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup,
                           KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove)
from dotenv import load_dotenv
from os import getenv, remove

# --- Импорт из Yandex Cloud ML SDK и "chroma" для векторного поиска ---
from yandex_cloud_ml_sdk import YCloudML
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

import chromadb
from chromadb.config import Settings

# -------------------- Конфигурация и инициализация --------------------

load_dotenv()  # если у вас есть .env с секретами

# Создаём экземпляр бота Aiogram
BOT_TOKEN = getenv('TOKEN')  # ваш Telegram-бот токен
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())

# Храним настройки, тексты, состояния и т.д.
bot.user_settings = {}
bot.texts = {}
bot.states = {}

# Настраиваем Яндекс Cloud ML
# В реальном проекте вместо getenv() укажите свои значения
YANDEX_FOLDER_ID = getenv('FOLDER_ID', 'b1g10f66fjjfuqg9ehje')
YANDEX_AUTH = getenv('YANDEX_AUTH', 'AQVN0zMfZzvnaQ_qeJz4mtiu3yYeTKJe2aupo1z5')

sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_AUTH)

embd_model = sdk.models.text_embeddings("doc")
llm_model = sdk.models.completions("yandexgpt")

client = chromadb.PersistentClient(path="chroma_db")

# Название коллекции для документов в Chroma
COLLECTION_NAME = "peterhof_docs"
bot.chroma_collection = None

CIS_COUNTRIES = ['ru', 'ua', 'by', 'kz', 'kg', 'am', 'uz', 'tj', 'az', 'md']

admin_chat = -1002411793280

# -------------------- Функции для работы с векторами и Chroma --------------------

from chromadb.api import EmbeddingFunction
from typing import List


class YandexEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embd_model):
        # embd_model = sdk.models.text_embeddings("doc")
        self.embd_model = embd_model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            result = self.embd_model.run(text)
            # result.embedding - это кортеж (tuple),
            # Chroma нужно list[float], конвертируем:
            emb_vector = list(result.embedding)
            vectors.append(emb_vector)
        return vectors

def split_text_by_tokens(text: str, max_chunk_size_tokens=700, chunk_overlap_tokens=300) -> list[str]:
    """
    Упрощённая функция разбиения текста на чанки ~по словам.
    В реальном проекте лучше использовать токенизацию,
    согласованную с моделью (yandexgpt), чтобы не выходить за лимиты.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_chunk_size_tokens
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (max_chunk_size_tokens - chunk_overlap_tokens)
    return chunks


def init_chroma():
    """
    Инициализирует коллекцию Chroma (получает или создаёт).
    Устанавливает функцию для эмбеддингов (yandex_embeddings).
    """
    embedding_fn = YandexEmbeddingFunction(embd_model)

    collection = client.get_or_create_collection(
        name="peterhof_docs",
        embedding_function=embedding_fn
    )
    return collection


def create_or_update_chroma_collection(collection):
    """
    Считывает данные из data.json, разбивает тексты на чанки
    и добавляет в коллекцию Chroma.
    Если коллекция не пуста и нужно всё пересоздать —
    либо очистите её вручную (collection.delete(...)), либо используйте другую логику.
    """
    with open('data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i, place in enumerate(data['places']):
        doc_text = f"{place['context']}\n\nimage_url для {place['title']}: {place['image_url_v2']}"
        chunks = split_text_by_tokens(doc_text)

        for idx, chunk in enumerate(chunks):
            doc_id = f"doc_{i}_{idx}"
            # Добавляем в Chroma
            collection.add(
                documents=[chunk],
                metadatas=[{"title": place['title'], "image_url": place['image_url_v2']}],
                ids=[doc_id]
            )


# -------------------- Работа с JSON (пользователи, тексты) --------------------

def load_dictionary(path='users.json'):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}  # если файла нет, вернуть пустой словарь


def write_dictionary(dictionary, path='users.json'):
    Path(path).write_text(json.dumps(dictionary, ensure_ascii=False, indent=4), encoding='utf-8')


# -------------------- Логика RAG: поиск и генерация ответа через LLM --------------------

async def get_answer(question: str, user_id: int) -> str:
    """
    1. Ищем релевантные документы в Chroma
    2. Склеиваем их в общий контекст
    3. Передаём всё (включая историю диалога) в yandexgpt (через sdk)
    4. Возвращаем итоговый текст ответа
    """
    # 1. Поиск документов
    results = bot.chroma_collection.query(
        query_texts=[question],
        n_results=3
    )
    retrieved_docs = results.get('documents', [[]])[0]
    relevant_context = "\n\n".join(retrieved_docs)

    # 2. История диалога
    memory = bot.user_settings[str(user_id)]['memory']
    prompt = f"""
1. Контекст и цель:
    - Вы являетесь виртуальным гидом для посетителей музея-заповедника Петергоф.
    - Ваша цель — предоставлять исчерпывающие ответы на вопросы пользователей относительно объектов музея и маршрутов, 
      основываясь на доступной базе данных. Поддерживайте интерес посетителя к посещению музея.

2. Коммуникация с пользователем:
    - Всегда стремитесь ответить до 1000 символов (это очень важно).
    - Если вопрос не может быть решён на основе данных, вежливо признайте, что не имеете ответа, 
      но предложите общую информацию о музее.

3. Подача информации:
    - При ответе на вопросы о конкретных объектах, предоставляйте название и завлекательное описание.
    - Если вопрос касается маршрутов, укажите несколько рекомендованных объектов последовательно, формируя маршрут.

4. Мотивация и вдохновение:
    - Используйте вдохновляющий и побуждающий язык, чтобы заинтересовать посетителя.
    - Подчёркивайте уникальные аспекты и ценность каждого объекта.
    - Иногда включайте приглашение посетить сайт музея.

5. Ограничения:
    - Не используйте символы форматирования вроде "**" (звёздочки).
    - Длина сообщения до 1000 символов.
    - Отвечайте только на основе имеющейся информации. Если данных недостаточно, честно скажите об этом.

6. Память (последние 3 обмена):
   1 (последний): вопрос: {memory["questions"][0]}; ваш ответ: {memory["answers"][0]};
   2 (предпоследний): вопрос: {memory["questions"][1]}; ваш ответ: {memory["answers"][1]};
   3 (предпредпоследний): вопрос: {memory["questions"][2]}; ваш ответ: {memory["answers"][2]};

---
Фрагменты релевантного контекста (не обязательно использовать всё):
{relevant_context}

Теперь ответьте пользователю.
    """.strip()

    # 3. Генерация ответа через Яндекс (yandexgpt) — аналогично вашему коду
    result = llm_model.run(prompt)
    # Сам текст ответа: result.text
    answer_text = result.alternatives[0].text
    # 4. Обновляем "память"
    bot.user_settings[str(user_id)]['memory']['questions'] = [
        question,
        memory['questions'][0],
        memory['questions'][1]
    ]
    bot.user_settings[str(user_id)]['memory']['answers'] = [
        answer_text,
        memory['answers'][0],
        memory['answers'][1]
    ]
    write_dictionary(bot.user_settings)
    return answer_text


async def get_answer_image(question: str) -> str:
    """
    Ищем объекты в Chroma, просим Яндекс вернуть одну ссылку image_url, если есть.
    """
    results = bot.chroma_collection.query(
        query_texts=[question],
        n_results=3
    )
    retrieved_docs = results.get('documents', [[]])[0]
    relevant_context = "\n\n".join(retrieved_docs)

    prompt = (
        "Предоставьте одну ссылку image_url, связанную с объектом(ами), упомянутыми в запросе. "
        "Если ничего не найдено, извинитесь и скажите, что нет данных.\n\n"
        f"Вот релевантные данные:\n{relevant_context}\n\n"
        "Дайте короткий ответ с одной ссылкой (или извинением)."
    )

    assistant = await sdk.assistants.create(
        name='rag-assistant',
        model='yandexgpt',
        temperature=0.1,
        instruction=prompt,
        max_prompt_tokens=2000
    )
    thread = await sdk.threads.create()
    try:
        await thread.write(question)
        run = await assistant.run(thread)
        result = await run
        return result.text.strip()
    finally:
        await thread.delete()
        await assistant.delete()


# -------------------- Хендлеры бота (команды, сообщения и т.д.) --------------------

@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['help'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['help'])
async def help_command(message: types.Message):
    user_id_str = str(message.from_user.id)
    user_lang = bot.user_settings[user_id_str]['language']
    await message.reply(bot.texts[user_lang]['help'])


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['start'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['start'])
async def start(message: types.Message):
    user_id_str = str(message.from_user.id)
    if user_id_str not in bot.user_settings:
        user_country = message.from_user.language_code if message.from_user.language_code else 'en'
        language = 'ru' if user_country in CIS_COUNTRIES else 'en'
        bot.user_settings[user_id_str] = {
            'language': language,
            'menu': 'off',
            'memory': {'questions': ['-', '-', '-'], 'answers': ['-', '-', '-']}
        }
        write_dictionary(bot.user_settings)
    user_lang = bot.user_settings[user_id_str]['language']
    await message.reply(bot.texts[user_lang]['start'])


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
async def settings(message: types.Message):
    user_id_str = str(message.from_user.id)
    user_lang = bot.user_settings[user_id_str]['language']
    keyboard = get_settings_keyboard(message.from_user.id)
    await message.reply(bot.texts[user_lang]['settings'], reply_markup=keyboard)


class SupportForm(StatesGroup):
    name = State()


def crop(text: str):
    return text if len(text) <= 10 else f'{text[:10]}...'


async def get_reply_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton('/help'), KeyboardButton('/settings'), KeyboardButton('/support')]
        ],
        resize_keyboard=True
    )


def get_settings_keyboard(user_id: int):
    user_id_str = str(user_id)
    user_lang = bot.user_settings[user_id_str]['language']
    button_lang = InlineKeyboardButton(
        bot.texts[user_lang]['language'], callback_data='toggle_language'
    )
    menu_status = bot.user_settings[user_id_str]['menu']
    mark = '✅' if menu_status == 'on' else '❌'
    button_menu = InlineKeyboardButton(
        bot.texts[user_lang]['menu'] + mark,
        callback_data='toggle_menu'
    )
    keyboard = InlineKeyboardMarkup()
    keyboard.add(button_lang)
    keyboard.add(button_menu)
    return keyboard


def get_support_keyboard(user_id: int):
    user_id_str = str(user_id)
    keyboard = InlineKeyboardMarkup()
    user_data = bot.user_settings[user_id_str]
    # Предположим, что 'tickets' — список словарей
    tickets = user_data.get('tickets', [])
    for t in tickets:
        ticket_id = t['id']
        keyboard.add(
            InlineKeyboardButton(crop(t['messages'][0]), callback_data=f'ticket_{ticket_id}')
        )
    keyboard.add(InlineKeyboardButton('Новый тикет', callback_data='new_ticket'))
    return keyboard


def get_ticket_keyboard(user_id: int, ticket_id: int):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton('Закрыть тикет', callback_data=f'close_ticket_{ticket_id}'))
    keyboard.add(InlineKeyboardButton('Назад', callback_data=f'back_ticket_{user_id}'))
    return keyboard


@dp.callback_query_handler(lambda call: call.data == "new_ticket")
async def new_ticket(call: types.CallbackQuery):
    await SupportForm.name.set()
    await bot.send_message(call.from_user.id, 'new_ticket')
    bot.states[call.from_user.id] = 'new'


@dp.message_handler(state=SupportForm.name)
async def support_finish(message: types.Message, state: FSMContext):
    user_id_str = str(message.from_user.id)
    user_lang = bot.user_settings[user_id_str]['language']

    if message.text == '/cancel':
        await message.reply(bot.texts[user_lang]['support_cancel'])
        await state.finish()
        return

    if bot.states.get(message.from_user.id) == 'new':
        # Предположим, что в 'bot.user_settings' хранится глобальный счётчик 'ticket'
        if 'ticket' not in bot.user_settings:
            bot.user_settings['ticket'] = 1

        ticket_number = bot.user_settings['ticket']
        bot.user_settings['ticket'] += 1

        if 'tickets' not in bot.user_settings[user_id_str]:
            bot.user_settings[user_id_str]['tickets'] = []

        bot.user_settings[user_id_str]['tickets'].append({
            "id": ticket_number,
            "messages": [message.text]
        })

        write_dictionary(bot.user_settings)
        await message.reply(f'Сообщение отправлено, создан новый тикет #{ticket_number}')

        # Оповещаем админ-чат
        await bot.send_message(
            admin_chat,
            f'{message.from_user.id} ({message.message_id}):\n```{message.text}```',
            parse_mode='Markdown'
        )
    else:
        # Другие состояния, если есть
        pass

    await state.finish()


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['support'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['support'])
async def support(message: types.Message):
    user_id_str = str(message.from_user.id)
    tickets = bot.user_settings[user_id_str].get('tickets', [])
    if not tickets:
        await message.reply('У вас нет открытых тикетов')
    else:
        await message.reply('Ваши тикеты:', reply_markup=get_support_keyboard(message.from_user.id))


@dp.callback_query_handler(lambda call: call.data == "toggle_language")
async def toggle_language(call: types.CallbackQuery):
    user_id_str = str(call.from_user.id)
    current_lang = bot.user_settings[user_id_str]['language']
    new_language = 'ru' if current_lang == 'en' else 'en'
    bot.user_settings[user_id_str]['language'] = new_language
    keyboard = get_settings_keyboard(call.from_user.id)
    await call.message.edit_text(bot.texts[new_language]['settings'], reply_markup=keyboard)
    write_dictionary(bot.user_settings)


@dp.callback_query_handler(lambda call: call.data == "toggle_menu")
async def toggle_menu(call: types.CallbackQuery):
    user_id_str = str(call.from_user.id)
    current_menu = bot.user_settings[user_id_str]['menu']
    new_menu = 'on' if current_menu == 'off' else 'off'
    bot.user_settings[user_id_str]['menu'] = new_menu

    keyboard = get_settings_keyboard(call.from_user.id)
    new_lang = bot.user_settings[user_id_str]['language']
    await call.message.edit_text(bot.texts[new_lang]['settings'], reply_markup=keyboard)

    if new_menu == 'on':
        msg = await bot.send_message(
            call.from_user.id,
            'Меню включено⌨️',
            reply_markup=await get_reply_keyboard()
        )
    else:
        msg = await bot.send_message(call.from_user.id, 'ㅤ', reply_markup=ReplyKeyboardRemove())
        await msg.delete()

    write_dictionary(bot.user_settings)


@dp.edited_message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
@dp.message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
async def on_message_chat(message: types.Message):
    if message.reply_to_message is None:
        await message.reply('Ответьте на тикет')
    elif message.reply_to_message.from_user.id == bot.id and message.text != 'Ответьте на тикет':
        # Извлекаем user_id и message_id
        text_parts = message.reply_to_message.text.split(' ')
        user_id_str = text_parts[0]
        msg_part = message.reply_to_message.text.split('(')[1].split(')')[0]
        user_id = int(user_id_str)
        user_msg_id = int(msg_part)

        await bot.send_message(
            user_id,
            f'```{message.text}```',
            reply_to_message_id=user_msg_id,
            parse_mode='Markdown'
        )
        await message.reply('Сообщение отправлено')


@dp.edited_message_handler(lambda message: message.chat.type == 'private')
@dp.message_handler(lambda message: message.chat.type == 'private')
async def on_message(message: types.Message):
    user_id_str = str(message.from_user.id)
    user_lang = bot.user_settings[user_id_str]['language']
    loading_msg = bot.texts[user_lang]['loading']

    msg = await message.reply(loading_msg)
    try:
        answer = await get_answer(message.text, message.from_user.id)
        # Пытаемся получить ссылку на картинку
        answer_img = await get_answer_image(message.text)

        # Если ответ слишком длинный, урезаем для caption
        if len(answer) > 1020:
            short_answer = ""
            for line in answer.split('\n'):
                if len(short_answer + f'\n{line}') > 1020:
                    break
                else:
                    short_answer += f'\n{line}'
            answer = short_answer.strip()

        # Пробуем отправить фото, если ответ_img — это валидная ссылка
        # (вы можете добавить проверку, что там http(s):// ... )
        await message.reply_photo(photo=answer_img, caption=answer)
        await msg.delete()
    except Exception as e:
        print("Error in on_message:", e)
        # если не вышло, просто редактируем сообщение
        await msg.edit_text(answer)


@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice_message(message: types.Message):
    user_id_str = str(message.from_user.id)
    user_lang = bot.user_settings[user_id_str]['language']
    loading_msg = bot.texts[user_lang]['loading']

    msg = await message.reply(loading_msg)

    file_info = await bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    local_file = f"{message.voice.file_id}.ogg"
    await bot.download_file(file_path, local_file)

    text = recognize(local_file)
    remove(local_file)

    try:
        answer_text = await get_answer(text, message.from_user.id)
        result_msg = f'Ваш вопрос: ```{text}```\n\n{answer_text}'
        await msg.edit_text(result_msg, parse_mode='Markdown')
    except Exception as e:
        print("Error in voice handler:", e)
        await msg.edit_text("Произошла ошибка при обработке голосового сообщения.")


def recognize(audio):
    """
    Распознаём речь через Yandex SpeechKit, как в вашем исходном коде.
    """
    model = model_repository.recognition_model()
    model.model = 'general'
    model.language = 'ru-RU'
    model.audio_processing_type = AudioProcessingType.Full
    result = model.transcribe_file(audio)
    return result[0].normalized_text


# -------------------- Основная функция запуска --------------------

async def main():
    # Загружаем словари и настраиваем SpeechKit
    bot.texts = load_dictionary('texts.json')
    bot.user_settings = load_dictionary('users.json')

    configure_credentials(
        yandex_credentials=creds.YandexCredentials(api_key=YANDEX_AUTH)
    )

    # Инициализируем Chroma и загружаем данные (при необходимости)
    bot.chroma_collection = init_chroma()
    # Если нужно, один раз на старте пересоздать или дозагрузить документы:
    # bot.chroma_collection.delete()  # если хотите всё пересоздать
    # create_or_update_chroma_collection(bot.chroma_collection)

    # Запускаем лонг-поллинг
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
