import asyncio
import json
import random
import re
from pathlib import Path
from aiogram import types
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup,
                           ReplyKeyboardRemove, MediaGroup)
from rout_suggestion import get_route_suggestion
from utils import create_json_chunks
from dotenv import load_dotenv
from os import getenv, remove
from yandex_cloud_ml_sdk import AsyncYCloudML, YCloudML
from yandex_cloud_ml_sdk.search_indexes import StaticIndexChunkingStrategy, TextSearchIndexType
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType
from raptor import DataEnlarger

load_dotenv()
bot = Bot(token=getenv('TOKEN'))
dp = Dispatcher(bot, storage=MemoryStorage())
bot.user_settings = {}
bot.texts = {}
bot.states = {}
bot.route_data = {}
bot.sdk = None
bot.files = None
CIS_COUNTRIES = ['ru', 'ua', 'by', 'kz', 'kg', 'am', 'uz', 'tj', 'az', 'md']
admin_chat = -1002411793280


async def files_delete():
    async for file in bot.sdk.files.list():
        await file.delete()


async def files_create():
    sdk = YCloudML(
        folder_id=getenv('FOLDER'),
        auth=getenv('AUTH'),
    )

    embd = sdk.models.text_embeddings('doc')
    model = sdk.models.completions('yandexgpt')

    path = 'data.json'
    de = DataEnlarger(llm=model, embd=embd, data_path=path)
    docs = de.chunks

    files = []
    for i, doc in enumerate(docs):
        print(len(doc))
        file_name = f'temp_doc_{i}.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(doc)
        file = await bot.sdk.files.upload(file_name)
        files.append(file)
        remove(file_name)
    return files


'''async def files_create():
    with open('data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    docs = [f'{place["context"]}\n\nimage_url для {place["title"]}: {place["image_url_v2"]}' for place in data['places']]
    files = []
    for i, doc in enumerate(docs):
        file_name = f'temp_doc_{i}.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(doc)
        file = await bot.sdk.files.upload(file_name)
        files.append(file)
        remove(file_name)
    return files'''


async def get_answer_test(question: str, user_id: int) -> str:
    operation = await bot.sdk.search_indexes.create_deferred(
        bot.files,
        index_type=TextSearchIndexType(
            chunking_strategy=StaticIndexChunkingStrategy(
                max_chunk_size_tokens=700,
                chunk_overlap_tokens=300,
            )
        )
    )
    search_index = await operation
    tool = bot.sdk.tools.search_index(search_index)
    memory = bot.user_settings[str(user_id)]['memory']
    random_route = '' if random.randint(1, 5) != 1 else 'Также обязательно предложите пользователю составить индивидуальный маршрут и упомяните точную команду: /route'
    prompt = \
f'''
1. Контекст и цель:
    - Вы являетесь виртуальным гидом для посетителей музея-заповедника Петергоф.
    - Ваша цель — предоставлять исчерпывающие ответы на вопросы пользователей относительно объектов музея и маршрутов, основываясь на доступной базе данных. Поддерживайте интерес посетителя к посещению музея.

2. Коммуникация с пользователем:
    - Всегда стремитесь ответить до 1000 символов. Я запрещаю отвечать длиной больше 1000 символов.
    - Если вопрос не может быть решён на основе данных, вежливо признайте, что не имеете ответа, но предложите общую информацию о музее.

3. Подача информации:
    - При ответе на вопросы о конкретных объектах, предоставляйте название и завлекательное описание.
    - Если вопрос касается маршрутов, укажите несколько рекомендованных объектов последовательно, формируя маршрут.

4. Мотивация и вдохновение:
    - Используйте вдохновляющий и побуждающий язык, чтобы заинтересовать пользователя в посещении музея.
    - Подчеркните уникальные аспекты и ценность каждого объекта, сделав акцент на незабываемом опыте, который ждёт посетителя.
    - Иногда включайте предложения посетить сайт музея для более полной информации, но не прикладывайте ссылку.

5. Ограничения:
    - Не используйте символы форматирования по типу \"**\" (звёздочки).
    - Длина сообщения до 1000 символов.
    - Отвечайте только на основе имеющейся информации. Если данных недостаточно, честно сообщите об этом, предлагая в качестве альтернативы общие советы по посещению музея.

6. Обязательно В КОНЦЕ ответа предоставьте параметр image_url (ссылку на изображение объекта, про который ты пишешь); несколько ссылок, если в вопросе ИЛИ вашем ответе упоминается несколько ОБЪЕКТОВ; одну ссылку, если упоминается один объект; не писать ничего, если не упоминается ни один объект.
    ещё раз, если ты сам в ответе упомянул какие-то объекты, например фонтаны, то приложи ссылки
    {random_route}
    Соблюдайте эти рекомендации, чтобы предоставить пользователям интересные, информативные и мотивирующие ответы, вдохновляя их на посещение музея Петергоф.
'''
    memory_text = \
f'''
6. Память:
    - Вот 3 последних запроса пользователя к вам и ваши ответы на них:
        1 (последний): вопрос: {memory["questions"][0]}; ваш ответ: {memory["answers"][0]};
        2 (предпоследний): вопрос: {memory["questions"][1]}; ваш ответ: {memory["answers"][1]};
        3 (предпредпоследний): вопрос: {memory["questions"][2]}; ваш ответ: {memory["answers"][2]};
    \"-\" означает отсутствие запроса. Пользователь может использовать местоимения или говорить в контексте этих сообщений, учитывайте это.
'''
    assistant = await bot.sdk.assistants.create(
        name='rag-assistant',
        model='yandexgpt',
        tools=[tool],
        temperature=0.1,
        instruction=prompt,
        max_prompt_tokens=2000
    )
    thread = await bot.sdk.threads.create()
    try:
        await thread.write(question)
        run = await assistant.run(thread)
        result = await run
        bot.user_settings[str(user_id)]['memory']['questions'] = [question, memory['questions'][0], memory['questions'][1]]
        bot.user_settings[str(user_id)]['memory']['answers'] = [result.text.split('image_url')[0].strip(), memory['answers'][0], memory['answers'][1]]
        write_dictionary(bot.user_settings)
        return result.text.replace('**', '')
    finally:
        await search_index.delete()
        await thread.delete()
        await assistant.delete()


async def get_answer_image(question: str) -> str:
    operation = await bot.sdk.search_indexes.create_deferred(
        bot.files,
        index_type=TextSearchIndexType(
            chunking_strategy=StaticIndexChunkingStrategy(
                max_chunk_size_tokens=700,
                chunk_overlap_tokens=300,
            )
        )
    )
    search_index = await operation
    tool = bot.sdk.tools.search_index(search_index)
    prompt = 'Предоставьте параметр image_url (ссылку на изображение), связанный с указанными, или одним из указанных в запросе объектов, если их несколько; или с указанным объектом, если в запросе он один. Отвечать на вопрос пользователя не требуется. В случае, если в запросе не фигурирует ни один объект, извинитесь.'
    assistant = await bot.sdk.assistants.create(
        name='rag-assistant',
        model='yandexgpt',
        tools=[tool],
        temperature=0.1,
        instruction=prompt,
        max_prompt_tokens=2000
    )
    thread = await bot.sdk.threads.create()
    try:
        await thread.write(question)
        run = await assistant.run(thread)
        result = await run
        return result.text
    finally:
        await search_index.delete()
        await thread.delete()
        await assistant.delete()


def load_dictionary(path='users.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_dictionary(dictionary, path='users.json'):
    Path(path).write_text(json.dumps(dictionary, ensure_ascii=False, sort_keys=False, indent=4), encoding='utf-8')


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['help'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['help'])
async def help_command(message: types.Message):
    await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['help'], reply_markup=get_route_keyboard())


async def get_route(user_id: int, request: str = None, latitude: str = None, longitude: str = None):
    if request is None:
        dialogue_user = bot.user_settings[str(user_id)]['memory']['questions'][0]
        dialogue_bot = bot.user_settings[str(user_id)]['memory']['answers'][0]
        user_dialogues = [
            {'user': dialogue_user if dialogue_user != '-' else dialogue_user},
            {'bot': dialogue_bot if dialogue_bot != '-' else dialogue_bot}
        ]
    else:
        user_dialogues = [
            {'user': f'Мои пожелания к маршруту: {request}'},
            {'bot': 'Хорошо, я учту ваши пожелания при составлении маршрута'}
        ]
    print(user_dialogues)
    data_chunks = create_json_chunks()
    res = get_route_suggestion(user_dialogues, data_chunks, initial_coordinates=['59.891802' if latitude is None else latitude, '29.913220' if longitude is None else longitude], objects_number=5)
    return res.replace('**', '')


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['route'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['route'])
async def route(message: types.Message):
    msg = await message.reply(get_route_text(message.from_user.id), disable_web_page_preview=True)
    await msg.edit_text(await get_route(message.from_user.id))
    await msg.edit_reply_markup(get_route_keyboard())


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['start'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['start'])
async def start(message: types.Message):
    if message.from_user.id not in bot.user_settings:
        user_country = message.from_user.language_code if message.from_user.language_code else 'en'
        bot.user_settings[str(message.from_user.id)] = {
            'language': 'ru' if user_country in CIS_COUNTRIES else 'en',
            'menu': 'off',
            'memory': {'questions': ['-', '-', '-'], 'answers': ['-', '-', '-']},
            'tickets': {}
        }
        write_dictionary(bot.user_settings)
    await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['start'])


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
async def settings(message: types.Message):
    keyboard = get_settings_keyboard(message.from_user.id)
    await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['settings'], reply_markup=keyboard)


class SupportForm(StatesGroup):
    name = State()


class RouteForm(StatesGroup):
    name = State()


class GeoForm(StatesGroup):
    name = State()


def crop(text: str):
    return text if len(text) <= 10 else f'{text[:10]}…'


def get_ticket_answer_keyboard(ticket_id):
    button1 = InlineKeyboardButton('Посмотреть', callback_data=f'ticket_answer_{ticket_id}')
    return InlineKeyboardMarkup().add(button1)


def get_reply_keyboard():
    return ReplyKeyboardMarkup(resize_keyboard=True, is_persistent=True, keyboard=[[KeyboardButton('/help'), KeyboardButton('/settings'), KeyboardButton('/support'), KeyboardButton('/route')]])


def get_settings_keyboard(user_id: int):
    button1 = InlineKeyboardButton(bot.texts[bot.user_settings[str(user_id)]['language']]['language'], callback_data='toggle_language')
    button2 = InlineKeyboardButton(bot.texts[bot.user_settings[str(user_id)]['language']]['menu'] + ('✅' if bot.user_settings[str(user_id)]['menu'] == 'on' else '❌'), callback_data='toggle_menu')
    return InlineKeyboardMarkup().add(button1).add(button2)


def get_route_keyboard():
    keyboard = InlineKeyboardMarkup()
    button1 = InlineKeyboardButton('♻️', callback_data=f'route_yes')
    button2 = InlineKeyboardButton('❌', callback_data=f'route_no')
    button3 = InlineKeyboardButton('📍', callback_data=f'route_geo')
    keyboard.row(button1, button2, button3)
    return keyboard


@dp.message_handler(state=GeoForm.name, content_types=['location'])
async def handle_location(message: types.Message):
    msg = await message.reply('Создаю маршрут с учётом вашей геолокации...', disable_web_page_preview=True)
    latitude, longitude = str(message.location.latitude), str(message.location.longitude)
    if bot.route_data.get(message.from_user.id) is None:
        bot.route_data[message.from_user.id] = {'geo': [latitude, longitude], 'request': None}
    else:
        bot.route_data[message.from_user.id]['geo'] = [latitude, longitude]
    await msg.edit_text(await get_route(message.from_user.id, bot.route_data[message.from_user.id]['request'], bot.route_data[message.from_user.id]['geo'][0], bot.route_data[message.from_user.id]['geo'][1]))
    await msg.edit_reply_markup(get_route_keyboard())


def get_support_keyboard(user_id: int):
    keyboard = InlineKeyboardMarkup()
    for i, j in bot.user_settings[str(user_id)]['tickets'].items():
        keyboard.add(InlineKeyboardButton(f'#t{i}: {crop(j[0][0])}', callback_data=f'exist_ticket_{i}'))
    if len(bot.user_settings[str(user_id)]['tickets'].keys()) < 10:
        keyboard.add(InlineKeyboardButton('Новый тикет', callback_data='new_ticket'))
    return keyboard


def get_ticket_keyboard(ticket_id: int):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton('Сообщение', callback_data=f'message_ticket_{ticket_id}'))
    keyboard.add(InlineKeyboardButton('Закрыть тикет', callback_data=f'close_ticket_{ticket_id}'))
    keyboard.add(InlineKeyboardButton('Назад', callback_data=f'back_ticket'))
    return keyboard


def get_ticket_message_keyboard(ticket_id: int):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton('Назад', callback_data=f'exist_backticket_{ticket_id}'))
    return keyboard


def get_new_ticket_cancel_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton('Назад', callback_data=f'cancel_ticket'))
    return keyboard


@dp.callback_query_handler(lambda call: call.data == 'new_ticket')
async def new_ticket(call: types.CallbackQuery):
    await SupportForm.name.set()
    await call.message.edit_text('Введите ваше сообщение поддержке или нажмите кнопку для отмены создания тикета')
    await call.message.edit_reply_markup(get_new_ticket_cancel_keyboard())
    bot.states[call.from_user.id] = 'new'


@dp.callback_query_handler(lambda call: call.data == 'back_ticket')
async def back_ticket(call: types.CallbackQuery):
    tickets_len = len(bot.user_settings[str(call.from_user.id)]['tickets'])
    await call.message.edit_text(get_tickets_text(tickets_len))
    await call.message.edit_reply_markup(get_support_keyboard(call.from_user.id))


@dp.callback_query_handler(lambda call: call.data == 'cancel_ticket', state=SupportForm.name)
async def cancel_ticket(call: types.CallbackQuery, state: FSMContext):
    await back_ticket(call)
    await state.finish()


def quote_text(text):
    return f'<blockquote>{text}</blockquote>'


@dp.callback_query_handler(lambda call: 'exist_ticket_' in call.data or 'ticket_answer_' in call.data)
async def exist_ticket(call: types.CallbackQuery):
    ticket_id = int(call.data.split('_')[2])
    message_history = '\n'.join(f'{i[1]}: {quote_text(i[0])}' for i in bot.user_settings[str(call.from_user.id)]['tickets'][str(ticket_id)])
    message_history = message_history if len(message_history) < 4090 else f'…\n{message_history[len(message_history) - 4080:]}'
    if message_history.count('<blockquote>') < message_history.count('</blockquote>'):
        message_history = f'<blockquote>{message_history}'
    print(message_history)
    await call.message.edit_text(f'Тикет #t{ticket_id}\n\nИстория:\n{message_history}', parse_mode='HTML')
    await call.message.edit_reply_markup(get_ticket_keyboard(ticket_id))


@dp.callback_query_handler(lambda call: 'exist_backticket_' in call.data, state=SupportForm.name)
async def exist_backticket(call: types.CallbackQuery, state: FSMContext):
    await exist_ticket(call)
    await state.finish()


def get_tickets_text(tickets_len):
    return 'У вас нет открытых тикетов' if tickets_len == 0 else f'У вас {tickets_len} открыты{"й" if tickets_len % 10 == 1 and tickets_len % 100 != 11 else "х"} тикет{"ов" if 5 <= tickets_len % 10 <= 9 or tickets_len % 10 == 0 or 11 <= tickets_len % 100 <= 19 else "" if tickets_len % 10 == 1 else "а"}'


@dp.callback_query_handler(lambda call: 'close_ticket_' in call.data)
async def close_ticket(call: types.CallbackQuery):
    ticket_id = int(call.data.replace('close_ticket_', ''))
    bot.user_settings[str(call.from_user.id)]['tickets'].pop(str(ticket_id))
    write_dictionary(bot.user_settings)
    tickets_len = len(bot.user_settings[str(call.from_user.id)]['tickets'])
    await call.message.edit_text(get_tickets_text(tickets_len))
    await call.message.edit_reply_markup(get_support_keyboard(call.from_user.id))
    await bot.send_message(admin_chat, f'Тикет #t{ticket_id} закрыт пользователем')


@dp.callback_query_handler(lambda call: 'message_ticket_' in call.data)
async def message_ticket(call: types.CallbackQuery):
    ticket_id = int(call.data.replace('message_ticket_', ''))
    await SupportForm.name.set()
    await call.message.edit_text('Введите ваше сообщение поддержке или нажмите кнопку для отмены')
    await call.message.edit_reply_markup(get_ticket_message_keyboard(ticket_id))
    bot.states[call.from_user.id] = f'exist_{ticket_id}'


@dp.message_handler(state=SupportForm.name)
async def support_finish(message: types.Message, state: FSMContext):
    if bot.states[message.from_user.id] == 'new':
        ticket_number = bot.user_settings['ticket']
        bot.user_settings['ticket'] += 1
        bot.user_settings[str(message.from_user.id)]['tickets'][str(ticket_number)] = [[message.text, '👤']]
        await message.reply(f'Сообщение отправлено, создан новый тикет #t{ticket_number}')
        text1 = 'Новый тикет'
    else:
        ticket_number = bot.states[message.from_user.id].split('exist_')[1]
        bot.user_settings[str(message.from_user.id)]['tickets'][str(ticket_number)].append([message.text, '👤'])
        await message.reply(f'Сообщение отправлено в существующий тикет #t{ticket_number}')
        text1 = 'Новое сообщение по тикету'
    await bot.send_message(admin_chat, f'{text1} \\#t{ticket_number} от [{message.from_user.id}](tg://user?id={message.from_user.id}):\n>' + text_v2(message.text).replace('\n', '\n>'), parse_mode='MarkdownV2')
    write_dictionary(bot.user_settings)
    await state.finish()


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['support'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['support'])
async def support(message: types.Message):
    tickets_len = len(bot.user_settings[str(message.from_user.id)]['tickets'])
    await message.reply(get_tickets_text(tickets_len), reply_markup=get_support_keyboard(message.from_user.id))


@dp.callback_query_handler(lambda call: call.data == 'toggle_language')
async def toggle_language(call: types.CallbackQuery):
    user_id = call.from_user.id
    new_language = 'ru' if bot.user_settings[str(user_id)]['language'] == 'en' else 'en'
    bot.user_settings[str(user_id)]['language'] = new_language
    keyboard = get_settings_keyboard(user_id)
    await call.message.edit_text(bot.texts[new_language]['settings'], reply_markup=keyboard)
    write_dictionary(bot.user_settings)


@dp.callback_query_handler(lambda call: call.data == 'toggle_menu')
async def toggle_menu(call: types.CallbackQuery):
    user_id = call.from_user.id
    new_menu = 'on' if bot.user_settings[str(user_id)]['menu'] == 'off' else 'off'
    bot.user_settings[str(user_id)]['menu'] = new_menu
    keyboard = get_settings_keyboard(user_id)
    await call.message.edit_text(bot.texts[bot.user_settings[str(user_id)]['language']]['settings'], reply_markup=keyboard)
    msg = await bot.send_message(user_id, 'Меню включено⌨️' if new_menu == 'on' else 'ㅤ', reply_markup=get_reply_keyboard() if new_menu == 'on' else ReplyKeyboardRemove())
    if new_menu != 'on':
        await msg.delete()
    write_dictionary(bot.user_settings)


@dp.callback_query_handler(lambda call: 'route_' in call.data)
async def route_inline_handler(call: types.CallbackQuery):
    if call.data == 'route_yes':
        await call.message.reply('Напишите пожелания к новому маршруту')
        await RouteForm.name.set()
    if call.data == 'route_geo':
        await call.message.reply('Отправьте свою геолокацию')
        await GeoForm.name.set()
        return
    else:
        await call.message.edit_reply_markup()


def get_route_text(user_id):
    return 'Создаю маршрут с начальной точкой по умолчанию...' if bot.route_data.get(user_id) is None or bot.route_data[user_id]['geo'][0] is None else 'Создаю маршрут с учётом вашей геолокации...'


@dp.message_handler(state=RouteForm.name)
async def route_finish(message: types.Message, state: FSMContext):
    msg = await message.reply(get_route_text(message.from_user.id), disable_web_page_preview=True)
    if bot.route_data.get(message.from_user.id) is None:
        bot.route_data[message.from_user.id] = {'geo': [None, None], 'request': message.text}
    else:
        bot.route_data[message.from_user.id]['request'] = [message.text, message.text]
    await msg.edit_text(await get_route(message.from_user.id, bot.route_data[message.from_user.id]['request'], bot.route_data[message.from_user.id]['geo'][0], bot.route_data[message.from_user.id]['geo'][1]))
    await msg.edit_reply_markup(get_route_keyboard())
    await state.finish()


def text_v2(text):
    for i in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
        text = text.replace(i, f'\\{i}')
    return text



@dp.edited_message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
@dp.message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
async def on_message_chat(message: types.Message):
    # if message.reply_to_message is None:
    #     await message.reply('Ответьте на тикет')
    if message.reply_to_message is not None and message.reply_to_message.from_user.id == bot.id and message.text != 'Ответьте на тикет':
        ticket_id = message.reply_to_message.text.split('#t')[1].split(' ')[0]
        user_id = message.reply_to_message.text.split('от ')[1].split(':')[0]
        if any([ticket_id == i for i in bot.user_settings[user_id]['tickets'].keys()]):
            await bot.send_message(int(user_id), f'Пришёл ответ от поддержки по тикету #t{ticket_id}', reply_markup=get_ticket_answer_keyboard(ticket_id))
            bot.user_settings[user_id]['tickets'][ticket_id].append([message.text, '🛠️'])
            await message.reply('Сообщение отправлено')
            write_dictionary(bot.user_settings)
        else:
            await message.reply('Этот тикет уже закрыт')


def shorten_text(text, length=1020):
    new_text = text
    if len(text) > length:
        new_text = ''
        for line in text.split('\n'):
            if len(new_text + f'\n{line}') > length:
                break
            else:
                new_text += f'\n{line}'
    return new_text


@dp.edited_message_handler(lambda message: message.chat.type == 'private')
@dp.message_handler(lambda message: message.chat.type == 'private')
async def on_message(message: types.Message):
    msg = await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['loading'])
    answer = ''
    try:
        answer = await get_answer_test(message.text, message.from_user.id)
        links = re.findall(r'https?://[^\s]+', answer)
        print(answer)
        answer = answer.split('image_url')[0].strip()
    except Exception as e:
        print(e)
        await msg.edit_text('Произошла непредвиденная ошибка')
        return
    if len(links) == 0:
        await msg.edit_text(shorten_text(answer, 4080))
        return
    try:
        answer_shorten = shorten_text(answer)
        if len(links) == 1:
            await message.reply_photo(photo=links[0], caption=answer_shorten)
        elif len(links) > 1:
            media_group = MediaGroup()
            for i, link in enumerate(links):
                if i == 0:
                    media_group.attach_photo(photo=link, caption=answer_shorten)
                else:
                    media_group.attach_photo(photo=link)
            await message.reply_media_group(media=media_group)
        await msg.delete()
        return
    except Exception as e:
        print(e)
    await msg.edit_text(shorten_text(answer, 4080))


@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice_message(message: types.Message):
    msg = await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['loading'])
    file_info = await bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    local_file = f'{message.voice.file_id}.ogg'
    await bot.download_file(file_path, local_file)
    text = recognize(local_file)
    text = text if text is not None and text != '' and len(text) >= 2 else '-'
    remove(local_file)
    await msg.edit_text(f'Ваш вопрос: {quote_text(text)}\n\n{(await get_answer_test(text, message.from_user.id)).split("image_url")[0].strip()}', parse_mode='HTML')


@dp.message_handler(content_types=[types.ContentType.ANY])
async def handle_any_message(message: types.Message):
    await message.reply('Извините, данный тип сообщений не поддерживается')


def recognize(audio):
    model = model_repository.recognition_model()
    model.model = 'general'
    model.language = 'ru-RU'
    model.audio_processing_type = AudioProcessingType.Full
    result = model.transcribe_file(audio)
    return result[0].normalized_text


async def main():
    bot.texts = load_dictionary('texts.json')
    bot.user_settings = load_dictionary('users.json')
    configure_credentials(
        yandex_credentials=creds.YandexCredentials(
            api_key=getenv('AUTH')
        )
    )
    bot.sdk = AsyncYCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))
    await files_delete()
    bot.files = await files_create()
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
