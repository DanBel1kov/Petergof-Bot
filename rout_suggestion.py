import pymorphy2
import math
from yandex_cloud_ml_sdk import YCloudML
from utils import parse_dialogue
import re
# from bot import get_answer, init_chroma
import chromadb
from yandex_cloud_ml_sdk.search_indexes import StaticIndexChunkingStrategy, TextSearchIndexType
from dotenv import load_dotenv
load_dotenv()
from chromadb.api import EmbeddingFunction
from typing import List
from os import getenv, remove
from chroma import create_or_update_chroma_collection, init_chroma
import nltk
from nltk.stem.snowball import RussianStemmer


YANDEX_FOLDER_ID = getenv('FOLDER_ID')
YANDEX_AUTH = getenv('YANDEX_AUTH')


def calculate_distance(coord1, coord2):
    return math.sqrt((float(coord1[0]) - float(coord2[0])) ** 2 + (float(coord1[1]) - float(coord2[1])) ** 2)

def normilize_text(text):
    text = text.replace('ё', 'е')
    text = text.replace('Ё', 'е')
    text = re.sub(r'[^A-Za-zА-Яа-я]', '', text)

    return stem_text(text.lower())


def sort_json_by_distance(initial, json_list):

    sorted_json_list = sorted(json_list, key=lambda obj: calculate_distance(
        initial, (float(obj['coordinates']['lat']), float(obj['coordinates']['lon']))
    ))

    return sorted_json_list


def generate_yandex_maps_route_url(coordinates,start_location_coordinates):
    """
    Генерация URL маршрута в Яндекс.Картах с использованием текстовых адресов.
    """

    route_param = '%2С'.join(start_location_coordinates) + '~'
    for coord in coordinates:
        route_param += (coord[0] + '%2C' + coord[1] + '~')


    base_url = "(https://yandex.ru/maps/?rtext="
    route_url = f"{base_url}{route_param[:-1]}&rtt=auto)"
    return f"[Ссылка Яндекс Карты]{route_url}"


def lemmatize_text(text):
    morph = pymorphy2.MorphAnalyzer()
    return ' '.join(morph.parse(word)[0].normal_form for word in text.split())


def stem_text(text):
    stemmer = RussianStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())



def suggest_route_by_gpt(model, route_description, link, last_message = None):

    prompt = f"""
Ты умный бот помощник по музею Петергоф. Тебя попросили подобрать оптимальный маршрут исходя из диалога и лучших объектов по рейтингу, которые сейчас работают
Ты не должен здороваться, так как уже ведешь диалог
Ты подобрал следующие объекты:
{route_description}

Теперь тебе нужно очень коротко рассказать про маршрут пользователю и спросить нужно ли что-то изменить (1 предложение на каждый объект). 
Если он хочет посмотреть конкретные объекты, то пусть он напишет и мы добавим их в маршрут (или уберем не интересующие)
Также скажи, что он может изучить маршрут по ссылке ниже:
{link}
"""

    if last_message is not None:
        prompt += f"Последнее сообщение пользователя, которое обязательно нужно обыграть в начале ответа. Ты должен сказать, что понял и поменял маршрут или добавил новый объект: {last_message}"

    result = model.run(prompt)

    return result.alternatives[0].text



def check_on_positivity(model, user_dialogues, mentioned_objects):
    dialogue_str = parse_dialogue(user_dialogues)

    objects_str = ", ".join(mentioned_objects)

    prompt = f"""Ты умный гид музея Петергоф и твоя задача понять какие объекты пользователь хочет увидеть и какие нет.
Ты получишь диалог пользователя с ботом и объекты про которые говорилось в переписке. 
Твоя задача убрать из списка те объекты, которые пользователь возможно не хочет посмотреть. 
Ты должен понять негативный контекст, и убрать только те объекты, которые явно относятся к этому негативному контексту
Ты должен вывести объекты в таком же формате, как они тебе и подавались, но без объектов, которые не хочет смотреть пользователь. 
Выведи только объекты через запятую, не меняй их окончание или большие буквы.
Если запрос пользователя включает просьбу удалить что-то, удали объекты, которые относятся к его запросу 

## Диалог с пользователем: ##
{dialogue_str}

## Объекты нашего музея, про которые говорилось в переписке: ##
{objects_str}
"""
    result = model.run(prompt)

    return result.alternatives[0].text.split(", ")



def get_route_suggestion(user_dialogues, data_chunks, initial_coordinates = ["59.891802", "29.913220"], objects_number = 5):

    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )

    model = sdk.models.completions(model_name="yandexgpt")
    model = model.configure(temperature=0.1)

    mentioned_objects = []
    all_objects_names = []
    for dialogue in user_dialogues:

        dialog_key ='user'
        if "user" not in dialogue:
            dialog_key = 'bot'

        lemmatized_user_text = normilize_text(dialogue[dialog_key])
        for chunk in data_chunks:
            lemmatized_chunk_name = normilize_text(chunk["name"])

            if lemmatized_chunk_name in lemmatized_user_text:
                all_objects_names.append(lemmatized_chunk_name)
                mentioned_objects.append(chunk)

    good_objects = check_on_positivity(model, user_dialogues, all_objects_names)
    mentioned_objects = list(filter(lambda x: normilize_text(x["name"]) in good_objects, mentioned_objects))

    bad_objects = set(all_objects_names) - set(good_objects)

    additional_objects = sorted(data_chunks, key=lambda x: x["score"], reverse=True)
    additional_objects = list(filter(lambda x: normilize_text(x["name"]) not in bad_objects, additional_objects))
    additional_objects = [obj for obj in additional_objects if obj not in mentioned_objects][:objects_number]

    route = mentioned_objects + additional_objects

    route_sorted = sort_json_by_distance(initial_coordinates, route)

    route_description = ""
    for index, obj in enumerate(route_sorted):
        route_description += f"{index + 1}. {obj['name']} - {obj['description'][:250]}...\n"

    coordinates_list = [[obj["coordinates"]["lat"], obj["coordinates"]["lon"]] for obj in route_sorted]

    link = generate_yandex_maps_route_url(coordinates_list, initial_coordinates)

    route_description = suggest_route_by_gpt(model, route_description, link)

    return route_description, route_sorted



def update_route(user_message, route_json, relevant_objects, model):

    objects_names = ", ".join([obj["name"] for obj in route_json])

    print("mentioned objects:")
    print(objects_names)

    prompt = f"""Ты умный гид музея Петергоф, и твоя задача — понять, какие объекты пользователь хочет увидеть, а какие нет.

Ты получишь список объектов, которые были предложены пользователю, и его последнее сообщение, в котором он хочет что-то поменять или добавить. Также тебе будет предложено пару дополнительных объектов с их описаниями, чтобы заменить какие-то из списка (если надо) или просто добавить к имеющимся.

ЗАПОМНИ: ты можешь говорить только про объекты из сообщения.

Желание пользователя может включать в себя:

1) Удалить объект из маршрута. Если пользователю не нравятся определённые объекты, это может быть явно выражено или ясно из контекста. В этом случае нужно удалить такие объекты из списка и вернуть его в том же формате, в котором они были даны.

2) Добавить объект в список. Если пользователь изъявляет желание увидеть что-то ещё, такие как "Добавь музей карт", добавь указанные объекты к текущему маршруту и не удаляй существующие. Важно: не переходи к заменам и не удаляй текущие объекты, если об этом нет явного указания.

3) Удалить объект и добавить ещё что-то в маршрут. Если же пользователь говорит о нежелательных объектах и высказывает желание добавить другие, удали упомянутые объекты и добавь указанные новые, сохраняя оставшиеся.

Примеры работы:
- Если пользователь пишет "Добавь музей карт", текущий список должен остаться прежним, просто с добавлением "музей карт".
- При отсутствии конкретного указания на удаление, ничто не должно удаляться.

Определи, какой из трех случаев наиболее вероятен, следуй инструкциям по нему и выведи только одну строку с итоговыми названиями объектов.
- Важно избегать любых отсылок к другим темам или ресурсам, кроме органов управления объектами.

ВАЖНО! Понять из сообщения пользователя, какие объекты нужно удалить, а какие добавлять, чтобы его маршрут стал более релевантным.

Выведи только итоговый список мест для посещения. Если подходящих мест нет, верни пустую строку.

## Возможные релевантные объекты для пользователя: ##
{relevant_objects}

## Последнее сообщение пользователя: ##
{user_message}

## Объекты нашего музея, которые были рекомендованы к посещению ##
{objects_names}

Обязательно ответь списком, как я тебе дал на входе
"""
#     prompt = f"""
# Ты — виртуальный гид музея Петергоф. Твоя задача — корректировать список объектов для посещения по запросу пользователя.
#
# У тебя есть:
# - Список предложенных объектов.
# - Сообщение пользователя, в котором он просит внести изменения.
# - Дополнительные объекты для возможного добавления.
#
# Следуй этим простым правилам:
#
# 1) Добавление объектов. Если пользователь говорит, что хочет добавить что-то (например, "Добавь музей карт"), просто добавь это к текущему списку без удаления уже предложенных объектов.
#
# 2) Удаление объектов. Если пользователь явно говорит, что что-то ему не нравится или не хочет видеть, удали эти объекты из списка.
#
# 3) Замена объектов. Если пользователь просит внести изменения и уточняет, что нужно убрать или поменять, замени ненужные объекты на новые из предложенного списка.
#
# Всегда проверяй, ясно ли указания пользователя, и работай только с указанными объектами.
#
# Выводи только итоговый список объектов. Если нет подходящих объектов, выведи пустую строку.
#
# ## Примеры:
# - Если пользователь просит добавить только один объект, просто дополните текущий список им.
# - Если пользователь хочет что-то убрать и что-то добавить, выполняйте соответствующие изменения.
#
# ## Возможные объекты: ##
# {relevant_objects}
#
# ## Сообщение пользователя: ##
# {user_message}
#
# ## Текущий список объектов: ##
# {objects_names}
#
# """

    result = model.run(prompt)
    return result.alternatives[0].text


def change_route_by_message(message, current_route_json, data_chunks, initial_coordinates = ["59.891802", "29.913220"]):

    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )

    model = sdk.models.completions(model_name="llama")
    model = model.configure(temperature=0.1)

    chroma_collection = init_chroma()
    collection_count = chroma_collection.count()

    # for idx in range(collection_count):
    #     chroma_collection.delete(f"doc_{idx}")

    if collection_count == 0:
        print("Creating a new collection")
        create_or_update_chroma_collection(chroma_collection)


    results = chroma_collection.query(
        query_texts=[message],
        n_results=2
    )


    retrieved_docs = results.get('documents', [[]])[0]
    # for doc in retrieved_docs:
    #     print(doc)
    #     print("------------")
    relevant_context = "\n\n".join(retrieved_docs)

    updated_object_names = update_route(message, current_route_json, relevant_context, model)
    print("____________________________")
    print("updated_object_names:", updated_object_names)
    normilized_names = normilize_text(updated_object_names)
    print(normilized_names)

    updated_objects_chunks = []
    updated_objects_names = []
    for chunk in data_chunks:
        lemmatized_chunk_name = normilize_text(chunk["name"])
        if lemmatized_chunk_name in normilized_names:
            updated_objects_names.append(lemmatized_chunk_name)
            updated_objects_chunks.append(chunk)

    print("Updated names in chunks:")
    print(updated_objects_names)
    route_sorted = sort_json_by_distance(initial_coordinates, updated_objects_chunks)

    route_description = ""
    for index, obj in enumerate(route_sorted):
        route_description += f"{index + 1}. {obj['name']} - {obj['description'][:250]}...\n"

    coordinates_list = [[obj["coordinates"]["lat"], obj["coordinates"]["lon"]] for obj in route_sorted]

    link = generate_yandex_maps_route_url(coordinates_list, initial_coordinates)

    route_description = suggest_route_by_gpt(model, route_description, link, message)

    return route_description, route_sorted

















