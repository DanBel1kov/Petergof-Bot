import json

import pymorphy2
import math
import re
import nltk
from nltk.stem.snowball import RussianStemmer
from numpy.core.defchararray import rfind
from yandex_cloud_ml_sdk import YCloudML
from chromadb.api import EmbeddingFunction
from os import getenv, remove
from chroma import create_or_update_chroma_collection, init_chroma
from dotenv import load_dotenv
from typing import List
import json

load_dotenv()

YANDEX_FOLDER_ID = getenv('FOLDER')
YANDEX_AUTH = getenv('AUTH')


def parse_json_output(text):
    text = text.strip()
    if '```json' in text:
        text = text[text.rfind('```json') + len('```json'): text.rfind('```')]
    else:
        text = text[text.find('```') + len('```'): text.rfind('```')]

    return text.strip()

def format_dialog(dialog):
    formatted_lines = []
    for message in dialog:
        if 'user' in message:
            formatted_lines.append("User: " + message['user'].strip())
        elif 'bot' in message:
            formatted_lines.append("Bot: " + message['bot'].strip())
    return "\n".join(formatted_lines)


def update_route(user_message, route_json, relevant_objects, model):
    objects_names = ", ".join([obj["name"] for obj in route_json])

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
ВАЖНО! Если пользователь не попросил уменьшить количество мест, то обязательно выведи 5 мест, если по контексту не ясно, что пользователь не хочет их видеть
ВАЖНО! Добавляй ответов всегда побольше чтобы пользователь мог выбрать сам (если только он не указал, что ему нужно меньше объектов)

Выведи только итоговый список мест для посещения. Если подходящих мест нет, предложи самые подходящие альтернативы

## Возможные релевантные объекты для пользователя: ##
{relevant_objects}

## Последнее сообщение пользователя: ##
{user_message}

## Объекты нашего музея, которые были рекомендованы к посещению ##
{objects_names}

Обязательно ответь списком, как я тебе дал на входе
"""

    result = model.run(prompt)
    return result.alternatives[0].text


def calculate_distance(coord1, coord2):
    lat1, lon1 = map(float, coord1)
    lat2, lon2 = map(float, coord2)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371000
    return R * c


def normilize_text(text):
    text = text.replace('ё', 'е').replace('Ё', 'е')
    text = re.sub(r'[^A-Za-zА-Яа-я]', '', text)
    return stem_text(text.lower())


def stem_text(text):
    stemmer = RussianStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())


def sort_json_by_distance(initial, json_list):
    return sorted(
        json_list,
        key=lambda obj: calculate_distance(
            initial, (float(obj['coordinates']['lat']), float(obj['coordinates']['lon']))
        )
    )


def filter_route_by_distance(route, max_distance=5000):
    if not route:
        return []
    filtered_route = [route[0]]
    for obj in route[1:]:
        obj_coords = (float(obj['coordinates']['lat']), float(obj['coordinates']['lon']))
        if any(calculate_distance(obj_coords, (
        float(existing['coordinates']['lat']), float(existing['coordinates']['lon']))) <= max_distance
               for existing in filtered_route):
            filtered_route.append(obj)
    return filtered_route


def generate_yandex_maps_route_url(coordinates, start_location_coordinates):
    route_param = '%2C'.join(start_location_coordinates) + '~'
    for coord in coordinates:
        route_param += (coord[0] + '%2C' + coord[1] + '~')
    base_url = "(https://yandex.ru/maps/?rtext="
    route_url = f"{base_url}{route_param[:-1]}&rtt=pd)"
    return f"[Ссылка Яндекс Карты]{route_url}"


def lemmatize_text(text):
    morph = pymorphy2.MorphAnalyzer()
    return ' '.join(morph.parse(word)[0].normal_form for word in text.split())


def suggest_route_by_gpt(model, route_description, link, dialog, last_message=None):
    dialogue_str = format_dialog(dialog)
    prompt = f"""
Вы ведёте живой и дружелюбный диалог с пользователем. Ты – отзывчивый и внимательный помощник музея Петергоф, который стремится сделать визит максимально интересным и удобным для пользователя. Ранее ты вел диалог с пользователем для выяснения его пожеланий:
{dialogue_str}

Исходя из этого диалога, ты подобрали следующий оптимальный маршрут:
{route_description}

Теперь кратко прокомментируй диалог и какой маршрут ты подобрал, а затем выведи его в виде булет списка.
Если в последнем сообщении пользователь, например, просит добавить объект, которого нет в списке (например, **Макдональдс**), объясните, что такой объект отсутствует на территории Петергофа и предложи альтернативное решение. Если запрос пользователя кажется не по теме или непонятным, сообщи, что ты не до конца понял запрос, и предложи возможный вариант маршрута с объяснением.
Общайся живо и поддерживающе, обязательно обращаясь к пользователю на «Вы» и не начиная сообщение с приветствия, если только пользователь не поздоровался.

Используй Markdown для Telegram следующим образом:
- Названия объектов оборачивайте в **двойные звёздочки**, чтобы выделять их жирным.
- Каждое описание объекта выводите с новой строки, чтобы текст не слипался: **Название объекта** – краткое описание.

Затем спроси, устраивает ли пользователя предложенный вариант маршрута, или требуется корректировка.
В конце сообщи, что подробности маршрута можно узнать по ссылке:
{link}
"""
    if last_message is not None:
        prompt += f" Ранее Вы уже предлагали маршрут, но пользователь решил его изменить. Вот его вариант: {last_message}"
    result = model.run(prompt)
    return result.alternatives[0].text



def change_route_by_gpt(model, route_description, link, last_message, user_dialog):
    dialogue_str = format_dialog(user_dialog)
    prompt = f"""
Ты ведёшь живой и дружелюбный диалог с пользователем. Ты – отзывчивый помощник музея Петергоф, который всегда старается сделать маршрут максимально удобным и интересным для пользователя. Ранее ты уже предлагал маршрут, но пользователь решил сменить или уточнить маршрут.

Вот история диалога с пользователем:
{dialogue_str}

Последнее сообщение пользователя: {last_message}

Новый маршрут, который ты подобрал:
{route_description}

ВНИМАНИЕ: Внимательно проанализируй предыдущий твой последний ответ, в котором ты рассказываешь про маршрут, новый маршрут и диалог с пользователем. Если новые предложения идентичны тем, что были предложены ранее, не повторяй их заново и не говори, что ты их добавил. Вместо этого сообщи, что маршрут не изменился, и попроси у пользователя дополнительные уточнения или пожелания.

Расскажи пользователю подробно, какие изменения ты внес относительно твоей прошлой рекомендации в маршрут (если изменения были произведены), чтобы он лучше соответствовал его последним пожеланиям. Общайся живо и разговорно, обращаясь к пользователю на «Вы». 

Если в последнем сообщении пользователь просит добавить объект, которого нет на территории (например, **Макдональдс**), сообщи, что такой объект не располагается на территории Петергофа и поэтому не может быть включён в маршрут, но предложи альтернативное решение. Если запрос пользователя кажется странным или не по теме, предупреди, что ты не до конца понял запрос, и предложи возможный вариант маршрута с объяснением.

Не начинай сообщение с приветствия, если только пользователь сам не поздоровался.

Используй в ответе Markdown для Telegram следующим образом:
- Названия объектов оборачивай в **двойные звёздочки**, чтобы выделять их жирным.
- Каждое описание объекта выводи с новой строки, чтобы текст не слипался:
  **Название объекта** – краткое описание.

Также сообщи, что маршрут можно изучить подробнее по ссылке:
{link}

"""
    print(prompt)
    result = model.run(prompt)
    response_text = result.alternatives[0].text
    return response_text

def check_on_positivity(model, user_dialogues, mentioned_objects):
    dialogue_str = format_dialog(user_dialogues)
    objects_str = ", ".join(mentioned_objects)

    prompt = f"""Ты умный гид музея Петергоф. В последнем сообщении пользователь захотел чтобы ты составил ему маршрут
Основываясь на диалоге с пользователем и уже вытащенных названий объектов из этого диалога тебе нужно определить какие объекты пользователь хочет посетить (которые должны быть в рекомендации по маршруту)

Обрати внимание на следующие моменты:
1. Выбирай для маршрута только те объекты, о которых пользователь говорит в положительном ключе. Если объект упоминается в отрицательном (например, "не нравится", "не надо"), то он исключается.
2. Если пользователь явно указывает, что маршрут должен состоять только из одного объекта (например, "Построй маршрут до дома игральных карт, других объектов не надо"), то в маршрут должен быть включен только этот объект, а остальные — исключены.
3. Если упоминаются граничные условия (начало или конец маршрута), то отрази их в выводе, иначе оставь соответствующие поля пустыми.
4. Выводи только те объекты, которые пользователь действительно хочет увидеть в своем маршруте, с учетом всех указанных ограничений.
5. Учти, что самая важная информация с пожеланием пользователя содержится в его самом последнем вопросе
6. В твоем ответе объекты не должны повторяться (только если пользователь сам не попросит начать и закончить маршрут в одной точке) 
7. Если добавляешь объект как начальную или конечную точку, то его не должно быть в списке других объектов

Вот доступная тебе информация:
<Диалог>
{dialogue_str}
</Диалог>

<Объекты, которые упоминаются в тексте> 
{objects_str}
</Объекты, которые упоминаются в тексте>

Первом делом подробно объясни почему пользователь хочет или не хочет видеть каждый объект из данного тебе списка объектов
После этого в конце ответа дай свои рассуждения насчет того, какие объекты хочет увидеть пользователь в своем будущем маршруте
В конце ответа выведи json в следующем формате:
```json
{{
    "thoughts": "<5-10 предложений с рассуждениями на тему того, нужно ли начинать маршрут с определенного объекта, закачивать в определенном объекте и какие еще объекты исходя из диалога можно включить в маршрут. Нужно написать про каждый объект из диалога>"
    "start": <название объекта для старта или пустой строкой>,
    "end": <название объекта для завершения или пустой строкой>,
    "others": [названия других объектов через запятую, которые хочет увидеть пользователь в следующем маршруте, может быть пустым списком]
}}
Важно! Не выводи ничего после json и важно использовать ```json перед самим джейсоном
```
"""

    result = model.run(prompt)
    response_text = result.alternatives[0].text.strip()
    # print('Response----------')
    # print(response_text)
    response_json = parse_json_output(response_text)
    try:
        response_json = json.loads(response_json)
    except json.JSONDecodeError:
        response_json = {"start": "", "end": "", "others": []}
    return response_json


def cluster_by_distance(objects, max_distance=5000):
    clusters = []
    for obj in objects:
        added = False
        for cluster in clusters:
            if any(calculate_distance(
                    (float(obj['coordinates']['lat']), float(obj['coordinates']['lon'])),
                    (float(existing['coordinates']['lat']), float(existing['coordinates']['lon']))
            ) <= max_distance for existing in cluster):
                cluster.append(obj)
                added = True
                break
        if not added:
            clusters.append([obj])
    return clusters


def select_cluster(clusters, min_objects):
    suitable = [cluster for cluster in clusters if len(cluster) >= min_objects]
    if suitable:
        return max(suitable, key=lambda c: len(c))
    return max(clusters, key=lambda c: len(c))


def build_route_with_boundaries(candidate_route, start_obj=None, end_obj=None,
                                initial_coordinates=["59.891802", "29.913220"]):
    if start_obj is None and end_obj is None:
        return candidate_route

    filtered_route = candidate_route.copy()
    if start_obj:
        filtered_route = [obj for obj in filtered_route if obj["name"] != start_obj["name"]]
    if end_obj:
        filtered_route = [obj for obj in filtered_route if obj["name"] != end_obj["name"]]

    if start_obj:
        current_coord = (float(start_obj['coordinates']['lat']), float(start_obj['coordinates']['lon']))
    else:
        current_coord = (float(initial_coordinates[0]), float(initial_coordinates[1]))

    route_ordered = []
    remaining = filtered_route.copy()
    while remaining:
        next_obj = min(
            remaining,
            key=lambda obj: calculate_distance(
                current_coord,
                (float(obj['coordinates']['lat']), float(obj['coordinates']['lon']))
            )
        )
        route_ordered.append(next_obj)
        current_coord = (float(next_obj['coordinates']['lat']), float(next_obj['coordinates']['lon']))
        remaining.remove(next_obj)

    if start_obj:
        route_ordered.insert(0, start_obj)
    if end_obj:
        route_ordered.append(end_obj)

    return route_ordered


def get_route_suggestion(user_dialogues, data_chunks, initial_coordinates=["59.891802", "29.913220"],
                         objects_number=5, initial_max_distance=1000, max_distance_limit=10000, distance_increment=500):
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    model = model.configure(temperature=0.2)

    all_objects_names = []
    mentioned_objects = []
    for dialogue in user_dialogues:
        dialog_key = 'user' if "user" in dialogue else 'bot'
        lemmatized_text = normilize_text(dialogue[dialog_key])
        for chunk in data_chunks:
            lemmatized_name = normilize_text(chunk["name"])
            if lemmatized_name in lemmatized_text:
                all_objects_names.append(chunk["name"])
                mentioned_objects.append(chunk)

    positivity_result = check_on_positivity(model, user_dialogues, all_objects_names)
    start_name = positivity_result.get("start", "")
    end_name = positivity_result.get("end", "")
    good_objects_names = positivity_result.get("others", [])

    # print(positivity_result)
    # print('------------------------')
    # print(f"start: {start_name}, end: {end_name}, others: {good_objects_names}")

    filtered_objects = [obj for obj in mentioned_objects if normilize_text(obj["name"]) in
                        [normilize_text(name) for name in good_objects_names]]

    candidate_route = filtered_objects.copy()

    user_message = user_dialogues[-1]["user"]
    user_dialog = "\n".join([d.get("user", d.get("bot", "")) for d in user_dialogues])

    chroma_collection = init_chroma()
    collection_count = chroma_collection.count()
    if collection_count == 0:
        print("Creating a new collection")
        create_or_update_chroma_collection(chroma_collection)

    all_objects_description = " ".join([junk['name'] + ' ' + junk['description'][:100] for junk in data_chunks])
    rephrazed_mes = rephrase(model, user_message, user_dialog, all_objects_description)
    # print(rephrazed_mes)
    additional_candidates = ''
    if rephrazed_mes != '':
        retriever_results = chroma_collection.query(
            query_texts=[rephrazed_mes],
            n_results=20
        )
        retrieved_docs = retriever_results.get('documents', [[]])[0]
        # Обрезаем каждый документ до 100 символов, чтобы уменьшить объём контекста
        retrieved_docs = [doc[:400] for doc in retrieved_docs]
        additional_candidates = "\n\n".join(retrieved_docs)

    current_old_names = [obj["name"] for obj in candidate_route]

    # print("Finding new objects----------------------------------------")
    new_objects_response = select_new_routes_by_gpt(model, additional_candidates, current_old_names, user_message,
                                                    user_dialog)
    norm_new_names = normilize_text(new_objects_response)
    new_objects = []
    for chunk in data_chunks:
        lemmatized_chunk_name = normilize_text(chunk["name"])
        if norm_new_names and lemmatized_chunk_name in norm_new_names and lemmatized_chunk_name not in normilize_text(
                " ".join(current_old_names)):
            new_objects.append(chunk)

    candidate_route += new_objects[:max(0, objects_number - len(candidate_route))]
    candidate_route = sort_json_by_distance(initial_coordinates, candidate_route)

    start_obj = next(
        (obj for obj in candidate_route if normilize_text(obj["name"]) == normilize_text(start_name)),
        None
    ) if start_name else None

    end_obj = next(
        (obj for obj in candidate_route if normilize_text(obj["name"]) == normilize_text(end_name)),
        None
    ) if end_name else None

    # Если объект для финиша не найден в candidate_route, ищем его в data_chunks и добавляем
    if end_name and not end_obj:
        end_obj = next(
            (obj for obj in data_chunks if normilize_text(obj["name"]) == normilize_text(end_name)),
            None
        )
        if end_obj:
            candidate_route.append(end_obj)

    final_route = build_route_with_boundaries(candidate_route, start_obj=start_obj, end_obj=end_obj,
                                              initial_coordinates=initial_coordinates)

    route_description = ""
    for index, obj in enumerate(final_route):
        route_description += f"{index + 1}. {obj['name']} - {obj['description'][:250]}...\n"
    coordinates_list = [[obj["coordinates"]["lat"], obj["coordinates"]["lon"]] for obj in final_route]
    link = generate_yandex_maps_route_url(coordinates_list,
                                          initial_coordinates if start_obj is None else [
                                              start_obj["coordinates"]["lat"], start_obj["coordinates"]["lon"]])
    route_description = suggest_route_by_gpt(model, route_description, link, user_dialogues)
    return route_description, final_route


def select_new_routes_by_gpt(model, retrieved_context, current_old_names, user_message, user_dialog, all_old_names=[]):
    user_dialog = format_dialog(user_dialog)
    prompt = f"""
## Основная инструкция:

1. Ты — опытный гид музея-заповедника Петергоф. Твоя задача — составить маршрут по объектам, строго следуя пожеланиям пользователя из последнего сообщения. **Никогда не добавляй объекты, которых пользователь явно не запрашивал, и не предлагай альтернативные варианты, если они не указаны в запросе.**

2. У тебя есть следующая информация:
   2.1. История диалога с пользователем.
   2.2. Текущий список объектов, уже включённых в маршрут, а также, если имеется, предыдущий маршрут.
   2.3. Список новых объектов-кандидатов, из которых можно выбрать дополнительные.

3. Твои действия:
   3.1. Внимательно проанализируй последнее сообщение пользователя, чтобы определить, какие объекты он хочет добавить.
   3.2. Если сообщение содержит конкретные названия объектов для добавления (например, "добавь X, Y"), выбери только их.
   3.3. Если сообщение содержит общее указание типа "добавь ещё один объект", выбери из списка кандидатов один наиболее релевантный объект, соответствующий тематике уже включённых объектов.
   3.4. Никогда не добавляй объекты, которые не были явно запрошены или подразумеваются запросом пользователя.
   3.5. Строго выполняй пожелания пользователя: если запрос сводится только к добавлению объектов, не предлагай никаких альтернативных маршрутов.

4. Формат итогового ответа:
   - Вывод должен быть строго в формате JSON с двумя полями:
     - "thoughts": В этом поле напиши:
         a) Что хочет пользователь (его запрос).
         b) Анализ текущего состояния маршрута с учётом уже включённых объектов.
         c) Недостающие элементы для полного удовлетворения запроса.
         d) Объекты из списка кандидатов, которые могут быть добавлены для удовлетворения запроса.
         Эти пункты должны быть изложены в одном логичном тексте, состоящем примерно из 5 предложений.
     - "new_objects": Названия объектов, которые нужно добавить (через запятую) или пустая строка ('') если изменений не требуется.

5. Если в последнем сообщении отсутствует явное указание на добавление объектов, поле "new_objects" должно быть пустым.

<История диалога>
{user_dialog}
</История диалога>

<Объекты, которые раньше были включены в маршрут>
{", ".join(all_old_names)}
<Объекты, которые раньше были включены в маршрут>

<Объекты, которые уже включены в маршрут>
{", ".join(current_old_names)}
</Объекты, которые уже включены в маршрут>

<Новые кандидаты для маршрута>
{retrieved_context}
</Новые кандидаты для маршрута>

Твоя задача:
1. Проанализируй кандидатов, сопоставляя их с фрагментами последнего сообщения и историей диалога.
2. Если в последнем сообщении присутствует явное указание на добавление объектов, выбери только те, которые удовлетворяют запросу, без лишних дополнений.
3. После анализа выведи итоговый JSON-ответ строго в следующем формате (без дополнительных пояснений и без текста после JSON):
```json
{{
    "thoughts": "<описание в 5 предложениях, где сначала указано, чего хочет пользователь, затем проведён анализ текущего маршрута, обозначены недостающие элементы и перечислены объекты для добавления>",
    "new_objects": "<названия объектов через запятую или пустая строка ('')>"
}}
"""

    result = model.run(prompt)
    output = result.alternatives[0].text
    # print(output)
    try:
        json_objects = json.loads(parse_json_output(output))
    except json.JSONDecodeError:
        json_objects = {'new_objects': ''}
    new_objects = json_objects["new_objects"]
    # print("Extracted objects -------------------")
    # print(new_objects)
    return new_objects


def filter_unwanted_objects(message, current_route_json, model):
    objects_names = ", ".join([obj["name"] for obj in current_route_json])
    prompt = f"""
## Основная инструкция:
Ты выступаешь в роли опытного гида музея Петергоф. Твоя задача — проанализировать последнее сообщение пользователя и текущий список объектов маршрута, чтобы определить, какие объекты следует оставить в маршруте. Обрати особое внимание на тон последнего сообщения: если оно носит характер добавления (например, "добавь еще дом игральных карт"), то оставь все уже включённые объекты и дополнительно выбери новые, соответствующие запросу; если же сообщение содержит ограничительные указания (например, "других объектов не надо" или "только до дома игральных карт"), то исключи из маршрута объекты, не удовлетворяющие последним требованиям.

Пользователь написал: "{message}"
Текущий маршрут содержит следующие объекты: {objects_names}

Требования:
1. Если последнее сообщение начинается с "добавь", трактуй его как запрос вывести все текущие объекты
2. Если же в последнем сообщении явно указано ограничение (например, "других объектов не надо", "только до ..."), пересмотри состав маршрута, оставив только объекты, подтверждающие новые требования.
3. Для каждого объекта проведи краткий анализ, опираясь на конкретные фрагменты последнего сообщения, объясни, почему он должен остаться или быть исключён.
4. Если пользователь указывает конкретное число объектов для маршрута, соблюдай это требование, но учти, что ты не можешь добавлять новые объекты, а можешь только удалять
5. После полного анализа выведи только итоговый ответ в формате JSON, где поле "remaining_objects" содержит через запятую названия объектов, которые должны остаться в маршруте. Никакого дополнительного текста после JSON выводить не нужно.

Формат итогового ответа:
```json
{{
    "thoughts": "<5 предложений с рассуждениями на тему того, какие объекты действительно нужны пользователю исходя из диалога. Нужно написать про каждый объект из диалога>"
    "remaining_objects": "<названия объектов через запятую или пустая строка ('')>"
}}
"""

    result = model.run(prompt)
    res = result.alternatives[0].text
    print("Updated objects -------------------")
    print(res)

    res = parse_json_output(res)
    res_json = json.loads(res)
    return res_json['remaining_objects']


def rephrase(model, last_message, user_dialog, objects_info):
    prompt = f"""Ты бот музея Петергоф.
Твоя задача — переформулировать запрос от пользователя на основе его последнего сообщения, учитывая общий контекст диалога, так чтобы запрос содержал только позитивные описания и был направлен на поиск информации и объектов в векторной базе данных, то есть объектов, которые пользователь хотел бы увидеть (например, красивые экспозиции, интересные музеи, впечатляющие архитектурные объекты, уютные выставочные залы и т.д.).

Используй следующую информацию:
<Контекст диалога>
{user_dialog}
</Контекст диалога>

<Информация об объектах>
{objects_info}
</Информация об объектах>

<Последнее сообщение пользователя>
{last_message}
</Последнее сообщение пользователя>

Если в последнем сообщении пользователя говорится исключительно об удалении объектов без замены, твой ответ должен быть пустой строкой. В противном случае переформулируй запрос так, чтобы он был максимально точным, информативным и не содержал негативных высказываний. Запрос должен содержать только позитивные аспекты, описывающие, какую информацию и какие объекты нужно искать.
"""
    result = model.run(prompt)
    res = result.alternatives[0].text.strip()

    print(f"Good query: {res}")
    return res




def change_route_by_message(message, current_route_json, data_chunks, user_dialog,
                            initial_coordinates=["59.891802", "29.913220"],
                            objects_number=20, initial_max_distance=500, max_distance_limit=1000,
                            distance_increment=100):
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    model = model.configure(temperature=0.2)

    chroma_collection = init_chroma()
    collection_count = chroma_collection.count()
    if collection_count == 0:
        print("Creating a new collection")
        create_or_update_chroma_collection(chroma_collection)

    all_objects_description = " ".join([junk['name'] + ' ' + junk['description'][:100] for junk in data_chunks])
    rephrazed_mes = rephrase(model, message, user_dialog, all_objects_description)
    retriever_results = chroma_collection.query(
        query_texts=[rephrazed_mes],
        n_results=30
    )
    retrieved_docs = retriever_results.get('documents', [[]])[0]
    # Обрезаем каждый документ до 400 символов, чтобы уменьшить объём контекста
    retrieved_docs = [doc[:400] for doc in retrieved_docs]
    retrieved_context = "\n\n".join(retrieved_docs)

    # print('Current obj----------------')
    # for data in current_route_json:
    #     print(data['name'])
    filtered_old_names = filter_unwanted_objects(message, current_route_json, model)

    # print("Good places (старые объекты):")
    # print(filtered_old_names)

    norm_old_names = normilize_text(filtered_old_names)
    old_objects = []
    for chunk in data_chunks:
        lemmatized_chunk_name = normilize_text(chunk["name"])
        if lemmatized_chunk_name in norm_old_names:
            old_objects.append(chunk)
    current_old_names = [obj["name"] for obj in old_objects]

    # print("Старые объекты, которые останутся в маршруте:")
    # print(current_old_names)

    old_names = ", ".join([obj["name"] for obj in current_route_json])
    new_objects_response = select_new_routes_by_gpt(model, retrieved_context, current_old_names, message, user_dialog, old_names)
    # print("Ответ на выбор новых объектов:")
    # print(new_objects_response)
    norm_new_names = normilize_text(new_objects_response)
    new_objects = []
    for chunk in data_chunks:
        lemmatized_chunk_name = normilize_text(chunk["name"])
        # Добавляем только новые объекты, которых ещё нет среди старых
        if (lemmatized_chunk_name in norm_new_names and
                lemmatized_chunk_name not in normilize_text(" ".join(current_old_names))):
            new_objects.append(chunk)

    num_needed = max(0, objects_number - len(old_objects))
    filtered_new_objects = []
    if num_needed > 0 and new_objects:
        current_threshold = initial_max_distance
        clusters = cluster_by_distance(new_objects, max_distance=current_threshold)
        selected_cluster_new = select_cluster(clusters, num_needed)
        while len(selected_cluster_new) < num_needed and current_threshold < max_distance_limit:
            current_threshold += distance_increment
            clusters = cluster_by_distance(new_objects, max_distance=current_threshold)
            selected_cluster_new = select_cluster(clusters, num_needed)
        filtered_new_objects = sort_json_by_distance(initial_coordinates, selected_cluster_new)
    else:
        filtered_new_objects = []

    final_route = old_objects + filtered_new_objects

    route_description = ""
    for index, obj in enumerate(final_route):
        route_description += f"{index + 1}. {obj['name']} - {obj['description'][:500]}...\n"
    coordinates_list = [[obj["coordinates"]["lat"], obj["coordinates"]["lon"]] for obj in final_route]
    link = generate_yandex_maps_route_url(coordinates_list, initial_coordinates)
    route_description = change_route_by_gpt(model, route_description, link, message, user_dialog)

    return route_description, final_route

