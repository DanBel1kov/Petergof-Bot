import json

def create_chunks(file_path='data.json'):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    docs = []

    for place in data['places']:
        all_info_text = f"""Название: 
{place['title']}

Информация об объекта: 
{place['context']}

Оценка объекта (по 100 бальной шкале):
{place['rate']}

Ссылка на объект:
{place['full_url']}
"""

        if len(place['c_real']) > 1:
            all_info_text += f"""Координты объекта:
Широта (lat): {place['c_real'][0]}
Долгота (lon): {place['c_real'][1]}
"""

        docs.append(all_info_text)

    return docs


def create_json_chunks(file_path='data.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data_chunks = []

    for place in data['places']:
        name = place['title']
        description = place['context']
        score = place['rate']
        link = place['full_url']

        if len(place['c_real']) > 1:
            coordinates = {
                'lat': place['c_real'][0],
                'lon': place['c_real'][1]
            }
        else:
            continue

        chunk = {
            'name': name,
            'description': description,
            'score': score,
            'link': link,
            'coordinates': coordinates
        }

        data_chunks.append(chunk)

    return data_chunks

user_dialogues = [
    {"user": "Расскажи мне про нижний парк"},
    {"bot": "Ну он норм"}
]

def parse_dialogue(dialogue):
    dialogue_str = ""

    for item in dialogue:
        if "user" in item:
            dialogue_str += "Собщение пользователя:\n"
            dialogue_str += item["user"] + "\n"
        else:
            dialogue_str += "Собщение бота:\n"
            dialogue_str += item["bot"] + "\n"

        dialogue_str += "---------------------------------------\n\n"

    return dialogue_str


