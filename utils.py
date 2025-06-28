import json

def create_chunks(file_path='data.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    docs = []
    if file_path == 'data.json':
        for place in data['places']:
            all_info_text = f"""Название: 
    {place['title']}
    
    Описание: 
    {place['context']}
    
    image_url изображение:
    {place['image_url_v2']}
    
    Оценка объекта по 100-балльной шкале:
    {place['rate']}
    """

            if len(place['c_real']) > 1:
                all_info_text += f"""Координаты объекта:
    Широта (lat): {place['c_real'][0]}
    Долгота (lon): {place['c_real'][1]}
    """

            docs.append(all_info_text)
    elif file_path == 'news.json':
        docs = [str(i) for i in data['news']]
        for i in data['rss']:
            docs.append(f'Событие: {i["text"]}. Ссылки на фото ({len(i["photos"])} штук): {", ".join(i["photos"])}')
    else:
        docs = [str(i) for i in data['base']]
    return docs


def create_json_chunks(file_path='data.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data_chunks = []

    for place in data['places']:
        name = place['title']
        description = place['context']
        score = place['rate']
        link = place['image_url_v2']

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

