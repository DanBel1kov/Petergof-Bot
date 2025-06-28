from utils import create_chunks
from chromadb.api import EmbeddingFunction
from typing import List
from os import getenv, remove
from yandex_cloud_ml_sdk import AsyncYCloudML, YCloudML
import chromadb
from raptor import DataEnlarger
from dotenv import load_dotenv
load_dotenv()


def get_links(text):
    raw_links = re.findall(r'https?://[^\s]+', text)
    links = [re.sub(r'[).,:;]+$', '', url) for url in raw_links]
    return links

import re

def create_or_update_chroma_collection(collection, data_path: str = 'data.json'):
    """
    Reads data from data.json, splits texts into chunks,
    and adds them to the Chroma collection.
    If the collection is not empty and needs to be recreated,
    either clear it manually (collection.delete(...)) or use different logic.
    """

    sdk = YCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))
    embd = sdk.models.text_embeddings("doc")
    model = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    de = DataEnlarger(llm=model, embd=embd, data_path=data_path)
    chunks = de.chunks

    for idx, chunk in enumerate(chunks):
        doc_id = f"doc_{idx}"
        image_url = get_links(chunk)
        image_url = image_url[0] if len(image_url) > 0 else False

        collection.add(
            documents=[chunk],
            metadatas=[{"image_url": image_url}],
            ids=[doc_id]
        )

class YandexEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embd_model):
        # embd_model = sdk.models.text_embeddings("doc")
        self.embd_model = embd_model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            result = self.embd_model.run(text[:4000])
            # result.embedding - это кортеж (tuple),
            # Chroma нужно list[float], конвертируем:
            emb_vector = list(result.embedding)
            vectors.append(emb_vector)
        return vectors


def init_chroma(remote: bool = False, name='data'):
    """
    Инициализирует коллекцию Chroma (получает или создаёт).
    Устанавливает функцию для эмбеддингов (yandex_embeddings).
    """
    sdk = YCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))
    embd_model = sdk.models.text_embeddings("doc")
    if remote:
        client = chromadb.HttpClient(host='158.160.179.7', port=8000)
    else:
        client = chromadb.PersistentClient(path="chroma_db")
    print(f'collections: {client.list_collections()}')
    embedding_fn = YandexEmbeddingFunction(embd_model)
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn
    )
    print(collection)
    print(collection.count())
    return collection
