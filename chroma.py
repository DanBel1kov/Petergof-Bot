from utils import create_chunks
from chromadb.api import EmbeddingFunction
from typing import List
from os import getenv, remove
from yandex_cloud_ml_sdk import AsyncYCloudML, YCloudML
import chromadb


YANDEX_FOLDER_ID = getenv('FOLDER_ID', 'b1g10f66fjjfuqg9ehje')
YANDEX_AUTH = getenv('YANDEX_AUTH', 'AQVN0zMfZzvnaQ_qeJz4mtiu3yYeTKJe2aupo1z5')

import re

def create_or_update_chroma_collection(collection):
    """
    Reads data from data.json, splits texts into chunks,
    and adds them to the Chroma collection.
    If the collection is not empty and needs to be recreated,
    either clear it manually (collection.delete(...)) or use different logic.
    """

    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    chunks = create_chunks(file_path='data.json')

    for idx, chunk in enumerate(chunks):
        doc_id = f"doc_{idx}"

        found_urls = re.findall(url_pattern, chunk)
        image_url = found_urls[0] if found_urls else None

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
            result = self.embd_model.run(text)
            # result.embedding - это кортеж (tuple),
            # Chroma нужно list[float], конвертируем:
            emb_vector = list(result.embedding)
            vectors.append(emb_vector)
        return vectors


def init_chroma():
    """
    Инициализирует коллекцию Chroma (получает или создаёт).
    Устанавливает функцию для эмбеддингов (yandex_embeddings).
    """
    sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_AUTH)
    embd_model = sdk.models.text_embeddings("doc")
    client = chromadb.PersistentClient(path="chroma_db")

    embedding_fn = YandexEmbeddingFunction(embd_model)

    collection = client.get_or_create_collection(
        name="peterhof_docs",
        embedding_function=embedding_fn
    )
    return collection