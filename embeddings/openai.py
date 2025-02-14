from dotenv import load_dotenv
from typing import List
import openai
import os

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def generate_text_embeddings(text: str) -> List[float]:
    text: str = text.replace("\n", " ")
    embeddings: List[float] = (
        openai.embeddings.create(input=[text], model="text-embedding-3-small")
        .data[0]
        .embedding
    )
    return embeddings
