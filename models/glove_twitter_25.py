import traceback
import nltk
import re
from typing import List
from gensim.models import KeyedVectors

word2vec_pretrained = None
NLTK_MODEL_NAME = "punkt"
MODEL_NAME = "glove-twitter-25"
MODEL_PATH = f"forked_models/{MODEL_NAME}.kv"


def initialize_word2vec():
    global word2vec_pretrained
    try:
        nltk.download(NLTK_MODEL_NAME)
        word2vec_pretrained = KeyedVectors.load(MODEL_PATH)
        print("Word2Vec model loaded successfully")
    except Exception as e:
        print(f"Error initializing Word2Vec model: {str(e)}")
        print(traceback.format_exc())


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def get_word_embedding(word: str) -> List[float]:
    word = clean_text(word)
    if word in word2vec_pretrained:
        return word2vec_pretrained[word].tolist()
    else:
        return []


def get_embedding(phrase: str) -> List[float]:
    cleaned_phrase = clean_text(phrase)
    words = cleaned_phrase.split()
    word_vectors: List[List[float]] = [
        get_word_embedding(word) for word in words if get_word_embedding(word)
    ]

    if not word_vectors:
        return []

    if len(word_vectors) == 1:
        return word_vectors[0]

    avg_vector: List[float] = [sum(x) / len(word_vectors) for x in zip(*word_vectors)]
    return avg_vector
