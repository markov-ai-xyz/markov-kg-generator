import spacy
import traceback

spacy_model = None
SPACY_MODEL_NAME = "en_core_web_sm"


def initialize_spacy():
    global spacy_model
    try:
        spacy_model = spacy.load(SPACY_MODEL_NAME)
        print("Spacy model loaded successfully")
    except Exception as e:
        print(f"Error initializing Spacy model: {str(e)}")
        print(traceback.format_exc())


def is_named_entity(text):
    doc = spacy_model(text)
    return any(entity.label_ != "" for entity in doc.ents)
