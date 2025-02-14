from transformers import AutoTokenizer, AutoModel
import torch
import traceback

bert_tokenizer = None
bert_model = None
SPACY_MODEL_NAME = "en_core_web_sm"
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MODEL_PATH = f"forked_models/{BERT_MODEL_NAME}"


def initialize_bert():
    global bert_tokenizer
    global bert_model
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        bert_model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        print("BERT tokenizer & model loaded successfully")
    except Exception as e:
        print(f"Error initializing BERT model: {str(e)}")
        print(traceback.format_exc())


def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
