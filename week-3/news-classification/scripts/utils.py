import os
import yaml

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaModel

TOKENIZER_RETURN_TYPE = RobertaTokenizerFast
MODEL_RETURN_TYPE = RobertaModel


def set_env():
    with open("config/credentials.yaml", "r") as stream:
        try:
            CREDENTIALS = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    os.environ["WANDB_API_KEY"] = CREDENTIALS.get('wamdb', {}).get('api_key')

def load_model() -> tuple():
    model = AutoModel.from_pretrained('youscan/ukr-roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('youscan/ukr-roberta-base')
    return tokenizer, model

def model_embedding(text: str, tokenizer: TOKENIZER_RETURN_TYPE, model: MODEL_RETURN_TYPE) -> np.ndarray:
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    outputs = model(input_ids, output_hidden_states=True)
    emb = outputs[2]
    emb = np.array([i[0].detach().numpy() for i in emb])[:]
    emb = emb.mean(axis=(0, 1))
    return emb

def load_data() -> pd.core.frame.DataFrame:
    data = pd.read_csv('data/train_small2.csv', index_col=0)
    data['text'] = data['text'].str.lower()
    return data

def train_test_split(data: pd.core.frame.DataFrame) -> tuple:
    rng = np.random.RandomState()
    #train_data1 = data.sample(frac=0.7, random_state=rng)
    #test_data1 = data.loc[~data.index.isin(train_data1.index)]
    train_data = data[data['target_numerical'] != 2].copy()
    test_data = data[(data["target_numerical"] == 1)].copy()
    return train_data, test_data

def get_embedding_list(labels_list, descriptions_list, tokenizer, model):
    embeddings_list = {}
    for label, descr in zip(labels_list, descriptions_list):
        tmp = embeddings_list.get(label, [])
        tmp.append(model_embedding(descr, tokenizer, model))
        embeddings_list[label] = tmp
    for label, embeddings in embeddings_list.items():
        embeddings_list[label] = np.mean(embeddings, axis=0)
    return embeddings_list

def distance(a, b):
    return sum([(i - j) ** 2 for i, j in zip(a, b)]) ** .5
