import numpy as np
import pandas as pd
import pytest
from transformers import EvalPrediction
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaModel

from scripts.utils import load_model, load_data, train_test_split, model_embedding, get_embedding_list, distance


def test_load_model():
    tokenizer, model = load_model()
    assert isinstance(tokenizer, RobertaTokenizerFast)
    assert isinstance(model, RobertaModel)


def test_train_test_split():
    """data = load_data()
    train_data, test_data = train_test_split(data)
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)"""
    pass


def test_model_embedding():
    pass


def test_get_embedding_list():
    pass


def test_distance():
    pass
