from pathlib import Path

import pytest
from transformers import Trainer, TrainingArguments
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaModel

from scripts.utils import load_model


def test_model():
    tokenizer, model = load_model()
    assert isinstance(tokenizer, RobertaTokenizerFast)
    assert isinstance(model, RobertaModel)
    assert tokenizer.model_max_length == 512
    assert tokenizer.max_model_input_sizes == {'roberta-base': 512, 'roberta-large': 512, 'roberta-large-mnli': 512, 'distilroberta-base': 512, 'roberta-base-openai-detector': 512, 'roberta-large-openai-detector': 512}

