import json
from typing import Tuple
from urllib.request import urlopen

import pytest
import pandas as pd
import great_expectations
from great_expectations.dataset.pandas_dataset import PandasDataset


data = pd.read_csv("data/train_small2.csv")
df = great_expectations.dataset.PandasDataset(data)

def test_expect_table_columns_to_match_ordered_list():
    df.expect_table_columns_to_match_ordered_list(
        column_list=["title", "text", "tags", "target", "target_numerical"]
    )


@pytest.mark.parametrize("test_input, test_expected", [("data/train_small2.csv", 6)])
def test_data_shape(test_input, test_expected):
    data = pd.read_csv(test_input)
    assert data.shape[1] == test_expected


def test_column_values_to_not_be_null():
    target_values_to_not_be_null = df.expect_column_values_to_not_be_null(column="target")
    text_values_to_not_be_null = df.expect_column_values_to_not_be_null(column="text")
    text_values_to_be_of_type = df.expect_column_values_to_be_of_type(column="text", type_="str")
    assert target_values_to_not_be_null.get("success") == True
    assert text_values_to_not_be_null.get("success") == True
    assert text_values_to_be_of_type.get("success") == True
