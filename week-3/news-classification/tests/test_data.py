import pytest
import pandas as pd
import great_expectations


@pytest.mark.parametrize("test_input", [("week-3/news-classification/data/train_small2.csv")])
def test_expect_table_columns_to_match_ordered_list(test_input):
    data = pd.read_csv(test_input)
    df = great_expectations.dataset.PandasDataset(data)
    df.expect_table_columns_to_match_ordered_list(
        column_list=["title", "text", "tags", "target", "target_numerical"]
    )


@pytest.mark.parametrize("test_input, test_expected", [("week-3/news-classification/data/train_small2.csv", 6)])
def test_data_shape(test_input, test_expected):
    data = pd.read_csv(test_input)
    assert data.shape[1] == test_expected


@pytest.mark.parametrize("test_input", [("week-3/news-classification/data/train_small2.csv")])
def test_column_values_to_not_be_null(test_input):
    data = pd.read_csv(test_input)
    df = great_expectations.dataset.PandasDataset(data)
    target_values_to_not_be_null = df.expect_column_values_to_not_be_null(column="target")
    text_values_to_not_be_null = df.expect_column_values_to_not_be_null(column="text")
    text_values_to_be_of_type = df.expect_column_values_to_be_of_type(column="text", type_="str")
    assert target_values_to_not_be_null.get("success") == True
    assert text_values_to_not_be_null.get("success") == True
    assert text_values_to_be_of_type.get("success") == True
