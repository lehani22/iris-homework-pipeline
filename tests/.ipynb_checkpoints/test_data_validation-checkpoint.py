import pytest
import pandas as pd
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, os.pardir)
sys.path.insert(0, parent_dir)

from pipeline_script import load_data, preprocess_data


def test_data_loading_returns_dataframe():
    df, _ = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_data_has_expected_columns():
    df, _ = load_data()
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    assert all(col in df.columns for col in expected_columns)

def test_target_column_has_valid_values():
    df, _ = load_data()
    assert df['target'].isin([0, 1, 2]).all() # Iris target values are 0, 1, 2

def test_preprocess_data_splits_correctly():
    df, _ = load_data()
    X, y = preprocess_data(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'target' not in X.columns
    assert y.name == 'target'
    assert len(X) == len(y)