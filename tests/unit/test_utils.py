# test_utils.py
import pytest
import pandas as pd
import tempfile
import os
import sys

sys.path.append(os.path.abspath("../../src"))
from utils.utils import normalize_for_rag, load_data

def test_normalize_for_rag_basic_cleaning():
    raw_text = "I applied for a loan on 01/02/2023 and it was rejected without reason. $$$ xx"
    cleaned = normalize_for_rag(raw_text)
    assert cleaned.startswith("i applied for a loan")
    assert "01/02/2023" not in cleaned
    assert "$" in cleaned
    assert "xx" not in cleaned

def test_normalize_for_rag_empty_and_null():
    assert normalize_for_rag(None) == ""
    assert normalize_for_rag("") == ""
    assert normalize_for_rag("   ") == ""

def test_load_data_csv():
    # Create a small temp CSV file
    df = pd.DataFrame({"Product": ["Loan"], "Complaint": ["Denied without explanation"]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    loaded = load_data(tmp_path)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == (1, 2)

    os.remove(tmp_path)

def test_load_data_parquet():
    # Create a small temp Parquet file
    df = pd.DataFrame({"Product": ["Credit Card"], "Complaint": ["Late fee applied unfairly"]})
    tmp_path = os.path.join(tempfile.gettempdir(), "test_complaints.parquet")
    df.to_parquet(tmp_path)

    loaded = load_data(tmp_path)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.loc[0, "Product"] == "Credit Card"

    os.remove(tmp_path)

def test_load_data_invalid_format():
    with pytest.raises(ValueError):
        load_data("complaints.txt")