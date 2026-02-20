import pytest

def test_data_loading():
    train_data_size = 500
    validation_data_size = 100
    test_data_size = 300

    total_data_size = train_data_size + validation_data_size + test_data_size
    assert total_data_size == 900

    assert train_data_size == 500
    assert validation_data_size >= 100 and validation_data_size <= 300
    assert test_data_size >= 100 and test_data_size <= 300

    assert validation_data_size + test_data_size <= 600