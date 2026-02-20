import pytest

@pytest.fixture(scope='session')
def dataset():
    return {
        'train': 500,
        'validation': 100,
        'test': 300
    }