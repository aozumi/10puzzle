from utils import first

def test_first__empty() -> None:
    assert first([], "empty") == "empty"


def test_first() -> None:
    assert first([1, 2, 3], None) == 1
