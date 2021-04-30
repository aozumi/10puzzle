from utils import first, subsums


def test_first__empty() -> None:
    assert first([], "empty") == "empty"


def test_first() -> None:
    assert first([1, 2, 3], None) == 1


def test_subsums() -> None:
    assert list(subsums([])) == []
    assert list(subsums([1])) == [1]
    assert list(subsums([1, 2])) == [1, 2, 3]
    assert list(subsums([1, 2, 4])) == [1, 2, 4, 3, 5, 6, 7]
