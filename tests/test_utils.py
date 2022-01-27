from ptera.utils import ABSENT


def test_named():
    assert str(ABSENT) == "ABSENT"
    assert repr(ABSENT) == "ABSENT"
