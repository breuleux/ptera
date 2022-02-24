import functools
import sys

import pytest

from ptera.utils import ABSENT, CodeNotFoundError, _build_refstring, refstring

from .common import one_test_per_assert


def test_named():
    assert str(ABSENT) == "ABSENT"
    assert repr(ABSENT) == "ABSENT"


def helloes():
    return 123


@functools.wraps(helloes)
def bonjours():
    return helloes()


class Corn:
    def __init__(self, x):
        self.x = x

    def __call__(self):
        return self.x


@one_test_per_assert
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_refstring():
    assert refstring(helloes) == "/tests.test_utils/helloes"
    assert refstring(bonjours) == "/tests.test_utils/helloes"
    assert refstring(one_test_per_assert) == "/tests.common/one_test_per_assert"
    assert refstring(refstring) == "/ptera.utils/refstring"
    assert refstring(Corn) == "/tests.test_utils/Corn/__init__"
    assert refstring(Corn(4)) == "/tests.test_utils/Corn/__call__"


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_refstring_closure():
    def inner():
        pass

    assert refstring(inner) == "/tests.test_utils/test_refstring_closure/inner"


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_refstring_bad():
    def liar():
        pass

    liar.__qualname__ = "truthteller"

    with pytest.raises(CodeNotFoundError):
        refstring(liar)


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_refstring_bad2():
    with pytest.raises(TypeError):
        refstring("not_a_function")


@one_test_per_assert
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_build_refstring():
    assert _build_refstring("a", "b") == "/a/b"
    assert _build_refstring("__main__", "b") == "//b"
