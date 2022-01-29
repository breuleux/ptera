from ptera import tooled
from ptera.selector import select
from ptera.tools import between, every, gt, gte, lt, lte, throttle

from .common import one_test_per_assert, tapping


@tooled
def boucle_d_or(seq):
    accum = 0
    for i in seq:
        accum = accum + i
    return accum


def _r(*args):
    return list(range(*args))


def _boucle(query, seq):
    query = select(query)
    with tapping(query) as xs:
        boucle_d_or(seq)
    return xs[query.main.capture]


@one_test_per_assert
def test_every():
    assert _boucle("i~every()", _r(10)) == _r(10)
    assert _boucle("i~every(2)", _r(10)) == _r(0, 10, 2)
    assert _boucle("i~every(3)", _r(10)) == _r(0, 10, 3)
    assert _boucle("i~every(3, start=5)", _r(10)) == _r(5, 10, 3)
    assert _boucle("i~every(3, end=5)", _r(10)) == _r(0, 5, 3)


@one_test_per_assert
def test_between():
    assert _boucle("i~between(2, 5)", _r(10)) == _r(2, 5)
    assert _boucle("i~between(-2, 2)", _r(-5, 5, 2)) == [-1, 1]


@one_test_per_assert
def test_lt():
    assert _boucle("i~lt(0)", _r(10)) == []
    assert _boucle("i~lt(5)", _r(10)) == _r(5)


@one_test_per_assert
def test_lte():
    assert _boucle("i~lte(0)", _r(10)) == [0]
    assert _boucle("i~lte(5)", _r(10)) == _r(6)


@one_test_per_assert
def test_gt():
    assert _boucle("i~gt(9)", _r(10)) == []
    assert _boucle("i~gt(5)", _r(10)) == _r(6, 10)


@one_test_per_assert
def test_gte():
    assert _boucle("i~gte(0)", _r(10)) == _r(10)
    assert _boucle("i~gte(5)", _r(10)) == _r(5, 10)


@one_test_per_assert
def test_throttle():
    assert _boucle("i~throttle(5)", _r(0, 20)) == [0, 5, 10, 15]
    assert _boucle("i~throttle(5)", _r(0, 20, 2)) == [0, 6, 10, 16]
    assert _boucle("i~throttle(5)", _r(-3, 20, 3)) == [-3, 3, 9, 12, 18]
