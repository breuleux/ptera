import numpy

from ptera import (
    Category,
    PatternCollection,
    overlay,
    ptera,
    selector as sel,
    to_pattern,
)

from .common import one_test_per_assert

Bouffe = Category("Bouffe")
Fruit = Category("Fruit", [Bouffe])
Legume = Category("Legume", [Bouffe])


@ptera
def brie(x, y):
    a: Bouffe = x * x
    b: Bouffe = y * y
    return a + b


@ptera
def extra(cheese):
    return cheese + 1


@ptera
def double_brie(x1, y1):
    a = brie[1](x1, x1 + 1)
    b = brie[2](y1, y1 + 1)
    aa = extra[1](a)
    bb = extra[2](b)
    return aa + bb


@one_test_per_assert
def test_normal_call():
    assert brie(3, 4) == 25
    assert double_brie(3, 4) == 68


class GrabAll:
    def __init__(self, pattern):
        self.results = []
        pattern = to_pattern(pattern)

        def listener(**kwargs):
            self.results.append(
                {name: cap.values for name, cap in kwargs.items()}
            )

        listener._ptera_argspec = None, set(pattern.all_captures())
        self.rules = {pattern: {"listeners": listener}}


def _dbrie(pattern):
    store = GrabAll(pattern)
    with overlay(store.rules):
        double_brie(2, 10)
    return store.results


@one_test_per_assert
def test_patterns():
    # Simple, test focus
    assert _dbrie("*{x}") == [{"x": [2, 10]}]
    assert _dbrie("*{!x}") == [{"x": [2]}, {"x": [10]}]
    assert _dbrie("*{!x, y}") == [{"x": [2], "y": [3]}, {"x": [10], "y": [11]}]
    assert _dbrie("*{x, y}") == [{"x": [2, 10], "y": [3, 11]}]

    # Simple
    assert _dbrie("*{!a}") == [{"a": [4]}, {"a": [100]}, {"a": [13]}]
    assert _dbrie("brie{!a}") == [{"a": [4]}, {"a": [100]}]

    # Multi-level
    assert _dbrie("double_brie{a} > brie{x}") == [{"a": [13], "x": [2, 10]}]
    assert _dbrie("double_brie{a} > brie{!x}") == [
        {"a": [13], "x": [2]},
        {"a": [13], "x": [10]},
    ]

    # Accumulate values across calls
    assert _dbrie("double_brie{extra{cheese}, brie{x}}") == [
        {"cheese": [13, 221], "x": [2, 10]}
    ]
    assert _dbrie("double_brie{extra{!cheese}, brie{x}}") == [
        {"cheese": [13], "x": [2, 10]},
        {"cheese": [221], "x": [2, 10]},
    ]

    # Indexing
    assert _dbrie("brie[$i]{!a}") == [
        {"a": [4], "i": [1]},
        {"a": [100], "i": [2]},
    ]
    assert _dbrie("brie[1]{!a}") == [{"a": [4]}]
    assert _dbrie("brie[2]{!a}") == [{"a": [100]}]

    # Parameter
    assert _dbrie("brie{$v:Bouffe}") == [{"v": [4, 9, 100, 121]}]
    assert _dbrie("brie{!$v:Bouffe}") == [
        {"v": [4]},
        {"v": [9]},
        {"v": [100]},
        {"v": [121]},
    ]
    assert _dbrie("*{a} >> brie{!$v:Bouffe}") == [
        {"a": [13], "v": [4]},
        {"a": [13], "v": [9]},
        {"a": [13], "v": [100]},
        {"a": [13], "v": [121]},
    ]


def test_nested_overlay():
    expectedx = [{"x": [2]}, {"x": [10]}]
    expectedy = [{"y": [3]}, {"y": [11]}]

    storex = GrabAll("brie > x")
    storey = GrabAll("brie > y")
    with overlay({**storex.rules, **storey.rules}):
        assert double_brie(2, 10) == 236
    assert storex.results == expectedx
    assert storey.results == expectedy

    storex = GrabAll("brie > x")
    storey = GrabAll("brie > y")
    with overlay(storex.rules):
        with overlay(storey.rules):
            assert double_brie(2, 10) == 236
    assert storex.results == expectedx
    assert storey.results == expectedy


@ptera
def mystery(hat):
    surprise: Fruit
    return surprise * hat


def test_provide_var():
    with overlay({"mystery{!surprise}": {"value": lambda: 4}}):
        assert mystery(10) == 40

    with overlay({"mystery{hat, !surprise}": {"value": lambda hat: hat.value}}):
        assert mystery(8) == 64


def test_tap_map():
    rval, acoll = double_brie.tap("brie{!a, b}")(2, 10)
    assert acoll.map("a") == [4, 100]
    assert acoll.map("b") == [9, 121]
    assert acoll.map(lambda a, b: a + b) == [13, 221]


def test_tap_map_full():
    rval, acoll = double_brie.tap("brie > $param:Bouffe")(2, 10)
    assert acoll.map_full(lambda param: param.value) == [4, 9, 100, 121]
    assert acoll.map_full(lambda param: param.name) == ["a", "b", "a", "b"]
