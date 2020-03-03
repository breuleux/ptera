import pytest

from ptera import Recurrence, cat, overlay, ptera, to_pattern
from ptera.core import Capture
from ptera.selector import Element, parse

from .common import one_test_per_assert


@ptera
def brie(x, y) -> cat.Fromage:
    a: cat.Bouffe = x * x
    b: cat.Bouffe = y * y
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


def _test(f, args, pattern):
    store = GrabAll(pattern)
    with overlay(store.rules):
        f(*args)
    return store.results


def _dbrie(pattern):
    return _test(double_brie, (2, 10), pattern)


@one_test_per_assert
def test_patterns():
    # Simple, test focus
    assert _dbrie("*{x}") == [{"x": [2]}, {"x": [10]}]
    assert _dbrie("*{!x}") == [{"x": [2]}, {"x": [10]}]
    assert _dbrie("*{!x, y}") == [{"x": [2], "y": [3]}, {"x": [10], "y": [11]}]
    assert _dbrie("*{x, y}") == [{"x": [2], "y": [3]}, {"x": [10], "y": [11]}]

    # Simple
    assert _dbrie("*{!a}") == [{"a": [4]}, {"a": [100]}, {"a": [13]}]
    assert _dbrie("brie{!a}") == [{"a": [4]}, {"a": [100]}]

    # Indirect
    assert _dbrie("a") == [{"a": [4]}, {"a": [100]}, {"a": [13]}]
    assert _dbrie("double_brie >> a") == [{"a": [13]}, {"a": [4]}, {"a": [100]}]
    assert _dbrie("double_brie >> x") == [{"x": [2]}, {"x": [10]}]

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
    assert _dbrie("brie{$v:cat.Bouffe}") == [{"v": [4, 9]}, {"v": [100, 121]}]
    assert _dbrie("brie{!$v:cat.Bouffe}") == [
        {"v": [4]},
        {"v": [9]},
        {"v": [100]},
        {"v": [121]},
    ]
    assert _dbrie("*{a} >> brie{!$v:cat.Bouffe}") == [
        {"a": [13], "v": [4]},
        {"a": [13], "v": [9]},
        {"a": [13], "v": [100]},
        {"a": [13], "v": [121]},
    ]

    # Function category
    assert _dbrie("*:cat.Fromage{a}") == [{"a": [4]}, {"a": [100]}]

    # Inexistent category
    assert _dbrie("brie > $x:cat.Xylophone") == []


@ptera
def snapple(x):
    a = cabanana(x + 1)
    b = cabanana(x + 2)
    return a + b


@ptera
def cabanana(y):
    return peacherry(y + 1)


@ptera
def peacherry(z):
    return z + 1


def test_deep():
    assert _test(snapple, [5], "snapple > cabanana{y} > peacherry > z") == [
        {"y": [6], "z": [7]},
        {"y": [7], "z": [8]},
    ]


@ptera
def fib(n):
    f = Recurrence(2)
    f[0] = 1
    f[1] = 1
    for i in range(2, n + 1):
        f[i] = f[i - 1] + f[i - 2]
    return f[n]


def test_indexing():
    assert fib(5) == 8

    res, fs = fib.using("f[0] as x")(5)
    assert fs.map("x") == [1]

    res, fs = fib.using("f[$i] as x")(5)
    intermediates = [1, 1, 2, 3, 5, 8]
    indices = list(range(6))
    assert fs.map("x") == intermediates
    assert fs.map("i") == indices
    assert fs.map("i", "x") == list(zip(indices, intermediates))


def test_indexing_2():
    res, fs = fib.using("fib{!n, f[3] as x}")(5)
    assert res == 8
    assert fs.map("n") == [5]
    assert fs.map("x") == [3]


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
    surprise: cat.MyStErY
    return surprise * hat


def test_provide_var():
    with overlay({"mystery{!surprise}": {"value": lambda surprise: 4}}):
        assert mystery(10) == 40

    with overlay(
        {"mystery{hat, !surprise}": {"value": lambda hat, surprise: hat.value}}
    ):
        assert mystery(8) == 64


def test_missing_var():
    with pytest.raises(NameError):
        mystery(3)


def test_tap_map():
    rval, acoll = double_brie.using("brie{!a, b}")(2, 10)
    assert acoll.map("a") == [4, 100]
    assert acoll.map("b") == [9, 121]
    assert acoll.map(lambda a, b: a + b) == [13, 221]
    assert acoll.map() == [{"a": 4, "b": 9}, {"a": 100, "b": 121}]
    assert acoll.map(lambda **kwargs: kwargs["a"] + kwargs["b"]) == [13, 221]


def test_tap_map_all():
    rval, acoll = double_brie.using("double_brie{!x1} >> brie{x}")(2, 10)
    with pytest.raises(ValueError):
        acoll.map("x1", "x")
    assert acoll.map_all("x1", "x") == [([2], [2, 10])]
    assert acoll.map_all() == [{"x1": [2], "x": [2, 10]}]


def test_tap_map_named():
    rval = double_brie.using(data="brie{!a, b}")(2, 10)
    assert rval.value == 236
    assert rval.data.map("a") == [4, 100]


def test_tap_map_full():
    rval, acoll = double_brie.using("brie > $param:cat.Bouffe")(2, 10)
    assert acoll.map_full(lambda param: param.value) == [4, 9, 100, 121]
    assert acoll.map_full(lambda param: param.name) == ["a", "b", "a", "b"]


def test_on():
    dbrie = double_brie.clone(return_object=True)

    @dbrie.on("brie > x")
    def minx(x):
        return -x

    @dbrie.on("brie > x", all=True)
    def minx_all(x):
        return [-v for v in x]

    @dbrie.on("brie > x", full=True)
    def minx_full(x):
        assert x.name == "x"
        return -x.value

    results = dbrie(2, 10)
    assert results.minx == [-2, -10]
    assert results.minx_all == [[-2], [-10]]
    assert results.minx_full == [-2, -10]


def test_use():
    dbrie = double_brie.clone(return_object=True)
    dbrie.use(data="brie{!a, b}")
    rval = dbrie(2, 10)
    assert rval.value == 236
    assert rval.data.map("a") == [4, 100]


def test_collect():
    dbrie = double_brie.clone(return_object=True)

    @dbrie.collect("brie > x")
    def sumx(xs):
        return sum(xs.map("x"))

    results = dbrie(2, 10)
    assert results.sumx == 12


@ptera
def square(x):
    rval = x * x
    return rval


@ptera
def sumsquares(x, y):
    xx = square(x)
    yy = square(y)
    rval = xx + yy
    return rval


def test_readme():
    results = sumsquares.using(q="x")(3, 4)
    assert results.q.map("x") == [3, 4, 3]

    results = sumsquares.using(q="square > x")(3, 4)
    assert results.q.map("x") == [3, 4]

    results = sumsquares.using(q="square{rval} > x")(3, 4)
    assert results.q.map("x", "rval") == [(3, 9), (4, 16)]

    results = sumsquares.using(
        q="sumsquares{x as ssx, y as ssy} > square{rval} > x"
    )(3, 4)
    assert results.q.map("ssx", "ssy", "x", "rval") == [
        (3, 4, 3, 9),
        (3, 4, 4, 16),
    ]

    results = sumsquares.using(
        q="sumsquares{!x as ssx, y as ssy} > square{rval, x}"
    )(3, 4)
    assert results.q.map_all("ssx", "ssy", "x", "rval") == [
        ([3], [4], [3, 4], [9, 16])
    ]

    result = sumsquares.tweak({"square > rval": 0})(3, 4)
    assert result == 0

    result = sumsquares.rewrite({"square{x} > rval": lambda x: x + 1})(3, 4)
    assert result == 9


@ptera.defaults(x=10, y=20)
def vanilla(x, y):
    return x * y


def test_ptera_defaults():
    assert vanilla() == 200
    assert vanilla(4, 5) == 20


def test_capture():
    cap = Capture(parse("x"))
    assert cap.name == "x"
    with pytest.raises(ValueError):
        cap.value
    cap.acquire("x", 1)
    assert cap.name == "x"
    assert cap.value == 1
    cap.acquire("x", 2)
    with pytest.raises(ValueError):
        cap.value

    assert str(cap) == "Capture(sel(\"!x\"), ['x', 'x'], [1, 2])"

    cap = Capture(Element(None))
    with pytest.raises(ValueError):
        cap.name
    cap.acquire("y", 7)
    assert cap.name == "y"
    assert cap.value == 7
    cap.acquire("z", 31)
    with pytest.raises(ValueError):
        cap.name
    with pytest.raises(ValueError):
        cap.value


@ptera
def cake():
    flavour: cat.Flavour
    return f"This is a {flavour} cake"


@ptera
def fruitcake():
    my_cake = cake.new(flavour="fruit").clone(return_object=True)

    @my_cake.on("flavour")
    def yum(flavour):
        return flavour * 2

    return my_cake()


def test_listener_within_ptera():
    res = fruitcake()
    assert res.value == "This is a fruit cake"
    assert res.yum == ["fruitfruit"]
