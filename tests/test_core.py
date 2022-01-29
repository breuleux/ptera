import sys
from collections import defaultdict
from types import SimpleNamespace as NS

import pytest

from ptera import BaseOverlay, Overlay, tag, tooled
from ptera.core import Capture, Immediate
from ptera.selector import Element, parse
from ptera.tools import every  # noqa

from .common import TapResults, full_tapping, one_test_per_assert, tapping


@tooled
def brie(x, y) -> tag.Fromage:
    """Brie is a sort of cheese."""
    a: tag.Bouffe = x * x
    b: "@Bouffe & @Agrement" = y * y  # type: ignore
    return a + b


@tooled
def extra(cheese):
    return cheese + 1


@tooled
@tooled
@tooled
def double_brie(x1, y1):
    a = brie(x1, x1 + 1)
    b = brie(y1, y1 + 1)
    aa = extra(a)
    bb = extra(b)
    return aa + bb


@one_test_per_assert
def test_normal_call():
    assert brie(3, 4) == 25
    assert double_brie(3, 4) == 68


def _test(f, args, pattern):
    with full_tapping(pattern) as results:
        f(*args)
    return results


def _dbrie(pattern):
    return _test(double_brie, (2, 10), pattern)


@one_test_per_assert
def test_patterns():
    # Simple, test focus
    assert _dbrie("*(x)") == [{"x": [2]}, {"x": [10]}]
    assert _dbrie("*(!x)") == [{"x": [2]}, {"x": [10]}]
    assert _dbrie("*(!x, y)") == [{"x": [2], "y": [3]}, {"x": [10], "y": [11]}]
    assert _dbrie("*(x, y)") == [{"x": [2], "y": [3]}, {"x": [10], "y": [11]}]

    # Simple
    assert _dbrie("*(!a)") == [{"a": [4]}, {"a": [100]}, {"a": [13]}]
    assert _dbrie("brie(!a)") == [{"a": [4]}, {"a": [100]}]

    # Indirect
    assert _dbrie("a") == [{"a": [4]}, {"a": [100]}, {"a": [13]}]

    # Multi-level
    assert _dbrie("double_brie(a) > brie(x)") == [{"a": [13], "x": [2, 10]}]
    assert _dbrie("double_brie(a) > brie(!x)") == [
        {"a": [13], "x": [2]},
        {"a": [13], "x": [10]},
    ]

    # Accumulate values across calls
    assert _dbrie("double_brie(extra(cheese), brie(x))") == [
        {"cheese": [13, 221], "x": [2, 10]}
    ]
    assert _dbrie("double_brie(extra(!cheese), brie(x))") == [
        {"cheese": [13], "x": [2, 10]},
        {"cheese": [221], "x": [2, 10]},
    ]

    # Parameter
    assert _dbrie("brie($v:tag.Bouffe)") == [{"v": [4, 9]}, {"v": [100, 121]}]
    assert _dbrie("brie($v:@Bouffe)") == [{"v": [4, 9]}, {"v": [100, 121]}]
    assert _dbrie("brie(!$v:tag.Bouffe)") == [
        {"v": [4]},
        {"v": [9]},
        {"v": [100]},
        {"v": [121]},
    ]
    assert _dbrie("*(a) > brie(!$v:tag.Bouffe)") == [
        {"a": [13], "v": [4]},
        {"a": [13], "v": [9]},
        {"a": [13], "v": [100]},
        {"a": [13], "v": [121]},
    ]

    # Function category
    assert _dbrie("*:tag.Fromage(a)") == [{"a": [4]}, {"a": [100]}]

    # Inexistent category
    assert _dbrie("brie > $x:tag.Xylophone") == []

    # Filter on value
    assert _dbrie("brie(!x, y, a=4)") == [{"a": [4], "x": [2], "y": [3]}]
    assert _dbrie("double_brie(x1=2) > brie > x") == [
        {"x1": [2], "x": [2]},
        {"x1": [2], "x": [10]},
    ]
    assert _dbrie("double_brie(#value=1234) > brie > x") == []


@tooled
def snapple(x):
    a = cabanana(x + 1)
    b = cabanana(x + 2)
    return a + b


@tooled
def cabanana(y):
    return peacherry(y + 1)


@tooled
def peacherry(z):
    return z + 1


def test_deep():
    assert _test(snapple, [5], "snapple > cabanana(y) > peacherry > z") == [
        {"y": [6], "z": [7]},
        {"y": [7], "z": [8]},
    ]


@tooled
def superbrie(n):
    result = 0
    k = 0
    for i in range(n):
        for j in range(n):
            result += brie(k, 2)
            k = k + 1
    return result


def test_nested_loops():
    assert superbrie(10) == 328750

    with tapping("superbrie(i=1, j) > brie > x") as res:
        superbrie(10)

    assert res["j"] == list(range(10))
    assert res["x"] == list(range(10, 20))

    with tapping("superbrie(i=1, j ~ every(3)) > brie > x") as res:
        superbrie(10)

    assert res["x"] == list(range(10, 20, 3))


def test_immediate_evaluation():
    # This uses an ImmediateAccumulator
    with Overlay.rewriting({"superbrie(k=7) > brie > x": (lambda args: 0)}):
        assert superbrie(10) == 328701

    # This uses an ImmediateAccumulator
    with Overlay.rewriting({"superbrie(i=9) > brie > x": (lambda args: 0)}):
        assert superbrie(10) == 239365

    # By default this uses a TotalAccumulator, which requires every
    # value of i to be 1 and every j to be a multiple of 3
    with full_tapping("superbrie(i=1, j~every(3)) > brie > x") as xs:
        superbrie(10)
    assert xs == []

    # Creates an ImmediateAccumulator which only takes into account the values
    # of i and j at the moment the focus variable x is triggered
    with tapping("superbrie(i=1, j~every(3)) > brie > x") as xs:
        superbrie(10)

    assert xs["x"] == list(range(10, 20, 3))


def test_nested_overlay():
    expectedx = [{"x": [2]}, {"x": [10]}]
    expectedy = [{"y": [3]}, {"y": [11]}]

    with full_tapping("brie > x") as resultsx:
        with full_tapping("brie > y") as resultsy:
            assert double_brie(2, 10) == 236

    assert resultsx == expectedx
    assert resultsy == expectedy


@tooled
def mystery(hat):
    surprise: tag.MyStErY
    return surprise * hat


def test_provide_var():
    with BaseOverlay(Immediate("mystery(!surprise)", intercept=lambda _: 4)):
        assert mystery(10) == 40

    with BaseOverlay(
        Immediate(
            "mystery(hat, !surprise)", intercept=lambda args: args["hat"].value
        )
    ):
        assert mystery(8) == 64


def test_missing_var():
    try:
        mystery(3)
    except NameError as err:
        assert err.varname == "surprise"
        assert err.function == mystery
        info = err.info()
        assert info["annotation"] == tag.MyStErY

    with pytest.raises(NameError):
        with Overlay.tweaking({"mystery(hat=10) > surprise": 0}):
            mystery(3)


@tooled
def square(x):
    rval = x * x
    return rval


@tooled
def sumsquares(x, y):
    xx = square(x)
    yy = square(y)
    rval = xx + yy
    return rval


def test_misc():
    # A version of these was in the old README

    with Overlay.tapping("x") as xs:
        sumsquares(3, 4)
    assert xs == [{"x": 3}, {"x": 3}, {"x": 4}]

    with Overlay.tapping("square > x", TapResults()) as xs:
        sumsquares(3, 4)
    assert xs["x"] == [3, 4]

    with full_tapping("square(rval) > x", all=False) as xs:
        sumsquares(3, 4)
    assert xs["x"] == [3, 4]
    assert xs["rval"] == [9, 16]

    with full_tapping(
        "sumsquares(x as ssx, y as ssy) > square(rval) > x", all=False
    ) as xs:
        sumsquares(3, 4)

    assert xs["ssx"] == [3, 3]
    assert xs["ssy"] == [4, 4]
    assert xs["x"] == [3, 4]
    assert xs["rval"] == [9, 16]

    with full_tapping(
        "sumsquares(!x as ssx, y as ssy) > square(rval, x)"
    ) as xs:
        sumsquares(3, 4)

    assert xs["ssx"] == [[3]]
    assert xs["ssy"] == [[4]]
    assert xs["x"] == [[3, 4]]
    assert xs["rval"] == [[9, 16]]

    with Overlay.tweaking({"square > rval": 0}):
        assert sumsquares(3, 4) == 0

    with Overlay.rewriting({"square(x) > rval": lambda args: args["x"] + 1}):
        assert sumsquares(3, 4) == 9


def test_capture():
    cap = Capture(parse("x"))
    assert cap.name == "x"
    with pytest.raises(ValueError):
        cap.value
    cap.accum("x", 1)
    assert cap.name == "x"
    assert cap.value == 1
    cap.accum("x", 2)
    with pytest.raises(ValueError):
        cap.value

    assert str(cap) == "Capture(sel(\"!x\"), ['x', 'x'], [1, 2])"

    cap = Capture(Element(name=None))
    with pytest.raises(ValueError):
        cap.name
    cap.accum("y", 7)
    assert cap.name == "y"
    assert cap.value == 7
    cap.accum("z", 31)
    with pytest.raises(ValueError):
        cap.name
    with pytest.raises(ValueError):
        cap.value


@tooled
def cake():
    flavour: tag.Flavour
    return f"This is a {flavour} cake"


def test_doc():
    assert brie.__doc__ == """Brie is a sort of cheese."""


class Matou:
    def __init__(self, species):
        self.species = species

    @tooled
    def meow(self, repeat=1):
        ms = "m"
        es = "e"
        os = "o" * repeat
        ws = "w" * len(self.species)
        cry = ms + es + os + ws
        meows = [cry] * repeat
        return " ".join(meows)

    def meow_nodeco(self, repeat=1):
        ms = "m"
        es = "e"
        os = "o" * repeat
        ws = "w" * len(self.species)
        cry = ms + es + os + ws
        meows = [cry] * repeat
        return " ".join(meows)


def test_method():
    siamese = Matou("siamese")
    assert siamese.meow() == "meowwwwwww"

    with Overlay.tweaking({"Matou.meow > es": "eee"}):
        assert siamese.meow() == "meeeowwwwwww"

    with Overlay.tweaking({"Matou.meow > repeat": 2}):
        assert siamese.meow() == "meoowwwwwww meoowwwwwww"

    with full_tapping("Matou.meow(repeat) > os") as results:
        for i in range(3):
            siamese.meow(i)
    assert results == [
        {"os": [""], "repeat": [0]},
        {"os": ["o"], "repeat": [1]},
        {"os": ["oo"], "repeat": [2]},
    ]


def test_redirect_method():
    siamese = Matou("siamese")

    tooled.inplace(Matou.meow_nodeco)

    assert siamese.meow_nodeco() == "meowwwwwww"

    with Overlay.tweaking({"Matou.meow_nodeco > es": "eee"}):
        assert siamese.meow_nodeco() == "meeeowwwwwww"

    with full_tapping("Matou.meow_nodeco(repeat) > os") as results:
        for i in range(3):
            siamese.meow_nodeco(i)
    assert results == [
        {"os": [""], "repeat": [0]},
        {"os": ["o"], "repeat": [1]},
        {"os": ["oo"], "repeat": [2]},
    ]


def test_on():
    def twice_mystery(x):
        return mystery(x), mystery(x + 1)

    results = defaultdict(list)

    ov = Overlay()
    ov.tweak({"surprise": 2})

    @ov.on("mystery > hat")
    def hats(args):
        hat = args["hat"]
        results["hats"].append(hat * hat)

    @ov.on("mystery(hat) > surprise")
    def shats(args):
        surprise = args["surprise"]
        hat = args["hat"]
        results["shats"].append((surprise, hat))

    with ov:
        assert twice_mystery(10) == (20, 22)

    assert results == {
        "hats": [100, 121],
        "shats": [(2, 10), (2, 11)],
    }


def test_on_2():
    results = defaultdict(list)

    ov = Overlay()

    @ov.on("brie > x")
    def minx(args):
        x = args["x"]
        results["minx"].append(-x)

    @ov.on("brie > x", all=True)
    def minx_all(args):
        x = args["x"]
        results["minx_all"].append([-v for v in x])

    @ov.on("brie > x", full=True)
    def minx_full(args):
        x = args["x"]
        assert x.name == "x"
        results["minx_full"].append(-x.value)

    with ov:
        double_brie(2, 10)

    assert results["minx"] == [-2, -10]
    assert results["minx_all"] == [[-2], [-10]]
    assert results["minx_full"] == [-2, -10]


@tooled
def multitag():
    y: tag.Bouffe = 10
    y = y * y
    return y


def test_samevar_multitag():
    assert multitag() == 100
    with Overlay.tweaking({"y:tag.Bouffe": 5}):
        assert multitag() == 25
    with Overlay.tweaking({"y:tag.Irrelevant": 5}):
        assert multitag() == 100


def test_redirect():
    def funkykong(x):
        surf: tag.Surfboard = True
        return x * x if surf else x

    orig_funky = funkykong
    new_funky = tooled.inplace(funkykong)

    assert funkykong is orig_funky
    assert funkykong is new_funky

    assert funkykong(10) == 100
    with Overlay.tweaking({"surf:tag.Surfboard": False}):
        assert funkykong(10) == 10
    assert funkykong(10) == 100


def test_redirect_noclobber():
    def one():
        x = 1
        return x

    def two():
        x = 1
        return x * 2

    tooled.inplace(one)
    tooled.inplace(two)

    assert one() == 1
    assert two() == 2

    with Overlay.tweaking({"x": 7}):
        assert one() == 7
        assert two() == 14


def exposure(n):
    x = 2
    return n ** x


def test_redirect_global():
    old_exposure = exposure

    tooled.inplace(exposure)
    assert exposure(8) == 64

    with Overlay.tweaking({"x": 3}):
        assert exposure(8) == 512

    assert old_exposure is exposure


def test_attr_assignment_ignored():
    @tooled
    def donkey(x):
        x.y = 3
        return x.y

    assert donkey(NS(y=7)) == 3
    with Overlay.tweaking({"x": NS(y=6)}):
        assert donkey(NS(y=7)) == 3


def test_enter_hashvar():
    @tooled
    def wumpus():
        return 3

    n = 5

    with full_tapping("wumpus > #enter") as results:
        for i in range(n):
            wumpus()

    assert results == [{"#enter": [True]}] * n


class Koala:
    @tooled
    def moo(self, x):
        self.x = x
        self.y = x


def test_intercept_attribute():
    k = Koala()

    with BaseOverlay(
        Immediate(
            "Koala.moo > self.x",
            intercept=lambda args: args["self.x"].value + 1,
        )
    ):
        k.moo(7)

    assert k.x == 8
    assert k.y == 7


def test_generator():
    @tooled
    def oxygen():
        x = 0
        while True:
            x = x + 1
            yield x * 2

    with full_tapping("oxygen > x") as results:
        for x in oxygen():
            if x > 10:
                break

    assert results == [{"x": [i]} for i in range(7)]

    with full_tapping("oxygen > #yield") as results:
        for x in oxygen():
            if x > 10:
                break

    assert results == [{"#yield": [i * 2]} for i in range(1, 7)]


def broccoli(n):
    factor = 2
    a = n * factor
    return a + 1


def cauliflower(n):
    factor = 2
    a = n * factor
    return a + 2


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_conform():
    from codefind import conform

    tooled.inplace(broccoli)

    with Overlay.tweaking({"factor": 3}):
        assert broccoli(10) == 31
        conformer = broccoli.__ptera__.fn._conformer
        conform(conformer.code, cauliflower)
        conform(conformer.code, cauliflower.__code__)
        assert broccoli(10) == 32
