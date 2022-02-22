import functools
import sys
from types import SimpleNamespace

import pytest

from ptera.selector import Element, SelectorError, select
from ptera.transform import Key, name_error, transform
from ptera.utils import ABSENT, keyword_decorator

from .common import one_test_per_assert


def _format_sym(entry):
    rval, key, typ, _, ovr = entry
    if key is not None:
        if key.type == "attr":
            rval += f".{key.value}"
        elif key.type == "index":
            rval += f"[{key.value}]"
    return rval


class Interactions(list):
    def has(self, sublist):
        i = 0
        for entry in self:
            if entry == sublist[i]:
                i += 1
                if i == len(sublist):
                    return True
        return False

    def syms(self):
        return type(self)(_format_sym(x) for x in self)

    def vals_for(self, sym):
        return type(self)(x[-2] for x in self if x[0] == sym)

    @property
    def ret(self):
        assert self[-1][0] == "#value"
        return self[-1][-2]


def test_Interactions():
    xs = Interactions([1, 2, 3, 7, 2, 9])
    assert xs.has(xs)
    assert xs.has([2, 3, 7])
    assert xs.has([1, 3, 9])
    assert not xs.has([3, 2, 3, 7])
    assert not xs.has([1, 1])

    xs = Interactions(
        [
            ("a", None, None, 6, True),
            ("b", None, None, 48, True),
            ("b", Key("attr", "wow"), None, 8, True),
            ("a", None, None, 9, True),
            ("c", None, None, -1, True),
        ]
    )
    assert xs.syms() == ["a", "b", "b.wow", "a", "c"]
    assert xs.vals_for("a") == [6, 9]
    assert xs.vals_for("b") == [48, 8]
    assert xs.vals_for("c") == [-1]


@keyword_decorator
def wrap(fn, all=False, names=True):
    results = Interactions()
    overrides = {}

    class SimpleInteractor:
        def __init__(self, fn):
            self.fn = fn
            self.results = results
            self.overrides = overrides

        def interact(self, sym, key, category, value, overridable):
            self.results.append((sym, key, category, value, overridable))
            rval = self.overrides.get(sym, value)
            if rval is ABSENT:
                raise name_error(sym, self.fn)
            return rval

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    new_fn = transform(
        fn,
        proceed=SimpleInteractor,
        to_instrument=True
        if names is True
        else [Element(name=name) for name in names],
    )

    @functools.wraps(fn)
    def wrapped(*args, **ovrd):
        kw = ovrd.pop("KW", {})
        results.clear()
        overrides.clear()
        overrides.update(ovrd)
        rval = new_fn(*args, **kw)
        results.actual_ret = rval
        if all:
            return results
        else:
            return rval

    wrapped.__ptera_info__ = wrapped.info = new_fn.__ptera_info__
    return wrapped


@wrap(all=True)
def iceberg(
    # The parameter x
    x: float,
    # The parameter y
    y,
):
    # The great
    # zee
    z: int
    return sum([x, y, z])


_iceberg_line = iceberg.__wrapped__.__code__.co_firstlineno


@wrap
def chocolat(x, y):
    z = x + y
    rval = z * z
    if add1:
        rval = rval + 1
    return rval


@wrap
def puerh(x=2, y=3):
    return x + y


@wrap
def helicopter(x, y):
    return min(x, y)


@wrap
def wishful_thinking():
    return unicorn


@wrap(all=True, names=["b", "d"])
def only_some(a):
    b = a + 1
    c = b + 1
    d = c + 1
    e = d + 1
    return e + 1


#################
# General tests #
#################


def test_interact():
    data = iceberg(7, 3, z=5)
    assert data.ret == 15
    assert data == [
        ("#enter", None, None, True, False),
        ("int", None, None, int, True),
        ("sum", None, None, sum, True),
        ("x", None, float, 7, True),
        ("y", None, None, 3, True),
        ("z", None, int, ABSENT, True),
        ("#value", None, None, 15, True),
    ]


def test_interact_only_some():
    data = only_some(10)
    assert data.actual_ret == 15
    assert data == [
        ("b", None, None, 11, True),
        ("d", None, None, 13, True),
    ]


@one_test_per_assert
def test_misc():
    assert iceberg(2, 3, z=5).ret == 10

    assert chocolat(2, 3, add1=False) == 25
    assert chocolat(2, 3, add1=True) == 26

    assert puerh() == 5
    assert puerh(x=7) == 10
    assert puerh(y=10) == 12
    assert puerh(4, 5) == 9

    assert helicopter(2, 10) == 2
    assert helicopter(2, 10, min=max) == 10


def test_name_error():
    with pytest.raises(NameError):
        iceberg(2, 3)
    with pytest.raises(NameError):
        chocolat(2, 3)
    with pytest.raises(NameError):
        wishful_thinking()


def test_missing_argument():
    with pytest.raises(TypeError):
        chocolat()


##############################
# Test information gathering #
##############################


def test_info():
    info = iceberg.info["z"]
    filename, fn, lineno = info.pop("location")
    assert fn.__name__ == "iceberg"
    assert lineno == _iceberg_line + 9
    assert info == {
        "name": "z",
        "annotation": int,
        "provenance": "body",
        "doc": "The great\nzee",
    }


def test_info_parameter():
    info = iceberg.info["x"]
    filename, fn, lineno = info.pop("location")
    assert fn.__name__ == "iceberg"
    assert lineno == _iceberg_line + 3
    assert info == {
        "name": "x",
        "annotation": float,
        "provenance": "argument",
        "doc": "The parameter x",
    }


def test_info_external():
    info = iceberg.info["sum"]
    filename, fn, lineno = info.pop("location")
    assert fn.__name__ == "iceberg"
    assert lineno is None
    assert info == {
        "name": "sum",
        "annotation": ABSENT,
        "provenance": "external",
        "doc": None,
    }


def test_docstring_preserved():
    @wrap
    def docteur(n):
        """Docstrings should be preserved."""
        diet = n * "pomme"
        return diet

    assert docteur.__doc__ == """Docstrings should be preserved."""


def _has_problem(selector, problem):
    with pytest.raises(SelectorError) as exc:
        select(selector, strict=True)
    exc_string = str(exc.value)
    assert problem in exc_string
    return True


@one_test_per_assert
def test_resolvability():
    assert _has_problem("a", "Wildcard")
    assert _has_problem("apple > banana", "Could not resolve 'apple'")
    assert _has_problem("chocolat > no", "Cannot find a variable named `no`")
    assert _has_problem(
        "iceberg > chocolat > no", "Cannot find a variable named `no`"
    )
    assert _has_problem("chocolat > x:@xyz", "does not have the category")
    assert _has_problem("chocolat > *:@xyz", "has the category")
    assert _has_problem("_has_problem > selector", "is not properly tooled")
    assert _has_problem("chocolat > #what", "#what is not a valid hashvar")

    assert select("chocolat > x", strict=True)
    assert select("chocolat > *", strict=True)
    assert select("chocolat > #enter", strict=True)
    assert select("iceberg > chocolat > x", strict=True)


@one_test_per_assert
def test_Key():
    assert str(Key("attr", "wow")) == "<Key attr='wow'>"
    assert Key("attr", "wow").affix_to("thing") == "thing.wow"
    assert Key("index", "wow").affix_to("thing") == "thing['wow']"


#################################
# Test language feature support #
#################################


@wrap
def spatula(x):
    a, b = x
    return a + b


def test_tuple_assignment():
    assert spatula((4, 5)) == 9
    assert spatula((4, 5), a=70) == 75
    assert spatula((4, 5), b=70) == 74


def test_nested_function():
    @wrap
    def kangaroo(x):
        def child(y):
            return y * y

        a = child(x)
        b = child(x + 1)
        return a + b

    assert kangaroo(3) == 25


def test_attribute_assignment():
    class X:
        pass

    @wrap(all=True)
    def obelisk(x):
        x.y = 2
        return x.y

    data = obelisk(X())
    assert data.ret == 2
    assert "x.y" in data.syms()


def test_index_assignment():
    @wrap(all=True)
    def limbo(x):
        x[0] = 2
        return x

    data = limbo([0, 1])
    assert data.ret == [2, 1]
    assert "x[0]" in data.syms()


def test_nested_no_crash():
    @wrap(all=True)
    def limbo(x):
        x[0][1] = 2
        return x

    data = limbo([[0, 1], 2])
    assert data.ret == [[0, 2], 2]
    assert data.syms() == ["#enter", "x", "#value"]


def test_empty_return():
    @wrap
    def foo():
        return

    assert foo() is None


def test_for_loop():
    @wrap
    def brooms(xs):
        rval = 0
        for i, x in enumerate(xs):
            rval = rval + (i + 1) * x
        return rval

    assert brooms([1, 2, 3]) == 14
    assert brooms([1, 2, 3], i=0) == 6


def test_while_loop():
    @wrap
    def broomy(n):
        rval = 0
        while n > 0:
            x = n * n
            rval = rval + x
            n -= 1
        return rval

    assert broomy(3) == 14
    assert broomy(3, x=2) == 6


def test_generator():
    @wrap
    def oxygen():
        j = 0
        for i in range(10):
            j = j + 1
            yield j
        return j

    results = list(oxygen())
    assert results == list(range(1, 11))

    results = list(oxygen(j=0))
    assert results == [0] * 10


def test_exception():
    @wrap
    def excite():
        try:
            1 / 0
        except ZeroDivisionError as exc:
            return exc
        except TypeError:
            return None

    assert isinstance(excite(), ZeroDivisionError)
    assert excite(exc="nope") == "nope"


def test_import_inside():
    from ptera import tools as T_orig

    @wrap
    def imp(x):
        import ptera.tools  # noqa
        import ptera.tools as T

        return T.gt(3)(x)

    assert imp(7)
    assert not imp(7, T=SimpleNamespace(gt=T_orig.lt))


def test_import_from_inside():
    from ptera.tools import lt as lt_orig

    @wrap
    def imp(x):
        from ptera.tools import gt

        return gt(3)(x)

    assert imp(7)
    assert not imp(7, gt=lt_orig)


def test_closure():
    x = 3
    y = 7

    @wrap(all=True)
    def inside_scoop():
        return x + y

    data = inside_scoop()
    assert data.ret == 10
    assert data == [
        ("#enter", None, None, True, False),
        ("x", None, None, 3, False),
        ("y", None, None, 7, False),
        ("#value", None, None, 10, True),
    ]


class Animal:
    def __init__(self, cry):
        self._cry = cry

    @wrap(all=True)
    def cry(self):
        intensity = 2
        return self._cry * intensity


def test_transform_method():
    cow = Animal("moo")
    data = cow.cry()
    assert data.ret == "moomoo"
    assert data == [
        ("#enter", None, None, True, False),
        ("self", None, None, cow, True),
        ("intensity", None, None, 2, True),
        ("#value", None, None, "moomoo", True),
    ]


def test_transform_type_error():
    with pytest.raises(TypeError, match="only works on functions"):
        transform(Animal("meow").cry, lambda *args: args)


def test_varargs():
    @wrap(all=True)
    def nightmare(x, y, *z, **k):
        return x + y + sum(z) + sum(k.values())

    # Keyword arguments in wrap() override variables, to pass
    # keyword arguments to nightmare we need to pass a special
    # KW key.
    data = nightmare(1, 2, 3, 4, KW=dict(a=5, b=6))
    assert data.ret == 21
    assert data == [
        ("#enter", None, None, True, False),
        ("sum", None, None, sum, True),
        ("x", None, None, 1, True),
        ("y", None, None, 2, True),
        ("z", None, None, (3, 4), True),
        ("k", None, None, {"a": 5, "b": 6}, True),
        ("#value", None, None, 21, True),
    ]


def test_kwonly():
    @wrap(all=True)
    def waycool(x, *, y):
        return x + y

    data = waycool(21, KW=dict(y=32))
    assert data.ret == 53
    assert data == [
        ("#enter", None, None, True, False),
        ("x", None, None, 21, True),
        ("y", None, None, 32, True),
        ("#value", None, None, 53, True),
    ]


def test_positional():
    @wrap(all=True)
    def gnarly(x, /, y):
        return x + y

    data = gnarly(21, 32)
    assert data.ret == 53
    assert data == [
        ("#enter", None, None, True, False),
        ("x", None, None, 21, True),
        ("y", None, None, 32, True),
        ("#value", None, None, 53, True),
    ]


if sys.version_info >= (3, 8, 0):

    def test_named_expression():
        from .walrus import ratatouille

        ratatouille = wrap(ratatouille)

        assert ratatouille(5) == 36
        assert ratatouille(5, y=3) == 9
