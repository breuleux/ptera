import sys

import pytest
import rx

from ptera import tooled
from ptera.core import BaseOverlay, Total
from ptera.probe import global_probe, probing

from .milk import cheese as ch, gouda


def f(x):
    a: "@flag" = x * x  # type: ignore
    return a + 1


def g(x):
    a = -x
    return a + 1


class Elephant:
    def __init__(self, secret):
        self.secret = secret

    def sing(self, x):
        wow = x + self.secret
        return wow


def loopy():
    el = Elephant(10)
    acc = 0
    for i in range(100):
        acc += f(i)
        acc += g(i)
        acc += el.sing(i)
    return acc


def nitrogen(n):
    for i in range(n):
        j = i * i
        yield j


def click(n):
    return n * n


def clack(n):
    a = click(n // 2)
    b = click(n // 3)
    return a + b


def test_probe():
    probe = global_probe("f > a")
    results = probe["a"].accum()
    loopy()
    assert results == [x * x for x in range(100)]

    probe.__exit__()

    results.clear()
    loopy()
    assert results == []


def test_probe_raw():
    with probing("f > a", raw=True) as probe:
        results = probe["a"].map(lambda x: x.value).accum()
        loopy()
        assert results == [x * x for x in range(100)]

    results.clear()
    loopy()
    assert results == []


def test_probe_method():
    with probing("Elephant.sing > wow") as probe:
        results = probe["wow"].accum()
        loopy()

    assert results == [x + 10 for x in range(100)]


def test_probe_tag():
    with probing("f > $x:@flag") as probe:
        results = probe["x"].accum()
        loopy()

    assert results == [x * x for x in range(100)]


def test_pipe():
    with probing("f > a") as probe:
        neg = probe.kmap(lambda a: -a)
        results = neg.accum()
        loopy()

    assert results == [-x * x for x in range(100)]


def test_merge():
    with probing("f > a") as probe:
        arr = rx.of(-1, -2, -3)
        merged = probe["a"] | arr
        results = merged.accum()

        loopy()

    assert set(results) == {x * x for x in range(100)} | {-1, -2, -3}


def test_probing_nested():
    with probing("click > n") as prb1:
        with probing("clack > b") as prb2:
            prb = prb1 | prb2
            results = prb.accum()
            clack(9)
            assert results == [
                {"n": 4},
                {"n": 3},
                {"b": 9},
            ]


def test_probing_multi():
    with probing("click > n", "clack > b") as prb:
        results = prb.accum()
        clack(9)
        assert results == [
            {"n": 4},
            {"n": 3},
            {"b": 9},
        ]


def test_two_probes():
    probe1 = global_probe("f > a")
    probe2 = global_probe("g > a")

    results1 = probe1["a"].accum()
    results2 = probe2["a"].accum()

    loopy()
    assert results1 == [x * x for x in range(100)]
    assert results2 == [-x for x in range(100)]

    probe1.__exit__()
    probe2.__exit__()


def test_probe_same_var_twice():
    probe1 = global_probe("f > a")
    results1 = probe1["a"].accum()

    loopy()
    assert results1 == [x * x for x in range(100)]

    probe2 = global_probe("f > a")  # Different probe for same var
    results2 = probe2["a"].accum()

    loopy()
    assert results1 == [x * x for x in range(100)] * 2
    assert results2 == [x * x for x in range(100)]


def test_bad_probe():
    with pytest.raises(NameError):
        global_probe("unknown > a")


def test_probing():
    with probing("f > a") as probe:
        results = probe["a"].max().accum()
        loopy()
    assert results == [99 ** 2]


def test_probing_generator():
    with probing("nitrogen > j") as probe:
        results = probe["j"].accum()
        for x in nitrogen(10):
            if x > 10:
                break
    assert results == [0, 1, 4, 9, 16]


def test_probing_format(capsys):
    with probing("f > a").print("a={a}"):
        loopy()
    captured = capsys.readouterr()
    assert captured.out == "".join(f"a={a * a}\n" for a in range(100))


def test_reactivate():
    prb = probing("f > a")
    with prb:
        pass

    with pytest.raises(Exception):
        with prb:
            pass


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_slash_probe():
    with probing("/tests.milk/cheese > a") as probe:
        results = probe["a"].accum()
        assert ch(4) == 16 + 4
        assert ch(5) == 25 + 4
    assert results == [16, 25]


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_slash_probe_multi():
    # Probing a function rewrites it and gives it a new code object, so
    # we need to check if we can find multiple times reliably

    with probing("/tests.milk/gouda > a") as probe:
        results = probe["a"].accum()
        assert gouda(4) == 64
    assert results == [8]

    with probing("/tests.milk/gouda > b") as probe:
        results = probe["b"].accum()
        assert gouda(4) == 64
    assert results == [64]

    with probing("/tests.milk/gouda() as ret") as probe:
        results = probe["ret"].accum()
        assert gouda(4) == 64
    assert results == [64]


def test_accumulate():
    with probing("f > a").values() as results:
        f(4)
        f(5)

    assert results == [{"a": 16}, {"a": 25}]


def test_accumulate2():
    with probing("f > a")["a"].values() as results:
        f(4)
        f(5)

    assert results == [16, 25]


def test_probe_override():
    with probing("f > a") as probe:
        probe.override(lambda a: 1234)
        assert f(5) == 1235

    with probing("f > a") as probe:
        probe.override(10)
        assert f(5) == 11

    with probing("f > a") as probe:
        probe.map(lambda _: 100).override()
        assert f(5) == 101


def test_probe_koverride():
    with probing("f > a") as probe:
        probe.koverride(lambda a: a * a)
        assert f(5) == 5 ** 4 + 1


def test_probing_no_arguments():
    with pytest.raises(TypeError):
        with probing():
            pass


@tooled
def fT(x):
    a: "@flag" = x * x  # type: ignore
    return a + 1


def test_probe_in_overlay():
    results = []

    def listener(args):
        results.append({name: cap.values for name, cap in args.items()})

    with BaseOverlay(Total("fT > a", listener)):
        fT(5)
        with probing("fT > a") as probe:
            results2 = probe.accum()
            fT(6)
        fT(7)

    assert results == [{"a": [25]}, {"a": [36]}, {"a": [49]}]
    assert results2 == [{"a": 36}]


def test_overlay_in_probe():
    results = []

    def listener(args):
        results.append({name: cap.values for name, cap in args.items()})

    with probing("fT > a") as probe:
        results2 = probe.accum()
        fT(5)
        with BaseOverlay(Total("fT > a", listener)):
            fT(6)
        fT(7)

    assert results == [{"a": [36]}]
    assert results2 == [{"a": 25}, {"a": 36}, {"a": 49}]
