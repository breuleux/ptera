import sys

import pytest

from ptera import selector as sel, tag

from .common import one_test_per_assert


def lex(code):
    return [(token.value, token.type) for token in sel.parser.lexer(code)]


@one_test_per_assert
def test_lexer():

    assert lex("apple > banana") == [
        ("apple", "WORD"),
        (">", "OPERATOR"),
        ("banana", "WORD"),
    ]
    assert lex("apple banana cherry") == [
        ("apple", "WORD"),
        ("", "OPERATOR"),
        ("banana", "WORD"),
        ("", "OPERATOR"),
        ("cherry", "WORD"),
    ]
    assert lex("apple:Fruit asparagus") == [
        ("apple", "WORD"),
        (":", "OPERATOR"),
        ("Fruit", "WORD"),
        ("", "OPERATOR"),
        ("asparagus", "WORD"),
    ]
    assert lex("radish > :cake") == [
        ("radish", "WORD"),
        (">", "OPERATOR"),
        (":", "OPERATOR"),
        ("cake", "WORD"),
    ]
    assert lex("radish as cake") == [
        ("radish", "WORD"),
        ("as", "OPERATOR"),
        ("cake", "WORD"),
    ]


@one_test_per_assert
def test_parser_equivalencies():
    assert sel.parse("apple") == sel.parse("(apple)")
    assert sel.parse("a > b > c") == sel.parse("a > (b > c)")
    assert sel.parse("* as x") == sel.parse("$x")
    assert sel.parse("a > b > c") == sel.parse("a\n> b\n> c")
    assert sel.parse("a > b > c") == sel.parse("\n  a > b > c\n")

    assert sel.parse("a > b") == sel.parse("a(!b)")
    assert sel.parse("a > b > c") == sel.parse("a > b(!c)")
    assert sel.parse("a > b > c") == sel.parse("a() > b(!c)")

    assert sel.parse("a > b(c)") == sel.parse("a(b(c))")

    assert sel.parse("a() as b") == sel.parse("a(!#value as b)")

    assert sel.parse("a:b(c)") == sel.parse("(a:b)(c)")

    assert sel.parse("a(b)=c") == sel.parse("a(b, #value=c)")


@one_test_per_assert
def test_parser():
    assert sel.parse("apple") == sel.Element(
        name="apple", capture="apple", tags=frozenset({1})
    )

    assert sel.parse("apple > banana") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        captures=(
            sel.Element(name="banana", capture="banana", tags=frozenset({1})),
        ),
        immediate=False,
    )

    assert sel.parse("apple > banana > cherry") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        children=(
            sel.Call(
                element=sel.Element(name=sel.VSymbol("banana")),
                captures=(
                    sel.Element(
                        name="cherry", capture="cherry", tags=frozenset({1})
                    ),
                ),
                immediate=False,
            ),
        ),
        immediate=False,
    )

    assert sel.parse("*:Fruit") == sel.Element(
        name=None, category=sel.VSymbol("Fruit"), capture=None
    )

    assert sel.parse("apple > :Fruit") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        captures=(
            sel.Element(
                name=None,
                category=sel.VSymbol("Fruit"),
                capture=None,
                tags=frozenset({1}),
            ),
        ),
        immediate=False,
    )

    assert sel.parse("apple(a)") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        captures=(sel.Element(name="a", capture="a"),),
    )

    assert sel.parse("apple(!a)") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        captures=(sel.Element(name="a", capture="a", tags=frozenset({1})),),
    )

    assert sel.parse("apple(a, b, c, d as e)") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        captures=(
            sel.Element(name="a", capture="a"),
            sel.Element(name="b", capture="b"),
            sel.Element(name="c", capture="c"),
            sel.Element(name="d", capture="e"),
        ),
    )

    assert sel.parse("$f:Fruit") == sel.Element(
        name=None, category=sel.VSymbol("Fruit"), capture="f"
    )

    assert sel.parse("!!x") == sel.Element(
        name="x", category=None, capture="x", tags=frozenset({1, 2})
    )


def test_bad_patterns():
    with pytest.raises(SyntaxError):
        sel.parse("{x}")

    with pytest.raises(SyntaxError):
        sel.parse("%")


def apple():
    pass


@one_test_per_assert
def test_select():

    assert sel.select("apple > banana:tag.Sublime") == sel.Call(
        element=sel.Element(name=apple, capture="/0"),
        captures=(
            sel.Element(
                name="banana",
                capture="banana",
                category=tag.Sublime,
                tags=frozenset({1}),
            ),
        ),
    )


def test_select_errors():
    with pytest.raises(TypeError):
        # Variable category cannot be a type
        sel.select("x:int")

    with pytest.raises(Exception):
        sel.select("x:blahblahblah")

    with pytest.raises(Exception):
        sel.select("pie:tag.Fruit", skip_modules=["tests"])


@one_test_per_assert
def test_validity():
    assert not sel.parse("a($b)").valid
    assert sel.parse("a(!$b)").valid
    assert not sel.parse("a(!$b, !$c)").valid
    assert sel.parse("a(b, c)").valid
    assert sel.parse("a(!b, c)").valid
    assert not sel.parse("a(!b, !c)").valid


def _rewrite(before, after, required, focus=None):
    bef = sel.parse(before)
    transformed = bef.rewrite(required, focus)
    aft = sel.parse(after)
    return transformed == aft


@one_test_per_assert
def test_rewrite():

    assert _rewrite(
        before="bug(world) > spider(w, e, b)",
        after="bug(world)",
        required=("world",),
    )

    assert _rewrite(
        before="bug(world) > spider(!w, e, b)",
        after="bug > spider(w, !b)",
        required=("w",),
        focus="b",
    )

    assert _rewrite(
        before="bug(world) > spider(!w, e, b)",
        after="bug > spider(!w, b)",
        required=("w", "b"),
    )

    assert _rewrite(
        before="a(b(c), d(e, f(g)), h(!i), j)",
        after="a(d(f(g)), h(!i), j)",
        required=("g", "j"),
    )

    assert _rewrite(
        before="a(!b, !c, !d)", after="a(b, !d)", required=("b",), focus="d"
    )

    assert _rewrite(before="a(!b)", after="a(!b)", required=(), focus="b")

    assert _rewrite(before="a(!b)", after="a(!b)", required=())


@one_test_per_assert
def test_specialize():
    assert sel.parse("co > co > $nut").specialize(
        {"nut": sel.Element(name=None, category=sel.VSymbol("Fruit"))}
    ) == sel.parse("co > co > $nut:Fruit")

    assert sel.parse("co > co > $nut").specialize(
        {"nut": sel.Element(name="coconut", category=sel.VSymbol("Fruit"))}
    ) == sel.parse("co > co > (coconut as nut):Fruit")


@one_test_per_assert
def test_main():
    assert sel.select("apple > pie").main.name == "pie"
    assert sel.select("apple(!x, y)").main.name == "x"
    assert sel.select("apple(x, !y)").main.name == "y"
    assert sel.select("apple(x, y)").main is None


def test_find_tag():
    expr = sel.parse("f(!!x) > g > y")
    t1 = expr.all_tags[1]
    assert len(t1) == 1
    (t1,) = t1
    assert t1.name == "y" and t1 in list(expr.children)[0].captures

    t2 = expr.all_tags[2]
    assert len(t2) == 1
    (t2,) = t2
    assert t2.name == "x" and t2 in expr.captures

    assert sel.parse("f(x) > y").all_tags[2] == set()


@one_test_per_assert
def test_all_captures():
    assert sel.parse("f(x, !y, g(z))").all_captures == {"x", "y", "z"}
    assert sel.parse("f(x as xx, !y, g(z))").all_captures == {"xx", "y", "z"}
    assert sel.parse("f() as x").all_captures == {"x"}
    assert sel.parse("f(x=3)").all_captures == {"x"}
    assert sel.parse("g > f()").all_captures == set()


@one_test_per_assert
def test_value_evaluate():
    assert sel.parse("x=3").value.eval(None) == 3
    assert sel.parse("x=3.7").value.eval(None) == 3.7
    assert sel.parse("x='wow'").value.eval(None) == "wow"


def _encode(x):
    return sel.parse(x).encode()


@one_test_per_assert
def test_encode():
    assert _encode("a > b") == "a(!b)"
    assert _encode("a(b)") == "a(b)"
    assert _encode("a(b, c(d))") == "a(b, c(d))"
    assert _encode("$x") == "$x"
    assert _encode("$x:Zoom") == "$x:Zoom"

    assert str(sel.parse("a")) == 'sel("!a")'
    assert str(sel.parse("a > b")) == 'sel("a(!b)")'


def test_local_resolve():
    x = 3

    def inside_scoop():
        return x

    selector = sel.select("inside_scoop > x")
    assert selector.element.name is inside_scoop


def test_bad_local_resolve():
    with pytest.raises(sel.SelectorError):
        sel.select("inside_scoop > x")


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_bad_slash_selector():
    with pytest.raises(sel.SelectorError):
        sel.select("//a/b.c > x")


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="requires python3.8 or higher"
)
def test_code_not_found():
    with pytest.raises(sel.CodeNotFoundError):
        sel.select("//what > x")
