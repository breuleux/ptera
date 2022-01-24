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
    assert sel.parse("a > b >> c > d") == sel.parse("a > (b >> (c > d))")
    assert sel.parse("* as x") == sel.parse("$x")
    assert sel.parse("a > b > c") == sel.parse("a\n> b\n> c")
    assert sel.parse("a > b > c") == sel.parse("\n  a > b > c\n")

    assert sel.parse("a > b") == sel.parse("a(!b)")
    assert sel.parse("a > b > c") == sel.parse("a > b(!c)")
    assert sel.parse("a > b > c") == sel.parse("a() > b(!c)")
    assert sel.parse("a >> b") == sel.parse("a(>> !b)")

    assert sel.parse("a > b(c)") == sel.parse("a(b(c))")
    assert sel.parse("a >> b(c)") == sel.parse("a(>> b(c))")

    # assert sel.parse("a[[b]]") == sel.parse("a(#key=b)")
    # assert sel.parse("a[[b]] as c") == sel.parse("a(#key=b, !#value as c)")
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

    assert sel.parse("> apple > banana") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        captures=(
            sel.Element(name="banana", capture="banana", tags=frozenset({1})),
        ),
        immediate=True,
    )

    assert sel.parse("> banana") == sel.Call(
        element=sel.Element(name=None),
        captures=(
            sel.Element(name="banana", capture="banana", tags=frozenset({1})),
        ),
        immediate=True,
    )

    assert sel.parse("apple >> banana") == sel.Call(
        element=sel.Element(name=sel.VSymbol("apple")),
        captures=(),
        children=(
            sel.Call(
                element=sel.Element(name=None),
                captures=(
                    sel.Element(
                        name="banana", capture="banana", tags=frozenset({1})
                    ),
                ),
                immediate=False,
                collapse=True,
            ),
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
                immediate=True,
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

    # assert sel.parse("apple[pie]") == sel.Call(
    #     element=sel.Element(name="apple"),
    #     captures=(sel.Element(name="#key", value=sel.VSymbol("pie")),),
    # )

    # assert sel.parse("apple[[pie]]") == sel.Call(
    #     element=sel.Element(name=sel.VSymbol("apple")),
    #     captures=(sel.Element(name="#key", value=sel.VSymbol("pie")),),
    # )

    # assert sel.parse("apple[0]") == sel.Call(
    #     element=sel.Element(name="apple"),
    #     captures=(sel.Element(name="#key", value=sel.VSymbol("0")),),
    # )

    # assert sel.parse("apple[* as filling]") == sel.Call(
    #     element=sel.Element(name="apple"),
    #     captures=(
    #         sel.Element(name="#key", capture="filling", key_field="value"),
    #     ),
    # )

    # assert sel.parse("axe > bow:Weapon > crowbar[* as length]") == sel.Call(
    #     element=sel.Element(name=sel.VSymbol("axe")),
    #     children=(
    #         sel.Call(
    #             element=sel.Element(
    #                 name=sel.VSymbol("bow"), category=sel.VSymbol("Weapon")
    #             ),
    #             children=(
    #                 sel.Call(
    #                     element=sel.Element(name="crowbar"),
    #                     captures=(
    #                         sel.Element(
    #                             name="#key", capture="length", key_field="value"
    #                         ),
    #                     ),
    #                     immediate=True,
    #                 ),
    #             ),
    #             immediate=True,
    #         ),
    #     ),
    #     immediate=False,
    # )

    assert sel.parse("$f:Fruit") == sel.Element(
        name=None, category=sel.VSymbol("Fruit"), capture="f", key_field="name"
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

    assert sel.select("apple") == sel.select(">> *(!apple)")
    assert sel.select("pie:tag.Fruit") == sel.select(">> *(!pie:tag.Fruit)")


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
    assert not sel.parse("a($b)").valid()
    assert sel.parse("a >> $b").valid()
    assert sel.parse("a(!$b)").valid()
    assert not sel.parse("a(!$b, !$c)").valid()
    assert sel.parse("a(b, c)").valid()
    assert sel.parse("a(!b, c)").valid()
    assert not sel.parse("a(!b, !c)").valid()


def _rewrite(before, after, required, focus=None):
    bef = sel.parse(before)
    transformed = bef.rewrite(required, focus)
    aft = sel.parse(after)
    return transformed == aft


@one_test_per_assert
def test_rewrite():

    assert _rewrite(
        before="bug(world) >> spider(w, e, b)",
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

    # assert _rewrite(
    #     before="a[0](x)", after="a[0](x)", required=("x",), focus=None
    # )

    assert _rewrite(before="a(!b)", after="a(!b)", required=(), focus="b")

    assert _rewrite(before="a(!b)", after="a(!b)", required=())


# @one_test_per_assert
# def test_key_captures():
#     assert sel.parse("bleu > blanc > rouge").key_captures() == set()
#     assert sel.parse("bleu > blanc[$b] > rouge").key_captures() == {
#         ("b", "value")
#     }
#     assert sel.parse("bleu > blanc[$b] > $rouge").key_captures() == {
#         ("b", "value"),
#         ("rouge", "name"),
#     }


@one_test_per_assert
def test_specialize():
    # assert sel.parse("co >> co($n) >> nut").specialize(
    #     {"n": sel.Element(name=None, value=sel.VSymbol("x"))}
    # ) == sel.parse("co >> co(x as n) >> nut")

    assert sel.parse("co >> co >> $nut").specialize(
        {"nut": sel.Element(name=None, category=sel.VSymbol("Fruit"))}
    ) == sel.parse("co >> co >> $nut:Fruit")

    assert sel.parse("co >> co >> $nut").specialize(
        {"nut": sel.Element(name="coconut", category=sel.VSymbol("Fruit"))}
    ) == sel.parse("co >> co >> (coconut as nut):Fruit")


def test_find_tag():
    expr = sel.parse("f(!!x) > g > y")
    t1 = expr.find_tag(1)
    assert len(t1) == 1
    (t1,) = t1
    assert t1.name == "y" and t1 in list(expr.children)[0].captures

    t2 = expr.find_tag(2)
    assert len(t2) == 1
    (t2,) = t2
    assert t2.name == "x" and t2 in expr.captures

    assert sel.parse("f(x) > y").find_tag(2) == set()


def _encode(x):
    return sel.parse(x).encode()


@one_test_per_assert
def test_encode():
    assert _encode("a > b") == "a(!b)"
    assert _encode("a(b)") == "a(b)"
    assert _encode("a(b) >> c") == "a(b, >> *(!c))"
    assert _encode("a(b, c(d))") == "a(b, > c(d))"
    assert _encode("$x") == "$x"
    assert _encode("$x:Zoom") == "$x:Zoom"

    assert str(sel.parse("a")) == 'sel("!a")'
    assert str(sel.parse("a > b")) == 'sel("a(!b)")'
