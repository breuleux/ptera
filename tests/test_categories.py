import pytest

from ptera.categories import Category, cat, match_category as mc

from .common import one_test_per_assert


@one_test_per_assert
def test_category():
    assert cat.Fruit == cat.Fruit
    assert cat.Fruit == Category("Fruit")


@one_test_per_assert
def test_category_set():
    assert (cat.Foo & cat.Baz & cat.Bar & cat.Baz).members == {
        cat.Foo,
        cat.Bar,
        cat.Baz,
    }
    assert (cat.Foo & int).members == {cat.Foo, int}
    assert (int & cat.Foo).members == {cat.Foo, int}


@one_test_per_assert
def test_category_repr():
    assert str(cat.Fruit) == "Fruit"
    assert repr(cat.Fruit) == "Fruit"
    assert str(cat.Foo & cat.Bar) == "Bar&Foo"
    assert str(cat.Foo & cat.Bar & cat.Baz) == "Bar&Baz&Foo"


@one_test_per_assert
def test_match_category():
    assert mc(cat.Fruit, cat.Fruit)
    assert mc(cat.Fruit, cat.Fruit & cat.Legume)
    assert not mc(cat.Fruit, cat.Legume)
    assert mc(cat.Fruit, cat.Fruit & int)
