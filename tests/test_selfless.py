# import sys

# import pytest

# from ptera.selfless import (
#     ABSENT,
#     ConflictError,
#     Override,
#     choose,
#     default,
#     override,
#     selfless,
# )

# from .common import one_test_per_assert


# @selfless
# def iceberg(
#     # The parameter x
#     x: float,
#     # The parameter y
#     y,
# ):
#     # The great
#     # zee
#     z: int
#     return sum([x, y, z])


# @selfless(add1=False)
# def chocolat(x, y):
#     z = x + y
#     rval = z * z
#     if add1:
#         rval = rval + 1
#     return rval


# @selfless
# def puerh(x=2, y=3):
#     return x + y


# @selfless
# def helicopter(x, y):
#     return min(x, y)


# @selfless
# def wishful_thinking():
#     return unicorn


# @one_test_per_assert
# def test_selfless():
#     assert iceberg.new(z=5)(2, 3) == 10

#     assert chocolat(2, 3) == 25
#     assert chocolat.new(add1=True)(2, 3) == 26
#     assert chocolat.new(x=2, y=3)() == 25
#     assert chocolat.new(x=default(2), y=default(3))(4) == 49
#     assert chocolat.new(x=default(2), y=default(3))(y=4) == 36

#     assert puerh() == 5
#     assert puerh.new(x=7)() == 10
#     assert puerh.new(y=10)() == 12
#     assert puerh(4, 5) == 9
#     assert puerh.new(x=default(10), y=default(15))(4, 5) == 9

#     assert helicopter(2, 10) == 2
#     assert helicopter.new(min=max)(2, 10) == 10


# def test_name_error():
#     with pytest.raises(NameError):
#         iceberg(2, 3)
#     with pytest.raises(NameError):
#         chocolat()
#     with pytest.raises(NameError):
#         wishful_thinking()


# @pytest.mark.xfail(
#     reason="Selfless raises a NameError but to be consistent with Python"
#     " it should be a TypeError."
# )
# def test_missing_argument():
#     with pytest.raises(TypeError):
#         chocolat()


# def test_state_get():

#     assert isinstance(puerh.state.x, Override)
#     assert puerh.state.x.value == 2
#     assert puerh.state.x.priority == -0.5
#     assert puerh.state.y.value == 3

#     puerh2 = puerh.new(x=4, y=5)

#     assert puerh2.state.x == 4
#     assert puerh2.state.y == 5

#     # Test that global variables are in the state
#     assert iceberg.state.sum is sum


# def test_state_set():
#     puerh2 = puerh.new(x=default(4), y=default(5))

#     # Test setting the state in place
#     assert puerh2() == 9
#     puerh2.state.x = 10
#     puerh2.state.y = 20
#     assert puerh2() == 30

#     # Verify that the original function still works as before
#     assert puerh() == 5


# def test_state_invalid_variables():
#     puerh2 = puerh.new(x=4, y=5)

#     with pytest.raises(AttributeError):
#         puerh2.state.z = 50


# def test_state_unset_variables():
#     with pytest.raises(AttributeError):
#         iceberg.state.x

#     with pytest.raises(AttributeError):
#         iceberg.state.z


# def test_override():
#     ov1 = override(1, priority=1)
#     ov2 = override(2, priority=2)
#     ov3 = override(3, priority=3)

#     assert choose([ov1, ov2, ov3], name=None) == 3

#     ov1x = override(ov1)
#     assert ov1x.value == 1
#     assert ov1x.priority == 1


# def test_override_return_value():
#     with pytest.raises(ConflictError):
#         chocolat.new(rval=1234)(2, 3)

#     assert chocolat.new(rval=override(1234))(2, 3) == 1234


# def test_override_arguments():
#     with pytest.raises(ConflictError):
#         chocolat.new(x=2, y=3)(4)
#     assert chocolat.new(x=default(2), y=3)(4) == 49
#     assert chocolat.new(x=override(2), y=3)(4) == 25
#     with pytest.raises(ConflictError):
#         chocolat.new(x=override(2), y=3)(override(4))
#     assert chocolat.new(x=override(2), y=3)(override(4, priority=2)) == 49


# def test_info():
#     info = dict(type(iceberg.state).__info__["z"])
#     filename, fn, lineno = info.pop("location")
#     assert fn.__name__ == "iceberg"
#     assert lineno == 27
#     assert info == {
#         "name": "z",
#         "annotation": int,
#         "provenance": "body",
#         "doc": "The great\nzee",
#     }


# def test_info_parameter():
#     info = dict(type(iceberg.state).__info__["x"])
#     filename, fn, lineno = info.pop("location")
#     assert fn.__name__ == "iceberg"
#     assert lineno == 21
#     assert info == {
#         "name": "x",
#         "annotation": float,
#         "provenance": "argument",
#         "doc": "The parameter x",
#     }


# def test_info_external():
#     info = dict(type(iceberg.state).__info__["sum"])
#     filename, fn, lineno = info.pop("location")
#     assert fn.__name__ == "iceberg"
#     assert lineno is None
#     assert info == {
#         "name": "sum",
#         "annotation": ABSENT,
#         "provenance": "external",
#         "doc": None,
#     }


# @selfless
# def spatula(x):
#     a, b = x
#     return a + b


# def test_tuple_assignment():
#     assert spatula((4, 5)) == 9
#     assert spatula.new(a=override(70))((4, 5)) == 75
#     assert spatula.new(b=override(70))((4, 5)) == 74


# def test_nested_function():
#     @selfless
#     def kangaroo(x):
#         def child(y):
#             return y * y

#         a = child(x)
#         b = child(x + 1)
#         return a + b

#     assert kangaroo(3) == 25


# def test_attribute_assignment():
#     class X:
#         pass

#     @selfless
#     def obelisk(x):
#         x.y = 2
#         return x.y

#     assert obelisk(X()) == 2


# def test_nested_key_assignment():
#     @selfless
#     def limbo(x):
#         x[0][1] = 2
#         return x

#     assert limbo([[0, 1], 2]) == [[0, 2], 2]


# def test_empty_return():
#     @selfless
#     def foo():
#         return

#     assert foo() is None


# if sys.version_info >= (3, 8, 0):

#     def test_named_expression():
#         from .walrus import ratatouille

#         assert ratatouille(5) == 36
#         assert ratatouille.new(y=override(3))(5) == 9
