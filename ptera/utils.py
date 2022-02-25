"""Miscellaneous utilities."""

import functools
import types


class Named:
    """A named object.

    This class can be used to construct objects with a name that will be used
    for the string representation.

    """

    def __init__(self, name):
        """Construct a named object.

        Arguments:
            name: The name of this object.
        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


ABSENT = Named("ABSENT")


def keyword_decorator(deco):
    """Wrap a decorator to optionally takes keyword arguments."""

    @functools.wraps(deco)
    def new_deco(fn=None, **kwargs):
        if fn is None:

            @functools.wraps(deco)
            def newer_deco(fn):
                return deco(fn, **kwargs)

            return newer_deco
        else:
            return deco(fn, **kwargs)

    return new_deco


class autocreate:
    """Automatically create an instance when called on the class.

    Basically makes it so that ``Klass.f()`` is equivalent to ``Klass().f()``.
    """

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype):
        if obj is None:
            obj = objtype()
        return self.fn.__get__(obj, objtype)


class cached_property:
    """Property that caches its value when we get it for the first time."""

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, cls):
        val = self.fn(obj)
        setattr(obj, self.fn.__name__, val)
        return val


_MISSING = Named("MISSING")


class DictPile:
    def __init__(self, *dicts, default=_MISSING):
        self.dicts = dicts
        self.default = default

    def __contains__(self, item):
        for d in self.dicts:
            if item in d:
                return True
        return False

    def __getitem__(self, item):
        for d in self.dicts:
            if item in d:
                return d[item]
        if self.default is _MISSING:  # pragma: no cover
            raise KeyError(item)
        else:
            return self.default


def _build_refstring(module, *path):
    if module == "__main__":
        module = ""

    return f"/{module}/" + "/".join(path)


class CodeNotFoundError(Exception):
    pass


def _extract_info(fn):
    if isinstance(fn, type) and hasattr(fn, "__init__"):
        return _extract_info(fn.__init__)
    elif (
        not isinstance(fn, types.FunctionType)
        and not isinstance(fn, types.MethodType)
        and not isinstance(fn, types.MethodWrapperType)
        and hasattr(fn, "__call__")
        and isinstance(fn.__call__, (types.FunctionType, types.MethodType))
    ):
        return _extract_info(fn.__call__)
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module is not None and qualname is not None:
        path = qualname.split(".")
        path = [p for p in path if p != "<locals>"]
        return (module, *path)
    else:
        return None


def _verify_existence(module, *path):
    import codefind

    try:
        # Verify that we find it
        codefind.find_code(*path, module=module)
        return True
    except KeyError:
        return False


def refstring(fn):
    """Return the canonical reference string to select fn.

    For example, if fn is called ``bloop`` and is located in module
    ``squid.game``, the refstring will be ``/squid.game/bloop``.
    """
    info = _extract_info(fn)
    if info is None:
        raise TypeError(f"Cannot make a refstring for {fn} of type {type(fn)}.")

    module, *path = info
    ref = _build_refstring(module, *path)

    if not _verify_existence(module, *path):
        raise CodeNotFoundError(
            f"Cannot find the canonical code reference for {fn}"
            f" (tried: '{ref}', but it did not work. Are the"
            " __module__ and __qualname__ properties accurate?)"
        )

    return ref


def is_tooled(fn):
    """Return whether a function has been tooled for Ptera."""
    return isinstance(fn, types.FunctionType) and hasattr(fn, "__ptera_info__")
