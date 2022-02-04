import functools
from itertools import count


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
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype):
        if obj is None:
            obj = objtype()
        return self.fn.__get__(obj, objtype)


_c = count()
_redirector = """
def {}(*args, **kwargs):
    return {}(*args, **kwargs)
"""


def redirect(fn, new_fn):
    """Redirect fn to new_fn.

    After this, calling fn(...) will be equivalent to calling new_fn(...).
    """
    # We must create a unique global variable to avoid clobbering the same
    # reference with multiple invocations of redirect in the same global
    # scope.
    uniq = f"____ptera_redirect_{next(_c)}"
    fname = f"{fn.__name__}__ptera_redirect"
    glb = {}
    exec(_redirector.format(fname, uniq), glb)
    # The new code will still use the same globals, so we need to inject
    # the new function in there. This is why we generated a unique name.
    fn.__globals__[uniq] = new_fn
    # We replace the code pointer
    try:
        from codefind import code_registry

        code_registry.update_cache_entry(fn, fn.__code__, glb[fname].__code__)
    except ImportError:  # pragma: no cover
        pass
    fn.__code__ = glb[fname].__code__
