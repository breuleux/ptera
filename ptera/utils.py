import functools
import inspect


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
# IMMEDIATE = Named("IMMEDIATE")
# NESTED = Named("NESTED")
ACTIVE = Named("ACTIVE")
COMPLETE = Named("COMPLETE")
FAILED = Named("FAILED")


class setvar:
    def __init__(self, var, value):
        self.var = var
        self.value = value

    def __enter__(self):
        self.reset = self.var.set(self.value)
        return self.value

    def __exit__(self, typ, exc, tb):
        self.var.reset(self.reset)


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


def call_with_captures(fn, captures, full=True):
    args = inspect.getfullargspec(fn)
    if args.varkw:
        kwargs = captures
    else:
        kwargs = {}
        for k, v in captures.items():
            if k in args.args or k in args.kwonlyargs or args.varkw:
                kwargs[k] = v
    if not full:
        kwargs = {k: v.value for k, v in kwargs.items()}
    return fn(**kwargs)
