import functools
import inspect
from types import FunctionType


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
    # TODO: merge with get_names
    if not hasattr(fn, "_ptera_argnames"):
        args = inspect.getfullargspec(fn)
        assert not args.varkw
        args = args.args + args.kwonlyargs
        if isinstance(fn, FunctionType):
            fn._ptera_argnames = args
    else:
        args = fn._ptera_argnames
    kwargs = {}
    for k in args:
        if k != "self":
            kwargs[k] = captures[k]
    if not full:
        kwargs = {k: v.value for k, v in kwargs.items()}
    return fn(**kwargs)
