import functools


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


class autocreate:
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype):
        if obj is None:
            obj = objtype()
        return self.fn.__get__(obj, objtype)
