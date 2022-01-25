from itertools import count

from .core import PteraFunction, interact
from .selfless import transform

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


class PteraDecorator:
    def __init__(self, inplace=False):
        self._inplace = inplace
        if inplace:
            self.inplace = self
        else:
            self.inplace = PteraDecorator(inplace=True)

    def __call__(self, fn):
        if isinstance(fn, PteraFunction) or hasattr(fn, "__ptera__"):
            return fn
        new_fn, state = transform(fn, interact=interact)
        new_fn = PteraFunction(new_fn, state)
        if self._inplace:
            redirect(fn, new_fn)
            fn.__ptera__ = new_fn
            return fn
        else:
            return new_fn


tooled = PteraDecorator()
