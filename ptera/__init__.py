from .categories import Category, cat
from .core import (
    PatternCollection,
    PteraFunction,
    interact,
    overlay,
    to_pattern,
)
from .recur import Recurrence
from .selfless import ConflictError, Override, default, override, transform
from .storage import Storage, initializer, updater, valuer
from .tools import Configurator, auto_cli, catalogue


class PteraDecorator:
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def defaults(self, **defaults):
        return PteraDecorator({**self.kwargs, "defaults": defaults})

    def __call__(self, fn):
        new_fn, state = transform(fn, interact=interact)
        fn = PteraFunction(new_fn, state)
        if "defaults" in self.kwargs:
            fn = fn.new(
                **{
                    k: override(v, -0.5)
                    for k, v in self.kwargs["defaults"].items()
                }
            )
        return fn


ptera = PteraDecorator({})
