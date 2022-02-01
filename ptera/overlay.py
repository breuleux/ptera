import functools
import inspect
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

from .interpret import Frame, Immediate, Total, interact
from .selector import check_element
from .transform import transform
from .utils import autocreate

_c = count()
_pattern_fit_cache = {}


def fits_pattern(pfn, pattern):
    fname = pfn.origin
    fcat = pfn.fn.__annotations__.get("return", None)
    fvars = pfn.info

    if not check_element(pattern.element, fname, fcat):
        return False

    capmap = {}

    for cap in pattern.captures:
        if cap.name is None:
            varnames = [
                var
                for var, info in fvars.items()
                if check_element(cap, var, info["annotation"])
            ]
            if not varnames:
                return False
            capmap[cap] = varnames
        else:
            name = cap.name.split(".")[0]
            if not cap.name.startswith("#") and name not in fvars:
                return False
            capmap[cap] = [cap.name]

    return capmap


class PatternCollection:
    current = ContextVar("PatternCollection.current", default=None)

    def __init__(self, patterns=None):
        self.patterns = list(patterns or [])

    def proceed(self, fn):
        frame = Frame(fn)
        next_patterns = []
        to_process = deque(self.patterns)
        while to_process:
            pattern, acc = to_process.pop()
            if not pattern.immediate:
                next_patterns.append((pattern, acc))
            cachekey = (fn, pattern)
            capmap = _pattern_fit_cache.get(cachekey)
            if capmap is None:
                capmap = fits_pattern(fn, pattern)
                _pattern_fit_cache[cachekey] = capmap
            if capmap is not False:
                is_template = acc.template
                if pattern.focus or is_template:
                    acc = acc.fork()
                frame.register(acc, capmap, close_at_exit=is_template)
                for child in pattern.children:
                    # if child.collapse:
                    #     # This feature is related to the >> operator which
                    #     # has been removed.
                    #     to_process.append((child, acc))
                    # else:
                    next_patterns.append((child, acc))
        rval = PatternCollection(next_patterns)
        return frame, rval


class proceed:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.curr = PatternCollection.current.get() or PatternCollection([])
        self.frame, new = self.curr.proceed(self.fn)
        self.frame_reset = Frame.top.set(self.frame)
        self.reset = PatternCollection.current.set(new)
        return new

    def __exit__(self, typ, exc, tb):
        if self.curr is not None:
            PatternCollection.current.reset(self.reset)
        Frame.top.reset(self.frame_reset)
        self.frame.exit()


class BaseOverlay:
    def __init__(self, *handlers):
        self.handlers = [(h.pattern, h) for h in handlers]

    def __enter__(self):
        if self.handlers:
            collection = PatternCollection(self.handlers)
            curr = PatternCollection.current.get()
            if curr is not None:
                collection.patterns = curr.patterns + collection.patterns
            self.reset = PatternCollection.current.set(collection)
            return collection

    def __exit__(self, typ, exc, tb):
        if self.handlers:
            PatternCollection.current.reset(self.reset)


class Overlay:
    def __init__(self, rules=()):
        self.rules = list(rules)

    def fork(self):
        return type(self)(rules=self.rules)

    def register(self, query, fn, full=False, all=False, immediate=True):
        def mapper(args):
            if all:
                args = {key: cap.values for key, cap in args.items()}
            elif not full:
                args = {key: cap.value for key, cap in args.items()}
            return fn(args)

        ruleclass = Immediate if immediate else Total
        self.rules.append(ruleclass(query, mapper))

    def on(self, query, **kwargs):
        def deco(fn):
            self.register(query, fn, **kwargs)
            return fn

        return deco

    def tap(self, query, dest=None, **kwargs):
        dest = [] if dest is None else dest
        self.register(query, dest.append, **kwargs)
        return dest

    def tweak(self, values):
        self.rules.extend(
            [
                Immediate(patt, intercept=(lambda _, _v=v: _v))
                for patt, v in values.items()
            ]
        )
        return self

    def rewrite(self, values, full=False):
        def _wrapfn(fn, full=True):
            @functools.wraps(fn)
            def newfn(args):
                if not full:
                    args = {k: v.value for k, v in args.items()}
                return fn(args)

            return newfn

        self.rules.extend(
            [
                Immediate(patt, intercept=_wrapfn(v, full=full))
                for patt, v in values.items()
            ]
        )
        return self

    @autocreate
    def tweaking(self, values):
        ol = self.fork()
        return ol.tweak(values)

    @autocreate
    def rewriting(self, values, full=False):
        ol = self.fork()
        return ol.rewrite(values, full=full)

    @autocreate
    @contextmanager
    def tapping(self, query, dest=None, **kwargs):
        ol = self.fork()
        dest = ol.tap(query, dest=dest, **kwargs)
        with ol:
            yield dest

    def __enter__(self):
        rulesets = [*self.rules]
        self._ol = BaseOverlay(*rulesets)
        self._ol.__enter__()
        return self

    def __exit__(self, typ, exc, tb):
        self._ol.__exit__(None, None, None)


class PteraFunction:
    def __init__(self, fn, info, origin=None, partial_args=()):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.info = info
        self.isgenerator = inspect.isgeneratorfunction(self.fn)
        self.origin = origin or self
        self.partial_args = partial_args

    def __get__(self, obj, typ):
        if obj is None:
            return self
        else:
            return type(self)(
                fn=self.fn,
                info=self.info,
                origin=self.origin,
                partial_args=self.partial_args + (obj,),
            )

    def __gcall__(self, *args, **kwargs):
        with proceed(self):
            interact("#enter", None, None, True)
            for entry in self.fn(*self.partial_args, *args, **kwargs):
                interact("#yield", None, None, entry)
                yield entry

    def __call__(self, *args, **kwargs):
        if self.isgenerator:
            return self.__gcall__(*args, **kwargs)

        with proceed(self):
            interact("#enter", None, None, True)
            return self.fn(*self.partial_args, *args, **kwargs)

    def __str__(self):
        return f"{self.fn.__name__}"


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