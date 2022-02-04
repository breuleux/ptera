import functools
import inspect
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

from .interpret import Frame, Immediate, Total, interact
from .selector import check_element, select, verify
from .transform import transform
from .utils import autocreate, redirect

# Cache whether functions match selectors
_selector_fit_cache = {}


def fits_selector(pfn, selector):
    """Check whether a PteraFunction matches a selector.

    Arguments:
        pfn: The PteraFunction.
        selector: The selector. We are trying to match the
            outer scope.
    """
    fname = pfn.origin
    fcat = pfn.fn.__annotations__.get("return", None)
    fvars = pfn.info

    if not check_element(selector.element, fname, fcat):
        return False

    capmap = {}

    for cap in selector.captures:
        if cap.name is None:
            # Capture is generic (*, $x)
            # Check that there is a matching variable
            varnames = [
                var
                for var, info in fvars.items()
                if check_element(cap, var, info["annotation"])
            ]
            if not varnames:
                return False
            capmap[cap] = varnames
        else:
            # Check that a variable by this name exists in the
            # function's namespace
            name = cap.name.split(".")[0]
            if not cap.name.startswith("#") and name not in fvars:
                return False
            capmap[cap] = [cap.name]

    return capmap


class SelectorCollection:
    current = ContextVar("SelectorCollection.current", default=None)

    def __init__(self, selectors=None):
        self.selectors = list(selectors or [])

    def proceed(self, fn):
        frame = Frame(fn)
        next_selectors = []
        to_process = deque(self.selectors)
        while to_process:
            selector, acc = to_process.pop()
            if not selector.immediate:
                next_selectors.append((selector, acc))
            cachekey = (fn, selector)
            capmap = _selector_fit_cache.get(cachekey)
            if capmap is None:
                capmap = fits_selector(fn, selector)
                _selector_fit_cache[cachekey] = capmap
            if capmap is not False:
                is_template = acc.template
                if selector.focus or is_template:
                    acc = acc.fork()
                frame.register(acc, capmap, close_at_exit=is_template)
                for child in selector.children:
                    # if child.collapse:
                    #     # This feature is related to the >> operator which
                    #     # has been removed.
                    #     to_process.append((child, acc))
                    # else:
                    next_selectors.append((child, acc))
        rval = SelectorCollection(next_selectors)
        return frame, rval


class proceed:
    """Context manager to wrap execution of a function.

    This pushes a new :class:`~ptera.interpret.Frame` on top and proceeds
    using the current :class:`~ptera.overlay.SelectorCollection`.

    Arguments:
        fn: The function that will be executed.
    """

    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.curr = SelectorCollection.current.get() or SelectorCollection([])
        self.frame, new = self.curr.proceed(self.fn)
        self.frame_reset = Frame.top.set(self.frame)
        self.reset = SelectorCollection.current.set(new)
        return new

    def __exit__(self, typ, exc, tb):
        if self.curr is not None:
            SelectorCollection.current.reset(self.reset)
        Frame.top.reset(self.frame_reset)
        self.frame.exit()


class BaseOverlay:
    def __init__(self, *handlers):
        self.handlers = [(h.selector, h) for h in handlers]

    def __enter__(self):
        if self.handlers:
            collection = SelectorCollection(self.handlers)
            curr = SelectorCollection.current.get()
            if curr is not None:
                collection.selectors = curr.selectors + collection.selectors
            self.reset = SelectorCollection.current.set(collection)
            return collection

    def __exit__(self, typ, exc, tb):
        if self.handlers:
            SelectorCollection.current.reset(self.reset)


class Overlay:
    """An Overlay contains a set of selectors and associated rules.

    When used as a context manager, the rules are applied within the with
    block.
    """

    def __init__(self, rules=()):
        self.rules = list(rules)

    def fork(self):
        """Create a clone of this Overlay."""
        return type(self)(rules=self.rules)

    def register(self, selector, fn, full=False, all=False, immediate=True):
        """Register a function to trigger on a selector.

        Arguments:
            selector: The selector to use.
            fn: The function to register.
            full: (default False) Whether to return a dictionary of Capture objects.
            all: (default False) If not full, whether to return a list of
                results for each variable or a single value.
            immediate: (default True) Whether to use an
                :func:`~ptera.interpret.Immediate` accumulator.
                If False, use a :func:`~ptera.interpret.Total` accumulator.
        """

        def mapper(args):
            if all:
                args = {key: cap.values for key, cap in args.items()}
            elif not full:
                args = {key: cap.value for key, cap in args.items()}
            return fn(args)

        ruleclass = Immediate if immediate else Total
        self.rules.append(ruleclass(selector, mapper))

    def on(self, selector, **kwargs):
        """Make a decorator for a function to trigger on a selector.

        Arguments:
            selector: The selector to use.
            full: (default False) Whether to return a dictionary of Capture objects.
            all: (default False) If not full, whether to return a list of
                results for each variable or a single value.
            immediate: (default True) Whether to use an
                :func:`~ptera.interpret.Immediate` accumulator.
                If False, use a :func:`~ptera.interpret.Total` accumulator.
        """

        def deco(fn):
            self.register(selector, fn, **kwargs)
            return fn

        return deco

    def tap(self, selector, dest=None, **kwargs):
        """Tap values from a selector into a list.

        Arguments:
            selector: The selector to use.
            dest: The list in which to append. If None, a list is created.

        Returns:
            The list in which to append.
        """
        dest = [] if dest is None else dest
        self.register(selector, dest.append, **kwargs)
        return dest

    def tweak(self, values):
        """Override the focus variables of selectors.

        Arguments:
            values: A ``{selector: value}`` dictionary.
        """
        self.rules.extend(
            [
                Immediate(sel, intercept=(lambda _, _v=v: _v))
                for sel, v in values.items()
            ]
        )
        return self

    def rewrite(self, rewriters, full=False):
        """Override the focus variables of selectors.

        Arguments:
            rewriters: A ``{selector: override_function}`` dictionary.
        """

        def _wrapfn(fn, full=True):
            @functools.wraps(fn)
            def newfn(args):
                if not full:
                    args = {k: v.value for k, v in args.items()}
                return fn(args)

            return newfn

        self.rules.extend(
            [
                Immediate(sel, intercept=_wrapfn(v, full=full))
                for sel, v in rewriters.items()
            ]
        )
        return self

    @autocreate
    def tweaking(self, values):
        """Fork this Overlay and :func:`~ptera.overlay.Overlay.tweak`.

        Can be called on the class (``with Overlay.tweaking(...):``).
        """
        ol = self.fork()
        return ol.tweak(values)

    @autocreate
    def rewriting(self, values, full=False):
        """Fork this Overlay and :func:`~ptera.overlay.Overlay.rewrite`.

        Can be called on the class (``with Overlay.rewriting(...):``).
        """
        ol = self.fork()
        return ol.rewrite(values, full=full)

    @autocreate
    @contextmanager
    def tapping(self, selector, dest=None, **kwargs):
        """Context manager yielding a list in which results will be accumulated.

        Can be called on the class (``with Overlay.tapping(...):``).
        """
        ol = self.fork()
        dest = ol.tap(selector, dest=dest, **kwargs)
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

    def _generator_call(self, *args, **kwargs):
        with proceed(self):
            interact("#enter", None, None, True)
            for entry in self.fn(*self.partial_args, *args, **kwargs):
                interact("#yield", None, None, entry)
                yield entry

    def __call__(self, *args, **kwargs):
        if self.isgenerator:
            return self._generator_call(*args, **kwargs)

        with proceed(self):
            interact("#enter", None, None, True)
            rval = self.fn(*self.partial_args, *args, **kwargs)
            rval = interact("#value", None, None, rval)
            return rval

    def __str__(self):
        return f"{self.fn.__name__}"


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


def autotool(selector):
    """Automatically tool functions inplace.

    Arguments:
        selector: The selector to use as a basis for the tooling. Any
            function it refers to will be tooled.
    """

    def _wrap(fn):
        tooled.inplace(fn)
        if hasattr(fn, "__ptera__"):
            return fn.__ptera__
        else:
            return fn

    rval = select(selector)
    rval = rval.wrap_functions(_wrap)
    verify(rval)
    return rval
