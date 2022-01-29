import functools
import inspect
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar

from .selector import Element, check_element, select
from .transform import PteraNameError
from .utils import ABSENT, autocreate

_pattern_fit_cache = {}


class Frame:

    top = ContextVar("Frame.top", default=None)

    def __init__(self, fn, accumulators=None):
        self.fn = fn
        self.accumulators = accumulators or defaultdict(list)
        self.to_close = []

    def register(self, acc, captures, close_at_exit):
        for element, varnames in captures.items():
            for v in varnames:
                self.accumulators[v].append((element, acc))
        if close_at_exit and acc.close:
            self.to_close.append(acc)

    def work_on(self, varname, key, category):
        return _WorkingFrame(varname, key, category, self.accumulators)

    def exit(self):
        for acc in self.to_close:
            acc.close()


class _WorkingFrame:
    def __init__(self, varname, key, category, accumulators):
        self.varname = varname
        self.key = key
        self.category = category
        self.accumulators = [
            (element, acc.accumulator_for(element))
            for element, acc in accumulators.get(varname, [])
            if check_element(element, varname, category)
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def intercept(self, tentative):
        rval = ABSENT
        for element, acc in self.accumulators:
            if element.focus and acc.intercept:
                tmp = acc.intercept(
                    element, self.varname, self.category, tentative
                )
                if tmp is not ABSENT:
                    rval = tmp
        return rval

    def log(self, value):
        for element, acc in self.accumulators:
            acc.log(element, self.varname, self.category, value)

    def trigger(self):
        for element, acc in self.accumulators:
            if element.focus and acc.trigger:
                acc.trigger()


class Capture:
    def __init__(self, element):
        self.element = element
        self.capture = element.capture
        self.names = []
        self.values = []

    @property
    def name(self):
        if self.element.name is not None:
            return self.element.name
        if len(self.names) == 1:
            return self.names[0]
        elif len(self.names) == 0:
            raise ValueError(f"No name for capture `{self.capture}`")
        else:
            raise ValueError(
                f"Multiple names stored for capture `{self.capture}`"
            )

    @property
    def value(self):
        if len(self.values) == 1:
            return self.values[0]
        elif len(self.values) == 0:
            raise ValueError(f"No value for capture `{self.capture}`")
        else:
            raise ValueError(
                f"Multiple values stored for capture `{self.capture}`"
            )

    def accum(self, varname, value):
        assert varname is not None
        self.names.append(varname)
        self.values.append(value)

    def set(self, varname, value):
        assert varname is not None
        self.names = [varname]
        self.values = [value]

    def snapshot(self):
        cap = Capture(self.element)
        cap.names = list(self.names)
        cap.values = list(self.values)
        return cap

    def __str__(self):
        return f"Capture({self.element}, {self.names}, {self.values})"

    __repr__ = __str__


class BaseAccumulator:
    def __init__(
        self,
        *,
        pattern,
        intercept=None,
        trigger=None,
        close=None,
        parent=None,
        template=True,
        check=True,
    ):
        pattern = select(pattern)

        self.pattern = pattern
        self.parent = parent
        self.template = template

        self._intercept = self.__check(intercept, check)
        if intercept is None:
            self.intercept = None

        self._trigger = self.__check(trigger, check)
        if trigger is None:
            self.trigger = None

        self._close = self.__check(close, check)
        if close is None:
            self.close = None

        self.children = []
        self.captures = {}

    def __check(self, fn, check):
        def new_fn(results):
            if self.pattern.check_captures(results):
                return fn(results)
            else:
                return ABSENT

        return new_fn if (fn and check and self.pattern.hasval) else fn

    def fork(self, pattern=None):
        parent = None if self.template else self
        return type(self)(
            pattern=pattern or self.pattern,
            intercept=self._intercept,
            trigger=self._trigger,
            close=self._close,
            parent=parent,
            template=False,
            check=False,  # False, because functions are already wrapped
        )

    def accumulator_for(self, element):
        return self

    def getcap(self, element):
        if element.capture not in self.captures:
            cap = Capture(element)
            self.captures[element.capture] = cap
            return cap
        else:
            return self.captures[element.capture]

    def build(self):
        if self.parent is None:
            return self.captures
        rval = {}
        curr = self
        while curr:
            rval.update(curr.captures)
            curr = curr.parent
        return rval

    def _call_with_snapshot(self, fn):
        args = {k: cap.snapshot() for k, cap in self.build().items()}
        return fn(args)

    def log(self, element, varname, category, value):  # pragma: no cover
        raise NotImplementedError()

    def intercept(self, element, varname, category, tentative):
        cap = Capture(element)
        self.captures[element.capture] = cap
        cap.names.append(varname)
        cap.set(varname, tentative)
        rval = self._call_with_snapshot(self._intercept)
        del self.captures[element.capture]
        return rval

    def trigger(self):
        self._call_with_snapshot(self._trigger)

    def close(self):  # pragma: no cover
        raise NotImplementedError()


class Total(BaseAccumulator):
    def __init__(self, pattern, close, trigger=None, **kwargs):
        super().__init__(
            pattern=pattern, trigger=trigger, close=close, **kwargs
        )
        if self.parent is None:
            self.names = self.pattern.all_captures
        else:
            self.names = self.parent.names
            self.parent.children.append(self)

    def accumulator_for(self, element):
        return self.fork(pattern=element) if element.focus else self

    def log(self, element, varname, category, value):
        cap = self.getcap(element)
        cap.accum(varname, value)
        return self

    def leaves(self):
        if isinstance(self.pattern, Element):
            return [self]
        else:
            rval = []
            for child in self.children:
                rval += child.leaves()
            return rval

    def close(self):
        if self.parent is None:
            leaves = self.leaves()
            for leaf in leaves or [self]:
                args = leaf.build()
                if set(args) == leaf.names:
                    leaf._close(args)


class Immediate(BaseAccumulator):
    def __init__(self, pattern, trigger=None, **kwargs):
        super().__init__(pattern=pattern, trigger=trigger, **kwargs)

    def log(self, element, varname, category, value):
        cap = self.getcap(element)
        cap.set(varname, value)
        return self


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


def interact(sym, key, category, value):
    if key is not None:
        sym = key.affix_to(sym)

    fr = Frame.top.get()

    with fr.work_on(sym, key, category) as wfr:

        fr_value = wfr.intercept(value)
        if fr_value is not ABSENT:
            value = fr_value

        if value is ABSENT:
            raise PteraNameError(sym, fr.fn)

        wfr.log(value)
        wfr.trigger()

    return value


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
