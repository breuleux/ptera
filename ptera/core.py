import functools
import inspect
import time
from collections import defaultdict, deque
from contextvars import ContextVar

from .selector import Element, select
from .selfless import PteraNameError
from .tags import match_tag
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
    ):
        pattern = select(pattern)

        self.pattern = pattern
        self.parent = parent
        self.template = template

        self._intercept = intercept
        if intercept is None:
            self.intercept = None

        self._trigger = trigger
        if trigger is None:
            self.trigger = None

        self._close = close
        if close is None:
            self.close = None

        self.children = []
        self.captures = {}

        if self.parent is None:
            self.names = set(pattern.all_captures)
        else:
            self.names = self.parent.names
            self.parent.children.append(self)

    def fork(self, pattern=None):
        parent = None if self.template else self
        return type(self)(
            pattern=pattern or self.pattern,
            intercept=self._intercept,
            trigger=self._trigger,
            close=self._close,
            parent=parent,
            template=False,
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

    def log(self, element, varname, category, value):
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

    def close(self):
        raise NotImplementedError()


class Total(BaseAccumulator):
    def __init__(self, pattern, close, trigger=None, **kwargs):
        super().__init__(
            pattern=pattern, trigger=trigger, close=close, **kwargs
        )

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


def Override(pattern, intercept, trigger=None):
    return Immediate(pattern, trigger, intercept=intercept)


def check_element(el, name, category):
    if el.name is not None and el.name != name:
        return False
    elif not match_tag(el.category, category):
        return False
    else:
        return True


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
            if not cap.name.startswith("#") and cap.name not in fvars:
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
                    if child.collapse:
                        to_process.append((child, acc))
                    else:
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
        if not self.handlers:
            return None

        else:
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


class PluginWrapper:
    def __init__(self, plugin, fn):
        self.hasoutput = plugin.hasoutput
        self.plugin = plugin
        self.fn = fn

    def instantiate(self):
        return ActivePluginWrapper(self.plugin.instantiate(), self.fn)


class ActivePluginWrapper:
    def __init__(self, active_plugin, fn):
        self.active_plugin = active_plugin
        self.fn = fn

    def rules(self):
        return self.active_plugin.rules()

    def finalize(self):
        rval = self.active_plugin.finalize()
        return self.fn(rval)


def selector_filterer(selector, fn):
    def new_fn(results):
        if selector.check_captures(results):
            return fn(results)
        else:
            return ABSENT

    return new_fn


class Collector:
    def __init__(self, pattern, mapper=None, immediate=False):
        self.data = []
        self.pattern = pattern
        self.mapper = mapper
        self.ruleclass = Immediate if immediate else Total

        if self.mapper:

            def listener(kwargs):
                result = self.mapper(kwargs)
                if result is not None:
                    self.data.append(result)

        else:

            def listener(kwargs):
                self.data.append(kwargs)

        if pattern.hasval:
            listener = selector_filterer(self.pattern, listener)

        self._listener = listener

    def __iter__(self):
        return iter(self.data)

    def _map_helper(self, args, transform_all, transform_one):
        if not args:
            return transform_all(self)
        elif isinstance(args[0], str):
            assert all(isinstance(arg, str) for arg in args)
            results = tuple(
                [transform_one(entry[arg]) for entry in self] for arg in args
            )
            if len(args) == 1:
                return results[0]
            else:
                return list(zip(*results))
        else:
            assert len(args) == 1
            (fn,) = args
            return [fn(entry) for entry in transform_all(self)]

    def map(self, *args):
        return self._map_helper(
            args=args,
            transform_all=lambda self: [
                {key: cap.value for key, cap in entry.items()} for entry in self
            ],
            transform_one=lambda entry: entry.value,
        )

    def map_all(self, *args):
        return self._map_helper(
            args=args,
            transform_all=lambda self: [
                {key: cap.values for key, cap in entry.items()}
                for entry in self
            ],
            transform_one=lambda entry: entry.values,
        )

    def map_full(self, *args):
        return self._map_helper(
            args=args,
            transform_all=lambda self: self,
            transform_one=lambda entry: entry,
        )

    def rules(self):
        return [self.ruleclass(self.pattern, self._listener)]

    def finalize(self):
        if self.mapper:
            return self.data
        else:
            return self


class Tap:
    hasoutput = True

    def __init__(self, selector, mapper=None, immediate=False):
        self.selector = select(selector)
        self.mapper = mapper
        self.immediate = immediate

    def hook(self, mapper):
        self.mapper = mapper
        return self

    def instantiate(self):
        return Collector(
            self.selector, mapper=self.mapper, immediate=self.immediate
        )


class CallResults:
    def __getitem__(self, item):
        if isinstance(item, int):
            item = str(item)
        try:
            return getattr(self, item)
        except AttributeError:
            raise IndexError(item)

    def __setitem__(self, item, value):
        setattr(self, str(item), value)


class StateOverlay:
    hasoutput = False

    def __init__(self, values):
        self._rules = [
            Immediate(patt, intercept=selector_filterer(select(patt), value))
            for patt, value in values.items()
        ]

    def rules(self):
        return self._rules

    def instantiate(self):
        return self

    def finalize(self):
        return self


def _to_plugin(spec, **kwargs):
    return Tap(spec, **kwargs) if isinstance(spec, str) else spec


class Overlay:
    def __init__(self, plugins={}):
        self.plugins = dict(plugins)

    @property
    def hasoutput(self):
        return any(p.hasoutput for name, p in self.plugins.items())

    def __use(self, plugins, kwplugins, immediate):
        plugins = {str(i + 1): p for i, p in enumerate(plugins)}
        plugins.update(kwplugins)
        plugins = {
            name: _to_plugin(p, immediate=immediate)
            for name, p in plugins.items()
        }
        self.plugins.update(plugins)
        return self

    def use(self, *plugins, **kwplugins):
        return self.__use(plugins, kwplugins, True)

    def full_tap(self, *plugins, **kwplugins):
        return self.__use(plugins, kwplugins, False)

    def tweak(self, values):
        values = {select(k): (lambda _, _v=v: _v) for k, v in values.items()}
        self.plugins[f"#{len(self.plugins)}"] = StateOverlay(values)
        return self

    def rewrite(self, values, full=False):
        def _wrapfn(fn, full=True):
            @functools.wraps(fn)
            def newfn(args):
                if not full:
                    args = {k: v.value for k, v in args.items()}
                return fn(args)

            return newfn

        values = {k: _wrapfn(v, full=full) for k, v in values.items()}
        self.plugins[f"#{len(self.plugins)}"] = StateOverlay(values)
        return self

    @autocreate
    def using(self, *plugins, **kwplugins):
        ol = Overlay(self.plugins)
        return ol.use(*plugins, **kwplugins)

    @autocreate
    def full_tapping(self, *plugins, **kwplugins):
        ol = Overlay(self.plugins)
        return ol.full_tap(*plugins, **kwplugins)

    @autocreate
    def tweaking(self, values):
        ol = Overlay(self.plugins)
        return ol.tweak(values)

    @autocreate
    def rewriting(self, values, full=False):
        ol = Overlay(self.plugins)
        return ol.rewrite(values, full=full)

    def collect(self, query):
        plugin = _to_plugin(query)

        def deco(fn):
            self.plugins[fn.__name__] = PluginWrapper(plugin, fn)

        return deco

    def on(self, query, full=False, all=False, immediate=True):
        plugin = _to_plugin(query, immediate=immediate)

        def deco(fn):
            def mapper(args):
                if all:
                    args = {key: cap.values for key, cap in args.items()}
                elif not full:
                    args = {key: cap.value for key, cap in args.items()}
                return fn(args)

            self.plugins[fn.__name__] = plugin.hook(mapper)

        return deco

    def __enter__(self):
        rulesets = []
        plugins = {name: p.instantiate() for name, p in self.plugins.items()}
        for plugin in plugins.values():
            rulesets.extend(plugin.rules())
        self._ol = BaseOverlay(*rulesets)
        self._ol.__enter__()
        self._results = CallResults()
        self._inst_plugins = plugins
        return self._results

    def __exit__(self, typ, exc, tb):
        self._ol.__exit__(None, None, None)
        for name, plugin in self._inst_plugins.items():
            setattr(self._results, name, plugin.finalize())


class PteraFunction:
    def __init__(
        self,
        fn,
        info,
        overlay=None,
        return_object=False,
        origin=None,
        attachments=None,
        partial_args=(),
    ):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.info = info
        self.isgenerator = inspect.isgeneratorfunction(self.fn)
        self.overlay = overlay or Overlay()
        self.return_object = return_object
        self.origin = origin or self
        self.partial_args = partial_args
        self.attachments = attachments or {}

    def clone(self, **kwargs):
        kwargs = {
            "fn": self.fn,
            "info": self.info,
            "overlay": Overlay(self.overlay.plugins),
            "return_object": self.return_object,
            "origin": self.origin,
            "partial_args": self.partial_args,
            "attachments": self.attachments,
            **kwargs,
        }
        return type(self)(**kwargs)

    def attach(self, **values):
        return self.clone(attachments={**self.attachments, **values})

    def use(self, *plugins, **kwplugins):
        self.overlay.use(*plugins, **kwplugins)
        return self

    def full_tap(self, *plugins, **kwplugins):
        self.overlay.full_tap(*plugins, **kwplugins)
        return self

    def tweak(self, values):
        self.overlay.tweak(values)
        return self

    def rewrite(self, values, full=False):
        self.overlay.rewrite(values, full=full)
        return self

    def tweaking(self, values):
        return self.clone(overlay=self.overlay.tweaking(values))

    def rewriting(self, values, full=False):
        return self.clone(overlay=self.overlay.rewriting(values, full=full))

    def using(self, *plugins, **kwplugins):
        ol = self.overlay.using(*plugins, **kwplugins)
        return self.clone(overlay=ol, return_object=ol.hasoutput)

    def full_tapping(self, *plugins, **kwplugins):
        ol = self.overlay.full_tapping(*plugins, **kwplugins)
        return self.clone(overlay=ol, return_object=ol.hasoutput)

    def collect(self, query):
        return self.overlay.collect(query)

    def on(self, query, full=False, all=False):
        return self.overlay.on(query, full=full, all=all)

    def __get__(self, obj, typ):
        if obj is None:
            return self
        else:
            return self.clone(partial_args=self.partial_args + (obj,))

    def _run_attachments(self):
        interact("#time", None, None, time.time())
        if self.attachments:
            for k, v in self.attachments.items():
                interact(f"#{k}", None, None, v)

    def __gcall__(self, *args, **kwargs):
        with self.overlay as _:
            with proceed(self):
                self._run_attachments()
                yield from self.fn(*self.partial_args, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.isgenerator:
            return self.__gcall__(*args, **kwargs)

        with self.overlay as callres:
            with proceed(self):
                self._run_attachments()
                rval = self.fn(*self.partial_args, *args, **kwargs)
                callres["0"] = callres["value"] = rval

        if self.return_object:
            return callres
        else:
            return callres.value

    def __str__(self):
        return f"{self.fn.__name__}"
