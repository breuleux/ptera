import functools
import inspect
from collections import deque
from contextvars import ContextVar
from copy import copy

from .categories import match_category
from .selector import Element, to_pattern
from .selfless import Override, Selfless, choose, override
from .utils import ABSENT, ACTIVE, COMPLETE, FAILED, call_with_captures

_pattern_fit_cache = {}


class Frame:

    top = ContextVar("Frame.top", default=None)

    def __init__(self):
        self.accumulators = set()
        self.getters = {}
        self.setters = {}
        self.to_close = []

    def register(self, acc, captures, close_at_exit):
        for cap, varnames in captures.items():
            for v in varnames:
                self.accumulators.add(v)
                if acc.rulename == "value" and cap.focus:
                    self.getters.setdefault(v, []).append((cap, acc))
                else:
                    self.setters.setdefault(v, []).append((cap, acc))
        if close_at_exit and acc.rulename == "listeners":
            self.to_close.append(acc)

    def set(self, varname, key, category, value):
        for element, acc in self.setters[varname]:
            if acc.status is ACTIVE:
                acc.varset(element, varname, category, value)

    def get(self, varname, key, category):
        rval = ABSENT
        for element, acc in self.getters[varname]:
            if acc.status is ACTIVE:
                tmp = acc.varget(element, varname, category)
                if tmp is not ABSENT:
                    rval = tmp
        return rval

    def exit(self):
        for acc in self.to_close:
            acc.close()


_empty_frame = Frame()


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

    def acquire(self, varname, value):
        assert varname is not None
        self.names.append(varname)
        self.values.append(value)

    def __str__(self):
        return f"Capture({self.element}, {self.names}, {self.values})"

    __repr__ = __str__


class Accumulator:
    def __init__(
        self,
        *,
        rulename,
        parent=None,
        rules=None,
        template=True,
        pattern=None,
        focus=True,
    ):
        self.pattern = pattern
        self.parent = parent
        self.children = []
        self.rules = rules or []
        self.captures = {}
        self.status = ACTIVE
        self.template = template
        self.focus = focus
        self.rulename = rulename
        if self.parent is not None:
            self.parent.children.append(self)

    def getcap(self, element):
        if element.capture is None:
            return None
        if element.capture not in self.captures:
            cap = Capture(element)
            self.captures[element.capture] = cap
            return cap
        else:
            return self.captures[element.capture]

    def fail(self):
        self.status = FAILED
        for leaf in self.leaves():
            leaf.status = FAILED

    def varset(self, element, varname, category, value):
        if element.value is ABSENT or element.value == value:
            acc = self.fork(pattern=element) if element.focus else self
            cap = acc.getcap(element)
            if cap:
                cap.acquire(varname, value)
        else:
            self.fail()

    def varget(self, element, varname, category):
        cap = Capture(element)
        self.captures[element.capture] = cap
        cap.names.append(varname)
        rval = self.run_value()
        del self.captures[element.capture]
        return rval

    def build(self):
        if self.parent is None:
            return self.captures
        rval = {}
        curr = self
        while curr:
            rval.update(curr.captures)
            curr = curr.parent
        return rval

    def run_value(self):
        rval = ABSENT
        args = self.build()
        for fn in self.rules:
            rval = fn(**args)
        return rval

    def run_listeners(self):
        args = self.build()
        for fn in self.rules:
            if set(args) != set(get_names(fn)):
                return ABSENT
            else:
                fn(**args)

    def leaves(self):
        if isinstance(self.pattern, Element) and self.focus:
            return [self]
        else:
            rval = []
            for child in self.children:
                rval += child.leaves()
            return rval

    def _to_merge(self):
        rval = []
        for child in self.children:
            if not child.focus:
                rval.append(child)
                rval += child._to_merge()
        return rval

    def close(self):
        assert self.rulename == "listeners"
        if self.status is ACTIVE:
            if self.parent is None:
                for acc in self._to_merge():
                    self.captures.update(acc.captures)
                leaves = self.leaves()
                for leaf in leaves or [self]:
                    leaf.run_listeners()
            self.status = COMPLETE

    def fork(self, focus=True, pattern=None):
        parent = None if self.template else self
        return Accumulator(
            parent=parent,
            rules=self.rules,
            template=False,
            pattern=pattern,
            focus=focus,
            rulename=self.rulename,
        )


def get_names(fn):
    if not hasattr(fn, "_ptera_argspec"):
        spec = inspect.getfullargspec(fn)
        if spec.args and spec.args[0] == "self":
            fn._ptera_argspec = spec.args[1:]
        else:
            fn._ptera_argspec = spec.args
    return fn._ptera_argspec


def dict_to_collection(*rulesets):
    tmp = {}
    for rules in rulesets:
        for pattern, triggers in rules.items():
            pattern = to_pattern(pattern)
            for name, entries in triggers.items():
                key = (name, pattern)
                if not isinstance(entries, (tuple, list)):
                    entries = [entries]
                for entry in entries:
                    if key not in tmp:
                        tmp[key] = Accumulator(rulename=name, pattern=pattern)
                    acc = tmp[key]
                    acc.rules.append(entry)
    return PatternCollection(
        [(pattern, acc) for (name, pattern), acc in tmp.items()]
    )


def check_element(el, name, category):
    if el.name is not None and el.name != name:
        return False
    elif not match_category(el.category, category):
        return False
    else:
        return True


def fits_pattern(pfn, pattern):
    if isinstance(pfn, str):
        fname = pfn
        fcat = None
        fvars = {}
    else:
        fname = pfn.fn.__name__
        fcat = pfn.fn.__annotations__.get("return", None)
        fvars = pfn.state_obj.__annotations__

    if not check_element(pattern.element, fname, fcat):
        return False

    capmap = {}

    for cap in pattern.captures:
        if cap.name is None:
            varnames = [
                var
                for var, ann in fvars.items()
                if check_element(cap, var, ann)
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
        self.patterns = patterns or []

    def proceed(self, fn):
        frame = _empty_frame
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
                if pattern.focus or is_template or pattern.hasval:
                    acc = acc.fork(
                        focus=pattern.focus or is_template, pattern=pattern
                    )
                if frame is _empty_frame:
                    frame = Frame()
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
        self.curr = PatternCollection.current.get()
        if self.curr is None:
            self.frame = _empty_frame
            self.frame_reset = Frame.top.set(self.frame)
            return None
        else:
            self.frame, new = self.curr.proceed(self.fn)
            self.frame_reset = Frame.top.set(self.frame)
            self.reset = PatternCollection.current.set(new)
            return new

    def __exit__(self, typ, exc, tb):
        if self.curr is not None:
            PatternCollection.current.reset(self.reset)
        Frame.top.reset(self.frame_reset)
        self.frame.exit()


class overlay:
    def __init__(self, *rulesets):
        self.rulesets = [rules for rules in rulesets if rules]

    def __enter__(self):
        if not self.rulesets:
            return None

        else:
            collection = dict_to_collection(*self.rulesets)
            curr = PatternCollection.current.get()
            if curr is not None:
                collection.patterns = curr.patterns + collection.patterns
            self.reset = PatternCollection.current.set(collection)
            return collection

    def __exit__(self, typ, exc, tb):
        if self.rulesets:
            PatternCollection.current.reset(self.reset)


def interact(sym, key, category, __self__, value):
    from_state = getattr(__self__.state_obj, sym, ABSENT)

    if key is None:
        fr = Frame.top.get()
        if sym not in fr.accumulators:
            if from_state is ABSENT and not isinstance(value, Override):
                if value is ABSENT:
                    raise NameError(f"Variable {sym} of {__self__} is not set.")
                return value
            elif value is ABSENT and not isinstance(from_state, Override):
                return from_state
            else:
                success, value = choose([value, from_state])
                if not success:
                    raise NameError(f"Variable {sym} of {__self__} is not set.")
                return value

        if sym in fr.getters:
            fr_value = fr.get(sym, key, category)
        else:
            fr_value = ABSENT
        if (
            value is ABSENT
            and fr_value is ABSENT
            and not isinstance(from_state, Override)
        ):
            value = from_state
            success = value is not ABSENT
        elif (
            fr_value is ABSENT
            and from_state is ABSENT
            and not isinstance(value, Override)
        ):
            success = value is not ABSENT
        else:
            success, value = choose([value, fr_value, from_state])
        if not success:
            raise NameError(f"Variable {sym} of {__self__} is not set.")

        if sym in fr.setters:
            fr.set(sym, key, category, value)
        return value

    else:
        assert value is not ABSENT
        with proceed(sym):
            interact("#key", None, None, __self__, key)
            # TODO: merge the return value of interact (currently raises
            # ConflictError)
            interact("#value", None, category, __self__, value)
            if value is ABSENT and not isinstance(from_state, Override):
                return from_state
            elif from_state is ABSENT and not isinstance(value, Override):
                return value
            success, value = choose([value, from_state])
            # TODO: it is not clear at the moment in what circumstance
            # success may fail to be true
            assert success
            return value


class Collector:
    def __init__(self, pattern, finalize=None):
        self.data = []
        self.pattern = pattern
        self.finalizer = finalize

        def listener(**kwargs):
            self.data.append(kwargs)

        listener._ptera_argspec = set(self.pattern.all_captures())
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
            return [fn(**entry) for entry in transform_all(self)]

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
        return {self.pattern: {"listeners": [self._listener]}}

    def finalize(self):
        if self.finalizer:
            return self.finalizer(self)
        else:
            return self


class Tap:
    hasoutput = True

    def __init__(self, selector, finalize=None):
        self.selector = to_pattern(selector)
        self.finalize = finalize

    def hook(self, finalize):
        self.finalize = finalize
        return self

    def instantiate(self):
        return Collector(self.selector, self.finalize)


class CallResults:
    def __init__(self, value):
        self.value = value
        setattr(self, "0", self.value)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = str(item)
        try:
            return getattr(self, item)
        except AttributeError:
            raise IndexError(item)


class StateOverlay:
    hasoutput = False

    def __init__(self, values):
        self._rules = {patt: {"value": value} for patt, value in values.items()}

    def rules(self):
        return self._rules

    def instantiate(self):
        return self

    def finalize(self):
        return self


def _to_plugin(spec):
    return Tap(spec) if isinstance(spec, str) else spec


def _collect_plugins(plugins, kwplugins):
    plugins = {str(i + 1): p for i, p in enumerate(plugins)}
    plugins.update(kwplugins)
    plugins = {name: _to_plugin(p) for name, p in plugins.items()}
    return plugins, any(p.hasoutput for name, p in plugins.items())


class PteraFunction(Selfless):
    def __init__(
        self, fn, state, callkey=None, plugins=None, return_object=False
    ):
        super().__init__(fn, state)
        self.callkey = callkey
        self.plugins = plugins or {}
        self.return_object = return_object
        self._match_cache = {}

    def clone(self, **kwargs):
        self.ensure_state()
        kwargs = {
            "fn": self.fn,
            "state": copy(self.state_obj),
            "callkey": self.callkey,
            "plugins": self.plugins,
            "return_object": self.return_object,
            **kwargs,
        }
        return type(self)(**kwargs)

    def __getitem__(self, callkey):
        assert self.callkey is None
        return self.clone(callkey=callkey)

    def tweak(self, values, priority=2):
        values = {
            to_pattern(k): lambda __v=v, **_: override(__v, priority)
            for k, v in values.items()
        }
        return self.using(StateOverlay(values))

    def rewrite(self, values, full=False, priority=2):
        def _wrapfn(fn, full=True):
            @functools.wraps(fn)
            def newfn(**kwargs):
                return override(
                    call_with_captures(fn, kwargs, full=full), priority=priority
                )

            newfn._ptera_argspec = get_names(fn)
            return newfn

        values = {k: _wrapfn(v, full=full) for k, v in values.items()}
        return self.using(StateOverlay(values))

    def using(self, *plugins, **kwplugins):
        plugins, return_object = _collect_plugins(plugins, kwplugins)
        return self.clone(
            plugins={**self.plugins, **plugins}, return_object=return_object,
        )

    def use(self, *plugins, **kwplugins):
        plugins, _ = _collect_plugins(plugins, kwplugins)
        self.plugins.update(plugins)
        return self

    def collect(self, query):
        plugin = _to_plugin(query)

        def deco(fn):
            self.plugins[fn.__name__] = plugin.hook(fn)

        return deco

    def on(self, query, full=False, all=False):
        plugin = _to_plugin(query)

        def deco(fn):
            def finalize(coll):
                if full:
                    return coll.map_full(fn)
                elif all:
                    return coll.map_all(fn)
                else:
                    return coll.map(fn)

            self.plugins[fn.__name__] = plugin.hook(finalize)

        return deco

    def __call__(self, *args, **kwargs):
        self.ensure_state()
        rulesets = []
        plugins = {name: p.instantiate() for name, p in self.plugins.items()}
        for plugin in plugins.values():
            rulesets.append(plugin.rules())
        with overlay(*rulesets):
            with proceed(self):
                if self.callkey is not None:
                    interact("#key", None, None, self, self.callkey)
                rval = super().__call__(*args, **kwargs)

        callres = CallResults(rval)
        for name, plugin in plugins.items():
            setattr(callres, name, plugin.finalize())

        if self.return_object:
            return callres
        else:
            return rval
