import functools
import inspect
from collections import defaultdict
from contextvars import ContextVar
from copy import copy
from itertools import chain

from .categories import match_category
from .selector import to_pattern
from .selfless import Selfless, choose, override
from .utils import ABSENT, ACTIVE, COMPLETE, FAILED, call_with_captures, setvar


def check_element(el, name, category, value=ABSENT):
    if el.name is not None and el.name != name:
        return False
    elif not match_category(el.category, category, value):
        return False
    elif el.value is not ABSENT and el.value != value:
        return False
    else:
        return True


class Frame:

    top = ContextVar("Frame.top", default=None)

    def __init__(self, fname):
        self.function_name = fname
        self.accumulators = defaultdict(list)
        self.to_close = []

    def register(self, acc, captures, close_at_exit):
        for cap, varnames in captures.items():
            for v in varnames:
                self.accumulators[v].append((cap, acc))
        if close_at_exit:
            self.to_close.append(acc)

    def run(self, method, varname, category, value=ABSENT, mayfail=True):
        rval = ABSENT
        for element, acc in self.accumulators[varname]:
            if acc.status is not ACTIVE:
                continue
            acc = acc.match(element, varname, category, value, mayfail=mayfail)
            if acc:
                tmp = getattr(acc, method)(element, varname, category, value)
                if tmp is not ABSENT:
                    rval = tmp
        return rval

    def set(self, varname, key, category, value):
        self.run("varset", varname, category, value)

    def get(self, varname, key, category):
        rval = self.run("varget", varname, category, mayfail=False)
        if rval is ABSENT:
            raise NameError(f"Cannot get value for variable `{varname}`")
        return rval

    def exit(self):
        for acc in self.to_close:
            acc.close()


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
        if len(self.names) == 0:
            raise ValueError(f"No name for capture `{self.capture}`")
        if len(self.names) > 1:
            raise ValueError(
                f"Multiple names stored for capture `{self.capture}`"
            )
        return self.names[0]

    @property
    def value(self):
        if len(self.values) == 0:
            raise ValueError(f"No value for capture `{self.capture}`")
        if len(self.values) > 1:
            raise ValueError(
                f"Multiple values stored for capture `{self.capture}`"
            )
        return self.values[0]

    def nomatch(self):
        return None if self.element.name is None else False

    def check(self, varname, category, value):
        rval = check_element(self.element, varname, category, value)
        if rval:
            return True
        else:
            return self.nomatch()

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
        names,
        parent=None,
        rules=None,
        template=True,
        pattern=None,
        focus=True,
    ):
        self.names = set(names)
        self.pattern = pattern
        self.parent = parent
        self.children = []
        self.rules = rules or defaultdict(list)
        self.captures = {}
        self.status = ACTIVE
        self.template = template
        self.focus = focus
        if self.parent is not None:
            self.parent.children.append(self)

    def getcap(self, element):
        if element.capture not in self.captures:
            cap = Capture(element)
            self.captures[element.capture] = cap
        return self.captures[element.capture]

    def fail(self):
        self.status = FAILED
        for leaf in self.leaves():
            leaf.status = FAILED

    def match(self, element, varname, category, value, mayfail=True):
        assert self.status is ACTIVE
        if element.focus:
            acc = self.fork()
        else:
            acc = self
        cap = acc.getcap(element)
        status = cap.check(varname, category, value)
        if status is True:
            return acc
        elif status is False:
            if mayfail:
                self.fail()
            return None
        else:
            return None

    def varset(self, element, varname, category, value):
        assert self.status is ACTIVE
        cap = self.getcap(element)
        cap.acquire(varname, value)

    def varget(self, element, varname, category, _):
        assert self.status is ACTIVE
        if not element.focus:
            return ABSENT
        cap = self.getcap(element)
        cap.names.append(varname)
        rval = self.run("value", may_fail=False)
        if rval is ABSENT:
            cap.names.pop()
        else:
            cap.values.append(rval)
        return rval

    def build(self):
        rval = {}
        curr = self
        while curr:
            rval.update(
                {
                    name: cap
                    for name, cap in curr.captures.items()
                    if (cap.values or cap.names) and name is not None
                }
            )
            curr = curr.parent
        return rval

    def run(self, rulename, may_fail):
        assert self.status is ACTIVE
        rval = ABSENT
        for fn in self.rules[rulename]:
            args = self.build()
            _, names = get_names(fn)
            if may_fail and set(args) != set(names):
                return ABSENT
            else:
                with setvar(PatternCollection.current, None):
                    rval = fn(**args)
        return rval

    def merge(self, child):
        for name, cap in child.captures.items():
            mycap = self.getcap(cap.element)
            mycap.names += cap.names
            mycap.values += cap.values

    def leaves(self):
        if not self.children and self.focus:
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
        if self.status is ACTIVE:
            if self.parent is None:
                for acc in self._to_merge():
                    self.merge(acc)
                leaves = self.leaves()
                for leaf in leaves:
                    leaf.run("listeners", may_fail=True)
                if not leaves:
                    self.run("listeners", may_fail=True)
            self.status = COMPLETE

    def fork(self, focus=True):
        parent = None if self.template else self
        return Accumulator(
            self.names,
            parent,
            rules=self.rules,
            template=False,
            pattern=self.pattern,
            focus=focus,
        )


def get_names(fn):
    if not hasattr(fn, "_ptera_argspec"):
        spec = inspect.getfullargspec(fn)
        if spec.args and spec.args[0] == "self":
            fn._ptera_argspec = None, spec.args[1:]
        else:
            fn._ptera_argspec = None, spec.args
    return fn._ptera_argspec


def dict_to_collection(*rulesets):
    tmp = {}
    for rules in rulesets:
        for pattern, triggers in rules.items():
            pattern = to_pattern(pattern)
            for name, entries in triggers.items():
                if not isinstance(entries, (tuple, list)):
                    entries = [entries]
                for entry in entries:
                    focus, names = get_names(entry)
                    this_pattern = pattern.rewrite(names, focus=focus)
                    if this_pattern not in tmp:
                        tmp[this_pattern] = Accumulator(
                            names, pattern=this_pattern
                        )
                    acc = tmp[this_pattern]
                    acc.rules[name].append(entry)
    return PatternCollection(list(tmp.items()))


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

    capmap = {cap: [cap.name] if cap.name else [] for cap in pattern.captures}

    for cap in pattern.captures:
        if cap.name and cap.name.startswith("#"):
            continue
        if cap.name is None:
            for var, ann in fvars.items():
                if check_element(cap, var, ann):
                    capmap[cap].append(var)
        elif cap.name not in fvars:
            return False

    if any(not varnames for varnames in capmap.values()):
        return False

    return capmap


class PatternCollection:
    current = ContextVar("PatternCollection.current", default=None)

    def __init__(self, patterns=None):
        self.patterns = patterns or []

    def proceed(self, fn, frame):
        next_patterns = []
        to_process = list(self.patterns)
        while to_process:
            pattern, acc = to_process.pop()
            if not pattern.immediate:
                next_patterns.append((pattern, acc))
            if isinstance(fn, PteraFunction):
                capmap = fn._match_cache.get(pattern, ABSENT)
                if capmap is ABSENT:
                    capmap = fits_pattern(fn, pattern)
                    fn._match_cache[pattern] = capmap
            else:
                capmap = fits_pattern(fn, pattern)
            if capmap is not False:
                is_template = acc.template
                acc = acc.fork(focus=pattern.focus or is_template)
                frame.register(acc, capmap, close_at_exit=is_template)
                for child in pattern.children:
                    if child.collapse:
                        to_process.append((child, acc))
                    else:
                        next_patterns.append((child, acc))
        rval = PatternCollection(next_patterns)
        return rval


class newframe:
    def __enter__(self):
        self.frame = Frame(None)
        self.reset = Frame.top.set(self.frame)
        return self.frame

    def __exit__(self, typ, exc, tb):
        Frame.top.reset(self.reset)
        self.frame.exit()


class proceed:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.curr = PatternCollection.current.get()
        frame = Frame.top.get()
        if self.curr is None:
            return None
        else:
            new = self.curr.proceed(self.fn, frame)
        self.reset = PatternCollection.current.set(new)
        return new

    def __exit__(self, typ, exc, tb):
        if self.curr is not None:
            PatternCollection.current.reset(self.reset)


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
        try:
            fr_value = fr.get(sym, key, category)
        except NameError:
            fr_value = ABSENT
        success, value = choose([value, fr_value, from_state])
        if not success:
            raise NameError(f"Variable {sym} of {__self__} is not set.")
        fr.set(sym, key, category, value)
        return value

    else:
        assert value is not ABSENT
        with newframe():
            with proceed(sym):
                interact("#key", None, None, __self__, key)
                # TODO: merge the return value of interact (currently raises
                # ConflictError)
                interact("#value", None, category, __self__, value)
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

        listener._ptera_argspec = None, set(self.pattern.all_captures())
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
            return [
                call_with_captures(fn, entry) for entry in transform_all(self)
            ]

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
            with newframe():
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
