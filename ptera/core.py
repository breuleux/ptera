import functools
import inspect
import time
from collections import deque
from contextvars import ContextVar
from copy import copy

from .selector import Element, MatchFunction, select
from .selfless import Override, Selfless, choose, name_error, override
from .tags import match_tag
from .utils import ABSENT, ACTIVE, COMPLETE, FAILED, autocreate

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
                if acc.varget and cap.focus:
                    self.getters.setdefault(v, []).append((cap, acc))
                else:
                    self.setters.setdefault(v, []).append((cap, acc))
        if close_at_exit and acc.close:
            self.to_close.append(acc)

    def set(self, varname, key, category, value):
        for element, acc in self.setters[varname]:
            if acc.status is ACTIVE:
                acc.varset(element, varname, category, value)

    def get(self, varname, key, category, tentative):
        rval = ABSENT
        for element, acc in self.getters[varname]:
            if acc.status is ACTIVE:
                tmp = acc.varget(element, varname, category, tentative)
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
        if self.parent is None:
            self.names = set(pattern.all_captures())
        else:
            self.names = self.parent.names
            self.parent.children.append(self)

    def fork(self, focus=True, pattern=None):
        parent = None if self.template else self
        return type(self)(
            parent=parent,
            rules=self.rules,
            template=False,
            pattern=pattern,
            focus=focus,
        )

    def getcap(self, element):
        if element.capture not in self.captures:
            cap = Capture(element)
            self.captures[element.capture] = cap
            return cap
        else:
            return self.captures[element.capture]

    def build_all(self):
        if self.parent is None:
            return self.captures
        rval = {}
        curr = self
        while curr:
            rval.update(curr.captures)
            curr = curr.parent
        return rval

    def build(self):
        rval = self.build_all()
        return rval and {k: v for k, v in rval.items() if not k.startswith("/")}

    def check_value(self, evalue, value):
        return (
            evalue is ABSENT
            or evalue == value
            or (isinstance(evalue, MatchFunction) and evalue.fn(value))
        )

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

    varset = None
    varget = None
    close = None


class TotalAccumulator(BaseAccumulator):
    def fail(self):
        self.status = FAILED
        for leaf in self.leaves():
            leaf.status = FAILED

    def varset(self, element, varname, category, value):
        if self.check_value(element.value, value):
            acc = self.fork(pattern=element) if element.focus else self
            cap = acc.getcap(element)
            cap.accum(varname, value)
            return acc
        else:
            self.fail()
            return None

    def run(self):
        args = self.build()
        for fn in self.rules:
            if set(args) != self.names:
                return ABSENT
            else:
                fn(args)

    def close(self):
        if self.status is ACTIVE:
            if self.parent is None:
                for acc in self._to_merge():
                    self.captures.update(acc.captures)
                leaves = self.leaves()
                for leaf in leaves or [self]:
                    leaf.run()
            self.status = COMPLETE


class ImmediateAccumulator(BaseAccumulator):
    def build_all(self):
        rval = super().build_all()

        for cap in rval.values():
            element = cap.element
            if element.value is not ABSENT:
                if not self.check_value(element.value, cap.value):
                    return None

        rval = {k: cap.snapshot() for k, cap in rval.items()}
        return rval

    def varset(self, element, varname, category, value):
        acc = self.fork(pattern=element) if element.focus else self
        cap = acc.getcap(element)
        cap.set(varname, value)
        return acc

    def run(self):
        rval = ABSENT
        args = self.build()
        if args is None:
            return ABSENT
        for fn in self.rules:
            rval = fn(args)
        return rval


class SetterAccumulator(ImmediateAccumulator):
    def varset(self, element, varname, category, value):
        acc = super().varset(element, varname, category, value)
        if acc and element.focus:
            acc.run()
        return acc


class GetterAccumulator(ImmediateAccumulator):
    def varget(self, element, varname, category, tentative):
        if not check_element(element, varname, category):
            return ABSENT
        cap = Capture(element)
        self.captures[element.capture] = cap
        cap.names.append(varname)
        cap.set(varname, tentative)
        rval = self.run()
        del self.captures[element.capture]
        return rval


accumulator_classes = {
    "listeners": TotalAccumulator,
    "immediate": SetterAccumulator,
    "value": GetterAccumulator,
}


def dict_to_pattern_list(*rulesets):
    tmp = {}
    for rules in rulesets:
        for pattern, triggers in rules.items():
            pattern = select(pattern)
            for name, entries in triggers.items():
                key = (name, pattern)
                if not isinstance(entries, (tuple, list)):
                    entries = [entries]
                for entry in entries:
                    if key not in tmp:
                        tmp[key] = accumulator_classes[name](pattern=pattern)
                    acc = tmp[key]
                    acc.rules.append(entry)

    return [(pattern, acc) for (name, pattern), acc in tmp.items()]


def dict_to_collection(*rulesets):
    return PatternCollection(dict_to_pattern_list(*rulesets))


def check_element(el, name, category):
    if el.name is not None and el.name != name:
        return False
    elif not match_tag(el.category, category):
        return False
    else:
        return True


def fits_pattern(pfn, pattern):
    if isinstance(pfn, str):
        fname = pfn
        fcat = None
        fvars = {}
    else:
        fname = pfn.origin
        fcat = pfn.fn.__annotations__.get("return", None)
        fvars = pfn.state_obj.__info__

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
        self.patterns = patterns or []

    def extend(self, patterns):
        self.patterns += patterns

    def remove_all(self, patterns):
        self.patterns = [p for p in self.patterns if p not in patterns]

    def proceed(self, fn):
        frame = _empty_frame
        next_patterns = []
        to_process = deque(self.patterns)
        while to_process:
            pattern, acc = to_process.pop()
            if acc.status is FAILED:
                continue
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

    if key is None:
        from_state = getattr(__self__.state_obj, sym, ABSENT)
        fr = Frame.top.get()
        if sym not in fr.accumulators:
            if from_state is ABSENT and not isinstance(value, Override):
                if value is ABSENT:
                    raise name_error(sym, __self__)
                return value
            elif value is ABSENT and not isinstance(from_state, Override):
                return from_state
            else:
                return choose([value, from_state], name=sym)

        if sym in fr.getters:
            fr_value = fr.get(sym, key, category, value)
        else:
            fr_value = ABSENT
        if (
            value is ABSENT
            and fr_value is ABSENT
            and not isinstance(from_state, Override)
        ):
            value = from_state
        elif (
            fr_value is ABSENT
            and from_state is ABSENT
            and not isinstance(value, Override)
        ):
            pass
        else:
            value = choose([value, fr_value, from_state], name=sym)
        if value is ABSENT:
            raise NameError(f"Variable {sym} of {__self__} is not set.")

        if sym in fr.setters:
            fr.set(sym, key, category, value)
        return value

    else:
        # TODO: it is not clear at the moment in what circumstance value may be
        # ABSENT
        assert value is not ABSENT
        with proceed(sym):
            interact("#key", None, None, __self__, key)
            # TODO: merge the return value of interact (currently raises
            # ConflictError)
            interact("#value", None, category, __self__, value)
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


class Collector:
    def __init__(self, pattern, mapper=None, immediate=False):
        self.data = []
        self.pattern = pattern
        self.mapper = mapper
        self.rulename = "immediate" if immediate else "listeners"

        if self.mapper:

            def listener(kwargs):
                result = self.mapper(kwargs)
                if result is not None:
                    self.data.append(result)

        else:

            def listener(kwargs):
                self.data.append(kwargs)

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
        return {self.pattern: {self.rulename: [self._listener]}}

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
        self._rules = {patt: {"value": value} for patt, value in values.items()}

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

    def tweak(self, values, priority=2):
        values = {
            select(k): lambda _, _v=v: override(_v, priority)
            for k, v in values.items()
        }
        self.plugins[f"#{len(self.plugins)}"] = StateOverlay(values)
        return self

    def rewrite(self, values, full=False, priority=2):
        def _wrapfn(fn, full=True):
            @functools.wraps(fn)
            def newfn(args):
                if not full:
                    args = {k: v.value for k, v in args.items()}
                return override(fn(args), priority=priority)

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
    def tweaking(self, values, priority=2):
        ol = Overlay(self.plugins)
        return ol.tweak(values, priority=priority)

    @autocreate
    def rewriting(self, values, full=False, priority=2):
        ol = Overlay(self.plugins)
        return ol.rewrite(values, full=full, priority=priority)

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
            rulesets.append(plugin.rules())
        self._ol = BaseOverlay(*rulesets)
        self._ol.__enter__()
        self._results = CallResults()
        self._inst_plugins = plugins
        return self._results

    def __exit__(self, typ, exc, tb):
        self._ol.__exit__(None, None, None)
        for name, plugin in self._inst_plugins.items():
            setattr(self._results, name, plugin.finalize())


class PteraFunction(Selfless):
    def __init__(
        self,
        fn,
        state,
        overlay=None,
        return_object=False,
        origin=None,
        attachments=None,
        partial_args=(),
    ):
        super().__init__(fn, state)
        self.isgenerator = inspect.isgeneratorfunction(self.fn)
        self.overlay = overlay or Overlay()
        self.return_object = return_object
        self.origin = origin or self
        self.partial_args = partial_args
        self.attachments = attachments or {}

    def clone(self, **kwargs):
        self.ensure_state()
        kwargs = {
            "fn": self.fn,
            "state": copy(self.state_obj),
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

    def __getitem__(self, callkey):
        assert isinstance(callkey, list)
        if len(callkey) == 1:
            (callkey,) = callkey
        attach = {"key": callkey}
        if isinstance(callkey, list):
            for i, v in enumerate(callkey):
                attach[f"key{i}"] = v
        return self.attach(**attach)

    def use(self, *plugins, **kwplugins):
        self.overlay.use(*plugins, **kwplugins)
        return self

    def full_tap(self, *plugins, **kwplugins):
        self.overlay.full_tap(*plugins, **kwplugins)
        return self

    def tweak(self, values, priority=2):
        self.overlay.tweak(values, priority=priority)
        return self

    def rewrite(self, values, full=False, priority=2):
        self.overlay.rewrite(values, full=full, priority=priority)
        return self

    def tweaking(self, values, priority=2):
        return self.clone(
            overlay=self.overlay.tweaking(values, priority=priority)
        )

    def rewriting(self, values, full=False, priority=2):
        return self.clone(
            overlay=self.overlay.rewriting(values, full=full, priority=priority)
        )

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
        interact("#time", None, None, self, time.time())
        if self.attachments:
            for k, v in self.attachments.items():
                interact(f"#{k}", None, None, self, v)

    def __gcall__(self, *args, **kwargs):
        with self.overlay as _:
            with proceed(self):
                self._run_attachments()
                yield from super().__call__(*self.partial_args, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.ensure_state()
        if self.isgenerator:
            return self.__gcall__(*args, **kwargs)

        with self.overlay as callres:
            with proceed(self):
                self._run_attachments()
                rval = super().__call__(*self.partial_args, *args, **kwargs)
                callres["0"] = callres["value"] = rval

        if self.return_object:
            return callres
        else:
            return callres.value
