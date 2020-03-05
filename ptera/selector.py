"""Specifications for call paths."""


import ast
import builtins
import sys

from . import opparse
from .categories import Category
from .utils import ABSENT


class InternedMC(type):
    def __new__(cls, name, bases, dct):
        dct["_cache"] = {}
        return super().__new__(cls, name, bases, dct)

    def __call__(cls, **kwargs):
        kwargs = {**cls._constructor_defaults, **kwargs}
        key = tuple(sorted(kwargs.items()))
        if key not in cls._cache:
            cls._cache[key] = super().__call__(**kwargs)
        return cls._cache[key]


class Element(metaclass=InternedMC):

    _constructor_defaults = {
        "value": ABSENT,
        "category": None,
        "capture": None,
        "tags": frozenset(),
        "key_field": None,
    }

    def __init__(self, *, name, value, category, capture, tags, key_field):
        self.name = name
        self.value = value
        self.category = category
        self.capture = capture
        self.tags = tags
        self.key_field = key_field
        self.focus = 1 in self.tags
        self.hasval = self.value is not ABSENT

    def with_focus(self):
        return self.clone(tags=self.tags | frozenset({1}))

    def without_focus(self):
        return self.clone(tags=self.tags - frozenset({1}))

    def clone(self, **changes):
        args = {
            "name": self.name,
            "value": self.value,
            "category": self.category,
            "capture": self.capture,
            "tags": self.tags,
            "key_field": self.key_field,
            **changes,
        }
        return Element(**args)

    def all_captures(self):
        if self.capture:
            return {self.capture}
        else:
            return set()

    def valid(self):
        if self.name is None:
            return self.focus
        else:
            return True

    def rewrite(self, required, focus=None):
        if focus is not None and focus == self.capture:
            return self.with_focus()
        elif focus is None and self.focus:
            return self
        elif self.capture not in required:
            if self.value is ABSENT:
                return None
            else:
                return self.clone(capture=None).without_focus()
        elif focus is not None:
            return self.without_focus()
        else:
            return self

    def key_captures(self):
        if self.key_field is not None:
            return {(self.capture, self.key_field)}
        else:
            return set()

    def specialize(self, specializations):
        spc = specializations.get(self.capture, ABSENT)
        if spc is ABSENT:
            return self
        rval = self.clone(
            name=self.name if spc.name is None else spc.name,
            category=self.category if spc.category is None else spc.category,
            value=self.value if spc.value is ABSENT else spc.value,
        )
        if rval.key_field == "name" and rval.name is not None:
            rval = rval.clone(key_field=None)
        if rval.key_field == "value" and rval.value is not ABSENT:
            rval = rval.clone(key_field=None)
        return rval

    def encode(self):
        if self.name is None and self.capture is not None:
            name = f"${self.capture}"
            cap = ""
        else:
            name = "*" if self.name is None else self.name
            cap = (
                ""
                if self.capture is None or self.capture == self.name
                else f" as {self.capture}"
            )
        cat = "" if self.category is None else f":{self.category}"
        focus = "!" * max(self.tags, default=0)
        val = f"={self.value}" if self.value is not ABSENT else ""
        return f"{focus}{name}{cap}{cat}{val}"

    def __str__(self):
        return f'sel("{self.encode()}")'

    __repr__ = __str__


class Call(metaclass=InternedMC):

    _constructor_defaults = {
        "children": (),
        "captures": (),
        "immediate": False,
        "collapse": False,
    }

    def __init__(self, *, element, children, captures, immediate, collapse):
        self.element = element
        self.children = children
        self.captures = captures
        self.immediate = immediate
        self.collapse = collapse
        self.focus = any(x.focus for x in self.captures + self.children)
        self.hasval = any(x.hasval for x in self.captures + self.children)

    def clone(self, **changes):
        args = {
            "element": self.element,
            "children": self.children,
            "captures": self.captures,
            "immediate": self.immediate,
            "collapse": self.collapse,
            **changes,
        }
        return Call(**args)

    def find_tag(self, tag):
        results = set()
        for child in self.children:
            results |= child.find_tag(tag)
        for cap in self.captures:
            if tag in cap.tags:
                results.add(cap)
        return results

    def all_captures(self):
        rval = set()
        for x in self.captures + self.children:
            rval.update(x.all_captures())
        return rval

    def valid(self):
        return (
            all(x.valid() for x in self.captures + self.children)
            and sum(x.focus for x in self.captures + self.children) <= 1
        )

    def rewrite(self, required, focus=None):
        captures = [x.rewrite(required, focus) for x in self.captures]
        captures = [x for x in captures if x is not None]

        children = [x.rewrite(required, focus) for x in self.children]
        children = [x for x in children if x is not None]

        if not captures and not children:
            return None

        return self.clone(captures=tuple(captures), children=tuple(children))

    def key_captures(self):
        rval = self.element.key_captures()
        for child in self.children:
            rval.update(child.key_captures())
        for cap in self.captures:
            rval.update(cap.key_captures())
        return rval

    def specialize(self, specializations):
        return self.clone(
            element=self.element and self.element.specialize(specializations),
            children=tuple(
                child.specialize(specializations) for child in self.children
            ),
            captures=tuple(
                cap.specialize(specializations) for cap in self.captures
            ),
        )

    def encode(self):
        name = self.element.encode()
        caps = []
        for cap in self.captures:
            caps.append(cap.encode())
        for child in self.children:
            enc = child.encode()
            enc = f"> {enc}" if child.immediate else f">> {enc}"
            caps.append(enc)
        caps = "" if not caps else "{" + ", ".join(caps) + "}"
        return f"{name}{caps}"

    def __str__(self):
        return f'sel("{self.encode()}")'

    __repr__ = __str__


parser = opparse.Parser(
    lexer=opparse.Lexer(
        {
            r"\s*(?:\bas\b|>>|!+|[(){}\[\]>:,$=])?\s*": "OPERATOR",
            r"[a-zA-Z_0-9#*.]+": "WORD",
        }
    ),
    order=opparse.OperatorPrecedenceTower(
        {
            ",": opparse.rassoc(10),
            ("", ">", ">>"): opparse.rassoc(100),
            "=": opparse.lassoc(120),
            ("!", "!!"): opparse.lassoc(150),
            ":": opparse.lassoc(300),
            "as": opparse.rassoc(350),
            "$": opparse.lassoc(400),
            ("(", "[", "{"): opparse.obrack(200),
            (")", "]", "}"): opparse.cbrack(500),
            ": WORD": opparse.lassoc(1000),
        }
    ),
)


def _guarantee_call(parent, context):
    if isinstance(parent, Element):
        parent = parent.clone(capture=None).without_focus()
        immediate = context == "incall"
        parent = Call(element=parent, captures=(), immediate=immediate)
    assert isinstance(parent, Call)
    return parent


class Evaluator:
    def __init__(self):
        self.actions = {}

    def register_action(self, *keys):
        def deco(fn):
            for key in keys:
                self.actions[key] = fn
            return fn

        return deco

    def __call__(self, ast, context="root"):
        assert ast is not None
        if isinstance(ast, opparse.Token):
            key = "SYMBOL"
        else:
            key = ast.key
        action = self.actions.get(key, None)
        if action is None:
            msg = f"Unrecognized operator: {key}"
            focus = ast.ops[0] if hasattr(ast, "ops") else ast
            raise focus.location.syntax_error(msg)
        return action(ast, *getattr(ast, "args", []), context=context)


evaluate = Evaluator()


@evaluate.register_action("_ ( X ) _")
def make_group(node, _1, element, _2, context):
    element = evaluate(element, context=context)
    return element


@evaluate.register_action("X > X")
def make_nested_imm(node, parent, child, context):
    parent = evaluate(parent, context=context)
    child = evaluate(child, context=context)
    parent = _guarantee_call(parent, context=context)
    if isinstance(child, Element):
        child = child.with_focus()
        return parent.clone(captures=parent.captures + (child,))
    else:
        return parent.clone(
            children=parent.children + (child.clone(immediate=True),),
        )


@evaluate.register_action("X >> X")
def make_nested(node, parent, child, context):
    parent = evaluate(parent, context=context)
    child = evaluate(child, context=context)
    parent = _guarantee_call(parent, context=context)
    if isinstance(child, Element):
        child = child.with_focus()
        child = Call(
            element=Element(name=None),
            captures=(child,),
            immediate=False,
            collapse=True,
        )
    return parent.clone(children=parent.children + (child,))


@evaluate.register_action("_ > X")
def make_nested_imm_pfx(node, _, child, context):
    child = evaluate(child, context=context)
    if isinstance(child, Element):
        return Call(
            element=Element(name=None), captures=(child,), immediate=True,
        )
    else:
        return child.clone(immediate=True)


@evaluate.register_action("_ >> X")
def make_nested_pfx(node, _, child, context):
    child = evaluate(child, context=context)
    if isinstance(child, Element):
        return Call(
            element=Element(name=None),
            captures=(child,),
            immediate=False,
            collapse=True,
        )
    else:
        return child.clone(immediate=False)


@evaluate.register_action("X : X")
def make_class(node, element, klass, context):
    element = evaluate(element, context=context)
    klass = evaluate(klass, context=context)
    assert isinstance(klass, Element)
    assert not element.category
    return element.clone(category=klass.name)


@evaluate.register_action("_ : X")
def make_class_prefix(node, _, klass, context):
    klass = evaluate(klass, context=context)
    return Element(name=None, category=klass.name, capture=None,)


@evaluate.register_action("_ ! X")
def make_focus(node, _, element, context):
    element = evaluate(element, context=context)
    assert isinstance(element, Element)
    return element.with_focus()


@evaluate.register_action("_ !! X")
def make_double_focus(node, _, element, context):
    element = evaluate(element, context=context)
    assert isinstance(element, Element)
    return element.clone(tags=frozenset(element.tags | {2}))


@evaluate.register_action("_ $ X")
def make_dollar(node, _, name, context):
    name = evaluate(name, context=context)
    return Element(
        name=None, category=None, capture=name.name, key_field="name"
    )


@evaluate.register_action("X [ X ] _")
def make_instance(node, element, key, _, context):
    element = evaluate(element, context=context)
    key = evaluate(key, context=context)
    assert isinstance(element, Element)
    assert isinstance(key, Element)
    element = _guarantee_call(element, context=context)
    key = Element(
        name="#key",
        value=key.name if key.name is not None else ABSENT,
        category=key.category,
        capture=key.capture if key.name != key.capture else None,
        key_field="value" if key.name is None else None,
    )
    return element.clone(captures=element.captures + (key,))


@evaluate.register_action("X { _ } _")
def make_call_empty_capture(node, fn, _1, _2, context):
    fn = evaluate(fn, context=context)
    fn = _guarantee_call(fn, context=context)
    return fn


@evaluate.register_action("X { X } _")
def make_call_capture(node, fn, names, _2, context):
    fn = evaluate(fn, context=context)
    names = evaluate(names, context="incall")
    names = names if isinstance(names, list) else [names]
    fn = _guarantee_call(fn, context=context)
    caps = tuple(name for name in names if isinstance(name, Element))
    children = tuple(name for name in names if isinstance(name, Call))
    return fn.clone(
        captures=fn.captures + caps, children=fn.children + children
    )


@evaluate.register_action("X , X")
def make_sequence(node, a, b, context):
    a = evaluate(a, context=context)
    b = evaluate(b, context=context)
    if not isinstance(b, list):
        b = [b]
    return [a, *b]


@evaluate.register_action("X as X")
def make_as(node, element, name, context):
    element = evaluate(element, context=context)
    name = evaluate(name, context=context)
    if isinstance(element, Element):
        return element.clone(
            capture=name.name,
            key_field="name" if element.name is None else None,
        )
    else:
        focus = context == "root"
        new_capture = Element(
            name="#value",
            capture=name.name,
            tags=frozenset({1}) if focus else frozenset(),
        )
        return element.clone(captures=element.captures + (new_capture,))


@evaluate.register_action("X = X")
def make_equals(node, element, value, context):
    element = evaluate(element, context=context)
    value = evaluate(value, context=context)
    assert isinstance(value, Element)
    return element.clone(value=value.name, capture=None)


@evaluate.register_action("SYMBOL")
def make_symbol(node, context):
    if node.value == "*":
        element = Element(name=None)
    else:
        value = node.value
        cap = node.value
        try:
            value = int(value)
            cap = None
        except ValueError:
            pass
        focus = context == "root"
        element = Element(
            name=value,
            capture=cap,
            tags=frozenset({1}) if focus else frozenset(),
        )
    return element


def parse(x):
    return evaluate(parser(x))


_string_cache = {}


class _StringFinder(ast.NodeVisitor):
    def __init__(self):
        self.strings = {}

    def visit_Str(self, node):
        self.strings[node.s] = node.lineno


def _find_string(s, filename):
    if filename not in _string_cache:
        with open(filename) as module:
            tree = ast.parse(module.read(), filename)
            finder = _StringFinder()
            finder.visit(tree)
            _string_cache[filename] = finder.strings
    return _string_cache[filename].get(s, None)


def _find_eval_env(s, fr):
    ev = None
    while fr is not None:
        filename = fr.f_code.co_filename
        if "/_pytest/" in filename or "/pluggy/" in filename:
            fr = fr.f_back
            continue
        lineno = _find_string(s, filename)
        if lineno is not None:
            if ev is not None:
                if ev[0] != filename:
                    raise Exception(f"Ambiguous env for selector '{s}'")
            ev = (filename, fr.f_globals)
        fr = fr.f_back
    return ev and ev[1]


def _eval(s, env):
    if not isinstance(s, str):
        return s
    start, *parts = s.split(".")
    if start in env:
        curr = env[start]
    elif hasattr(builtins, start):
        return getattr(builtins, start)
    else:
        raise Exception(f"Could not resolve '{start}'.")

    for part in parts:
        curr = getattr(curr, part)
    return curr


def _resolve(pattern, env):
    if isinstance(pattern, Call):
        el = _resolve(pattern.element, env)
        return pattern.clone(
            element=el,
            captures=tuple(_resolve(x, env) for x in pattern.captures),
            children=tuple(_resolve(x, env) for x in pattern.children),
        )
    elif isinstance(pattern, Element):
        category = _eval(pattern.category, env)
        if category is not None and not isinstance(category, Category):
            raise TypeError(f"A pattern can only be a Category.")
        return pattern.clone(category=category)


def _to_pattern(pattern, context="root"):
    if isinstance(pattern, str):
        pattern = parse(pattern)
    if isinstance(pattern, Element):
        pattern = Call(
            element=Element(name=None),
            captures=(pattern.with_focus(),),
            immediate=False,
        )
    assert isinstance(pattern, Call)
    return pattern


def to_pattern(s, env=None):
    if not isinstance(s, str):
        return s
    if env is None:
        fr = sys._getframe(1)
        env = _find_eval_env(s, fr)
    if env is None:
        raise Exception(f"Could not find env for selector '{s}'")
    pattern = _to_pattern(s)
    return _resolve(pattern, env)
