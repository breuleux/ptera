"""Specifications for call paths."""


import builtins
import re
import sys
from dataclasses import dataclass
from itertools import count

from . import opparse
from .tags import Tag
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
        if self.capture and not self.capture.startswith("/"):
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
        caps = "" if not caps else "(" + ", ".join(caps) + ")"
        return f"{name}{caps}"

    def __str__(self):
        return f'sel("{self.encode()}")'

    __repr__ = __str__


parser = opparse.Parser(
    lexer=opparse.Lexer(
        {
            r"\s*(?:\bas\b|>>|!+|\[\[|\]\]|[(){}\[\]>:,$=~])?\s*": "OPERATOR",
            r"[a-zA-Z_0-9#*.-]+": "WORD",
            r"'[^']*'": "STRING",
        }
    ),
    order=opparse.OperatorPrecedenceTower(
        {
            ",": opparse.rassoc(10),
            ("", ">", ">>"): opparse.rassoc(100),
            ("=", "~"): opparse.lassoc(120),
            ("!", "!!"): opparse.lassoc(150),
            ":": opparse.lassoc(300),
            "as": opparse.rassoc(350),
            "$": opparse.lassoc(400),
            ("(", "[", "{", "[["): opparse.obrack(200),
            (")", "]", "}", "]]"): opparse.cbrack(500),
            ": WORD": opparse.lassoc(1000),
            ": STRING": opparse.lassoc(1000),
        }
    ),
)


def _guarantee_call(parent, context, resolve=True):
    if isinstance(parent, Element):
        name = VSymbol(parent.name) if parent.name and resolve else parent.name
        parent = parent.clone(capture=None, name=name).without_focus()
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


@evaluate.register_action("_ : X")
@evaluate.register_action("X : X")
def make_class(node, element, tag, context):
    element = (
        evaluate(element, context=context) if element else Element(name=None)
    )
    tag = value_evaluate(tag)
    return element.clone(category=tag)


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
def make_index(node, element, key, _, context, resolve_call=False):
    def _make_key(key, suffix):
        assert isinstance(key, Element)
        if key.value is not ABSENT:
            assert key.name is None
            val = key.value
        else:
            val = VSymbol(key.name) if key.name is not None else ABSENT
        return Element(
            name=f"#key{suffix}",
            value=val,
            category=key.category,
            capture=key.capture if key.name != key.capture else None,
            key_field="value" if key.name is None else None,
        )

    element = evaluate(element, context=context)
    key = evaluate(key, context=context)
    assert isinstance(element, Element)
    element = _guarantee_call(element, context=context, resolve=resolve_call)

    if isinstance(key, list):
        keys = tuple(_make_key(key, i) for i, key in enumerate(key))
    else:
        keys = (_make_key(key, ""),)

    return element.clone(captures=element.captures + keys)


@evaluate.register_action("X [[ X ]] _")
def make_function_index(node, element, key, _, context):
    return make_index(node, element, key, _, context, resolve_call=True)


@evaluate.register_action("X ( _ ) _")
@evaluate.register_action("X ( X ) _")
def make_call_capture(node, fn, names, _, context):
    fn = evaluate(fn, context=context)
    names = evaluate(names, context="incall") if names else []
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


@evaluate.register_action("_ = X")
@evaluate.register_action("X = X")
def make_equals(node, element, value, context, matchfn=False):
    if element is None:
        element = Element(name=None)
    else:
        element = evaluate(element, context=context)
    value = value_evaluate(value)
    if matchfn:
        value = VCall(MatchFunction, (value,))
    if isinstance(element, Element):
        capture = element.capture if matchfn else None
        return element.clone(value=value, capture=capture)
    else:
        new_element = Element(name="#value", value=value, capture=None)
        return element.clone(captures=element.captures + (new_element,))


@evaluate.register_action("_ ~ X")
@evaluate.register_action("X ~ X")
def make_matchfn(node, element, value, context):
    return make_equals(node, element, value, context, matchfn=True)


@evaluate.register_action("SYMBOL")
def make_symbol(node, context):
    if node.value == "*":
        element = Element(name=None)
    else:
        value = node.value
        cap = node.value[1:] if node.value.startswith("#") else node.value
        focus = context == "root"
        element = Element(
            name=value,
            capture=cap,
            tags=frozenset({1}) if focus else frozenset(),
        )
    return element


class VNode:
    pass


@dataclass(frozen=True)
class VSymbol(VNode):
    value: str

    def eval(self, env):
        x = self.value

        if re.match(r"-?[0-9]+\.[0-9]*", x):
            return float(x)
        elif re.match(r"-?[0-9]+", x):
            return int(x)
        elif re.match(r"'[^']*'", x):
            return x[1:-1]

        start, *parts = x.split(".")
        if start in env:
            curr = env[start]
        elif hasattr(builtins, start):
            return getattr(builtins, start)
        else:
            raise Exception(f"Could not resolve '{start}'.")

        for part in parts:
            curr = getattr(curr, part)

        return curr

    def __str__(self):
        return str(self.value)


@dataclass(frozen=True)
class VCall(VNode):
    fn: object
    args: tuple

    def eval(self, env):
        fn = _eval(self.fn, env)
        args = []
        kwargs = {}
        for arg in self.args:
            if isinstance(arg, VKeyword):
                kwargs[arg.key.value] = _eval(arg.value, env)
            else:
                args.append(_eval(arg, env))
        return fn(*args, **kwargs)


@dataclass(frozen=True)
class VKeyword(VNode):
    key: object
    value: object


def _eval(x, env):
    if isinstance(x, VNode):
        return x.eval(env)
    else:
        return x


value_evaluate = Evaluator()


@value_evaluate.register_action("X , X")
def vmake_sequence(node, a, b, context):
    a = value_evaluate(a)
    b = value_evaluate(b)
    if not isinstance(b, list):
        b = [b]
    return [a, *b]


@value_evaluate.register_action("X ( _ ) _")
@value_evaluate.register_action("X ( X ) _")
def vmake_call(node, fn, args, _, context):
    fn = value_evaluate(fn)
    args = value_evaluate(args) if args else []
    if not isinstance(args, list):
        args = [args]
    return VCall(fn, tuple(args))


@value_evaluate.register_action("X = X")
def vmake_keyword(node, key, value, context):
    key = value_evaluate(key)
    assert isinstance(key, VSymbol)
    value = value_evaluate(value)
    return VKeyword(key, value)


@value_evaluate.register_action("SYMBOL")
def vmake_symbol(node, context):
    return VSymbol(node.value)


def parse(x):
    return evaluate(parser(x))


def _find_eval_env(s, fr, skip):
    while fr is not None:
        glb = fr.f_globals
        name = glb["__name__"]
        if all(not name.startswith(pfx) for pfx in skip):
            return glb
        fr = fr.f_back
    raise AssertionError("Unreachable outside ptera.")  # pragma: no cover


@dataclass(frozen=True)
class MatchFunction:
    fn: object


def _resolve(pattern, env, cnt):
    if isinstance(pattern, Call):
        el = _resolve(pattern.element, env, cnt)
        return pattern.clone(
            element=el,
            captures=tuple(_resolve(x, env, cnt) for x in pattern.captures),
            children=tuple(_resolve(x, env, cnt) for x in pattern.children),
        )
    elif isinstance(pattern, Element):
        name = _eval(pattern.name, env)
        category = _eval(pattern.category, env)
        value = _eval(pattern.value, env)
        capture = (
            f"/{next(cnt)}" if pattern.capture is None else pattern.capture
        )
        if category is not None and not isinstance(category, Tag):
            raise TypeError(f"A pattern can only be a Tag.")
        return pattern.clone(
            name=name, category=category, value=value, capture=capture
        )


def _select(pattern, context="root"):
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


def select(s, env=None, skip_modules=[]):
    if not isinstance(s, str):
        return s
    if env is None:
        fr = sys._getframe(1)
        env = _find_eval_env(s, fr, skip=["ptera", *skip_modules])
    pattern = _select(s)
    return _resolve(pattern, env, count())
