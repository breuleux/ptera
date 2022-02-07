"""Specifications for call paths."""


import builtins
import inspect
import re
import sys
from collections import defaultdict
from itertools import count

from . import opparse
from .tags import Tag, match_tag, tag as tag_factory
from .utils import ABSENT, cached_property

_valid_hashvars = ("#enter", "#value", "#yield")


class SelectorError(Exception):
    """Error raised for invalid selectors."""


def check_element(el, name, category):
    """Check if Element el matches the given name and category."""
    if el.name is not None and el.name != name:
        return False
    elif not match_tag(el.category, category):
        return False
    else:
        return True


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


class Selector(metaclass=InternedMC):
    def check_captures(self, captures):
        for v in self.all_values:
            if v.capture in captures:
                cap = captures[v.capture]
                for value in cap.values:
                    match = v.value == value or (
                        isinstance(v.value, MatchFunction) and v.value.fn(value)
                    )
                    if not match:
                        return False
        return True

    def __str__(self):
        return f'sel("{self.encode()}")'

    __repr__ = __str__


class Element(Selector):
    """Represents a variable or some other atom."""

    _constructor_defaults = {
        "value": ABSENT,
        "category": None,
        "capture": None,
        "tags": frozenset(),
    }

    def __init__(self, *, name, value, category, capture, tags):
        self.name = name
        self.value = value
        self.category = category
        self.capture = capture
        self.tags = tags

    @cached_property
    def focus(self):
        return 1 in self.tags

    @cached_property
    def hasval(self):
        return self.value is not ABSENT

    @cached_property
    def main(self):
        return self if self.focus else None

    @cached_property
    def all_captures(self):
        if self.capture and not self.capture.startswith("/"):
            return {self.capture}
        else:  # pragma: no cover
            # Does not currently happen
            return set()

    @cached_property
    def all_values(self):
        return [self] if self.hasval else []

    @cached_property
    def valid(self):
        if self.name is None:
            return self.focus
        else:
            return True

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
            **changes,
        }
        return Element(**args)

    def rewrite(self, required, focus=None):
        if focus is not None and focus == self.capture:
            return self.with_focus()
        elif focus is None and self.focus:
            return self
        elif self.capture not in required:
            return None
        elif focus is not None:
            return self.without_focus()
        else:
            return self

    def specialize(self, specializations):
        """Replace $variables in the selector using a specializations dict."""
        spc = specializations.get(self.capture, ABSENT)
        if spc is ABSENT:
            return self
        rval = self.clone(
            name=self.name if spc.name is None else spc.name,
            category=self.category if spc.category is None else spc.category,
            value=self.value if spc.value is ABSENT else spc.value,
        )
        return rval

    def wrap_functions(self, wrap):
        return self.clone(
            name=wrap(self.name),
        )

    def encode(self):
        """Return a string representation of the selector."""
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


class Call(Selector):
    """Represents a call in the call stack."""

    _constructor_defaults = {
        "children": (),
        "captures": (),
        "immediate": False,
    }

    def __init__(self, *, element, children, captures, immediate):
        self.element = element
        self.children = children
        self.captures = captures
        self.immediate = immediate

    @cached_property
    def focus(self):
        return any(x.focus for x in self.captures + self.children)

    @cached_property
    def hasval(self):
        return any(x.hasval for x in self.captures + self.children)

    @cached_property
    def all_tags(self):
        results = defaultdict(set)
        for child in self.children:
            for tag, values in child.all_tags.items():
                results[tag] |= values
        for cap in self.captures:
            for tag in cap.tags:
                results[tag].add(cap)
        return results

    @cached_property
    def main(self):
        for x in self.captures + self.children:
            if x.main is not None:
                return x.main
        return None

    @cached_property
    def all_captures(self):
        rval = set()
        for x in self.captures + self.children:
            rval.update(x.all_captures)
        return rval

    @cached_property
    def all_values(self):
        rval = []
        for x in self.captures + self.children:
            rval += x.all_values
        return rval

    @cached_property
    def valid(self):
        return (
            all(x.valid for x in self.captures + self.children)
            and sum(x.focus for x in self.captures + self.children) <= 1
        )

    def clone(self, **changes):
        args = {
            "element": self.element,
            "children": self.children,
            "captures": self.captures,
            "immediate": self.immediate,
            **changes,
        }
        return Call(**args)

    def rewrite(self, required, focus=None):
        captures = [x.rewrite(required, focus) for x in self.captures]
        captures = [x for x in captures if x is not None]

        children = [x.rewrite(required, focus) for x in self.children]
        children = [x for x in children if x is not None]

        if not captures and not children:
            return None

        return self.clone(captures=tuple(captures), children=tuple(children))

    def specialize(self, specializations):
        """Replace $variables in the selector using a specializations dict."""
        return self.clone(
            element=self.element and self.element.specialize(specializations),
            children=tuple(
                child.specialize(specializations) for child in self.children
            ),
            captures=tuple(
                cap.specialize(specializations) for cap in self.captures
            ),
        )

    def wrap_functions(self, wrap):
        return self.clone(
            element=self.element.wrap_functions(wrap),
            children=tuple(
                child.wrap_functions(wrap) for child in self.children
            ),
        )

    def encode(self):
        """Return a string representation of the selector."""
        name = self.element.encode()
        caps = []
        for cap in self.captures:
            caps.append(cap.encode())
        for child in self.children:
            caps.append(child.encode())
        caps = "" if not caps else "(" + ", ".join(caps) + ")"
        return f"{name}{caps}"

    def problems(self):
        problems = []

        func = self.element.name
        info = getattr(func, "info", None)

        if func is None:
            problems.append("Wildcard function is not allowed")

        elif info is None:
            problems.append(f"{func} is not properly tooled")

        else:
            for x in self.captures:
                if x.name is None:
                    for _, data in info.items():
                        if check_element(x, x.name, data["annotation"]):
                            break
                    else:
                        problems.append(
                            f"No variable in `{func}` has the category `{x.category}`"
                        )

                elif x.name.startswith("#"):
                    if x.name not in _valid_hashvars:
                        problems.append(
                            f"{x.name} is not a valid hashvar. Valid hashvars are: {', '.join(_valid_hashvars)}"
                        )

                else:
                    data = info.get(x.name, None)
                    if not data:
                        problems.append(
                            f"Cannot find a variable named `{x.name}` in `{func}`"
                        )

                    elif not check_element(x, x.name, data["annotation"]):
                        problems.append(
                            f"Variable `{func} > {x.name}` does not have the category `{x.category}`"
                        )

        for x in self.children:
            problems.extend(x.problems())

        return problems


parser = opparse.Parser(
    lexer=opparse.Lexer(
        {
            # r"\s*(?:\bas\b|>>|!+|\[\[|\]\]|[(){}\[\]>:,$=~])?\s*": "OPERATOR",
            r"\s*(?:\bas\b|>>|!+|\[\[|\]\]|[(){}\[\]>:,$=~])\s*|\s+": "OPERATOR",
            r"[a-zA-Z_0-9#@*./-]+": "WORD",
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
    """Always returns a Call instance.

    If given an Element, return a Call with that Element as the function
    to call. The focus and capture name are removed, if there are any.
    """
    if isinstance(parent, Element):
        name = VSymbol(parent.name) if parent.name and resolve else parent.name
        parent = parent.clone(capture=None, name=name).without_focus()
        parent = Call(element=parent, captures=(), immediate=False)
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
            children=parent.children + (child.clone(immediate=False),),
        )


# LEGACY
# @evaluate.register_action("X >> X")
# def make_nested(node, parent, child, context):
#     parent = evaluate(parent, context=context)
#     child = evaluate(child, context=context)
#     parent = _guarantee_call(parent, context=context)
#     if isinstance(child, Element):
#         child = child.with_focus()
#         child = Call(
#             element=Element(name=None),
#             captures=(child,),
#             immediate=False,
#             collapse=True,
#         )
#     return parent.clone(children=parent.children + (child,))


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
    return Element(name=None, category=None, capture=name.name)


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
        return element.clone(capture=name.name)
    else:
        focus = context == "root"
        new_capture = Element(
            name="#value",
            capture=name.name,
            tags=frozenset({1}) if focus else frozenset(),
        )
        return element.clone(captures=element.captures + (new_capture,))


@evaluate.register_action("X = X")
def make_equals(node, element, value, context, matchfn=False):
    element = evaluate(element, context=context)
    value = value_evaluate(value)
    if matchfn:
        value = VCall(MatchFunction, (value,))
    if isinstance(element, Element):
        return element.clone(value=value, capture=element.capture)
    else:
        new_element = Element(name="#value", value=value, capture="#value")
        return element.clone(captures=element.captures + (new_element,))


@evaluate.register_action("X ~ X")
def make_matchfn(node, element, value, context):
    return make_equals(node, element, value, context, matchfn=True)


@evaluate.register_action("SYMBOL")
def make_symbol(node, context):
    if node.value == "*":
        element = Element(name=None)
    else:
        focus = context == "root"
        element = Element(
            name=node.value,
            capture=node.value,
            tags=frozenset({1}) if focus else frozenset(),
        )
    return element


def dict_resolver(env):
    """Resolve a symbol from a dictionary, e.g. the globals directory."""

    def resolve(x):
        if x.startswith("/"):
            import codefind

            _, module, *parts = x.split("/")
            co = codefind.find_code(*parts, module=module or "__main__")
            funcs = [
                fn
                for fn in codefind.get_functions(co)
                if inspect.isfunction(fn)
                and not fn.__name__.endswith("__ptera_redirect")
            ]
            if not funcs:  # pragma: no cover
                raise Exception(f"Reference `{x}` cannot be resolved.")
            elif len(funcs) > 1:  # pragma: no cover
                raise Exception(f"Reference `{x}` is ambiguous.")
            (curr,) = funcs

        elif x.startswith("@"):
            return getattr(tag_factory, x[1:])

        else:
            start, *parts = x.split(".")
            if start in env:
                curr = env[start]
            elif hasattr(builtins, start):
                return getattr(builtins, start)
            else:
                raise SelectorError(f"Could not resolve '{start}'.")

            for part in parts:
                curr = getattr(curr, part)

        return getattr(curr, "__ptera__", curr)

    return resolve


class VNode:
    pass


class VSymbol(VNode):
    def __init__(self, value):
        self.value = value

    def eval(self, env):
        x = self.value

        if re.fullmatch(r"-?[0-9]+\.[0-9]*", x):
            return float(x)
        elif re.fullmatch(r"-?[0-9]+", x):
            return int(x)
        elif re.fullmatch(r"'[^']*'", x):
            return x[1:-1]
        elif isinstance(env, dict):
            return dict_resolver(env)(x)
        else:
            return env(x)

    def __eq__(self, other):
        return isinstance(other, VSymbol) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return str(self.value)


class VCall(VNode):
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

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

    def __eq__(self, other):
        return (
            isinstance(other, VCall)
            and self.fn == other.fn
            and self.args == other.args
        )

    def __hash__(self):
        return hash((self.fn, self.args))


class VKeyword(VNode):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __eq__(self, other):  # pragma: no cover
        return (
            isinstance(other, VCall)
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self):
        return hash((self.key, self.value))


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
        if "__ptera_resolver__" in glb:
            return glb["__ptera_resolver__"]
        name = glb["__name__"]
        if all(not name.startswith(pfx) for pfx in skip):
            return glb
        fr = fr.f_back
    raise AssertionError("Unreachable outside ptera.")  # pragma: no cover


class MatchFunction:
    def __init__(self, fn):
        self.fn = fn


def _resolve(selector, env, cnt):
    if isinstance(selector, Call):
        el = _resolve(selector.element, env, cnt)
        return selector.clone(
            element=el,
            captures=tuple(_resolve(x, env, cnt) for x in selector.captures),
            children=tuple(_resolve(x, env, cnt) for x in selector.children),
        )
    elif isinstance(selector, Element):
        name = _eval(selector.name, env)
        category = _eval(selector.category, env)
        value = _eval(selector.value, env)
        capture = (
            f"/{next(cnt)}" if selector.capture is None else selector.capture
        )
        if category is not None and not isinstance(category, Tag):
            raise TypeError("A selector's category can only be a Tag.")
        return selector.clone(
            name=name, category=category, value=value, capture=capture
        )


def _select(selector, context="root"):
    if isinstance(selector, str):
        selector = parse(selector)
    if isinstance(selector, Element):
        selector = Call(
            element=Element(name=None),
            captures=(selector.with_focus(),),
            immediate=False,
        )
    assert isinstance(selector, Call)
    return selector


def select(s, env=None, skip_modules=[], skip_frames=0, strict=False):
    """Create a selector from a string.

    Arguments:
        s: The string to compile to a Selector, or a Selector to return
            unchanged.
        env: The environment to use to evaluate symbols in the selector.
            If not given, the environment chosen is the parent scope.
        skip_modules: Modules to skip when looking for an environment.
            We will go up through the stack until we get to a scope that
            is outside these modules.
        skip_frames: Number of frames to skip when looking for an
            environment.
        strict: Whether to require functions and variables in the selector
            to be statically resolvable (will give better errors).
    """
    if not isinstance(s, str):
        return s
    if env is None:
        fr = sys._getframe(skip_frames + 1)
        env = _find_eval_env(s, fr, skip=["ptera", "contextlib", *skip_modules])
    sel = _select(s)
    rval = _resolve(sel, env, count())

    if strict:
        verify(rval, display=s)

    return rval


def verify(selector, display=None):
    display = display or selector
    problems = selector.problems()
    if problems:
        raise SelectorError(repr(display) + ": " + "\n".join(problems))
    return selector
