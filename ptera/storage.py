import functools
from collections import defaultdict
from dataclasses import dataclass

from .core import Capture, get_names
from .selector import Element, to_pattern
from .utils import ABSENT, call_with_captures, keyword_decorator


@dataclass
class Role:
    role: str
    target: object
    full: bool

    def make_capture(self):
        cap = Capture(self.target)
        if self.target.name is not None:
            cap.acquire(self.target.name, self.target.value)
        return {self.target.capture: cap}


def role_wrapper(role):
    @keyword_decorator
    def wrap(
        fn,
        target=None,
        target_name=None,
        target_category=None,
        target_value=ABSENT,
        full=False,
    ):
        fn._ptera_role = Role(
            role=role,
            target=Element(
                capture=target,
                name=target_name,
                category=target_category,
                value=target_value,
            ),
            full=full,
        )
        fn._ptera_argspec = get_names(fn)
        return fn

    return wrap


initializer = role_wrapper("initializer")
updater = role_wrapper("updater")
valuer = role_wrapper("valuer")


class Storage:

    pattern = None
    default_target = None
    hasoutput = False

    def __init__(self):
        self._prepare()
        self.store = {}
        self.update_queue = {}

    def _init_wrap(self, fn):
        @functools.wraps(fn)
        def wrapped(**cap):
            role = fn._ptera_role
            cap = {**role.make_capture(), **cap}
            key = tuple(
                getattr(cap[k], field) for k, field in self._key_captures
            )
            if key not in self.store:
                self.store[key] = call_with_captures(fn, cap, full=role.full)
            return self.store[key]

        wrapped._ptera_argspec = fn._ptera_argspec
        return wrapped

    def _update_wrap(self, fn):
        @functools.wraps(fn)
        def wrapped(**cap):
            role = fn._ptera_role
            cap = {**role.make_capture(), **cap}
            key = tuple(
                getattr(cap[k], field) for k, field in self._key_captures
            )
            # TODO: It could be appropriate to catch a ValueError here
            assert key in self.store
            self.update_queue[key] = call_with_captures(fn, cap, full=role.full)

        wrapped._ptera_argspec = (
            set(fn._ptera_argspec) | set(k for k, _ in self._key_captures)
        )
        return wrapped

    def _value_wrap(self, fn):
        @functools.wraps(fn)
        def wrapped(**cap):
            role = fn._ptera_role
            cap = {**role.make_capture(), **cap}
            return call_with_captures(fn, cap, full=role.full)

        wrapped._ptera_argspec = (
            set(fn._ptera_argspec) | set(k for k, _ in self._key_captures)
        )
        return wrapped

    def _prepare(self):
        assert self.pattern
        pattern = to_pattern(self.pattern)
        self._rules = defaultdict(lambda: defaultdict(list))

        self._key_captures = pattern.key_captures()

        for method in dir(self):
            fn = getattr(self, method)
            role = getattr(fn, "_ptera_role", None)
            if role is None:
                continue

            if role.target.capture is None:
                role.target = role.target.clone(capture=self.default_target)

            names = get_names(fn)
            names = {*names, *[k for k, param in self._key_captures]}
            patt = pattern.rewrite(names, focus=role.target.capture)
            patt = patt.specialize({role.target.capture: role.target})

            if role.role == "valuer":
                self._rules[patt]["value"].append(self._value_wrap(fn))

            elif role.role == "initializer":
                self._rules[patt]["value"].append(self._init_wrap(fn))

            elif role.role == "updater":
                self._rules[patt]["listeners"].append(self._update_wrap(fn))

            else:  # pragma: no cover
                raise AssertionError(f"Unknown role: {role.role}")

    def instantiate(self):
        return self

    def rules(self):
        return self._rules

    def finalize(self):
        for key, value in self.update_queue.items():
            self.store[key] = value
        return self
