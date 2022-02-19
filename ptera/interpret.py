"""Create events on transformed functions based on selectors.

There are some extra features here compared to the standard Probe interface,
namely :class:`~ptera.interpret.Total` which can accumulate
multiple values for non-focus variables and is only triggered when the
selector's outer function finishes.
"""

from collections import defaultdict

from .selector import Element, check_element, select
from .transform import PteraNameError
from .utils import ABSENT


class OverrideException(Exception):
    """Exception raised when trying to override a closure variable."""


class Interactor:
    """Represents an interactor for a tooled function.

    Define an ``interact`` method called by the tooled function
    when variables are changed.
    """

    def __init__(self, fn, accumulators=None):
        self.fn = fn
        self.accumulators = accumulators or defaultdict(list)
        self.to_close = []

    def register(self, acc, captures, close_at_exit):
        """Register an accumulator for a certain set of captures.

        Arguments:
            acc: An Accumulator.
            captures: A dictionary of elements to sets of matching
                variable names for which the accumulator will be
                triggered.
            close_at_exit: Whether to call the accumulator's close
                function when the interactor exits.
        """
        for element, varnames in captures.items():
            for v in varnames:
                self.accumulators[v].append((element, acc))
        if close_at_exit and acc.close:
            self.to_close.append(acc)

    def work_on(self, varname, key, category):
        """Return a :class:`WorkingFrame` for the given variable.

        Arguments:
            varname: The name of the variable.
            key: The key (attribute or index) that is being set on the
                variable.
            category: The variable's category or tag.
        """
        return WorkingFrame(varname, key, category, self.accumulators)

    def interact(self, varname, key, category, value, overridable):
        """Interaction function called when setting a variable in a tooled function.

        Arguments:
            varname: The variable's name.
            key: The attribute or index set on the variable (as a Key object)
            category: The variable's category or tag (annotation)
            value: The value given to the variable in the original code.
            overridable: Whether the value can be overriden.

        Returns:
            The value to actually set the variable to.
        """
        if key is not None:
            varname = key.affix_to(varname)

        with self.work_on(varname, key, category) as wfr:

            fr_value = wfr.intercept(value)
            if fr_value is not ABSENT:
                if not overridable:
                    raise OverrideException(
                        f"The value of '{varname}' cannot be overriden"
                    )
                value = fr_value

            if value is ABSENT:
                raise PteraNameError(varname, self.fn)

            wfr.log(value)
            wfr.trigger()

        return value

    def exit(self):
        """Exit the interactor.

        This triggers the close function on available accumulators.
        """
        for acc in self.to_close:
            acc.close()


class WorkingFrame:
    """Context manager to facilitate working on a variable."""

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
        """Execute the intercepts of all matching accumulators.

        The last intercept that does not return ABSENT wins.

        Arguments:
            tentative: The tentative value for the variable, as
                provided in the original code.

        Returns:
            The value the intercepted variable should take.
        """
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
        """Log a value for the variable."""
        for element, acc in self.accumulators:
            acc.log(element, self.varname, self.category, value)

    def trigger(self):
        """Trigger an event using what was accumulated."""
        for element, acc in self.accumulators:
            if element.focus and acc.trigger:
                acc.trigger()


class Capture:
    """Represents captured values for a variable.

    Arguments:
        element: The selector element for which we are capturing.

    Attributes:
        element: The selector element for which we are capturing.
        capture: The variable name or alias corresponding to the
            capture (same as element.capture).
        names: The list of names of the variables that match the
            element.
        values: The list of values taken by matching variables.
    """

    def __init__(self, element):
        self.element = element
        self.capture = element.capture
        self.names = []
        self.values = []

    @property
    def name(self):
        """Name of the capture.

        For a generic element such as ``$x``, there may be multiple
        names, in which case the ``.names`` attribute should be used
        instead.
        """
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
        """Value of the capture.

        This only works if there is a unique value. Otherwise, you must
        use ``.values``.
        """
        if len(self.values) == 1:
            return self.values[0]
        elif len(self.values) == 0:
            raise ValueError(f"No value for capture `{self.capture}`")
        else:
            raise ValueError(
                f"Multiple values stored for capture `{self.capture}`"
            )

    def accum(self, varname, value):
        """Accumulate a variable name and value."""
        assert varname is not None
        self.names.append(varname)
        self.values.append(value)

    def set(self, varname, value):
        """Set a variable name and value, overwriting the previous capture."""
        assert varname is not None
        self.names = [varname]
        self.values = [value]

    def snapshot(self):
        """Return a snapshot of the capture at this moment."""
        cap = Capture(self.element)
        cap.names = list(self.names)
        cap.values = list(self.values)
        return cap

    def __str__(self):
        return f"Capture({self.element}, {self.names}, {self.values})"

    __repr__ = __str__


class BaseAccumulator:
    """Accumulates the values of variables in Capture objects.

    Under certain conditions, call user-provided event functions.

    Any function given to the constructor must take one argument which is the
    dictionary of captures.

    Arguments:
        selector: The selector to use.
        trigger: The function to call when the focus variable is set.
        intercept: The function to call to override the value of the
            focus variable.
        close: The function to call when the selector is closed.
        parent: The parent Accumulator.
        template: Whether the Accumulator is a "template" and should be
            cloned prior to accumulating anything.
        check: Whether to filter that the values are correct in a selector
            such as ``f(x=1) > y``. Otherwise the ``=1`` would be ignored.
    """

    def __init__(
        self,
        *,
        selector,
        intercept=None,
        trigger=None,
        close=None,
        parent=None,
        template=True,
        check=True,
    ):
        selector = select(selector)

        self.selector = selector
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
            if self.selector.check_captures(results):
                return fn(results)
            else:
                return ABSENT

        return new_fn if (fn and check and self.selector.hasval) else fn

    def fork(self, selector=None):
        """Fork the Accumulator, possibly with a new selector.

        Children Accumulators can accumulate new data while sharing what is
        accumulated by their parents.
        """
        parent = None if self.template else self
        return type(self)(
            selector=selector or self.selector,
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
        """Get the Capture object for a leaf element."""
        if element.capture not in self.captures:
            cap = Capture(element)
            self.captures[element.capture] = cap
            return cap
        else:
            return self.captures[element.capture]

    def build(self):
        """Build the dictionary of captures.

        The built dictionary includes captures from the parents.
        """
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
    """Accumulator usually triggered when the selector's outer function ends.

    The Total accumulator keeps all values taken by the variables in the
    selector for each value taken by the focus variable. For example, if the
    selector is ``f(x) > g(!y) > h(z)`` and h is called multiple times for
    multiple values of ``z``, they will all be accumulated together. However, if
    ``y`` is set multiple times, there will be multiple events.

    Any function given to the constructor must take one argument which is the
    dictionary of captures.

    Arguments:
        selector: The selector to use.
        close: The function to call when the selector is closed.
        trigger: The function to call when the focus variable is set.
        intercept: The function to call to override the value of the
            focus variable.
    """

    def __init__(self, selector, close, trigger=None, **kwargs):
        super().__init__(
            selector=selector, trigger=trigger, close=close, **kwargs
        )
        if self.parent is None:
            self.names = self.selector.all_captures
        else:
            self.names = self.parent.names
            self.parent.children.append(self)

    def accumulator_for(self, element):
        # If the element is the focus of the selector or contains the
        # focus, we fork the accumulator. Any values set downstream
        # will be accumulated in that child, but all children will
        # share what gets accumulated in this accumulator. This stands
        # in contrast with Immediate, which does not need to fork
        # because it gets activated immediately with all the current
        # captures.
        return self.fork(selector=element) if element.focus else self

    def log(self, element, varname, category, value):
        cap = self.getcap(element)
        cap.accum(varname, value)
        return self

    def leaves(self):
        if isinstance(self.selector, Element):
            return [self]
        else:
            rval = []
            for child in self.children:
                rval += child.leaves()
            return rval

    def close(self):
        # We only close the accumulator from the root
        if self.parent is None:
            leaves = self.leaves()
            for leaf in leaves or [self]:
                # Each leaf is a separate set of captures
                args = leaf.build()
                if set(args) == leaf.names:
                    # We only call the function if all of the names that should
                    # have been captured are there. Otherwise we may get some
                    # incomplete captures where e.g. the focus variable is
                    # missing because it was never set in that leaf.
                    leaf._close(args)


class Immediate(BaseAccumulator):
    """Accumulator triggered when the focus variable is set.

    The Immediate accumulator only keeps the last value of each variable in the
    selector.

    Any function given to the constructor must take one argument which is the
    dictionary of captures.

    Arguments:
        selector: The selector to use.
        trigger: The function to call when the focus variable is set.
        intercept: The function to call to override the value of the
            focus variable.
        close: The function to call when the selector is closed.
    """

    def __init__(self, selector, trigger=None, **kwargs):
        super().__init__(selector=selector, trigger=trigger, **kwargs)

    def log(self, element, varname, category, value):
        cap = self.getcap(element)
        cap.set(varname, value)
        return self
