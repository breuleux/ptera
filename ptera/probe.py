import atexit
import inspect

from giving import SourceProxy

from .core import BaseOverlay
from .deco import tooled
from .selector import select
from .selfless import override
from .tags import tag
from .utils import ABSENT

global_probes = set()


def make_resolver(*namespaces):
    """Hook into ptera's selector resolution.

    * Resolve functions in the given namespaces. They will automatically be
      instrumented with ptera.tooled.
    * When a tag is found, instrument all functions in all modules that might
      use that tag (with ptera.tooled).

    Arguments:
        namespaces: A list of globals dicts to find functions and variables in.
    """

    def __ptera_resolver__(x):
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
            (curr,) = funcs

        else:
            varname, *rest = x.split(".")

            if varname.startswith("@"):
                curr = getattr(tag, varname[1:])

            else:
                for ns in namespaces:
                    if varname in ns:
                        curr = ns[varname]
                        seq = [(ns, varname, dict.__setitem__)]
                        break
                else:
                    raise NameError(f"Could not resolve '{varname}'.")

                for part in rest:
                    seq.append((curr, part, setattr))
                    curr = getattr(curr, part)

        if inspect.isfunction(curr):
            # Instrument the function directly
            tooled.inplace(curr)

        # elif isinstance(curr, Tag):
        #     # Instrument existing modules and update substitutions
        #     instrument_for_tag(curr)

        return getattr(curr, "__ptera__", curr)

    return __ptera_resolver__


class Probe(SourceProxy):
    """Observable which generates a stream of values from program variables.

    Probes should be created with `ptera.probing` or `ptera.global_probe`.

    Arguments:
        selectors: The selector strings describing the variables to probe (at least one).
        raw: Defaults to False. If True, produce a stream of Capture objects that
            contain extra information about the capture. Mostly relevant for
            advanced selectors such as "f > $x:@Parameter" which captures the value
            of any variable with the Parameter tag under the generic name "x".
            When raw is True, the actual name of the variable is preserved in a
            Capture object associated to x.
    """

    def __init__(self, *selectors, raw=False, _obs=None, _root=None):
        if not selectors and _obs is None:
            raise TypeError("Probe() takes at least one selector argument.")

        super().__init__(_obs=_obs, _root=_root)

        if selectors:
            self._selectors = [
                select(selector, env_wrapper=make_resolver)
                for selector in selectors
            ]
            self._ol = BaseOverlay(
                {sel: {"value": self._emit} for sel in self._selectors}
            )
            self._raw = raw
            self._activated = False

    def override(self, setter=lambda x: x):
        """Override the value of the focus variable using a setter function.

        .. code-block:: python

            # Increment a whenever it is set (will not apply recursively)
            Probe("f > a")["a"].override(lambda value: value + 1)

        .. note::
            **Important:** override() only overrides the **focus variable**. The focus
            variable is the one to the right of ``>``, or the one prefixed with ``!``.

            This is because a Ptera selector is triggered when the focus variable is set,
            so realistically it is the only one that it makes sense to override.

            Be careful, because it is easy to write misleading code:

            .. code-block:: python

                # THIS WILL SET y = x + 1, NOT x
                Probe("f(x) > y")["x"].override(lambda x: x + 1)

        .. note::
            ``override`` will only work at the end of a synchronous pipe (map/filter are OK,
            but not e.g. sample)

        Arguments:
            setter:

                A function that takes a value from the pipeline and produces the value
                to set the focus variable to.

                * If not set, the value from the stream is used as-is.
                * If not callable, set the variable to the value of setter
        """
        if not callable(setter):
            value = setter

            def setter(_):
                return value

        def _override(data):
            self._root._value = override(setter(data))

        return self.subscribe(_override)

    def koverride(self, setter):
        """Override the value of the focus variable using a setter function with kwargs.

        .. code-block:: python

            def f(x):
                ...
                y = 123
                ...

            # This will basically override y = 123 to become y = x + 123
            Probe("f(x) > y").koverride(lambda x, y: x + y)

        .. note::

            **Important:** override() only overrides the **focus variable**. The focus
            variable is the one to the right of ``>``, or the one prefixed with ``!``.

            See :meth:`~ptera.probe.ProbeProxy.override`.

        Arguments:
            setter: A function that takes a value from the pipeline as keyword arguments
                and produces the value to set the focus variable to.
        """

        def _override(data):
            self._root._value = override(setter(**data))

        return self.subscribe(_override)

    ######################
    # Changed from Given #
    ######################

    def breakpoint(self, *args, skip=[], **kwargs):  # pragma: no cover
        skip = ["ptera.*", *skip]
        return super().breakpoint(*args, skip=skip, **kwargs)

    def breakword(self, *args, skip=[], **kwargs):  # pragma: no cover
        skip = ["ptera.*", *skip]
        return super().breakword(*args, skip=skip, **kwargs)

    def fail(self, *args, skip=[], **kwargs):  # pragma: no cover
        skip = ["ptera.*", "rx.*", "giving.*", *skip]
        return super().fail(*args, skip=skip, **kwargs)

    ###################
    # Context manager #
    ###################

    def _emit(self, data):
        """Emit data on the stream.

        This is used internally.
        """
        self._value = ABSENT
        if not self._raw:
            data = {name: cap.value for name, cap in data.items()}
        self._push(data)
        # self._value is set by override(), but this will only work if the pipeline
        # is synchronous
        return self._value

    def _enter(self):
        if self._activated:
            raise Exception("An instance of Probe can only be entered once")

        self._activated = True
        global_probes.add(self)
        self._ol.__enter__()
        return self

    def _exit(self):
        self._ol.__exit__(None, None, None)
        global_probes.remove(self)


def probing(*selectors, raw=False):
    """Probe that can be used as a context manager.

    Example:

    >>> def f(x):
    ...     a = x * x
    ...     return a

    >>> with probing("f > a").print():
    ...     f(4)  # Prints {"a": 16}

    Arguments:
        selectors: The selector strings describing the variables to probe (at least one).
        do: A function to execute on each data point.
        format: A format string (implies do=print)
        raw: Defaults to False. If True, produce a stream of Capture objects that
            contain extra information about the capture.
    """
    return Probe(*selectors, raw=raw)


def global_probe(*selectors, raw=False):
    """Set a probe globally.

    Example:

    >>> def f(x):
    ...     a = x * x
    ...     return a

    >>> probe = global_probe("f > a")
    >>> probe["a"].print()
    >>> f(4)  # Prints 16

    Arguments:
        selectors: The selector strings describing the variables to probe (at least one).
        raw: Defaults to False. If True, produce a stream of Capture objects that
            contain extra information about the capture.
    """
    prb = Probe(*selectors, raw=raw)
    prb.__enter__()
    return prb


@atexit.register
def _terminate_global_probes():  # pragma: no cover
    for probe in list(global_probes):
        probe.__exit__(None, None, None)
