
Guide
=====

.. contents:: Contents
   :depth: 2
   :local:

Probing
-------

.. _probe-retval:

Probe the return value
~~~~~~~~~~~~~~~~~~~~~~

To probe the return value of ``f``, use the selector ``f() as result`` (you can name the result however you like):

.. code-block:: python

    def f(x, y):
        return x + y

    with probing("f() as result").values() as values:
        f(2, 5)

    assert values == [{"result": 7}]


Probe multiple variables
~~~~~~~~~~~~~~~~~~~~~~~~

Ptera is not limited to probing a single variable in a function: it can probe several at the same time (this is different from passing more than one selector to ``probing``).

When probing multiple variables at the same time, it is important to understand the concept of **focus variable**. The **focus variable**, if present, is the variable that triggers the events in the pipeline when it is assigned to (note that parameters are considered to be "assigned to" at the beginning of the function):

1. ``probing("f(x) > y")``: The focus is ``y``, this triggers when ``y`` is set. (Probe type: :class:`~ptera.interpret.Immediate`)
2. ``probing("f(y) > x")``: The focus is ``x``, this triggers when ``x`` is set. (Probe type: :class:`~ptera.interpret.Immediate`)
3. ``probing("f(x, y)")``: There is no focus, this triggers when ``f`` returns. (Probe type: :class:`~ptera.interpret.Total` -- these may be a bit less intuitive, see the section on  :ref:`total-probes` but don't feel like you have to use them)

To wit:

.. code-block:: python

    def f():
        x = 1
        y = 2
        x = 3
        y = 4
        x = 5
        return x

    # Case 1: focus on y
    with probing("f(x) > y").values() as values:
        f()

    assert values == [
        {"x": 1, "y": 2},
        {"x": 3, "y": 4},
    ]

    # Case 2: focus on x
    with probing("f(y) > x").values() as values:
        f()

    assert values == [
        {"x": 1},  # y is not set yet, so it is not in this entry
        {"x": 3, "y": 2},
        {"x": 5, "y": 4},
    ]

    # Case 3: no focus
    # See the section on total probes
    with probing("f(x, y)", raw=True).values() as values:
        f()

    assert values[0]["x"].values == [1, 3, 5]
    assert values[0]["y"].values == [2, 4]


.. note::
    The selector syntax does not necessarily mirror the syntax of actual function calls. For example, ``f(x)`` does not necessarily refer to a *parameter* of ``f`` called ``x``. As shown above, you can put any local variable between the parentheses. You can also probe global/closure variables that are used in the body of ``f``.

.. note::
    The selector ``f(x, !y)`` is an alternative syntax for ``f(x) > y``. The exclamation mark denotes the focus variable. There can only be one in a selector.


Probe across scopes
~~~~~~~~~~~~~~~~~~~

Sometimes you would like to get some context about whatever you are probing, and the context might not be in the same scope: it might be, for example, in the caller. Thankfully, Ptera has you covered.

.. code-block:: python

    def outer(n):
        x = 0
        for i in range(n):
            x += inner(i)
        return x

    def inner(x):
        a = x * x
        return a + 1

    with probing("outer(n) > inner > a").values() as values:
        outer(3)

    assert values == [
        {"n": 3, "a": 0},
        {"n": 3, "a": 1},
        {"n": 3, "a": 4},
    ]

As you can see, this probe gives us the context of what the value of ``n`` is in the outer scope, and that context is attached to every entry.

.. note::
    The selector ``outer > inner > a`` does not require ``inner`` to be called *directly* within ``outer``. The call can be indirect, for example if ``outer`` calls ``middle``, and ``middle`` calls ``inner``, the selector will still match. This makes it even more practical, since you can easily capture context quite removed from the focus variable.


Probe sibling calls
~~~~~~~~~~~~~~~~~~~

Now we're getting into power features that are a bit more niche, but Ptera goes even beyond probing across caller/callee scopes: it can also attach results from sibling calls!

.. code-block:: python

    def main(x):
        return negmul(side(3), side(6))

    def side(x):
        return x + 1

    def negmul(x, y):
        a = x * y
        return -a

    with probing("main(x, side(x as x2), negmul(!a))", raw=True).values() as values:
        main(12)

    assert values == [
        {"x": 12, "x2": 6, "a": 28}
    ]

Here we use the ``!`` notation to indicate the focus variable, but it is not fundamentally different from doing ``... > negmul > a``. The probe above gives us, all at once:

* The value of ``x`` in the main function.
* The latest value of ``x`` in ``side`` (under a different name, to avoid clashing)
* The value of the local variable ``a`` in ``negmul``

.. _total-probes:

Total probes
~~~~~~~~~~~~

A probe that does not have a focus variable is a "total" probe. Total probes function differently:

* Instead of triggering when a specific focus variable is set, they trigger when the outermost function in the selector ends.
* Instead of providing the latest values of all the variables, they collect *all* the values the variables have taken (hence the name "total").
* Since the default interface of ``probing`` assumes there is only one value for each variable in each entry, total probes will fail if multiple values are captured for the same variable in the same entry, unless you pass ``raw=True`` to ``probing``. This will cause :class:`~ptera.interpret.Capture` instances to be provided instead.

For example, if we remove the focus from the previous example (and add ``raw=True``):

.. code-block:: python

    def main(x):
        return negmul(side(3), side(6))

    def side(x):
        return x + 1

    def negmul(x, y):
        a = x * y
        return -a

    with probing("main(x, side(x as x2), negmul(a))", raw=True).values() as values:
        main(12)

    assert values[0]["x"].values == [12]
    assert values[0]["x2"].values == [3, 6]
    assert values[0]["a"].values == [28]

In this example, each call to ``main`` will produce exactly one event, because ``main`` is the outermost call in the selector. You can observe that ``x2`` is associated to two values, because ``side`` was called twice.

.. note::
    You can in fact create a total probe that has a focus with ``probing(selector, probe_type="total")``. In this case, it will essentially duplicate the data for the outer scopes for each value of the focus variable.

Global probes
~~~~~~~~~~~~~

The :func:`~ptera.probe.global_probe` function can be used to set up a probe that remains active for the rest of the program. Unlike ``probing`` it is not a context manager.

.. code-block:: python

    def f(x):
        a = x * x
        return a

    gprb = global_probe("f > a")
    gprb.print()

    f(4)  # prints 16
    f(5)  # prints 25

    gprb.deactivate()

    f(6)  # prints nothing

.. note::
    Probes can only be activated once, so after calling deactivate you will need to make a new probe if you want to reactivate it.

.. note::
    Reduction operators such as :func:`~giving.operators.min` or :func:`~giving.operators.sum` are finalized when the probe exits. With ``probing``, that happens at the end of the ``with`` block. With ``global_probe``, that happens either when ``deactivate`` is called or when the program exits.


Operations
----------

In all of the previous examples, I have used the ``.values()`` method to gather all the results into a list. This is a perfectly fine way to use Ptera and it has the upside of being simple and easy to understand. There are however many other ways to interact with the streams produced by ``probing``.


Printing
~~~~~~~~

Use ``.print(<format>)`` or ``.display()`` to print each element of the stream on its own line.

.. code-block:: python

    def f(x):
        y = 0
        for i in range(1, x + 1):
            y = y + x
        return y

    with probing("f > y").print("y = {y}"):
        f(3)

    # Prints:
    # y = 0
    # y = 1
    # y = 3
    # y = 6

If ``print`` is given no arguments it will use plain ``str()`` to convert the elements to strings. ``display()`` displays dictionaries a bit more nicely.

Subscribe
~~~~~~~~~

You can, of course, subscribe arbitrary functions to a probe's stream. You can do so with:

1. The ``>>`` operator
2. The ``subscribe`` method (passes the dictionary as a positional argument)
3. The ``ksubscribe`` method (passes the dictionary as keyword arguments)

For example:

.. code-block:: python

    def f(x):
        y = 0
        for i in range(1, x + 1):
            y = y + x
        return y

    with probing("f > y") as prb:
        # 1. The >> operator
        prb >> print

        # 2. The subscribe method
        @prb.subscribe
        def _(data):
            print("subscribe", data)

        # 3. The ksubscribe method
        @prb.ksubscribe
        def _(y):
            print("ksubscribe", y)

        f(3)

    # Prints:
    # {"y": 0}
    # subscribe {"y": 0}
    # ksubscribe 0
    # ...


Map, filter, reduce
~~~~~~~~~~~~~~~~~~~

Let's say you have a sequence and you want to print out the maximum absolute value. You can do it like this:

.. code-block:: python

    def f():
        y = 1
        y = -7
        y = 3
        y = 6
        y = -2

    with probing("f > y") as prb:
        maximum = prb["y"].map(abs).max()
        maximum.print("The maximum is {}")

        f()

    # Prints: The maximum is 7

* The ``[...]`` notation indexes each element in the stream (you can use it multiple times to get deep into the structure, if you're probing lists or dictionaries. There is also a ``.getattr()`` operator if you want to get deep into arbitrary objects)
* ``map`` maps a function to each element, here the absolute value
* ``min`` reduces the stream using the minimum function

.. note::
    ``map`` is different from ``subscribe``. The pipelines are lazy, so ``map`` might not execute if there is no subscriber down the pipeline.

If the stream interface is getting in your way and you would rather get the maximum value as an integer that you can manipulate normally, you have two (pretty much equivalent) options:

.. code-block:: python

    # With values()
    with probing("f > y")["y"].map(abs).max().values() as values:
        f()

    assert values == [7]

    # With accum()
    with probing("f > y") as prb:
        maximum = prb["y"].map(abs).max()
        values = maximum.accum()

        f()

    assert values == [7]

That same advice goes for pretty much all the other operators.

Overriding values
~~~~~~~~~~~~~~~~~

Ptera's probes are able to override the values of the variables being probed (unless the probe is total; nonlocal variables are also not overridable). For example:

.. code-block:: python

    def f(x):
        hidden = 1
        return x + hidden

    assert f(10) == 11

    with probing("f > hidden") as prb:
        prb.override(2)

        assert f(10) == 12

The argument to :meth:`~ptera.probe.Probe.override` can also be a function that takes the current value of the stream. Also see :meth:`~ptera.probe.Probe.koverride`.

.. warning::

    ``override()`` only overrides the **focus variable**. Recall that the focus variable is the one to the right of ``>``, or the one prefixed with ``!``.

    This is because a Ptera selector is triggered when the focus variable is set, so realistically it is the only one that it makes sense to override.

    Be careful, because it is easy to write misleading code:

    .. code-block:: python

        # THIS WILL SET y = x + 1, NOT x
        Probe("f(x) > y")["x"].override(lambda x: x + 1)

.. note::
    ``override`` will only work at the end of a synchronous pipe (map/filter are OK, but not e.g. sample)

If the focus variable is the return value of a function (as explained in :ref:`probe-retval`), ``override`` will indeed override that return value.

Asserts
~~~~~~~

The ``fail()`` method can be used to raise an exception. If you put it after a ``filter``, you can effectively fail when certain conditions occur. This can be a way to beef up a test suite.

.. code-block:: python

    def median(xs):
        # Don't copy this because it's incorrect if the length is even
        return xs[len(xs) // 2]

    with probing("median > xs") as prb:
        prb.kfilter(lambda xs: len(xs) == 0).fail("List is empty!")
        prb.kfilter(lambda xs: list(sorted(xs)) != xs).fail("List is not sorted!")

        median([])               # Fails immediately
        median([1, 2, 5, 3, 4])  # Also fails

Note the use of the :func:`~giving.operator.kfilter` operator, which receives the data as keyword arguments. Whenever it returns False, the corresponding datum is omitted from the stream. An alternative to using ``kfilter`` here would be to simply write ``prb["xs"].filter(...)``.

Conditional breakpoints
~~~~~~~~~~~~~~~~~~~~~~~

Interestingly, you can use probes to set conditional breakpoints. Modifying the previous example:

.. code-block:: python

    def median(xs):
        return xs[len(xs) // 2]

    with probing("median > xs") as prb:
        prb.kfilter(lambda xs: list(sorted(xs)) != xs).breakpoint()

        median([1, 2, 5, 3, 4])  # Enters breakpoint
        median([1, 2, 3, 4])     # Does not enter breakpoint

Using this code, you can set a breakpoint in ``median`` that is triggered only if the input list is not sorted. The breakpoint will occur wherever in the function the focus variable is set, in this case the beginning of the function since the focus variable is a parameter.


Miscellaneous
-------------

Meta-variables
~~~~~~~~~~~~~~

There are a few meta-variables recognized by Ptera that start with a hash sign:

* ``#enter`` is triggered immediately when entering a function. For example, if you want to set a breakpoint at the start of a function with no arguments you can use ``probing("f > #enter").breakpoint()``.
* ``#value`` stands in for the return value of a function. ``f() as x`` is sugar for ``f > #value as x``.
* ``#yield`` is triggered whenever a generator yields.

Generic variables
~~~~~~~~~~~~~~~~~

It is possible to indiscriminately capture all variables from a function, or all variables that have a certain "tag". Simply prefix a variable with ``$`` to indicate it is generic. When doing so, you will need to set ``raw=True`` if you want to be able to access the variable names. For example:

.. code-block:: python

    def f(a):
        b = a + 1
        c = b + 1
        d = c + 1
        return d

    with probing("f > $x", raw=True) as prb:
        prb.print("{x.name} is {x.value}").

        f(10)

    # Prints:
    # a is 10
    # b is 11
    # c is 12
    # d is 13

.. note::
    ``$x`` will also pick up global and nonlocal variables, so if for example you use the ``sum`` builtin in the function, you will get an entry for ``sum`` in the stream. It will not pick up meta-variables such as ``#value``, however.

Selecting based on tags
~~~~~~~~~~~~~~~~~~~~~~~

This feature admittedly clashes with type annotations, but Ptera recognizes a specific kind of annotation on variables:

.. code-block:: python

    def f(a):
        b = a + sum([1])
        c: "@Cool" = b + 1
        d: "@Cool & @Hot" = c + 1
        return d

    with probing("f > $x:@Cool", raw=True) as prb:
        prb.print("{x.name} is {x.value}")

        f(10)

    # Prints:
    # c is 12
    # d is 13

In the above code, only variables tagged as ``@Cool`` will be instrumented. Multiple tags can be combined using the ``&`` operator.

Probe methods
~~~~~~~~~~~~~

Probing methods works as one would expect. When using a selector such as ``self.f > x``, it will be interpreted as ``cls.f(self = <self>) > x`` so that it only triggers when it is called on this particular ``self``.


Absolute references
~~~~~~~~~~~~~~~~~~~

Ptera inspects the locals and globals of the frame in which ``probing`` is called in order to figure out what to instrument. In addition to this system, there is a second system whereas each function corresponds to a unique reference. These references always start with ``/``:

.. code-block:: python

    global_probe("/xyz.submodule/Klass/method > x")

    # is essentially equivalent to:

    from xyz.submodule import Klass
    global_probe("Klass.method > x")

The slashes represent a physical nesting rather than object attributes. For example, ``/module.submodule/x/y`` means:

* Go in the file that defines ``module.submodule``
* Enter ``def x`` or ``class x`` (it will *not* work if ``x`` is imported from elsewhere)
* Within that definition, enter ``def y`` or ``class y``

The helper function :func:`~ptera.utils.refstring` can be used to get the absolute reference for a function.

.. note::
    * Unlike the normal notation, the absolute notation bypasses decorators. ``/module/function`` will probe the function inside the ``def function(): ...`` in ``module.py``, so it will work even if the function was wrapped by a decorator (unless the decorator does not actually call the function).
    * Use ``/module.submodule/func``, *not* ``/module/submodule/func``. The former roughly corresponds to ``from module.submodule import func`` and the latter to ``from module import submodule; func = submodule.func``, which can be different in Python. It's a bit odd, but it works that way to properly address Python quirks.
