
What is Ptera?
==============

Ptera is a way to instrument code from the outside. More precisely, it allows you to specify a set of variables to watch in an arbitrary Python call graph and manipulate a stream of their values.

For example, the following code will print out the minimum value taken by the variable `loss` across all calls of function `step`, but only when it called in the function `train`:

.. code-block:: python

    with probing("train > step > loss") as prb:
        # The following line declares a processing pipeline. It must be done
        # before the main function is called.
        prb["loss"].min().print()
        main()

You can use Ptera to:

* Instrument code that you do not control.
* Collect data across function scopes.
* Perform complex reductions on the stream of values.


Getting started
===============

Install
-------

.. code-block:: bash

    pip install ptera

Usage
-----

The main API for Ptera is :func:`~ptera.probe.probing`, which is used as a context manager.

Here's an example involving `a fun function <https://en.wikipedia.org/wiki/Collatz_conjecture>`_. Even though the function returns nothing, we can use Ptera to extract all sorts of interesting things:

.. code-block:: python

    from ptera import probing

    def collatz(n):
        while n != 1:
            n = (3 * n + 1) if n % 2 else (n // 2)

    # `collatz > n` means: probe variable `n` in function `collatz`
    # Every time `n` is set (including when it is given as a parameter)
    # an event it sent through `prb`
    with probing("collatz > n") as prb:
        # Declare one or more pipelines on the data.
        prb["n"].print("n = {}")
        prb["n"].max().print("max(n) = {}")
        prb["n"].count().print("number of steps: {}")

        # We can also ask for all values to be accumulated into a list
        values = prb["n"].accum()

        # Call the function once all the pipelines are set up.
        collatz(2021)

        # Print the values
        print(values)

    # Output:
    # n = 2021
    # ...
    # n = 1
    # [2021, ..., 1]
    # max(n) = 6064
    # number of steps: 63

Note that in the example above the max/count are printed after the with block ends (that's how they know there is no more data), which is why ``print(values)`` happens before.
