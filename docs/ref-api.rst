
Main API
========

This page collates the main API functions. Other reference files contain further details that may or may not be relevant to typical users.

.. contents::
    :local:


Probing API
-----------

The preferred API to Ptera's functionality. It is the most powerful.

.. list-table::

   * - :func:`~ptera.probe.probing`
     - Context manager for probing
   * - :func:`~ptera.probe.global_probe`
     - Create a global probe
   * - :class:`~ptera.probe.Probe`
     - Probe class returned by probing and global_probe


Overlay API
-----------

The Overlay API is more low level than the probing API (the latter uses the former under the hood).

.. list-table::

   * - :func:`~ptera.overlay.tooled`
     - Transform a function to report variable changes
   * - :func:`~ptera.overlay.is_tooled`
     - Return whether a function is tooled or not
   * - :func:`~ptera.overlay.autotool`
     - Automatically tool inplace all functions a selector refers to
   * - :class:`~ptera.overlay.BaseOverlay`
     - Simple context manager to apply handlers corresponding to selectors
   * - :class:`~ptera.overlay.Overlay`
     - A BaseOverlay with extra functionality
   * - :class:`~ptera.interpret.Immediate`
     - A handler that triggers when the focus variable is set
   * - :class:`~ptera.interpret.Total`
     - A handler that triggers when the outer scope for a selector ends


Low level API
-------------

.. list-table::

   * - :func:`~ptera.selector.select`
     - Parse a string into a :class:`~ptera.selector.Selector` object
   * - :func:`~ptera.transform.transform`
     - Transform a function to filter its behavior
