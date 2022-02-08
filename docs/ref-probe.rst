
ptera.probe
===========

This module defines the probing functionality. The interface for probes is built on `giving <https://giving.readthedocs.io/en/latest/>`_

.. automodule:: ptera.probe

    .. autofunction:: probing

    .. autofunction:: global_probe

    .. autoclass:: Probe(*selectors, raw=False)

        .. automethod:: accum
        .. automethod:: breakpoint
        .. automethod:: breakword
        .. automethod:: display
        .. automethod:: eval
        .. automethod:: exec
        .. automethod:: fail
        .. automethod:: fail_if_empty
        .. automethod:: give
        .. automethod:: koverride
        .. automethod:: ksubscribe
        .. automethod:: override
        .. automethod:: pipe(*ops)
        .. *
        .. automethod:: print
        .. automethod:: subscribe(observer=None, on_next=None, on_error=None, on_completed=None)
        .. automethod:: values
        .. automethod:: wrap
        .. automethod:: __or__
        .. automethod:: __rshift__
        .. automethod:: __getitem__

        **Other methods:** All operators in the :ref:`operator list<OperatorList>` have a corresponding method on Probe.

        * :func:`~giving.operators.affix`
        * :func:`~giving.operators.all`
        * :func:`~giving.operators.amb`
        * :func:`~giving.operators.as_`
        * :func:`~giving.operators.as_observable`
        * :func:`~giving.operators.augment`
        * :func:`~giving.operators.average`
        * :func:`~giving.operators.average_and_variance`
        * :func:`~giving.operators.bottom`
        * :func:`~giving.operators.buffer`
        * :func:`~giving.operators.buffer_toggle`
        * :func:`~giving.operators.buffer_when`
        * :func:`~giving.operators.buffer_with_count`
        * :func:`~giving.operators.buffer_with_time`
        * :func:`~giving.operators.buffer_with_time_or_count`
        * :func:`~giving.operators.catch`
        * :func:`~giving.operators.collect_between`
        * :func:`~giving.operators.combine_latest`
        * :func:`~giving.operators.concat`
        * :func:`~giving.operators.contains`
        * :func:`~giving.operators.count`
        * :func:`~giving.operators.debounce`
        * :func:`~giving.operators.default_if_empty`
        * :func:`~giving.operators.delay`
        * :func:`~giving.operators.delay_subscription`
        * :func:`~giving.operators.delay_with_mapper`
        * :func:`~giving.operators.dematerialize`
        * :func:`~giving.operators.distinct`
        * :func:`~giving.operators.distinct_until_changed`
        * :func:`~giving.operators.do`
        * :func:`~giving.operators.do_action`
        * :func:`~giving.operators.do_while`
        * :func:`~giving.operators.element_at`
        * :func:`~giving.operators.element_at_or_default`
        * :func:`~giving.operators.exclusive`
        * :func:`~giving.operators.expand`
        * :func:`~giving.operators.filter`
        * :func:`~giving.operators.filter_indexed`
        * :func:`~giving.operators.finally_action`
        * :func:`~giving.operators.find`
        * :func:`~giving.operators.find_index`
        * :func:`~giving.operators.first`
        * :func:`~giving.operators.first_or_default`
        * :func:`~giving.operators.flat_map`
        * :func:`~giving.operators.flat_map_indexed`
        * :func:`~giving.operators.flat_map_latest`
        * :func:`~giving.operators.fork_join`
        * :func:`~giving.operators.format`
        * :func:`~giving.operators.getitem`
        * :func:`~giving.operators.group_by`
        * :func:`~giving.operators.group_by_until`
        * :func:`~giving.operators.group_join`
        * :func:`~giving.operators.group_wrap`
        * :func:`~giving.operators.ignore_elements`
        * :func:`~giving.operators.is_empty`
        * :func:`~giving.operators.join`
        * :func:`~giving.operators.keep`
        * :func:`~giving.operators.kfilter`
        * :func:`~giving.operators.kmap`
        * :func:`~giving.operators.kmerge`
        * :func:`~giving.operators.kscan`
        * :func:`~giving.operators.last`
        * :func:`~giving.operators.last_or_default`
        * :func:`~giving.operators.map`
        * :func:`~giving.operators.map_indexed`
        * :func:`~giving.operators.materialize`
        * :func:`~giving.operators.max`
        * :func:`~giving.operators.merge`
        * :func:`~giving.operators.merge_all`
        * :func:`~giving.operators.min`
        * :func:`~giving.operators.multicast`
        * :func:`~giving.operators.observe_on`
        * :func:`~giving.operators.on_error_resume_next`
        * :func:`~giving.operators.pairwise`
        * :func:`~giving.operators.partition`
        * :func:`~giving.operators.partition_indexed`
        * :func:`~giving.operators.pluck`
        * :func:`~giving.operators.pluck_attr`
        * :func:`~giving.operators.publish`
        * :func:`~giving.operators.publish_value`
        * :func:`~giving.operators.reduce`
        * :func:`~giving.operators.ref_count`
        * :func:`~giving.operators.repeat`
        * :func:`~giving.operators.replay`
        * :func:`~giving.operators.retry`
        * :func:`~giving.operators.roll`
        * :func:`~giving.operators.sample`
        * :func:`~giving.operators.scan`
        * :func:`~giving.operators.sequence_equal`
        * :func:`~giving.operators.share`
        * :func:`~giving.operators.single`
        * :func:`~giving.operators.single_or_default`
        * :func:`~giving.operators.single_or_default_async`
        * :func:`~giving.operators.skip`
        * :func:`~giving.operators.skip_last`
        * :func:`~giving.operators.skip_last_with_time`
        * :func:`~giving.operators.skip_until`
        * :func:`~giving.operators.skip_until_with_time`
        * :func:`~giving.operators.skip_while`
        * :func:`~giving.operators.skip_while_indexed`
        * :func:`~giving.operators.skip_with_time`
        * :func:`~giving.operators.slice`
        * :func:`~giving.operators.some`
        * :func:`~giving.operators.sole`
        * :func:`~giving.operators.starmap`
        * :func:`~giving.operators.starmap_indexed`
        * :func:`~giving.operators.start_with`
        * :func:`~giving.operators.subscribe_on`
        * :func:`~giving.operators.sum`
        * :func:`~giving.operators.switch_latest`
        * :func:`~giving.operators.tag`
        * :func:`~giving.operators.take`
        * :func:`~giving.operators.take_last`
        * :func:`~giving.operators.take_last_buffer`
        * :func:`~giving.operators.take_last_with_time`
        * :func:`~giving.operators.take_until`
        * :func:`~giving.operators.take_until_with_time`
        * :func:`~giving.operators.take_while`
        * :func:`~giving.operators.take_while_indexed`
        * :func:`~giving.operators.take_with_time`
        * :func:`~giving.operators.throttle`
        * :func:`~giving.operators.throttle_first`
        * :func:`~giving.operators.throttle_with_mapper`
        * :func:`~giving.operators.throttle_with_timeout`
        * :func:`~giving.operators.time_interval`
        * :func:`~giving.operators.timeout`
        * :func:`~giving.operators.timeout_with_mapper`
        * :func:`~giving.operators.timestamp`
        * :func:`~giving.operators.to_dict`
        * :func:`~giving.operators.to_future`
        * :func:`~giving.operators.to_iterable`
        * :func:`~giving.operators.to_list`
        * :func:`~giving.operators.to_marbles`
        * :func:`~giving.operators.to_set`
        * :func:`~giving.operators.top`
        * :func:`~giving.operators.variance`
        * :func:`~giving.operators.where`
        * :func:`~giving.operators.where_any`
        * :func:`~giving.operators.while_do`
        * :func:`~giving.operators.window`
        * :func:`~giving.operators.window_toggle`
        * :func:`~giving.operators.window_when`
        * :func:`~giving.operators.window_with_count`
        * :func:`~giving.operators.window_with_time`
        * :func:`~giving.operators.window_with_time_or_count`
        * :func:`~giving.operators.with_latest_from`
        * :func:`~giving.operators.zip`
        * :func:`~giving.operators.zip_with_iterable`
        * :func:`~giving.operators.zip_with_list`
