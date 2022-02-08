
.. _OperatorList:

List of operators
=================

The operators listed here are all available as methods on the :class:`~ptera.probe.Probe` objects yielded by :func:`~ptera.probe.probing`.

Ptera's operators come from the giving_ package, which itself derives most of its operators from the rx_ package.

.. _giving: https://giving.readthedocs.io
.. _rx: https://rxpy.readthedocs.io/en/latest/reference_operators.html

.. automodule:: giving.operators

    .. autofunction:: affix
    .. autofunction:: all
    .. autofunction:: amb
    .. autofunction:: as_
    .. autofunction:: as_observable
    .. autofunction:: augment
    .. autofunction:: average(*, scan=False)
    .. autofunction:: average_and_variance(*, scan=False)
    .. autofunction:: bottom
    .. autofunction:: buffer
    .. autofunction:: buffer_toggle
    .. autofunction:: buffer_when
    .. autofunction:: buffer_with_count
    .. autofunction:: buffer_with_time
    .. autofunction:: buffer_with_time_or_count
    .. autofunction:: catch
    .. autofunction:: collect_between
    .. autofunction:: combine_latest
    .. autofunction:: concat
    .. autofunction:: contains
    .. autofunction:: count(*, predicate=None, scan=False)
    .. autofunction:: debounce
    .. autofunction:: default_if_empty
    .. autofunction:: delay
    .. autofunction:: delay_subscription
    .. autofunction:: delay_with_mapper
    .. autofunction:: dematerialize
    .. autofunction:: distinct
    .. autofunction:: distinct_until_changed
    .. autofunction:: do
    .. autofunction:: do_action
    .. autofunction:: do_while
    .. autofunction:: element_at
    .. autofunction:: element_at_or_default
    .. autofunction:: exclusive
    .. autofunction:: expand
    .. autofunction:: filter
    .. autofunction:: filter_indexed
    .. autofunction:: finally_action
    .. autofunction:: find
    .. autofunction:: find_index
    .. autofunction:: first
    .. autofunction:: first_or_default
    .. autofunction:: flat_map
    .. autofunction:: flat_map_indexed
    .. autofunction:: flat_map_latest
    .. autofunction:: fork_join
    .. autofunction:: format
    .. autofunction:: getitem
    .. autofunction:: group_by
    .. autofunction:: group_by_until
    .. autofunction:: group_join
    .. autofunction:: group_wrap
    .. autofunction:: ignore_elements
    .. autofunction:: is_empty
    .. autofunction:: join
    .. autofunction:: keep
    .. autofunction:: kfilter
    .. autofunction:: kmap
    .. autofunction:: kmerge
    .. autofunction:: kscan
    .. autofunction:: last
    .. autofunction:: last_or_default
    .. autofunction:: map
    .. autofunction:: map_indexed
    .. autofunction:: materialize
    .. autofunction:: max(*, key=None, comparer=None, scan=False)
    .. autofunction:: merge
    .. autofunction:: merge_all
    .. autofunction:: min(*, key=None, comparer=None, scan=False)
    .. autofunction:: multicast
    .. autofunction:: observe_on
    .. autofunction:: on_error_resume_next
    .. autofunction:: pairwise
    .. autofunction:: partition
    .. autofunction:: partition_indexed
    .. autofunction:: pipe
    .. autofunction:: pluck
    .. autofunction:: pluck_attr
    .. autofunction:: publish
    .. autofunction:: publish_value
    .. autofunction:: reduce
    .. autofunction:: ref_count
    .. autofunction:: repeat
    .. autofunction:: replay
    .. autofunction:: retry
    .. autofunction:: roll
    .. autofunction:: sample
    .. autofunction:: scan
    .. autofunction:: sequence_equal
    .. autofunction:: share
    .. autofunction:: single
    .. autofunction:: single_or_default
    .. autofunction:: single_or_default_async
    .. autofunction:: skip
    .. autofunction:: skip_last
    .. autofunction:: skip_last_with_time
    .. autofunction:: skip_until
    .. autofunction:: skip_until_with_time
    .. autofunction:: skip_while
    .. autofunction:: skip_while_indexed
    .. autofunction:: skip_with_time
    .. autofunction:: slice
    .. autofunction:: sole
    .. autofunction:: some
    .. autofunction:: starmap
    .. autofunction:: starmap_indexed
    .. autofunction:: start_with
    .. autofunction:: subscribe_on
    .. autofunction:: sum(*, scan=False)
    .. autofunction:: switch_latest
    .. autofunction:: tag
    .. autofunction:: take
    .. autofunction:: take_last
    .. autofunction:: take_last_buffer
    .. autofunction:: take_last_with_time
    .. autofunction:: take_until
    .. autofunction:: take_until_with_time
    .. autofunction:: take_while
    .. autofunction:: take_while_indexed
    .. autofunction:: take_with_time
    .. function:: throttle(window_duration, scheduler=None)

        :func:`throttle` is an alias of :func:`throttle_first`
    .. autofunction:: throttle_first
    .. autofunction:: throttle_with_mapper
    .. autofunction:: throttle_with_timeout
    .. autofunction:: time_interval
    .. autofunction:: timeout
    .. autofunction:: timeout_with_mapper
    .. autofunction:: timestamp
    .. autofunction:: to_dict
    .. autofunction:: to_future
    .. autofunction:: to_iterable
    .. autofunction:: to_list
    .. autofunction:: to_marbles
    .. autofunction:: to_set
    .. autofunction:: top
    .. autofunction:: variance(*, scan=False)
    .. autofunction:: where
    .. autofunction:: where_any
    .. autofunction:: while_do
    .. autofunction:: window
    .. autofunction:: window_toggle
    .. autofunction:: window_when
    .. autofunction:: window_with_count
    .. autofunction:: window_with_time
    .. autofunction:: window_with_time_or_count
    .. autofunction:: with_latest_from
    .. autofunction:: zip
    .. autofunction:: zip_with_iterable
    .. autofunction:: zip_with_list
