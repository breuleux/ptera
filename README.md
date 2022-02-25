
# Ptera

Ptera is a powerful way to instrument your code for logging, debugging and testing purposes. With a simple call to `ptera.probing()`, you can:

* Obtain a stream of the values taken by any variable.
* Probe multiple variables from multiple functions in multiple scopes.
* Apply maps, filters, reductions, and much more to the streams.
* Override the values of variables based on complex conditions.
* Create external asserts or conditional breakpoints.
* Et cetera :)

**TODO** link to rtd documentation

## Example

You can use Ptera to observe assignments to any variable in your program:

```python
from ptera import probing

def f(x):
    y = 10
    for i in range(1, x + 1):
        y = y + i
    return y

with probing("f > y").values() as values:
    f(3)

# These are all the values taken by the y variable in f.
assert values == [
    {"y": 10},
    {"y": 11},
    {"y": 13},
    {"y": 16},
]
```

In the above,

1. We *select* the variable `y` of function `f` using the selector `f > y`.
2. We use the `values()` method to obtain a list in which the values of `y` will be progressively accumulated.
3. When `f` is called within the `probing` block, assignments to `y` are intercepted and appended to the list.
4. When the `probing` block finishes, the instrumentation is removed and `f` reverts to its normal behavior.

## Creating probes

* `ptera.probing`: Probe variables inside a `with` block.
* `ptera.global_probe`: Activate a global probe.

## Using probes

The interface for Ptera's probes is inspired from functional reactive programming and is identical to the interface of [giving](https://github.com/breuleux/giving) (itself based on `rx`). [See here for a complete list of operators.](https://giving.readthedocs.io/en/latest/ref-operators.html)

You can always use `with probing(...).values()` as in the example at the top if you want to keep it simple and just obtain a list of values. You can also use `with probing(...).display()` to print the values instead.

Beyond that, you can also define complex data processing pipelines. For example:

```python
with probing("f > x") as probe:
    probe["x"].map(abs).max().print()
    f(1234)
```

The above defines a pipeline that extracts the value of `x`, applies the `abs` function on every element, takes the maximum of these absolute values, and then prints it out. Keep in mind that this statement doesn't really *do* anything at the moment it is executed, it only *declares* a pipeline that will be activated whenever a probed variable is set afterwards. That is why `f` is called after and not before.

## More examples

Ptera is all about providing new ways to inspect what your programs are doing, so all examples will be based on this simple binary search function:

```python
from ptera import global_probe, probing

def f(arr, key):
    lo = -1
    hi = len(arr)
    while lo < hi - 1:
        mid = lo + (hi - lo) // 2
        if (elem := arr[mid]) > key:
            hi = mid
        else:
            lo = mid
    return lo + 1

##############################
# THE PROBING CODE GOES HERE #
##############################

f(list(range(1, 350, 7)), 136)
```

To get the output listed in the right column of the table below, the code in the left column should be inserted before the call to `f`, where the big comment is. Most of the methods on `global_probe` define the pipeline through which the probed values will be routed (the interface is inspired from functional reactive programming), so it is important to define them before the instrumented functions are called.

<table>
<tr>
<th>Code</th>
<th>Output</th>
</tr>

<!--
####################
####### ROW ########
####################
-->

<tr>
<td>

The `display` method provides a simple way to log values.

```python
global_probe("f > mid").display()
```

</td>
<td>

```
mid: 24
mid: 11
mid: 17
mid: 20
mid: 18
mid: 19
```

</td>
</tr>
<tr></tr>

<!--
####################
####### ROW ########
####################
-->

<tr>
<td>

The `print` method lets you specify a format string.

```python
global_probe("f(mid) > elem").print("arr[{mid}] == {elem}")
```

</td>
<td>

```
arr[24] == 169
arr[11] == 78
arr[17] == 120
arr[20] == 141
arr[18] == 127
arr[19] == 134
```

</td>
</tr>
<tr></tr>

<!--
####################
####### ROW ########
####################
-->

<tr>
<td>

Reductions are easy: extract the key and use `min`, `max`, etc.

```python
global_probe("f > lo")["lo"].max().print("max(lo) = {}")
global_probe("f > hi")["hi"].min().print("min(hi) = {}")
```

</td>
<td>

```
max(lo) = 19
min(hi) = 20
```

</td>
</tr>
<tr></tr>

<!--
####################
####### ROW ########
####################
-->

<tr>
<td>

Define assertions with `fail()` (for debugging, also try `.breakpoint()`!)

```python
def unordered(xs):
    return any(x > y for x, y in zip(xs[:-1], xs[1:]))

probe = global_probe("f > arr")["arr"]
probe.filter(unordered).fail("List is unordered: {}")

f([1, 6, 30, 7], 18)
```

</td>
<td>

```
Traceback (most recent call last):
  ...
  File "test.py", line 30, in <module>
    f([1, 6, 30, 7], 18)
  File "<string>", line 3, in f__ptera_redirect
  File "test.py", line 3, in f
    def f(arr, key):
giving.gvn.Failure: List is unordered: [1, 6, 30, 7]
```

</td>
</tr>
<tr></tr>

<!--
####################
####### ROW ########
####################
-->

<tr>
<td>

Accumulate into a list:

```python
results = global_probe("f > mid")["mid"].accum()
f(list(range(1, 350, 7)), 136)
print(results)
```

OR

```python
with probing("f > mid")["mid"].values() as results:
    f(list(range(1, 350, 7)), 136)

print(results)
```

</td>
<td>

```
[24, 11, 17, 20, 18, 19]
```

</td>
</tr>
<tr></tr>

</table>


## probing

Usage: `with ptera.probing(selector) as probe: ...`

The **selector** is a specification of which variables in which functions we want to stream through the probe. One of the variables must be the **focus** of the selector, meaning that the probe is triggered when *that* variable is set. The focus may be indicated either as `f(!x)` or `f > x` (the focus is `x` in both cases).

The **probe** is a wrapper around [rx.Observable](https://github.com/ReactiveX/RxPY) and supports a large number of [operators](https://giving.readthedocs.io/en/latest/ref-operators.html) such as `map`, `filter`, `min`, `average`, `throttle`, etc. (the interface is the same as in [giving](https://github.com/breuleux/giving)).


### Example 1: intermediate variables

Ptera is capable of capturing any variable in a function, not just inputs and return values:

```python
def fact(n):
    curr = 1
    for i in range(n):
        curr = curr * (i + 1)
    return curr

with probing("fact(i, !curr)").print():
    fact(3)
    # {'curr': 1}
    # {'curr': 1, 'i': 0}
    # {'curr': 2, 'i': 1}
    # {'curr': 6, 'i': 2}
```

The "!" in the selector above means that the focus is `curr`. This means it is triggered when `curr` is set. This is why the first result does not have a value for `i`. You can use the selector `fact(!i, curr)` to focus on `i` instead:

```python
with probing("fact(!i, curr)").print():
    fact(3)
    # {'i': 0, 'curr': 1}
    # {'i': 1, 'curr': 1}
    # {'i': 2, 'curr': 2}
```

You can see that the associations are different (curr is 2 when i is 2, whereas it was 6 with the other selector), but this is simply because they are now triggered when `i` is set.

### Example 2: multiple scopes

A selector may act on several nested scopes in a call graph. For example, the selector `f(x) > g(y) > h > z` would capture variables `x`, `y` and `z` from the scopes of three different functions, but only when `f` calls `g` and `g` calls `h` (either directly or indirectly).

```python
def f(x):
    return g(x + 1) * g(-x - 1)

def g(x):
    return x * 2

# Use "as" to rename a variable if there is a name conflict
with probing("f(x) > g > x as gx").print():
    f(5)
    # {'gx': 6, 'x': 5}
    # {'gx': -6, 'x': 5}
    g(10)
    # Prints nothing
```

### Example 3: overriding variables

It is also possible to override the value of a variable with the `override` (or `koverride`) methods:


```python
def add_ct(x):
    ct = 1
    return x + ct

with probing("add_ct(x) > ct") as probe:
    # The value of other variables can be used to compute the new value of ct
    probe.override(lambda data: data["x"])

    # You can also use koverride, which calls func(**data)
    # probe.koverride(lambda x: x)

    print(add_ct(3))   # sets ct = x = 3; prints 6
    print(add_ct(10))  # sets ct = x = 20; prints 20
```

**Important:** override() only overrides the **focus variable**. As explained earlier, the focus variable is the one to the right of `>`, or the one prefixed with `!`. A Ptera selector is only triggered when the focus variable is set, so realistically it is the only one that it makes sense to override.

This is worth keeping in mind, because otherwise it's not always obvious what override is doing. For example:

```python
with probing("add_ct(x) > ct") as probe:
    # The focus is ct, so override will always set ct
    # Therefore, this sets ct = 10 when x == 3:
    probe.where(x=3).override(10)

    print(add_ct(3))   # sets ct = 10; prints 13
    print(add_ct(10))  # does not override anything; prints 11
```
