# Features only present in Python >= 3.8


# Positional-only argument
def gnarly(x, /, y):
    return x + y


# Walrus operators
def ratatouille(x):
    return (y := x + 1) * y
