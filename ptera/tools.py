class Range:
    def __init__(self, start=0, end=None, modulo=None):
        self.start = start
        self.end = end
        self.modulo = modulo

    def __call__(self, value):
        if self.start is not None and value < self.start:
            return False
        if self.end is not None and value >= self.end:
            return False
        if self.modulo is not None:
            return (
                (value - (self.start or 0)) + self.modulo
            ) % self.modulo == 0
        return True


def every(modulo=None, start=0, end=None):
    return Range(modulo=modulo, start=start, end=end)


def between(start, end, modulo=None):
    return Range(modulo=modulo, start=start, end=end)


def lt(end):
    return lambda x: x < end


def gt(start):
    return lambda x: x > start


def lte(end):
    return lambda x: x <= end


def gte(start):
    return lambda x: x >= start


class throttle:
    def __init__(self, period):
        self.period = period
        self.current = None
        self.trigger = None

    def __call__(self, value):
        if self.current is None:
            self.current = value
            self.trigger = self.current + self.period

        if value == self.current:
            return True
        elif value >= self.trigger:
            self.current = value
            self.trigger += self.period
            return True
        else:
            return False
