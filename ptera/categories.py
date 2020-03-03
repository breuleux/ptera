from .utils import ABSENT


def _merge(a, b):
    members = set()
    members.update(a.members if isinstance(a, CategorySet) else {a})
    members.update(b.members if isinstance(b, CategorySet) else {b})
    return CategorySet(members)


class Category:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Category) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    __and__ = _merge
    __rand__ = _merge

    def __repr__(self):
        return self.name

    __str__ = __repr__


class CategorySet:
    def __init__(self, members):
        self.members = frozenset(members)

    __and__ = _merge
    __rand__ = _merge

    def __eq__(self, other):
        return isinstance(other, CategorySet) and other.members == self.members

    def __hash__(self):
        return hash(self.members)

    def __repr__(self):
        return "&".join(sorted(map(str, self.members)))

    __str__ = __repr__


class _CategoryFactory:
    def __getattr__(self, name):
        return Category(name)


def match_category(to_match, category):
    if to_match is None:
        return True
    if category is None:
        return False
    elif isinstance(category, CategorySet):
        return any(cat == to_match for cat in category.members)
    else:
        return category == to_match


cat = _CategoryFactory()
