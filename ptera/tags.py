"""Tag system for variables.

Variables can be tagged as e.g. ``x: ptera.tag.Important`` and the
selectors ``x:Important`` or ``*:Important`` would match it.
Alternatively, Ptera recognizes ``x: "@Important"`` as referring
to these tags.
"""


def _merge(a, b):
    members = set()
    members.update(a.members if isinstance(a, TagSet) else {a})
    members.update(b.members if isinstance(b, TagSet) else {b})
    return TagSet(members)


class Tag:
    """Tag for a variable, to be used as an annotation.

    Arguments:
        name: The name of the tag.
    """

    def __init__(self, name):
        self.name = name

    __and__ = _merge
    __rand__ = _merge

    def __repr__(self):
        return f"ptera.tag.{self.name}"

    __str__ = __repr__


class TagSet:
    """Set of multiple tags."""

    def __init__(self, members):
        self.members = frozenset(members)

    __and__ = _merge
    __rand__ = _merge

    def __eq__(self, other):
        return isinstance(other, TagSet) and other.members == self.members

    def __repr__(self):
        return " & ".join(sorted(map(str, self.members)))

    __str__ = __repr__


class _TagFactory:
    def __init__(self):
        self._cache = {}

    def __getattr__(self, name):
        if name not in self._cache:
            self._cache[name] = Tag(name)
        return self._cache[name]


def match_tag(to_match, tg):
    """Return whether two Tags or TagSets match.

    Only tg can be a TagSet.
    """

    if to_match is None:
        return True
    if tg is None:
        return False
    elif isinstance(tg, TagSet):
        return any(cat == to_match for cat in tg.members)
    else:
        return tg == to_match


def get_tags(*tags):
    """Build a Tag or TagSet from strings."""
    tags = [getattr(tag, tg) if isinstance(tg, str) else tg for tg in tags]
    if len(tags) == 1:
        return tags[0]
    else:
        return TagSet(tags)


tag = _TagFactory()
