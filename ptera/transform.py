"""Code transform that instruments probed functions."""

import ast
import inspect
import re
import tokenize
import types
from ast import NodeTransformer, NodeVisitor
from contextlib import contextmanager
from copy import deepcopy
from itertools import count
from textwrap import dedent
from types import TracebackType

from .tags import get_tags
from .utils import ABSENT

_IDX = count()


class Key:
    """Represents an attribute or index on a variable.

    Attributes:
        type: Either "attr" or "index".
        value: The value of the attribute or index.
    """

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def affix_to(self, sym):
        """Return a string representing getting the key from sym.

        >>> Key("attr", "y").affix_to("x")
        "x.y"
        >>> Key("index", "y").affix_to("x")
        "x['y']"
        """
        if self.type == "attr":
            return f"{sym}.{self.value}"
        elif self.type == "index":
            return f"{sym}[{self.value!r}]"
        else:  # pragma: no cover
            raise NotImplementedError(self.type)

    def __str__(self):
        return f"<Key {self.type}={self.value!r}>"

    __repr__ = __str__


class PteraNameError(NameError):
    """The Ptera equivalent of a NameError, which gives more information."""

    def __init__(self, varname, function):
        self.varname = varname
        self.function = function
        prov = self.info().get("provenance", None)
        if prov == "external":
            msg = (
                f"Global or nonlocal variable '{varname}' used"
                f" in function '{function}' is not set."
                " Note that ptera tries to fetch its value before"
                " executing the function."
            )
        else:
            msg = (
                f"Variable '{varname}' in function '{function}' is not set"
                " and was not given a value in the dynamic environment."
            )
        super().__init__(msg)

    def info(self):
        """Return information about the missing variable."""
        return self.function.__ptera_info__[self.varname]


def name_error(varname, function, pop_frames=1):
    """Raise a PteraNameError pointing to the right location."""
    fr = inspect.currentframe()
    for _ in range(pop_frames + 1):
        if fr:
            fr = fr.f_back
    err = PteraNameError(varname, function)
    try:  # pragma: no cover
        tb = TracebackType(
            tb_next=None,
            tb_frame=fr,
            tb_lasti=fr.f_lasti,
            tb_lineno=fr.f_lineno,
        )
        return err.with_traceback(tb)
    except TypeError:  # pragma: no cover
        return err


def _readline_mock(src):
    """Line reader for the given text.

    This is meant to be used with Python's tokenizer.
    """
    curr = -1
    src = bytes(src, encoding="utf8")
    lines = [line + b"\n" for line in src.split(b"\n")]

    def readline():
        nonlocal curr
        curr = curr + 1
        if curr >= len(lines):
            raise StopIteration
        return lines[curr]

    return readline


def _gensym():
    """Generate a fresh symbol."""
    return f"_ptera__{next(_IDX)}"


class ExternalVariableCollector(NodeVisitor):
    """Collect variables referred to but not defined in the given AST.

    The attributes are filled after the object is created.

    Attributes:
        used: Set of used variable names (does not include the names of
            inner functions).
        assigned: Set of assigned variable names.
        vardoc: Dict that maps variable names to matching comments.
        provenance: Dict that maps variable names to "body" or "argument"
            if they are defined as variables in the body or as function
            arguments.
        funcnames: Set of function names defined in the body.
    """

    def __init__(self, comments, tree):
        self.used = set()
        self.assigned = set()
        self.comments = comments
        self.vardoc = {}
        self.provenance = {}
        self.funcnames = set()
        self.visit(tree)
        self.used -= self.funcnames

    def visit_FunctionDef(self, node):
        self.funcnames.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        else:
            if node.lineno in self.comments:
                self.vardoc[node.id] = self.comments[node.lineno]
            self.provenance[node.id] = "body"
            self.assigned.add(node.id)

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.provenance[node.name] = "body"
            self.assigned.add(node.name)

    def visit_Import(self, node):
        self.visit_ImportFrom(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            name = name.split(".")[0]
            self.provenance[name] = "body"
            self.assigned.add(name)

    def visit_arg(self, node):
        if node.lineno in self.comments:
            self.vardoc[node.arg] = self.comments[node.lineno]
        self.provenance[node.arg] = "argument"
        self.assigned.add(node.arg)


class PteraTransformer(NodeTransformer):
    """Transform the AST of a function to instrument it with ptera.

    The `result` field is set to the AST of the transformed function
    after instantiation of the PteraTransformer.
    """

    def __init__(self, tree, comments, lib):
        super().__init__()
        evc = ExternalVariableCollector(comments, tree)
        self.vardoc = evc.vardoc
        self.used = evc.used
        self.assigned = evc.assigned
        self.external = evc.used - evc.assigned
        self.provenance = evc.provenance
        for ext in self.external:
            self.provenance[ext] = "external"
        self.annotated = {}
        self.linenos = {}
        self.defaults = {}
        self.lib = lib
        self.result = self.visit_FunctionDef(tree, root=True)

    def _ann(self, ann):
        if isinstance(ann, ast.Str) and ann.s.startswith("@"):
            tags = re.split(r" *& *", ann.s)
            ann = ast.Call(
                func=ast.Name(id="__ptera_get_tags", ctx=ast.Load()),
                args=[
                    ast.Str(s=tag[1:]) for tag in tags if tag.startswith("@")
                ],
                keywords=[],
            )
        return ann

    def _absent(self):
        """Create a Name that represents the lack of a value."""
        return ast.Name(id="__ptera_ABSENT", ctx=ast.Load())

    def _get(self, name):
        return ast.Name(id=self.lib[name][0], ctx=ast.Load())

    def _set(self, name):
        return ast.Name(id=self.lib[name][0], ctx=ast.Store())

    def _interact(self, *args):
        args = [
            arg if isinstance(arg, ast.AST) else ast.Constant(arg)
            for arg in args
        ]
        return ast.Call(
            func=self._get("interact"),
            args=args,
            keywords=[],
        )

    def _wrap_call(self, sym, *args):
        args = [
            (a if isinstance(a, ast.AST) else ast.Constant(value=a))
            for a in args
        ]
        return ast.Call(
            func=ast.Name(id=sym, ctx=ast.Load()),
            args=list(args),
            keywords=[],
        )

    def make_interaction(self, target, ann, value, orig=None, expression=False):
        """Create code for setting the value of a variable."""
        if ann and isinstance(target, ast.Name):
            self.annotated[target.id] = ann
            self.linenos[target.id] = target.lineno
        ann_arg = ann if ann else ast.Constant(value=None)
        value_arg = self._absent() if value is None else value
        if isinstance(target, ast.Name):
            value_args = [
                ast.Constant(value=target.id),
                ast.Constant(value=None),
                ann_arg,
                value_arg,
            ]
        elif isinstance(target, ast.Subscript) and isinstance(
            target.value, ast.Name
        ):
            slc = target.slice
            slc = slc.value if isinstance(target.slice, ast.Index) else slc
            value_args = [
                ast.Constant(value=target.value.id),
                self._wrap_call("__ptera_Key", "index", deepcopy(slc)),
                ann_arg,
                value_arg,
            ]
        elif isinstance(target, ast.Attribute) and isinstance(
            target.value, ast.Name
        ):
            value_args = [
                ast.Constant(value=target.value.id),
                self._wrap_call("__ptera_Key", "attr", target.attr),
                ann_arg,
                value_arg,
            ]
        else:
            value_args = None

        if value_args is None:
            new_value = value
        else:
            new_value = ast.Call(
                func=self._get("interact"), args=value_args, keywords=[]
            )
        if expression:
            return ast.NamedExpr(
                target=target,
                value=new_value,
                lineno=orig.lineno,
                col_offset=orig.col_offset,
            )
        else:
            return [
                ast.Assign(
                    targets=[target],
                    value=new_value,
                    lineno=orig.lineno,
                    col_offset=orig.col_offset,
                )
            ]

    def visit_body(self, stmts):
        new_body = []
        for stmt in map(self.visit, stmts):
            if isinstance(stmt, list):
                new_body.extend(stmt)
            else:
                new_body.append(stmt)
        return new_body

    def generate_interactions(self, target):
        if isinstance(target, ast.arguments):
            assert not target.vararg
            assert not target.kwonlyargs
            assert not target.kwarg
            stmts = []
            for arg in target.args:
                stmts.extend(self.generate_interactions(arg))
            return stmts

        elif isinstance(target, ast.arg):
            return self.make_interaction(
                target=ast.copy_location(
                    ast.Name(id=target.arg, ctx=ast.Store()), target
                ),
                ann=self._ann(target.annotation),
                value=ast.copy_location(
                    ast.Name(id=target.arg, ctx=ast.Load()), target
                ),
                orig=target,
            )

        elif isinstance(target, ast.Name):
            return self.make_interaction(
                target=ast.copy_location(
                    ast.Name(id=target.id, ctx=ast.Store()), target
                ),
                ann=None,
                value=ast.copy_location(
                    ast.Name(id=target.id, ctx=ast.Load()), target
                ),
                orig=target,
            )

        elif isinstance(target, ast.Tuple):
            stmts = []
            for entry in target.elts:
                stmts.extend(self.generate_interactions(entry))
            return stmts

        else:  # pragma: no cover
            raise NotImplementedError(target)

    def visit_FunctionDef(self, node, root=False):
        if not root:
            return node

        wrapped_body = []

        new_body = [ast.Expr(self._interact("#enter", None, None, True))]

        new_body += self.generate_interactions(node.args)

        for external in sorted(self.external):
            new_body.extend(
                self.make_interaction(
                    target=ast.Name(id=external, ctx=ast.Store()),
                    ann=None,
                    value=ast.Subscript(
                        value=ast.Name(id="__ptera_globals", ctx=ast.Load()),
                        slice=ast.Constant(external),
                        ctx=ast.Load(),
                    ),
                    orig=node,
                )
            )

        body = node.body
        first = body[0]
        if isinstance(first, ast.Expr):
            v = first.value
            if (
                isinstance(v, ast.Str)
                or isinstance(v, ast.Constant)
                and isinstance(v.value, str)
            ):
                wrapped_body.append(first)
                body = body[1:]

        new_body += self.visit_body(node.body)

        wrapped_body.append(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=self._get("proceed"),
                            args=[self._get("self")],
                            keywords=[],
                        ),
                        optional_vars=self._set("frame"),
                    ),
                ],
                body=new_body,
            )
        )

        return ast.copy_location(
            ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=wrapped_body,
                decorator_list=node.decorator_list,
                returns=node.returns,
            ),
            node,
        )

    def visit_For(self, node):

        new_body = self.generate_interactions(node.target)
        new_body.extend(self.visit_body(node.body))

        return ast.copy_location(
            ast.For(
                target=node.target,
                iter=self.visit(node.iter),
                body=new_body,
                orelse=self.visit_body(node.orelse),
            ),
            node,
        )

    def visit_ExceptHandler(self, node):
        if node.name is None:
            new_body = []
        else:
            target = ast.copy_location(
                ast.Name(id=node.name, ctx=ast.Store()), node
            )
            new_body = self.generate_interactions(target)
        new_body.extend(self.visit_body(node.body))
        return ast.copy_location(
            ast.ExceptHandler(
                name=node.name,
                type=self.visit(node.type),
                body=new_body,
            ),
            node,
        )

    def visit_NamedExpr(self, node):
        """Rewrite an assignment expression.

        Before:
            x := y + z

        After:
            x := _ptera_interact('x', None, y + z)
        """
        return self.make_interaction(
            node.target,
            None,
            self.visit(node.value),
            orig=node,
            expression=True,
        )

    def visit_AnnAssign(self, node):
        """Rewrite an annotated assignment statement.

        Before:
            x: int

        After:
            x: int = _ptera_interact('x', int)
        """
        return self.make_interaction(
            node.target, self._ann(node.annotation), node.value, orig=node
        )

    def visit_Assign(self, node):
        """Rewrite an assignment statement.

        Before:
            x = y + z

        After:
            x = _ptera_interact('x', None, y + z)
        """
        (target,) = node.targets
        if isinstance(target, ast.Tuple):
            var_all = _gensym()
            ass_all = ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id=var_all, ctx=ast.Store())],
                    value=node.value,
                ),
                node,
            )
            accum = [ass_all]
            for i, tgt in enumerate(target.elts):
                accum += self.visit_Assign(
                    ast.copy_location(
                        ast.Assign(
                            targets=[tgt],
                            value=ast.Subscript(
                                value=ast.Name(id=var_all, ctx=ast.Load()),
                                slice=ast.Index(value=ast.Constant(i)),
                                ctx=ast.Load(),
                            ),
                        ),
                        node,
                    )
                )
            return accum
        else:
            return self.make_interaction(target, None, node.value, orig=node)

    def visit_Import(self, node):
        """Rewrite an import statement.

        Before:
            import kangaroo

        After:
            import kangaroo
            kangaroo = _ptera_interact('kangaroo', None, kangaroo)
        """
        return self.visit_ImportFrom(node)

    def visit_ImportFrom(self, node):
        """Rewrite an import statement.

        Before:
            from kangaroo import jump

        After:
            from kangaroo import jump
            jump = _ptera_interact('jump', None, jump)
        """
        stmts = [node]
        for alias in node.names:
            name = alias.asname or alias.name
            if "." not in name:
                name_node = ast.copy_location(
                    ast.Name(id=name, context=ast.Load()),
                    node,
                )
                stmts.extend(self.generate_interactions(name_node))
        return stmts

    def visit_Return(self, node):
        new_value = self._interact(
            "#value",
            None,
            None,
            self.visit(node.value or ast.Constant(value=None)),
        )
        return ast.copy_location(ast.Return(value=new_value), node)


class _Conformer:
    """Implements codefind's __conform__ protocol.

    This allows a package like jurigged to hot patch functions that
    are modified by ptera, and ptera will be able to re-run the
    tooling on the new version. Might not work perfectly reliably.
    """

    __slots__ = ("code", "ptera_fn", "interact")

    def __init__(self, fn, ptera_fn, interact):
        self.ptera_fn = ptera_fn
        self.code = fn.__code__
        self.interact = interact

    def __conform__(self, new):
        from codefind import code_registry

        if isinstance(new, types.CodeType):
            self.code = new
            return

        new_fn = new
        new_code = new.__code__

        result, _ = transform(new_fn, self.interact)
        self.ptera_fn.__code__ = result.__code__

        code_registry.update_cache_entry(self, self.code, new_code)
        self.code = new_code


class _Resolver:
    def __init__(self, *dicts):
        self.dicts = dicts

    def __getitem__(self, item):
        for d in self.dicts:
            if item in d:
                return d[item]
        return ABSENT


@contextmanager
def _default_proceed(fn):
    yield


def transform(fn, interact, proceed=_default_proceed):
    """Return an instrumented version of fn.

    Arguments:
        fn: The function to instrument.
        interact: The function to call each time the value of an instrumented
          variable is changed. It receives the arguments
          ``(symbol, key, category, value)``
          (see :func:`~ptera.interact.interact`)

    Returns:
        A (newfn, info) tuple.

        * newfn: A new function that is an instrumented version of the old one.
        * info: A dictionary mapping each instrumented variable to information
          about whether it is local or global, and comments about the variable.
    """

    src = dedent(inspect.getsource(fn))

    # Scrape the comments in the function's source and map them to lines.
    comments = {}
    for tok in tokenize.tokenize(_readline_mock(src)):
        if tok.type == tokenize.COMMENT:
            if tok.line.strip().startswith("#"):
                line = tok.end[0]
                comments[line + 1] = tok.string[1:].strip()
                if line in comments:
                    comments[line + 1] = (
                        comments[line] + "\n" + comments[line + 1]
                    )
                    del comments[line]

    # Perform the transform
    filename = inspect.getsourcefile(fn)
    tree = ast.parse(src, filename)
    tree = tree.body[0]
    assert isinstance(tree, ast.FunctionDef)
    tree.decorator_list = []

    fnsym = _gensym()
    glb = fn.__globals__
    lib = {
        "interact": (f"__ptera_{id(interact)}", interact),
        "proceed": (f"__ptera_{id(proceed)}", proceed),
        "globals": ("__ptera_globals", _Resolver(glb, __builtins__)),
        "ABSENT": ("__ptera_ABSENT", ABSENT),
        "Key": ("__ptera_Key", Key),
        "get_tags": ("__ptera_get_tags", get_tags),
        "self": (fnsym, None),
        "frame": ("__ptera_frame", None),
    }
    glb.update(
        {name: value for name, value in lib.values() if value is not None}
    )

    transformer = PteraTransformer(tree, comments, lib)
    new_tree = transformer.result
    ast.fix_missing_locations(new_tree)
    _, lineno = inspect.getsourcelines(fn)
    ast.increment_lineno(new_tree, lineno - 1)
    new_fn = compile(
        ast.Module(body=[new_tree], type_ignores=[]), filename, "exec"
    )

    fname = fn.__name__
    save = glb.get(fname, None)
    exec(new_fn, glb, glb)

    try:
        from codefind import code_registry

        co = fn.__code__
        code_registry.assimilate(co, (co.co_filename,))
    except ImportError:  # pragma: no cover
        pass

    # Get the new function (populated with exec)
    actual_fn = glb[fname]
    glb[fnsym] = actual_fn

    # However, we don't want to change the existing mapping of fn
    glb[fname] = save

    all_vars = transformer.used | transformer.assigned

    info = {
        k: {
            "name": k,
            "annotation": (
                eval(
                    compile(
                        ast.Expression(transformer.annotated[k]),
                        filename,
                        "eval",
                    ),
                    glb,
                    glb,
                )
                if k in transformer.annotated
                else ABSENT
            ),
            "provenance": transformer.provenance.get(k),
            "doc": transformer.vardoc.get(k),
            "location": (
                filename,
                fn,
                transformer.linenos[k] + lineno - 1
                if k in transformer.linenos
                else None,
            ),
        }
        for k in all_vars
    }

    actual_fn._conformer = _Conformer(fn, actual_fn, interact)
    actual_fn.__ptera_info__ = info
    actual_fn.__ptera_token__ = fnsym
    return actual_fn, info
