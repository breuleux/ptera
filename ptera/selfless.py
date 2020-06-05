import ast
import builtins
import inspect
import tokenize
from ast import NodeTransformer, NodeVisitor
from copy import copy, deepcopy
from textwrap import dedent

from .utils import ABSENT, keyword_decorator

idx = 0


def readline_mock(src):
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


def gensym():
    global idx
    idx += 1
    return f"_ptera_tmp_{idx}"


class ExternalVariableCollector(NodeVisitor):
    def __init__(self, comments, tree):
        self.used = set()
        self.assigned = set()
        self.comments = comments
        self.vardoc = {}
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
            self.assigned.add(node.id)

    def visit_arg(self, node):
        if node.lineno in self.comments:
            self.vardoc[node.arg] = self.comments[node.lineno]
        self.assigned.add(node.arg)


class PteraTransformer(NodeTransformer):
    def __init__(self, tree, comments):
        super().__init__()
        evc = ExternalVariableCollector(comments, tree)
        self.vardoc = evc.vardoc
        self.used = evc.used
        self.assigned = evc.assigned
        self.external = evc.used - evc.assigned
        self.annotated = {}
        self.linenos = {}
        self.defaults = {}
        self.result = self.visit_FunctionDef(tree, root=True)

    def fself(self):
        return ast.Name("__self__", ctx=ast.Load())

    def _absent(self):
        return ast.Name("__ptera_ABSENT", ctx=ast.Load())

    def make_interaction(self, target, ann, value, orig=None):
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
                self.fself(),
                value_arg,
            ]
        elif isinstance(target, ast.Subscript) and isinstance(
            target.value, ast.Name
        ):
            value_args = [
                ast.Constant(value=target.value.id),
                deepcopy(target.slice.value),
                ann_arg,
                self.fself(),
                value_arg,
            ]
        else:
            value_args = None

        if value_args is None:
            new_value = value
        else:
            new_value = ast.Call(
                func=ast.Name("__ptera_interact", ctx=ast.Load()),
                args=value_args,
                keywords=[],
            )
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
                ann=target.annotation,
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

        new_body = self.generate_interactions(node.args)

        for external in self.external:
            new_body.extend(
                self.make_interaction(
                    target=ast.Name(id=external, ctx=ast.Store()),
                    ann=None,
                    value=ast.Name(id="__ptera_ABSENT", ctx=ast.Load()),
                    orig=node,
                )
            )
        new_args = ast.arguments(
            posonlyargs=[],
            args=list(node.args.args),
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[
                ast.copy_location(
                    ast.Name(id="__ptera_ABSENT", ctx=ast.Load()), arg
                )
                for arg in node.args.args
            ],
        )
        for dflt, arg in zip(
            node.args.defaults, node.args.args[-len(node.args.defaults) :]
        ):
            self.defaults[arg.arg] = dflt
        new_args.args.insert(0, ast.arg("__self__"))

        first = node.body[0]
        if isinstance(first, ast.Expr):
            v = first.value
            if (
                isinstance(v, ast.Str)
                or isinstance(v, ast.Constant)
                and isinstance(v.value, str)
            ):
                new_body.insert(0, first)

        new_body += self.visit_body(node.body)

        return ast.copy_location(
            ast.FunctionDef(
                name=node.name,
                args=new_args,
                body=new_body,
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

    def visit_Return(self, node):
        new_value = ast.Call(
            func=ast.Name("__ptera_interact", ctx=ast.Load()),
            args=[
                ast.Constant(value="#value"),
                ast.Constant(value=None),
                ast.Constant(value=None),
                self.fself(),
                node.value or ast.Constant(value=None),
            ],
            keywords=[],
        )
        return ast.copy_location(ast.Return(value=new_value), node)

    def visit_AnnAssign(self, node):
        """Rewrite an annotated assignment expression.

        Before::
            x: int

        After::
            x: int = ptera.interact('x', int)
        """
        return self.make_interaction(
            node.target, node.annotation, node.value, orig=node
        )

    def visit_Assign(self, node):
        """Rewrite an assignment expression.

        Before::
            x = y + z

        After::
            x = ptera.interact('x', None, y + z)
        """
        (target,) = node.targets
        if isinstance(target, ast.Tuple):
            var_all = gensym()
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


def transform(fn, interact):
    src = dedent(inspect.getsource(fn))

    comments = {}
    for tok in tokenize.tokenize(readline_mock(src)):
        if tok.type == tokenize.COMMENT:
            if tok.line.strip().startswith("#"):
                line = tok.end[0]
                comments[line + 1] = tok.string[1:].strip()
                if line in comments:
                    comments[line + 1] = (
                        comments[line] + "\n" + comments[line + 1]
                    )
                    del comments[line]

    filename = inspect.getsourcefile(fn)
    tree = ast.parse(src, filename)
    tree = tree.body[0]
    assert isinstance(tree, ast.FunctionDef)
    tree.decorator_list = []
    transformer = PteraTransformer(tree, comments)
    new_tree = transformer.result
    ast.fix_missing_locations(new_tree)
    _, lineno = inspect.getsourcelines(fn)
    ast.increment_lineno(new_tree, lineno - 1)
    new_fn = compile(
        ast.Module(body=[new_tree], type_ignores=[]), filename, "exec"
    )
    glb = fn.__globals__
    glb["__ptera_interact"] = interact
    glb["__ptera_ABSENT"] = ABSENT
    exec(new_fn, glb, glb)

    state = {
        k: override(
            eval(compile(ast.Expression(v), filename, "eval"), glb, glb),
            priority=-0.5,
        )
        for k, v in transformer.defaults.items()
    }

    fname = fn.__name__
    actual_fn = glb[fname]
    all_vars = transformer.used | transformer.assigned

    info = {
        k: {
            "doc": transformer.vardoc.get(k),
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

    state_obj = state_class(fname, info)(state)

    # The necessary globals may not yet be set, so we create a "PreState" that
    # will be filled in whenever we first need to fetch the state.
    state_obj = PreState(state=state_obj, names=transformer.external, glbls=glb)
    return actual_fn, state_obj


class Override:
    def __init__(self, value, priority=1):
        assert not isinstance(value, Override)
        self.value = value
        self.priority = priority


def override(value, priority=1):
    if isinstance(value, Override):
        return value
    else:
        return Override(value, priority=priority)


def default(value, priority=-10):
    return override(value, priority)


class PreState:
    def __init__(self, state, names, glbls):
        self.state = state
        self.names = names
        self.glbls = glbls

    def make(self):
        for varname in self.names:
            val = self.glbls.get(varname, ABSENT)
            if val is ABSENT:
                val = getattr(builtins, varname, ABSENT)
            setattr(self.state, varname, val)
        return self.state


class BaseState:
    __slots__ = ()

    def __init__(self, values):
        for k, v in values.items():
            setattr(self, k, v)


def state_class(fname, info):
    return type(
        f"{fname}.state",
        (BaseState,),
        {"__slots__": tuple(info.keys()), "__info__": info},
    )


class Selfless:
    def __init__(self, fn, state):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.state_obj = state

    @property
    def state(self):
        self.ensure_state()
        return self.state_obj

    def ensure_state(self):
        if isinstance(self.state_obj, PreState):
            self.state_obj = self.state_obj.make()

    def new(self, **values):
        rval = self.clone()
        for k, v in values.items():
            setattr(rval.state_obj, k, v)
        return rval

    def clone(self, **kwargs):
        self.ensure_state()
        kwargs = {"fn": self.fn, "state": copy(self.state_obj), **kwargs}
        return type(self)(**kwargs)

    def get(self, name):
        return getattr(self.state_obj, name, ABSENT)

    def __call__(self, *args, **kwargs):
        self.ensure_state()
        return self.fn(self, *args, **kwargs)

    def __str__(self):
        return f"{self.fn.__name__}"


class ConflictError(Exception):
    pass


def choose(opts, name):
    real_opts = [opt for opt in opts if opt is not ABSENT]
    if not real_opts:
        return ABSENT
    elif len(real_opts) == 1:
        (opt,) = real_opts
        return opt.value if isinstance(opt, Override) else opt
    else:
        with_prio = [
            (opt.value, -opt.priority)
            if isinstance(opt, Override)
            else (opt, 0)
            for opt in real_opts
        ]
        with_prio.sort(key=lambda x: x[1])
        p = with_prio[0][1]
        if with_prio[1][1] == p:
            conflicts = [str(x) for x, y in with_prio if y == p]
            raise ConflictError(
                f"Multiple values with same priority conflict for "
                f"variable '{name}': {', '.join(conflicts)}"
            )
        return with_prio[0][0]


def selfless_interact(sym, key, category, __self__, value):
    from_state = __self__.get(sym)
    rval = choose([value, from_state], name=sym)
    if rval is ABSENT:
        raise NameError(f"Variable {sym} of {__self__} is not set.")
    assert not isinstance(rval, Override)
    return rval


@keyword_decorator
def selfless(fn, **defaults):
    new_fn, state = transform(fn, interact=selfless_interact)
    rval = Selfless(new_fn, state)
    if defaults:
        rval = rval.new(**defaults)
    return rval
