import ast
import builtins
import inspect
import tokenize
from ast import NodeTransformer, NodeVisitor
from copy import copy
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
        self.defaults = {}
        self.result = self.visit_FunctionDef(tree, root=True)

    def fself(self):
        return ast.Name("__self__", ctx=ast.Load())

    def _absent(self):
        return ast.Name("__ptera_ABSENT", ctx=ast.Load())

    def make_interaction(self, target, ann, value):
        if ann and isinstance(target, ast.Name):
            self.annotated[target.id] = ann
        ann_arg = ann if ann else ast.Constant(value=None)
        value_arg = self._absent() if value is None else value
        if isinstance(target, ast.Name):
            value_args = [
                ast.Constant(value=target.id),
                ast.Constant(value=None),
                # self._absent(),
                ann_arg,
                self.fself(),
                value_arg,
            ]
        elif isinstance(target, ast.Subscript):
            value_args = [
                ast.Constant(value=target.value.id),
                target.slice.value,
                ann_arg,
                self.fself(),
                value_arg,
            ]
        else:
            raise SyntaxError(target)
        new_value = ast.Call(
            func=ast.Name("__ptera_interact", ctx=ast.Load()),
            args=value_args,
            keywords=[],
        )
        if ann is None:
            return [ast.Assign(targets=[target], value=new_value)]
        else:
            return [
                ast.AnnAssign(
                    target=target, value=new_value, annotation=ann, simple=True
                )
            ]

    def visit_FunctionDef(self, node, root=False):
        # assert not node.args.posonlyargs
        assert not node.args.vararg
        assert not node.args.kwonlyargs
        assert not node.args.kwarg
        # (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
        #  expr* kw_defaults, arg? kwarg, expr* defaults)

        new_body = []
        for arg in node.args.args:
            new_body.extend(
                self.make_interaction(
                    target=ast.Name(id=arg.arg, ctx=ast.Store()),
                    ann=arg.annotation,
                    value=ast.Name(id=arg.arg, ctx=ast.Load()),
                )
            )

        if root:
            for external in self.external:
                new_body.extend(
                    self.make_interaction(
                        target=ast.Name(id=external, ctx=ast.Store()),
                        ann=None,
                        value=ast.Name(id="__ptera_ABSENT", ctx=ast.Load()),
                    )
                )
            new_args = ast.arguments(
                # posonlyargs=[],
                args=list(node.args.args),
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[
                    ast.Name(id="__ptera_ABSENT", ctx=ast.Load())
                    for _ in node.args.args
                ],
            )
            for dflt, arg in zip(
                node.args.defaults, node.args.args[-len(node.args.defaults) :]
            ):
                self.defaults[arg.arg] = dflt
            new_args.args.insert(0, ast.arg("__self__"))
        else:
            new_args = node.args

        # node.args.args = new_args
        for stmt in map(self.visit, node.body):
            if isinstance(stmt, list):
                new_body.extend(stmt)
            else:
                new_body.append(stmt)
        return ast.FunctionDef(
            name=node.name,
            args=new_args,
            body=new_body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            # type_comment=node.type_comment,
        )

    def visit_Return(self, node):
        new_value = ast.Call(
            func=ast.Name("__ptera_interact", ctx=ast.Load()),
            args=[
                ast.Constant(value="#value"),
                ast.Constant(value=None),
                ast.Constant(value=None),
                self.fself(),
                node.value,
            ],
            keywords=[],
        )
        return ast.Return(value=new_value)

    def visit_AnnAssign(self, node):
        """Rewrite an annotated assignment expression.

        Before::
            x: int

        After::
            x: int = ptera.interact('x', int)
        """
        return self.make_interaction(node.target, node.annotation, node.value)

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
            ass_all = ast.Assign(
                targets=[ast.Name(id=var_all, ctx=ast.Store())],
                value=node.value,
            )
            accum = [ass_all]
            for i, tgt in enumerate(target.elts):
                accum += self.visit_Assign(
                    ast.Assign(
                        targets=[tgt],
                        value=ast.Subscript(
                            value=ast.Name(id=var_all, ctx=ast.Load()),
                            slice=ast.Index(value=ast.Constant(i)),
                            ctx=ast.Load(),
                        ),
                    )
                )
            return accum
        else:
            return self.make_interaction(target, None, node.value)


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

    annotations = {
        k: eval(compile(ast.Expression(v), filename, "eval"), glb, glb)
        for k, v in transformer.annotated.items()
    }

    fname = fn.__name__
    actual_fn = glb[fname]
    all_vars = transformer.used | transformer.assigned
    state_obj = state_class(fname, all_vars, transformer.vardoc, annotations)(
        state
    )
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


def default(value, priority=-1):
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


def state_class(fname, slots, vardoc, annotations):
    for slot in slots:
        annotations.setdefault(slot, ABSENT)
    return type(
        f"{fname}.state",
        (BaseState,),
        {
            "__slots__": tuple(slots),
            "__vardoc__": vardoc,
            "__annotations__": annotations,
        },
    )


class Selfless:
    def __init__(self, fn, state):
        self.fn = fn
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


def choose(opts):
    real_opts = [opt for opt in opts if opt is not ABSENT]
    if not real_opts:
        return False, None
    elif len(real_opts) == 1:
        (opt,) = real_opts
        return True, opt.value if isinstance(opt, Override) else opt
    else:
        with_prio = [
            (opt.value, -opt.priority)
            if isinstance(opt, Override)
            else (opt, 0)
            for opt in real_opts
        ]
        with_prio.sort(key=lambda x: x[1])
        if with_prio[1][1] == with_prio[0][1]:
            raise ConflictError("Multiple values with same priority conflict.")
        return True, with_prio[0][0]


def selfless_interact(sym, key, category, __self__, value):
    from_state = __self__.get(sym)
    success, rval = choose([value, from_state])
    if not success:
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
