import ast
import inspect
from ast import NodeTransformer
from contextlib import contextmanager
from textwrap import dedent

from _pytest.assertion.rewrite import AssertionRewriter

from ptera.interpret import Immediate, Total
from ptera.overlay import BaseOverlay
from ptera.selector import select


class AssertTransformer(NodeTransformer):
    def visit_FunctionDef(self, node):
        newfns = []
        for i, stmt in enumerate(node.body):
            if not isinstance(stmt, ast.Assert):
                raise Exception(
                    "@one_test_per_assert requires all statements to be asserts"
                )
            else:
                newfns.append(
                    ast.FunctionDef(
                        name=f"{node.name}_assert{i + 1}",
                        args=node.args,
                        body=[stmt],
                        decorator_list=node.decorator_list,
                        returns=node.returns,
                    )
                )
        return ast.Module(body=newfns, type_ignores=[])


def one_test_per_assert(fn):
    src = dedent(inspect.getsource(fn))
    filename = inspect.getsourcefile(fn)
    tree = ast.parse(src, filename)
    tree = tree.body[0]
    assert isinstance(tree, ast.FunctionDef)
    tree.decorator_list = []
    new_tree = AssertTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    _, lineno = inspect.getsourcelines(fn)
    ast.increment_lineno(new_tree, lineno - 1)
    # Use pytest's assertion rewriter for nicer error messages
    AssertionRewriter(filename, None, None).run(new_tree)
    new_fn = compile(new_tree, filename, "exec")
    glb = fn.__globals__
    exec(new_fn, glb, glb)
    if hasattr(fn, "pytestmark"):
        for name, value in glb.items():
            if name.startswith(fn.__name__):
                value.pytestmark = fn.pytestmark
    return None


class TapResults(list):
    def __getitem__(self, item):
        if isinstance(item, str):
            return [x[item] for x in self if item in x]
        else:
            return super().__getitem__(item)


@contextmanager
def tapping(pattern, all=False, full=False):
    results = TapResults()

    def listener(args):
        results.append(
            {
                name: cap.values if all else cap.value
                for name, cap in args.items()
            }
        )

    cls = Total if full else Immediate
    with BaseOverlay(cls(select(pattern, skip_frames=1), listener)):
        yield results


def full_tapping(pattern, all=True):
    return tapping(pattern, all=all, full=True)


def all_tapping(pattern, full=False):
    return tapping(pattern, all=True, full=full)
