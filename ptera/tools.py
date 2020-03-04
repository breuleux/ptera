import argparse
import json
import os
import re
import sys
from collections import defaultdict

from .categories import CategorySet, match_category
from .core import PteraFunction, overlay
from .selector import to_pattern
from .selfless import PreState
from .utils import ABSENT


def _catalogue(seen, results, fn):
    if id(fn) in seen:
        return

    seen.add(id(fn))

    if isinstance(fn, PteraFunction):
        state = fn.state.state if isinstance(fn.state, PreState) else fn.state
        res = {}
        results[fn] = res
        tst = type(state)
        for name, ann in tst.__annotations__.items():
            res[name] = {
                "annotation": ann,
                "doc": tst.__vardoc__.get(name, None),
            }
            val = getattr(state, name, ABSENT)
            _catalogue(seen, results, val)

    elif isinstance(fn, (list, tuple)):
        for entry in fn:
            _catalogue(seen, results, entry)


def catalogue(functions):
    results = {}
    _catalogue(set(), results, functions)
    return results


def _find_configurable(catalogue, category):
    rval = defaultdict(dict)
    for fn, variables in catalogue.items():
        for name, data in variables.items():
            ann = data["annotation"]
            if match_category(category, ann):
                rval[name][fn] = data
    return rval


def _fill_argparser(parser, names):
    entries = list(sorted(list(names.items())))
    for name, data in entries:
        docs = set()
        for fn, entry in data.items():
            if entry["doc"]:
                docs.add(entry["doc"])
            else:
                docs.add(f"Parameter in {fn}")

        optname = name.replace("_", "-")
        typ = []
        for x in data.values():
            ann = x["annotation"]
            if isinstance(ann, CategorySet):
                members = ann.members
            else:
                members = [ann]
            for m in members:
                if isinstance(m, type):
                    typ.append(m)
        if len(typ) != 1:
            typ = None
        else:
            (typ,) = typ

        if typ is bool:
            parser.add_argument(
                f"--{optname}",
                dest=name,
                action="store_true",
                help="; ".join(docs),
            )
            parser.add_argument(
                f"--no-{optname}",
                dest=name,
                action="store_false",
                help=f"Set --{optname} to False",
            )
        else:
            parser.add_argument(
                f"--{optname}",
                dest=name,
                type=typ or None,
                action="store",
                metavar="VALUE",
                help="; ".join(docs),
            )

    return parser


class Configurator:
    def __init__(
        self,
        *,
        entry_point=None,
        category=None,
        cli=True,
        description=None,
        argparser=None,
        eval_env=None,
        argv=None,
        config_option=False,
        default_config_file=None,
    ):
        cg = catalogue(entry_point)
        self.category = category
        self.argv = argv
        self.names = _find_configurable(cg, category)
        if cli:
            if argparser is None:
                argparser = argparse.ArgumentParser(
                    description=description, argument_default=argparse.SUPPRESS
                )
        self.argparser = _fill_argparser(argparser, self.names)
        if config_option:
            if config_option is True:
                config_option = "config"
            self.argparser.add_argument(
                f"--{config_option}",
                action="store",
                dest="#config",
                metavar="FILE",
                help="Configuration file to read options from.",
            )
            self.argparser.add_argument(
                f"--save-{config_option}",
                action="store",
                dest="#save_config",
                metavar="FILE",
                help="Configuration file to save the options to.",
            )
        self.eval_env = eval_env
        self.default_config_file = default_config_file

    def resolve(self, arg):
        if arg is ABSENT:
            return arg
        elif not isinstance(arg, str):
            return arg
        elif arg in ("True", "False", "None"):
            return eval(arg)
        elif re.match(r"^:[A-Za-z_0-9]+:", arg):
            _, modname, code = arg.split(":", 2)
            mod = __import__(modname)
            return eval(code, vars(mod))
        elif arg.startswith(":"):
            if not self.eval_env:
                raise Exception(f"No environment to evaluate {arg}")
            return eval(arg[1:], self.eval_env)
        else:
            try:
                float(arg)
                return eval(arg)
            except ValueError:
                return arg

    def get_options(self):
        args = self.argparser.parse_args(self.argv)
        opts = {k: v for k, v in vars(args).items() if not k.startswith("#")}

        cfg, must_exist = getattr(args, "#config", None), True
        if not cfg:
            cfg, must_exist = self.default_config_file, False
        if cfg:
            exists = os.path.exists(cfg)
            if exists:
                with open(cfg) as f:
                    opts2 = json.load(f)
                    opts = {**opts2, **opts}
            elif must_exist:
                print(
                    f"Error: configuration file '{cfg}' does not exist",
                    file=sys.stderr,
                )
                sys.exit(1)

        save = getattr(args, "#save_config", None)
        if save:
            print(f"Saving configuration to {save}")
            with open(save, "w") as f:
                json.dump(opts, f, indent=4)
                f.write("\n")
            sys.exit(0)
        return opts

    def __enter__(self):
        def _resolve(arg):
            return opts.get(arg.name, ABSENT)

        opts = self.get_options()
        opts = {name: self.resolve(value) for name, value in opts.items()}
        pattern = to_pattern(f"$arg:##X", env={"##X": self.category})
        self.ov = overlay({pattern: {"value": _resolve}})
        self.ov.__enter__()
        return self

    def __exit__(self, exc, typ, tb):
        return self.ov.__exit__(exc, typ, tb)


def auto_cli(fn, args=(), **kwargs):
    with Configurator(entry_point=fn, **kwargs):
        return fn(*args)
