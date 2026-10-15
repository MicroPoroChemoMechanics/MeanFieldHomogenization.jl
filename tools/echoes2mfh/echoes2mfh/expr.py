"""Python expression -> Julia expression.

This is the leaf translator. It handles the ordinary-Python half of a script
(arithmetic, comprehensions, lambdas, indexing) and defers every Echoes-API
construct to the caller through `on_echoes`, so that `extract.py` can decide
whether an expression is a phase access, a homogenize call, or plain algebra.

Anything it cannot translate faithfully raises `Untranslatable` rather than
guessing. A wrong number is far more expensive than a refusal.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import mapping


class Untranslatable(Exception):
    def __init__(self, reason: str, node: ast.AST | None = None, suggestion: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.suggestion = suggestion
        self.lineno = getattr(node, "lineno", 0)


# Julia operator precedence, high binds tighter. Used to parenthesize minimally
# so the output reads like hand-written code.
_PREC = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 3,
    "<=": 3,
    ">": 3,
    ">=": 3,
    "+": 4,
    "-": 4,
    "*": 5,
    "/": 5,
    "%": 5,
    "÷": 5,
    "^": 7,
    "unary": 6,
}

_BINOP = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.Pow: "^",
    ast.FloorDiv: "÷",
    ast.MatMult: "*",
}

_CMPOP = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


# numpy array attributes that have a direct Julia spelling.
_ARRAY_ATTRS = {
    "T": "transpose({0})",
    "real": "real({0})",
    "imag": "imag({0})",
    "shape": "size({0})",
    "size": "length({0})",
    "ndim": "ndims({0})",
}

# Zero-argument methods that are identity or near-identity in Julia.
_NOOP_METHODS = {
    "tolist": "{0}",
    "copy": "copy({0})",
    "flatten": "vec({0})",
    "ravel": "vec({0})",
    "conjugate": "conj({0})",
    "item": "only({0})",
}


@dataclass
class Context:
    """What the translator knows about the names in scope."""

    #: names bound to an Echoes `rve(...)`
    rve_vars: set[str] = field(default_factory=set)
    #: names bound to a tensor-valued expression
    tensor_vars: set[str] = field(default_factory=set)
    #: loop variables that are used *only* as subscripts, so their range was
    #: emitted 1-based and their subscripts need no shift
    unshifted_indices: set[str] = field(default_factory=set)
    #: names of user-defined helpers (so calls to them are left alone)
    helpers: set[str] = field(default_factory=set)
    #: local variables currently in scope
    locals: set[str] = field(default_factory=set)
    #: callback invoked for Echoes API calls the caller wants to intercept
    on_echoes: Optional[Callable[[ast.Call, str], Optional[str]]] = None
    #: callback invoked for attribute reads (phase accessors like `.eE`)
    on_attribute: Optional[Callable[[ast.Attribute], Optional[str]]] = None


def julia_number(v) -> str:
    """Render a Python literal as Julia, keeping float-ness explicit."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v == float("inf"):
            return "Inf"
        if v == float("-inf"):
            return "-Inf"
        s = repr(v)
        # Julia wants 1.0e-6 rather than 1e-06
        if "e" in s or "E" in s:
            mant, _, exp = s.partition("e")
            if "." not in mant:
                mant += ".0"
            return f"{mant}e{int(exp)}"
        return s if "." in s else s + ".0"
    if isinstance(v, complex):
        return f"({julia_number(v.real)} + {julia_number(v.imag)}im)"
    raise Untranslatable(f"unsupported literal {v!r}")


def julia_string(s: str) -> str:
    """Render a Python string as a Julia string, escaping Julia's specials."""
    out = s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return f'"{out}"'


class ExprTranslator:
    def __init__(self, ctx: Context, source_lines: list[str] | None = None):
        self.ctx = ctx
        self.source_lines = source_lines or []
        #: set when an emitted expression needs `using Printf`
        self.needs_printf = False

    # -- entry point ------------------------------------------------------

    def translate(self, node: ast.expr) -> str:
        code, _ = self._tr(node)
        return code

    def _tr(self, node: ast.expr) -> tuple[str, int]:
        """Return (julia_code, precedence)."""
        m = getattr(self, "_v_" + type(node).__name__, None)
        if m is None:
            raise Untranslatable(
                f"unsupported Python expression `{type(node).__name__}`", node
            )
        return m(node)

    def _paren(self, node: ast.expr, min_prec: int) -> str:
        code, prec = self._tr(node)
        return f"({code})" if prec < min_prec else code

    # -- literals ---------------------------------------------------------

    def _v_Constant(self, n: ast.Constant) -> tuple[str, int]:
        if n.value is None:
            return "nothing", 100
        if isinstance(n.value, str):
            return julia_string(n.value), 100
        if isinstance(n.value, bool):
            return ("true" if n.value else "false"), 100
        v = julia_number(n.value)
        # a bare negative literal still needs parens in `x^-2`
        return v, (100 if not v.startswith("-") else _PREC["unary"])

    def _v_Name(self, n: ast.Name) -> tuple[str, int]:
        name = n.id
        if name in self.ctx.locals or name in self.ctx.helpers:
            return name, 100
        if name in mapping.CONSTANTS:
            return mapping.CONSTANTS[name], 100
        if name in mapping.SCHEMES:
            spec = mapping.SCHEMES[name]
            return (spec.ctor if spec.ctor.endswith(")") else spec.ctor + "()"), 100
        if name in mapping.MATH_CONSTANTS:
            return mapping.MATH_CONSTANTS[name], 100
        if name in mapping.GEOMETRY and name == "spherical":
            return mapping.GEOMETRY[name].emit, 100
        if name in mapping.SYMMETRIZE:
            return mapping.SYMMETRIZE[name], 100
        if name in mapping.VISCO_LAW_TYPE:
            return mapping.VISCO_LAW_TYPE[name], 100
        if name in mapping.INTERFACE_TYPE:
            return mapping.INTERFACE_TYPE[name], 100
        if name in mapping.ESHELBY_ALGO:
            return mapping.ESHELBY_ALGO[name], 100
        if name in mapping.REFUSED:
            r = mapping.REFUSED[name]
            raise Untranslatable(f"`{name}`: {r.reason}", n, r.suggestion)
        return name, 100

    def _v_Tuple(self, n: ast.Tuple) -> tuple[str, int]:
        parts = [self.translate(e) for e in n.elts]
        if len(parts) == 1:
            return f"({parts[0]},)", 100
        return "(" + ", ".join(parts) + ")", 100

    def _v_List(self, n: ast.List) -> tuple[str, int]:
        return "[" + ", ".join(self.translate(e) for e in n.elts) + "]", 100

    def _v_Dict(self, n: ast.Dict) -> tuple[str, int]:
        pairs = []
        for k, v in zip(n.keys, n.values):
            if k is None:
                raise Untranslatable("dict unpacking `**` is not supported", n)
            pairs.append(f"{self.translate(k)} => {self.translate(v)}")
        return "Dict(" + ", ".join(pairs) + ")", 100

    # -- operators --------------------------------------------------------

    def _v_BinOp(self, n: ast.BinOp) -> tuple[str, int]:
        op_t = type(n.op)
        if op_t is ast.Mod and isinstance(n.left, ast.Constant) and isinstance(
            n.left.value, str
        ):
            # printf-style formatting as a *value*: Julia's @sprintf takes the
            # same format string, so this is a direct rewrite.
            fmt = julia_string(n.left.value)
            vals = n.right.elts if isinstance(n.right, ast.Tuple) else [n.right]
            rendered = ", ".join(self.translate(v) for v in vals)
            self.needs_printf = True
            return f"@sprintf({fmt}, {rendered})", 90
        if op_t not in _BINOP:
            raise Untranslatable(f"unsupported operator {op_t.__name__}", n)
        op = _BINOP[op_t]
        prec = _PREC[op]
        if op == "^":
            # right-associative, and the exponent binds loosely in Julia
            left = self._paren(n.left, prec + 1)
            right = self._paren(n.right, prec)
            return f"{left}^{right}", prec
        left = self._paren(n.left, prec)
        right = self._paren(n.right, prec + 1)
        return f"{left} {op} {right}", prec

    def _v_UnaryOp(self, n: ast.UnaryOp) -> tuple[str, int]:
        if isinstance(n.op, ast.USub):
            return f"-{self._paren(n.operand, _PREC['unary'])}", _PREC["unary"]
        if isinstance(n.op, ast.UAdd):
            return self._tr(n.operand)
        if isinstance(n.op, ast.Not):
            return f"!{self._paren(n.operand, _PREC['unary'])}", _PREC["unary"]
        if isinstance(n.op, ast.Invert):
            return f"~{self._paren(n.operand, _PREC['unary'])}", _PREC["unary"]
        raise Untranslatable("unsupported unary operator", n)

    def _v_BoolOp(self, n: ast.BoolOp) -> tuple[str, int]:
        op = "&&" if isinstance(n.op, ast.And) else "||"
        prec = _PREC[op]
        parts = [self._paren(v, prec + 1) for v in n.values]
        return f" {op} ".join(parts), prec

    def _v_Compare(self, n: ast.Compare) -> tuple[str, int]:
        # Julia supports chained comparison with the same semantics as Python.
        for op in n.ops:
            if type(op) not in _CMPOP:
                if isinstance(op, (ast.In, ast.NotIn)):
                    if len(n.ops) != 1:
                        raise Untranslatable("chained `in` comparison", n)
                    left = self.translate(n.left)
                    right = self.translate(n.comparators[0])
                    code = f"{left} in {right}"
                    if isinstance(op, ast.NotIn):
                        code = f"!({code})"
                    return code, _PREC["=="]
                if isinstance(op, (ast.Is, ast.IsNot)):
                    if len(n.ops) != 1:
                        raise Untranslatable("chained `is` comparison", n)
                    left = self.translate(n.left)
                    right = self.translate(n.comparators[0])
                    sym = "===" if isinstance(op, ast.Is) else "!=="
                    return f"{left} {sym} {right}", _PREC["=="]
                raise Untranslatable(
                    f"unsupported comparison {type(op).__name__}", n
                )
        prec = _PREC["=="]
        out = self._paren(n.left, prec + 1)
        for op, comp in zip(n.ops, n.comparators):
            out += f" {_CMPOP[type(op)]} {self._paren(comp, prec + 1)}"
        return out, prec

    def _v_IfExp(self, n: ast.IfExp) -> tuple[str, int]:
        test = self._paren(n.test, 1)
        body = self._paren(n.body, 1)
        orelse = self._paren(n.orelse, 1)
        return f"{test} ? {body} : {orelse}", 0

    # -- indexing ---------------------------------------------------------

    def _v_Subscript(self, n: ast.Subscript) -> tuple[str, int]:
        base = self._paren(n.value, 90)
        sl = n.slice

        # Dict-style phase lookup `ver["PORE"]` is handled by the caller; if it
        # reaches here the base is an ordinary dict.
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return f"{base}[{julia_string(sl.value)}]", 90

        if isinstance(sl, ast.Tuple):
            idx = ", ".join(self._index(e) for e in sl.elts)
            return f"{base}[{idx}]", 90

        return f"{base}[{self._index(sl)}]", 90

    def _index(self, node: ast.expr) -> str:
        """Translate one subscript index, applying the 0->1 base shift.

        Decidable cases only:
          * integer literal  -> +1, or `end`-relative when negative
          * a loop variable known to have been emitted over a 1-based range
            -> left alone (see `Context.unshifted_indices`)
          * anything else    -> an explicit `+ 1`, which is always correct
        """
        if isinstance(node, ast.Slice):
            return self._slice(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            v = node.value
            if v >= 0:
                return str(v + 1)
            return f"end{v + 1}" if v < -1 else "end"
        if isinstance(node, ast.Name) and node.id in self.ctx.unshifted_indices:
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self.translate(node.operand)
            return f"end - {inner} + 1"
        code = self._paren(node, _PREC["+"] + 1)
        return f"{code} + 1"

    def _slice(self, node: ast.Slice) -> str:
        if node.step is not None:
            step = self.translate(node.step)
            lo = self._index(node.lower) if node.lower else "1"
            hi = self.translate(node.upper) if node.upper else "end"
            return f"{lo}:{step}:{hi}"
        lo = self._index(node.lower) if node.lower is not None else "1"
        # Python's upper bound is exclusive and 0-based; Julia's is inclusive
        # and 1-based, so the two corrections cancel exactly.
        hi = self.translate(node.upper) if node.upper is not None else "end"
        return f"{lo}:{hi}"

    # -- attributes -------------------------------------------------------

    def _v_Attribute(self, n: ast.Attribute) -> tuple[str, int]:
        # phase accessors (`ver["PORE"].eE`) need the enclosing RVE, so the
        # caller resolves them
        if self.ctx.on_attribute is not None:
            hit = self.ctx.on_attribute(n)
            if hit is not None:
                return hit, 100

        # numpy submodule functions: linalg.inv etc.
        dotted = self._dotted(n)
        if dotted and dotted in mapping.NUMPY_ATTR:
            return mapping.NUMPY_ATTR[dotted], 100
        if dotted and dotted.split(".")[0] in ("np", "numpy", "math"):
            tail = dotted.split(".")[-1]
            if tail in mapping.NUMPY:
                t = mapping.NUMPY[tail]
                if "{" not in t:
                    return t, 100
            if tail in mapping.MATH_CONSTANTS:
                return mapping.MATH_CONSTANTS[tail], 100

        base = self._paren(n.value, 90)
        attr = n.attr

        # numpy array attributes with a direct Julia spelling
        if attr in _ARRAY_ATTRS:
            return _ARRAY_ATTRS[attr].format(base), 90

        if attr in mapping.TENSOR_ATTRS:
            return mapping.TENSOR_ATTRS[attr].format(base), 90
        if attr in mapping.PHASE_ACCESSORS:
            raise Untranslatable(
                f"phase accessor `.{attr}` outside a recognized RVE context",
                n,
                "MFH computes concentration tensors on demand; the emitter "
                "needs to know the RVE and reference medium",
            )
        raise Untranslatable(f"unsupported attribute `.{attr}`", n)

    def _dotted(self, n: ast.AST) -> str | None:
        parts = []
        cur = n
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

    # -- calls ------------------------------------------------------------

    def _v_Call(self, n: ast.Call) -> tuple[str, int]:
        # give the caller first refusal on Echoes constructs
        if self.ctx.on_echoes is not None:
            fname = self._callee_name(n)
            if fname:
                hit = self.ctx.on_echoes(n, fname)
                if hit is not None:
                    return hit, 100

        fname = self._callee_name(n)
        if fname is None:
            # calling an expression, e.g. `f[i](x)`
            callee = self._paren(n.func, 90)
            args = ", ".join(self.translate(a) for a in n.args)
            return f"{callee}({args})", 90

        if fname in mapping.REFUSED:
            r = mapping.REFUSED[fname]
            raise Untranslatable(f"`{fname}`: {r.reason}", n, r.suggestion)

        # `.dot(x)` -> matrix product
        if isinstance(n.func, ast.Attribute) and n.func.attr == "dot":
            left = self._paren(n.func.value, _PREC["*"])
            if len(n.args) != 1:
                raise Untranslatable("`.dot` with unexpected arity", n)
            right = self._paren(n.args[0], _PREC["*"] + 1)
            return f"{left} * {right}", _PREC["*"]

        # zero-argument numpy/list methods
        if (
            isinstance(n.func, ast.Attribute)
            and n.func.attr in _NOOP_METHODS
            and not n.args
        ):
            base = self._paren(n.func.value, 90)
            return _NOOP_METHODS[n.func.attr].format(base), 90

        if fname in ("time.time", "time.perf_counter"):
            return "time()", 90

        # list/array `.append(x)` is a statement, not an expression
        if isinstance(n.func, ast.Attribute) and n.func.attr == "append":
            raise Untranslatable(
                "`.append` used as an expression", n, "handled at statement level"
            )

        args = [self.translate(a) for a in n.args]
        kwargs = {
            kw.arg: self.translate(kw.value) for kw in n.keywords if kw.arg is not None
        }

        return self._call_named(n, fname, args, kwargs), 90

    def _call_named(
        self, n: ast.Call, fname: str, args: list[str], kwargs: dict[str, str]
    ) -> str:
        # Echoes' `...c` family is the complex-scalar instantiation of the same
        # C++ templates. Julia's constructors are generic over the scalar type,
        # so the suffix simply drops.
        short, _ = mapping.strip_complex(fname.split(".")[-1])

        if fname in mapping.NUMPY_ATTR:
            return f"{mapping.NUMPY_ATTR[fname]}({', '.join(args)})"

        if short in mapping.NUMPY and (
            fname == short or fname.split(".")[0] in ("np", "numpy", "math")
        ):
            tmpl = mapping.NUMPY[short]
            return self._apply_template(short, tmpl, args, kwargs, n)

        if short in mapping.TENSOR_BUILDERS:
            return self._apply_template(
                short, mapping.TENSOR_BUILDERS[short], args, kwargs, n
            )
        if short in mapping.SCALAR_CONVERSIONS:
            return self._apply_template(
                short, mapping.SCALAR_CONVERSIONS[short], args, kwargs, n
            )
        if short in mapping.GEOMETRY:
            return self._geometry(short, args, kwargs, n)
        if short in mapping.ROTATIONAL_AVERAGE:
            return f"{mapping.ROTATIONAL_AVERAGE[short]}({', '.join(args)})"
        if short in mapping.FUNCTIONS:
            return f"{mapping.FUNCTIONS[short]}({', '.join(args)})"

        if short in mapping.KM_FUNCTIONS:
            return f"{mapping.KM_FUNCTIONS[short]}({', '.join(args)})"

        # `visco_law(f)` -> ViscoLaw(f, mode). Echoes defaults to relaxation.
        if short == "visco_law":
            mode = ":relaxation"
            for kw in n.keywords:
                if kw.arg in ("visco_law_type", "type") and isinstance(
                    kw.value, ast.Name
                ):
                    mode = mapping.VISCO_LAW_TYPE.get(kw.value.id, mode)
            if len(n.args) >= 2 and isinstance(n.args[1], ast.Name):
                mode = mapping.VISCO_LAW_TYPE.get(n.args[1].id, mode)
            return f"ViscoLaw({args[0]}, {mode})"

        if short in mapping.VISCO_FUNCTIONS:
            return f"{mapping.VISCO_FUNCTIONS[short]}({', '.join(args)})"

        # `law.mat(T)` / `law.inv_mat()` -- evaluating a visco law over a time
        # series and inverting the resulting Volterra block matrix.
        if short in mapping.VISCO_METHODS and isinstance(n.func, ast.Attribute):
            base = self.translate(n.func.value)
            fn = mapping.VISCO_METHODS[short]
            inner = ", ".join([base] + args)
            return f"{fn}({inner})"

        # `visco_paramsym(M, ISO)` -> the block-parameter extractor
        if short == "visco_paramsym":
            sym = "ISO"
            if len(n.args) >= 2 and isinstance(n.args[1], ast.Name):
                sym = n.args[1].id
            for kw in n.keywords:
                if kw.arg == "sym" and isinstance(kw.value, ast.Name):
                    sym = kw.value.id
            if sym not in mapping.VISCO_PARAMSYM:
                raise Untranslatable(
                    f"visco_paramsym({sym}) has no MFH block extractor", n
                )
            return f"{mapping.VISCO_PARAMSYM[sym]}({args[0]})"

        # `.paramsym(ISO)` -> best_fit_iso(...)
        if short == "paramsym" and isinstance(n.func, ast.Attribute):
            base = self.translate(n.func.value)
            sym = "ISO"
            if n.args and isinstance(n.args[0], ast.Name):
                sym = n.args[0].id
            for kw in n.keywords:
                if kw.arg == "sym" and isinstance(kw.value, ast.Name):
                    sym = kw.value.id
            if sym not in mapping.PARAMSYM:
                raise Untranslatable(f"paramsym({sym}) has no MFH projection", n)
            return f"{mapping.PARAMSYM[sym]}({base})"

        # Python builtins that survive unchanged
        if short in ("len",):
            return f"length({', '.join(args)})"
        if short in ("abs", "max", "min", "sum", "round", "float", "int", "str"):
            builtin = {
                "float": "Float64",
                "int": "Int",
                "str": "string",
                "round": "round",
            }.get(short, short)
            return f"{builtin}({', '.join(args)})"
        if short == "range":
            return self._range(args)
        if short == "enumerate":
            return f"enumerate({', '.join(args)})"
        if short == "zip":
            return f"zip({', '.join(args)})"
        if short == "print":
            raise Untranslatable("`print` used as an expression", n)

        if short in self.ctx.helpers or short in self.ctx.locals:
            all_args = list(args) + [f"{k} = {v}" for k, v in kwargs.items()]
            return f"{short}({', '.join(all_args)})"

        raise Untranslatable(f"unmapped function `{fname}`", n)

    def _apply_template(
        self,
        name: str,
        tmpl: str,
        args: list[str],
        kwargs: dict[str, str],
        n: ast.Call,
    ) -> str:
        if "{" not in tmpl:
            all_args = list(args) + [f"{k} = {v}" for k, v in kwargs.items()]
            return f"{tmpl}({', '.join(all_args)})"
        needed = tmpl.count("{")
        if len(args) < needed:
            raise Untranslatable(
                f"`{name}` needs {needed} positional arguments, got {len(args)}", n
            )
        # Operator-shaped templates splice their arguments into an expression,
        # so a compound argument must keep its own grouping: `pow(f, 1./3.)`
        # is `f^(1.0 / 3.0)`, not `f^1.0 / 3.0`.
        if any(op in tmpl for op in "^*/+-"):
            args = [_atom(a) for a in args]
        return tmpl.format(*args)

    def _geometry(
        self, name: str, args: list[str], kwargs: dict[str, str], n: ast.Call
    ) -> str:
        spec = mapping.GEOMETRY[name]
        angle_start = mapping.GEOMETRY_ANGLE_START.get(name)
        if name == "spherical":
            return spec.emit
        # a single list argument: ellipsoidal([a,b,c,th,phi,psi])
        if len(args) == 1 and args[0].startswith("["):
            inner = args[0][1:-1].split(", ")
            args = inner
        if angle_start is not None and len(args) > angle_start:
            shape_args = args[:angle_start]
            angles = args[angle_start:]
            base = spec.emit.format(*shape_args)
            ang = ", ".join(angles)
            return base[:-1] + f"; euler_angles = ({ang},))" if len(
                angles
            ) == 1 else base[:-1] + f"; euler_angles = ({ang}))"
        if "angles" in kwargs:
            base = spec.emit.format(*args)
            return base[:-1] + f"; euler_angles = Tuple({kwargs['angles']}))"
        needed = spec.emit.count("{")
        if len(args) < needed:
            raise Untranslatable(
                f"`{name}` needs {needed} arguments, got {len(args)}", n
            )
        return spec.emit.format(*args)

    def _range(self, args: list[str]) -> str:
        if len(args) == 1:
            return f"0:({args[0]} - 1)"
        if len(args) == 2:
            return f"{args[0]}:({args[1]} - 1)"
        return f"{args[0]}:{args[2]}:({args[1]} - 1)"

    def _callee_name(self, n: ast.Call) -> str | None:
        if isinstance(n.func, ast.Name):
            return n.func.id
        if isinstance(n.func, ast.Attribute):
            return self._dotted(n.func) or n.func.attr
        return None

    # -- comprehensions and lambdas ---------------------------------------

    def _v_ListComp(self, n: ast.ListComp) -> tuple[str, int]:
        return self._comp(n, n.elt), 100

    def _v_GeneratorExp(self, n: ast.GeneratorExp) -> tuple[str, int]:
        return self._comp(n, n.elt), 100

    def _comp(self, n: ast.expr, elt: ast.expr) -> str:
        gens = n.generators  # type: ignore[attr-defined]
        saved = set(self.ctx.locals)
        clauses = []
        for g in gens:
            if g.is_async:
                raise Untranslatable("async comprehension", n)
            tgt = self._target_names(g.target)
            self.ctx.locals |= set(tgt)
            tgt_s = tgt[0] if len(tgt) == 1 else "(" + ", ".join(tgt) + ")"
            clauses.append(f"{tgt_s} in {self.translate(g.iter)}")
            for cond in g.ifs:
                clauses.append(f"if {self.translate(cond)}")
        body = self.translate(elt)
        self.ctx.locals = saved
        return "[" + body + " for " + ", ".join(clauses) + "]"

    def _v_DictComp(self, n: ast.DictComp) -> tuple[str, int]:
        saved = set(self.ctx.locals)
        clauses = []
        for g in n.generators:
            tgt = self._target_names(g.target)
            self.ctx.locals |= set(tgt)
            tgt_s = tgt[0] if len(tgt) == 1 else "(" + ", ".join(tgt) + ")"
            clauses.append(f"{tgt_s} in {self.translate(g.iter)}")
            for cond in g.ifs:
                clauses.append(f"if {self.translate(cond)}")
        pair = f"{self.translate(n.key)} => {self.translate(n.value)}"
        self.ctx.locals = saved
        return "Dict(" + pair + " for " + ", ".join(clauses) + ")", 100

    def _v_Lambda(self, n: ast.Lambda) -> tuple[str, int]:
        names = [a.arg for a in n.args.args]
        saved = set(self.ctx.locals)
        self.ctx.locals |= set(names)
        body = self.translate(n.body)
        self.ctx.locals = saved
        params = names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"
        return f"{params} -> {body}", 0

    def _v_Starred(self, n: ast.Starred) -> tuple[str, int]:
        return f"{self.translate(n.value)}...", 100

    def _v_JoinedStr(self, n: ast.JoinedStr) -> tuple[str, int]:
        out = []
        for part in n.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(
                    part.value.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("$", "\\$")
                )
            elif isinstance(part, ast.FormattedValue):
                if part.format_spec is not None:
                    raise Untranslatable(
                        "f-string format spec; use @printf", n,
                        "the emitter converts print() format specs separately",
                    )
                out.append("$(" + self.translate(part.value) + ")")
            else:
                raise Untranslatable("unsupported f-string part", n)
        return '"' + "".join(out) + '"', 100

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _target_names(t: ast.expr) -> list[str]:
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)):
            out: list[str] = []
            for e in t.elts:
                out.extend(ExprTranslator._target_names(e))
            return out
        raise Untranslatable("unsupported assignment target")


def _atom(code: str) -> str:
    """Parenthesize `code` unless it is already a single indivisible term."""
    s = code.strip()
    if not s:
        return s
    if s.isidentifier() or s.replace(".", "", 1).replace("e", "", 1).lstrip(
        "-"
    ).isdigit():
        return s
    # already fully wrapped, e.g. `(a + b)` or `f(x, y)`
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                break
        elif depth == 0 and ch in "+-*/^ ":
            break
    else:
        if s[0] in "([" and depth == 0:
            return s
    if not any(ch in s for ch in "+-*/^ "):
        return s
    return f"({s})"


def loop_var_is_index_only(body: list[ast.stmt], var: str) -> bool:
    """True when `var` appears exclusively as a subscript index in `body`.

    When that holds, the loop range can be emitted 1-based and every
    subscript left unshifted, which is both correct and idiomatic. When it
    does not (the variable is also used as a value, e.g. `2*i`), the range
    stays 0-based and each subscript gets an explicit `+ 1`.
    """
    uses_as_index = 0
    uses_total = 0
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == var and isinstance(
                node.ctx, ast.Load
            ):
                uses_total += 1
        for node in ast.walk(stmt):
            if isinstance(node, ast.Subscript):
                sl = node.slice
                targets = sl.elts if isinstance(sl, ast.Tuple) else [sl]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == var:
                        uses_as_index += 1
    return uses_total > 0 and uses_as_index == uses_total
