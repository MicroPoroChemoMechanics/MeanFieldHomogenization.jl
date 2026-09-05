"""Python AST -> the IR in `model.py`.

The extractor recognizes the constructs that carry MFH meaning and records
them structurally; everything else it hands to `expr.ExprTranslator`. The
important structural job is the **RVE lift**: Echoes mutates a module-level
RVE in place, while MFH's RVE is immutable (`set_param` returns a new one and
the matrix amount is derived, not settable). So every RVE becomes a builder
function parameterized by whatever the script varies -- exactly the shape the
hand-written MFH demos use.
"""

from __future__ import annotations

import ast
from typing import Optional

from . import mapping
from .expr import Context, ExprTranslator, Untranslatable, julia_string
from .model import (
    Assign,
    For,
    Helper,
    HomogenizeCall,
    If,
    JuliaExpr,
    JuliaStmt,
    Param,
    PhaseAccess,
    PhaseDef,
    PlotCall,
    PrintCall,
    Return,
    RVEDef,
    Script,
    Stmt,
    Sweep,
    TryCatch,
    Untranslated,
    While,
)

PHASE_CTORS = {
    "ellipsoid": "ellipsoid",
    "crack": "crack",
    "sphere_nlayers": "sphere_nlayers",
    "spheroid_nlayers": "spheroid_nlayers",
    "inclusion_generic_ellipsoid": "ellipsoid",
}

def _is_string_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    # an f-string is a string too
    return isinstance(node, ast.JoinedStr)


HOMOGENIZE_FNS = {
    "homogenize": "elastic",
    "homogenize_visco": "alv",
    "homogenize_derivative": "derivative",
}


class Extractor:
    def __init__(self, tree: ast.Module, source: str, path: str):
        self.tree = tree
        self.source = source
        self.lines = source.splitlines()
        self.path = path
        self.script = Script(source_path=path)
        self.ctx = Context()
        self.tr = ExprTranslator(self.ctx, self.lines)
        self.ctx.on_echoes = self._on_echoes
        self.ctx.on_attribute = self._on_attribute
        #: rve variable -> RVEDef
        self.rve_map: dict[str, RVEDef] = {}
        #: names bound to a homogenize result, so `.k` resolves correctly
        self._plot_axes = "p"
        self._n_figs = 0
        self._new_axes: set[str] = set()
        self._subplot_used = False

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def run(self) -> Script:
        self._collect_imports()
        self.script.header = self._leading_comment()
        body = self._block(self.tree.body, top_level=True)
        self.script.body = body
        self.script.declared_axes = set(self._new_axes)
        self.script.used_subplots = self._subplot_used
        if self.tr.needs_printf:
            self.script.imports.add("Printf")
        return self.script

    def _collect_imports(self) -> None:
        for node in self.tree.body:
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in mapping.OUT_OF_SCOPE_IMPORTS:
                    self._refuse(
                        node,
                        f"module `{m}` is out of scope",
                        mapping.OUT_OF_SCOPE_IMPORTS[m],
                        severity="review",
                        symbol=m,
                    )
                if m in ("matplotlib", "pylab"):
                    self.script.needs_plots = True
                if m == "numpy":
                    self.script.imports.add("LinearAlgebra")

    def _leading_comment(self) -> str:
        out = []
        for line in self.lines:
            s = line.strip()
            if s.startswith("#!"):
                continue
            if s.startswith("#"):
                out.append(s.lstrip("#").strip())
            elif not s:
                if out:
                    break
            else:
                break
        return "\n".join(out)

    # ------------------------------------------------------------------
    # statement dispatch
    # ------------------------------------------------------------------

    def _block(self, stmts: list[ast.stmt], top_level: bool = False) -> list[Stmt]:
        out: list[Stmt] = []
        i = 0
        while i < len(stmts):
            node = stmts[i]
            # try the multi-statement sweep idiom first
            consumed, sweep = self._try_sweep(stmts, i)
            if sweep is not None:
                out.append(sweep)
                i += consumed
                continue
            try:
                res = self._stmt(node, top_level=top_level)
            except Untranslatable as e:
                res = [self._refusal_node(node, e)]
            if res:
                out.extend(res)
            i += 1
        return out

    def _stmt(self, node: ast.stmt, top_level: bool = False) -> list[Stmt]:
        m = getattr(self, "_s_" + type(node).__name__, None)
        if m is None:
            raise Untranslatable(
                f"unsupported statement `{type(node).__name__}`", node
            )
        return m(node, top_level)

    # -- trivia ---------------------------------------------------------

    def _s_Import(self, node, top_level) -> list[Stmt]:
        return []

    def _s_ImportFrom(self, node, top_level) -> list[Stmt]:
        return []

    def _s_Pass(self, node, top_level) -> list[Stmt]:
        return []

    def _s_Global(self, node, top_level) -> list[Stmt]:
        # `global ver` exists only to permit in-place mutation of a module
        # RVE, which the lift removes entirely.
        return []

    def _s_Break(self, node, top_level) -> list[Stmt]:
        return [JuliaStmt(code="break", lineno=node.lineno)]

    def _s_Continue(self, node, top_level) -> list[Stmt]:
        return [JuliaStmt(code="continue", lineno=node.lineno)]

    # -- expressions used as statements ---------------------------------

    def _s_Expr(self, node: ast.Expr, top_level) -> list[Stmt]:
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return []  # docstring
        if isinstance(v, ast.Call):
            name = self._callee(v)
            if name in ("exit", "sys.exit", "quit"):
                return [
                    JuliaStmt(
                        code="# NOTE: original script called exit() here",
                        lineno=node.lineno,
                    )
                ]
            if name == "print":
                return [self._print(v)]
            if name and name.startswith(("plt.", "pylab.", "pyplot.")):
                return self._plot(v)
            if name == "set_printoptions":
                return []
            if name and name.endswith(".append"):
                return [self._append(v)]
            # ver["SPN"].set_radius(0, r) -- a method on a phase, not the RVE
            if isinstance(v.func, ast.Attribute) and isinstance(
                v.func.value, ast.Subscript
            ):
                sub = v.func.value
                if (
                    isinstance(sub.value, ast.Name)
                    and sub.value.id in self.rve_map
                    and isinstance(sub.slice, ast.Constant)
                ):
                    if v.func.attr == "set_radius":
                        return self._set_radius(v, sub.value.id, str(sub.slice.value))
                    raise Untranslatable(
                        f"unsupported phase method `.{v.func.attr}`", v
                    )
            if name and (
                name.endswith(".set_prop")
                or name.endswith(".set_ref")
                or name.endswith(".set_param_eshelby")
                or name.endswith(".set_radius")
                or name.endswith(".set_interf_prop")
            ):
                return self._rve_setter(v)
        code = self.tr.translate(v)
        return [JuliaStmt(code=code, lineno=node.lineno, py_src=self._src(node))]

    # -- assignment ------------------------------------------------------

    def _s_Assign(self, node: ast.Assign, top_level) -> list[Stmt]:
        if len(node.targets) != 1:
            raise Untranslatable("chained assignment `a = b = ...`", node)
        tgt = node.targets[0]

        # ver = rve(matrix="SOLID")  /  verc = rvec(matrix="SOLID")
        if isinstance(node.value, ast.Call):
            base, is_cplx = mapping.strip_complex(self._callee(node.value) or "")
            if base == "rve":
                return self._new_rve(tgt, node.value, node, complex_valued=is_cplx)

        # ver["PORE"] = ellipsoid(...)
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            base = tgt.value.id
            if base in self.rve_map and isinstance(tgt.slice, ast.Constant):
                return self._add_phase(base, str(tgt.slice.value), node.value, node)

        # ver["PORE"].fraction = f   /  .density = d
        if isinstance(tgt, ast.Attribute):
            return self._attr_assign(tgt, node.value, node)

        # indexed assignment `a[i] = x` / `a[i, j] = x`
        if isinstance(tgt, ast.Subscript):
            lhs = self.tr.translate(tgt)
            rhs = self.tr.translate(node.value)
            return [
                JuliaStmt(
                    code=f"{lhs} = {rhs}",
                    lineno=node.lineno,
                    py_src=self._src(node),
                )
            ]

        # ordinary binding
        names = self._targets(tgt)
        value = self.tr.translate(node.value)
        # a homogenize result bound to a name becomes the self-consistent
        # reference medium for any localization read later in this scope
        if len(names) == 1 and "homogenize(" in value:
            self._current_result = names[0]
        for n_ in names:
            self.ctx.locals.add(n_)
        if top_level and len(names) == 1 and self._is_constantish(node.value):
            p = Param(
                name=names[0],
                value=JuliaExpr(value),
                lineno=node.lineno,
                py_src=self._src(node),
            )
            self.script.params.append(p)
            return []
        return [
            Assign(
                targets=names,
                value=JuliaExpr(value),
                lineno=node.lineno,
                py_src=self._src(node),
            )
        ]

    def _s_AugAssign(self, node: ast.AugAssign, top_level) -> list[Stmt]:
        from .expr import _BINOP

        op = _BINOP.get(type(node.op))
        if op is None:
            raise Untranslatable("unsupported augmented assignment", node)
        tgt = self.tr.translate(node.target)
        val = self.tr.translate(node.value)
        return [
            JuliaStmt(
                code=f"{tgt} {op}= {val}", lineno=node.lineno, py_src=self._src(node)
            )
        ]

    def _is_constantish(self, node: ast.expr) -> bool:
        """True for module-level definitions worth emitting as `const`."""
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.UnaryOp, ast.BinOp)):
            return all(
                self._is_constantish(c)
                for c in ast.iter_child_nodes(node)
                if isinstance(c, ast.expr)
            )
        if isinstance(node, ast.Call):
            name = self._callee(node) or ""
            short = name.split(".")[-1]
            return short in mapping.TENSOR_BUILDERS or short in mapping.CONSTANTS
        if isinstance(node, ast.Name):
            return node.id in mapping.CONSTANTS
        return False

    # -- RVE construction ------------------------------------------------

    def _new_rve(
        self,
        tgt: ast.expr,
        call: ast.Call,
        node: ast.stmt,
        complex_valued: bool = False,
    ) -> list[Stmt]:
        if not isinstance(tgt, ast.Name):
            raise Untranslatable("`rve(...)` bound to a non-name target", node)
        matrix = ""
        for kw in call.keywords:
            if kw.arg == "matrix" and isinstance(kw.value, ast.Constant):
                matrix = str(kw.value.value)
        # The same variable name can hold structurally different RVEs in
        # different function scopes -- `bfhf.py` builds one `ver` in Chom and
        # a different one in Cu. Qualifying the builder by scope keeps them
        # apart instead of letting the second silently overwrite the first.
        builder = f"build_{tgt.id}"
        if self._scope:
            builder = f"build_{self._scope}_{tgt.id}"
        rve = RVEDef(
            var=tgt.id,
            matrix_name=matrix,
            builder_name=builder,
            complex_valued=complex_valued,
            lineno=node.lineno,
            py_src=self._src(node),
        )
        self.rve_map[tgt.id] = rve
        self.script.rves.append(rve)
        self.ctx.rve_vars.add(tgt.id)
        # properties passed straight to rve(...)
        for kw in call.keywords:
            if kw.arg == "prop" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant):
                        rve.reference[f":{k.value}"] = JuliaExpr(self.tr.translate(v))
        return []

    def _add_phase(
        self, rve_var: str, phase: str, value: ast.expr, node: ast.stmt
    ) -> list[Stmt]:
        rve = self.rve_map[rve_var]
        if not isinstance(value, ast.Call):
            raise Untranslatable("phase assigned a non-constructor value", node)
        ctor = self._callee(value) or ""
        short, _is_cplx = mapping.strip_complex(ctor.split(".")[-1])
        if short not in PHASE_CTORS:
            raise Untranslatable(
                f"unsupported inclusion constructor `{ctor}`",
                node,
                "user_inclusion subclasses map to CustomInclusion; see "
                "scripts/80_custom_inclusion_contract.jl",
            )
        kind = PHASE_CTORS[short]
        pd = PhaseDef(
            name=phase,
            kind="crack" if short == "crack" else kind,
            is_matrix=(phase == rve.matrix_name),
            lineno=node.lineno,
            py_src=self._src(node),
        )
        kw = {k.arg: k.value for k in value.keywords if k.arg}

        # geometry
        if "shape" in kw:
            pd.geometry = JuliaExpr(self.tr.translate(kw["shape"]))
        if short == "crack":
            pd.geometry = self._crack_geometry(kw.get("shape"))

        # layered inclusions
        if short == "sphere_nlayers":
            self._layered_sphere(pd, kw, node)
        elif short == "spheroid_nlayers":
            self._layered_spheroid(pd, kw, node)
        else:
            if "prop" in kw and isinstance(kw["prop"], ast.Dict):
                for k, v in zip(kw["prop"].keys, kw["prop"].values):
                    if isinstance(k, ast.Constant):
                        pd.props[f":{k.value}"] = JuliaExpr(self.tr.translate(v))

        # visco properties: prop={"C": (Js, CREEP)}
        if "visco_prop" in kw and isinstance(kw["visco_prop"], ast.Dict):
            for k, v in zip(kw["visco_prop"].keys, kw["visco_prop"].values):
                if not isinstance(k, ast.Constant):
                    continue
                mode = ":relaxation"
                law = v
                if isinstance(v, ast.Tuple) and len(v.elts) == 2:
                    law = v.elts[0]
                    if isinstance(v.elts[1], ast.Name):
                        mode = mapping.VISCO_LAW_TYPE.get(v.elts[1].id, ":relaxation")
                pd.visco_props[f":{k.value}"] = (
                    JuliaExpr(self.tr.translate(law)),
                    mode,
                )
            self.script.imports.add("MeanFieldHomogenization")

        # amount
        if not pd.is_matrix:
            if "fraction" in kw:
                pd.amount = ("fraction", JuliaExpr(self.tr.translate(kw["fraction"])))
            elif "density" in kw:
                pd.amount = ("density", JuliaExpr(self.tr.translate(kw["density"])))
            else:
                pd.amount = ("fraction", JuliaExpr("0.0"))

        # symmetrize=[ISO]
        if "symmetrize" in kw:
            sym = kw["symmetrize"]
            elts = sym.elts if isinstance(sym, (ast.List, ast.Tuple)) else [sym]
            for e in elts:
                if isinstance(e, ast.Name) and e.id in mapping.SYMMETRIZE:
                    pd.symmetrize = mapping.SYMMETRIZE[e.id]

        # interface properties
        if "interf_prop" in kw:
            self._interfaces(pd, kw["interf_prop"], node)

        # Whatever the enclosing function varies, the builder must take.
        self._note_params(rve, *value.args, *(k.value for k in value.keywords))
        rve.phases.append(pd)
        return []

    def _crack_geometry(self, shape: Optional[ast.expr]) -> JuliaExpr:
        """`crack(shape=spheroidal(w))` -> PennyCrack / EllipticCrack.

        Echoes describes a crack by the spheroid it degenerates from; MFH has
        dedicated crack types. A `spheroidal` shape is axisymmetric, hence a
        penny crack of unit radius.
        """
        if shape is None:
            return JuliaExpr("PennyCrack(1.0)")
        if isinstance(shape, ast.Call):
            name = self._callee(shape) or ""
            if name.split(".")[-1] == "spheroidal":
                return JuliaExpr("PennyCrack(1.0)")
            if name.split(".")[-1] == "ellipsoidal" and len(shape.args) >= 2:
                a = self.tr.translate(shape.args[0])
                b = self.tr.translate(shape.args[1])
                return JuliaExpr(f"EllipticCrack({a}, {b})")
        if isinstance(shape, ast.Name) and shape.id == "spherical":
            return JuliaExpr("PennyCrack(1.0)")
        return JuliaExpr("PennyCrack(1.0)")

    def _layered_sphere(self, pd: PhaseDef, kw: dict, node: ast.stmt) -> None:
        """`sphere_nlayers(radii=[...], prop={"C":[C0,C1]})` -> LayeredSphere."""
        radii: list[str] = []
        if "radii" in kw and isinstance(kw["radii"], (ast.List, ast.Tuple)):
            radii = [self.tr.translate(e) for e in kw["radii"].elts]
        elif "layer_fractions" in kw:
            pd.kind = "sphere_nlayers_fractions"
            radii = [self.tr.translate(kw["layer_fractions"])]
        elif "nb_layers" in kw:
            raise Untranslatable(
                "`sphere_nlayers(nb_layers=...)` without explicit radii", node,
                "MFH LayeredSphere takes an explicit ascending radii tuple",
            )
        # Both libraries list the outer radius of every layer, ascending, with
        # r = 0 implicit at the center. A leading 0.0 is therefore a real
        # (degenerate) core layer, not padding -- `porous.py` grows exactly
        # that layer through set_radius(0, f^(1/3)) to sweep porosity.
        moduli: list[str] = []
        if "prop" in kw and isinstance(kw["prop"], ast.Dict):
            for k, v in zip(kw["prop"].keys, kw["prop"].values):
                if not isinstance(k, ast.Constant):
                    continue
                if isinstance(v, (ast.List, ast.Tuple)):
                    moduli = [self.tr.translate(e) for e in v.elts]
                    pd.props[f":{k.value}"] = JuliaExpr("")  # filled by emitter
        pd.layers = [JuliaExpr(r) for r in radii]
        pd.props["__moduli__"] = JuliaExpr(", ".join(moduli))
        pd.geometry = JuliaExpr("")  # emitter builds LayeredSphere(...)

    def _layered_spheroid(self, pd: PhaseDef, kw: dict, node: ast.stmt) -> None:
        axis: list[str] = []
        if "small_radii" in kw and isinstance(kw["small_radii"], (ast.List, ast.Tuple)):
            axis = [self.tr.translate(e) for e in kw["small_radii"].elts]
        ar = self.tr.translate(kw["aspect_ratio"]) if "aspect_ratio" in kw else "1.0"
        nseries = self.tr.translate(kw["N"]) if "N" in kw else "5"
        pd.layers = [JuliaExpr(a) for a in axis]
        pd.props["__aspect_ratio__"] = JuliaExpr(ar)
        pd.props["__nseries__"] = JuliaExpr(nseries)
        moduli: list[str] = []
        if "prop" in kw and isinstance(kw["prop"], ast.Dict):
            for k, v in zip(kw["prop"].keys, kw["prop"].values):
                if isinstance(v, (ast.List, ast.Tuple)):
                    moduli = [self.tr.translate(e) for e in v.elts]
        pd.props["__moduli__"] = JuliaExpr(", ".join(moduli))

    def _interfaces(self, pd: PhaseDef, node_val: ast.expr, node: ast.stmt) -> None:
        """`interf_prop={"C": [[kn, kt, PRIMALDISC]]}` -> SpringInterface(...)."""
        if not isinstance(node_val, ast.Dict):
            return
        for _k, v in zip(node_val.keys, node_val.values):
            if not isinstance(v, (ast.List, ast.Tuple)):
                continue
            for entry in v.elts:
                if not isinstance(entry, (ast.List, ast.Tuple)):
                    continue
                parts = [self.tr.translate(e) for e in entry.elts]
                kind = "PerfectInterface()"
                if parts and isinstance(entry.elts[-1], ast.Name):
                    tname = entry.elts[-1].id
                    ctor = mapping.INTERFACE_TYPE.get(tname, "PerfectInterface()")
                    nums = parts[:-1]
                    kind = ctor if ctor.endswith(")") else (
                        f"{ctor}({', '.join(nums)})" if nums else f"{ctor}()"
                    )
                pd.interfaces.append(JuliaExpr(kind))

    def _rve_setter(self, call: ast.Call) -> list[Stmt]:
        name = self._callee(call) or ""
        base, _, meth = name.rpartition(".")
        rve = self.rve_map.get(base)
        if rve is None:
            raise Untranslatable(f"`{name}` on an unknown RVE", call)
        if meth == "set_ref" and len(call.args) >= 2:
            key = call.args[0]
            if isinstance(key, ast.Constant):
                rve.reference[f":{key.value}"] = JuliaExpr(
                    self.tr.translate(call.args[1])
                )
            return []
        if meth == "set_prop" and len(call.args) >= 2:
            key = call.args[0]
            if isinstance(key, ast.Constant):
                rve.reference[f":{key.value}"] = JuliaExpr(
                    self.tr.translate(call.args[1])
                )
            return []
        if meth == "set_param_eshelby":
            for kw in call.keywords:
                if kw.arg == "algo" and isinstance(kw.value, ast.Name):
                    rve.eshelby["method"] = mapping.ESHELBY_ALGO.get(
                        kw.value.id, ":auto"
                    )
                elif kw.arg in mapping.ESHELBY_KW_RENAME:
                    rve.eshelby[mapping.ESHELBY_KW_RENAME[kw.arg]] = self.tr.translate(
                        kw.value
                    )
            return []
        raise Untranslatable(f"unsupported RVE method `{meth}`", call)

    def _set_radius(self, call: ast.Call, rve_var: str, phase: str) -> list[Stmt]:
        """`spn.set_radius(i, r)` -> the i-th radius becomes a builder parameter.

        MFH's LayeredSphere is immutable, so a swept radius has to enter
        through the builder rather than be assigned after the fact.
        """
        rve = self.rve_map[rve_var]
        if not isinstance(call.args[0], ast.Constant):
            raise Untranslatable(
                "`set_radius` with a non-literal layer index", call,
                "the layer index must be statically known to rebuild the tuple",
            )
        idx = int(call.args[0].value)
        val = JuliaExpr(self.tr.translate(call.args[1]))
        for pd in rve.phases:
            if pd.name != phase:
                continue
            if idx < 0 or idx >= len(pd.layers):
                raise Untranslatable(
                    f"`set_radius({idx}, ...)` is out of range for "
                    f"{len(pd.layers)} declared layers",
                    call,
                )
            pd.layers[idx] = val
            for free in self._free_names(call.args[1]):
                if free not in rve.params:
                    rve.params.append(free)
            return []
        raise Untranslatable(f"`set_radius` on unknown phase `{phase}`", call)

    def _attr_assign(
        self, tgt: ast.Attribute, value: ast.expr, node: ast.stmt
    ) -> list[Stmt]:
        """`ver["PORE"].fraction = f` -> a builder parameter, or set_param."""
        attr = tgt.attr
        sub = tgt.value
        if (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id in self.rve_map
            and isinstance(sub.slice, ast.Constant)
        ):
            rve = self.rve_map[sub.value.id]
            phase = str(sub.slice.value)
            expr = JuliaExpr(self.tr.translate(value))
            for pd in rve.phases:
                if pd.name != phase:
                    continue
                if attr == "shape":
                    # the swept geometry becomes a builder parameter too
                    pd.geometry = expr
                    self._note_params(rve, value)
                    return []
                if attr not in ("fraction", "density"):
                    raise Untranslatable(
                        f"assignment to phase attribute `.{attr}`", node,
                        "only .fraction, .density and .shape are recognized; "
                        "anything else would change the model silently",
                    )
                if pd.is_matrix:
                    # MFH derives the matrix amount as 1 - sum(f_inclusions);
                    # setting it explicitly is both unnecessary and an error.
                    return [
                        JuliaStmt(
                            code=(
                                f"# `{phase}` is the matrix: MFH derives its "
                                f"fraction as 1 - Σ f_inclusions"
                            ),
                            lineno=node.lineno,
                        )
                    ]
                pd.amount = (attr, expr)
                self._note_params(rve, value)
                return []
        raise Untranslatable(f"unsupported attribute assignment `.{attr}`", node)

    def _free_names(self, node: ast.expr) -> list[str]:
        out = []
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id in self.ctx.locals and n.id not in [
                    p.name for p in self.script.params
                ]:
                    out.append(n.id)
        return sorted(set(out))

    # -- functions -------------------------------------------------------

    def _s_FunctionDef(self, node: ast.FunctionDef, top_level) -> list[Stmt]:
        args = [a.arg for a in node.args.args]
        saved_locals = set(self.ctx.locals)
        saved_scope = self._scope
        saved_rves = dict(self.rve_map)
        self._scope = node.name
        self._fn_args = list(args)
        self.ctx.locals |= set(args)
        self.ctx.helpers.add(node.name)
        defaults: dict[str, JuliaExpr] = {}
        n_req = len(args) - len(node.args.defaults)
        for i, d in enumerate(node.args.defaults):
            defaults[args[n_req + i]] = JuliaExpr(self.tr.translate(d))
        body = self._block(node.body)
        self.ctx.locals = saved_locals
        self._scope = saved_scope
        self._fn_args = []
        self.rve_map = saved_rves
        h = Helper(
            name=node.name,
            args=args,
            defaults=defaults,
            body=body,
            lineno=node.lineno,
            py_src=self._src(node),
        )
        self.script.helpers.append(h)
        return []

    def _s_Return(self, node: ast.Return, top_level) -> list[Stmt]:
        if node.value is None:
            return [Return(lineno=node.lineno)]
        return [
            Return(value=JuliaExpr(self.tr.translate(node.value)), lineno=node.lineno)
        ]

    def _s_ClassDef(self, node: ast.ClassDef, top_level) -> list[Stmt]:
        bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
        if "user_inclusion" in bases:
            raise Untranslatable(
                "`user_inclusion` subclass: MFH's counterpart is CustomInclusion "
                "with explicit callbacks, which is a different contract "
                "(three entry gates rather than one build_all)",
                node,
                "see scripts/80_custom_inclusion_contract.jl and "
                "check_inclusion_interface; the four tensors eE/sE/eS/sS "
                "become strain_strain_loc / stress_strain_loc / "
                "strain_stress_loc / stress_stress_loc callbacks",
            )
        raise Untranslatable(f"class `{node.name}` has no MFH counterpart", node)

    # -- control flow ----------------------------------------------------

    def _s_If(self, node: ast.If, top_level) -> list[Stmt]:
        test = JuliaExpr(self.tr.translate(node.test))
        return [
            If(
                test=test,
                body=self._block(node.body),
                orelse=self._block(node.orelse),
                lineno=node.lineno,
            )
        ]

    def _s_For(self, node: ast.For, top_level) -> list[Stmt]:
        targets = self._targets(node.target)
        saved = set(self.ctx.locals)
        saved_unshifted = set(self.ctx.unshifted_indices)
        self.ctx.locals |= set(targets)

        iter_code, unshift = self._for_iter(node)
        if unshift:
            self.ctx.unshifted_indices |= set(targets)

        body = self._block(node.body)
        self.ctx.locals = saved
        self.ctx.unshifted_indices = saved_unshifted
        return [
            For(
                targets=targets,
                iter=JuliaExpr(iter_code),
                body=body,
                lineno=node.lineno,
            )
        ]

    def _for_iter(self, node: ast.For) -> tuple[str, bool]:
        """Translate the iterable, choosing a 1-based range when it is safe.

        `for i in range(n)` becomes `for i in 1:n` when every use of `i` in
        the body is a subscript index -- then no `+1` shift is needed anywhere
        and the Julia reads naturally. Otherwise the range stays 0-based and
        subscripts get an explicit shift, which is always correct.
        """
        from .expr import loop_var_is_index_only

        it = node.iter
        if (
            isinstance(it, ast.Call)
            and self._callee(it) == "range"
            and isinstance(node.target, ast.Name)
        ):
            var = node.target.id
            if loop_var_is_index_only(node.body, var):
                args = [self.tr.translate(a) for a in it.args]
                if len(args) == 1:
                    return f"1:{args[0]}", True
                if len(args) == 2:
                    return f"({args[0]} + 1):{args[1]}", True
        return self.tr.translate(it), False

    def _s_While(self, node: ast.While, top_level) -> list[Stmt]:
        return [
            While(
                test=JuliaExpr(self.tr.translate(node.test)),
                body=self._block(node.body),
                lineno=node.lineno,
            )
        ]

    def _s_Try(self, node: ast.Try, top_level) -> list[Stmt]:
        handler: list[Stmt] = []
        for h in node.handlers:
            handler.extend(self._block(h.body))
        return [
            TryCatch(
                body=self._block(node.body),
                handler=handler,
                lineno=node.lineno,
                py_src=(
                    "Echoes raises on non-convergence; MFH's select_best "
                    "returns the best iterate instead, so this catch may "
                    "no longer fire"
                ),
            )
        ]

    # -- the sweep idiom -------------------------------------------------

    def _try_sweep(self, stmts: list[ast.stmt], i: int) -> tuple[int, Optional[Sweep]]:
        """Recognize `acc = []` ... `for x in xs: acc.append(f(x))`.

        Collapsing this into a comprehension is the single biggest readability
        win available, because it is the dominant idiom in the corpus.
        """
        accs: dict[str, ast.stmt] = {}
        j = i
        while j < len(stmts):
            s = stmts[j]
            if (
                isinstance(s, ast.Assign)
                and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)
                and isinstance(s.value, ast.List)
                and not s.value.elts
            ):
                accs[s.targets[0].id] = s
                j += 1
                continue
            break
        if not accs or j >= len(stmts):
            return 0, None
        loop = stmts[j]
        if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
            return 0, None

        # every statement in the loop must be `acc.append(expr)` or a plain
        # binding that the appends depend on
        appends: dict[str, ast.expr] = {}
        prelude: list[ast.stmt] = []
        for s in loop.body:
            if (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute)
                and s.value.func.attr == "append"
                and isinstance(s.value.func.value, ast.Name)
                and s.value.func.value.id in accs
                and len(s.value.args) == 1
            ):
                appends[s.value.func.value.id] = s.value.args[0]
            elif isinstance(s, ast.Assign):
                prelude.append(s)
            else:
                return 0, None
        if set(appends) != set(accs) or not appends:
            return 0, None

        var = loop.target.id
        saved = set(self.ctx.locals)
        self.ctx.locals.add(var)
        try:
            iter_code = self.tr.translate(loop.iter)
            # a prelude binding is inlined into the element expression only if
            # it is a single tuple-unpacking of one call, the common
            # `a,b = Chom(phi,sch)` shape
            inline: dict[str, str] = {}
            if len(prelude) == 1 and isinstance(prelude[0], ast.Assign):
                p = prelude[0]
                if isinstance(p.targets[0], ast.Tuple):
                    names = self._targets(p.targets[0])
                    self.ctx.locals |= set(names)
                    call = self.tr.translate(p.value)
                    for k, nm in enumerate(names):
                        inline[nm] = f"{call}[{k + 1}]"
                elif isinstance(p.targets[0], ast.Name):
                    self.ctx.locals.add(p.targets[0].id)
                    inline[p.targets[0].id] = self.tr.translate(p.value)
            elif prelude:
                self.ctx.locals = saved
                return 0, None

            elements: dict[str, JuliaExpr] = {}
            for name, expr in appends.items():
                code = self.tr.translate(expr)
                for k, v in inline.items():
                    if code == k:
                        code = v
                elements[name] = JuliaExpr(code)
        except Untranslatable:
            self.ctx.locals = saved
            return 0, None
        finally:
            self.ctx.locals = saved

        for name in accs:
            self.ctx.locals.add(name)
        sweep = Sweep(
            var=var,
            iterable=JuliaExpr(iter_code),
            accumulators=elements,
            comprehensible=True,
            lineno=loop.lineno,
        )
        return (j - i) + 1, sweep

    def _append(self, call: ast.Call) -> Stmt:
        assert isinstance(call.func, ast.Attribute)
        target = self.tr.translate(call.func.value)
        val = self.tr.translate(call.args[0])
        return JuliaStmt(code=f"push!({target}, {val})", lineno=call.lineno)

    # -- outputs ---------------------------------------------------------

    def _print(self, call: ast.Call) -> Stmt:
        args = []
        for a in call.args:
            # `print("x = %g" % v)` -> @printf
            if (
                isinstance(a, ast.BinOp)
                and isinstance(a.op, ast.Mod)
                and isinstance(a.left, ast.Constant)
                and isinstance(a.left.value, str)
            ):
                fmt = a.left.value
                vals = (
                    a.right.elts if isinstance(a.right, ast.Tuple) else [a.right]
                )
                self.script.imports.add("Printf")
                rendered = ", ".join(self.tr.translate(v) for v in vals)
                return PrintCall(
                    args=[JuliaExpr(rendered)],
                    fmt=julia_string(fmt + "\\n"),
                    lineno=call.lineno,
                )
            args.append(JuliaExpr(self.tr.translate(a)))
        return PrintCall(args=args, lineno=call.lineno)

    def _plot(self, call: ast.Call) -> list[Stmt]:
        name = (self._callee(call) or "").split(".")[-1]
        self.script.needs_plots = True

        if name in ("show", "clf", "cla", "close", "tight_layout", "draw"):
            return []
        if name == "figure":
            # A new matplotlib figure starts a new Plots.jl object. Naming them
            # p, p2, p3 keeps every series attached to the right one, which
            # matplotlib's implicit current-figure hides.
            self._n_figs += 1
            self._plot_axes = "p" if self._n_figs <= 1 else f"p{self._n_figs}"
            self._new_axes.add(self._plot_axes)
            return [JuliaStmt(code=f"{self._plot_axes} = plot()",
                              lineno=call.lineno)]
        if name in ("subplot", "subplots", "add_subplot"):
            # matplotlib's grid-of-axes has a Plots.jl counterpart, but the
            # layout is declared at the end rather than switched into mid-script.
            self._n_figs += 1
            self._plot_axes = f"p{self._n_figs}"
            self._new_axes.add(self._plot_axes)
            self._subplot_used = True
            return [
                JuliaStmt(
                    code=f"{self._plot_axes} = plot()  # was plt.{name}(...)",
                    lineno=call.lineno,
                )
            ]
        if name == "axis":
            if call.args and isinstance(call.args[0], ast.Constant):
                mode = call.args[0].value
                if mode == "equal":
                    return [
                        JuliaStmt(
                            code=f"plot!({self._plot_axes}; aspect_ratio = :equal)",
                            lineno=call.lineno,
                        )
                    ]
                if mode in ("off", "on"):
                    flag = "false" if mode == "off" else "true"
                    return [
                        JuliaStmt(
                            code=f"plot!({self._plot_axes}; showaxis = {flag})",
                            lineno=call.lineno,
                        )
                    ]
            if call.args:
                a0 = call.args[0]
                if isinstance(a0, (ast.List, ast.Tuple)) and len(a0.elts) == 4:
                    x0, x1, y0, y1 = (self.tr.translate(e) for e in a0.elts)
                    return [
                        JuliaStmt(
                            code=(
                                f"plot!({self._plot_axes}; "
                                f"xlims = ({x0}, {x1}), ylims = ({y0}, {y1}))"
                            ),
                            lineno=call.lineno,
                        )
                    ]
                lim = self.tr.translate(a0)
                return [
                    JuliaStmt(
                        code=(
                            f"plot!({self._plot_axes}; "
                            f"xlims = ({lim}[1], {lim}[2]), "
                            f"ylims = ({lim}[3], {lim}[4]))"
                        ),
                        lineno=call.lineno,
                    )
                ]
            return []
        if name == "savefig":
            path = self.tr.translate(call.args[0]) if call.args else '"figure.png"'
            return [JuliaStmt(code=f"savefig({self._plot_axes}, {path})",
                              lineno=call.lineno)]
        if name in ("xlabel", "ylabel", "title", "xlim", "ylim", "xscale", "yscale"):
            key = mapping.PLOT_SETTERS.get(name, name)
            val = self.tr.translate(call.args[0]) if call.args else '""'
            return [
                JuliaStmt(
                    code=f"plot!({self._plot_axes}; {key} = {val})",
                    lineno=call.lineno,
                )
            ]
        if name == "legend":
            return [
                JuliaStmt(
                    code=f"plot!({self._plot_axes}; legend = :best)",
                    lineno=call.lineno,
                )
            ]
        if name == "grid":
            return [
                JuliaStmt(
                    code=f"plot!({self._plot_axes}; grid = true)", lineno=call.lineno
                )
            ]
        if name in mapping.PLOT_FUNCS:
            return [self._plot_series(call, name)]
        raise Untranslatable(f"unsupported matplotlib call `{name}`", call)

    def _plot_series(self, call: ast.Call, name: str) -> Stmt:
        args: list[JuliaExpr] = []
        style: dict[str, str] = {}
        runtime_style: Optional[str] = None
        for pos, a in enumerate(call.args):
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                style.update(self._fmt_string(a.value))
            elif pos >= 2:
                # matplotlib's third positional is the format string. When it
                # is a variable (`plt.plot(x, y, lbl)` inside a zip over
                # styles) it cannot be decoded statically, so it is decoded at
                # run time by the emitted `mpl_style` helper instead.
                runtime_style = self.tr.translate(a)
                self.script.needs_style_helper = True
            else:
                args.append(JuliaExpr(self.tr.translate(a)))
        kwargs: dict[str, JuliaExpr] = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            key = {"label": "label", "color": "color", "linewidth": "lw",
                   "lw": "lw", "linestyle": "linestyle",
                   "marker": "marker"}.get(kw.arg)
            if not key:
                continue
            code = self.tr.translate(kw.value)
            # matplotlib stringifies whatever it is handed as a label; Plots.jl
            # requires an actual string and fails with a `length` MethodError
            # otherwise. `label=sch` over a list of scheme constants is the
            # common case, and a plain `string` there would dump the scheme's
            # entire configuration into the legend, so it gets its own helper.
            if key == "label" and not _is_string_literal(kw.value):
                code = f"mfh_label({code})"
                self.script.needs_label_helper = True
            kwargs[key] = JuliaExpr(code)
        if name in mapping.PLOT_SCALE_KW:
            for part in mapping.PLOT_SCALE_KW[name].split(", "):
                k, _, v = part.partition(" = ")
                kwargs[k] = JuliaExpr(v)
        # matplotlib legends only the series that were given a label; Plots.jl
        # invents `y1`, `y2`, … for the rest, which fills the legend with the
        # companion curves the original deliberately left out.
        if "label" not in kwargs:
            kwargs["label"] = JuliaExpr('""')
        return PlotCall(
            func=mapping.PLOT_FUNCS[name],
            args=args,
            kwargs=kwargs,
            style=style,
            runtime_style=runtime_style,
            axes=self._plot_axes,
            lineno=call.lineno,
        )

    @staticmethod
    def _fmt_string(fmt: str) -> dict[str, str]:
        """Decode a matplotlib format string like 'r--' or 'g-d'."""
        out: dict[str, str] = {}
        rest = fmt
        for ls in ("--", "-.", "-", ":"):
            if ls in rest:
                out["linestyle"] = mapping.PLOT_LINESTYLES[ls]
                rest = rest.replace(ls, "", 1)
                break
        for ch in rest:
            if ch in mapping.PLOT_COLORS:
                out["color"] = mapping.PLOT_COLORS[ch]
            elif ch in mapping.PLOT_MARKERS:
                out["marker"] = mapping.PLOT_MARKERS[ch]
        if "linestyle" not in out and "marker" in out:
            out["seriestype"] = ":scatter"
        return out

    # ------------------------------------------------------------------
    # Echoes-call interception (used by ExprTranslator)
    # ------------------------------------------------------------------

    def _on_echoes(self, call: ast.Call, fname: str) -> Optional[str]:
        short = fname.split(".")[-1]

        if short in HOMOGENIZE_FNS:
            return self._homogenize(call, HOMOGENIZE_FNS[short])

        if short == "tensor":
            return self._tensor(call)

        # `x.data()` on an RVE iteration yields the phase itself
        if short == "data" and not call.args:
            raise Untranslatable(
                "iteration over an RVE (`for x in ver` / `x.data()`)", call,
                "MFH exposes the phases by name: iterate "
                "inclusion_phase_names(rve) and use volume_fraction(rve, name) "
                "plus the localization functions",
            )
        return None

    def _on_attribute(self, attr: ast.Attribute) -> Optional[str]:
        return self._phase_access(attr)

    def _angles_arg(self, kw: dict, max_angles: int) -> str:
        """Render `angles=[θ, φ, ψ]` as trailing positional arguments.

        Echoes pads a short angle list with zeros
        (`angles_from_incomplete_angles`), and the emitted helper defaults the
        missing ones to zero, so a shorter list needs no padding here.
        """
        node = kw.get("angles")
        if node is None:
            return ""
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [self.tr.translate(e) for e in node.elts][:max_angles]
            return "".join(f", {v}" for v in vals)
        # a variable holding the angle list
        code = self.tr.translate(node)
        return "".join(f", {code}[{i + 1}]" for i in range(max_angles))

    def _tensor(self, call: ast.Call) -> str:
        """Echoes' generic `tensor(...)` builder.

        Its overloads are discriminated by how the data arrives:

        * `tensor([a, b])` -- the two Walpole scalars of a 4th-order isotropic
          tensor. The corpus writes them as `tensor([3.*k, 2.*mu])`, which is
          exactly TensND's `TensISO{3}(3k, 2mu)` storage, so this is a direct
          rewrite.
        * `tensor(M)` -- a raw array whose *order follows its shape* (6x6 is
          4th order, 3x3 is 2nd). The shape is generally not known statically,
          so the emitted script dispatches on it at run time.
        * `tensor(M, SYM)` -- an array projected onto a symmetry class, which
          is a reporting projection, i.e. `best_fit_*`.
        * longer parameter lists (5, 7, 9, 12 scalars) map to TI / orthotropic
          / anisotropic layouts whose component ordering differs between the
          two libraries. Those are refused rather than guessed.
        """
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        args = list(call.args)

        sym_node = kw.get("sym")
        if sym_node is None and len(args) >= 2 and isinstance(args[1], ast.Name):
            if args[1].id in mapping.PARAMSYM:
                sym_node = args[1]

        if args and isinstance(args[0], (ast.List, ast.Tuple)):
            elts = args[0].elts
            n = len(elts)
            if n == 2:
                # C = αJ + βK is exactly TensND's TensISO storage.
                a = self.tr.translate(elts[0])
                b = self.tr.translate(elts[1])
                return f"TensISO{{3}}({a}, {b})"
            if n not in (3, 5, 9):
                raise Untranslatable(
                    f"`tensor([...])` with {n} parameters",
                    call,
                    "Echoes registers builders for 1, 2, 3, 5, 9 and 21 "
                    "parameters only (tensor_builder.h:466); the 21-parameter "
                    "triclinic layout has no verified component ordering",
                )
            params = ", ".join(self.tr.translate(e) for e in elts)
            self.script.needs_tensor_helper = True
            ang = self._angles_arg(kw, max_angles=2 if n == 5 else 3)
            return f"echoes_tensor(({params}){ang})"

        if not args:
            raise Untranslatable("`tensor()` with no arguments", call)

        inner = self.tr.translate(args[0])
        if sym_node is not None and isinstance(sym_node, ast.Name):
            fn = mapping.PARAMSYM[sym_node.id]
            self.script.needs_tensor_helper = True
            return f"{fn}(echoes_tensor({inner}))"

        self.script.needs_tensor_helper = True
        return f"echoes_tensor({inner})"

    def _phase_access(self, attr: ast.Attribute) -> Optional[str]:
        """`ver["PORE"].eE` -> the corresponding MFH localization call.

        Echoes stores the four localization tensors on the phase as a side
        effect of `homogenize`; MFH computes them on demand from the inclusion,
        its stiffness and the *reference medium*. Which medium that is depends
        on the scheme -- the matrix for the matrix-based schemes, the
        homogenized result for the self-consistent ones -- so the extractor
        resolves it from the homogenize call in scope rather than guessing.
        """
        sub = attr.value
        if not (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id in self.rve_map
            and isinstance(sub.slice, ast.Constant)
            and attr.attr in mapping.PHASE_ACCESSORS
        ):
            return None

        rve = self.rve_map[sub.value.id]
        phase = str(sub.slice.value)
        pd = next((p for p in rve.phases if p.name == phase), None)
        if pd is None:
            raise Untranslatable(
                f"phase `{phase}` is read but never declared on `{rve.var}`", attr
            )

        rve_expr = f"{rve.builder_name}({', '.join(rve.params)})"

        if attr.attr in ("factor", "fraction"):
            return f"volume_fraction({rve_expr}, :{phase})"
        if attr.attr == "density":
            return f"crack_density({rve_expr}, :{phase})"

        transport = ":K" in pd.props and ":C" not in pd.props
        table = (
            mapping.PHASE_ACCESSORS_TRANSPORT if transport
            else mapping.PHASE_ACCESSORS
        )
        fn = table[attr.attr]

        prop_key = ":K" if transport else ":C"
        c_incl = pd.props.get(prop_key)
        if c_incl is None or not c_incl.code:
            raise Untranslatable(
                f"phase `{phase}` carries no `{prop_key}` property to localize "
                f"with", attr,
            )
        c_ref = self._reference_medium(rve, prop_key, attr)
        return f"{fn}({self._geometry_expr(pd)}, {c_incl.code}, {c_ref})"

    def _geometry_expr(self, pd: PhaseDef) -> str:
        from .emit import Emitter

        return Emitter(self.script, "", False)._geometry(pd)

    def _reference_medium(
        self, rve: RVEDef, prop_key: str, node: ast.AST
    ) -> str:
        """The medium the scheme localizes against."""
        scheme = self._current_scheme
        if scheme is None:
            raise Untranslatable(
                "a localization tensor is read with no homogenize call in "
                "scope to fix the reference medium",
                node,
                "MFH needs the reference explicitly: "
                "strain_strain_loc(inclusion, C_inclusion, C_reference)",
            )
        if scheme.startswith(("SelfConsistent", "AsymmetricSelfConsistent")):
            if self._current_result is None:
                raise Untranslatable(
                    "a self-consistent localization needs the homogenized "
                    "stiffness as its reference, but the result is not bound "
                    "to a name",
                    node,
                )
            return self._current_result
        # matrix-based schemes localize against the matrix
        for pd in rve.phases:
            if pd.is_matrix:
                c = pd.props.get(prop_key)
                if c is not None and c.code:
                    return c.code
        return f"matrix_property({rve.builder_name}"
        f"({', '.join(rve.params)}), {prop_key})"

    _current_scheme: Optional[str] = None
    _current_result: Optional[str] = None
    #: name of the function currently being extracted, "" at module level
    _scope: str = ""
    #: its parameters, which are the candidates for builder parameters
    _fn_args: list = []

    def _note_params(self, rve: RVEDef, *nodes: ast.expr) -> None:
        """Any function argument an RVE depends on becomes a builder argument."""
        for node in nodes:
            if node is None:
                continue
            for n in ast.walk(node):
                if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
                    continue
                if n.id in self._fn_args and n.id not in rve.params:
                    rve.params.append(n.id)

    def _homogenize(self, call: ast.Call, flavour: str) -> str:
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        rve_node = kw.get("rve")
        rve_name = (
            rve_node.id
            if isinstance(rve_node, ast.Name)
            else (self.tr.translate(rve_node) if rve_node is not None else "rve")
        )
        rve = self.rve_map.get(rve_name)
        rve_expr = rve.builder_name + "(" + ", ".join(rve.params) + ")" if rve else rve_name

        prop = ":C"
        if "prop" in kw and isinstance(kw["prop"], ast.Constant):
            prop = f":{kw['prop'].value}"

        scheme_node = kw.get("scheme")
        scheme = self._scheme(scheme_node, kw)
        # Remember what the localization tensors read afterwards refer to.
        self._current_scheme = scheme
        if rve is not None and scheme.startswith(("Maxwell", "PonteCastanedaWillis")):
            rve.needs_distribution_shape = True

        if flavour == "alv":
            times = kw.get("time_series")
            t = self.tr.translate(times) if times is not None else "times"
            self.script.imports.add("MeanFieldHomogenization")
            return f"homogenize_alv({rve_expr}, {scheme}, {prop}; times = {t})"

        if flavour == "derivative":
            lens = self._derivative_lens(kw)
            return f"derivative({rve_expr}, {scheme}, {lens}; output = {prop})"

        return f"homogenize({rve_expr}, {scheme}, {prop})"

    def _scheme(self, node: Optional[ast.expr], kw: dict) -> str:
        """Build the MFH scheme instance, folding in Echoes solver keywords."""
        if node is None:
            return "MoriTanaka()"
        if not isinstance(node, ast.Name) or node.id not in mapping.SCHEMES:
            # a variable holding a scheme: pass it through untouched
            return self.tr.translate(node)
        spec = mapping.SCHEMES[node.id]
        if spec.ctor.endswith(")"):
            return spec.ctor
        opts: list[str] = []
        for ekw, mkw in mapping.SOLVER_KW_RENAME.items():
            if ekw in kw and mkw in spec.solver_kw:
                opts.append(f"{mkw} = {self.tr.translate(kw[ekw])}")
        # Echoes' iterative stop is purely relative, so a faithful translation
        # of `epsrel` must also switch MFH's absolute term off. Left at its
        # `1e-12` default it would silently dominate `abstol + reltol * ‖x‖`
        # whenever the caller asked for a tight `epsrel` — exactly the case
        # (`epsrel = 1e-15`) where the number was chosen to matter.
        if "epsrel" in kw and "epsabs" not in kw and "abstol" in spec.solver_kw:
            opts.append("abstol = 0.0")
        if node.id == "DIFF" and "maxnb" in kw:
            opts = [f"nsteps = {self.tr.translate(kw['maxnb'])}"]
        return f"{spec.ctor}(; {', '.join(opts)})" if opts else f"{spec.ctor}()"

    def _derivative_lens(self, kw: dict) -> str:
        phase = kw.get("phase")
        pname = (
            f":{phase.value}"
            if isinstance(phase, ast.Constant)
            else (self.tr.translate(phase) if phase is not None else ":PORE")
        )
        if "index" in kw:
            idx = self.tr.translate(kw["index"])
            prop = ":C"
            if "prop" in kw and isinstance(kw["prop"], ast.Constant):
                prop = f":{kw['prop'].value}"
            return f"property({pname}, {prop}, {idx})"
        return f"amount({pname})"

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def _callee(self, call: ast.Call) -> Optional[str]:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            parts = []
            cur: ast.AST = call.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
            return call.func.attr
        return None

    @staticmethod
    def _targets(t: ast.expr) -> list[str]:
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)):
            out: list[str] = []
            for e in t.elts:
                out.extend(Extractor._targets(e))
            return out
        raise Untranslatable("unsupported assignment target")

    def _src(self, node: ast.AST) -> str:
        lo = getattr(node, "lineno", 1) - 1
        hi = getattr(node, "end_lineno", lo + 1)
        return "\n".join(self.lines[lo:hi])

    def _refuse(
        self,
        node: ast.AST,
        reason: str,
        suggestion: str = "",
        severity: str = "blocking",
        symbol: str = "",
    ) -> Untranslated:
        u = Untranslated(
            reason=reason,
            suggestion=suggestion,
            severity=severity,
            symbol=symbol,
            lineno=getattr(node, "lineno", 0),
            py_src=self._src(node),
        )
        self.script.findings.append(u)
        return u

    def _refusal_node(self, node: ast.AST, e: Untranslatable) -> Untranslated:
        return self._refuse(node, e.reason, e.suggestion)


def extract(source: str, path: str = "<script>") -> Script:
    tree = ast.parse(source)
    return Extractor(tree, source, path).run()
