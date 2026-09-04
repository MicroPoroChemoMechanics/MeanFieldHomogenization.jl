"""Model → Julia.

The conventions here are the ones `tools/echoes2mfh/echoes2mfh/emit.py` already
establishes, and the two generators are kept deliberately consistent: a script
written by the studio and one translated from Echoes should read the same way.

Three of those conventions are load-bearing and worth restating, because the
whole point of the interface is that the user never has to remember them:

* `iso_stiffness(k, μ)` takes *physical* moduli, while the raw `TensISO{3}(a, b)`
  constructor takes `(3k, 2μ)`. Only the former is ever emitted.
* The matrix amount is derived (`1 − Σ f`) and assigning it raises. The matrix
  phase therefore has no amount in the generated code.
* Solver options belong to the **scheme instance**, not to `homogenize`.
"""

from __future__ import annotations

import json
from typing import Optional

from .model import Cell, Model, Phase, Property

IND = "    "
RULE = "# " + "=" * 77
MODEL_TAG = "mfhstudio-model v1"


def _rule(title: str) -> str:
    bar = "─" * max(3, 72 - len(title))
    return f"# ── {title} {bar}"


def _angles(euler) -> list:
    """The ZYZ angles, padded to three and stripped of a canonical frame."""
    a = [x for x in (euler or []) if x != "" and x is not None]
    a = (a + [0.0, 0.0, 0.0])[:3]
    return [] if all(x == 0 for x in a) else a


def _is_canonical_normal(n) -> bool:
    """Is this the default stacking direction, e₃?

    Worth asking, because `Laminate()` with no keyword yields a
    `CanonicalBasis` and the kernel then skips the frame rotation entirely.
    Writing `normal = (0.0, 0.0, 1.0)` would say the same thing at a cost.
    """
    vals = list(n or [])
    if len(vals) != 3 or any(isinstance(x, str) for x in vals):
        return False
    return vals[0] == 0 and vals[1] == 0 and vals[2] > 0


def _angle_num(x) -> str:
    """An angle is a number *or* a Julia expression the user typed.

    `π/4` says what `0.7853981633974483` only approximates, and it is what
    comes back out when the script is read again, so a typed expression is
    emitted verbatim rather than evaluated here.
    """
    return x if isinstance(x, str) else _fnum(x)


def _frame_expr(euler) -> str:
    """The basis the constants are written in.

    `TensOrtho` and `Tens(…, basis)` both demand a basis object, so the
    canonical frame has to be spelled out rather than omitted.
    """
    a = _angles(euler)
    if not a:
        return "CanonicalBasis{3, Float64}()"
    return "RotatedBasis(" + ", ".join(_angle_num(x) for x in a) + ")"


def _axis_expr(euler, axis=None) -> str:
    """The symmetry axis of a transversely isotropic tensor.

    TI tensors carry an axis, not a basis — the third vector of the frame the
    Euler angles define, ψ being irrelevant since the transverse plane is
    isotropic. Taking it from `vecbasis` rather than writing
    `(sinθcosφ, sinθsinφ, cosθ)` keeps one definition of "the frame" in the
    script: change the angles and the axis follows.
    """
    a = _angles(euler)
    if a:
        return f"vecbasis({_frame_expr(a)})[:, 3]"
    if isinstance(axis, (list, tuple)) and len(axis) == 3:
        return _tuple(_fnum(x) for x in axis)
    return "(0.0, 0.0, 1.0)"


def _axis_tuple_expr(euler, axis=None) -> str:
    """`_axis_expr`, but as a Julia *tuple*.

    `LayeredSpheroid` and `layered_spheroid_from_fractions` declare
    `axis::Tuple`, which rejects the `Vector` that `vecbasis(...)[:, 3]`
    returns — the same trap the `TensTI{2}` builder documents above.
    """
    e = _axis_expr(euler, axis)
    return e if e.startswith("(") else f"Tuple({e})"


def _tuple(items) -> str:
    """A Julia tuple; only a 1-tuple needs the trailing comma."""
    parts = [p for p in items if p]
    if not parts:
        return "()"
    if len(parts) == 1:
        return f"({parts[0]},)"
    return "(" + ", ".join(parts) + ")"


def _fnum(x) -> str:
    """Like `_num`, but never yields a bare integer.

    Geometry constructors take an `NTuple{N, T}`: mixing `1` and `0.6` gives a
    `Tuple{Float64, Int64}` and no method matches. Sizes are therefore always
    written as floats, while genuine counts (`nsteps`, `length`) keep `_num`.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return _num(x)
    if isinstance(x, int):
        return _num(float(x))
    return _num(x)


def _num(x) -> str:
    """Render a value as Julia, keeping float-ness explicit."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        s = repr(x)
        if "e" in s or "E" in s:
            mant, _, exp = s.partition("e")
            if "." not in mant:
                mant += ".0"
            return f"{mant}e{int(exp)}"
        return s if "." in s else s + ".0"
    return str(x)


# The activation preamble is shared with echoes2mfh: a generated script must
# run from wherever it is saved, not only from `scripts/`.
_ACTIVATE = """import Pkg
let d = @__DIR__
    while true
        pt = joinpath(d, "Project.toml")
        if isfile(pt) && occursin("MeanFieldHomogenization", read(pt, String))
            Pkg.activate(d; io = devnull)
            break
        end
        parent = dirname(d)
        parent == d && break
        d = parent
    end
end"""


class CodeGen:
    def __init__(self, model: Model, embed_model: bool = True):
        self.m = model
        self.embed = embed_model
        self.out: list = []

    # -- plumbing ---------------------------------------------------------

    def w(self, line: str = "") -> None:
        self.out.append(line)

    def blank(self) -> None:
        if self.out and self.out[-1] != "":
            self.out.append("")

    # -- entry point ------------------------------------------------------

    def generate(self) -> str:
        problems = self.m.validate()
        self._header(problems)
        self._preamble()
        self._params()
        self._builders()
        self._opaque()
        self._main()
        self._embedded_model()
        return "\n".join(self.out).rstrip() + "\n"

    # -- sections ---------------------------------------------------------

    def _header(self, problems: list) -> None:
        self.w(RULE)
        self.w(f"#  {self.m.title}.jl")
        self.w("#")
        self.w("#  Built with MFH Studio. Editing this file by hand is fine: the")
        self.w("#  studio reads it back and preserves anything it does not")
        self.w("#  recognize.")
        if self.m.description:
            self.w("#")
            for line in self.m.description.splitlines():
                self.w(f"#  {line}")
        if problems:
            self.w("#")
            self.w("#  UNRESOLVED PROBLEMS — this script will not run as-is:")
            for p in problems:
                self.w(f"#    * {p}")
        self.w(RULE)
        self.w()

    def _preamble(self) -> None:
        for line in _ACTIVATE.splitlines():
            self.w(line)
        self.blank()
        self.w("using MeanFieldHomogenization")
        self.w("using TensND")
        self.w("using LinearAlgebra")
        self.w("using Printf")
        # Only when the main that actually plots is the one being emitted:
        # sensitivities print a table, and an unused `using Plots` costs the
        # reader a question and the script a load.
        if self.m.alv.enabled:
            plots = self.m.alv.plot
        elif self.m.sens.enabled:
            plots = False
        else:
            plots = self.m.sweep.enabled and self.m.sweep.mode != "single" and self.m.sweep.plot
        if plots:
            self.w("using Plots")
            self.w("gr()")
        self.blank()

    def _params(self) -> None:
        if not self.m.params:
            return
        self.w(_rule("Parameters"))
        for p in self.m.params:
            # A parameter read from a file keeps its original text until the
            # user actually changes it: re-printing from the AST would collapse
            # deliberate multi-line layout for no benefit.
            if p.origin and not p.edited:
                for line in p.origin.splitlines():
                    self.w(line)
                continue
            line = f"const {p.name} = {p.value}"
            if p.comment:
                line += f"  # {p.comment}"
            self.w(line)
        self.blank()

    # -- cells ------------------------------------------------------------

    def _builders(self) -> None:
        if not self.m.cells:
            return
        try:
            order = self.m.topological_order()
        except ValueError:
            # The cycle is already reported in the header; emit in declaration
            # order so the rest of the file is still readable.
            order = list(self.m.cells)

        if self.m.uses_multiscale():
            self.w(_rule("Scales"))
            self.w("#")
            self.w("# Inner scales are defined first. A phase property holding a")
            self.w("# `Homogenized(cell, scheme)` is the seam: the outer scheme")
            self.w("# resolves the inner scale when it reads that property, and")
            self.w("# memoizes it for the duration of the call.")
            self.blank()

        for c in order:
            self._builder(c)

    def _builder(self, c: Cell) -> None:
        args = ", ".join(c.params)
        self.w(f"function {c.builder}({args})")
        for line in self._cell_body(c):
            self.w(line)
        self.w("end")
        self.blank()

    def _cell_body(self, c: Cell) -> list:
        """The builder's statements, without the `function`/`end` wrapper.

        Kept apart from `_builder` because the 3-D view needs the same
        statements as a single expression (`let … end`), and two spellings of
        one construction is how the picture and the script start disagreeing.
        """
        saved, self.out = self.out, []
        try:
            if c.is_laminate():
                self._laminate_body(c)
            else:
                self._rve_body(c)
            return self.out
        finally:
            self.out = saved

    def _rve_body(self, c: Cell) -> None:
        opts = c.rve_options or {}
        tail = ("; " + ", ".join(f"{k} = {v}" for k, v in sorted(opts.items()))) if opts else ""
        # No phase name on the constructor: an RVE designates no matrix. The
        # phase whose amount is "rest" is emitted first, as it reads best, but
        # the order carries no meaning.
        self.w(f"{IND}rve = RVE({tail})")   # `tail` is "" or "; k = v, ..." 
        rest = c.remainder()
        if rest is not None:
            self._emit_phase(rest, c)
        for ph in c.inclusions():
            self._emit_phase(ph, c)
        self.w(f"{IND}return rve")

    def _laminate_body(self, c: Cell) -> None:
        """`Laminate(...)` then one `add_layer!` per layer, in stacking order.

        The frame is given one way only — MeanFieldHomogenization takes at most
        one of `normal`, `euler_angles` and `basis`, and rejects two. The
        canonical `(0, 0, 1)` is emitted as nothing at all, which is both the
        default and the case where the kernel skips the frame rotation.
        """
        opts = []
        if c.frame_mode == "euler" and _angles(c.euler_angles):
            vals = ", ".join(
                x if isinstance(x, str) else _fnum(x) for x in c.euler_angles
            )
            opts.append(
                f"euler_angles = ({vals},)" if len(c.euler_angles) == 1
                else f"euler_angles = ({vals})"
            )
        elif c.frame_mode != "euler" and not _is_canonical_normal(c.normal):
            vals = ", ".join(
                x if isinstance(x, str) else _fnum(x) for x in (c.normal or [])
            )
            opts.append(f"normal = ({vals})")
        for k, v in sorted((c.rve_options or {}).items()):
            opts.append(f"{k} = {v}")
        tail = ("; " + ", ".join(opts)) if opts else ""
        self.w(f"{IND}lam = Laminate({tail})")

        for lay in c.layers:
            props = self._properties(lay)
            kw = [f"{lay.amount_kind} = {self._amount(lay)}"]
            itf = self._interface_expr(lay.interface)
            if itf:
                kw.append(f"interface = {itf}")
            self._call(
                f"add_layer!(lam, :{lay.name}, {props}; " + ", ".join(kw) + ")"
            )
        self.w(f"{IND}return lam")

    @staticmethod
    def _interface_expr(itf) -> str:
        """A layer's interface, or `""` for the perfect one.

        Perfect is the default of `add_layer!`, and writing it out adds a
        keyword that says nothing.
        """
        itf = itf or {}
        kind = itf.get("kind", "PerfectInterface")
        if kind == "PerfectInterface":
            return ""
        args = ", ".join(
            (v if isinstance(v, str) else _num(v))
            for v in (itf.get("args") or {}).values()
        )
        return f"{kind}({args})" if args else f"{kind}()"

    def _emit_phase(self, ph: Phase, c: Cell) -> None:
        geom = self._geometry(ph)
        props = self._properties(ph)
        opts = []
        if ph.symmetrize and ph.symmetrize != "none":
            opts.append(
                "symmetrize = "
                + {"iso": "IsoSymmetrize()", "ti": "TISymmetrize()"}[ph.symmetrize]
            )

        # One declaration for every phase. `fraction = :rest` marks the one
        # that takes up the volume the others leave; a matrix-based scheme
        # picks it up as its reference medium unless it names another.
        if ph.amount_kind == "rest":
            opts.insert(0, "fraction = :rest")
        else:
            opts.insert(0, f"{ph.amount_kind} = {self._amount(ph)}")
        self._call(f"add_phase!(rve, :{ph.name}, {geom}, {props}; " + ", ".join(opts) + ")")

    def _call(self, line: str) -> None:
        full = IND + line
        if len(full) <= 92:
            self.w(full)
            return
        head, _, tail = line.partition("(")
        body, _, _close = tail.rpartition(")")
        # The split must happen at the call's *own* keyword separator. A naive
        # search finds the one inside a nested constructor
        # (`SelfConsistent(; abstol = …)`) and produces unbalanced Julia, so
        # only depth-zero separators count.
        cut = _toplevel_kw_split(body)
        if cut is None:
            # nothing safe to break on — a long line beats a broken one
            self.w(full)
            return
        pos, kw = body[:cut], body[cut + 2:]
        self.w(f"{IND}{head}(")
        self.w(f"{IND}{IND}{pos};")
        self.w(f"{IND}{IND}{kw}")
        self.w(f"{IND})")

    def _amount(self, ph: Phase) -> str:
        return ph.amount if isinstance(ph.amount, str) else _fnum(ph.amount)

    def _geometry(self, ph: Phase) -> str:
        g = ph.geometry
        a = g.args
        v = lambda k, d=0.0: (a[k] if isinstance(a.get(k), str) else _fnum(a.get(k, d)))
        ang = ""
        if g.euler_angles:
            # Angles are physical quantities: a bare `0` next to `0.5` would
            # make the tuple `Tuple{Int, Float64}`.
            vals = ", ".join(
                x if isinstance(x, str) else _fnum(x) for x in g.euler_angles
            )
            ang = f"; euler_angles = ({vals},)" if len(g.euler_angles) == 1 else f"; euler_angles = ({vals})"

        k = g.kind
        if k == "spheroid":
            return f"Spheroid({v('omega', 1.0)}{ang})"
        if k == "ellipsoid":
            return f"Ellipsoid({v('a', 1.0)}, {v('b', 1.0)}, {v('c', 1.0)}{ang})"
        if k == "cylinder":
            return f"Cylinder({v('b', 1.0)}, {v('c', 1.0)}{ang})"
        if k == "penny_crack":
            return f"PennyCrack({v('a', 1.0)}{ang})"
        if k == "elliptic_crack":
            return f"EllipticCrack({v('a', 1.0)}, {v('b', 0.5)}{ang})"
        if k == "ribbon_crack":
            return f"RibbonCrack({v('b', 1.0)}{ang})"
        if k == "layered_sphere":
            return self._layered_sphere(g)
        if k == "layered_spheroid":
            return self._layered_spheroid(g)
        return "Spheroid(1.0)"

    def _layered_sphere(self, g) -> str:
        radii = ", ".join(
            (l["radius"] if isinstance(l.get("radius"), str) else _fnum(l.get("radius", 1.0)))
            for l in g.layers
        )
        moduli = ", ".join(self._prop_expr(Property.from_dict(l["property"])) for l in g.layers)
        radii_t = _tuple(radii.split(", ")) if radii else "()"
        mod_t = _tuple(moduli.split(", ")) if moduli else "()"
        ifaces = [l.get("interface") for l in g.layers]
        if any(i and i.get("kind", "PerfectInterface") != "PerfectInterface" for i in ifaces):
            parts = []
            for i in ifaces:
                i = i or {"kind": "PerfectInterface", "args": {}}
                kind = i.get("kind", "PerfectInterface")
                args = ", ".join(_num(x) for x in (i.get("args") or {}).values())
                parts.append(f"{kind}({args})" if args else f"{kind}()")
            it = f"({parts[0]},)" if len(parts) == 1 else "(" + ", ".join(parts) + ")"
            return f"LayeredSphere({radii_t}, {mod_t}; interfaces = {it})"
        return f"LayeredSphere({radii_t}, {mod_t})"

    def _layered_spheroid(self, g) -> str:
        """`layered_spheroid_from_fractions`, not the raw constructor.

        `LayeredSpheroid(axis_radii, disk_radii, …)` demands that every layer
        share one focal distance, and radii typed in layer by layer essentially
        never do — the old form here scaled one radius list by ω, which is not
        confocal and threw. The fraction constructor takes the outer aspect
        ratio and size and solves for the confocal inner radii itself, which is
        the only form a person can drive.
        """
        fractions = _tuple(
            (
                l["fraction"] if isinstance(l.get("fraction"), str)
                else _fnum(l.get("fraction", 1.0))
            )
            for l in g.layers
        )
        moduli = _tuple(
            self._prop_expr(Property.from_dict(l["property"])) for l in g.layers
        )
        omega = g.args.get("omega", 0.5)
        radius = g.args.get("radius", 1.0)
        ns = g.args.get("Nseries", 5)
        # A spheroid of revolution is orientable, and the solver honors it:
        # `scheme_integration.jl` returns `TensTI{2}(αt, αa, s.axis)`, so the
        # concentration tensors come back TI about the axis the geometry
        # carries. Dropping the angles here left the axis at its (0,0,1)
        # default, silently ignoring the interface's own orientation fields —
        # in the computation as much as in the 3-D view.
        axis = ""
        if _angles(g.euler_angles):
            axis = f", axis = {_axis_tuple_expr(g.euler_angles)}"
        return (
            f"layered_spheroid_from_fractions({_fnum(omega)}, {_fnum(radius)}, "
            f"{fractions}, {moduli}; Nseries = {_num(ns)}{axis})"
        )

    def _properties(self, ph: Phase) -> str:
        entries = [f"{p.key} => {self._prop_expr(p)}" for p in ph.properties]
        if not entries:
            return "Dict{Symbol, Any}()"
        return "Dict(" + ", ".join(entries) + ")"

    def _prop_expr(self, p: Property) -> str:
        if p.source == "expr":
            return p.expr or "one(TensISO{3})"

        if p.source == "cell":
            # The multiscale seam.
            inner = self.m.cell(p.cell)
            if inner is None:
                return "#= missing inner cell =# one(TensISO{3})"
            call = f"{inner.builder}(" + ", ".join(inner.params) + ")"
            scheme = self._scheme(p.scheme or "MoriTanaka", p.scheme_options or {})
            return f"Homogenized({call}, {scheme})"

        if p.source == "visco" or p.visco:
            return self._visco_expr(p.visco or {})

        a = p.args
        num = lambda k, d=0.0: (a[k] if isinstance(a.get(k), str) else _fnum(a.get(k, d)))
        b = p.builder
        if b == "iso_stiffness":
            return f"iso_stiffness({num('k', 1.0)}, {num('mu', 1.0)})"
        if b == "iso_stiffness_E_nu":
            return f"iso_stiffness_E_nu({num('E', 1.0)}, {num('nu', 0.2)})"
        if b == "hoenig_stiffness":
            # The axis is a required argument, not a keyword with a default:
            # MeanFieldHomogenization declares only the six-argument method.
            return (
                f"hoenig_stiffness({num('E1', 30.0)}, {num('h', 0.3)}, "
                f"{num('nu1', 0.2)}, {num('nu2', 0.25)}, {num('gamma', 0.5)}, "
                f"{_axis_expr(p.euler_angles)})"
            )
        if b == "TensISO{3}":
            # One argument to TensISO{dim} is the 2nd-order (conductivity) form.
            return f"TensISO{{3}}({num('k', 1.0)})"
        if b == "TensTI2":
            # The outer constructor, not `TensTI{2, Float64, 2}(data, n)`: the
            # fully parameterized one takes the axis as a 3-*tuple* and rejects
            # the vector `vecbasis(...)[:, 3]` returns.
            return (
                f"TensTI{{2}}({num('kt', 1.0)}, {num('ka', 1.0)}, "
                f"{_axis_expr(p.euler_angles, a.get('axis'))})"
            )
        if b == "TensDiag2":
            # `Tens(A, basis)` stores A as the components *in that basis*, which
            # is what "the conductivity is diagonal in the material frame" means.
            diag = (
                f"[{num('k1', 1.0)} 0.0 0.0; 0.0 {num('k2', 1.0)} 0.0; "
                f"0.0 0.0 {num('k3', 1.0)}]"
            )
            if _angles(p.euler_angles):
                return f"Tens({diag}, {_frame_expr(p.euler_angles)})"
            return f"Tens({diag})"
        if b == "TensOrtho":
            consts = ", ".join(
                num(k, d) for k, d in (
                    ("C11", 120.0), ("C22", 90.0), ("C33", 70.0),
                    ("C12", 40.0), ("C13", 35.0), ("C23", 30.0),
                    ("C44", 25.0), ("C55", 22.0), ("C66", 20.0),
                )
            )
            return f"TensOrtho({consts}, {_frame_expr(p.euler_angles)})"
        if b == "maxwell_iso":
            return (
                f"maxwell_iso({num('k', 10.0)}, {num('mu', 5.0)}, "
                f"{num('eta_k', 1.0)}, {num('eta_mu', 1.0)})"
            )
        if b == "kelvin_iso":
            return (
                f"kelvin_iso({num('k0', 10.0)}, {num('mu0', 5.0)}, "
                f"[{num('k1', 20.0)}], [{num('mu1', 10.0)}], "
                f"[{num('tau_k', 1.0)}], [{num('tau_mu', 1.0)}])"
            )
        if b == "heaviside_law":
            return f"heaviside_law(iso_stiffness({num('k', 10.0)}, {num('mu', 5.0)}))"
        if b == "ViscoLaw":
            expr = a.get("expr") or "iso_stiffness(10.0, 5.0)"
            mode = str(a.get("mode") or "creep").strip().lstrip(":")
            mode = mode if mode in ("creep", "relaxation") else "creep"
            return f"ViscoLaw((t, t\u2032) -> {expr}, :{mode})"
        return f"iso_stiffness({num('k', 1.0)}, {num('mu', 1.0)})"

    def _visco_expr(self, v: dict) -> str:
        kind = v.get("kind", "maxwell_iso")
        a = v.get("args", {})
        num = lambda k, d=1.0: _num(a.get(k, d))
        if kind == "maxwell_iso":
            return f"maxwell_iso({num('k', 10.0)}, {num('mu', 5.0)}, {num('tau')})"
        if kind == "kelvin_iso":
            return f"kelvin_iso({num('k', 10.0)}, {num('mu', 5.0)}, {num('tau')})"
        if kind == "heaviside":
            return (
                f"heaviside_law(iso_stiffness({num('k', 10.0)}, {num('mu', 5.0)}))"
            )
        expr = a.get("expr", "1.0")
        mode = v.get("mode", "creep")
        return f"ViscoLaw((t, t′) -> {expr}, :{mode})"

    # -- schemes ----------------------------------------------------------

    def _scheme(self, name: str, options: dict) -> str:
        opts = {k: v for k, v in (options or {}).items() if v is not None}
        if not opts:
            return f"{name}()"
        # Solver options attach to the scheme instance, not to `homogenize`.
        parts = ", ".join(f"{k} = {_num(v)}" for k, v in sorted(opts.items()))
        return f"{name}(; {parts})"

    # -- opaque -----------------------------------------------------------

    def _opaque(self) -> None:
        if not self.m.opaque:
            return
        self.w(_rule("Preserved from the original script"))
        self.w("#")
        self.w("# MFH Studio did not recognize the code below, so it is kept")
        self.w("# exactly as it was rather than rewritten.")
        self.blank()
        for blk in sorted(self.m.opaque, key=lambda b: b.order):
            if blk.note:
                self.w(f"# {blk.note}")
            for line in blk.source.splitlines():
                self.w(line)
            self.blank()

    # -- main -------------------------------------------------------------

    def _main(self) -> None:
        root = self.m.root()
        if root is None:
            return
        self.w(_rule("Result"))
        if self.m.alv.enabled:
            self._alv_main(root)
        elif self.m.sens.enabled:
            self._sens_main(root)
        elif not self.m.sweep.enabled or self.m.sweep.mode == "single":
            self._single_main(root)
        else:
            self._sweep_main(root, self.m.sweep)

    # -- outputs ----------------------------------------------------------

    _PROJECTIONS = {
        "iso": "best_fit_iso", "ti": "best_fit_ti", "ortho": "best_fit_ortho",
    }

    def _project(self, var: str) -> str:
        fn = self._PROJECTIONS.get(self.m.sweep.projection)
        return f"{fn}({var})" if fn else var

    def _clamped(self, expr: str) -> str:
        """`max(…, 0.0)` when the sweep asks for it, as `scripts/28` does."""
        return f"max({expr}, 0.0)" if self.m.sweep.clamp_zero else expr

    #: What each output kind reduces the effective tensor with, and the local
    #: name to bind it to. `k` and `μ` come out of one `k_mu` call, `E` and `ν`
    #: out of one `E_nu`, a component and a trace out of one `Array` — so
    #: asking for both members of a pair must not compute the pair twice.
    _REDUCERS = {
        "k": ("km", "k_mu({var})"),
        "mu": ("km", "k_mu({var})"),
        "E": ("Enu", "E_nu({var})"),
        "nu": ("Enu", "E_nu({var})"),
        "km": ("KMC", "KM({var})"),
        "comp": ("arr", "Array({var})"),
        "trace3": ("arr", "Array({var})"),
    }

    def _bindings(self, outputs: list, var: str, suffix: str = "") -> list:
        """`[(name, expr)]` for every reduction used more than once.

        Plotting `k` and `μ` together used to emit `k_mu(C)[1], k_mu(C)[2]` —
        the same solve run twice, at every point of every sweep, for every
        scheme. Binding it once is both what one would write by hand and a
        straight halving of the reduction work.

        A reduction used once is left inline: a name introduced for a single
        use is a line of noise in a script somebody has to read.
        """
        seen: dict = {}
        for o in outputs:
            entry = self._REDUCERS.get(o.get("kind", "k"))
            if entry is None:
                continue
            name, tmpl = entry
            seen.setdefault(name, [tmpl, 0])
            seen[name][1] += 1
        return [
            (name + suffix, tmpl.format(var=var))
            for name, (tmpl, n) in seen.items() if n > 1
        ]

    def _output_expr(self, o: dict, var: str = "C", binds: Optional[dict] = None) -> str:
        """One plotted quantity.

        `k`/`μ`/`E`/`ν` exist only for an isotropic tensor — MeanFieldHomogenization's
        `k_mu` has no method for anything else, which is exactly the error an
        oriented inclusion without an orientation average produces. Kelvin-
        Mandel components are defined whatever the symmetry, so they are the
        way out rather than a silent projection.

        `binds` maps a reduction's name to the local it was bound to by
        `_bindings`; anything not in it is emitted inline.
        """
        binds = binds or {}

        def red(kind: str) -> str:
            """The reduced value: the local when there is one, else the call."""
            name, tmpl = self._REDUCERS[kind]
            return binds.get(name) or tmpl.format(var=var)

        kind = o.get("kind", "k")
        i, j = int(o.get("i", 1)), int(o.get("j", 1))
        if kind == "k":
            return f"{red('k')}[1]"
        if kind == "mu":
            return f"{red('mu')}[2]"
        if kind == "E":
            return f"{red('E')}[1]"
        if kind == "nu":
            return f"{red('nu')}[2]"
        if kind == "km":
            return f"{red('km')}[{i}, {j}]"
        if kind == "comp":
            return f"{red('comp')}[{i}, {j}]"
        if kind == "trace3":
            return f"tr({red('trace3')}) / 3"
        return f"{red('k')}[1]"

    @staticmethod
    def _output_label(o: dict) -> str:
        kind = o.get("kind", "k")
        i, j = int(o.get("i", 1)), int(o.get("j", 1))
        return {
            "k": "k", "mu": "mu", "E": "E", "nu": "nu",
            "km": f"KM{i}{j}", "comp": f"C{i}{j}", "trace3": "tr/3",
        }.get(kind, kind)

    def _scheme_list(self, sw) -> list:
        return [
            (x.get("name", "MoriTanaka"), self._scheme(x.get("name", "MoriTanaka"),
                                                       x.get("options") or {}))
            for x in (sw.schemes or [{"name": "MoriTanaka", "options": {}}])
        ]

    # -- single point ------------------------------------------------------

    def _single_main(self, root: Cell) -> None:
        sw = self.m.sweep
        call = f"{root.builder}(" + ", ".join(root.params) + ")"
        self.w("# One homogenization with the amounts entered in the model.")
        self.w(f"const cell = {call}")
        self.blank()
        names = [self._output_label(o) for o in sw.outputs] or ["value"]
        for scheme_name, scheme in self._scheme_list(sw):
            var = f"C_{scheme_name}"
            self.w(f"{var} = homogenize(cell, {scheme}, {sw.property})")
            proj = self._project(var)
            if proj != var:
                self.w(f"{var} = {proj}")
            # One binding per scheme: these are top-level names, so they carry
            # the scheme's suffix rather than colliding with the previous one.
            binds = {}
            for name, expr in self._bindings(sw.outputs, var, f"_{scheme_name}"):
                self.w(f"{name} = {expr}")
                binds[name[: -len(scheme_name) - 1]] = name
            # `@printf` takes its arguments space-separated: commas would make
            # them a single tuple and the format-specifier count would not match.
            vals = " ".join(
                self._clamped(self._output_expr(o, var, binds)) for o in sw.outputs
            ) or var
            fmt = "  ".join(f"{n} = %.6f" for n in names)
            nl = "\\n"
            self.w(f'@printf "{scheme_name:<24}  {fmt}{nl}" {vals}')
        self.blank()

    # -- sweep -------------------------------------------------------------

    def _sweep_main(self, root: Cell, sw) -> None:
        lens = self._lens_expr(sw.lens)
        names = [self._output_label(o) for o in sw.outputs] or ["value"]
        schemes = self._scheme_list(sw)

        self.w(
            f"const {sw.variable}s = range({_fnum(sw.start)}, {_fnum(sw.stop)}; "
            f"length = {_num(sw.length)})"
        )
        self.blank()
        self.w("#")
        self.w("# `set_param` returns a *new* cell, leaving the original intact,")
        self.w("# so the sweep is a pure map rather than a mutation.")
        self.w(f"const base_cell = {root.builder}(" + ", ".join(root.params) + ")")
        self.blank()
        self.w(f"function evaluate(scheme, {sw.variable})")
        self.w(f"{IND}cell = set_param(base_cell, {lens}, {sw.variable})")
        self.w(f"{IND}C = homogenize(cell, scheme, {sw.property})")
        proj = self._project("C")
        if proj != "C":
            self.w(f"{IND}C = {proj}")
        binds = {}
        for name, expr in self._bindings(sw.outputs, "C"):
            self.w(f"{IND}{name} = {expr}")
            binds[name] = name
        outs = ", ".join(
            self._clamped(self._output_expr(o, "C", binds)) for o in sw.outputs
        ) or "C"
        self.w(f"{IND}return ({outs},)" if len(names) == 1 else f"{IND}return ({outs})")
        self.w("end")
        self.blank()

        self.w("const SCHEMES = [")
        for scheme_name, scheme in schemes:
            self.w(f'{IND}("{scheme_name}", {scheme}),')
        self.w("]")
        self.blank()
        self.w("const RESULTS = Dict{String, Any}()")
        self.w("for (name, scheme) in SCHEMES")
        self.w(f"{IND}rows = [evaluate(scheme, {sw.variable}) for {sw.variable} in {sw.variable}s]")
        for i, n in enumerate(names):
            self.w(f'{IND}RESULTS["$(name) {n}"] = [r[{i + 1}] for r in rows]')
        self.w("end")
        self.blank()

        self.w("# Published for MFH Studio; harmless when the script runs alone.")
        self.w(
            f'const MFHSTUDIO_RESULTS = merge('
            f'Dict("x" => collect({sw.variable}s), "xlabel" => "{sw.variable}"), RESULTS)'
        )
        self.blank()
        # Sorted once and reused: the table and the figure walk the same
        # curves, in the same order, and ordering them twice would let the two
        # disagree if the key ever changed.
        self.w("const CURVES = sort!(collect(RESULTS); by = first)")
        self.blank()
        self.w("for (label, ys) in CURVES")
        self.w(f'{IND}@printf "%-28s  first = %.6f   last = %.6f\\n" label first(ys) last(ys)')
        self.w("end")

        if sw.plot:
            self.blank()
            self.w(
                f'p = plot(; xlabel = "{sw.variable}", ylabel = "effective property", '
                f"legend = :best)"
            )
            self.w("for (label, ys) in CURVES")
            self.w(f"{IND}plot!(p, {sw.variable}s, ys; label = label, lw = 2)")
            self.w("end")
            self.w("display(p)")

    # -- sensitivities -----------------------------------------------------

    def _sens_main(self, root: Cell) -> None:
        """`derivative` / `gradient` / `jacobian`, straight from the lenses.

        These wrappers take the cell and the parameter lens themselves — they
        build the perturbed cell with `set_param` and run `homogenize` inside
        the ForwardDiff pass — so there is no closure to write. What the
        interface supplies is the lens it already models and the scalar
        extraction the sweep already emits, handed over as `indexer`.

        A jacobian passes no `indexer`: it differentiates the whole effective
        tensor, flattened. That is also the way out when the result is not
        isotropic and `k_mu` therefore has no method for it.
        """
        s = self.m.sens
        cell = self.m.cell(s.cell) or root
        call = f"{cell.builder}(" + ", ".join(cell.params) + ")"
        scheme = self._scheme(s.scheme, s.scheme_options or {})
        lenses = [self._lens_expr(l) for l in s.lenses] or ["amount(:PHASE)"]

        self.w(f"const cell = {call}")
        self.w(f"const scheme = {scheme}")
        self.blank()
        if s.kind == "derivative":
            self.w("const param = " + lenses[0])
        else:
            self.w("const params = [")
            for l in lenses:
                self.w(f"{IND}{l},")
            self.w("]")
        self.blank()

        indexer = ""
        if s.kind != "jacobian":
            # The projection sits inside the differentiated function, not after
            # it: `best_fit_*` is a least-squares fit, so what comes out is the
            # derivative of the *reported* quantity, which is the one asked for.
            fn = self._PROJECTIONS.get(s.projection)
            var = f"{fn}(C)" if fn else "C"
            indexer = f", indexer = C -> {self._output_expr(s.output, var)}"
        arg = "param" if s.kind == "derivative" else "params"
        self.w("# `get_param` reads the point the derivative is taken at, so the")
        self.w("# values in the model are the x₀ — nothing is entered twice.")
        self.w(f"const D = {s.kind}(cell, scheme, {arg}; output = {s.property}{indexer})")
        self.blank()

        label = self._output_label(s.output)
        if s.kind == "derivative":
            self.w(f'@printf "d({label}) / d(%s) = %.8g\\n" "{self._lens_label(s.lenses[0])}" D')
            return

        self.w("const LABELS = [")
        for l in s.lenses:
            self.w(f'{IND}"{self._lens_label(l)}",')
        self.w("]")
        if s.kind == "gradient":
            self.w(f'@printf "%-34s  d({label}) / d(p)\\n" "parameter"')
            self.w("for (name, g) in zip(LABELS, D)")
            self.w(f'{IND}@printf "%-34s  %.8g\\n" name g')
            self.w("end")
            return

        self.w("# One row per component of the flattened effective tensor.")
        self.w('println("jacobian ", size(D, 1), " x ", size(D, 2), "  columns: ",')
        self.w(f'{IND}join(LABELS, ", "))')
        self.w("for i in axes(D, 1)")
        # The row is taken once. `@view` allocates nothing, but naming it says
        # plainly that the test and the print look at the same thing.
        self.w(f"{IND}row = @view D[i, :]")
        self.w(f"{IND}any(!iszero, row) || continue")
        self.w(f'{IND}@printf "  [%3d]  %s\\n" i join(')
        self.w(f'{IND}{IND}[@sprintf("%12.6g", x) for x in row], "")')
        self.w("end")

    @staticmethod
    def _lens_label(lens) -> str:
        """How a parameter is named in the printed table."""
        k = lens.kind
        if k == "amount":
            return f"amount({lens.phase})"
        if k == "thickness":
            return f"thickness({lens.phase})"
        if k == "interface_param":
            return f"interface {lens.index}.{lens.field_name}"
        if k == "property":
            return f"{lens.phase}{lens.property}[{lens.index}]"
        if k == "geometry":
            return f"{lens.phase}.{lens.field_name}[{lens.index}]"
        if k == "shape_param":
            return f"shape.{lens.field_name}[{lens.index}]"
        if k == "nested":
            from .model import Lens as _L

            inner = _L.from_dict(lens.inner or {})
            return f"{lens.member}{lens.property} → {CodeGen._lens_label(inner)}"
        return k

    # -- ageing viscoelasticity -------------------------------------------

    def _alv_main(self, root: Cell) -> None:
        alv = self.m.alv
        cell = self.m.cell(alv.cell) or root
        call = f"{cell.builder}(" + ", ".join(cell.params) + ")"
        i, j = int(alv.component[0]), int(alv.component[1])
        names_schemes = self._scheme_list(self.m.sweep)

        if alv.log_time:
            self.w(
                f"const times = vcat(0.0, 10 .^ range({_fnum(alv.t_start)}, "
                f"{_fnum(alv.t_stop)}; length = {_num(alv.length)}))"
            )
        else:
            self.w(
                f"const times = range({_fnum(alv.t_start)}, {_fnum(alv.t_stop)}; "
                f"length = {_num(alv.length)})"
            )
        self.blank()
        self.w(f"const cell = {call}")
        self.blank()
        self.w("#")
        self.w("# `homogenize_alv` returns the effective relaxation operator as a")
        self.w("# 6n x 6n block matrix on the time grid. Its Volterra inverse is")
        self.w("# the creep operator, and the response to a unit stress step is")
        self.w(f"# the row sum of its ({i}{j}) blocks.")
        self.w("function uniaxial(R)")
        self.w(f"{IND}J = volterra_inverse(R; block_size = 6)")
        self.w(f"{IND}n = size(J, 1) ÷ 6")
        self.w(
            f"{IND}return [sum(J[6 * (a - 1) + {i}, 6 * (b - 1) + {j}] for b in 1:n) "
            f"for a in 1:n]"
        )
        self.w("end")
        self.blank()
        self.w("const RESULTS = Dict{String, Any}()")
        for scheme_name, scheme in names_schemes:
            self.w(
                f'RESULTS["{scheme_name}"] = uniaxial(homogenize_alv(cell, {scheme}, '
                f"{alv.property}; times = times))"
            )
        self.blank()
        self.w("# Published for MFH Studio; harmless when the script runs alone.")
        self.w(
            'const MFHSTUDIO_RESULTS = merge('
            'Dict("x" => collect(times), "xlabel" => "t"), RESULTS)'
        )
        self.blank()
        # Sorted once, as in the sweep: the table and the figure show the same
        # curves in the same order.
        self.w("const CURVES = sort!(collect(RESULTS); by = first)")
        self.blank()
        self.w("for (label, ys) in CURVES")
        self.w(f'{IND}@printf "%-24s  J(t₁) = %.6g   J(t_end) = %.6g\\n" label first(ys) last(ys)')
        self.w("end")
        if alv.plot:
            self.blank()
            self.w(
                'p = plot(; xlabel = "t", ylabel = "uniaxial creep", legend = :best'
                + (", xscale = :log10)" if alv.log_time else ")")
            )
            self.w("for (label, ys) in CURVES")
            self.w(f"{IND}plot!(p, times, ys; label = label, lw = 2)")
            self.w("end")
            self.w("display(p)")

    # -- the embedded model ----------------------------------------------

    def _embedded_model(self) -> None:
        if not self.embed:
            return
        self.blank()
        self.w("#=" + f" {MODEL_TAG}")
        self.w("The studio reads this block to reopen the model exactly as it was.")
        self.w("Deleting it costs nothing but a best-effort re-reading of the code.")
        self.w(json.dumps(self.m.to_dict(), indent=1, sort_keys=True))
        self.w("=#")

    # -- lenses -----------------------------------------------------------

    def _lens_expr(self, lens) -> str:
        k = lens.kind
        if k == "amount":
            return f"amount(:{lens.phase})"
        if k == "property":
            return f"property(:{lens.phase}, {lens.property}, {lens.index})"
        if k == "geometry":
            return f"geometry(:{lens.phase}, :{lens.field_name}, {lens.index})"
        if k == "shape_param":
            return f"shape_param(:{lens.field_name}, {lens.index})"
        if k == "thickness":
            return f"thickness(:{lens.phase})"
        if k == "interface_param":
            # The field is a Symbol naming one scalar of the interface — `:kn`,
            # `:κs`, `:resistance`, … — and the index is the interface's
            # position, on top of the layer of the same index.
            return f"interface_param({lens.index}, :{lens.field_name})"
        if k == "nested":
            from .model import Lens as _L

            inner = _L.from_dict(lens.inner or {})
            return f"nested(:{lens.member}, {lens.property}, {self._lens_expr(inner)})"
        return f"amount(:{lens.phase})"


def _toplevel_kw_split(body: str) -> Optional[int]:
    """Index of the `; ` that separates positional from keyword arguments.

    Only a separator at bracket depth zero belongs to this call; any deeper one
    belongs to a nested constructor.
    """
    depth = 0
    for i, ch in enumerate(body):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth == 0:
            return i
    return None


def generate(model: Model, embed_model: bool = True) -> str:
    return CodeGen(model, embed_model).generate()


def cell_expression(model: Model, cell: Cell) -> str:
    """The cell as one Julia expression, for the 3-D view.

    A phase's shape is already an expression (`Spheroid(0.4)`) and the viewer
    evaluates it directly. A laminate has no per-member shape — the geometry of
    the whole cell *is* the stack — so what has to be drawn is the cell, and a
    cell takes several statements to build. `let … end` is the expression form
    of exactly those statements, so the picture is built from the same code as
    the script rather than from a parallel description of it.

    A builder taking parameters cannot be drawn this way; the caller checks.
    """
    g = CodeGen(model, embed_model=False)
    body = []
    for ln in g._cell_body(cell):
        if not ln.strip():
            continue
        # `let` yields its last expression: the `return` has to go, and the
        # lines keep the indentation the builder gave them.
        stripped = ln.lstrip()
        if stripped.startswith("return "):
            ln = ln[: len(ln) - len(stripped)] + stripped[len("return "):]
        body.append(ln)
    return "let\n" + "\n".join(body) + "\nend"


def render_cell(model: Model, cell: Cell) -> str:
    """Just one builder, as it would appear in the script.

    Used by the reader to check that a cell it *thinks* it understood would be
    written back unchanged. A construct the studio can parse but not reproduce
    is one it should leave alone.
    """
    g = CodeGen(model, embed_model=False)
    g._builder(cell)
    return "\n".join(g.out).rstrip()


def extract_embedded(source: str) -> Optional[dict]:
    """The model a studio-written script carries, if any."""
    i = source.find("#=" + f" {MODEL_TAG}")
    if i < 0:
        return None
    j = source.find("\n=#", i)
    if j < 0:
        return None
    body = source[i:j]
    brace = body.find("{")
    if brace < 0:
        return None
    try:
        return json.loads(body[brace:])
    except json.JSONDecodeError:
        return None
