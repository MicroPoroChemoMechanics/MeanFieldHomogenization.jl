"""The intermediate representation.

The IR is a *program model*, not a token stream. Statements that carry MFH
meaning (RVE construction, phase declaration, homogenization, sweeps, plots)
become rich typed nodes that the emitter can restructure idiomatically;
ordinary Python becomes `JuliaStmt` holding already-translated Julia; and only
genuinely unmappable constructs become `Untranslated`.

That split is what lets the emitter produce idiomatic Julia rather than a
line-by-line transcription: it can hoist an RVE construction into a
`build_rve(...)` function, fold Echoes' loose solver keywords into the MFH
scheme constructor, and turn an append-in-a-loop into a comprehension --
none of which is a local rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """Base: every node remembers where it came from."""

    lineno: int = 0
    py_src: str = ""


@dataclass
class JuliaExpr:
    """An already-translated Julia expression plus what we know about it."""

    code: str
    #: 'scalar' | 'tensor4' | 'tensor2' | 'array' | 'rve' | 'scheme' |
    #: 'phase' | 'unknown'
    kind: str = "unknown"

    def __str__(self) -> str:  # convenience in f-strings
        return self.code


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


@dataclass
class Param(Node):
    """A top-level constant: `ks = 72.` or `Cs = stiff_kmu(ks, mus)`."""

    name: str = ""
    value: JuliaExpr = field(default_factory=lambda: JuliaExpr(""))
    is_const: bool = True


@dataclass
class Helper(Node):
    """A pure Python function or lambda: translates near-verbatim.

    These are the hydration formulas, the shape factors, the scalar
    correlations -- the physics that is *not* an Echoes API call. They are
    the bulk of scripts like `cementpaste_mortar_sanahuja.py` and they
    translate essentially one-to-one.
    """

    name: str = ""
    args: list[str] = field(default_factory=list)
    defaults: dict[str, JuliaExpr] = field(default_factory=dict)
    body: list["Stmt"] = field(default_factory=list)
    #: set when the helper was a `lambda`, so the emitter can use `f(x) = ...`
    is_lambda: bool = False


# ---------------------------------------------------------------------------
# The physical model
# ---------------------------------------------------------------------------


@dataclass
class PhaseDef(Node):
    """One phase of an RVE.

    `kind` selects the MFH constructor family:
      'ellipsoid'      -> add_phase!(..., Spheroid/Ellipsoid, ...)
      'crack'          -> add_phase!(..., PennyCrack/EllipticCrack, ...; density=)
      'sphere_nlayers' -> add_phase!(..., LayeredSphere(radii, moduli), ...)
      'spheroid_nlayers' -> add_phase!(..., LayeredSpheroid(...), ...)
      'custom'         -> a CustomInclusion built from a user_inclusion subclass
    """

    name: str = ""
    kind: str = "ellipsoid"
    geometry: JuliaExpr = field(default_factory=lambda: JuliaExpr("Spheroid(1.0)"))
    #: property symbol -> Julia expression, e.g. {':C': 'iso_stiffness(72.0, 32.0)'}
    props: dict[str, JuliaExpr] = field(default_factory=dict)
    #: viscoelastic properties: symbol -> (law expression, ':creep'|':relaxation')
    visco_props: dict[str, tuple[JuliaExpr, str]] = field(default_factory=dict)
    #: ('fraction' | 'density', expr) or None when the phase is the matrix
    amount: Optional[tuple[str, JuliaExpr]] = None
    symmetrize: Optional[str] = None
    is_matrix: bool = False
    #: layered inclusions only
    layers: list[JuliaExpr] = field(default_factory=list)
    interfaces: list[JuliaExpr] = field(default_factory=list)


@dataclass
class RVEDef(Node):
    """An RVE construction, hoisted out of wherever it appeared.

    Echoes mutates a module-level RVE in place; MFH's RVE is immutable
    (`set_param` returns a new one, and the matrix amount is derived, not
    settable). So every RVE becomes a *builder function* parameterized by
    whatever the script varies -- which is also what the hand-written MFH
    demos do (`build_rve(scheme, phi)` in scripts/28_porous_schemes.jl).
    """

    var: str = ""
    matrix_name: str = ""
    phases: list[PhaseDef] = field(default_factory=list)
    #: free parameters the builder must take (discovered from the sweep)
    params: list[str] = field(default_factory=list)
    #: rve.set_ref("C", C0)
    reference: dict[str, JuliaExpr] = field(default_factory=dict)
    #: rve.set_param_eshelby(...) -> per-call kwargs in MFH
    eshelby: dict[str, str] = field(default_factory=dict)
    builder_name: str = ""
    #: built from Echoes' `rvec`, the complex-scalar twin of `rve`. Julia needs
    #: no separate family -- only a complex element type.
    complex_valued: bool = False
    #: True when some `homogenize` on this RVE uses MAX or PCW. Those two read
    #: the distribution shape, and MFH no longer supplies a default one: an
    #: undeclared shape raises rather than quietly collapsing the scheme onto
    #: Mori-Tanaka. Echoes' own default is the unit sphere carried by
    #: `ver.self()`, so that is what the builder has to declare to reproduce
    #: the original run.
    needs_distribution_shape: bool = False


@dataclass
class HomogenizeCall(Node):
    """`homogenize(...)`, `homogenize_visco(...)`, `homogenize_derivative(...)`."""

    #: 'elastic' | 'alv' | 'derivative'
    flavour: str = "elastic"
    rve: JuliaExpr = field(default_factory=lambda: JuliaExpr("rve"))
    scheme: JuliaExpr = field(default_factory=lambda: JuliaExpr("MoriTanaka()"))
    prop: str = ":C"
    kwargs: dict[str, JuliaExpr] = field(default_factory=dict)
    #: alv only
    times: Optional[JuliaExpr] = None
    #: derivative only -- the MFH lens, e.g. `amount(:PORE)`
    lens: Optional[JuliaExpr] = None


@dataclass
class PhaseAccess(Node):
    """`ver["PORE"].eE` -> `strain_strain_loc(...)` and friends.

    Echoes stores localization tensors on the phase after `homogenize`;
    MFH computes them on demand. The emitter therefore needs the RVE, the
    phase name and the reference medium, which this node carries.
    """

    rve: JuliaExpr = field(default_factory=lambda: JuliaExpr("rve"))
    phase: str = ""
    accessor: str = "eE"
    transport: bool = False


# ---------------------------------------------------------------------------
# Control flow and outputs
# ---------------------------------------------------------------------------


@dataclass
class Stmt(Node):
    """Base for anything that lands in a function body or at top level."""


@dataclass
class JuliaStmt(Stmt):
    """A statement we translated expression-wise; `code` is ready to emit."""

    code: str = ""


@dataclass
class Assign(Stmt):
    targets: list[str] = field(default_factory=list)
    value: JuliaExpr = field(default_factory=lambda: JuliaExpr(""))
    #: True for `x += 1` style
    augmented: Optional[str] = None


@dataclass
class If(Stmt):
    test: JuliaExpr = field(default_factory=lambda: JuliaExpr("true"))
    body: list[Stmt] = field(default_factory=list)
    orelse: list[Stmt] = field(default_factory=list)


@dataclass
class For(Stmt):
    targets: list[str] = field(default_factory=list)
    iter: JuliaExpr = field(default_factory=lambda: JuliaExpr("1:1"))
    body: list[Stmt] = field(default_factory=list)


@dataclass
class While(Stmt):
    test: JuliaExpr = field(default_factory=lambda: JuliaExpr("true"))
    body: list[Stmt] = field(default_factory=list)


@dataclass
class TryCatch(Stmt):
    body: list[Stmt] = field(default_factory=list)
    handler: list[Stmt] = field(default_factory=list)


@dataclass
class Return(Stmt):
    value: Optional[JuliaExpr] = None


@dataclass
class Sweep(Stmt):
    """An accumulate-in-a-loop that the emitter can turn into a comprehension.

    Recognized shape:
        k = []
        for x in xs:
            k.append(f(x))
    becomes
        k = [f(x) for x in xs]
    which is the single most common idiom in the corpus and the difference
    between transcribed and idiomatic Julia.
    """

    var: str = ""
    iterable: JuliaExpr = field(default_factory=lambda: JuliaExpr("1:1"))
    #: accumulator name -> element expression
    accumulators: dict[str, JuliaExpr] = field(default_factory=dict)
    body: list[Stmt] = field(default_factory=list)
    #: True when every accumulator is a clean one-liner
    comprehensible: bool = False


@dataclass
class PlotCall(Stmt):
    """One matplotlib call, normalized."""

    func: str = "plot"
    args: list[JuliaExpr] = field(default_factory=list)
    kwargs: dict[str, JuliaExpr] = field(default_factory=dict)
    #: derived from a matplotlib format string like 'r--' or 'g-d'
    style: dict[str, str] = field(default_factory=dict)
    #: a Julia expression evaluating to a format string, decoded at run time
    #: when the style could not be resolved statically
    runtime_style: Optional[str] = None
    #: which figure/axes this belongs to
    axes: str = "p"


@dataclass
class PrintCall(Stmt):
    args: list[JuliaExpr] = field(default_factory=list)
    #: a Julia format string when the Python used % or .format or an f-string
    fmt: Optional[str] = None


@dataclass
class Untranslated(Stmt):
    """A refusal. Carries the original Python so the user can finish in place."""

    reason: str = ""
    suggestion: str = ""
    severity: str = "blocking"  # 'blocking' | 'review'
    symbol: str = ""


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@dataclass
class Script:
    source_path: str = ""
    #: docstring / leading comment block of the original
    header: str = ""
    params: list[Param] = field(default_factory=list)
    helpers: list[Helper] = field(default_factory=list)
    rves: list[RVEDef] = field(default_factory=list)
    body: list[Stmt] = field(default_factory=list)
    findings: list[Untranslated] = field(default_factory=list)
    #: modules the emitter must `using`
    imports: set[str] = field(default_factory=lambda: {"MeanFieldHomogenization", "TensND"})
    #: True when any output is a plot
    needs_plots: bool = False
    #: True when a matplotlib format string had to be decoded at run time
    needs_style_helper: bool = False
    #: True when `tensor(M)` needs run-time dispatch on the array shape
    needs_tensor_helper: bool = False
    #: True when a plot label is not a string literal and needs shortening
    needs_label_helper: bool = False
    #: plot objects the script creates itself (from plt.figure / plt.subplot)
    declared_axes: set[str] = field(default_factory=set)
    #: True when the original used a subplot grid, which Plots.jl declares
    #: with a `layout` at the end rather than by switching axes mid-script
    used_subplots: bool = False

    def finding_count(self, severity: str | None = None) -> int:
        if severity is None:
            return len(self.findings)
        return sum(1 for f in self.findings if f.severity == severity)
