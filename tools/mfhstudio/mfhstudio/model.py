"""The authoring model.

This is what the interface edits and what the code generator reads. It is a
*graph of cells*, not a single RVE, because MeanFieldHomogenization chains scales
declaratively: a phase property may hold a `Homogenized(inner_cell, scheme)`
instead of a tensor, and the outer scheme resolves the inner scale when it
reads the property.

Everything is plain dataclasses with `to_dict`/`from_dict`, so the whole model
serializes to JSON — for the browser, for the embedded round-trip block, and
for the test fixtures.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


#: The only schemes defined for a `Laminate`. `Laminated` is the exact
#: periodic solution; `Voigt` and `Reuss` apply because neither needs a matrix
#: phase, and they bracket it — two of the bracketings being equalities. Every
#: other scheme needs a matrix and a reference medium, and MeanFieldHomogenization
#: says so explicitly rather than returning a wrong number.
LAMINATE_SCHEMES = ("Laminated", "Voigt", "Reuss")


# ---------------------------------------------------------------------------
# Values: either a literal, a named parameter, or a nested scale
# ---------------------------------------------------------------------------


@dataclass
class Property:
    """One entry of a phase's property dictionary.

    `source` selects where the value comes from:

    - ``"builder"``  — a tensor built from moduli, e.g. ``iso_stiffness(k, μ)``
    - ``"expr"``     — a raw Julia expression the user typed
    - ``"cell"``     — **the multiscale seam**: the value is the effective
      property of another cell, emitted as ``Homogenized(cell, scheme)``
    """

    key: str = ":C"
    source: str = "builder"
    builder: str = "iso_stiffness"
    #: which catalog entry produced this. Several entries share one builder
    #: (`iso_stiffness` backs both the plain isotropic form and the near-zero
    #: pore preset), so the builder alone cannot identify the form.
    form: str = "iso_kmu"
    args: dict = field(default_factory=lambda: {"k": 10.0, "mu": 5.0})
    expr: str = ""
    #: ZYZ Euler angles of the frame the anisotropic constants are written in.
    #: A tensor's frame is not the inclusion's: a tilted fiber in an untilted
    #: matrix and an untilted fiber in a tilted matrix are different materials,
    #: so the two orientations are stored and emitted separately.
    euler_angles: list = field(default_factory=list)
    # multiscale seam
    cell: Optional[str] = None
    scheme: Optional[str] = None
    scheme_options: dict = field(default_factory=dict)
    # viscoelastic law instead of a tensor
    visco: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Property":
        return Property(**{k: v for k, v in d.items() if k in Property.__annotations__})


@dataclass
class Geometry:
    kind: str = "spheroid"
    args: dict = field(default_factory=lambda: {"omega": 1.0})
    euler_angles: list = field(default_factory=list)
    #: layered inclusions only: list of {radius, property, interface}
    layers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Geometry":
        return Geometry(**{k: v for k, v in d.items() if k in Geometry.__annotations__})


@dataclass
class Phase:
    name: str = "PHASE"
    is_matrix: bool = False
    geometry: Geometry = field(default_factory=Geometry)
    properties: list = field(default_factory=list)  # list[Property]
    #: ("fraction" | "density", value-or-parameter-name); ignored for the matrix,
    #: whose amount MFH derives as 1 - Σ f_inclusions and refuses to be set.
    amount_kind: str = "fraction"
    amount: Any = 0.1
    symmetrize: str = "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["geometry"] = self.geometry.to_dict()
        d["properties"] = [
            p.to_dict() if isinstance(p, Property) else p for p in self.properties
        ]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Phase":
        return Phase(
            name=d.get("name", "PHASE"),
            is_matrix=bool(d.get("is_matrix", False)),
            geometry=Geometry.from_dict(d.get("geometry", {})),
            properties=[Property.from_dict(p) for p in d.get("properties", [])],
            amount_kind=d.get("amount_kind", "fraction"),
            amount=d.get("amount", 0.1),
            symmetrize=d.get("symmetrize", "none"),
        )


@dataclass
class Layer:
    """One layer of a `Laminate`.

    A layer is *not* a phase. It carries no inclusion geometry of its own —
    the geometry of a laminate is the stacking direction, and that belongs to
    the cell. What it does carry is a property dictionary, and the same
    `Property` serves here as in a phase, which is what gives a layer the
    multiscale seam, the viscoelastic laws and the anisotropic forms for free.

    `amount_kind` is a *cell-wide* setting mirrored onto each layer for
    convenience: MeanFieldHomogenization refuses a stack that mixes absolute
    thicknesses with volume fractions, because with an imperfect interface the
    period is physically meaningful and a half-specified stack is ambiguous.

    `interface` is the condition **on top of** this layer; the last layer's
    closes the cell back onto the first by periodicity.
    """

    name: str = "A"
    properties: list = field(default_factory=list)  # list[Property]
    amount_kind: str = "fraction"  # fraction | thickness
    amount: Any = 0.5
    interface: dict = field(default_factory=lambda: {"kind": "PerfectInterface", "args": {}})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["properties"] = [
            p.to_dict() if isinstance(p, Property) else p for p in self.properties
        ]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Layer":
        return Layer(
            name=d.get("name", "A"),
            properties=[Property.from_dict(p) for p in d.get("properties", [])],
            amount_kind=d.get("amount_kind", "fraction"),
            amount=d.get("amount", 0.5),
            interface=dict(d.get("interface") or {"kind": "PerfectInterface", "args": {}}),
        )


@dataclass
class Cell:
    """One scale — either an RVE or a laminate.

    `kind` picks which:

    - ``"rve"``      — a matrix with inclusion phases, the random morphology
      the mean-field schemes describe;
    - ``"laminate"`` — a periodic stack of parallel layers of common normal,
      with no matrix and no reference medium, solved *exactly* rather than
      estimated.

    They are two cells, not a cell and an inclusion: a laminate is a unit of
    homogenization, and embedding one in a matrix would need its Hill tensor,
    which MeanFieldHomogenization does not have. So the two live side by side
    in the graph and connect through the same seam.
    """

    id: str = field(default_factory=_new_id)
    name: str = "rve"
    kind: str = "rve"
    matrix_name: str = "MATRIX"
    phases: list = field(default_factory=list)  # list[Phase]
    #: laminate only, in stacking order
    layers: list = field(default_factory=list)  # list[Layer]
    #: how the stacking direction is given: "normal" or "euler"
    frame_mode: str = "normal"
    normal: list = field(default_factory=lambda: [0.0, 0.0, 1.0])
    euler_angles: list = field(default_factory=list)
    #: parameters this cell's builder takes (discovered from the sweep)
    params: list = field(default_factory=list)
    #: the builder's function name. Read-back keeps whatever the file used, so
    #: a script whose builder is called `_ec_equiv` keeps that name and its
    #: callers keep working; only new cells get the `build_<name>` convention.
    builder_name: Optional[str] = None
    #: extra keywords on the RVE constructor, e.g. `T = ComplexF64`
    rve_options: dict = field(default_factory=dict)
    #: where the user dragged this scale in the graph view. Purely cosmetic,
    #: but it rides along in the embedded model so a reopened file looks the
    #: way it was left.
    ui: dict = field(default_factory=lambda: {"x": 40, "y": 40})

    @property
    def builder(self) -> str:
        return self.builder_name or f"build_{self.name}"

    def is_laminate(self) -> bool:
        return self.kind == "laminate"

    def members(self) -> list:
        """The property-carrying members, whichever kind of cell this is.

        Phases for an RVE, layers for a laminate. Both have a `name` and a
        `properties` list, and that is all the multiscale seam ever needs, so
        the dependency walk, the validation and the ports are written once
        against this view rather than twice against the two shapes.
        """
        return self.layers if self.is_laminate() else self.phases

    def matrix(self) -> Optional[Phase]:
        return next((p for p in self.phases if p.is_matrix), None)

    def inclusions(self) -> list:
        return [p for p in self.phases if not p.is_matrix]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "matrix_name": self.matrix_name,
            "phases": [p.to_dict() for p in self.phases],
            "layers": [l.to_dict() for l in self.layers],
            "frame_mode": self.frame_mode,
            "normal": list(self.normal),
            "euler_angles": list(self.euler_angles),
            "params": list(self.params),
            "builder_name": self.builder_name,
            "rve_options": dict(self.rve_options),
            "ui": dict(self.ui),
        }

    @staticmethod
    def from_dict(d: dict) -> "Cell":
        return Cell(
            id=d.get("id") or _new_id(),
            name=d.get("name", "rve"),
            kind=d.get("kind", "rve"),
            matrix_name=d.get("matrix_name", "MATRIX"),
            phases=[Phase.from_dict(p) for p in d.get("phases", [])],
            layers=[Layer.from_dict(l) for l in d.get("layers", [])],
            frame_mode=d.get("frame_mode", "normal"),
            normal=list(d.get("normal") or [0.0, 0.0, 1.0]),
            euler_angles=list(d.get("euler_angles") or []),
            params=list(d.get("params", [])),
            builder_name=d.get("builder_name"),
            rve_options=dict(d.get("rve_options", {})),
            ui=dict(d.get("ui") or {"x": 40, "y": 40}),
        )


# ---------------------------------------------------------------------------
# Parameters, sweeps, outputs
# ---------------------------------------------------------------------------


@dataclass
class Param:
    """A named constant emitted as `const name = value`.

    `origin` holds the exact text the parameter had in the file it was read
    from. As long as the user has not edited it, that text is what gets written
    back — so a carefully laid-out multi-line constant survives instead of
    being collapsed into one line by the AST round-trip. Re-formatting code the
    interface did not author is a form of damage, even when the meaning is
    preserved.
    """

    name: str = "k"
    value: str = "1.0"
    comment: str = ""
    origin: Optional[str] = None
    edited: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Param":
        return Param(**{k: v for k, v in d.items() if k in Param.__annotations__})


@dataclass
class Lens:
    """A sensitivity/sweep lens, mirroring `src/Schemes/parameters.jl`.

    `nested` wraps another lens to reach into an inner scale, which is how a
    sweep crosses scales without any hand-written closure.

    `thickness` and `interface_param` are the two lenses a laminate adds.
    `property` is shared with the RVE, its `phase` field naming a layer there.
    """

    #: amount | property | geometry | shape_param | nested
    #: | thickness | interface_param
    kind: str = "amount"
    phase: str = ""
    property: str = ":C"
    field_name: str = "semi_axes"
    index: int = 1
    member: str = ""
    inner: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Lens":
        return Lens(**{k: v for k, v in d.items() if k in Lens.__annotations__})


@dataclass
class Sweep:
    """What to compute and what to plot.

    `mode` decides the shape of the run:

    - ``"single"`` — homogenize once with the amounts entered in Scales. This
      is the answer to "I just want the number for the fractions I typed".
    - ``"sweep"``  — vary one lens over a range and plot.

    `schemes` is a *list*, so several can share one figure; comparing schemes
    on the same microstructure is the usual reason to draw one at all.

    `outputs` are explicit specs rather than the fixed `k`/`μ` pair: those two
    only exist for an isotropic result, and an oriented inclusion without an
    orientation average does not give one.
    """

    enabled: bool = False
    mode: str = "sweep"
    variable: str = "φ"
    start: float = 0.0
    stop: float = 1.0
    length: int = 21
    lens: Lens = field(default_factory=Lens)
    cell: Optional[str] = None
    #: [{"name": "MoriTanaka", "options": {...}}, …]
    schemes: list = field(default_factory=lambda: [{"name": "MoriTanaka", "options": {}}])
    property: str = ":C"
    projection: str = "none"
    outputs: list = field(default_factory=lambda: [{"kind": "k"}, {"kind": "mu"}])
    #: clip negative values to zero when reporting. A scheme pushed outside its
    #: range gives negative moduli — `Dilute` does, well before f = 1 — and on
    #: a comparison figure that one curve sets the scale for all the others.
    #: It is a *display* choice, not a correction: the run is unchanged, and a
    #: single scheme plotted alone is better left unclamped, where a negative
    #: modulus is the useful signal that the estimate has stopped meaning
    #: anything.
    clamp_zero: bool = False
    plot: bool = True

    #: kinds that are only defined for an isotropic tensor
    ISOTROPIC_ONLY = ("k", "mu", "E", "nu")

    def needs_isotropy(self) -> bool:
        return any(o.get("kind") in Sweep.ISOTROPIC_ONLY for o in self.outputs)

    def to_dict(self) -> dict:
        d = {
            k: v for k, v in asdict(self).items()
            if k not in ("lens", "ISOTROPIC_ONLY")
        }
        d["lens"] = self.lens.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Sweep":
        s = Sweep(
            **{
                k: v for k, v in d.items()
                if k in Sweep.__annotations__ and k not in ("lens", "schemes", "outputs")
            }
        )
        s.lens = Lens.from_dict(d.get("lens", {}))

        # Models written before schemes became a list, and before outputs were
        # specs, still open: migrate rather than lose them.
        schemes = d.get("schemes")
        if not schemes:
            schemes = [{
                "name": d.get("scheme") or "MoriTanaka",
                "options": dict(d.get("scheme_options") or {}),
            }]
        s.schemes = [
            {"name": x.get("name", "MoriTanaka"), "options": dict(x.get("options") or {})}
            for x in schemes
        ]

        outs = d.get("outputs") or []
        s.outputs = [
            {"kind": o} if isinstance(o, str) else dict(o) for o in outs
        ] or [{"kind": "k"}, {"kind": "mu"}]
        return s


@dataclass
class Sens:
    """Autodiff sensitivities of the effective property.

    MeanFieldHomogenization's wrappers take the cell and the lens directly —
    `derivative(cell, scheme, param; output, indexer)` — so there is no closure
    to write here. The lens is the one the sweep already models, and the
    `indexer` is the scalar extraction the sweep already emits: this panel is
    the two of them put together, not new machinery.

    `kind`:

    - ``"derivative"`` — exactly one lens, `f'(x₀)`;
    - ``"gradient"``   — several lenses, the gradient of one scalar;
    - ``"jacobian"``   — several lenses, the whole effective tensor flattened.

    A gradient is not a curve, so nothing is plotted: the answer is a table.
    """

    enabled: bool = False
    kind: str = "derivative"
    cell: Optional[str] = None
    scheme: str = "MoriTanaka"
    scheme_options: dict = field(default_factory=dict)
    property: str = ":C"
    lenses: list = field(default_factory=lambda: [Lens()])  # list[Lens]
    #: the scalar read off the effective tensor, same specs as `Sweep.outputs`
    output: dict = field(default_factory=lambda: {"kind": "k"})
    #: reporting projection applied before the extraction, as in the sweep.
    #: `best_fit_*` is a least-squares fit, so differentiating through it is
    #: meaningful — and it is the same escape the sweep offers when `k` is
    #: asked of a tensor that need not be isotropic.
    projection: str = "iso"

    def needs_isotropy(self) -> bool:
        return (
            self.output.get("kind") in Sweep.ISOTROPIC_ONLY
            and self.projection == "none"
        )

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "lenses"}
        d["lenses"] = [
            l.to_dict() if isinstance(l, Lens) else l for l in self.lenses
        ]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Sens":
        s = Sens(
            **{
                k: v for k, v in d.items()
                if k in Sens.__annotations__ and k not in ("lenses", "output")
            }
        )
        s.lenses = [Lens.from_dict(x) for x in d.get("lenses") or []] or [Lens()]
        out = d.get("output")
        s.output = {"kind": out} if isinstance(out, str) else dict(out or {"kind": "k"})
        # `jacobian` flattens the whole tensor, so no scalar is extracted; the
        # other two need one.
        if s.kind == "derivative":
            s.lenses = s.lenses[:1]
        return s


@dataclass
class Alv:
    """Ageing linear viscoelasticity settings."""

    enabled: bool = False
    t_start: float = 0.0
    t_stop: float = 10.0
    length: int = 41
    log_time: bool = False
    cell: Optional[str] = None
    scheme: str = "MoriTanaka"
    property: str = ":C"
    #: which Kelvin-Mandel component of the creep operator to follow; (1, 1)
    #: is the uniaxial response.
    component: list = field(default_factory=lambda: [1, 1])
    plot: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Alv":
        a = Alv(**{k: v for k, v in d.items() if k in Alv.__annotations__})
        a.component = list(a.component or [1, 1])[:2] or [1, 1]
        return a


@dataclass
class OpaqueBlock:
    """Source the studio did not recognize, preserved verbatim.

    This is the whole reason the interface is safe to point at a hand-written
    script: what it does not understand, it does not touch. The block is
    re-emitted byte for byte and shown read-only in the UI.
    """

    source: str = ""
    #: where it sat in the original file, so ordering survives
    order: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "OpaqueBlock":
        return OpaqueBlock(
            **{k: v for k, v in d.items() if k in OpaqueBlock.__annotations__}
        )


# ---------------------------------------------------------------------------
# The whole model
# ---------------------------------------------------------------------------


@dataclass
class Model:
    title: str = "model"
    description: str = ""
    params: list = field(default_factory=list)   # list[Param]
    cells: list = field(default_factory=list)    # list[Cell]
    sweep: Sweep = field(default_factory=Sweep)
    sens: Sens = field(default_factory=Sens)
    alv: Alv = field(default_factory=Alv)
    opaque: list = field(default_factory=list)   # list[OpaqueBlock]
    #: id of the cell that carries the final result
    root_cell: Optional[str] = None

    # -- lookups ----------------------------------------------------------

    def cell(self, cid: Optional[str]) -> Optional[Cell]:
        if cid is None:
            return None
        return next((c for c in self.cells if c.id == cid), None)

    def cell_by_name(self, name: str) -> Optional[Cell]:
        return next((c for c in self.cells if c.name == name), None)

    def root(self) -> Optional[Cell]:
        return self.cell(self.root_cell) or (self.cells[-1] if self.cells else None)

    # -- the multiscale graph --------------------------------------------

    def dependencies(self, cell: Cell) -> list:
        """The cells this one reads through a `Homogenized` seam.

        Over `members()`, not `phases`: a laminate's layers carry the seam
        exactly as a phase does, and walking only the phases would leave an
        inner scale out of the topological order — the script would then call
        a builder before defining it.
        """
        out = []
        for mb in cell.members():
            for pr in mb.properties:
                if pr.source == "cell" and pr.cell:
                    out.append(pr.cell)
        return out

    def topological_order(self) -> list:
        """Cells ordered so every cell comes after the ones it depends on.

        Raises `ValueError` naming the cycle when the graph has one — a scale
        cannot be built from itself, and catching it here means the interface
        refuses at construction rather than emitting a script that recurses
        forever.
        """
        state: dict = {}
        order: list = []

        def visit(cid: str, trail: list) -> None:
            st = state.get(cid)
            if st == "done":
                return
            if st == "visiting":
                names = [self.cell(x).name if self.cell(x) else x for x in trail + [cid]]
                start = names.index(names[-1])
                raise ValueError(
                    "multiscale cycle: " + " → ".join(names[start:])
                )
            c = self.cell(cid)
            if c is None:
                return
            state[cid] = "visiting"
            for dep in self.dependencies(c):
                visit(dep, trail + [cid])
            state[cid] = "done"
            order.append(c)

        for c in self.cells:
            visit(c.id, [])
        return order

    def uses_multiscale(self) -> bool:
        return any(self.dependencies(c) for c in self.cells)

    def validate(self) -> list:
        """Problems worth blocking on, each as a human-readable string."""
        problems = []
        try:
            self.topological_order()
        except ValueError as exc:
            problems.append(str(exc))

        names = [c.name for c in self.cells]
        for n in set(names):
            if names.count(n) > 1:
                problems.append(f"two cells are both named `{n}`")

        for c in self.cells:
            if c.is_laminate():
                problems.extend(self._laminate_problems(c))
            elif c.matrix() is None:
                problems.append(f"cell `{c.name}` has no matrix phase")
            what = "layers" if c.is_laminate() else "phases"
            mn = [m.name for m in c.members()]
            for n in set(mn):
                if mn.count(n) > 1:
                    problems.append(f"cell `{c.name}` has two {what} named `{n}`")

        problems.extend(self._scheme_problems())

        # `k_mu` and `E_nu` have methods for TensISO only. Asking for them
        # from an oriented inclusion with no orientation average throws a
        # MethodError deep in the run; saying it here costs nothing.
        # A laminate of more than one layer is transversely isotropic about its
        # normal even when every layer is isotropic — that is the whole point
        # of Backus — so it lands in exactly the same trap.
        anisotropic = any(
            ph.symmetrize == "none" for c in self.cells for ph in c.phases
        ) or any(len(c.layers) > 1 for c in self.cells if c.is_laminate())
        if (
            self.sweep.enabled
            and self.sweep.needs_isotropy()
            and self.sweep.projection == "none"
            and anisotropic
        ):
            problems.append(
                "k, μ, E and ν are only defined for an isotropic result. This "
                "model has phases with no orientation average, so the effective "
                "tensor need not be isotropic: pick a reporting projection, or "
                "plot Kelvin-Mandel components instead."
            )

        if self.sens.enabled:
            problems.extend(self._sens_problems())

        # Documented MFH constraint: an inner Homogenized cannot sit inside an
        # ageing-viscoelastic chain, because the inner result would have to be
        # re-expressible as a ViscoLaw (src/Core/cells.jl).
        if self.alv.enabled and self.uses_multiscale():
            problems.append(
                "ageing viscoelasticity cannot be combined with a nested scale: "
                "MeanFieldHomogenization cannot re-express a homogenized inner result as a "
                "ViscoLaw"
            )
        return problems

    # -- per-kind checks ---------------------------------------------------

    @staticmethod
    def _laminate_problems(c: Cell) -> list:
        """What MeanFieldHomogenization refuses about a stack, said early.

        Each of these raises inside `add_layer!` or `validate_laminate`; saying
        it here turns a stack trace at the end of a run into a sentence beside
        the form.
        """
        problems = []
        if not c.layers:
            problems.append(f"laminate `{c.name}` has no layer")
            return problems

        kinds = {l.amount_kind for l in c.layers}
        if len(kinds) > 1:
            problems.append(
                f"laminate `{c.name}` mixes absolute thicknesses with volume "
                "fractions. MeanFieldHomogenization refuses that: with an "
                "imperfect interface the period carries the size effect, so a "
                "half-specified stack is ambiguous."
            )
        elif kinds == {"fraction"}:
            # `validate_laminate` checks Σf ≈ 1 rather than rescaling silently,
            # so a stack summing to anything else is an error, not a hint.
            vals = [l.amount for l in c.layers]
            if all(isinstance(v, (int, float)) for v in vals):
                total = sum(vals)
                if abs(total - 1.0) > 1.0e-8:
                    problems.append(
                        f"the fractions of laminate `{c.name}` sum to "
                        f"{total:.6g}, not 1"
                    )
        for l in c.layers:
            if isinstance(l.amount, (int, float)) and l.amount < 0:
                problems.append(f"layer `{l.name}` has a negative {l.amount_kind}")
        if c.frame_mode == "normal" and all(
            isinstance(x, (int, float)) and x == 0 for x in (c.normal or [])
        ):
            problems.append(f"laminate `{c.name}` has a null normal")
        return problems

    def _sens_problems(self) -> list:
        s = self.sens
        problems = []
        if not s.lenses:
            problems.append("the sensitivity needs at least one parameter")
        if s.kind == "derivative" and len(s.lenses) != 1:
            problems.append(
                "a derivative is with respect to one parameter; use a gradient "
                "for several"
            )
        target = self.cell(s.cell) or self.root()
        if target is not None and target.is_laminate():
            # `AmountParameter` raises on a Laminate and points at
            # `ThicknessParameter`; the interface can say so first.
            for l in s.lenses:
                if l.kind == "amount":
                    problems.append(
                        "a laminate has no phase amount: differentiate a layer "
                        "thickness instead (changing hᵢ also moves the period, "
                        "which is what carries the interface size effect)."
                    )
                    break
        for l in s.lenses:
            if l.kind in ("thickness", "interface_param") and (
                target is None or not target.is_laminate()
            ):
                problems.append(
                    f"the `{l.kind}` parameter only exists on a laminate"
                )
                break
        if s.kind != "jacobian" and s.needs_isotropy():
            free = target is not None and (
                target.is_laminate() and len(target.layers) > 1
                or any(ph.symmetrize == "none" for ph in target.phases)
            )
            if free:
                problems.append(
                    "k, μ, E and ν are only defined for an isotropic result. "
                    "Pick a reporting projection, or differentiate a "
                    "Kelvin-Mandel component instead."
                )
        return problems

    def _scheme_problems(self) -> list:
        """Schemes asked of a cell that does not support them."""
        problems = []

        def check(cid, name, where):
            c = self.cell(cid) or self.root()
            if c is None or not c.is_laminate():
                return
            if name not in LAMINATE_SCHEMES:
                problems.append(
                    f"{where}: `{name}` needs a matrix phase, so it does not "
                    f"apply to the laminate `{c.name}`. A laminate takes "
                    + ", ".join(f"`{s}`" for s in LAMINATE_SCHEMES) + "."
                )

        if self.sweep.enabled:
            for s in self.sweep.schemes:
                check(self.sweep.cell, s.get("name", "MoriTanaka"), "Sweep")
        if self.alv.enabled:
            check(self.alv.cell, self.alv.scheme, "Viscoelastic")
            # `laminate_alv` builds the order-2 kernel in the canonical frame
            # and says so; a tilted stack has to stay on `:C`.
            c = self.cell(self.alv.cell) or self.root()
            if (
                c is not None and c.is_laminate() and self.alv.property != ":C"
                and (c.frame_mode != "normal" or list(c.normal) != [0.0, 0.0, 1.0])
            ):
                problems.append(
                    "ageing viscoelasticity of a laminate in transport (`:K`) "
                    "requires the canonical frame: MeanFieldHomogenization "
                    "builds that kernel with the normal along e₃."
                )
        if self.sens.enabled:
            check(self.sens.cell, self.sens.scheme, "Sensitivity")
        # Every phase property a `Homogenized` reads carries its own scheme.
        for c in self.cells:
            for mb in c.members():
                for pr in mb.properties:
                    if pr.source == "cell" and pr.cell:
                        check(pr.cell, pr.scheme or "MoriTanaka",
                              f"`{c.name}`/`{mb.name}`{pr.key}")
        return problems

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "cells": [c.to_dict() for c in self.cells],
            "sweep": self.sweep.to_dict(),
            "sens": self.sens.to_dict(),
            "alv": self.alv.to_dict(),
            "opaque": [o.to_dict() for o in self.opaque],
            "root_cell": self.root_cell,
        }

    @staticmethod
    def from_dict(d: dict) -> "Model":
        m = Model(
            title=d.get("title", "model"),
            description=d.get("description", ""),
            params=[Param.from_dict(p) for p in d.get("params", [])],
            cells=[Cell.from_dict(c) for c in d.get("cells", [])],
            opaque=[OpaqueBlock.from_dict(o) for o in d.get("opaque", [])],
            root_cell=d.get("root_cell"),
        )
        m.sweep = Sweep.from_dict(d.get("sweep", {}))
        m.sens = Sens.from_dict(d.get("sens", {}))
        m.alv = Alv.from_dict(d.get("alv", {}))
        return m


def default_model() -> Model:
    """The porous benchmark, which is the shortest useful thing to open on."""
    solid = Phase(
        name="SOLID", is_matrix=True,
        geometry=Geometry(kind="spheroid", args={"omega": 1.0}),
        properties=[
            Property(key=":C", builder="iso_stiffness", form="iso_kmu",
                     args={"k": 72.0, "mu": 32.0})
        ],
    )
    pore = Phase(
        name="PORE", is_matrix=False,
        geometry=Geometry(kind="spheroid", args={"omega": 1.0}),
        properties=[
            Property(key=":C", builder="iso_stiffness", form="void",
                     args={"k": 1.0e-6, "mu": 1.0e-6})
        ],
        amount_kind="fraction", amount=0.1,
    )
    cell = Cell(name="rve", matrix_name="SOLID", phases=[solid, pore])
    m = Model(title="porous_benchmark", cells=[cell], root_cell=cell.id)
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="φ", start=0.0, stop=0.9, length=19,
        lens=Lens(kind="amount", phase="PORE"), cell=cell.id,
        schemes=[{"name": "MoriTanaka", "options": {}}],
        projection="iso", outputs=[{"kind": "k"}, {"kind": "mu"}],
    )
    return m


def default_layers() -> list:
    """The 30/70 bilayer of `scripts/33_laminate_basics.jl`.

    A stiff layer and a compliant one is the shortest stack that shows what a
    laminate is: the effective stiffness comes out transversely isotropic about
    the normal even though both layers are isotropic.
    """
    return [
        Layer(
            name="A", amount_kind="fraction", amount=0.3,
            properties=[
                Property(key=":C", builder="iso_stiffness", form="iso_kmu",
                         args={"k": 2.0, "mu": 0.8})
            ],
        ),
        Layer(
            name="B", amount_kind="fraction", amount=0.7,
            properties=[
                Property(key=":C", builder="iso_stiffness", form="iso_kmu",
                         args={"k": 0.5, "mu": 0.2})
            ],
        ),
    ]
