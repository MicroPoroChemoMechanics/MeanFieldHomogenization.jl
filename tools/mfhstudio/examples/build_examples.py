"""Regenerate the studio-format examples.

    python3 examples/build_examples.py

The examples are *generated*, not hand-written, and that is the point. Each one
is a model built with the same dataclasses the interface edits, written out by
the same code generator — so an example cannot drift from what the studio would
produce, and when the emitter changes the examples change with it.

Each file therefore carries its embedded model block and reopens in the studio
exactly as it was left, which a hand-written demo from `scripts/` does not: those
build their cells at top level rather than in a builder function, so the studio
preserves them verbatim instead of offering them for editing.

Every example is a scaled-down sibling of a script under `scripts/`, named in
its own header. Read the script for the physics; open the example to see the
same model as a form you can change.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from mfhstudio.codegen import generate  # noqa: E402
from mfhstudio.model import (  # noqa: E402
    Alv,
    Cell,
    Geometry,
    Layer,
    Lens,
    Model,
    Phase,
    Property,
    Sens,
    Sweep,
)


def iso(k, mu, key=":C", **kw):
    return Property(key=key, builder="iso_stiffness", form="iso_kmu",
                    args={"k": k, "mu": mu}, **kw)


def cond(k, key=":K"):
    return Property(key=key, builder="TensISO{3}", form="iso_conduction",
                    args={"k": k})


def void(key=":C"):
    return Property(key=key, builder="iso_stiffness", form="void",
                    args={"k": 1.0e-6, "mu": 1.0e-6})


# ---------------------------------------------------------------------------
# 01 — the porous benchmark, four schemes on one figure
# ---------------------------------------------------------------------------


def porous_schemes() -> Model:
    """After `scripts/28_porous_schemes.jl` and `scripts/20_voigt_reuss_bounds.jl`.

    Spherical pores in an isotropic solid, swept over the porosity, with every
    scheme the package offers on one figure — which is the usual reason to draw
    one. Voigt and Reuss bracket everything; Dilute and DiluteDual are only
    honest at small ``f`` and leave the bounds beyond it; the self-consistent
    scheme loses all rigidity at ``f = 1/2`` while Mori-Tanaka, Maxwell and PCW
    never do; the differential scheme percolates only at ``f = 1``.

    Solver options ride on the scheme that reads them, not on `homogenize`.
    They are the script's own: `select_best` matters for the self-consistent
    schemes near percolation, where several fixed points exist.

    Negative values are clipped, as the script clips them. `Dilute` goes below
    zero well before ``f = 1``, and on a figure shared by ten schemes that one
    curve would set the scale for all of them. It is a display choice: run a
    single scheme with the box unticked and the negative modulus shows, which
    is the useful signal that the estimate has left its range.
    """
    solid = Phase(name="SOLID", is_matrix=True, properties=[iso(72.0, 32.0)])
    pore = Phase(name="PORE", amount=0.1, properties=[void()])
    c = Cell(name="rve", matrix_name="SOLID", phases=[solid, pore])
    m = Model(title="porous_schemes", cells=[c], root_cell=c.id)
    m.description = (
        "Spherical pores in an isotropic solid, under every scheme at once. "
        "Voigt and Reuss bracket; the self-consistent schemes vanish at "
        "f = 1/2; Mori-Tanaka, Maxwell and PCW never do."
    )
    sc = {"abstol": 1.0e-10, "maxiters": 300, "select_best": True}
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="φ", start=0.0, stop=1.0, length=101,
        lens=Lens(kind="amount", phase="PORE"), cell=c.id, clamp_zero=True,
        schemes=[
            {"name": "Voigt", "options": {}},
            {"name": "Reuss", "options": {}},
            {"name": "Dilute", "options": {}},
            {"name": "DiluteDual", "options": {}},
            {"name": "MoriTanaka", "options": {}},
            {"name": "Maxwell", "options": {}},
            {"name": "PonteCastanedaWillis", "options": {}},
            {"name": "SelfConsistent", "options": dict(sc)},
            {"name": "AsymmetricSelfConsistent", "options": dict(sc)},
            {"name": "DifferentialScheme", "options": {"nsteps": 300}},
        ],
        projection="iso", outputs=[{"kind": "k"}, {"kind": "mu"}],
    )
    return m


# ---------------------------------------------------------------------------
# 02 — oriented cracks, and what an orientation average is for
# ---------------------------------------------------------------------------


def cracked_solid() -> Model:
    """After `scripts/15_cracks_iso_interface.jl` and `scripts/86_crack_distributions.jl`.

    Penny cracks enter with a *density* rather than a volume fraction — a flat
    crack has none — and an isotropic orientation average turns a family of
    parallel cracks into an isotropic damaged solid. Remove the average and the
    result is transversely isotropic, which is why `k` is not asked for here
    without a reporting projection.
    """
    solid = Phase(name="SOLID", is_matrix=True, properties=[iso(72.0, 32.0)])
    cracks = Phase(
        name="CRACKS", amount_kind="density", amount=0.1, symmetrize="iso",
        geometry=Geometry(kind="penny_crack", args={"a": 1.0}),
        properties=[void()],
    )
    c = Cell(name="rve", matrix_name="SOLID", phases=[solid, cracks])
    m = Model(title="cracked_solid", cells=[c], root_cell=c.id)
    m.description = (
        "Penny cracks with a random orientation distribution, averaged exactly "
        "in the kernel. The abscissa is the crack density ε = N a³ / V, not a "
        "volume fraction."
    )
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="ε", start=0.0, stop=0.6, length=25,
        lens=Lens(kind="amount", phase="CRACKS"), cell=c.id,
        schemes=[{"name": "MoriTanaka", "options": {}},
                 {"name": "SelfConsistent", "options": {}}],
        projection="iso", outputs=[{"kind": "k"}, {"kind": "mu"}],
    )
    return m


# ---------------------------------------------------------------------------
# 03 — a coated inclusion
# ---------------------------------------------------------------------------


def coated_inclusion() -> Model:
    """After `scripts/30_average_nlayers.jl`.

    A `LayeredSphere` is one inclusion with concentric shells, given by their
    outer radii with r = 0 implicit at the center — not a nested cell. The
    interface between two shells can be imperfect; here they are perfect and
    the interest is in the coating alone.
    """
    matrix = Phase(name="MATRIX", is_matrix=True, properties=[iso(20.0, 8.0)])
    coated = Phase(
        name="COATED", amount=0.2,
        geometry=Geometry(kind="layered_sphere", args={}, layers=[
            {"radius": 0.7, "interface": {"kind": "PerfectInterface", "args": {}},
             "property": iso(60.0, 30.0).to_dict()},
            {"radius": 1.0, "interface": {"kind": "PerfectInterface", "args": {}},
             "property": iso(5.0, 2.0).to_dict()},
        ]),
        properties=[iso(60.0, 30.0)],
    )
    c = Cell(name="rve", matrix_name="MATRIX", phases=[matrix, coated])
    m = Model(title="coated_inclusion", cells=[c], root_cell=c.id)
    m.description = (
        "A stiff core in a compliant shell, dispersed in a matrix. The shells "
        "are given by outer radius, ascending, with r = 0 at the center."
    )
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="f", start=0.0, stop=0.5, length=21,
        lens=Lens(kind="amount", phase="COATED"), cell=c.id,
        schemes=[{"name": "MoriTanaka", "options": {}}],
        projection="iso", outputs=[{"kind": "k"}, {"kind": "mu"}],
    )
    return m


# ---------------------------------------------------------------------------
# 04 — transport rather than elasticity
# ---------------------------------------------------------------------------


def conductivity_fibers() -> Model:
    """After `scripts/32_spheroid_effective_conductivity.jl`.

    Everything above is stiffness; nothing about the schemes is. Store a
    second-order tensor under `:K`, ask for `:K`, and the same machinery
    answers a conduction problem. The inclusions are prolate spheroids —
    fibers — with no orientation average, so the effective conductivity is
    transversely isotropic and the plot follows components, not a scalar.
    """
    matrix = Phase(name="MATRIX", is_matrix=True, properties=[cond(1.0)])
    fibers = Phase(
        name="FIBERS", amount=0.2,
        geometry=Geometry(kind="spheroid", args={"omega": 10.0}),
        properties=[cond(50.0)],
    )
    c = Cell(name="rve", matrix_name="MATRIX", phases=[matrix, fibers])
    m = Model(title="conductivity_fibers", cells=[c], root_cell=c.id)
    m.description = (
        "Aligned conductive fibers (ω = 10) in a poor matrix. The result is "
        "transversely isotropic about e₃: K₃₃ follows the fibers, K₁₁ does not."
    )
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="f", start=0.0, stop=0.4, length=21,
        lens=Lens(kind="amount", phase="FIBERS"), cell=c.id,
        schemes=[{"name": "MoriTanaka", "options": {}}],
        property=":K", projection="none",
        outputs=[{"kind": "comp", "i": 1, "j": 1}, {"kind": "comp", "i": 3, "j": 3}],
    )
    return m


# ---------------------------------------------------------------------------
# 05 — two scales
# ---------------------------------------------------------------------------


def two_scales() -> Model:
    """After `scripts/42_cementpaste_iso.jl`.

    The multiscale seam: a phase property of the outer cell holds the effective
    property of the inner one, written `Homogenized(inner, scheme)`. The outer
    scheme resolves it when it reads the key, so the two scales are declared
    rather than chained by hand — and the sweep below reaches through the seam
    with a `nested` lens, in one pass.
    """
    inner = Cell(name="foam", matrix_name="GEL", phases=[
        Phase(name="GEL", is_matrix=True, properties=[iso(20.0, 9.0)]),
        Phase(name="GELPORE", amount=0.28, properties=[void()]),
    ], ui={"x": 40, "y": 40})
    outer = Cell(name="paste", matrix_name="FOAM", phases=[
        Phase(name="FOAM", is_matrix=True, properties=[
            Property(key=":C", source="cell", cell=inner.id,
                     scheme="SelfConsistent", scheme_options={}),
        ]),
        Phase(name="CLINKER", amount=0.15, properties=[iso(112.0, 50.0)]),
    ], ui={"x": 320, "y": 110})
    m = Model(title="two_scales", cells=[inner, outer], root_cell=outer.id)
    m.description = (
        "A gel foam homogenized first, then used as the matrix of a paste. The "
        "sweep varies the inner porosity through the seam."
    )
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="φ_gel", start=0.0, stop=0.5, length=21,
        cell=outer.id,
        lens=Lens(kind="nested", member="FOAM", property=":C",
                  inner={"kind": "amount", "phase": "GELPORE", "property": ":C",
                         "field_name": "semi_axes", "index": 1, "member": "",
                         "inner": None}),
        schemes=[{"name": "MoriTanaka", "options": {}}],
        projection="iso", outputs=[{"kind": "k"}, {"kind": "mu"}],
    )
    return m


# ---------------------------------------------------------------------------
# 06 — the exact laminate
# ---------------------------------------------------------------------------


def laminate_basics() -> Model:
    """After `scripts/33_laminate_basics.jl`.

    A laminate is a *cell*, not an inclusion: a periodic stack with no matrix
    and no reference medium, and an exact effective behavior rather than an
    estimate. Two isotropic layers give a transversely isotropic result — that
    is Backus (1962) — and the bounds saturate: exactly Reuss across the layers
    (KM 3,3) and exactly Voigt in their plane (KM 6,6), at once.
    """
    lam = Cell(name="lam", kind="laminate", layers=[
        Layer(name="A", amount=0.3, properties=[iso(2.0, 0.8)]),
        Layer(name="B", amount=0.7, properties=[iso(0.5, 0.2)]),
    ])
    m = Model(title="laminate_basics", cells=[lam], root_cell=lam.id)
    m.description = (
        "A 30/70 bilayer normal to e₃. Compare the three numbers: Laminated "
        "equals Reuss on KM[3,3] and Voigt on KM[6,6], exactly."
    )
    m.sweep = Sweep(
        enabled=True, mode="single", cell=lam.id,
        schemes=[{"name": n, "options": {}} for n in ("Laminated", "Voigt", "Reuss")],
        projection="none",
        outputs=[{"kind": "km", "i": 3, "j": 3}, {"kind": "km", "i": 6, "j": 6}],
    )
    return m


# ---------------------------------------------------------------------------
# 07 — an imperfect interface, and the size effect it brings
# ---------------------------------------------------------------------------


def laminate_interfaces() -> Model:
    """After `scripts/34_laminate_interfaces.jl`.

    With a spring interface the answer stops depending on the volume fractions
    alone: the interface enters with weight 1/L, an interface *density*, so the
    absolute period matters. That is why the layers are given by thickness here
    and not by fraction — and why the sweep varies a thickness, which moves the
    period as well as the fraction.
    """
    lam = Cell(name="lam", kind="laminate", layers=[
        Layer(name="A", amount_kind="thickness", amount=0.3,
              properties=[iso(2.0, 0.8)],
              interface={"kind": "SpringInterface",
                         "args": {"kn": 1.0e-2, "kt": 2.0e-2}}),
        Layer(name="B", amount_kind="thickness", amount=0.7,
              properties=[iso(0.5, 0.2)]),
    ])
    m = Model(title="laminate_interfaces", cells=[lam], root_cell=lam.id)
    m.description = (
        "A spring interface on top of layer A. Because it enters with weight "
        "1/L, thickening the stack stiffens it — a size effect a perfect "
        "interface does not have."
    )
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="h_A", start=0.1, stop=2.0, length=21,
        lens=Lens(kind="thickness", phase="A"), cell=lam.id,
        schemes=[{"name": "Laminated", "options": {}}],
        projection="none",
        outputs=[{"kind": "km", "i": 3, "j": 3}, {"kind": "km", "i": 4, "j": 4}],
    )
    return m


# ---------------------------------------------------------------------------
# 08 — a laminate whose layer is itself a homogenized cell
# ---------------------------------------------------------------------------


def laminate_multiscale() -> Model:
    """After `scripts/36_laminate_multiscale.jl`.

    The seam works on a laminate exactly as on an RVE: a layer property may be
    a `Homogenized`. Drag a connector onto a layer's `:C` slot in the graph and
    this is what you get.
    """
    inner = Cell(name="porous", matrix_name="SOLID", phases=[
        Phase(name="SOLID", is_matrix=True, properties=[iso(2.0, 0.8)]),
        Phase(name="PORE", amount=0.25, properties=[void()]),
    ], ui={"x": 40, "y": 40})
    lam = Cell(name="lam", kind="laminate", ui={"x": 320, "y": 100}, layers=[
        Layer(name="POROUS", amount=0.4, properties=[
            Property(key=":C", source="cell", cell=inner.id,
                     scheme="MoriTanaka", scheme_options={}),
        ]),
        Layer(name="DENSE", amount=0.6, properties=[iso(0.5, 0.2)]),
    ])
    m = Model(title="laminate_multiscale", cells=[inner, lam], root_cell=lam.id)
    m.description = (
        "A three-scale chain: a porous RVE becomes one layer of a stack. The "
        "sweep varies the inner porosity through the seam."
    )
    m.sweep = Sweep(
        enabled=True, mode="sweep", variable="φ", start=0.0, stop=0.5, length=21,
        cell=lam.id,
        lens=Lens(kind="nested", member="POROUS", property=":C",
                  inner={"kind": "amount", "phase": "PORE", "property": ":C",
                         "field_name": "semi_axes", "index": 1, "member": "",
                         "inner": None}),
        schemes=[{"name": "Laminated", "options": {}}],
        projection="none", outputs=[{"kind": "km", "i": 3, "j": 3}],
    )
    return m


# ---------------------------------------------------------------------------
# 09 — ageing viscoelasticity
# ---------------------------------------------------------------------------


def ageing_creep() -> Model:
    """After `scripts/53_ageing_creep_solid.jl` and `scripts/62_alv_schemes.jl`.

    A phase becomes viscoelastic through its *property*, not through a separate
    panel: pick a Kelvin chain or a Maxwell law in Properties, and the
    Viscoelastic tab then only chooses the time grid and the component to
    follow. `(1, 1)` is the uniaxial creep response.
    """
    paste = Phase(
        name="PASTE", is_matrix=True,
        properties=[Property(
            key=":C", builder="kelvin_iso", form="kelvin_iso",
            args={"k0": 12.0, "mu0": 8.0, "k1": 20.0, "mu1": 12.0,
                  "tau_k": 2.0, "tau_mu": 2.0},
        )],
    )
    sand = Phase(
        name="SAND", amount=0.35,
        properties=[Property(
            key=":C", builder="heaviside_law", form="visco_elastic",
            args={"k": 40.0, "mu": 30.0},
        )],
    )
    c = Cell(name="mortar", matrix_name="PASTE", phases=[paste, sand])
    m = Model(title="ageing_creep", cells=[c], root_cell=c.id)
    m.description = (
        "Elastic sand grains in a creeping paste. The curve is the uniaxial "
        "creep response, read off the Volterra inverse of the effective "
        "relaxation operator."
    )
    m.sweep = Sweep(
        enabled=False, cell=c.id,
        schemes=[{"name": "MoriTanaka", "options": {}}],
    )
    m.alv = Alv(
        enabled=True, t_start=0.0, t_stop=20.0, length=41, cell=c.id,
        scheme="MoriTanaka", property=":C", component=[1, 1], plot=True,
    )
    return m


# ---------------------------------------------------------------------------
# 10 — sensitivities
# ---------------------------------------------------------------------------


def sensitivities() -> Model:
    """After `scripts/26_sensitivities.jl`.

    ForwardDiff straight through the scheme. The point x₀ is the model itself —
    `get_param` reads it off the cell — so the amounts and moduli entered in
    Scales are where the derivative is taken, and nothing is typed twice.
    """
    solid = Phase(name="SOLID", is_matrix=True, properties=[iso(72.0, 32.0)])
    pore = Phase(name="PORE", amount=0.2, properties=[void()])
    c = Cell(name="rve", matrix_name="SOLID", phases=[solid, pore])
    m = Model(title="sensitivities", cells=[c], root_cell=c.id)
    m.description = (
        "How the effective bulk modulus moves with the porosity and with the "
        "solid's own stiffness, at the fractions entered above."
    )
    m.sweep = Sweep(enabled=False, cell=c.id,
                    schemes=[{"name": "MoriTanaka", "options": {}}])
    m.sens = Sens(
        enabled=True, kind="gradient", cell=c.id, scheme="MoriTanaka",
        property=":C", output={"kind": "k"}, projection="iso",
        lenses=[
            Lens(kind="amount", phase="PORE"),
            Lens(kind="property", phase="SOLID", property=":C", index=1),
        ],
    )
    return m


EXAMPLES = [
    ("01_porous_schemes.jl", porous_schemes),
    ("02_cracked_solid.jl", cracked_solid),
    ("03_coated_inclusion.jl", coated_inclusion),
    ("04_conductivity_fibers.jl", conductivity_fibers),
    ("05_two_scales.jl", two_scales),
    ("06_laminate_basics.jl", laminate_basics),
    ("07_laminate_interfaces.jl", laminate_interfaces),
    ("08_laminate_multiscale.jl", laminate_multiscale),
    ("09_ageing_creep.jl", ageing_creep),
    ("10_sensitivities.jl", sensitivities),
]


def _stabilize(m: Model) -> Model:
    """Give the cells deterministic ids.

    `Cell` ids are random by default, which is right in a live session and
    useless in a file kept under version control: regenerating would rewrite
    every example with a fresh set of hexadecimal names and no change of
    meaning. Numbering them by position makes the examples diffable, and makes
    "regenerate and see nothing move" a check worth running.
    """
    mapping = {c.id: f"cell{i + 1}" for i, c in enumerate(m.cells)}
    for c in m.cells:
        c.id = mapping[c.id]
        for mb in c.members():
            for pr in mb.properties:
                if pr.source == "cell" and pr.cell in mapping:
                    pr.cell = mapping[pr.cell]
    m.root_cell = mapping.get(m.root_cell, m.root_cell)
    for holder in (m.sweep, m.alv, m.sens):
        holder.cell = mapping.get(holder.cell, holder.cell)
    return m


def build(name: str, fn) -> str:
    m = _stabilize(fn())
    problems = m.validate()
    if problems:
        raise SystemExit(f"{name}: the model does not validate: {problems}")
    header = "\n".join(
        "# " + line if line else "#"
        for line in (fn.__doc__ or "").strip().splitlines()
    )
    return _with_header(generate(m), header)


def _with_header(source: str, header: str) -> str:
    """Put the docstring just under the generated banner.

    The banner is the emitter's; the note below it is this file's, and keeping
    them apart means regenerating never eats the explanation.
    """
    lines = source.splitlines()
    end = 0
    for i, ln in enumerate(lines):
        if ln.startswith("# ===") and i:
            end = i + 1
            break
    return "\n".join(lines[:end] + ["#", header, "#"] + lines[end:]) + "\n"


def main() -> None:
    for name, fn in EXAMPLES:
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build(name, fn))
        print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


if __name__ == "__main__":
    main()
