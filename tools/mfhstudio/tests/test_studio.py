"""Tests for MFH Studio.

    python3 tests/test_studio.py            model, codegen, graph — no Julia
    python3 tests/test_studio.py --julia    adds the sidecar-backed tests

The load-bearing test is `test_preserves_every_demo_script`: the interface may
only be pointed at somebody's existing work if opening and saving cannot damage
it.
"""

from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from mfhstudio.codegen import (  # noqa: E402
    cell_expression,
    extract_embedded,
    generate,
    render_cell,
)
from mfhstudio.model import (  # noqa: E402
    Cell,
    Geometry,
    Layer,
    Lens,
    Model,
    Param,
    Phase,
    Property,
    Sens,
    Sweep,
    default_layers,
    default_model,
)

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WITH_JULIA = "--julia" in sys.argv


# ---------------------------------------------------------------------------
# Conventions the interface exists to remove
# ---------------------------------------------------------------------------


def test_matrix_has_no_amount():
    """MFH derives it as 1 - Σ f and raises if it is set."""
    src = generate(default_model(), embed_model=False)
    add_matrix = next(l for l in src.splitlines() if "add_matrix!" in l)
    assert "fraction" not in add_matrix


def test_physical_moduli_not_raw_tensiso():
    """`iso_stiffness(k, μ)` takes physical moduli; TensISO{3} takes (3k, 2μ)."""
    src = generate(default_model(), embed_model=False)
    assert "iso_stiffness(72.0, 32.0)" in src
    assert "TensISO{3}(216" not in src


def test_solver_options_attach_to_the_scheme():
    m = default_model()
    m.sweep.schemes = [{
        "name": "SelfConsistent",
        "options": {"abstol": 1e-10, "maxiters": 300, "select_best": True},
    }]
    src = generate(m, embed_model=False)
    assert "SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true)" in src
    assert "homogenize(cell, scheme, :C)" in src


def test_several_schemes_share_one_figure():
    m = default_model()
    m.sweep.schemes = [
        {"name": "MoriTanaka", "options": {}},
        {"name": "Voigt", "options": {}},
    ]
    src = generate(m, embed_model=False)
    assert '("MoriTanaka", MoriTanaka())' in src
    assert '("Voigt", Voigt())' in src
    assert "for (name, scheme) in SCHEMES" in src


def test_single_point_uses_the_amounts_as_entered():
    """The answer to "just compute with the fractions I typed"."""
    m = default_model()
    m.sweep.mode = "single"
    src = generate(m, embed_model=False)
    assert "One homogenization with the amounts entered" in src
    assert "set_param" not in src
    assert "homogenize(cell, MoriTanaka(), :C)" in src


def test_kelvin_mandel_output_needs_no_isotropy():
    """`k_mu` has a method for TensISO alone; an oriented inclusion with no
    orientation average does not give one, and the run dies with a
    MethodError deep inside. Components are defined whatever the symmetry."""
    m = default_model()
    m.sweep.projection = "none"
    m.sweep.outputs = [{"kind": "km", "i": 1, "j": 1}, {"kind": "km", "i": 3, "j": 3}]
    src = generate(m, embed_model=False)
    # Both components come off one `KM` call, bound to a local.
    assert "KMC = KM(C)" in src
    assert "KMC[1, 1]" in src and "KMC[3, 3]" in src
    assert "k_mu" not in src


def test_negative_values_can_be_clipped_for_the_figure():
    """`scripts/28` clips them, because Dilute goes below zero well before
    f = 1 and would otherwise set the scale for all ten curves. It is a
    display choice, so it is off unless asked for."""
    m = default_model()
    assert "max(" not in generate(m, embed_model=False)
    m.sweep.clamp_zero = True
    src = generate(m, embed_model=False)
    assert "return (max(km[1], 0.0), max(km[2], 0.0))" in src


def test_a_reduction_is_computed_once():
    """`k` and `μ` come out of one `k_mu` call.

    Emitting `k_mu(C)[1], k_mu(C)[2]` solved the same thing twice at every
    point of every sweep, for every scheme.
    """
    m = default_model()
    src = generate(m, embed_model=False)
    assert "km = k_mu(C)" in src
    assert src.count("k_mu(C)") == 1, "k_mu must be called once per point"

    # One use needs no name: a local introduced for a single reference is
    # noise in a script somebody has to read.
    m.sweep.outputs = [{"kind": "k"}]
    src = generate(m, embed_model=False)
    assert "km = " not in src and "k_mu(C)[1]" in src

    # Two Kelvin-Mandel components share one `KM` call, and an unrelated
    # single `k` stays inline beside it.
    m.sweep.outputs = [{"kind": "km", "i": 1, "j": 1},
                       {"kind": "km", "i": 3, "j": 3}, {"kind": "k"}]
    src = generate(m, embed_model=False)
    assert "KMC = KM(C)" in src
    assert src.count("KM(C)") == 1
    assert "k_mu(C)[1]" in src

    # Top-level bindings carry the scheme they belong to, so two schemes in a
    # single-point run cannot collide.
    m.sweep.mode = "single"
    m.sweep.outputs = [{"kind": "k"}, {"kind": "mu"}]
    m.sweep.schemes = [{"name": "MoriTanaka", "options": {}},
                       {"name": "Voigt", "options": {}}]
    src = generate(m, embed_model=False)
    assert "km_MoriTanaka = k_mu(C_MoriTanaka)" in src
    assert "km_Voigt = k_mu(C_Voigt)" in src


def test_isotropic_only_output_without_a_projection_is_flagged():
    m = default_model()
    m.sweep.projection = "none"
    m.sweep.outputs = [{"kind": "k"}]
    for c in m.cells:
        for ph in c.phases:
            ph.symmetrize = "none"
    assert any("isotropic result" in p for p in m.validate())


def test_viscoelastic_laws_use_the_real_signatures():
    """`maxwell_iso` takes two relaxation times, not one."""
    from mfhstudio.codegen import CodeGen

    g = CodeGen(Model())
    assert g._prop_expr(Property(
        builder="maxwell_iso",
        args={"k": 10.0, "mu": 5.0, "eta_k": 2.0, "eta_mu": 3.0},
    )) == "maxwell_iso(10.0, 5.0, 2.0, 3.0)"
    assert g._prop_expr(Property(
        builder="kelvin_iso",
        args={"k0": 10.0, "mu0": 5.0, "k1": 20.0, "mu1": 10.0,
              "tau_k": 1.0, "tau_mu": 2.0},
    )) == "kelvin_iso(10.0, 5.0, [20.0], [10.0], [1.0], [2.0])"


def test_anisotropic_conductivity_forms():
    from mfhstudio.codegen import CodeGen

    g = CodeGen(Model())
    # The OUTER constructor: `TensTI{2, Float64, 2}(data, n)` demands a 3-tuple
    # axis and rejects the vector an oriented frame yields.
    assert g._prop_expr(Property(builder="TensTI2", args={"kt": 1.0, "ka": 5.0})) == (
        "TensTI{2}(1.0, 5.0, (0.0, 0.0, 1.0))"
    )
    assert g._prop_expr(Property(
        builder="TensDiag2", args={"k1": 1.0, "k2": 2.0, "k3": 5.0}
    )) == "Tens([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 5.0])"


def test_anisotropic_properties_carry_their_own_frame():
    """The frame a tensor's constants are written in is not the shape's.

    A transversely isotropic tensor takes an axis (the third vector of the
    frame); an orthotropic one takes the basis itself, its components then
    being read *in* that basis.
    """
    from mfhstudio.codegen import CodeGen

    g = CodeGen(Model())
    ang = ["pi/4", 0.7, 0.0]

    ti = g._prop_expr(Property(builder="TensTI2", args={"kt": 1.0, "ka": 5.0},
                               euler_angles=ang))
    assert ti == "TensTI{2}(1.0, 5.0, vecbasis(RotatedBasis(pi/4, 0.7, 0.0))[:, 3])"

    # `hoenig_stiffness` declares no five-argument method: the axis is required.
    hoenig = g._prop_expr(Property(builder="hoenig_stiffness", euler_angles=ang))
    assert hoenig.startswith("hoenig_stiffness(")
    assert hoenig.endswith("vecbasis(RotatedBasis(pi/4, 0.7, 0.0))[:, 3])")
    assert g._prop_expr(Property(builder="hoenig_stiffness")).endswith("(0.0, 0.0, 1.0))")

    ortho = g._prop_expr(Property(builder="TensDiag2",
                                  args={"k1": 1.0, "k2": 2.0, "k3": 5.0},
                                  euler_angles=ang))
    assert ortho.endswith(", RotatedBasis(pi/4, 0.7, 0.0))")

    # `RotatedBasis` with fewer than three angles builds a 2-D basis, so the
    # frame is always spelled out in full.
    assert g._prop_expr(Property(builder="TensOrtho", euler_angles=[0.3])) \
        .endswith("RotatedBasis(0.3, 0.0, 0.0))")


def test_hoenig_defaults_are_not_the_isotropic_point():
    """h = 1 with ν₁ = ν₂ and γ = 1 is isotropy wearing a TI type."""
    from mfhstudio.catalog import PROPERTIES

    form = next(f for f in PROPERTIES if f["name"] == "ti_hoenig")
    d = {f["name"]: f["default"] for f in form["fields"]}
    assert not (d["h"] == 1.0 and d["nu1"] == d["nu2"] and d["gamma"] == 1.0)


def test_alv_curve_follows_the_documented_extraction():
    m = default_model()
    m.alv.enabled = True
    src = generate(m, embed_model=False)
    assert "volterra_inverse(R; block_size = 6)" in src
    assert "homogenize_alv(" in src
    assert "using Plots" in src, "the ALV run plots too"


def _layered_spheroid_model() -> Model:
    def layer(fr, k):
        return {
            "fraction": fr,
            "property": {
                "key": ":K", "source": "builder", "builder": "TensISO{3}",
                "form": "iso_conduction", "args": {"k": k},
            },
        }

    g = Geometry(
        kind="layered_spheroid",
        args={"omega": 0.5, "radius": 1.0, "Nseries": 5},
        layers=[layer(0.3, 1.0), layer(0.7, 5.0)],
    )
    c = Cell(name="r", matrix_name="M", phases=[
        Phase(name="M", is_matrix=True, properties=[
            Property(key=":K", source="builder", builder="TensISO{3}",
                     form="iso_conduction", args={"k": 2.0})]),
        Phase(name="I", amount=0.2, geometry=g, properties=[]),
    ])
    return Model(cells=[c])


def test_layered_spheroid_orientation_reaches_the_generated_call():
    """A spheroid of revolution is orientable and the solver honors it —
    `scheme_integration.jl` returns `TensTI{2}(αt, αa, s.axis)`. The angles
    used to be dropped for this shape alone, leaving the axis at its (0,0,1)
    default: the interface's orientation fields did nothing, in the
    computation as much as in the 3-D view.

    `axis::Tuple` is the declared type, so the vector `vecbasis(...)[:, 3]`
    returns has to be wrapped — the trap the `TensTI{2}` builder documents.
    """
    m = _layered_spheroid_model()
    m.cells[0].phases[1].geometry.euler_angles = [0.7, 1.1]
    src = generate(m, embed_model=False)
    assert "axis = Tuple(vecbasis(RotatedBasis(0.7, 1.1" in src
    # …and no axis keyword at all when the shape is left unrotated.
    m2 = _layered_spheroid_model()
    assert "axis" not in generate(m2, embed_model=False)


def test_layered_spheroid_uses_the_fraction_constructor():
    """The raw constructor demands confocal layers, which typed-in radii are
    not: it threw, and the shape drew nothing."""
    src = generate(_layered_spheroid_model(), embed_model=False)
    assert "layered_spheroid_from_fractions(0.5, 1.0, (0.3, 0.7)" in src
    assert "LayeredSpheroid(" not in src


def test_conductivity_builder_has_the_right_arity():
    """`TensISO{dim}` — one argument is the 2nd-order form. `TensISO{2, 3}`
    named neither the right dimension nor the right order and threw."""
    src = generate(_layered_spheroid_model(), embed_model=False)
    assert "TensISO{3}(2.0)" in src
    assert "TensISO{2, 3}" not in src


def test_orientation_reaches_the_generated_call():
    c = Cell(name="r", matrix_name="M", phases=[
        Phase(name="M", is_matrix=True, properties=[Property()]),
        Phase(name="I", amount=0.2, geometry=Geometry(
            kind="spheroid", args={"omega": 0.3}, euler_angles=[0.7, 1.1])),
    ])
    src = generate(Model(cells=[c]), embed_model=False)
    assert "Spheroid(0.3; euler_angles = (0.7, 1.1))" in src


def test_a_single_angle_gets_the_tuple_comma():
    c = Cell(name="r", matrix_name="M", phases=[
        Phase(name="M", is_matrix=True, properties=[Property()]),
        Phase(name="I", amount=0.2, geometry=Geometry(
            kind="spheroid", args={"omega": 0.3}, euler_angles=[1.2])),
    ])
    assert "euler_angles = (1.2,)" in generate(Model(cells=[c]), embed_model=False)


def test_angles_are_floats_like_every_other_size():
    """A bare `0` next to `1.1` would make the tuple `Tuple{Int, Float64}`."""
    c = Cell(name="r", matrix_name="M", phases=[
        Phase(name="M", is_matrix=True, properties=[Property()]),
        Phase(name="I", amount=0.2, geometry=Geometry(
            kind="ellipsoid", args={"a": 2, "b": 1, "c": 0.5},
            euler_angles=[0, 1.1, 0.3])),
    ])
    src = generate(Model(cells=[c]), embed_model=False)
    assert "euler_angles = (0.0, 1.1, 0.3)" in src


def test_no_angles_means_no_keyword():
    src = generate(default_model(), embed_model=False)
    assert "euler_angles" not in src


def test_geometry_sizes_are_floats():
    """An NTuple mixing Int and Float64 fails to dispatch."""
    g = Geometry(kind="layered_sphere", layers=[
        {"radius": 0.6, "property": {"key": ":C", "source": "builder",
                                     "builder": "iso_stiffness", "args": {"k": 1, "mu": 1}}},
        {"radius": 1, "property": {"key": ":C", "source": "builder",
                                   "builder": "iso_stiffness", "args": {"k": 30, "mu": 12}}},
    ])
    c = Cell(name="r", matrix_name="M", phases=[
        Phase(name="M", is_matrix=True, properties=[Property()]),
        Phase(name="I", amount=1, geometry=g),
    ])
    src = generate(Model(cells=[c]), embed_model=False)
    assert "LayeredSphere((0.6, 1.0)" in src
    assert "(0.6, 1)" not in src


# ---------------------------------------------------------------------------
# Laminates — a cell, not an inclusion
# ---------------------------------------------------------------------------


def _bilayer(**kw) -> Model:
    """The 30/70 stack of `scripts/33_laminate_basics.jl`."""
    lam = Cell(name="lam", kind="laminate", layers=default_layers(), **kw)
    m = Model(title="bilayer", cells=[lam], root_cell=lam.id)
    m.sweep = Sweep(
        enabled=True, mode="single", cell=lam.id,
        schemes=[{"name": "Laminated", "options": {}}],
        projection="none", outputs=[{"kind": "km", "i": 3, "j": 3}],
    )
    return m


def test_laminate_emits_add_layer_in_stacking_order():
    src = generate(_bilayer(), embed_model=False)
    assert "lam = Laminate()" in src
    i = src.index("add_layer!(lam, :A, Dict(:C => iso_stiffness(2.0, 0.8)); fraction = 0.3)")
    j = src.index("add_layer!(lam, :B, Dict(:C => iso_stiffness(0.5, 0.2)); fraction = 0.7)")
    assert i < j, "layers must be emitted in stacking order"


def test_canonical_normal_is_not_written_out():
    """`Laminate()` already means e₃, and the kernel then skips the rotation."""
    src = generate(_bilayer(), embed_model=False)
    assert "normal" not in src


def test_a_tilted_stack_carries_its_normal():
    m = _bilayer()
    # Angles are the default way in; the normal is the other one, and a cell
    # is only read that way when it says so.
    m.cells[0].frame_mode = "normal"
    m.cells[0].normal = [1.0, 0.0, 1.0]
    assert "Laminate(; normal = (1.0, 0.0, 1.0))" in generate(m, embed_model=False)
    m.cells[0].frame_mode = "euler"
    m.cells[0].euler_angles = ["π/4", 0.3]
    src = generate(m, embed_model=False)
    # An expression reaches the script as written, exactly as for a shape:
    # `π/4` says what its seventeen digits only approximate, and it is what
    # comes back out when the file is read again.
    assert "Laminate(; euler_angles = (π/4, 0.3))" in src
    assert "normal" not in src


def test_perfect_interface_is_left_implicit():
    """It is the default of `add_layer!`; writing it out says nothing."""
    assert "PerfectInterface" not in generate(_bilayer(), embed_model=False)


def test_an_imperfect_interface_reaches_the_call():
    m = _bilayer()
    m.cells[0].layers[0].interface = {
        "kind": "SpringInterface", "args": {"kn": 1.0e-3, "kt": 2.0e-3}
    }
    assert "interface = SpringInterface(0.001, 0.002)" in generate(m, embed_model=False)


def test_a_stack_may_not_mix_fractions_and_thicknesses():
    """MFH refuses it: with an imperfect interface the period is meaningful,
    so a half-specified stack is ambiguous rather than merely unusual."""
    m = _bilayer()
    m.cells[0].layers[0].amount_kind = "thickness"
    assert any("mixes absolute thicknesses" in p for p in m.validate())


def test_fractions_must_sum_to_one():
    """`validate_laminate` checks Σf ≈ 1 rather than rescaling silently."""
    m = _bilayer()
    m.cells[0].layers[0].amount = 0.5
    assert any("sum to 1.2" in p for p in m.validate())


def test_a_laminate_needs_no_matrix():
    """The RVE rule must not leak: a laminate has no matrix by construction."""
    assert not any("matrix" in p for p in _bilayer().validate())


def test_a_scheme_that_needs_a_matrix_is_refused_on_a_laminate():
    m = _bilayer()
    m.sweep.schemes = [{"name": "MoriTanaka", "options": {}}]
    problems = m.validate()
    assert any("does not apply to the laminate" in p for p in problems)
    assert any("`Laminated`" in p for p in problems)


def test_a_layer_carries_the_multiscale_seam():
    """A layer property may be a `Homogenized`, and the topological order has
    to see it — otherwise the script calls a builder before defining it."""
    inner = Cell(name="foam", matrix_name="SOLID", phases=[
        Phase(name="SOLID", is_matrix=True,
              properties=[Property(args={"k": 72.0, "mu": 32.0})]),
        Phase(name="PORE", amount=0.3,
              properties=[Property(args={"k": 1e-6, "mu": 1e-6})]),
    ])
    lam = Cell(name="lam", kind="laminate", layers=[
        Layer(name="A", amount=0.4, properties=[
            Property(key=":C", source="cell", cell=inner.id, scheme="MoriTanaka"),
        ]),
        Layer(name="B", amount=0.6,
              properties=[Property(args={"k": 0.5, "mu": 0.2})]),
    ])
    m = Model(title="ms_lam", cells=[lam, inner], root_cell=lam.id)
    src = generate(m, embed_model=False)
    assert "Homogenized(build_foam(), MoriTanaka())" in src
    assert src.index("function build_foam") < src.index("function build_lam")


def test_the_3d_view_is_built_from_the_generated_code():
    """A laminate has no per-member shape, so the picture is of the cell — and
    it comes from the code generator, not from a second description of it."""
    m = _bilayer()
    expr = cell_expression(m, m.cells[0])
    assert expr.startswith("let") and expr.rstrip().endswith("end")
    assert "return" not in expr
    assert expr.count("add_layer!") == 2


def test_laminate_lenses_reach_the_script():
    m = _bilayer()
    m.sweep.mode = "sweep"
    m.sweep.lens = Lens(kind="thickness", phase="A")
    assert "thickness(:A)" in generate(m, embed_model=False)
    m.sweep.lens = Lens(kind="interface_param", index=2, field_name="kn")
    assert "interface_param(2, :kn)" in generate(m, embed_model=False)


def test_a_laminate_reopens_exactly():
    m = _bilayer()
    m.cells[0].normal = [0.0, 1.0, 1.0]
    m.cells[0].layers[1].interface = {
        "kind": "KapitzaInterface", "args": {"h": 0.25}
    }
    back = Model.from_dict(extract_embedded(generate(m)))
    assert back.to_dict() == m.to_dict()


# ---------------------------------------------------------------------------
# Sensitivities
# ---------------------------------------------------------------------------


def _sens_model(**kw) -> Model:
    m = default_model()
    m.sweep.enabled = False
    m.sens = Sens(
        enabled=True, cell=m.cells[0].id, scheme="MoriTanaka",
        output={"kind": "k"}, projection="iso",
        lenses=[Lens(kind="amount", phase="PORE")],
        **kw,
    )
    return m


def test_derivative_passes_the_lens_and_the_indexer():
    """The wrappers take the cell and the lens themselves, so the panel hands
    over exactly the two things it already models."""
    src = generate(_sens_model(), embed_model=False)
    assert "const param = amount(:PORE)" in src
    assert "derivative(cell, scheme, param; output = :C, " \
           "indexer = C -> k_mu(best_fit_iso(C))[1])" in src


def test_gradient_takes_a_vector_of_lenses():
    m = _sens_model(kind="gradient")
    m.sens.lenses = [Lens(kind="amount", phase="PORE"),
                     Lens(kind="property", phase="SOLID", property=":C", index=1)]
    src = generate(m, embed_model=False)
    assert "const params = [" in src
    assert "amount(:PORE)," in src and "property(:SOLID, :C, 1)," in src
    assert "gradient(cell, scheme, params;" in src


def test_a_jacobian_extracts_nothing():
    """It differentiates the whole tensor, which is also the way out when the
    result is not isotropic."""
    src = generate(_sens_model(kind="jacobian"), embed_model=False)
    assert "jacobian(cell, scheme, params; output = :C)" in src
    assert "indexer" not in src


def test_a_derivative_takes_exactly_one_parameter():
    m = _sens_model()
    m.sens.lenses = [Lens(kind="amount", phase="PORE"),
                     Lens(kind="amount", phase="PORE")]
    assert any("one parameter" in p for p in m.validate())


def test_amount_is_refused_on_a_laminate():
    """`AmountParameter` raises there and points at `ThicknessParameter`; the
    interface says so before the run does."""
    m = _bilayer()
    m.sweep.enabled = False
    m.sens = Sens(enabled=True, cell=m.cells[0].id, scheme="Laminated",
                  lenses=[Lens(kind="amount", phase="A")],
                  output={"kind": "km", "i": 3, "j": 3}, projection="none")
    assert any("no phase amount" in p for p in m.validate())


def test_sensitivities_do_not_pull_in_plots():
    """A gradient is a table, not a curve."""
    assert "using Plots" not in generate(_sens_model(), embed_model=False)


# ---------------------------------------------------------------------------
# Multiscale
# ---------------------------------------------------------------------------


def _two_scale() -> Model:
    inner = Cell(name="foam", matrix_name="SOLID", phases=[
        Phase(name="SOLID", is_matrix=True,
              properties=[Property(args={"k": 72.0, "mu": 32.0})]),
        Phase(name="PORE", amount=0.3,
              properties=[Property(args={"k": 1e-6, "mu": 1e-6})]),
    ])
    outer = Cell(name="paste", matrix_name="FOAM", phases=[
        Phase(name="FOAM", is_matrix=True, properties=[
            Property(key=":C", source="cell", cell=inner.id, scheme="SelfConsistent",
                     scheme_options={"abstol": 1e-10}),
        ]),
        Phase(name="CLINKER", amount=0.2,
              properties=[Property(args={"k": 100.0, "mu": 50.0})]),
    ])
    return Model(title="ms", cells=[inner, outer], root_cell=outer.id)


def test_seam_emits_homogenized():
    src = generate(_two_scale(), embed_model=False)
    assert "Homogenized(build_foam(), SelfConsistent(; abstol = 1.0e-10))" in src


def test_inner_scale_is_emitted_first():
    src = generate(_two_scale(), embed_model=False)
    assert src.index("function build_foam") < src.index("function build_paste")


def test_cycle_is_refused_at_construction():
    m = _two_scale()
    foam, paste = m.cells
    foam.phases[0].properties[0] = Property(key=":C", source="cell", cell=paste.id)
    problems = m.validate()
    assert any("cycle" in p for p in problems)
    assert "foam" in problems[0] and "paste" in problems[0]


def test_nested_lens_crosses_scales():
    m = _two_scale()
    m.sweep = Sweep(
        enabled=True, cell=m.cells[1].id,
        lens=Lens(kind="nested", member="FOAM", property=":C",
                  inner=Lens(kind="amount", phase="PORE").to_dict()),
    )
    src = generate(m, embed_model=False)
    assert "nested(:FOAM, :C, amount(:PORE))" in src


def test_alv_and_multiscale_are_refused_together():
    """MFH cannot re-express a homogenized inner result as a ViscoLaw."""
    m = _two_scale()
    m.alv.enabled = True
    assert any("viscoelast" in p.lower() for p in m.validate())


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_embedded_model_reopens_exactly():
    m = default_model()
    src = generate(m)
    back = extract_embedded(src)
    assert back is not None
    assert generate(Model.from_dict(back)) == src


def test_graph_positions_survive_the_round_trip():
    m = _two_scale()
    m.cells[0].ui = {"x": 123, "y": 456}
    back = Model.from_dict(extract_embedded(generate(m)))
    assert back.cells[0].ui == {"x": 123, "y": 456}


def test_untouched_parameter_keeps_its_original_text():
    m = default_model()
    m.params.append(Param(name="T", value="[1, 2]", origin="const T = [\n    1,\n    2,\n]"))
    src = generate(m, embed_model=False)
    assert "const T = [\n    1,\n    2,\n]" in src


def test_edited_parameter_is_regenerated():
    m = default_model()
    m.params.append(Param(name="T", value="99.0", origin="const T = 1.0", edited=True))
    assert "const T = 99.0" in generate(m, embed_model=False)


# ---------------------------------------------------------------------------
# Working without Julia
#
# The interface must come up whether or not the sidecar does. When it did not,
# `S.model` stayed null in the browser and every control threw a TypeError
# nobody sees — the whole thing looked broken with no clue why.
# ---------------------------------------------------------------------------


def test_catalog_is_complete_without_julia():
    from mfhstudio import catalog as catalog_module

    cat = catalog_module.base_catalog()
    assert cat["introspected"] is False
    for key in ("schemes", "geometries", "properties", "symmetrize",
                "projections", "interfaces", "lenses", "visco"):
        assert cat[key], f"{key} is empty without Julia"


def test_introspected_schemes_replace_the_fallback_wholesale():
    """A scheme MeanFieldHomogenization drops must disappear, not linger from a merge."""
    from mfhstudio import catalog as catalog_module

    merged = catalog_module.merge({
        "schemes": [{"name": "OnlyOne", "options": [], "singleton": True}],
        "mfh_version": "9.9.9", "julia_version": "1.x",
    })
    assert [s["name"] for s in merged["schemes"]] == ["OnlyOne"]
    assert merged["introspected"] is True
    assert merged["geometries"], "form definitions must survive the merge"


def test_session_serves_a_catalog_when_the_sidecar_is_dead():
    from mfhstudio.server import Session

    s = Session()
    s.bridge.julia = "/nonexistent-julia"
    cat = s.catalog()
    assert cat["introspected"] is False
    assert cat["schemes"] and cat["geometries"]
    assert s.catalog_error, "the failure must be reported, not swallowed"
    # and the model still generates a script
    assert "add_matrix!" in s.script()


def test_startup_failure_is_diagnosed_not_dumped():
    """A stack trace says what happened; the user needs to know what to do."""
    from mfhstudio.juliabridge import _diagnose

    log = ("ERROR: LoadError: ArgumentError: Package JSON3 [0f8b85d8] is "
           "required but does not seem to be installed:\n"
           " - Run `Pkg.instantiate()` to install all recorded dependencies.")
    msg = _diagnose(log)
    assert "instantiate" in msg
    assert "julia --project=" in msg
    assert log in msg, "the original error must still be there"


# ---------------------------------------------------------------------------
# Julia-backed
# ---------------------------------------------------------------------------


def _bridge():
    from mfhstudio.juliabridge import Bridge

    b = Bridge()
    b.start()
    return b


def test_catalog_covers_every_exported_scheme():
    """The interface must not fall behind MeanFieldHomogenization."""
    b = _bridge()
    try:
        cat = b.catalog()
        names = {s["name"] for s in cat["schemes"]}
        # Julia sources are UTF-8; without saying so this reads as cp1252 on
        # Windows and dies on the first `φ`.
        src = open(
            os.path.join(REPO, "src", "Schemes", "scheme_types.jl"), encoding="utf-8"
        ).read()
        for expected in ("Voigt", "Reuss", "MoriTanaka", "SelfConsistent",
                         "AsymmetricSelfConsistent", "DifferentialScheme",
                         "Dilute", "DiluteDual", "Maxwell",
                         "PonteCastanedaWillis", "Laminated"):
            assert f"struct {expected}" in src or expected in src
            assert expected in names, f"{expected} missing from the catalog"
    finally:
        b.stop()


def test_self_consistent_offers_only_what_it_reads():
    """The kwargs bag accepts anything, so the option list must not come from
    probing the constructor."""
    b = _bridge()
    try:
        cat = b.catalog()
        sc = next(s for s in cat["schemes"] if s["name"] == "SelfConsistent")
        editable = {o["name"] for o in sc["options"] if o["editable"]}
        assert "abstol" in editable and "select_best" in editable
        assert "nsteps" not in editable, "nsteps is meaningless for SelfConsistent"
        diff = next(s for s in cat["schemes"] if s["name"] == "DifferentialScheme")
        deditable = {o["name"] for o in diff["options"] if o["editable"]}
        assert "nsteps" in deditable
        assert "select_best" not in deditable
    finally:
        b.stop()


def test_preserves_every_demo_script():
    """Open and save must not lose a line of anybody's script."""
    from mfhstudio.readback import model_from_script

    b = _bridge()
    try:
        files = sorted(glob.glob(os.path.join(REPO, "scripts", "*.jl")))
        assert files, "no demo scripts found"
        lossy = []
        for f in files:
            src = open(f, encoding="utf-8", errors="replace").read()
            model, _ = model_from_script(src, b)
            out = generate(model, embed_model=False)
            missing = [
                l.strip() for l in src.splitlines()
                if l.strip() and not l.strip().startswith("#") and l.strip() not in out
            ]
            if missing:
                lossy.append((os.path.basename(f), missing[:2]))
        assert not lossy, f"lost lines in {len(lossy)} script(s): {lossy[:3]}"
    finally:
        b.stop()


def test_generated_script_matches_the_echoes_reference():
    """The porous benchmark must reproduce the captured Echoes 1.0 values."""
    b = _bridge()
    try:
        m = default_model()
        m.sweep.start, m.sweep.stop, m.sweep.length = 0.3, 0.3, 2
        m.sweep.plot = False
        r = b.run(generate(m, embed_model=False), timeout=300)
        assert r["ok"], r.get("error")
        got = {}
        for line in r["stdout"].splitlines():
            mm = re.match(r"\s*MoriTanaka (\w+)\s+first = ([-\d.eE+]+)", line)
            if mm:
                got[mm.group(1)] = float(mm.group(2))
        assert got, r["stdout"]
        assert abs(got["k"] - 33.460582) < 1e-5, got
        assert abs(got["mu"] - 17.626742) < 1e-5, got
    finally:
        b.stop()


def test_traces_come_back_as_real_json():
    b = _bridge()
    try:
        sc = b.traces("Spheroid(0.4)")
        assert set(sc) == {"data", "layout"}
        assert sc["data"] and sc["data"][0]["type"] == "surface"
        # guides must not clutter the legend
        assert all(t.get("showlegend") is not True for t in sc["data"])
    finally:
        b.stop()


def test_a_tilted_layered_spheroid_is_drawn_tilted():
    """`LayeredSpheroid` stores a unit revolution axis, not a `.basis`, and
    `inclusion_basis` returns the canonical one whatever that axis is. The
    trace builder parametrized every layer with the axis hard-coded along z,
    so a tilted spheroid drew upright — the picture disagreeing with the
    script, exactly the failure the `_rot(::Ellipsoid)` comment records.
    """
    import math

    b = _bridge()
    try:
        mods = "(TensISO{3}(1.0), TensISO{3}(5.0))"
        base = f"layered_spheroid_from_fractions(0.5, 1.0, (0.3, 0.7), {mods}; Nseries = 5"
        th, ph = 0.9, 0.4
        axis = (
            math.sin(th) * math.cos(ph),
            math.sin(th) * math.sin(ph),
            math.cos(th),
        )
        upright = b.traces(base + ")")
        tilted = b.traces(base + f", axis = {axis})")

        def revolution_dir(scene):
            """Shortest principal direction of the outer layer's point cloud.

            ω = 0.5 is oblate, so the *short* semi-axis is the revolution one.
            """
            tr = scene["data"][-1]
            pts = [
                (x, y, z)
                for xs, ys, zs in zip(tr["x"], tr["y"], tr["z"])
                for x, y, z in zip(xs, ys, zs)
            ]
            n = len(pts)
            cov = [[sum(p[i] * p[j] for p in pts) / n for j in range(3)] for i in range(3)]
            # Power iteration on (tr(C)·I − C) converges to C's *smallest*
            # eigenvector, avoiding a numpy dependency in the test suite.
            tr_c = sum(cov[i][i] for i in range(3))
            v = [0.3, 0.5, 0.81]
            for _ in range(400):
                w = [
                    sum((tr_c * (i == j) - cov[i][j]) * v[j] for j in range(3))
                    for i in range(3)
                ]
                nrm = math.sqrt(sum(c * c for c in w)) or 1.0
                v = [c / nrm for c in w]
            return v

        d_up = revolution_dir(upright)
        d_tl = revolution_dir(tilted)
        # A principal direction has no sign, so compare |cos|.
        assert abs(d_up[2]) > 0.99, d_up
        assert abs(sum(a * b_ for a, b_ in zip(d_tl, axis))) > 0.99, d_tl
        assert abs(d_tl[2]) < 0.9, d_tl
    finally:
        b.stop()


def test_every_example_validates_and_is_up_to_date():
    """The examples are generated, so they cannot be allowed to drift.

    An example that no longer matches what the emitter produces is worse than
    no example: it is the interface's own output, shown to a newcomer as a
    model answer, quietly out of date.
    """
    sys.path.insert(0, os.path.join(HERE, "..", "examples"))
    import build_examples

    stale = []
    for name, fn in build_examples.EXAMPLES:
        m = build_examples._stabilize(fn())
        assert not m.validate(), f"{name}: {m.validate()}"
        path = os.path.join(HERE, "..", "examples", name)
        assert os.path.exists(path), f"{name} has not been generated"
        with open(path, encoding="utf-8") as fh:
            if fh.read() != build_examples.build(name, fn):
                stale.append(name)
    assert not stale, (
        f"out of date: {stale} — run `python3 examples/build_examples.py`"
    )


def test_every_example_reopens_exactly():
    """Each one carries its model, which is what makes it a starting point
    rather than a listing."""
    sys.path.insert(0, os.path.join(HERE, "..", "examples"))
    import build_examples

    for name, fn in build_examples.EXAMPLES:
        with open(os.path.join(HERE, "..", "examples", name), encoding="utf-8") as fh:
            embedded = extract_embedded(fh.read())
        assert embedded is not None, f"{name} carries no model block"
        want = build_examples._stabilize(fn()).to_dict()
        assert Model.from_dict(embedded).to_dict() == want, name


def test_the_bilayer_reproduces_backus():
    """The whole point of the laminate: it is exact, not an estimate.

    `scripts/33_laminate_basics.jl` checks the studio-independent version of
    this against the closed form of Backus (1962). Two of the bound
    bracketings are equalities — exactly Reuss across the layers and exactly
    Voigt in their plane — which is what pins the orientation as well as the
    values.
    """
    b = _bridge()
    try:
        m = _bilayer()
        m.sweep.schemes = [
            {"name": n, "options": {}} for n in ("Laminated", "Voigt", "Reuss")
        ]
        m.sweep.outputs = [
            {"kind": "km", "i": 3, "j": 3}, {"kind": "km", "i": 6, "j": 6}
        ]
        r = b.run(generate(m, embed_model=False), timeout=300)
        assert r["ok"], r.get("error")
        got = {}
        for line in r["stdout"].splitlines():
            mm = re.match(
                r"\s*(\w+)\s+KM33 = ([-\d.eE+]+)\s+KM66 = ([-\d.eE+]+)", line
            )
            if mm:
                got[mm.group(1)] = (float(mm.group(2)), float(mm.group(3)))
        assert set(got) == {"Laminated", "Voigt", "Reuss"}, r["stdout"]
        # C₃₃₃₃ = 1 / ⟨1/(λ+2μ)⟩ and 2·C₁₂₁₂ = 2⟨μ⟩ for the 30/70 stack.
        # The tolerance is the emitter's `%.6f`, not the solver's accuracy:
        # these are read off the printed table, which is what a user sees.
        assert abs(got["Laminated"][0] - 0.98924731) < 5e-7, got
        assert abs(got["Laminated"][1] - 0.76) < 5e-7, got
        assert got["Laminated"][0] == got["Reuss"][0], got
        assert got["Laminated"][1] == got["Voigt"][1], got
    finally:
        b.stop()


def test_a_laminate_written_by_the_studio_is_read_back():
    """Without its embedded block, so the answer comes from `Meta.parse` and
    the recognizer rather than from the model the file carries."""
    from mfhstudio.readback import model_from_script

    b = _bridge()
    try:
        m = _bilayer()
        m.cells[0].frame_mode = "normal"
        m.cells[0].normal = [1.0, 0.0, 1.0]
        m.cells[0].layers[0].interface = {
            "kind": "SpringInterface", "args": {"kn": 1.0e-3, "kt": 2.0e-3}
        }
        src = generate(m)
        src = src[: src.index("#= mfhstudio-model")]
        back, _ = model_from_script(src, b)
        assert [c.kind for c in back.cells] == ["laminate"]
        c = back.cells[0]
        # A stack written as a vector comes back as one: the mode follows the
        # file, not the studio's default.
        assert c.frame_mode == "normal"
        assert c.normal == [1.0, 0.0, 1.0]
        assert [l.name for l in c.layers] == ["A", "B"]
        assert [l.amount for l in c.layers] == [0.3, 0.7]
        assert c.layers[0].interface["kind"] == "SpringInterface"
        assert c.layers[0].interface["args"] == {"kn": 0.001, "kt": 0.002}
        assert c.layers[1].interface["kind"] == "PerfectInterface"
    finally:
        b.stop()


def test_a_laminate_is_drawn_as_a_stack_along_its_normal():
    """The cell is the shape. A tilted normal must tilt the slabs, or the
    picture is agreeing with a model nobody entered."""
    import math

    b = _bridge()
    try:
        m = _bilayer()
        upright = b.traces(cell_expression(m, m.cells[0]))
        m.cells[0].frame_mode = "normal"
        m.cells[0].normal = [1.0, 0.0, 1.0]
        tilted = b.traces(cell_expression(m, m.cells[0]))
        # The same tilt stated the default way. θ is the angle off e₃ and φ the
        # azimuth, so θ = π/4 alone lays the normal in the x–z plane at 45° —
        # the picture must not care which of the two forms said so.
        m.cells[0].frame_mode = "euler"
        m.cells[0].euler_angles = [math.pi / 4]
        tilted_by_angles = b.traces(cell_expression(m, m.cells[0]))

        def slabs(scene):
            return [t for t in scene["data"] if t["type"] == "mesh3d"]

        assert len(slabs(upright)) == 2, upright["data"]
        assert [t["name"].split()[0] for t in slabs(upright)] == ["A", "B"]

        def normal_dir(scene):
            """The stacking direction, from the two slab centroids."""
            a, c = slabs(scene)
            ca = [sum(a[k]) / len(a[k]) for k in "xyz"]
            cc = [sum(c[k]) / len(c[k]) for k in "xyz"]
            d = [cc[i] - ca[i] for i in range(3)]
            n = math.sqrt(sum(x * x for x in d)) or 1.0
            return [x / n for x in d]

        assert abs(normal_dir(upright)[2]) > 0.99, normal_dir(upright)
        want = [1 / math.sqrt(2), 0.0, 1 / math.sqrt(2)]
        for got in (normal_dir(tilted), normal_dir(tilted_by_angles)):
            assert abs(sum(a * b_ for a, b_ in zip(got, want))) > 0.99, got
    finally:
        b.stop()


JULIA_TESTS = {
    "test_catalog_covers_every_exported_scheme",
    "test_self_consistent_offers_only_what_it_reads",
    "test_preserves_every_demo_script",
    "test_generated_script_matches_the_echoes_reference",
    "test_traces_come_back_as_real_json",
    "test_a_tilted_layered_spheroid_is_drawn_tilted",
    "test_the_bilayer_reproduces_backus",
    "test_a_laminate_written_by_the_studio_is_read_back",
    "test_a_laminate_is_drawn_as_a_stack_along_its_normal",
}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = skipped = 0
    for name, fn in fns:
        if name in JULIA_TESTS and not WITH_JULIA:
            skipped += 1
            print(f"  skip  {name}  (pass --julia to run)")
            continue
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}  {type(e).__name__}: {e}")
    total = len(fns) - skipped
    print(f"\n{total - failed}/{total} passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
