"""Golden and behavioral tests for echoes2mfh.

Run with:  python3 -m pytest tools/echoes2mfh/tests -q
(or plain `python3 tests/test_translate.py` for a dependency-free run).

The refusal set is part of the expected output: a regression that starts
silently translating something it used to refuse fails here.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from echoes2mfh.extract import extract  # noqa: E402
from echoes2mfh.emit import emit  # noqa: E402


def translate(src: str) -> str:
    return emit(extract(src, "<test>"), "test")


def findings(src: str) -> list[str]:
    return [f.reason for f in extract(src, "<test>").findings]


# ---------------------------------------------------------------------------
# The RVE lift
# ---------------------------------------------------------------------------

PROUS = """
from echoes import *
ks=72.;mus=32.;Cs=stiff_kmu(ks,mus)
kp=1.e-6;mup=1.e-6;Cp=stiff_kmu(kp,mup)
ver=rve(matrix="SOLID")
ver["SOLID"]=ellipsoid(shape=spheroidal(1.),symmetrize=[ISO],prop={"C":Cs})
ver["PORE"]=ellipsoid(shape=spheroidal(1.),symmetrize=[ISO],prop={"C":Cp})
def Chom(f):
    ver["PORE"].fraction=f ; ver["SOLID"].fraction=1-f
    C=homogenize(prop="C",rve=ver,scheme=MT)
    return C.k
"""


def test_rve_becomes_a_builder():
    out = translate(PROUS)
    assert "function build_ver(f)" in out
    assert "rve = RVE()" in out
    # Wrapped by the 92-column emitter, so assert on the pieces.
    assert "rve, :SOLID, Spheroid(1.0), Dict(:C => Cs);" in out
    assert "fraction = :rest, symmetrize = IsoSymmetrize()" in out
    assert "fraction = f" in out


def test_matrix_fraction_is_dropped_not_translated():
    """MFH derives it; assigning it would throw."""
    out = translate(PROUS)
    assert "fraction = 1 - f" not in out
    assert "is the matrix" in out


def test_symmetrize_maps_to_the_kernel_average():
    assert "symmetrize = IsoSymmetrize()" in translate(PROUS)


def test_iso_stiffness_is_used_not_the_raw_constructor():
    """`iso_stiffness(k, μ)` takes physical moduli; TensISO{3} takes (3k, 2μ)."""
    out = translate(PROUS)
    assert "iso_stiffness(ks, mus)" in out
    assert "TensISO{3}(3" not in out


# ---------------------------------------------------------------------------
# Scheme options migrate into the constructor
# ---------------------------------------------------------------------------


def test_solver_options_move_into_the_scheme():
    src = """
from echoes import *
ver=rve(matrix="M")
ver["M"]=ellipsoid(shape=spherical,prop={"C":stiff_kmu(1.,1.)})
C=homogenize(prop="C",rve=ver,scheme=SC,epsrel=1.e-10,maxnb=300,select_best=True)
"""
    out = translate(src)
    assert "SelfConsistent(;" in out
    assert "abstol = 1.0e-10" in out
    assert "maxiters = 300" in out
    assert "select_best = true" in out


def test_scheme_without_solver_options_stays_bare():
    src = """
from echoes import *
ver=rve(matrix="M")
ver["M"]=ellipsoid(shape=spherical,prop={"C":stiff_kmu(1.,1.)})
C=homogenize(prop="C",rve=ver,scheme=MT,epsrel=1.e-6,verbose=True)
"""
    out = translate(src)
    assert "MoriTanaka()" in out
    assert "abstol" not in out


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def test_append_loop_becomes_a_comprehension():
    src = """
from numpy import *
def f(x): return x*2
lphi=linspace(0.,1.,11)
k=[]
for phi in lphi:
    k.append(f(phi))
"""
    out = translate(src)
    assert "k = [f(phi) for phi in lphi]" in out


def test_paired_accumulators_evaluate_the_call_once():
    src = """
from numpy import *
def g(x): return x, x
lphi=linspace(0.,1.,11)
k=[];mu=[]
for phi in lphi:
    a,b=g(phi);k.append(a);mu.append(b)
"""
    out = translate(src)
    assert "_sweep = [g(phi) for phi in lphi]" in out
    assert "k = [t[1] for t in _sweep]" in out
    assert "mu = [t[2] for t in _sweep]" in out


# ---------------------------------------------------------------------------
# Correctness hazards
# ---------------------------------------------------------------------------


def test_pow_keeps_its_grouping():
    """`math.pow(f, 1./3.)` is `f^(1.0 / 3.0)`, never `f^1.0 / 3.0`."""
    src = "import math\nx=math.pow(f,1./3.)\n"
    assert "f^(1.0 / 3.0)" in translate(src)


def test_index_only_loop_var_gets_a_one_based_range():
    src = """
a=[1,2,3]
for i in range(3):
    print(a[i])
"""
    out = translate(src)
    assert "for i in 1:3" in out
    assert "a[i]" in out
    assert "i + 1" not in out


def test_value_used_loop_var_keeps_zero_based_range_and_shifts():
    src = """
a=[1,2,3]
for i in range(3):
    print(a[i]+2*i)
"""
    out = translate(src)
    assert "for i in 0:(3 - 1)" in out
    assert "a[i + 1]" in out


def test_literal_index_is_shifted():
    assert "v[3]" in translate("x=v[2]\n")


def test_python_slice_bounds_convert_exactly():
    """Python [0:3] is 0-based exclusive; Julia [1:3] is 1-based inclusive."""
    assert "B[1:3]" in translate("x=B[0:3]\n")


def test_try_bound_names_are_declared_local():
    """A name bound inside `try` does not escape the block in Julia."""
    src = """
from echoes import *
ver=rve(matrix="M")
ver["M"]=ellipsoid(shape=spherical,prop={"C":stiff_kmu(1.,1.)})
def f():
    try:
        C=homogenize(prop="C",rve=ver,scheme=MT)
    except:
        return 0.
    return C.k
"""
    out = translate(src)
    assert "local C" in out
    assert out.index("local C") < out.index("try")


def test_non_string_plot_label_is_stringified():
    """matplotlib stringifies any label; Plots.jl raises a `length`
    MethodError instead. `label=sch` over scheme constants is the common case."""
    src = """
from echoes import *
import matplotlib.pyplot as plt
for sch in [MT, SC]:
    plt.plot(x, y, label=sch)
"""
    out = translate(src)
    assert "label = mfh_label(sch)" in out


def test_string_literal_label_is_not_wrapped():
    src = """
import matplotlib.pyplot as plt
plt.plot(x, y, label='HF')
"""
    out = translate(src)
    assert 'label = "HF"' in out
    assert "string(" not in out


def test_scheme_label_is_the_short_type_name():
    """`string(DifferentialScheme(...))` dumps the whole solver configuration
    into the legend and squashes the axes."""
    src = """
from echoes import *
import matplotlib.pyplot as plt
for sch in [MT, DIFF]:
    plt.plot(x, y, label=sch)
"""
    out = translate(src)
    assert (
        "mfh_label(s::MeanFieldHomogenization.HomogenizationScheme) = "
        "string(nameof(typeof(s)))" in out
    )


def test_unlabeled_series_stays_out_of_the_legend():
    """matplotlib legends only labeled series; Plots.jl invents `y1`, `y2`, …"""
    src = """
import matplotlib.pyplot as plt
plt.plot(x, y)
"""
    out = translate(src)
    assert 'label = ""' in out


def test_preamble_finds_the_project_from_anywhere():
    """A translated script is dropped wherever the user wants, so a fixed
    `joinpath(@__DIR__, "..")` is wrong outside `scripts/`."""
    out = translate("x=1\n")
    assert 'occursin("MeanFieldHomogenization", read(pt, String))' in out
    assert 'Pkg.activate(joinpath(@__DIR__, ".."))' not in out


# ---------------------------------------------------------------------------
# Refusals -- these must NOT silently translate
# ---------------------------------------------------------------------------


def test_unknown_phase_attribute_is_refused_not_guessed():
    """`.shape` once fell through to the `fraction` branch. Never again."""
    src = """
from echoes import *
ver=rve(matrix="M")
ver["M"]=ellipsoid(shape=spherical,prop={"C":stiff_kmu(1.,1.)})
ver["P"]=ellipsoid(shape=spherical,fraction=0.1,prop={"C":stiff_kmu(1.,1.)})
def f(w):
    ver["P"].colour=w
"""
    assert any("colour" in r for r in findings(src))


def test_shape_assignment_updates_the_geometry():
    src = """
from echoes import *
ver=rve(matrix="M")
ver["M"]=ellipsoid(shape=spherical,prop={"C":stiff_kmu(1.,1.)})
ver["P"]=ellipsoid(shape=spherical,fraction=0.1,prop={"C":stiff_kmu(1.,1.)})
def f(omega):
    ver["P"].shape=spheroidal(omega)
"""
    out = translate(src)
    assert "Spheroid(omega)" in out
    assert "fraction = Spheroid" not in out


def test_user_inclusion_subclass_is_refused():
    src = """
from echoes import *
class myincl(user_inclusion):
    def build_all(self):
        return {}
"""
    rs = findings(src)
    assert any("user_inclusion" in r for r in rs)


def test_rve_iteration_is_refused():
    src = """
from echoes import *
ver=rve(matrix="M")
ver["M"]=ellipsoid(shape=spherical,prop={"C":stiff_kmu(1.,1.)})
m=sum([x.data().factor for x in ver],0)
"""
    assert any("iteration over an RVE" in r for r in findings(src))


def test_refusal_emits_a_runtime_error():
    src = """
from echoes import *
class myincl(user_inclusion):
    def build_all(self):
        return {}
"""
    out = translate(src)
    assert "UNTRANSLATED" in out
    assert 'error("echoes2mfh:' in out


def test_out_of_scope_import_is_flagged():
    assert any("nlopt" in r for r in findings("import nlopt\n"))


# ---------------------------------------------------------------------------
# `tensor(...)` conventions -- all transcribed from the Echoes C++ sources and
# checked numerically against TensND (agreement ~9e-16).
# ---------------------------------------------------------------------------


def test_tensor_two_params_is_iso_walpole_pair():
    """tensor_iso.h: C = αJ + βK, which is TensISO{3}(α, β) exactly."""
    src = "from echoes import *\nC=tensor([3.*k, 2.*mu])\n"
    assert "TensISO{3}(3.0 * k, 2.0 * mu)" in translate(src)


def test_tensor_five_params_is_the_sym_walpole_basis():
    """tensor_builder.h:470 dispatches 5 params to tensor_ti::build, which
    reads them as sym-Walpole coefficients -- *not* as the (C1111, ...)
    components the class docstring advertises."""
    src = "from echoes import *\nC=tensor([1.,2.,3.,4.,5.])\n"
    out = translate(src)
    assert "echoes_tensor((1.0, 2.0, 3.0, 4.0, 5.0))" in out
    assert "TensTI{4, T, 5}(p, T.(echoes_axis(θ, φ)))" in out


def test_tensor_nine_params_is_cijkl():
    """tensor_ortho.h:94 really is the Cijkl form -- the two arities disagree
    with each other, which is why each is transcribed separately."""
    src = "from echoes import *\nC=tensor([1.,2.,3.,4.,5.,6.,7.,8.,9.])\n"
    out = translate(src)
    assert "echoes_tensor((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))" in out
    assert "M[4, 4] = 2 * C2323" in out


def test_tensor_three_params_is_second_order():
    src = "from echoes import *\nK=tensor([1.,2.,3.])\n"
    assert "echoes_tensor((1.0, 2.0, 3.0))" in translate(src)


def test_tensor_angles_become_trailing_arguments():
    src = "from echoes import *\nC=tensor([1.,2.,3.,4.,5.],angles=[th,ph])\n"
    assert "echoes_tensor((1.0, 2.0, 3.0, 4.0, 5.0), th, ph)" in translate(src)


def test_tensor_ti_takes_only_two_angles():
    """TI has 2 angles (θ, φ); a third would be meaningless about its axis."""
    src = "from echoes import *\nC=tensor([1.,2.,3.,4.,5.],angles=[a,b,c])\n"
    out = translate(src)
    assert "echoes_tensor((1.0, 2.0, 3.0, 4.0, 5.0), a, b)" in out


def test_tensor_unregistered_arity_is_refused():
    """Echoes registers 1, 2, 3, 5, 9, 21 only; 21 has no verified ordering."""
    src = "from echoes import *\nC=tensor([1.,2.,3.,4.])\n"
    assert any("4 parameters" in r for r in findings(src))


def test_tensor_array_dispatches_on_shape_at_runtime():
    src = "from echoes import *\nC=tensor(M)\n"
    out = translate(src)
    assert "echoes_tensor(M)" in out
    assert "size(M) == (3, 3) ? Tens(M) : inv_KM(M)" in out


def test_km_maps_to_tensnd_counterparts():
    src = "from echoes import *\nA=KM(t)\nB=invKM(M)\n"
    out = translate(src)
    assert "KM(t)" in out
    assert "inv_KM(M)" in out


# ---------------------------------------------------------------------------
# Complex twins
# ---------------------------------------------------------------------------


def test_rvec_becomes_a_complex_eltype_rve():
    src = """
from echoes import *
ver=rvec(matrix="M")
ver["M"]=ellipsoidc(shape=spherical,prop={"C":stiff_kmu(1.,1.)})
"""
    out = translate(src)
    assert "RVE(; T = ComplexF64)" in out


def test_complex_suffix_drops_on_constructors():
    src = """
from echoes import *
ver=rvec(matrix="M")
ver["M"]=ellipsoidc(shape=spheroidal(0.5),prop={"C":stiff_kmu(1.,1.)})
"""
    out = translate(src)
    assert "Spheroid(0.5)" in out
    assert "ellipsoidc" not in out


# ---------------------------------------------------------------------------
# Layered inclusions
# ---------------------------------------------------------------------------


def test_sphere_nlayers_maps_to_layered_sphere():
    src = """
from echoes import *
Cp=stiff_kmu(1.,1.);Cs=stiff_kmu(72.,32.)
ver=rve(matrix="SPN")
ver["SPN"]=sphere_nlayers(radii=[0.,1.],fraction=1.,prop={"C":[Cp,Cs]})
"""
    out = translate(src)
    assert "LayeredSphere((0.0, 1.0), (Cp, Cs))" in out


def test_set_radius_becomes_a_builder_parameter():
    src = """
from echoes import *
import math
Cp=stiff_kmu(1.,1.);Cs=stiff_kmu(72.,32.)
ver=rve(matrix="SPN")
ver["SPN"]=sphere_nlayers(radii=[0.,1.],fraction=1.,prop={"C":[Cp,Cs]})
def Chom(f):
    ver["SPN"].set_radius(0,math.pow(f,1./3.))
    return homogenize(prop="C",rve=ver,scheme=SC)
"""
    out = translate(src)
    assert "function build_ver(f)" in out
    assert "LayeredSphere((f^(1.0 / 3.0), 1.0)" in out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}  {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}  {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
