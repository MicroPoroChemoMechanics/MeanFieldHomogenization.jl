# Cross-check `homogenize_alv_order2` (Julia) vs ECHOES `homogenize_visco`
# on the order-2 Maxwell ageing-creep setup:
#   - iso ALV matrix with Dirichlet 2-element chain + ageing prefactor
#   - iso inhomogeneity with similar law, fraction φ = 0.2
#   - spherical (ω=1) and prolate spheroidal (ω=0.1) shapes
#   - schemes Maxwell, Dilute, Mori-Tanaka
#
# We compare the (3n)×(3n) trapezoidal matrices `V` produced by both
# implementations and report the relative error block-by-block.
import Pkg
Pkg.activate(@__DIR__; io = devnull)

using LinearAlgebra
using Printf
using PyCall
using MeanFieldHomogenization

const echoes = pyimport("echoes")
const np = pyimport("numpy")

# Build the same kernels as the Python script.
# matrix R: instantaneous + 2 R//C, ageing prefactor exp(-(t/30)^2).
function build_R(r0, r1, r2, τ1, τ2, fag, finst)
    return (t, tp) -> begin
        je = finst(tp) * r0 +
            fag(tp) * (
            r1 * (1 - exp(-(t - tp) / τ1)) +
                r2 * (1 - exp(-(t - tp) / τ2))
        )
        je * Matrix{Float64}(I, 3, 3)
    end
end

const Rs = build_R(
    1.0, 2.0, 3.0, 2.0, 10.0,
    t -> exp(-(t / 30.0)^2), _ -> 1.0
)
const Ri = build_R(
    0.2, 0.3, 1.2, 1.0, 15.0,
    t -> exp(-(t / 15.0)^2),
    t -> exp(-(t / 30.0)^2)
)

const law_M = ViscoLaw(Rs, :creep)
const law_I = ViscoLaw(Ri, :creep)

# Python ECHOES helpers --------------------------------------------------
py"""
import numpy as np
def py_Rs(t, tp):
    r0, r1, r2 = 1., 2., 3.
    tau1, tau2 = 2., 10.
    je = r0 + np.exp(-(tp/30.)**2)*(r1*(1.-np.exp(-(t-tp)/tau1))+r2*(1.-np.exp(-(t-tp)/tau2)))
    return je*np.eye(3)

def py_Ri(t, tp):
    r0, r1, r2 = 0.2, 0.3, 1.2
    tau1, tau2 = 1., 15.
    je = np.exp(-(tp/30.)**2)*r0 + np.exp(-(tp/15.)**2)*(r1*(1.-np.exp(-(t-tp)/tau1))+r2*(1.-np.exp(-(t-tp)/tau2)))
    return je*np.eye(3)
"""

const py_Rs = py"py_Rs"
const py_Ri = py"py_Ri"

# Build comparable Julia / ECHOES homogenize calls -----------------------
function julia_run(omega, T, sch, frac)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:Y => law_M); fraction = :rest)
    add_phase!(
        rve, :I,
        omega == 1.0 ? Ellipsoid(1.0, 1.0, 1.0) : Spheroid(omega),
        Dict(:Y => law_I); fraction = frac
    )
    return homogenize_alv_order2(rve, sch, :Y; times = T)
end

py"""
from echoes import rve, ellipsoid, ISO, CREEP, MT, MAX, DIL, REUSS, VOIGT, homogenize_visco
import numpy as np

_SCH_MAP = {"MT": MT, "MAX": MAX, "DIL": DIL, "REUSS": REUSS, "VOIGT": VOIGT}

def run_order2(sh_M, sh_I, T, sch_name, frac, py_Rs, py_Ri):
    ver = rve(matrix="MAT")
    ver["MAT"] = ellipsoid(sh_M, symmetrize=[ISO],
                            visco_prop={"Y": (py_Rs, CREEP)},
                            fraction=1.0 - frac)
    ver["INC"] = ellipsoid(sh_I, symmetrize=[ISO],
                            visco_prop={"Y": (py_Ri, CREEP)},
                            fraction=frac)
    T_np = np.asarray(T, dtype=np.float64)
    sch = _SCH_MAP[sch_name]
    V = homogenize_visco(prop="Y", rve=ver, time_series=T_np,
                         unitsize=3, scheme=sch,
                         epsrel=1.e-6, maxnb=100, verbose=False)
    return np.asarray(V)
"""

const py_run_order2 = py"run_order2"

function echoes_run(omega, T, sch_str, frac)
    sh_M = echoes.spherical
    sh_I = omega == 1.0 ? echoes.spherical : echoes.spheroidal(omega)
    V_np = py_run_order2(sh_M, sh_I, collect(T), sch_str, frac, py_Rs, py_Ri)
    return Matrix{Float64}(V_np)
end

function compare(omega, T, sch_julia, sch_str, frac)
    V_jl = julia_run(omega, T, sch_julia, frac)
    V_py = echoes_run(omega, T, sch_str, frac)
    rel = norm(V_jl - V_py) / max(norm(V_py), 1.0e-30)
    @printf "  ω=%.2f  scheme=%-3s  size=%4d  rel err = %.3e\n" omega sch_str size(V_jl, 1) rel
    return V_jl, V_py
end

# ── Sweep ────────────────────────────────────────────────────────────────
println("=== Order-2 ALV: Julia vs ECHOES ===")
for n in (5, 11, 21)
    T = collect(range(0.0, 5.0; length = n))
    @printf "n_times = %d\n" n
    for omega in (1.0, 0.1)
        compare(omega, T, MoriTanaka(), "MT", 0.2)
        compare(omega, T, Dilute(), "DIL", 0.2)
        compare(omega, T, Maxwell(), "MAX", 0.2)
    end
    println()
end
