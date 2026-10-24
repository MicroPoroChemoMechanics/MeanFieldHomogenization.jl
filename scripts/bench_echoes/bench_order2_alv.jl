# Cross-check `homogenize_alv_order2` (Julia) vs ECHOES `homogenize_visco`
# on the order-2 Maxwell aging-creep setup:
#   - iso ALV matrix with Dirichlet 2-element chain + aging prefactor
#   - iso inhomogeneity with similar law, fraction φ = 0.2
#   - spherical (ω=1) and prolate spheroidal (ω=0.1) shapes
#   - schemes Maxwell, Dilute, Mori-Tanaka
#
# We compare the (3n)×(3n) trapezoidal matrices `V` produced by both
# implementations and report the relative error block-by-block.
import Pkg
Pkg.activate(@__DIR__; io = devnull)

# Which Python holds Echoes: chosen before PythonCall is loaded
include(joinpath(@__DIR__, "echoes_python.jl"))

using LinearAlgebra
using Printf
using PythonCall
using MeanFieldHomogenization

const echoes = pyimport("echoes")
const np = pyimport("numpy")

# Build the same kernels as the Python script.
# matrix R: instantaneous + 2 R//C, aging prefactor exp(-(t/30)^2).
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
pyimport("sys").path.insert(0, @__DIR__)
include(joinpath(@__DIR__, "pyjl.jl"))
const O2 = pyimport("order2_alv_echoes")
const py_Rs = O2.py_Rs
const py_Ri = O2.py_Ri

# Build comparable Julia / ECHOES homogenize calls -----------------------
function julia_run(omega, T, sch, frac)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:Y => law_M); fraction = :rest)
    add_phase!(
        rve, :I,
        omega == 1.0 ? Ellipsoid(1.0, 1.0, 1.0) : Spheroid(omega),
        Dict(:Y => law_I); fraction = frac
    )
    return homogenize_alv_order2(rve, sch, :Y; times = T)
end

const py_run_order2 = O2.run_order2

function echoes_run(omega, T, sch_str, frac)
    sh_M = echoes.spherical
    sh_I = omega == 1.0 ? echoes.spherical : echoes.spheroidal(omega)
    V_np = py_run_order2(sh_M, sh_I, pylist(T), sch_str, frac, py_Rs, py_Ri)
    return pyconvert(Matrix{Float64}, V_np)
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
