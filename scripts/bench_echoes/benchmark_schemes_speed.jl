# =============================================================================
#  scripts/bench_echoes/benchmark_schemes_speed.jl
#
#  Speed of whole homogenization schemes, MeanFieldHomogenization.jl against
#  Echoes (C++ through its Python API), on the same RVEs with the same
#  tolerances: numerical Hill tensors to a relative 1e-6 (absolute 1e-8), fixed
#  points to a relative 1e-8. Each case is timed with `@belapsed` (the minimum
#  of the samples) on both sides, one thread each, and the two results are
#  compared, so that the ratio is between two computations of the same number.
#
#  Cases: elasticity and conduction, from an isotropic matrix with spheres to an
#  anisotropic one with rotated ellipsoids; dilute, Mori-Tanaka, self-consistent
#  and differential schemes; aging viscoelastic Mori-Tanaka and self-consistent
#  schemes on a grid of 11 instants; the three-scale strength model of
#  `scripts/common/quasibrittle_strength.jl` (value and derivative).
#
#  Writes `figures/benchmark_schemes_speed.json`, which the "Performance vs
#  Echoes" page of the documentation reads. Echoes must be importable as
#  `import echoes` from the Python interpreter PythonCall uses (see README.md).
#
#  Run from the package root, alone on the machine:
#    julia --project=scripts/bench_echoes scripts/bench_echoes/benchmark_schemes_speed.jl
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

# Which Python holds Echoes: chosen before PythonCall is loaded
include(joinpath(@__DIR__, "echoes_python.jl"))

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using BenchmarkTools
using ForwardDiff
using PythonCall
using JSON

const MFH = MeanFieldHomogenization
const CO = MFH.Core
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5.0

# ── Echoes side: the cases and their homogenization, in Python ──────────────
#
# `schemes_speed_echoes.py` (next to this file) defines the cases once and runs
# them in Echoes; run on its own, it times the Echoes side alone.

pyimport("sys").path.insert(0, @__DIR__)
include(joinpath(@__DIR__, "pyjl.jl"))
const ES = pyimport("schemes_speed_echoes")
const ES_STRENGTH = pyimport("strength_echoes")

# ── MFH side, built from the same description ────────────────────────────────

km(X) = CO.mandel66_minor(TensND.components_canon(X))
canon(X, elastic) = elastic ? km(X) : Matrix(TensND.components_canon(X))
tens(v, elastic) = TensND.Tens(elastic ? CO.array_from_mandel66(v) : v)

function mfh_rve(phases, elastic)
    prop = elastic ? :C : :K
    rve = RVE()
    for (k, p) in enumerate(phases)
        shape = p.axes === nothing ? Ellipsoid(1.0) : Ellipsoid(p.axes...; euler_angles = p.euler)
        kw = p.iso ? (; symmetrize = :iso) : (;)
        frac = k == 1 ? :rest : p.fraction
        add_phase!(rve, Symbol(p.name), shape, Dict(prop => _structured(p.value, elastic)); fraction = frac, kw...)
    end
    return rve
end
# An isotropic medium as the structured type a user writes, so that each code
# takes its isotropic branch; any other as canonical components.
function _structured(v, elastic)
    if elastic && v ≈ CO.mandel66_minor(TensND.components_canon(MFH.Core.isotropify(tens(v, true))))
        α, β = TensND.get_data(MFH.Core.isotropify(tens(v, true)))
        return TensISO{3}(α, β)
    elseif !elastic && v ≈ (tr(v) / 3) * I
        return TensISO{3}(tr(v) / 3)
    end
    return tens(v, elastic)
end

const SCHEMES = Dict(
    "dilute" => Dilute(), "mori_tanaka" => MoriTanaka(),
    "self_consistent" => SelfConsistent(reltol = 1.0e-8),
    "differential" => DifferentialScheme(; reltol = 1.0e-8, abstol = 1.0e-10),
)
# A phase of `ES.CASES` as a named tuple; the matrix comes first.
_phase(p) = (;
    name = p["name"], value = _mat(p["value"]), fraction = p["fraction"],
    axes = p["axes"] === nothing ? nothing : Tuple(Float64.(p["axes"])),
    euler = Tuple(Float64.(p["euler"])), iso = p["iso"],
)
_mat(rows::AbstractMatrix) = Float64.(rows)
_mat(rows) = [Float64(rows[i][j]) for i in eachindex(rows), j in eachindex(rows[1])]

function time_case(i, j, label, elastic, pyphases, scheme)
    phases = map(_phase, pyphases)
    rve = mfh_rve(phases, elastic)
    sch = SCHEMES[scheme]
    prop = elastic ? :C : :K
    jl = canon(homogenize(rve, sch, prop), elastic)
    py = pyconvert(Matrix{Float64}, ES.homogenize_case(i, j))
    tJ = @belapsed homogenize($rve, $sch, $prop)
    tE = @belapsed $(ES.homogenize_case)($i, $j)
    gap = norm((jl + transpose(jl)) / 2 - py) / norm(py)
    @printf "%-62s %-16s %10.3g s %10.3g s %9.1f×   gap %.1e\n" label scheme tJ tE tE / tJ gap
    return Dict(
        "case" => label, "scheme" => scheme, "t_mfh" => tJ, "t_echoes" => tE,
        "ratio" => tE / tJ, "gap" => gap
    )
end

println("="^120)
println("Schemes: MeanFieldHomogenization vs Echoes, same RVEs and tolerances (t(E)/t(J) > 1: MFH faster)")
println("="^120)
rows = Any[]
for (i, (label, elastic, phases, schemes)) in enumerate(pyjl(ES.CASES)), (j, s) in enumerate(schemes)
    push!(rows, time_case(i - 1, j - 1, label, elastic, phases, s))   # 0-based on the Python side
end

# ── Aging linear viscoelasticity, on the time grid of `ES.ALV_TIMES` ─────────

_alv_law(law) = law[1] == "maxwell" ? maxwell_iso(law[2], law[3], law[4], law[5]) :
    heaviside_law(TensISO{3}(3 * law[2], 2 * law[3]))   # an elastic phase: a Heaviside kernel

function mfh_rve_alv(pyphases)
    rve = RVE()
    for (k, p) in enumerate(pyphases)
        shape = p["axes"] === nothing ? Ellipsoid(1.0) : Ellipsoid(Float64.(p["axes"])...)
        kw = p["iso"] ? (; symmetrize = :iso) : (;)
        add_phase!(
            rve, Symbol(p["name"]), shape, Dict(:C => _alv_law(p["law"]));
            fraction = k == 1 ? :rest : p["fraction"], kw...
        )
    end
    return rve
end

const ALV_TIMES = pyconvert(Vector{Float64}, ES.ALV_TIMES)
for (i, (label, pyphases, schemes)) in enumerate(pyjl(ES.ALV_CASES)), (j, s) in enumerate(schemes)
    rve = mfh_rve_alv(pyphases)
    sch = SCHEMES[s]
    jl = Matrix{Float64}(homogenize_alv(rve, sch, :C; times = ALV_TIMES))
    py = pyconvert(Matrix{Float64}, ES.homogenize_alv_case(i - 1, j - 1))
    tJ = @belapsed homogenize_alv($rve, $sch, :C; times = $ALV_TIMES)
    tE = @belapsed $(ES.homogenize_alv_case)($(i - 1), $(j - 1))
    gap = norm(jl - py) / norm(py)
    @printf "%-62s %-16s %10.3g s %10.3g s %9.1f×   gap %.1e\n" label s tJ tE tE / tJ gap
    push!(
        rows, Dict(
            "case" => label, "scheme" => s, "t_mfh" => tJ, "t_echoes" => tE,
            "ratio" => tE / tJ, "gap" => gap
        )
    )
end

# ── The three-scale strength model: value and derivative of one point ────────

include(joinpath(@__DIR__, "..", "common", "quasibrittle_strength.jl"))
# The Echoes side of the model: `strength_echoes.py`, with the same tiny
# regularization of water and air as `quasibrittle_strength.jl`.
py_strength(wc, α, sc, ω, nθ) = ES_STRENGTH.py_strength_reference(wc, α, sc, ω, nθ, TINY)
# MFH takes Echoes' route by default (`start = :iso`): the foam solved with one
# isotropically averaged needle family, then the 20 families from it. The
# direct route (`start = :direct`), the families from the hydrate stiffness, is
# timed as well, for reference.
let wc = 0.5, α = 0.43
    fc_py = pyconvert(Float64, py_strength(wc, α, 0.0, ω_aspect, NTHETA)[4])   # 0-based on the Python side
    tE = @belapsed py_strength($wc, $α, 0.0, $ω_aspect, $NTHETA) samples = 3 evals = 1
    for (start, scheme) in ((:iso, "SC + MT + MT, from the isotropic foam"), (:direct, "SC + MT + MT, direct"))
        fc_jl = compute_point(wc, α; start).fc
        tJ = @belapsed compute_point($wc, $α; start = $start) samples = 3 evals = 1
        label = "three-scale strength model (20 needle families), value and derivative"
        @printf "%-62s %-38s %10.3g s %10.3g s %9.2f×   gap %.1e\n" label scheme tJ tE tE / tJ abs(fc_jl - fc_py) / fc_py
        push!(
            rows, Dict(
                "case" => label, "scheme" => scheme, "t_mfh" => tJ, "t_echoes" => tE,
                "ratio" => tE / tJ, "gap" => abs(fc_jl - fc_py) / fc_py
            )
        )
    end
end

out = Dict(
    "generator" => "scripts/bench_echoes/benchmark_schemes_speed.jl",
    "julia" => string(VERSION), "mfh" => string(pkgversion(MeanFieldHomogenization)),
    "tensnd" => string(pkgversion(TensND)), "echoes" => pyconvert(String, pygetattr(pyimport("echoes"), "__version__", "unknown")),
    "cpu" => Sys.cpu_info()[1].model, "threads" => Threads.nthreads(),
    "protocol" => "BenchmarkTools @belapsed minimum, one thread; Hill tensors to rtol 1e-6, atol 1e-8; fixed points to rtol 1e-8",
    "rows" => rows,
)
path = joinpath(@__DIR__, "figures", "benchmark_schemes_speed.json")
mkpath(dirname(path))
write(path, JSON.json(out; pretty = true))
println("\nwrote ", path)
println("=== bench done benchmark_schemes_speed.jl")
