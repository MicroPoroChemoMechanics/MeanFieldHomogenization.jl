# =============================================================================
#  bench_ellipsoid2.jl — debug script 38 against ECHOES Python.
#
#  Reproduces the exact ellipsoid-2 aging-creep setup and compares
#  the homogenized effective relaxation matrix `R̃_eff` block-by-block,
#  plus the trapezoidal inputs (matrix + inclusion creep / relaxation
#  matrices), so we can pinpoint the divergence layer-by-layer.
# =============================================================================
import Pkg
Pkg.activate(@__DIR__; io = devnull)

# Which Python holds Echoes: chosen before PythonCall is loaded
include(joinpath(@__DIR__, "echoes_python.jl"))

using LinearAlgebra
using Printf
using PythonCall
using MeanFieldHomogenization
using TensND

const echoes = pyimport("echoes")
const np = pyimport("numpy")

# ─── Phase definitions (identical to script 38 / Python) ───────────────────

const Eₛ = 1.0;  const νₛ = 0.2
const kₛ = Eₛ / (3 * (1 - 2 * νₛ))
const μₛ = Eₛ / (2 * (1 + νₛ))
const Cₛ_t = TensISO{3}(3 * kₛ, 2 * μₛ)

fk(t) = 0.5 * exp(-t / 20.0) + 0.5
fμ(t) = 0.5 * exp(-t / 20.0) + 0.5

const _, 𝕁₄, 𝕂₄ = TensND.iso_projectors(Val(3), Val(Float64))
const _J_M = TensND.mandel66(𝕁₄)
const _K_M = TensND.mandel66(𝕂₄)

# Julia matrix law (creep mode).
function Js(t, tp)
    α = fk(tp) / (3 * kₛ) + 1.0e-1 / kₛ * log(1 + (t - tp) / 2.0)
    β = fμ(tp) / (2 * μₛ) + 1.0e-1 / μₛ * log(1 + (t - tp) / 1.0)
    return α .* _J_M .+ β .* _K_M
end
const law_M = ViscoLaw(Js, :creep)

const Cᵢ_inv = (1.0 / 1.0e6) * Matrix{Float64}(I, 6, 6)
const law_I = ViscoLaw((t, tp) -> Cᵢ_inv, :creep)

# ─── Python side ──────────────────────────────────────────────────────────

pyimport("sys").path.insert(0, @__DIR__)
include(joinpath(@__DIR__, "pyjl.jl"))
const E2 = pyimport("ellipsoid2_echoes")
py_trapz_creep(T) = pymat(E2.trapz_creep(T))
py_trapz_relax(T) = pymat(E2.trapz_relax(T))
py_homogenize(omega, T, sch_name, frac) = pymat(E2.homogenize_py(omega, T, sch_name, frac))

# ─── Helpers ───────────────────────────────────────────────────────────────

block(M, i, j) = M[(6 * (i - 1) + 1):(6 * i), (6 * (j - 1) + 1):(6 * j)]

function diff_summary(label, M_jl, M_py)
    rel = norm(M_jl - M_py) / max(norm(M_py), 1.0e-30)
    @printf "  %-30s shape=%s  rel err = %.3e\n" label string(size(M_jl)) rel
    if rel > 1.0e-6
        @printf "    block(1,1) Julia : %s\n" string(round.(block(M_jl, 1, 1); digits = 5))
        @printf "    block(1,1) Python: %s\n" string(round.(block(M_py, 1, 1); digits = 5))
    end
    return rel
end

# ─── Step 1: trapezoidal compliance / relaxation of the matrix ─────────────

const T_grid = vcat(0.0, 10 .^ range(-2, log10(50.0); length = 5))   # n=6 for clarity
println("=== Trapezoidal (creep) and relaxation matrices of the matrix ===")
println("  T = ", T_grid)
const T_grid_py = np.array(pylist(T_grid))   # the grid as a numpy array, for Echoes

J̃_M_jl = trapezoidal_matrix(law_M, T_grid)
J̃_M_py = py_trapz_creep(T_grid_py)
diff_summary("J̃_M (creep, trapezoidal)", J̃_M_jl, J̃_M_py)

R̃_M_jl = volterra_inverse(J̃_M_jl; block_size = 6)
R̃_M_py = py_trapz_relax(T_grid_py)
diff_summary("R̃_M (relaxation = inv J̃)", R̃_M_jl, R̃_M_py)

# ─── Step 2: full homogenization across schemes ────────────────────────────
println()

const _SCH_MAP = Dict(
    "MT" => MoriTanaka(),
    "DIFF" => DifferentialScheme(; nsteps = 100),
    "SC" => SelfConsistent(),
    "ASC" => AsymmetricSelfConsistent(),
    "MAX" => Maxwell(),
    "PCW" => PonteCastanedaWillis(),
)

for sch_name in ("MT", "DIFF", "SC", "ASC", "MAX", "PCW")
    println("=== Effective relaxation matrix ($sch_name) ===")
    for omega in (1.0, 0.1)
        for f in (0.0, 0.4)
            R̃_jl_eff = let
                rve = RVE(; distribution_shape = Ellipsoid(1.0))
                add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => law_M); fraction = :rest)
                sh = omega == 1.0 ? Ellipsoid(1.0, 1.0, 1.0) : Spheroid(omega)
                add_phase!(
                    rve, :I, sh, Dict(:C => law_I);
                    fraction = f, symmetrize = :iso
                )
                homogenize_alv(rve, _SCH_MAP[sch_name], :C; times = T_grid)
            end
            R̃_py_eff = try
                py_homogenize(omega, T_grid_py, sch_name, f)
            catch e
                println("    $sch_name not in Python map; skipping py-side")
                nothing
            end
            label = "$(sch_name)  ω=$omega  f=$f"
            if R̃_py_eff === nothing
                @printf "  %-30s shape=%s  (Julia only)\n" label string(size(R̃_jl_eff))
            else
                diff_summary(
                    label, Matrix{Float64}(R̃_jl_eff), R̃_py_eff
                )
            end
        end
    end
    println()
end
