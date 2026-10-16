# =============================================================================
#  bench_layered_alv_step.jl — Julia counterpart of `bench_layered_alv_step.py`.
#  Tests step-activated layers (the pattern in script 37 :layers).
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON3
using Printf

const HERE = @__DIR__
const json_path = joinpath(HERE, "bench_layered_alv_step_python.json")
isfile(json_path) || error("Run `bench_layered_alv_step.py` first")
ref = JSON3.read(read(json_path, String))

T_grid = collect(Float64.(ref.T))
n = length(T_grid)
const k0 = Float64(ref.k0)
const mu0 = Float64(ref.mu0)
const eta0 = Float64(ref.eta0)
const gamma0 = Float64(ref.gamma0)
const k1 = Float64(ref.k1)
const mu1 = Float64(ref.mu1)
const eta1 = Float64(ref.eta1)
const gamma1 = Float64(ref.gamma1)
const kp = Float64(ref.kp)
const mup = Float64(ref.mup)
const fp = Float64(ref.fp)
const finf = Float64(ref.finf)
const f0 = Float64(ref.f0)
const N = Int(ref.N)
const t0 = Float64(ref.t0)
const lT = collect(Float64.(ref.lT))

const C_p_tens = TensISO{3}(3 * kp, 2 * mup)

make_R0() = maxwell_iso(k0, mu0, eta0, gamma0)
make_R1() = maxwell_iso(k1, mu1, eta1, gamma1)

# Layer modulus matching Python visco_prop with `tp >= lt` history-dep:
function inclusion_law(t_set::Real)
    R1 = make_R1()
    return ViscoLaw(
        function (t, tp)
            if t < tp
                return zero(C_p_tens)
            elseif tp ≥ t_set
                return R1.eval_fun(t, tp)
            else
                return C_p_tens
            end
        end, :relaxation
    )
end

# Layers: pore + N solidifying shells.
# Outer-most carries lT[0] (innermost solidifying = lT[1] in Julia, earliest setting).
# Python: prop[0]=Cp, prop[i+1]=lT[N-1-i].  In Julia 1-indexed:
#   k=1 → pore; k=2..N+1 → lT[N-(k-2)] = lT[N - k + 2].
moduli = ntuple(N + 1) do k
    if k == 1
        heaviside_law(C_p_tens)
    else
        inclusion_law(lT[N - k + 2])
    end
end

cumulative = cumsum(vcat([fp], fill(finf / N, N)))
radii = ntuple(k -> cumulative[k]^(1 / 3), N + 1)

println("Setting times lT = ", lT)
println("Cumulative fractions: ", cumulative)
println("Radii: ", radii)

sphere = LayeredSphere(radii, moduli)
C0_law = make_R0()

α_jl = bulk_concentration_alv(sphere, C0_law, T_grid)
β_jl = shear_concentration_alv(sphere, C0_law, T_grid)

# Compare against Python reference per layer.
to_matrix(v) = reduce(hcat, [collect(Float64.(row)) for row in v])'

function compare(label, A_jl, A_py)
    @printf "\n── %s ──\n" label
    diff = A_jl .- A_py
    rel = maximum(abs, diff) / max(maximum(abs, A_py), 1.0e-300)
    @printf "  max |Δ| = %.3e   max relative = %.3e\n" maximum(abs, diff) rel
    println("  jl:")
    for i in 1:size(A_jl, 1)
        @printf "    [%d, :] = %s\n" i join([@sprintf("%+.5e", x) for x in A_jl[i, :]], "  ")
    end
    println("  py:")
    for i in 1:size(A_py, 1)
        @printf "    [%d, :] = %s\n" i join([@sprintf("%+.5e", x) for x in A_py[i, :]], "  ")
    end
    return
end

for k in 1:(N + 1)
    α_py = Matrix(to_matrix(ref.layers_alpha[k]))
    β_py = Matrix(to_matrix(ref.layers_beta[k]))
    compare("Layer $k α", α_jl[k], α_py)
    compare("Layer $k β", β_jl[k], β_py)
end
