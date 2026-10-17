# Compare per-layer (k, mu) Volterra matrices Julia vs Python ECHOES.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf

const json_path = joinpath(@__DIR__, "bench_step_layer_kernel_python.json")
ref = JSON.parsefile(json_path)

T_grid = collect(Float64.(ref.T))
n = length(T_grid)
lT = collect(Float64.(ref.lT))
N = Int(ref.N)
t0 = Float64(ref.t0)

# Same physical setup as bench_step_kernel2.py.
E0 = 1.0; nu0 = 0.2
k0 = E0 / (3 * (1 - 2 * nu0)); mu0 = E0 / (2 * (1 + nu0))
eta0 = 0.2; gamma0 = 0.133
E1 = 5.0; nu1 = 0.3
k1 = E1 / (3 * (1 - 2 * nu1)); mu1 = E1 / (2 * (1 + nu1))
eta1 = 1.0; gamma1 = 1.67
Ep = E0 * 1.0e-8; nup = 0.2
kp = Ep / (3 * (1 - 2 * nup)); mup = Ep / (2 * (1 + nup))

C_p_tens = TensISO{3}(3 * kp, 2 * mup)
R1_law = maxwell_iso(k1, mu1, eta1, gamma1)

function inclusion_law(t_set::Real)
    return ViscoLaw(
        function (t, tp)
            if t < tp
                return zero(C_p_tens)
            elseif tp ≥ t_set
                return R1_law.eval_fun(t, tp)
            else
                return C_p_tens
            end
        end, :relaxation
    )
end

# Julia layers (innermost = pore, then solidifying out to N+1).
moduli = ntuple(N + 1) do k
    if k == 1
        heaviside_law(C_p_tens)
    else
        inclusion_law(lT[N - k + 2])
    end
end

# Compute Julia trapezoidals.
function compare_matrix(label, A_jl, A_py)
    diff = A_jl .- A_py
    rel = maximum(abs, diff) / max(maximum(abs, A_py), 1.0e-300)
    @printf "%s : max |Δ| = %.3e   relative = %.3e\n" label maximum(abs, diff) rel
    return if rel > 1.0e-10
        println("  jl:")
        for i in 1:size(A_jl, 1)
            @printf "    [%d, :] = %s\n" i join([@sprintf("%+.4e", x) for x in A_jl[i, :]], "  ")
        end
        println("  py:")
        for i in 1:size(A_py, 1)
            @printf "    [%d, :] = %s\n" i join([@sprintf("%+.4e", x) for x in A_py[i, :]], "  ")
        end
    end
end

to_matrix(v) = reduce(hcat, [collect(Float64.(row)) for row in v])'

for k in 1:(N + 1)
    R = trapezoidal_matrix(moduli[k], T_grid)
    α, β = iso_params_from_blocks(R)
    M_κ = α ./ 3
    M_μ = β ./ 2

    M_κ_py = Matrix(to_matrix(ref.layers_k[k]))
    M_μ_py = Matrix(to_matrix(ref.layers_mu[k]))

    println("\n=== Layer $(k - 1) (Python idx) ===")
    compare_matrix("M_κ", M_κ, M_κ_py)
    compare_matrix("M_μ", M_μ, M_μ_py)
end
