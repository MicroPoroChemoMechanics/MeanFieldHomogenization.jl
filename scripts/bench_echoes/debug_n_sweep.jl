import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using MeanFieldHomogenization, TensND, LinearAlgebra, JSON, Printf

ref = JSON.parsefile(joinpath(@__DIR__, "debug_n_sweep_python.json"))
T_grid = collect(Float64.(ref.T))
k0 = Float64(ref.k0); mu0 = Float64(ref.mu0)
eta0 = Float64(ref.eta0); gamma0 = Float64(ref.gamma0)
C0_law = maxwell_iso(k0, mu0, eta0, gamma0)
to_matrix(v) = reduce(hcat, [collect(Float64.(row)) for row in v])'
finf = 0.4

for item in ref.Ns
    N = Int(item.N)
    ks = collect(Float64.(item.ks))
    mus = collect(Float64.(item.mus))
    moduli = ntuple(k -> heaviside_law(TensISO{3}(3 * ks[k], 2 * mus[k])), N)
    cumulative = cumsum(fill(1.0 / N, N))
    radii = ntuple(k -> cumulative[k]^(1 / 3), N)
    sphere = LayeredSphere(radii, moduli)
    α_jl = bulk_concentration_alv(sphere, C0_law, T_grid)
    @printf "\n=== N = %d ===\n" N
    for k in 1:N
        α_py = Matrix(to_matrix(item.alpha[k]))
        Δ = maximum(abs, α_jl[k] .- α_py)
        rel = Δ / max(maximum(abs, α_py), 1.0e-300)
        @printf "  Layer %d : max |Δα| = %.3e (rel %.3e)\n" k Δ rel
    end
end
