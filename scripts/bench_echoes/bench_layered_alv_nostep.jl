import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using MeanFieldHomogenization, TensND, LinearAlgebra, JSON, Printf

ref = JSON.parsefile(joinpath(@__DIR__, "bench_layered_alv_nostep_python.json"))
T_grid = collect(Float64.(ref.T))
N = Int(ref.N)
k0 = Float64(ref.k0); mu0 = Float64(ref.mu0)
eta0 = Float64(ref.eta0); gamma0 = Float64(ref.gamma0)
k1 = Float64(ref.k1); mu1 = Float64(ref.mu1)
kp = Float64(ref.kp); mup = Float64(ref.mup); fp = Float64(ref.fp); finf = Float64(ref.finf)

C_p_tens = TensISO{3}(3 * kp, 2 * mup)
C_1_tens = TensISO{3}(3 * k1, 2 * mu1)
moduli = (heaviside_law(C_p_tens), heaviside_law(C_1_tens), heaviside_law(C_1_tens), heaviside_law(C_1_tens))
cumulative = cumsum(vcat([fp], fill(finf / N, N)))
radii = ntuple(k -> cumulative[k]^(1 / 3), N + 1)
sphere = LayeredSphere(radii, moduli)

C0_law = maxwell_iso(k0, mu0, eta0, gamma0)
α_jl = bulk_concentration_alv(sphere, C0_law, T_grid)
β_jl = shear_concentration_alv(sphere, C0_law, T_grid)

to_matrix(v) = reduce(hcat, [collect(Float64.(row)) for row in v])'

for k in 1:(N + 1)
    α_py = Matrix(to_matrix(ref.layers_alpha[k]))
    β_py = Matrix(to_matrix(ref.layers_beta[k]))
    diff_α = maximum(abs, α_jl[k] .- α_py)
    diff_β = maximum(abs, β_jl[k] .- β_py)
    rel_α = diff_α / max(maximum(abs, α_py), 1.0e-300)
    rel_β = diff_β / max(maximum(abs, β_py), 1.0e-300)
    @printf "Layer %d: max |Δα| = %.3e (rel %.3e)   max |Δβ| = %.3e (rel %.3e)\n" k diff_α rel_α diff_β rel_β
end
