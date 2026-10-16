import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using MeanFieldHomogenization, TensND, LinearAlgebra, JSON, Printf

ref = JSON.parsefile(joinpath(@__DIR__, "bench_layered_alv_nopore_python.json"))
T_grid = collect(Float64.(ref.T))
N = Int(ref.N)
k0 = Float64(ref.k0); mu0 = Float64(ref.mu0)
eta0 = Float64(ref.eta0); gamma0 = Float64(ref.gamma0)
ks = collect(Float64.(ref.ks))
mus = collect(Float64.(ref.mus))
finf = Float64(ref.finf)

moduli = ntuple(k -> heaviside_law(TensISO{3}(3 * ks[k], 2 * mus[k])), N)
cumulative = cumsum(fill(1.0 / N, N))
radii = ntuple(k -> cumulative[k]^(1 / 3), N)
sphere = LayeredSphere(radii, moduli)

C0_law = maxwell_iso(k0, mu0, eta0, gamma0)
α_jl = bulk_concentration_alv(sphere, C0_law, T_grid)
β_jl = shear_concentration_alv(sphere, C0_law, T_grid)

to_matrix(v) = reduce(hcat, [collect(Float64.(row)) for row in v])'

for k in 1:N
    α_py = Matrix(to_matrix(ref.layers_alpha[k]))
    β_py = Matrix(to_matrix(ref.layers_beta[k]))
    @printf "Layer %d: max |Δα| = %.3e (rel %.3e)   max |Δβ| = %.3e (rel %.3e)\n" k maximum(abs, α_jl[k] .- α_py) maximum(abs, α_jl[k] .- α_py) / maximum(abs, α_py) maximum(abs, β_jl[k] .- β_py) maximum(abs, β_jl[k] .- β_py) / maximum(abs, β_py)
    @printf "    Diagonals α jl vs py: %s vs %s\n" repr([α_jl[k][i, i] for i in 1:5]) repr([α_py[i, i] for i in 1:5])
    @printf "    Diagonals β jl vs py: %s vs %s\n" repr([β_jl[k][i, i] for i in 1:5]) repr([β_py[i, i] for i in 1:5])
end
