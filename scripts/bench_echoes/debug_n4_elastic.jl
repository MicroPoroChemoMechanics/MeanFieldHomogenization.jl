# Debug: compare Julia's elastic _bulk_concentration (state-space) to
# Julia's ALV bulk_concentration_alv (amplitude-space) when fed
# heaviside laws. They should give exactly the same answer (just
# different representations).

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using MeanFieldHomogenization, TensND, LinearAlgebra, Printf

const k0 = 0.555555; const mu0 = 0.4167; const eta0 = 0.2; const gamma0 = 0.133
const ks = [0.5, 1.0, 2.0, 3.0]
const mus = [0.3, 0.6, 1.0, 1.5]
const N = 4

T_grid = [0.5, 0.7, 1.0, 1.5, 2.0]

# Build the LayeredSphere.
moduli = ntuple(k -> heaviside_law(TensISO{3}(3 * ks[k], 2 * mus[k])), N)
cumulative = cumsum(fill(1.0 / N, N))
radii = ntuple(k -> cumulative[k]^(1 / 3), N)
sphere = LayeredSphere(radii, moduli)

# Elastic state-space concentration (uses heaviside path).
moduli_elastic = ntuple(k -> TensISO{3}(3 * ks[k], 2 * mus[k]), N)
sphere_elastic = LayeredSphere(radii, moduli_elastic)
α_elastic = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(sphere_elastic, k0, mu0)
println("Julia elastic (state-space) α_k:")
for (k, αv) in enumerate(α_elastic)
    println("  Layer $k: α = ", αv)
end

# ALV amplitude-space concentration with HEAVISIDE matrix (= same as elastic).
C0_law = heaviside_law(TensISO{3}(3 * k0, 2 * mu0))
α_alv = bulk_concentration_alv(sphere, C0_law, T_grid)
println("\nJulia ALV amplitude-space α_k diagonals (Heaviside matrix → should be elastic):")
for k in 1:N
    @printf "  Layer %d: diag = %s\n" k repr([α_alv[k][i, i] for i in 1:length(T_grid)])
end

# Test 2: Maxwell matrix.
C0_maxwell = maxwell_iso(k0, mu0, eta0, gamma0)
α_alv2 = bulk_concentration_alv(sphere, C0_maxwell, T_grid)
println("\nJulia ALV α_k with Maxwell matrix (diagonals should equal elastic at each t):")
for k in 1:N
    @printf "  Layer %d: diag = %s\n" k repr([α_alv2[k][i, i] for i in 1:length(T_grid)])
end

println("\n=> α_alv2 diagonals should equal α_elastic (elastic concentration recomputed at each fixed t).")
