# =============================================================================
#  90_pair_interaction_tensor.jl
#
#  The two-inclusion interaction tensor 𝕋^{ab}: what it looks like, how it
#  decays, why its isotropic part vanishes, and how the three back-ends
#  (closed form, multipole, quadrature) compare.
#
#  This is the object shared by both N-body schemes of the package — Brisard
#  et al. (2014) note in their §3.1 that their order-zero influence
#  pseudotensors coincide with the interaction tensors of Molinari & El Mouden
#  (1996) and Berveiller et al. (1987).
#
#  SIGN.  The package follows Brisard, Bertin & Legoll (2023), Eq. (9): the
#  Green operator maps a polarization onto MINUS the induced field, so
#  𝕋^{aa} = +ℙ.  Molinari and Berveiller use the opposite sign; §2 below is
#  where the difference is visible.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

const μ₀, ν₀ = 1.0, 0.3
const k₀ = 2μ₀ * (1 + ν₀) / (3 * (1 - 2ν₀))
const C₀ = TensISO{3}(3k₀, 2μ₀)
const K₀ = TensISO{3}(1.0)

println("=" ^ 74)
println("§1  Structure of 𝕋^{ab} for two spheres on the e₃ axis")
println("=" ^ 74)

a, b, R = 1.0, 1.0, 4.0
𝕋 = interaction_tensor(Ellipsoid(a), Ellipsoid(b), [0.0, 0.0, R], C₀)
A = get_array(𝕋)
@printf "  a = %.1f   b = %.1f   R = %.1f   ν = %.2f\n" a b R ν₀
println("  type: ", typeof(𝕋), "  (transversely isotropic about the line of centers)")
@printf "  T_1111 = %+.6e    T_1122 = %+.6e\n" A[1, 1, 1, 1] A[1, 1, 2, 2]
@printf "  T_1133 = %+.6e    T_1212 = %+.6e\n" A[1, 1, 3, 3] A[1, 2, 1, 2]
@printf "  T_1313 = %+.6e    T_3333 = %+.6e\n" A[1, 3, 1, 3] A[3, 3, 3, 3]

println()
println("  In-plane isotropy of a TI tensor, T_1212 = (T_1111 - T_1122)/2:")
@printf "    %.12e  vs  %.12e\n" A[1, 2, 1, 2] (A[1, 1, 1, 1] - A[1, 1, 2, 2]) / 2

println()
println("  Vanishing isotropic part — the reason a cubic array keeps the")
println("  Mori-Tanaka bulk modulus exactly:")
@printf "    T_iijj = %+.3e     T_ijij = %+.3e\n" sum(A[i, i, j, j] for i in 1:3, j in 1:3) sum(A[i, j, i, j] for i in 1:3, j in 1:3)

println()
println("=" ^ 74)
println("§2  The self term IS the Hill tensor (Brisard convention)")
println("=" ^ 74)

for incl in (Ellipsoid(1.0), Ellipsoid(1.0, 0.5, 0.25))
    S = get_array(self_interaction_tensor(incl, C₀))
    P = get_array(hill_tensor(incl, C₀))
    @printf "  %-28s  max|𝕋^{aa} - ℙ| = %.3e\n" string(incl.semi_axes) maximum(abs.(S .- P))
end

println()
println("=" ^ 74)
println("§3  Far-field decay and the finite-size correction in ρ²")
println("=" ^ 74)
println("  ‖𝕋‖ ∝ V_b / R³ at leading order; the ρ² = (a²+b²)/R² term is the")
println("  first (and, for balls, last) correction.")
println()
println("     R/a      ‖𝕋‖·R³/V_b        relative weight of the ρ² term")
for Rr in (3.0, 5.0, 10.0, 20.0, 40.0)
    V_b = 4π / 3
    n_full = maximum(abs.(get_array(interaction_tensor(Ellipsoid(1.0), Ellipsoid(1.0), [0.0, 0.0, Rr], C₀))))
    # Same pair, shrunk to a vanishing size at fixed separation: the ρ² term
    # switches off and only the point-dipole part survives.
    ε = 1.0e-4
    n_dip = maximum(abs.(get_array(interaction_tensor(Ellipsoid(ε), Ellipsoid(ε), [0.0, 0.0, Rr], C₀)))) / ε^3
    @printf "   %6.1f    %.6f          %+.2e\n" Rr (n_full * Rr^3 / V_b) ((n_full - n_dip) / n_full)
end

println()
println("=" ^ 74)
println("§4  Back-ends: closed form vs multipole vs quadrature")
println("=" ^ 74)
println("  For two BALLS the multipole series terminates (the elastic Green")
println("  function is biharmonic), so `:multipole` is exact, not approximate.")
println()
r = [1.0, 2.0, 3.0]
ia, ib = Ellipsoid(1.0), Ellipsoid(0.8)
Aa = get_array(interaction_tensor(ia, ib, r, C₀))
Am = get_array(interaction_tensor(ia, ib, r, C₀; method = :multipole))
Aq = get_array(interaction_tensor(ia, ib, r, C₀; method = :quadrature, nodes = (10, 10, 20)))
@printf "  balls, elasticity   multipole: %.2e   quadrature: %.2e   (relative)\n" (maximum(abs.(Am .- Aa)) / maximum(abs.(Aa))) (maximum(abs.(Aq .- Aa)) / maximum(abs.(Aa)))

Ka = get_array(interaction_tensor(ia, ib, r, K₀))
Km = get_array(interaction_tensor(ia, ib, r, K₀; method = :multipole))
Kq = get_array(interaction_tensor(ia, ib, r, K₀; method = :quadrature, nodes = (10, 10, 20)))
@printf "  balls, conduction   multipole: %.2e   quadrature: %.2e\n" (maximum(abs.(Km .- Ka)) / maximum(abs.(Ka))) (maximum(abs.(Kq .- Ka)) / maximum(abs.(Ka)))

println()
println("  For general ELLIPSOIDS it does not terminate — order 2 converges as")
println("  the separation grows, order 0 (point dipole) much more slowly:")
println()
println("     R        order 0        order 2")
ea, eb = Ellipsoid(1.0, 0.6, 0.4), Ellipsoid(0.8, 0.8, 0.3)
for Rr in (6.0, 12.0, 24.0)
    rr = [0.0, 0.0, Rr]
    Q = get_array(interaction_tensor(ea, eb, rr, C₀; method = :quadrature, nodes = (10, 10, 20)))
    M0 = get_array(interaction_tensor(ea, eb, rr, C₀; method = :multipole, order = 0))
    M2 = get_array(interaction_tensor(ea, eb, rr, C₀; method = :multipole, order = 2))
    s = maximum(abs.(Q))
    @printf "   %5.1f    %.3e      %.3e\n" Rr (maximum(abs.(M0 .- Q)) / s) (maximum(abs.(M2 .- Q)) / s)
end

println()
println("=" ^ 74)
println("§5  Lattice sums over periodic images")
println("=" ^ 74)
println("  The truncation is a SPHERE of images, not a box: Molinari &")
println("  El Mouden's App. B proves convergence from the kernel integrating to")
println("  zero outside a sphere centered on the receiver.")
println()
println("     R_c/L    images    T̄_1111        T̄_iijj (must stay 0)")
for R_c in (0.5, 1.5, 2.5, 3.5, 4.5)
    n_img = length(periodic_images([0.0, 0.0, 0.0], 1.0, R_c; skip_self = true))
    T̄ = get_array(
        lattice_interaction_tensor(Ellipsoid(0.3), Ellipsoid(0.3), [0.0, 0.0, 0.0], C₀, 1.0, R_c)
    )
    @printf "   %6.1f  %8d    %+.6e   %+.2e\n" R_c n_img T̄[1, 1, 1, 1] sum(T̄[i, i, j, j] for i in 1:3, j in 1:3)
end

println()
println("Done.")
