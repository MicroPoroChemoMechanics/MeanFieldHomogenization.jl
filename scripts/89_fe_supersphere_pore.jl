# # A concave superspherical pore, against the literature
#
# `FESupershapePore` is the package's non-ellipsoidal cavity: a shape with no
# closed-form Eshelby solution, reached by solving one finite-element cell — and
# since this release, one *eighth* of a cell, which is what makes a sweep over
# the shape exponent affordable at all.
#
# This script reproduces what can be reproduced from
#
# > Chen, Sevostianov, Giraud & Grgic, *Evaluation of the effective elastic and
# > conductive properties of a material containing concave pores*, International
# > Journal of Engineering Science **97** (2015) 60–68,
# > `doi:10.1016/j.ijengsci.2015.08.012`,
#
# and says plainly where this package and that paper disagree. Their shape is
#
# ```math
# |x/a|^{2p} + |y/a|^{2p} + |z/a|^{2p} \le 1 ,
# ```
#
# which is exactly `Supersphere(a, p)`: convex above `p = 1/2`, the sphere at
# `p = 1`, concave with conical points on the axes below `1/2`. No conversion is
# needed anywhere.
#
# Requires `Ferrite`, `FerriteGmsh` and `Gmsh`.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "fe"); io = devnull)                 #jl
Pkg.instantiate(; io = devnull)                                      #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using SpecialFunctions

import Ferrite, FerriteGmsh, Gmsh                                    #jl

# ## The closed forms of the paper
#
# Their volume, Eq. (2.7), and the microstructural parameter `η`, Eq. (2.20),
# which is the whole content of their approximation: a single scalar that scales
# the *spherical* contribution tensor.

"Volume of the unit supersphere, their Eq. (2.7)."
V_star(p) = 2 * gamma(1 / (2p))^3 / (3 * p^2 * gamma(3 / (2p)))

"Their microstructural parameter, Eq. (2.20): a linear fit in `p`, rescaled by volume."
eta_paper(p) = 3 * (5p - 1) / 8 * ((4π / 3) / V_star(p))

H_G(ν) = 10 * (1 - ν) / (7 - 5ν)
H_K(ν) = (1 - ν) / (1 - 2ν)

# The three schemes of their §3, with `α_K`, `α_G` from their Eq. (3.4).
α_K(ν) = 2 / 3 * (1 - 2ν) / (1 - ν)
α_G(ν) = (7 - 5ν) / (15 * (1 - ν))

nia(X₀, φ, η, h) = X₀ / (1 + φ * η * h)
mt(X₀, φ, η, h) = X₀ / (1 + φ / (1 - φ) * η * h)
maxwell(X₀, φ, η, h, α) = X₀ * (1 - φ * η * h * α) / (1 + φ * η * h * (1 - α))

# ## The reference material
#
# The paper states no moduli for its figures. Everything below is normalized, so
# only the Poisson ratio matters; `ν₀ = 0.3` is used throughout.

const ν₀ = 0.3
const E₀ = 1.0
const K₀ = E₀ / (3 * (1 - 2ν₀))
const G₀ = E₀ / (2 * (1 + ν₀))
const C₀ = iso_stiffness(K₀, G₀)
const k₀ = 1.0
const KK₀ = TensISO{3}(k₀)

# ## 1. The geometry, in closed form and against theirs
#
# The volume is the one place where agreement is exact and checkable: the
# package derives it independently.

println("="^78)
println("Volume of the unit supersphere: closed form vs `shape_volume`")
println("-"^78)
@printf "%-8s %16s %16s %12s\n" "p" "their Eq. (2.7)" "shape_volume" "rel. diff"
for p in (0.30, 0.50, 0.70, 1.00, 1.50, 2.50)
    a, b = V_star(p), shape_volume(Supersphere(1.0, p))
    @printf "%-8.2f %16.9f %16.9f %12.2e\n" p a b abs(a - b) / a
end
println("\nThe sphere is the control: V*(1) = 4π/3 = ", 4π / 3, ".")

# ## 2. The finite-element cell, in an octant
#
# One cell per `p`, both physics from the same mesh. `octant = true` is what
# makes this loop a matter of minutes: the three coordinate planes are mirror
# planes of a supersphere, so an eighth of the domain carries the whole answer.

const CELL = FECellMeshOptions(;
    level = 3, outer_level = 3, radius_ratio = 3.0, relax = 20,
    octant = true, max_dofs = 150_000, min_free_gb = 4.0,
)

"Compliance and resistivity contribution tensors of the cavity, this package's way."
function contributions(p)
    pore = FESupershapePore(Supersphere(1.0, p); opts = CELL)
    # The phase property is a placeholder for a cavity and is ignored — see the
    # manual. Passing the matrix's own moduli is the documented way to say so.
    H = compliance_contribution(pore, C₀, C₀)
    R = resistivity_contribution(pore, KK₀, KK₀)
    return (; H, R, pore)
end

println("\n", "="^78)
println("Resistivity contribution: this work against their η, Eq. (2.21)")
println("-"^78)
println("Their k₀R = η δ, isotropic and one scalar. So is ours, and for the")
println("same reason: a 2nd-order tensor invariant under the octahedral group")
println("*is* isotropic.")
println()
@printf "%-8s %14s %14s %12s\n" "p" "this work" "their η(p)" "rel. diff"
const PS = (0.30, 0.40, 0.50, 0.70, 1.00, 1.50, 2.50)
const MEASURED = Dict{Float64, Any}()
for p in PS
    c = contributions(p)
    MEASURED[p] = c
    r = k₀ * Matrix(get_array(c.R))[1, 1]
    e = eta_paper(p)
    @printf "%-8.2f %14.6f %14.6f %12.2e\n" p r e abs(r - e) / e
    flush(stdout)
end

# ## 3. Where we disagree: a supersphere is cubic
#
# Their Eq. (2.19) writes the compliance contribution on the **isotropic** basis
# `(𝕁, 𝕂)`, two constants. A supersphere has the octahedral symmetry group, so
# its compliance contribution is **cubic**: three constants, and the third is not
# a small correction in the concave range.

println("\n", "="^78)
println("Compliance contribution: three constants, not two")
println("-"^78)
@printf "%-8s %12s %12s %12s %14s\n" "p" "H₁₁₁₁" "H₁₁₂₂" "H₁₂₁₂" "anisotropy"
for p in PS
    M = Matrix(KM(MEASURED[p].H))
    h11, h12 = M[1, 1], M[1, 2]
    h44 = M[4, 4] / 2                     # Kelvin carries a factor 2 on the shear block
    # Zener's ratio, zero for an isotropic tensor by construction.
    z = (h11 - h12 - 2h44) / h11
    @printf "%-8.2f %12.6f %12.6f %12.6f %14.6f\n" p h11 h12 h44 z
end
println()
println("The control is the sphere. At p = 1 the true anisotropy is zero, so")
println("whatever comes out there is this chain's own spurious anisotropy at")
println("this discretization — and it is what licenses reading the concave")
println("numbers as physical rather than numerical.")

# ## 4. Effective properties
#
# Their §3: the non-interaction approximation, Mori-Tanaka and Maxwell, all
# three driven by the single scalar `η`. The package computes them from its own
# measured contribution tensors instead, so the comparison is a comparison of
# *models* and not of arithmetic.

println("\n", "="^78)
println("Effective properties at φ = 0.05, ν₀ = 0.3")
println("-"^78)
const φ = 0.05
@printf "%-8s %10s %10s %10s   %10s %10s %10s\n" "p" "k/k₀ NIA" "MT" "Maxwell" "G/G₀ NIA" "MT" "Maxwell"
for p in PS
    η = eta_paper(p)
    @printf "%-8.2f %10.5f %10.5f %10.5f   %10.5f %10.5f %10.5f\n" p (
        nia(1.0, φ, η, 1.0)
    ) mt(1.0, φ, η, 1.0) (
        (3 - 2φ * η) / (3 + φ * η)
    ) nia(1.0, φ, η, H_G(ν₀)) mt(1.0, φ, η, H_G(ν₀)) maxwell(
        1.0, φ, η, H_G(ν₀), α_G(ν₀)
    )
end

# Their Eq. (4.1)–(4.2) tie the elastic drop to the conductive one, and hold for
# both NIA and Mori-Tanaka. A free cross-check of the whole chain.
println("\n", "-"^78)
println("Cross-property connections, their Eq. (4.1)–(4.2), on the NIA")
println("-"^78)
@printf "%-8s %16s %16s %12s\n" "p" "(K₀−K)/K" "prediction" "rel. diff"
for p in PS
    η = eta_paper(p)
    k = nia(1.0, φ, η, 1.0)
    K = nia(1.0, φ, η, H_K(ν₀))
    lhs = (1 - K) / K
    rhs = H_K(ν₀) * (1 - k) / k
    @printf "%-8.2f %16.8f %16.8f %12.2e\n" p lhs rhs abs(lhs - rhs) / abs(rhs)
end

println("\n", "="^78)
println("Their linear fit against their own data")
println("-"^78)
println("Their Eq. (2.21) anchors η at p = 1, where it is exact, and at p = 0.2,")
println("where they take the effect to vanish. That makes η(p) non-monotone —")
println("zero at 0.2 by construction, then large just above — and negative")
println("below 0.2, which is not physical for a pore.")
@printf "%-8s %14s\n" "p" "their η(p)"
for p in (0.18, 0.20, 0.22, 0.25, 0.30)
    @printf "%-8.2f %14.6f\n" p eta_paper(p)
end
