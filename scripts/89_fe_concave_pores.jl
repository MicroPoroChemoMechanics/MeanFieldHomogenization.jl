# # Concave pores, against the literature
#
# Two families, two papers, one script. A **supersphere** is cubic and reached
# by a three-dimensional cell in an octant; a **superspheroid** is transversely
# isotropic and reached by a two-dimensional Fourier cell on the meridian
# half-plane. Neither has a closed-form Eshelby solution.
#
# ## Part 1 — the supersphere, against Chen et al. (2015)
#
# This part reproduces what can be reproduced from
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

# ## Part 2 — the superspheroid, against Sevostianov et al. (2016)
#
# The axisymmetric companion, and the only one of the two papers to **tabulate**
# its numbers:
#
# > Sevostianov, Chen, Giraud & Grgic, *Compliance and resistivity contribution
# > tensors of axisymmetric concave pores*, International Journal of Engineering
# > Science **101** (2016) 14–28, `doi:10.1016/j.ijengsci.2015.12.005`.
#
# Their shape, Eq. (1.2), is
#
# ```math
# \frac{(x_1^2+x_2^2)^p}{a^{2p}} + \frac{|x_3|^{2p}}{a^{2p}\gamma^{2p}} = 1 ,
# ```
#
# which is `(ρ/a)^{2p} + (|z|/(aγ))^{2p} = 1` — exactly `Superspheroid(a, aγ, p)`.
# The study is at `a = γ = 1`, so `Superspheroid(1, 1, p)`, and the answer is
# transversely isotropic: five constants for ℍ, two for ℝ.
#
# ### The material, which the paper does not state
#
# No modulus appears anywhere in it. The values below are **inferred** from its
# own `p = 1` row, where the body is an exact sphere and Eq. (3.9) applies:
# `H₁₁₁₁/(−H₁₁₂₂) = (9+5ν₀)/(1+5ν₀)` gives `ν₀ = 0.330`, and `ν₀ = 1/3` with
# `E₀ = 1` reproduces all five components and both resistivities to 0.06 % —
# which is their own finite-element error. Presented as an inference, not as a
# datum.

const ν₀_axi = 1 / 3
const E₀_axi = 1.0
const C₀_axi = iso_stiffness(
    E₀_axi / (3 * (1 - 2ν₀_axi)), E₀_axi / (2 * (1 + ν₀_axi))
)
const K₀_axi = TensISO{3}(1.0)

const AXI = FEAxiMeshOptions(; nradial = 28, radius_ratio = 6.0)

# A subset of their Table B.1 and B.4, for comparison. Seven of eighteen rows,
# spanning the concave range they study and their `p = 1` control.
# Table B.1 in full, all eighteen rows, transcribed from the paper. Their own
# mesh-convergence check is Table B.3, and it is the honest yardstick for any
# comparison: between their two meshes the components move by up to 2.50 % at
# `p = 0.30` (on `H₁₁₃₃`) and 0.4 % on `H₃₃₃₃`, so agreement closer than that
# is agreement to within their own resolution.
const THEIR_H = Dict(          # p => (H₁₁₁₁, H₁₁₂₂, H₁₁₃₃, H₃₃₃₃, H₁₃₁₃)
    0.20 => (1.887796, -0.477459, -0.964299, 27.731870, 8.024717),
    0.25 => (1.808015, -0.419517, -0.842265, 12.354200, 3.654240),
    0.30 => (1.819960, -0.420280, -0.783590, 7.405500, 2.558693),
    0.33 => (1.840120, -0.421878, -0.739390, 5.895420, 1.901430),
    0.35 => (1.854860, -0.424255, -0.720632, 5.217026, 1.742868),
    0.40 => (1.894290, -0.434329, -0.681435, 4.065400, 1.506599),
    0.45 => (1.918003, -0.443068, -0.642262, 3.352013, 1.388400),
    0.50 => (1.937640, -0.451064, -0.611918, 2.916446, 1.323735),
    0.55 => (1.952870, -0.459580, -0.587291, 2.639410, 1.289338),
    0.60 => (1.963770, -0.466431, -0.568153, 2.456980, 1.269780),
    0.65 => (1.973077, -0.472413, -0.553477, 2.332502, 1.269780),
    0.70 => (1.979451, -0.477634, -0.541160, 2.241809, 1.258690),
    0.75 => (1.984812, -0.482264, -0.531049, 2.174690, 1.252894),
    0.80 => (1.989424, -0.486274, -0.522567, 2.122560, 1.251279),
    0.85 => (1.993319, -0.489852, -0.515269, 2.081938, 1.250837),
    0.90 => (1.996558, -0.492969, -0.508869, 2.049180, 1.250650),
    0.95 => (1.999258, -0.496250, -0.501454, 2.027340, 1.250050),
    1.00 => (2.001203, -0.498053, -0.498057, 2.001212, 1.249900),
)

# Their Table B.3, the relative change between their two meshes, kept alongside
# so a discrepancy can be compared against their own uncertainty rather than
# against zero. Same column order as `THEIR_H`, in percent.
const THEIR_H_MESH_PCT = Dict(
    0.20 => (0.21, 0.46, 0.17, 0.41, 0.42),
    0.25 => (0.07, 0.18, 1.32, 0.43, 0.33),
    0.30 => (0.33, 0.42, 2.50, 0.10, 0.83),
    0.33 => (0.02, 0.06, 1.47, 0.44, 0.31),
    0.35 => (0.01, 0.04, 0.11, 0.40, 0.27),
    0.40 => (0.00, 0.01, 0.13, 0.40, 0.16),
    0.45 => (0.00, 0.00, 0.11, 0.29, 0.07),
    0.50 => (0.01, 0.00, 0.11, 0.21, 0.06),
    0.55 => (0.01, 0.01, 0.07, 0.16, 0.02),
    0.60 => (0.01, 0.01, 0.05, 0.08, 0.02),
    0.65 => (0.02, 0.02, 0.04, 0.05, 0.91),
    0.70 => (0.01, 0.01, 0.02, 0.03, 0.18),
    0.75 => (0.01, 0.01, 0.04, 0.04, 0.23),
    0.80 => (0.01, 0.01, 0.02, 0.02, 0.22),
    0.85 => (0.01, 0.01, 0.02, 0.02, 0.11),
    0.90 => (0.01, 0.01, 0.01, 0.01, 0.06),
    0.95 => (0.27, 0.03, 0.02, 0.02, 0.02),
    1.00 => (0.00, 0.00, 0.00, 0.00, 0.04),
)

# Table B.4 in full, seventeen rows — they report `p = 0.35` here where the
# compliance table also carries `p = 0.33`.
const THEIR_R = Dict(          # p => (R₁₁, R₃₃)
    0.20 => (2.024681, 15.1971865),
    0.25 => (1.715557, 6.631911),
    0.30 => (1.640000, 3.939221),
    0.35 => (1.548567, 2.823389),
    0.40 => (1.528998, 2.289616),
    0.45 => (1.511922, 1.999021),
    0.50 => (1.506300, 1.832213),
    0.55 => (1.505546, 1.728305),
    0.60 => (1.505895, 1.662567),
    0.65 => (1.507544, 1.616467),
    0.70 => (1.509126, 1.587973),
    0.75 => (1.509902, 1.559419),
    0.80 => (1.509923, 1.539122),
    0.85 => (1.508904, 1.523779),
    0.90 => (1.507043, 1.509388),
    0.95 => (1.507253, 1.505122),
    1.00 => (1.501244, 1.496256),
)

"Compliance and resistivity contribution of the axisymmetric cavity."
function axi_contributions(p)
    pore = FEAxiSupershapePore(Superspheroid(1.0, 1.0, p); opts = AXI)
    return (
        H = compliance_contribution(pore, C₀_axi, C₀_axi),
        R = resistivity_contribution(pore, K₀_axi, K₀_axi),
        pore,
    )
end

println("\n", "="^78)
println("The sphere is the control, and it is exact")
println("-"^78)
println("At p = 1 the body is a sphere, whose contribution tensors the package")
println("knows in closed form. Nothing about the paper enters this row.")
let
    c = axi_contributions(1.0)
    Hm = Matrix(KM(c.H))
    Rm = Matrix(get_array(c.R))
    # Eq. (3.9) of the paper, which is the classical result.
    HG = 10 * (1 - ν₀_axi) / (7 - 5ν₀_axi)
    G₀ = E₀_axi / (2 * (1 + ν₀_axi))
    h1111 = 3 * (1 - ν₀_axi) * (9 + 5ν₀_axi) / (4 * (7 - 5ν₀_axi) * (1 + ν₀_axi) * G₀)
    @printf "H1111 = %.6f   closed form %.6f   rel %.2e\n" Hm[1, 1] h1111 abs(Hm[1, 1] - h1111) / h1111
    @printf "k0 R11 = %.6f  closed form %.6f   rel %.2e\n" Rm[1, 1] 1.5 abs(Rm[1, 1] - 1.5) / 1.5
    @printf "transverse isotropy of H: |H1212 - (H1111-H1122)/2| / H1111 = %.2e\n" (
        abs(Hm[6, 6] / 2 - (Hm[1, 1] - Hm[1, 2]) / 2) / abs(Hm[1, 1])
    )
end

println("\n", "="^78)
println("Compliance contribution against their Table B.1")
println("-"^78)
@printf "%-6s %-22s %-22s %-22s %s\n" "p" "H1111  (this/theirs)" "H3333  (this/theirs)" "H1313  (this/theirs)" "worst dev / their mesh"
# One solve per `p`, reused by both tables. Their two tables do not carry the
# same abscissae — B.1 has `p = 0.33`, B.4 has `p = 0.35` — so the union is
# solved once and each table reads the rows it has.
const AXI_P = sort(collect(union(keys(THEIR_H), keys(THEIR_R))))
const AXI_MEASURED = Dict{Float64, Any}()
for p in AXI_P
    AXI_MEASURED[p] = axi_contributions(p)
end
for p in sort(collect(keys(THEIR_H)))
    H = Matrix(KM(AXI_MEASURED[p].H))
    t = THEIR_H[p]
    # Kelvin-Mandel carries a factor 2 on the shear block.
    mine = (H[1, 1], H[1, 2], H[1, 3], H[3, 3], H[5, 5] / 2)
    dev = maximum(abs(mine[i] - t[i]) / abs(t[i]) for i in 1:5)
    # The same five, as their own mesh study reports them: a deviation below
    # this is a deviation inside their resolution.
    theirs = maximum(THEIR_H_MESH_PCT[p]) / 100
    @printf "%-6.2f %8.4f /%8.4f     %8.4f /%8.4f     %8.4f /%8.4f     %5.2f %% (%.2f %%)\n" p H[1, 1] t[1] H[3, 3] t[4] H[5, 5] / 2 t[5] (100 * dev) (100 * theirs)
    flush(stdout)
end

println("\n", "="^78)
println("Resistivity contribution against their Table B.4")
println("-"^78)
@printf "%-6s %-20s %-20s\n" "p" "R11 (this / theirs)" "R33 (this / theirs)"
for p in sort(collect(keys(THEIR_R)))
    R = Matrix(get_array(AXI_MEASURED[p].R))
    t = THEIR_R[p]
    @printf "%-6.2f %8.4f /%8.4f (%+6.2f %%)   %8.4f /%8.4f (%+6.2f %%)\n" p R[1, 1] t[1] (
        100 * (R[1, 1] - t[1]) / t[1]
    ) R[3, 3] t[2] (100 * (R[3, 3] - t[2]) / t[2])
end

# The two tables, as delimited text, so the figure generator plots exactly the
# numbers this run produced rather than a transcription of them.
let path = joinpath(tempdir(), "axi_vs_sevostianov.csv")
    open(path, "w") do io
        println(io, "p,H1111,H1122,H1133,H3333,H1313,R11,R33")
        for p in AXI_P
            H = Matrix(KM(AXI_MEASURED[p].H))
            R = Matrix(get_array(AXI_MEASURED[p].R))
            @printf io "%.2f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n" p H[1, 1] H[1, 2] H[1, 3] H[3, 3] (
                H[5, 5] / 2
            ) R[1, 1] R[3, 3]
        end
    end
    println("\nwrote ", path)
end

println("\n", "="^78)
println("What the correction buys, and the proof of its sign")
println("-"^78)
println("The uncorrected answer drifts as (a/R)^3; the corrected one does not.")
println("A wrong sign would leave twice the bias instead of none.")
@printf "%-8s %-14s %-14s %s\n" "R/a" "corrected" "uncorrected" "ratio"
let
    # The sphere again, where the exact answer is known.
    SJ = (1 + ν₀_axi) / (3 * (1 - ν₀_axi))
    SK = 2 * (4 - 5ν₀_axi) / (15 * (1 - ν₀_axi))
    v = [1.0, 1, 1, 0, 0, 0]
    J = v * v' / 3
    Kd = Matrix(1.0I, 6, 6) - J
    Aex = inv(Matrix(1.0I, 6, 6) - (SJ * J + SK * Kd))
    for rr in (2.5, 4.0, 6.0, 8.0)
        pore = FEAxiSupershapePore(
            Superspheroid(1.0, 1.0, 1.0);
            opts = FEAxiMeshOptions(; nradial = 20, radius_ratio = rr)
        )
        b = fe_axi_pore_breakdown(pore, C₀_axi)
        ec = norm(Matrix(KM(b.A)) - Aex) / norm(Aex)
        eu = norm(Matrix(KM(b.A_uncorrected)) - Aex) / norm(Aex)
        @printf "%-8.1f %-14.2e %-14.2e %.0fx\n" rr ec eu eu / ec
        flush(stdout)
    end
end
