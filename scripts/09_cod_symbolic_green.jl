# # [From the Green operator to the COD tensor 𝐁](@id tut-cod-symbolic-green)
#
# End-to-end **symbolic** derivation of the crack-opening-displacement tensor
# ``\boldsymbol{B}``, starting from the Fourier Green operator and stopping only
# where the elliptic integrals appear. Everything in between is done by SymPy.
#
# The chain is the one of [barthelemySifAniso](@cite): with
# ``\boldsymbol{N}(\underline{\xi}) = \underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}``
# the acoustic (Christoffel) tensor,
#
# ```math
# \hat{\mathbb{\Gamma}}(\underline{\xi})
#   = \underline{\xi}\stackrel{s}{\otimes}\boldsymbol{N}^{-1}(\underline{\xi})
#     \stackrel{s}{\otimes}\underline{\xi},
# \qquad
# \hat{\mathbb{Q}}(\underline{\xi})
#   = \mathbb{C} - \mathbb{C}:\hat{\mathbb{\Gamma}}(\underline{\xi}):\mathbb{C},
# ```
#
# ```math
# \hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\xi}^{\star})
#   = \frac{1}{2\pi}\int_{-\infty}^{+\infty}
#     \underline{n}\cdot\hat{\mathbb{Q}}(\underline{\xi}^{\star}+\xi_3\,\underline{n})
#     \cdot\underline{n}\;\mathrm{d}\xi_3,
# \qquad
# b\boldsymbol{\Lambda} = \frac14\int_0^{2\pi}
#     \hat{\boldsymbol{Q}}^{\star}_{nn}
#     \bigl(\eta\cos\varphi\,\underline{\ell}+\sin\varphi\,\underline{m}\bigr)\,
#     \mathrm{d}\varphi,
# \qquad
# \boldsymbol{B} = \chi\,(b\boldsymbol{\Lambda})^{-1}.
# ```
#
# The article defines ``\hat{\boldsymbol{Q}}^{\star}_{nn}``, proves that its
# integral converges, and then **defers the evaluation** to the literature. This
# script performs it. The hard step is the ``\xi_3`` integral: the integrand is
# a rational function of ``\xi_3`` whose denominator is
# ``\det\boldsymbol{N}(\underline{\xi}^{\star}+\xi_3\underline{n})``, a *sextic*
# whose roots are the Stroh eigenvalues. Three structural reductions (§ 4) bring
# it within reach of `integrate` for an isotropic matrix, and the factorization
# of that sextic (§ 7) does the same for a transversely isotropic matrix whose
# crack lies in the plane of isotropy — where the radical ``\sigma_\gamma`` of
# the published closed form comes *out of* the calculation instead of being
# quoted.
#
# The oracles are the closed forms already shipped by the package
# ([`cod_tensor`](@ref)), which share no code with anything below: a clean run
# of this script is a proof that the two routes agree.
#
# Prerequisite: `Pkg.add("SymPy")`.
#
# **§0** Setup · **§1** Green operator, order 4 · **§2** Reduced identity ·
# **§3** Behavior at infinity · **§4** The ``\xi_3`` integral (isotropic) ·
# **§5** The crack-plane integral → elliptic integrals · **§6** 𝐁 isotropic,
# penny and ribbon · **§7** Transversely isotropic, crack in the isotropy plane ·
# **§8** Where the symbolic route stops.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using SymPy
using LinearAlgebra
using QuadGK
using Printf

const OO = sympy.oo
const IU = sympy.I

println("=== From the Green operator to the COD tensor 𝐁 — symbolic ===")   #jl
println()                                                                   #jl

# ## §0 Setup — an isotropic matrix, a crack plane, a wave vector
#
# The crack normal is ``\underline{n} = \underline{e}_3``, so an in-plane vector
# is ``\underline{\xi}^{\star} = \rho\,\underline{u}`` with ``\underline{u}`` a
# unit vector of the ``(\underline{e}_1,\underline{e}_2)`` plane. Both the
# isotropic and the aligned-TI matrix are invariant under rotations about
# ``\underline{n}``, and ``\underline{n}`` is also the direction integrated
# over, so ``\hat{\boldsymbol{Q}}^{\star}_{nn}`` is *equivariant* under those
# rotations: computing it for the single direction
# ``\underline{u}=\underline{e}_1`` determines it for every azimuth. That is the
# first of the three reductions of § 4, and it is why one integral suffices.

println("="^78)                                                #jl
println("  § 0  SETUP — isotropic matrix, crack normal e₃")     #jl
println("="^78)                                                #jl

@syms E::positive ν::positive ρ::positive ξ₃::real

## (3k, 2μ) parameterization of `TensISO{3}`, written in (E, ν).
μ = E / (2 * (1 + ν))
λ = E * ν / ((1 + ν) * (1 - 2 * ν))
ℂ = TensISO{3}(E / (1 - 2 * ν), E / (1 + ν))

ℬ = CanonicalBasis{3, Sym}()
𝐞₁, 𝐞₂, 𝐞₃ = tens_basis(ℬ, 1), tens_basis(ℬ, 2), tens_basis(ℬ, 3)
𝐧 = 𝐞₃
𝛏 = ρ * 𝐞₁ + ξ₃ * 𝐧

println("  ℂ = TensISO{3}(E/(1-2ν), E/(1+ν))   (= 3k 𝕁 + 2μ 𝕂)")   #jl
println("  n̲ = e₃,   ξ̲ = ρ e₁ + ξ₃ e₃")                            #jl
println()                                                          #jl

# ## §1 The Fourier Green operator, order 4
#
# Built exactly as in `TensND`'s own symbolic tutorials: the acoustic tensor,
# its inverse, ``\hat{\mathbb{\Gamma}}``, then ``\hat{\mathbb{Q}}``. Note that
# ``\hat{\mathbb{Q}}`` already carries the subtraction of the large-``\xi_3``
# asymptote — which is why the integral of § 4 converges with no *ad hoc*
# regularization.

println("="^78)                                            #jl
println("  § 1  FOURIER GREEN OPERATOR (order 4)")         #jl
println("="^78)                                            #jl

𝐍 = 𝛏 ⋅ ℂ ⋅ 𝛏                     # acoustic (Christoffel) tensor
ℾ = 𝛏 ⊗ˢ inv(𝐍) ⊗ˢ 𝛏
ℚ = ℂ - ℂ ⊡ ℾ ⊡ ℂ
𝐐ₙₙ = tsimplify(𝐧 ⋅ ℚ ⋅ 𝐧)

println("  Q̂ₙₙ[1,1] = ", 𝐐ₙₙ[1, 1])   #jl
println("  Q̂ₙₙ[2,2] = ", 𝐐ₙₙ[2, 2])   #jl
println("  Q̂ₙₙ[3,3] = ", 𝐐ₙₙ[3, 3])   #jl
println("  Q̂ₙₙ[1,3] = ", 𝐐ₙₙ[1, 3])   #jl
println()                              #jl

# ## §2 The reduced identity — no order-4 product needed
#
# Contracting twice with ``\underline{n}`` collapses the order-4 algebra to a
# 3×3 one:
#
# ```math
# \underline{n}\cdot\hat{\mathbb{Q}}(\underline{\xi})\cdot\underline{n}
# = \boldsymbol{A}
#   - \boldsymbol{V}(\underline{\xi})\cdot\boldsymbol{N}^{-1}(\underline{\xi})
#     \cdot\boldsymbol{V}^{\mathsf{T}}(\underline{\xi}),
# \qquad
# \boldsymbol{A} = \underline{n}\cdot\mathbb{C}\cdot\underline{n},
# \qquad
# \boldsymbol{V}(\underline{\xi}) = \underline{n}\cdot\mathbb{C}\cdot\underline{\xi}.
# ```
#
# This is the identity every numerical back-end of the package already uses
# (`Core/green_helpers.jl`). Verified here rather than asserted, and used from
# § 7 on, where the order-4 route on five symbolic constants would be
# needlessly expensive.

println("="^78)                                               #jl
println("  § 2  REDUCED IDENTITY   n̲·Q̂·n̲ = A - V·N⁻¹·Vᵗ")    #jl
println("="^78)                                               #jl

"Components of `n̲·ℂ·n̲`, `n̲·ℂ·ξ̲` and `ξ̲·ℂ·ξ̲` as plain 3×3 `Sym` matrices."
reduced_blocks(C, n, ξ) = (Matrix(n ⋅ C ⋅ n), Matrix(n ⋅ C ⋅ ξ), Matrix(ξ ⋅ C ⋅ ξ))

let (A, V, Nm) = reduced_blocks(ℂ, 𝐧, 𝛏)
    gap = tsimplify.(A - V * inv(Nm) * transpose(V) - Matrix(𝐐ₙₙ))
    @assert all(iszero, gap) "the reduced identity does not reproduce n̲·Q̂·n̲"
    println("  reduced identity vs order-4 route : exact match")   #jl
    println()                                                     #jl
end

# ## §3 Behavior at infinity — why the integral converges
#
# ``\hat{\mathbb{Q}}`` is positively homogeneous of degree 0, so
# ``\hat{\mathbb{Q}}(\underline{\xi}^{\star}+\xi_3\underline{n})\cdot\underline{n}
# = \hat{\mathbb{Q}}(\underline{\xi}^{\star}/\xi_3+\underline{n})\cdot\underline{n}``
# and a large-``\xi_3`` expansion gives
#
# ```math
# \hat{\mathbb{Q}}(\underline{\xi}^{\star}+\xi_3\,\underline{n})\cdot\underline{n}
# = -\frac{1}{\xi_3}\,\hat{\mathbb{Q}}(\underline{n})\cdot\underline{\xi}^{\star}
#   + \mathcal{O}(\xi_3^{-2}),
# \qquad
# \underline{n}\cdot\hat{\mathbb{Q}}(\underline{\xi}^{\star}+\xi_3\,\underline{n})
# \cdot\underline{n} = \mathcal{O}(\xi_3^{-2}).
# ```
#
# The *second* contraction with ``\underline{n}`` is what kills the ``1/\xi_3``
# term: the doubly contracted kernel is absolutely integrable, whereas the
# singly contracted one converges only once replaced by its even part.
#
# Rather than compute limits at infinity, exploit the homogeneity directly:
# substituting ``\rho \to s\,\xi_3`` makes every component a function of the
# single variable ``s = \rho/\xi_3`` alone, and the large-``\xi_3`` expansion
# becomes an ordinary Taylor expansion at ``s = 0`` — three `subs` instead of
# eighteen `limit` calls.

println("="^78)                                                        #jl
println("  § 3  BEHAVIOR AT INFINITY  (degree-0 homogeneity)")         #jl
println("="^78)                                                        #jl

@syms s::positive

"Componentwise coefficient of `x^k` in the Taylor expansion of `a` at `x = 0`."
taylor_coeff(a, x, k) = [tsimplify(subs(sympy.diff(e, x, k), x => 0) / factorial(k)) for e in a]

"Components of `a̲ · T` after the homogeneity substitution `ρ → s ξ₃`."
in_s(T, a) = [subs(e, ρ => s * ξ₃) for e in Matrix(a ⋅ T)]

let 𝐐ₙ = ℚ ⋅ 𝐧,
        𝐐₁ = ℚ ⋅ 𝐞₁,
        dirs = (𝐞₁, 𝐞₂, 𝐞₃)
    ## s⁰ term: Q̂(n̲)·n̲ = 0 — the Green operator annihilates its own argument.
    @assert all(all(iszero, taylor_coeff(in_s(𝐐ₙ, a), s, 0)) for a in dirs) "Q̂(n̲)·n̲ ≠ 0"

    ## s¹ term: the article's eq-Qn, `Q̂(ξ̲*+ξ₃n̲)·n̲ = -(1/ξ₃) Q̂(n̲)·ξ̲* + O(ξ₃⁻²)`.
    ## Q̂(n̲) is `ℚ` at ρ = 0 (degree-0 homogeneity again), and ξ̲*/ξ₃ = s e₁.
    for a in dirs
        lhs = taylor_coeff(in_s(𝐐ₙ, a), s, 1)
        rhs = [tsimplify(-subs(e, ρ => 0)) for e in Matrix(a ⋅ 𝐐₁)]
        @assert all(iszero, tsimplify.(lhs - rhs)) "eq-Qn fails on the $(a) row"
    end
    println("  s⁰ term of Q̂·n̲ vanishes, s¹ term = -Q̂(n̲)·e₁   → the article's eq-Qn")   #jl

    ## Double contraction: the s¹ term dies too, so n̲·Q̂·n̲ = O(s²) = O(1/ξ₃²).
    nQn = in_s(ℚ ⋅ 𝐧, 𝐧)
    @assert all(iszero, taylor_coeff(nQn, s, 1)) "n̲·Q̂·n̲ should have no 1/ξ₃ term"
    @assert !all(iszero, taylor_coeff(nQn, s, 2)) "n̲·Q̂·n̲ should not vanish identically"
    e₁Qn = in_s(ℚ ⋅ 𝐧, 𝐞₁)
    @assert !all(iszero, taylor_coeff(e₁Qn, s, 1)) "e₁·Q̂·n̲ is expected to decay only like 1/ξ₃"
    println("  n̲·Q̂·n̲ = O(s²) = O(1/ξ₃²), integrable — the article's eq-nQn")   #jl
    println("  e₁·Q̂·n̲ = O(1/ξ₃) only → needs the even-part regularization")     #jl
    println()                                                                   #jl
end

# ## §4 The ``\xi_3`` integral — the hard step
#
# Three reductions come *before* any integration, and they are what make the
# calculation possible at all.
#
# 1. **Equivariance** (§ 0): one in-plane direction suffices.
# 2. **Homogeneity.** ``\hat{\mathbb{Q}}`` has degree 0, so
#    ``\hat{\boldsymbol{Q}}^{\star}_{nn}`` has degree 1 and the whole result
#    carries a single factor ``\rho``.
# 3. **Parity.** With ``\underline{u}=\underline{e}_1``, the
#    ``\underline{u}``–``\underline{n}`` block of the integrand is *odd* in
#    ``\xi_3`` and integrates to zero, so
#    ``\hat{\boldsymbol{Q}}^{\star}_{nn}`` is diagonal in
#    ``(\underline{u},\underline{w},\underline{n})``, with
#    ``\underline{w} = \underline{n}\wedge\underline{u}``:
#
# ```math
# \hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\xi}^{\star}) = \rho\,\bigl[
#   a_1\,\underline{u}\otimes\underline{u}
# + a_2\,\underline{w}\otimes\underline{w}
# + a_3\,\underline{n}\otimes\underline{n}\bigr].
# ```
#
# For an isotropic matrix the sextic degenerates completely,
# ``\det\boldsymbol{N} = \mu^2(\lambda+2\mu)\,\|\underline{\xi}\|^6``, leaving
# the single pair of double poles ``\xi_3 = \pm i\rho`` — and `integrate` closes
# the whole matrix in one pass.

println("="^78)                                                #jl
println("  § 4  THE ξ₃ INTEGRAL  (isotropic matrix)")          #jl
println("="^78)                                                #jl

let detN = tfactor(sympy.expand(det(Matrix(𝐍))))
    @assert iszero(tsimplify(detN - μ^2 * (λ + 2 * μ) * (ρ^2 + ξ₃^2)^3))
    println("  det N = ", detN)                                               #jl
    println("  → only the double poles ξ₃ = ±iρ; no Stroh sextic to solve")    #jl
end

"`(1/2π) ∫_{-∞}^{+∞} M(ξ₃) dξ₃`, component by component, shape preserved."
xi3_integral(M, x) = [integrate(M[i, j], (x, -OO, OO)) / (2 * PI) for i in axes(M, 1), j in axes(M, 2)]

𝐐ˢ = tsimplify.(xi3_integral(Matrix(𝐐ₙₙ), ξ₃))

a₁ = tsimplify(𝐐ˢ[1, 1] / ρ)
a₂ = tsimplify(𝐐ˢ[2, 2] / ρ)
a₃ = tsimplify(𝐐ˢ[3, 3] / ρ)

## Degree-1 homogeneity and the vanishing of the odd block, checked not assumed.
@assert all(iszero, [tsimplify(sympy.diff(e / ρ, ρ)) for e in 𝐐ˢ]) "Q̂*ₙₙ is not linear in ρ"
@assert iszero(tsimplify(𝐐ˢ[1, 3])) "the u̲–n̲ block should vanish by parity"
@assert iszero(tsimplify(a₁ - μ / (2 * (1 - ν)))) "a₁ ≠ μ/(2(1-ν))"
@assert iszero(tsimplify(a₂ - μ / 2)) "a₂ ≠ μ/2"
@assert iszero(tsimplify(a₃ - μ / (2 * (1 - ν)))) "a₃ ≠ μ/(2(1-ν))"

println("\n  Q̂*ₙₙ = ρ · diag(a₁, a₂, a₃) with")                            #jl
println("    a₁ = ", a₁, "   (u̲ ⊗ u̲, in-plane)")                          #jl
println("    a₂ = ", a₂, "   (w̲ ⊗ w̲, antiplane)")                         #jl
println("    a₃ = ", a₃, "   (n̲ ⊗ n̲, opening)")                           #jl
println("  i.e. a₁ = a₃ = μ/(2(1-ν)) = E/(4(1-ν²))  — plane-strain modulus")   #jl
println("       a₂ = μ/2                            — antiplane modulus")     #jl
println()                                                                     #jl

# ### The same result by Cauchy residues
#
# The integrand decays like ``\xi_3^{-2}``, so
# ``\int_{\mathbb{R}} = 2i\pi\sum_{\text{UHP}}\mathrm{Res}`` and
# ``\hat{\boldsymbol{Q}}^{\star}_{nn} = i\sum_{\text{UHP}}\mathrm{Res}``. This
# is the route the package's `Residue` back-end takes numerically
# (`Cracks/green_residue.jl`): there the poles are the roots of the degree-6
# ``\det\boldsymbol{N}(z)``, here they collapse to the single ``+i\rho``.

let Q = Matrix(𝐐ₙₙ),
        res = [IU * sympy.residue(Q[i, j], ξ₃, IU * ρ) for i in 1:3, j in 1:3]
    @assert all(iszero, tsimplify.(res - 𝐐ˢ)) "residue sum ≠ direct integration"
    println("  residue route vs direct integration : exact match")   #jl
    println()                                                        #jl
end

# ## §5 The crack-plane integral — where the elliptic integrals appear
#
# On the crack contour
# ``\underline{\xi}^{\star}(\varphi) = \eta\cos\varphi\,\underline{\ell}
# + \sin\varphi\,\underline{m}``, so that
# ``\rho = \sqrt{\eta^2\cos^2\varphi + \sin^2\varphi}``,
# ``\underline{u} = \underline{\xi}^{\star}/\rho`` and
# ``\underline{w} = (\eta\cos\varphi\,\underline{m} - \sin\varphi\,\underline{\ell})/\rho``.
# The ``\rho`` of the degree-1 homogeneity cancels the ``1/\rho^2`` of the two
# in-plane dyads, so **exactly three** angular integrals survive:
#
# ```math
# \frac14\!\int_0^{2\pi}\!\rho\,\mathrm{d}\varphi = \mathcal{E}_\eta,
# \qquad
# \frac14\!\int_0^{2\pi}\!\frac{\eta^2\cos^2\varphi}{\rho}\,\mathrm{d}\varphi
#   = \eta^2\mathcal{S}_\eta,
# \qquad
# \frac14\!\int_0^{2\pi}\!\frac{\sin^2\varphi}{\rho}\,\mathrm{d}\varphi
#   = \mathcal{C}_\eta,
# ```
#
# with ``\mathcal{C}_\eta = (\mathcal{E}_\eta-\eta^2\mathcal{K}_\eta)/(1-\eta^2)``
# and ``\mathcal{S}_\eta = (\mathcal{K}_\eta-\mathcal{E}_\eta)/(1-\eta^2)`` — the
# very combinations the package stores in `_elliptic_CS`. Hence
#
# ```math
# b\boldsymbol{\Lambda} = \mathrm{diag}\bigl(
#   a_1\eta^2\mathcal{S}_\eta + a_2\mathcal{C}_\eta,\;
#   a_1\mathcal{C}_\eta + a_2\eta^2\mathcal{S}_\eta,\;
#   a_3\mathcal{E}_\eta\bigr),
# \qquad
# \boldsymbol{B}^{\mathcal{E}} = \tfrac{2}{3}\,(b\boldsymbol{\Lambda})^{-1}.
# ```
#
# The cross term ``\int\cos\varphi\sin\varphi/\rho`` vanishes by parity, which
# is why ``\boldsymbol{B}`` comes out diagonal in the crack frame. These three
# quadratures are the one step SymPy cannot close, so they are checked
# numerically against `ell_K` / `ell_E`.
#
# !!! note "Why cos² and sin² look exchanged against the literature"
#     [barthelemySifAniso](@cite) defines the same two quantities over the
#     **complementary** angle ``\vartheta = \pi/2-\varphi``, for which the
#     radical reads ``\sqrt{\cos^{2}\vartheta+\eta^{2}\sin^{2}\vartheta} = \rho``:
#     there ``\mathcal{C}_\eta`` is the ``\cos^{2}`` integral and
#     ``\mathcal{S}_\eta`` the ``\sin^{2}`` one. Substituting swaps ``\cos`` and
#     ``\sin`` in the integrand *and* in the radical, so the numbers are the
#     same — that is what the check below measures, against the Legendre forms
#     that both conventions share. The ``\varphi`` used here is the angle the
#     package integrates over (`Cracks._cod_elliptic_numerical`).

println("="^78)                                                            #jl
println("  § 5  CRACK-PLANE INTEGRAL → ELLIPTIC INTEGRALS")                #jl
println("="^78)                                                            #jl

@syms η::positive

𝒦η = ell_K(1 - η^2)
ℰη = ell_E(1 - η^2)
𝒞η = (ℰη - η^2 * 𝒦η) / (1 - η^2)
𝒮η = (𝒦η - ℰη) / (1 - η^2)

"COD tensor of an elliptic crack from the three coefficients of `Q̂*ₙₙ`."
function cod_from_coefficients(c₁, c₂, c₃, ηs, Cη, Sη, Eη)
    bΛ = (c₁ * ηs^2 * Sη + c₂ * Cη, c₁ * Cη + c₂ * ηs^2 * Sη, c₃ * Eη)
    return [Sym(2) / 3 / e for e in bΛ]
end

ρφ(ηv, φ) = sqrt(ηv^2 * cos(φ)^2 + sin(φ)^2)

"The four candidate angular integrals of § 5, by adaptive quadrature."
function master_numeric(ηv)
    q(f) = quadgk(f, 0.0, 2π; rtol = 1.0e-13)[1] / 4
    return (
        q(φ -> ρφ(ηv, φ)),
        q(φ -> ηv^2 * cos(φ)^2 / ρφ(ηv, φ)),
        q(φ -> sin(φ)^2 / ρφ(ηv, φ)),
        q(φ -> ηv * cos(φ) * sin(φ) / ρφ(ηv, φ)),
    )
end

"""
    sifaniso_CS(ηv) -> (𝒞, 𝒮)

`𝒞_η` and `𝒮_η` in the published form: over the complementary angle
``ϑ = π/2 - φ``, so a `cos²` integral for `𝒞` and a `sin²` one for `𝒮`.
Measuring both forms against the same Legendre reference is what shows the two
conventions to be one and the same quantity.
"""
function sifaniso_CS(ηv)
    r(ϑ) = sqrt(cos(ϑ)^2 + ηv^2 * sin(ϑ)^2)
    q(f) = quadgk(f, 0.0, π / 2; rtol = 1.0e-13)[1]
    return q(ϑ -> cos(ϑ)^2 / r(ϑ)), q(ϑ -> sin(ϑ)^2 / r(ϑ))
end

for ηv in (0.85, 0.5, 0.2)
    Eq, S2q, Cq, cross = master_numeric(ηv)
    ref = (
        Float64(N(ℰη(η => ηv))),
        ηv^2 * Float64(N(𝒮η(η => ηv))),
        Float64(N(𝒞η(η => ηv))),
    )
    err = maximum(abs, (Eq, S2q, Cq) .- ref)
    ## Same two numbers, written the other way round (see the note above).
    C_sif, S_sif = sifaniso_CS(ηv)
    err_sif = max(abs(C_sif - ref[3]), abs(S_sif - Float64(N(𝒮η(η => ηv)))))
    @printf "  η = %.2f : max|quadrature - (ℰη, η²𝒮η, 𝒞η)| = %.2e,  cross term = %.2e,  vs published form = %.2e\n" ηv err abs(cross) err_sif   #jl
    @assert err < 1.0e-9 "master integral mismatch at η = $ηv"
    @assert abs(cross) < 1.0e-12 "the cross term should vanish by parity"
    @assert err_sif < 1.0e-9 "the complementary-angle form disagrees at η = $ηv"
end
println()   #jl

# ## §6 The isotropic COD tensor, and its penny and ribbon limits
#
# Assembling § 4 and § 5, and comparing with [`cod_tensor`](@ref).

println("="^78)                                                          #jl
println("  § 6  ISOTROPIC 𝐁  vs  cod_tensor")                            #jl
println("="^78)                                                          #jl

B_iso = tsimplify.(cod_from_coefficients(a₁, a₂, a₃, η, 𝒞η, 𝒮η, ℰη))

println("  B_ℓℓ = ", B_iso[1])   #jl
println("  B_mm = ", B_iso[2])   #jl
println("  B_nn = ", B_iso[3])   #jl

let B_code = cod_tensor(EllipticCrack(one(Sym), η), ℂ)
    gap = [tsimplify(B_iso[i] - B_code[i, i]) for i in 1:3]
    @assert all(iszero, gap) "symbolic 𝐁 ≠ cod_tensor for an isotropic matrix"
    println("\n  vs cod_tensor(EllipticCrack(1, η), ℂ) : exact match")   #jl
end

## χ = 8(1-ν²)/(3E) — the common prefactor, with no η in it.
@assert iszero(tsimplify(B_iso[3] * ℰη - 8 * (1 - ν^2) / (3 * E))) "χ ≠ 8(1-ν²)/(3E)"
println("  common prefactor χ = 8(1-ν²)/(3E)   (no η)")   #jl

# ### Penny crack ``\eta = 1``
#
# ``\mathcal{K}_1 = \mathcal{E}_1 = \pi/2`` and
# ``\mathcal{C}_1 = \mathcal{S}_1 = \pi/4``, which the limit recovers on its own.

let B_penny = [tsimplify(sympy.limit(e, η, 1)) for e in B_iso],
        Bnn = 16 * (1 - ν^2) / (3 * PI * E)
    @assert iszero(tsimplify(sympy.limit(𝒞η, η, 1) - PI / 4)) "𝒞₁ ≠ π/4"
    @assert iszero(tsimplify(sympy.limit(𝒮η, η, 1) - PI / 4)) "𝒮₁ ≠ π/4"
    @assert iszero(tsimplify(B_penny[3] - Bnn)) "penny B_nn ≠ 16(1-ν²)/(3πE)"
    @assert iszero(tsimplify(B_penny[1] - Bnn / (1 - ν / 2))) "penny B_ℓℓ ≠ B_nn/(1-ν/2)"
    @assert iszero(tsimplify(B_penny[1] - B_penny[2])) "penny 𝐁 is not in-plane isotropic"
    println("\n  penny (η→1) : B_nn = ", B_penny[3], ",  B_ℓℓ = B_mm = B_nn/(1-ν/2)")   #jl
end

# ### Ribbon crack
#
# Here ``\underline{u} = \underline{m}``, so
# ``\underline{w} = \underline{n}\wedge\underline{m} = -\underline{\ell}`` and
# the roles of ``a_1`` and ``a_2`` swap. The prefactor is
# ``\chi^{\mathcal{R}} = \pi/4`` instead of ``\chi^{\mathcal{E}} = 2/3``, which
# is the entire origin of the ``3\pi/8`` between
# ``\boldsymbol{B}^{\mathcal{R}}`` and
# ``\lim_{\eta\to0}\boldsymbol{B}^{\mathcal{E}}``.

B_rib = [PI / 4 / a for a in (a₂, a₁, a₃)]

let B_code = cod_tensor(RibbonCrack(one(Sym)), ℂ)
    gap = [tsimplify(B_rib[i] - B_code[i, i]) for i in 1:3]
    @assert all(iszero, gap) "symbolic 𝐁ᴿ ≠ cod_tensor for a ribbon"
    println("\n  ribbon : vs cod_tensor(RibbonCrack(1), ℂ) : exact match")   #jl
end

# The ``3\pi/8`` follows from ``\mathcal{C}_\eta\to1``, ``\eta^2\mathcal{S}_\eta\to0``
# and ``\mathcal{E}_\eta\to1``, which turn ``b\boldsymbol{\Lambda}^{\mathcal{E}}``
# into ``(a_2,a_1,a_3)`` — the very triple the ribbon uses. `sympy.limit` cannot
# take that limit (``\mathcal{K}_\eta`` diverges logarithmically and `gruntz`
# gives up), so the convergence is exhibited numerically instead.

let vals = Dict(E => Sym(1), ν => Sym(1) / 4)
    ref = [Float64(N(subs(e, vals))) for e in B_rib]
    err = 0.0
    for ηv in (1.0e-2, 1.0e-3, 1.0e-4)
        approx = [Float64(N(subs(subs(3 * PI / 8 * e, vals), η => Sym(ηv)))) for e in B_iso]
        err = maximum(abs, approx - ref) / maximum(abs, ref)
        @printf "  (3π/8)·𝐁ᴱ(η=%.0e) vs 𝐁ᴿ : relative gap = %.2e\n" ηv err   #jl
    end
    @assert err < 1.0e-6 "𝐁ᴿ ≠ (3π/8)·lim_{η→0} 𝐁ᴱ"
    println()   #jl
end

# ## §7 Transversely isotropic matrix, crack in the plane of isotropy
#
# The TI axis is ``\underline{e}_3 = \underline{n}``. Two facts make this case
# barely harder than the isotropic one.
#
# **The acoustic tensor block-decouples.** Every component
# ``N_{12}, N_{23}`` carries an odd number of `2` indices, so it vanishes: the
# antiplane (SH) polarization ``\underline{e}_2`` separates from the in-plane
# pair ``(\underline{e}_1,\underline{e}_3)``, and
#
# ```math
# \det\boldsymbol{N}
# = \underbrace{C_{2323}\bigl(\xi_3^2+\gamma_3^2\rho^2\bigr)}_{\text{antiplane}}
#   \cdot
#   \underbrace{C_{3333}C_{2323}\bigl(\xi_3^2+\gamma_1^2\rho^2\bigr)
#               \bigl(\xi_3^2+\gamma_2^2\rho^2\bigr)}_{\text{in-plane}} .
# ```
#
# The in-plane factor is *biquadratic*, hence solvable by radicals, and
#
# ```math
# \gamma_3^2 = \frac{C_{1212}}{C_{2323}},
# \qquad
# \gamma_1\gamma_2 = \sqrt{\frac{C_{1111}}{C_{3333}}},
# \qquad
# \gamma_1+\gamma_2 = \sigma_\gamma .
# ```
#
# That last identity *is* the published radical ``\sigma_\gamma``, obtained here
# rather than quoted.
#
# **The angular structure is unchanged.** The matrix is still invariant under
# rotations about ``\underline{n}``, so § 5 applies verbatim — same three
# elliptic integrals, same assembly. This is exactly why the shipped TI closed
# form has the same ``\mathcal{C},\mathcal{S},\mathcal{E}`` skeleton as the
# isotropic one.

println("="^78)                                                              #jl
println("  § 7  TRANSVERSELY ISOTROPIC, CRACK IN THE ISOTROPY PLANE")        #jl
println("="^78)                                                              #jl

## The two shear moduli are taken as independent symbols, *declared positive*.
## `C₁₁₂₂ = C₁₁₁₁ - 2C₁₂₁₂` is then derived. Writing it the other way round —
## `C₁₂₁₂ = (C₁₁₁₁-C₁₁₂₂)/2` with `C₁₁₂₂` merely real — leaves the sign of the
## antiplane denominator undecided and `integrate` answers with a `Piecewise`.
@syms C₁₁₁₁::positive C₁₂₁₂::positive C₁₁₃₃::real C₃₃₃₃::positive C₂₃₂₃::positive

C₁₁₂₂ = C₁₁₁₁ - 2 * C₁₂₁₂
ℂᵗⁱ = tens_TI(C₁₁₁₁, C₁₁₂₂, C₁₁₃₃, C₃₃₃₃, C₂₃₂₃, [Sym(0), Sym(0), Sym(1)])

Aᵗⁱ, Vᵗⁱ, Nᵗⁱ = reduced_blocks(ℂᵗⁱ, 𝐧, 𝛏)

@assert iszero(tsimplify(Nᵗⁱ[1, 2])) "the SH polarization should decouple (N₁₂)"
@assert iszero(tsimplify(Nᵗⁱ[2, 3])) "the SH polarization should decouple (N₂₃)"
@assert iszero(tsimplify(Vᵗⁱ[2, 1])) "the SH polarization should decouple (V₂₁)"
@assert iszero(tsimplify(Vᵗⁱ[2, 3])) "the SH polarization should decouple (V₂₃)"
println("  N₁₂ = N₂₃ = V₂₁ = V₂₃ = 0  → the SH polarization decouples")   #jl

# ### The antiplane block — one line
#
# ``N_{22} = C_{1212}\rho^2 + C_{2323}\xi_3^2`` and ``V_{22} = C_{2323}\xi_3``,
# so the ``\underline{w}`` coefficient is the geometric mean of the two shear
# moduli.

a₂ᵗⁱ = tsimplify(
    integrate(Aᵗⁱ[2, 2] - Vᵗⁱ[2, 2]^2 / Nᵗⁱ[2, 2], (ξ₃, -OO, OO)) / (2 * PI) / ρ
)

println("  a₂ = ", a₂ᵗⁱ, "   = √(C₂₃₂₃·C₁₂₁₂)/2")   #jl

## Compared through the squares: SymPy keeps `√C₂₃₂₃·√C₁₂₁₂` unmerged, and both
## sides are positive (`a₂` integrates a positive quantity), so this is equality.
@assert iszero(tsimplify(a₂ᵗⁱ^2 - C₂₃₂₃ * C₁₂₁₂ / 4)) "a₂ᵗⁱ ≠ √(C₂₃₂₃C₁₂₁₂)/2"

# ### The in-plane block — factorize, then integrate
#
# The 2×2 inverse is written with the cofactor formula so the denominator is
# *exactly* ``\det\boldsymbol{N}_{(13)}``, which is then replaced by its
# factored form in ``\gamma_1,\gamma_2``. Without that substitution `integrate`
# faces an unfactored quartic and answers with `RootOf` expressions.

@syms γ₁::positive γ₂::positive

N₂ = Sym[Nᵗⁱ[1, 1] Nᵗⁱ[1, 3]; Nᵗⁱ[3, 1] Nᵗⁱ[3, 3]]
V₂ = Sym[Vᵗⁱ[1, 1] Vᵗⁱ[1, 3]; Vᵗⁱ[3, 1] Vᵗⁱ[3, 3]]
A₂ = Sym[Aᵗⁱ[1, 1] Aᵗⁱ[1, 3]; Aᵗⁱ[3, 1] Aᵗⁱ[3, 3]]

quart = sympy.expand(det(N₂))
quart_fac = C₃₃₃₃ * C₂₃₂₃ * (ξ₃^2 + γ₁^2 * ρ^2) * (ξ₃^2 + γ₂^2 * ρ^2)

"Coefficient of `x^n` in a polynomial expression, by differentiation."
poly_coeff(e, x, n) = tsimplify(subs(sympy.diff(e, x, n), x => 0) / factorial(n))

c₄ = poly_coeff(quart, ξ₃, 4)
c₂ = poly_coeff(quart, ξ₃, 2)
c₀ = poly_coeff(quart, ξ₃, 0)

e₂γ = tsimplify(c₀ / c₄ / ρ^4)          # γ₁²γ₂²
p₂γ = tsimplify(c₂ / c₄ / ρ^2)          # γ₁²+γ₂²
σᵞ = sqrt(p₂γ + 2 * sqrt(e₂γ))          # γ₁+γ₂

σᵞ_paper = sqrt(
    (C₁₁₁₁ * C₃₃₃₃ - C₁₁₃₃^2 - 2 * C₁₁₃₃ * C₂₃₂₃) / (C₂₃₂₃ * C₃₃₃₃) +
        2 * sqrt(C₁₁₁₁ / C₃₃₃₃),
)

## `σᵞ = √(p₂γ + 2√e₂γ)` by construction, so proving that it is the published
## radical only takes the two *rational* identities below — no nested-radical
## comparison, which SymPy would not close.
@assert all(iszero, [poly_coeff(quart, ξ₃, n) for n in (1, 3)]) "the in-plane factor is not biquadratic"
@assert iszero(tsimplify(c₄ - C₃₃₃₃ * C₂₃₂₃)) "leading coefficient ≠ C₃₃₃₃C₂₃₂₃"
@assert iszero(tsimplify(e₂γ - C₁₁₁₁ / C₃₃₃₃)) "γ₁²γ₂² ≠ C₁₁₁₁/C₃₃₃₃"
@assert iszero(
    tsimplify(
        p₂γ - (C₁₁₁₁ * C₃₃₃₃ - C₁₁₃₃^2 - 2 * C₁₁₃₃ * C₂₃₂₃) / (C₂₃₂₃ * C₃₃₃₃)
    )
) "γ₁²+γ₂² does not match the published radicand"

println("  γ₁²γ₂² = ", e₂γ)                                        #jl
println("  γ₁²+γ₂² = ", p₂γ)                                       #jl
println("  σᵞ = γ₁+γ₂ = ", σᵞ,  "   → matches the published σᵞ")   #jl

"""
    factored_xi3_integral(expr, x, den_expanded, den_factored)

`(1/2π) ∫ expr dx` over the whole real line, where the denominator of `expr` is
proportional to `den_expanded` and is rewritten over `den_factored` first. The
proportionality factor is checked to be free of `x`.
"""
function factored_xi3_integral(expr, x, den_expanded, den_factored)
    num, den = sympy.fraction(sympy.cancel(sympy.together(expr)))
    r = tsimplify(den / den_expanded)
    @assert iszero(tsimplify(sympy.diff(r, x))) "the denominator is not ∝ the expected one"
    return integrate(num / (r * den_factored), (x, -OO, OO)) / (2 * PI)
end

Q₂ˢ = let adj = Sym[N₂[2, 2] -N₂[1, 2]; -N₂[2, 1] N₂[1, 1]],
        M = V₂ * adj * transpose(V₂)
    [
        factored_xi3_integral(A₂[i, j] - M[i, j] / quart, ξ₃, quart, quart_fac)
            for i in 1:2, j in 1:2
    ]
end

a₁ᵗⁱ = tsimplify(Q₂ˢ[1, 1] / ρ)
a₃ᵗⁱ = tsimplify(Q₂ˢ[2, 2] / ρ)

@assert iszero(tsimplify(Q₂ˢ[1, 2])) "the u̲–n̲ block should vanish by parity"
@assert iszero(
    tsimplify(a₁ᵗⁱ * 2 * (γ₁ + γ₂) * C₃₃₃₃ - (C₁₁₁₁ * C₃₃₃₃ - C₁₁₃₃^2))
) "a₁ᵗⁱ ≠ (C₁₁₁₁C₃₃₃₃-C₁₁₃₃²)/(2σᵞC₃₃₃₃)"
@assert iszero(tsimplify(a₃ᵗⁱ * γ₁ * γ₂ - a₁ᵗⁱ)) "a₃ᵗⁱ ≠ a₁ᵗⁱ/(γ₁γ₂)"

println("  a₁ = ", a₁ᵗⁱ)                                                #jl
println("  a₃ = ", a₃ᵗⁱ)                                                #jl
println("  i.e. a₁ = (C₁₁₁₁C₃₃₃₃-C₁₁₃₃²)/(2σᵞC₃₃₃₃),  a₃ = a₁/(γ₁γ₂)")   #jl
println()                                                                #jl

# ### The TI COD tensor
#
# The two identities just proved eliminate ``\gamma_1,\gamma_2`` in favor of
# ``\sigma_\gamma`` and ``\sqrt{C_{1111}/C_{3333}}``, § 5 is reused unchanged,
# and the result is compared with [`cod_tensor`](@ref) on the same `TensTI`
# stiffness. The package reaches its closed form through the engineering
# parameters ``(E,\nu_1,\nu_2,H,\Gamma)`` while this script never leaves the
# stiffness components, so the agreement cross-checks two independent
# parameterizations. The symbolic identity between them is a heavy `simplify`,
# so it is verified on exact rational values instead.

## `σ` stands in for `σᵞ` so that the two coefficients stay *rational* in it;
## substituting the radical only at the end keeps every `simplify` tractable.
@syms σ::positive

a₁_σ = (C₁₁₁₁ * C₃₃₃₃ - C₁₁₃₃^2) / (2 * σ * C₃₃₃₃)
a₃_σ = a₁_σ * sqrt(C₃₃₃₃ / C₁₁₁₁)          # a₃ = a₁/(γ₁γ₂), γ₁γ₂ = √(C₁₁₁₁/C₃₃₃₃)

a₁ᵞ = subs(a₁_σ, σ => σᵞ_paper)
a₃ᵞ = subs(a₃_σ, σ => σᵞ_paper)
B_ti = cod_from_coefficients(a₁ᵞ, a₂ᵗⁱ, a₃ᵞ, η, 𝒞η, 𝒮η, ℰη)

## `Cv = (C₁₁₁₁, C₁₁₂₂, C₁₁₃₃, C₃₃₃₃, C₂₃₂₃)` — the argument order of `tens_TI`;
## the derivation's independent symbol is `C₁₂₁₂ = (C₁₁₁₁-C₁₁₂₂)/2`.
ti_subs(Cv) = Dict(
    C₁₁₁₁ => Sym(Cv[1]), C₁₂₁₂ => Sym(Cv[1] - Cv[2]) / 2, C₁₁₃₃ => Sym(Cv[3]),
    C₃₃₃₃ => Sym(Cv[4]), C₂₃₂₃ => Sym(Cv[5]),
)

for Cv in ((14, 6, 5, 11, 3), (9, 4, 3, 20, 5)), ηv in (2 // 5, 7 // 10, 9 // 10)
    ℂnum = tens_TI(Sym.(Cv)..., [Sym(0), Sym(0), Sym(1)])
    B_code = cod_tensor(EllipticCrack(one(Sym), Sym(ηv)), ℂnum)
    B_here = [subs(subs(e, ti_subs(Cv)), η => Sym(ηv)) for e in B_ti]
    err = maximum(abs(Float64(N(B_here[i] - B_code[i, i]))) for i in 1:3)
    @printf "  C = %-18s η = %-5s : max|𝐁_here - 𝐁_code| = %.2e\n" string(Cv) string(ηv) err   #jl
    @assert err < 1.0e-12 "TI 𝐁 mismatch for C = $Cv, η = $ηv"
end

## Penny: substitute the limit values 𝒞₁ = 𝒮₁ = π/4, ℰ₁ = π/2 (verified in § 6)
## rather than take a 0/0 limit of 𝒮η at η = 1.
let Cv = (14, 6, 5, 11, 3),
        B_pen = cod_from_coefficients(a₁ᵞ, a₂ᵗⁱ, a₃ᵞ, Sym(1), PI / 4, PI / 4, PI / 2),
        B_code = cod_tensor(PennyCrack(one(Sym)), tens_TI(Sym.(Cv)..., [Sym(0), Sym(0), Sym(1)]))
    err = maximum(abs(Float64(N(subs(B_pen[i], ti_subs(Cv)) - B_code[i, i]))) for i in 1:3)
    @printf "  C = %-18s penny   : max|𝐁_here - 𝐁_code| = %.2e\n" string(Cv) err   #jl
    @assert err < 1.0e-12 "TI penny 𝐁 mismatch"
end

# ### Isotropic reduction
#
# Setting ``C_{1111}=C_{3333}=\lambda+2\mu``, ``C_{1122}=C_{1133}=\lambda`` and
# ``C_{2323}=\mu`` must give back § 4 — and in particular
# ``\sigma_\gamma = 2``, the value the published reduction rule postulates.

let iso = Dict(
            C₁₁₁₁ => λ + 2 * μ, C₁₂₁₂ => μ, C₁₁₃₃ => λ,
            C₃₃₃₃ => λ + 2 * μ, C₂₃₂₃ => μ,
        )
    σ_iso = tsimplify(subs(σᵞ_paper, iso))
    @assert iszero(tsimplify(σ_iso - 2)) "σᵞ does not reduce to 2"
    @assert iszero(tsimplify(subs(subs(a₁_σ, iso), σ => σ_iso) - a₁)) "a₁ᵗⁱ does not reduce to a₁ᶦˢᵒ"
    @assert iszero(tsimplify(subs(a₂ᵗⁱ, iso)^2 - a₂^2)) "a₂ᵗⁱ does not reduce to a₂ᶦˢᵒ"
    @assert iszero(tsimplify(subs(subs(a₃_σ, iso), σ => σ_iso) - a₃)) "a₃ᵗⁱ does not reduce to a₃ᶦˢᵒ"
    println("\n  isotropic reduction : σᵞ = 2, and a₁, a₂, a₃ recover § 4")   #jl
    println()                                                                #jl
end

# ## §8 Where the symbolic route stops
#
# Tilt the TI axis away from the crack normal and both enabling structures
# collapse at once — no need to go to a full triclinic stiffness. With an axis
# that lies in no symmetry plane of the crack frame:
#
# * ``N_{12} \ne 0``: the SH polarization no longer decouples, so the sextic
#   does not split off a quadratic factor;
# * odd powers of ``\xi_3`` appear in ``\det\boldsymbol{N}``, so what remains is
#   not even a polynomial in ``\xi_3^2`` — the biquadratic trick of § 7 has
#   nothing to work on, and the six Stroh roots have no radical expression in
#   general.
#
# On top of that the crack plane has lost its rotational symmetry, so a *single*
# in-plane direction no longer determines the others and the ``\varphi``
# integral no longer reduces to three master integrals. Both integrals become
# numerical — which is precisely what the `Residue` and cubature back-ends of
# the package do, the first summing residues over the six roots located
# numerically.

println("="^78)                                             #jl
println("  § 8  WHERE THE SYMBOLIC ROUTE STOPS")            #jl
println("="^78)                                             #jl

let Cv = (14, 6, 5, 11, 3),
        axis = [Sym(1), Sym(1), Sym(1)] / sqrt(Sym(3)),
        ℂtilt = tens_TI(Sym.(Cv)..., axis),
        Ntilt = Matrix(𝛏 ⋅ ℂtilt ⋅ 𝛏),
        dets = sympy.expand(det(Ntilt)),
        odd = [poly_coeff(dets, ξ₃, n) for n in (1, 3, 5)]
    @assert !iszero(tsimplify(Ntilt[1, 2])) "expected no SH decoupling for a tilted axis"
    @assert !all(iszero, odd) "expected odd powers of ξ₃ in det N"
    @assert iszero(poly_coeff(dets, ξ₃, 7)) "det N should have degree 6 in ξ₃"
    println("  TI axis tilted to (1,1,1)/√3 :")                                  #jl
    println("    N₁₂ = ", Ntilt[1, 2], "  ≠ 0   → no SH decoupling")             #jl
    println("    coefficients of ξ₃¹, ξ₃³, ξ₃⁵ = ", odd, "  → not biquadratic")   #jl
    println("  ⇒ hand over to cod_tensor(…; method = :residues | :decuhr)")      #jl
end

println()                                                                 #jl
println("="^78)                                                           #jl
println("  All symbolic identities verified: Green operator → Q̂*ₙₙ → 𝐁.")  #jl
println("="^78)                                                           #jl
