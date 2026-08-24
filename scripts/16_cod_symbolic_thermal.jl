# # [From the Green operator to the thermal COD scalar](@id tut-cod-symbolic-thermal)
#
# The transport twin of `09_cod_symbolic_green.jl`, carried out symbolically —
# and carried **further**, because the order-2 problem is analytic where the
# elastic one is not.
#
# The chain is the same as in elasticity, with the conductivity ``\boldsymbol{K}_0``
# in place of the stiffness:
#
# ```math
# \hat{\boldsymbol{\Gamma}}(\underline{\xi})
#   = \frac{\underline{\xi}\otimes\underline{\xi}}
#          {\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}},
# \qquad
# \hat{\boldsymbol{Q}}(\underline{\xi})
#   = \boldsymbol{K}_0 - \boldsymbol{K}_0\cdot\hat{\boldsymbol{\Gamma}}\cdot\boldsymbol{K}_0,
# \qquad
# \hat{Q}^{\star}_{nn}(\underline{\xi}^{\star})
#   = \frac{1}{2\pi}\int_{-\infty}^{+\infty}
#     \underline{n}\cdot\hat{\boldsymbol{Q}}
#     (\underline{\xi}^{\star}+\xi_n\underline{n})\cdot\underline{n}\;\mathrm{d}\xi_n .
# ```
#
# One difference decides everything. The elastic acoustic tensor
# ``\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}`` is a 3×3 matrix, so
# ``\det\boldsymbol{N}`` is a **sextic** in ``\xi_n`` and the ``\xi_n`` integral
# only closes when that sextic factorizes — which is why
# `09_cod_symbolic_green.jl` stops at transverse isotropy aligned with the crack.
# Here ``\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}`` is a
# **scalar**, a quadratic in ``\xi_n``, and the integral closes for *every*
# anisotropy. There is no § "where the symbolic route stops".
#
# What the script establishes, with the six components of ``\boldsymbol{K}_0``
# free symbols throughout:
#
# ```math
# \hat{Q}^{\star}_{nn}(\underline{\xi}^{\star})
# = \tfrac12\sqrt{(\underline{n}\cdot\boldsymbol{K}_0\underline{n})
#                 (\underline{\xi}^{\star}\cdot\boldsymbol{K}_0\underline{\xi}^{\star})
#               - (\underline{n}\cdot\boldsymbol{K}_0\underline{\xi}^{\star})^{2}}
# = \tfrac12\sqrt{(\underline{n}\wedge\underline{\xi}^{\star})\cdot
#                 \mathrm{adj}\,\boldsymbol{K}_0\cdot
#                 (\underline{n}\wedge\underline{\xi}^{\star})},
# ```
#
# and then, on the crack contour, that an arbitrarily anisotropic conductor
# behaves as an **isotropic** one of conductivity ``\sqrt{\lambda_1}`` around a
# crack of **effective** aspect ratio ``\eta' = \sqrt{\lambda_2/\lambda_1}``,
# where ``\lambda_1\ge\lambda_2`` are the eigenvalues of a 2×2 in-plane
# restriction of ``\mathrm{adj}\,\boldsymbol{K}_0`` — a quadratic, hence closed
# form. That reduction is the analytic maximum of the problem.
#
# Prerequisite: `Pkg.add("SymPy")`.
#
# **§0** Setup · **§1** Order-2 Green operator · **§2** The numerator collapse ·
# **§3** The ``\xi_n`` integral, for any anisotropy · **§4** The crack-plane
# integral and the effective ellipse · **§5** Specializations · **§6**
# Independent checks.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using SymPy
using LinearAlgebra
using Tensors
using QuadGK
using Printf

const OO = sympy.oo

println("=== From the Green operator to the thermal COD scalar — symbolic ===")   #jl
println()                                                                        #jl

# ## §0 Setup — a fully anisotropic conductor from the start
#
# Unlike the elastic script, which had to specialize the symmetry class before
# the integral would close, here the six independent components of
# ``\boldsymbol{K}_0`` stay free the whole way.

println("="^78)                                                     #jl
println("  § 0  SETUP — fully anisotropic K₀, crack normal e₃")      #jl
println("="^78)                                                     #jl

@syms K₁₁::positive K₂₂::positive K₃₃::positive K₁₂::real K₁₃::real K₂₃::real
@syms ρ::positive ξₙ::real

ℬ = CanonicalBasis{3, Sym}()
𝐞₁, 𝐞₂, 𝐞₃ = tens_basis(ℬ, 1), tens_basis(ℬ, 2), tens_basis(ℬ, 3)
𝐧 = 𝐞₃

Karr = Sym[K₁₁ K₁₂ K₁₃; K₁₂ K₂₂ K₂₃; K₁₃ K₂₃ K₃₃]
𝐊 = Tens(SymmetricTensor{2, 3}((i, j) -> Karr[i, j]), ℬ)

𝛏★ = ρ * 𝐞₁                       # one in-plane direction; the azimuth is restored in § 4
𝛏 = 𝛏★ + ξₙ * 𝐧

println("  K₀ : 6 free symbols,  n̲ = e₃,  ξ̲ = ρ e₁ + ξₙ e₃")   #jl
println()                                                      #jl

# ## §1 The order-2 Green operator — why it is easier
#
# The acoustic object is
# ``N(\underline{\xi}) = \underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}``,
# a **scalar**. No inverse of a 3×3 matrix, no adjugate polynomial, no Stroh
# roots: ``\hat{\boldsymbol{\Gamma}}`` is a rank-1 tensor divided by that scalar.

println("="^78)                                            #jl
println("  § 1  ORDER-2 GREEN OPERATOR")                   #jl
println("="^78)                                            #jl

Nq = 𝛏 ⋅ 𝐊 ⋅ 𝛏
@assert Nq isa Sym "the order-2 acoustic form must be a scalar"

𝚪 = (𝛏 ⊗ 𝛏) / Nq
𝐐 = 𝐊 - 𝐊 ⋅ 𝚪 ⋅ 𝐊
Qₙₙ = tsimplify(𝐧 ⋅ 𝐐 ⋅ 𝐧)

println("  N(ξ̲) = ξ̲·K₀·ξ̲ is a ", typeof(Nq), " — a scalar, quadratic in ξₙ")   #jl
println("  n̲·Q̂·n̲ = ", Qₙₙ)                                                    #jl
println()                                                                      #jl

# ## §2 The numerator collapses to a constant
#
# Write the three contractions
#
# ```math
# a = \underline{n}\cdot\boldsymbol{K}_0\underline{n},
# \qquad
# b = \underline{n}\cdot\boldsymbol{K}_0\underline{\xi}^{\star},
# \qquad
# c = \underline{\xi}^{\star}\cdot\boldsymbol{K}_0\underline{\xi}^{\star},
# ```
#
# so that ``N = a\,\xi_n^{2} + 2b\,\xi_n + c`` and
# ``\underline{n}\cdot\boldsymbol{K}_0\underline{\xi} = b + a\,\xi_n``. Then
#
# ```math
# a\,N - (b + a\,\xi_n)^{2} = ac - b^{2}
# \qquad\text{identically —  every power of } \xi_n \text{ cancels,}
# ```
#
# so the kernel is a **constant over a quadratic**:
# ``\underline{n}\cdot\hat{\boldsymbol{Q}}\cdot\underline{n} = (ac-b^{2})/N``.
# That is the whole reason the transport problem is analytic: the elastic kernel
# is a quartic over a sextic and its numerator does not collapse.

println("="^78)                                                  #jl
println("  § 2  THE NUMERATOR COLLAPSES TO A CONSTANT")          #jl
println("="^78)                                                  #jl

a₀ = tsimplify(𝐧 ⋅ 𝐊 ⋅ 𝐧)
b₀ = tsimplify(𝐧 ⋅ 𝐊 ⋅ 𝛏★)
c₀ = tsimplify(𝛏★ ⋅ 𝐊 ⋅ 𝛏★)

@assert iszero(tsimplify(Nq - (a₀ * ξₙ^2 + 2 * b₀ * ξₙ + c₀))) "N is not the expected quadratic"
@assert iszero(tsimplify(a₀ * Nq - (b₀ + a₀ * ξₙ)^2 - (a₀ * c₀ - b₀^2))) "the numerator does not collapse"
@assert iszero(tsimplify(Qₙₙ - (a₀ * c₀ - b₀^2) / Nq)) "n̲·Q̂·n̲ ≠ (ac-b²)/N"

println("  a = ", a₀, ",  b = ", b₀, ",  c = ", c₀)                        #jl
println("  a·N - (b+a ξₙ)² - (ac-b²) = 0   (identically, 6 free symbols)")  #jl
println("  ⇒ n̲·Q̂·n̲ = (ac-b²)/N : a constant over a quadratic")            #jl
println()                                                                  #jl

# ## §3 The ``\xi_n`` integral — elementary, for every anisotropy
#
# Completing the square, ``N = a\bigl[(\xi_n+b/a)^{2}+p^{2}\bigr]`` with
# ``p = \sqrt{ac-b^{2}}/a``, and ``p > 0`` **because** ``\boldsymbol{K}_0`` is
# positive definite: ``ac-b^{2}`` is the Gram determinant of
# ``\boldsymbol{K}_0`` restricted to
# ``\mathrm{span}\{\underline{n},\underline{\xi}^{\star}\}``. Making that
# substitution explicit is not cosmetic — handed the raw quadratic, `integrate`
# cannot decide the sign of the discriminant and silently answers `0`.
#
# One elementary integral, ``\int\mathrm{d}u/(u^{2}+p^{2}) = \pi/p``, then gives
#
# ```math
# \boxed{\;
# \hat{Q}^{\star}_{nn}(\underline{\xi}^{\star}) = \tfrac12\sqrt{ac-b^{2}}
# \;}
# ```

println("="^78)                                                       #jl
println("  § 3  THE ξₙ INTEGRAL — ELEMENTARY AT ANY ANISOTROPY")      #jl
println("="^78)                                                       #jl

@syms u::real p::positive

## `p` is carried as a positive symbol so the integral is unconditional; the
## substitution itself is verified against `N` before being used.
@assert iszero(
    tsimplify(a₀ * ((ξₙ + b₀ / a₀)^2 + (a₀ * c₀ - b₀^2) / a₀^2) - Nq)
) "completing the square does not reproduce N"

Q★ = tsimplify(
    integrate((a₀ * c₀ - b₀^2) / (a₀ * (u^2 + p^2)), (u, -OO, OO)) / (2 * PI)
)
Q★ = tsimplify(subs(Q★, p => sqrt(a₀ * c₀ - b₀^2) / a₀))

@assert iszero(tsimplify(Q★ - sqrt(a₀ * c₀ - b₀^2) / 2)) "Q̂*ₙₙ ≠ √(ac-b²)/2"

println("  Q̂*ₙₙ = ", Q★)   #jl

# ### The adjugate form
#
# The Gram determinant is a quadratic form on the *normal* of the plane
# ``\{\underline{n},\underline{\xi}^{\star}\}``: for symmetric
# ``\boldsymbol{K}`` and any ``\underline{u},\underline{v}``,
#
# ```math
# (\underline{u}\cdot\boldsymbol{K}\underline{u})
# (\underline{v}\cdot\boldsymbol{K}\underline{v})
# - (\underline{u}\cdot\boldsymbol{K}\underline{v})^{2}
# = (\underline{u}\wedge\underline{v})\cdot\mathrm{adj}\,\boldsymbol{K}\cdot
#   (\underline{u}\wedge\underline{v}),
# \qquad
# \mathrm{adj}\,\boldsymbol{K} = \det\boldsymbol{K}\;\boldsymbol{K}^{-1}.
# ```
#
# So ``\hat{Q}^{\star}_{nn}`` is a norm of ``\underline{n}\wedge\underline{\xi}^{\star}``
# in the metric ``\mathrm{adj}\,\boldsymbol{K}_0`` — manifestly homogeneous of
# degree 1, as it must be.

adjK = Sym[
    K₂₂ * K₃₃-K₂₃^2 K₁₃ * K₂₃-K₁₂ * K₃₃ K₁₂ * K₂₃-K₁₃ * K₂₂
    K₁₃ * K₂₃-K₁₂ * K₃₃ K₁₁ * K₃₃-K₁₃^2 K₁₂ * K₁₃-K₁₁ * K₂₃
    K₁₂ * K₂₃-K₁₃ * K₂₂ K₁₂ * K₁₃-K₁₁ * K₂₃ K₁₁ * K₂₂-K₁₂^2
]
@assert all(iszero, tsimplify.(adjK - det(Karr) * inv(Karr))) "adj K ≠ det(K)·K⁻¹"

"Componentwise `subs` on a matrix — `subs.(M, d)` is not allowed, `d` being a `Dict`."
subs_all(M, d) = [tsimplify(subs(M[i, j], d)) for i in axes(M, 1), j in axes(M, 2)]

let w = Sym[Sym(0), ρ, Sym(0)]            # n̲ ∧ ξ̲★ = e₃ ∧ ρe₁ = ρ e₂
    @assert iszero(tsimplify(dot(w, adjK * w) - (a₀ * c₀ - b₀^2))) "the adjugate identity fails"
    println("  = ½√[(n̲∧ξ̲★)·adj K₀·(n̲∧ξ̲★)]   (adjugate identity verified)")   #jl
    println()                                                                   #jl
end

# ## §4 The crack-plane integral — an effective ellipse
#
# On the contour ``\underline{\xi}^{\star}(\varphi) =
# \eta\cos\varphi\,\underline{\ell} + \sin\varphi\,\underline{m}``,
# ``\underline{n}\wedge\underline{\xi}^{\star}
# = \eta\cos\varphi\,\underline{m} - \sin\varphi\,\underline{\ell}``, so with
# ``\underline{v} = (\cos\varphi,\sin\varphi)``
#
# ```math
# \hat{Q}^{\star}_{nn} = \tfrac12\sqrt{\underline{v}\cdot\boldsymbol{Q}_2\cdot\underline{v}},
# \qquad
# \boldsymbol{Q}_2 = \begin{pmatrix}
#   \eta^{2}\,\underline{m}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{m} &
#  -\eta\,\underline{m}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{\ell} \\
#  -\eta\,\underline{\ell}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{m} &
#        \underline{\ell}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{\ell}
# \end{pmatrix}.
# ```
#
# Diagonalizing the **2×2** ``\boldsymbol{Q}_2`` — a quadratic, so closed form,
# where the elastic problem faced a sextic — and rotating to its principal axes
# turns the integral into the isotropic one:
#
# ```math
# b\Lambda = \frac14\!\int_0^{2\pi}\!\hat{Q}^{\star}_{nn}\,\mathrm{d}\varphi
# = \frac{\sqrt{\lambda_1}}{2}\,\mathcal{E}_{\eta'},
# \qquad
# \eta' = \sqrt{\lambda_2/\lambda_1},
# \qquad
# b = \frac{\chi^{\mathcal{E}}}{b\Lambda} = \frac{4}{3\sqrt{\lambda_1}\,\mathcal{E}_{\eta'}} .
# ```
#
# An arbitrarily anisotropic conductor therefore behaves as an isotropic one of
# conductivity ``\sqrt{\lambda_1}`` around a crack of *effective* aspect ratio
# ``\eta'``. In general ``\eta'\ne\eta``: even a circular crack acquires an
# effective ellipticity from the anisotropy.

println("="^78)                                                       #jl
println("  § 4  CRACK-PLANE INTEGRAL — THE EFFECTIVE ELLIPSE")        #jl
println("="^78)                                                       #jl

@syms η::positive

"""
    inplane_Q2(adjK, η) -> (T, D)

Trace and determinant of the 2×2 in-plane restriction of `adjK` scaled by the
aspect ratio, in the crack frame `(ℓ, m, n) = (e₁, e₂, e₃)`.
"""
function inplane_Q2(adjK, η)
    A = adjK[2, 2]           # m̲·adjK·m̲
    B = adjK[2, 1]           # m̲·adjK·ℓ̲
    C = adjK[1, 1]           # ℓ̲·adjK·ℓ̲
    Q2 = Sym[A*η^2 -B*η; -B*η C]
    return tsimplify(Q2[1, 1] + Q2[2, 2]), tsimplify(det(Q2))
end

"Eigenvalues `λ₁ ≥ λ₂` of a 2×2 symmetric form from its trace and determinant."
eig2(T, D) = ((T + sqrt(T^2 - 4 * D)) / 2, (T - sqrt(T^2 - 4 * D)) / 2)

"Thermal COD scalar in the package convention: `b = χᴱ / (bΛ)`, `χᴱ = 2/3`."
cod_thermal(λ₁, η′) = Sym(2) / 3 / (sqrt(λ₁) / 2 * ell_E(1 - η′^2))

# The eigenvalues are closed form, but for a *free* symbolic ``\eta`` SymPy
# cannot order them: it holds ``\sqrt{(1-\eta^{2})^{2}}`` unevaluated, because
# nothing in the assumption system says ``\eta \le 1``. The general formula is
# therefore checked on exact rational data, against a direct quadrature of the
# contour integral; the symmetry classes of § 5 are then done symbolically, where
# ``\boldsymbol{Q}_2`` is diagonal and no ordering is needed.

let Kv = Sym[4 1 1; 1 5 2; 1 2 6],
        adjKv = subs_all(
        adjK,
        Dict(
            K₁₁ => Kv[1, 1], K₂₂ => Kv[2, 2], K₃₃ => Kv[3, 3],
            K₁₂ => Kv[1, 2], K₁₃ => Kv[1, 3], K₂₃ => Kv[2, 3],
        ),
    )
    adjF = [Float64(N(e)) for e in adjKv]
    for ηv in (1.0, 0.6, 0.25)
        T, D = inplane_Q2(adjKv, Sym(ηv))
        λ₁, λ₂ = eig2(Float64(N(T)), Float64(N(D)))
        b_closed = (2 / 3) / (sqrt(λ₁) / 2 * ell_E(1 - λ₂ / λ₁))
        ## Direct quadrature of  bΛ = ¼∫₀²ᵖ Q̂*ₙₙ dφ, no eigen-decomposition.
        bΛ = quadgk(0.0, 2π; rtol = 1.0e-12) do φ
            w = [-sin(φ), ηv * cos(φ), 0.0]          # n̲ ∧ ξ̲★(φ)
            sqrt(dot(w, adjF * w)) / 2
        end[1] / 4
        b_quad = (2 / 3) / bΛ
        @printf "  η = %.2f : η′ = %.4f, √λ₁ = %.4f | b closed %.10f  vs quadrature %.10f\n" ηv sqrt(λ₂ / λ₁) sqrt(λ₁) b_closed b_quad   #jl
        @assert abs(b_closed - b_quad) < 1.0e-9 "the effective-ellipse reduction fails at η = $ηv"
    end
    println()   #jl
end

# ## §5 Specializations
#
# Every case below is the *same* two formulas of § 3 and § 4, evaluated on a more
# symmetric ``\boldsymbol{K}_0``. In each, the off-diagonal
# ``\underline{m}\cdot\mathrm{adj}\boldsymbol{K}_0\cdot\underline{\ell}``
# vanishes, so the principal axes of ``\boldsymbol{Q}_2`` *are* the crack axes
# and the eigenvalues can be read off: ``\lambda_2 = \eta^{2}A``,
# ``\lambda_1 = C``, ordered because ``\eta \le 1``.

println("="^78)                                     #jl
println("  § 5  SPECIALIZATIONS")                   #jl
println("="^78)                                     #jl

"""
    diagonal_lambdas(adjK, η) -> (λ₁, λ₂)

Eigenvalues of `Q₂` when the in-plane off-diagonal vanishes — asserted here
rather than assumed, so the shortcut cannot be applied to a case it does not fit.
"""
function diagonal_lambdas(adjK, η)
    @assert iszero(tsimplify(adjK[2, 1])) "m̲·adjK·ℓ̲ ≠ 0: Q₂ is not diagonal"
    return tsimplify(adjK[1, 1]), tsimplify(adjK[2, 2] * η^2)
end

# ### Isotropic conductor
#
# ``\mathrm{adj}(k_0\boldsymbol{1}) = k_0^{2}\boldsymbol{1}``, so
# ``\boldsymbol{Q}_2 = k_0^{2}\,\mathrm{diag}(\eta^{2},1)``, giving
# ``\lambda_1 = k_0^{2}``, ``\eta' = \eta`` and
# ``b = 4/(3k_0\mathcal{E}_\eta)``.

@syms k₀::positive

iso_subs = Dict(
    K₁₁ => k₀, K₂₂ => k₀, K₃₃ => k₀, K₁₂ => Sym(0), K₁₃ => Sym(0), K₂₃ => Sym(0)
)

b_iso = let (λ₁, λ₂) = diagonal_lambdas(subs_all(adjK, iso_subs), η)
    @assert iszero(tsimplify(λ₁ - k₀^2)) "λ₁ ≠ k₀² in the isotropic case"
    @assert iszero(tsimplify(λ₂ - k₀^2 * η^2)) "λ₂ ≠ k₀²η² in the isotropic case"
    println("  isotropic  : λ₁ = ", λ₁, ",  η′ = ", tsimplify(sqrt(λ₂ / λ₁)))   #jl
    tsimplify(cod_thermal(λ₁, sqrt(λ₂ / λ₁)))
end

@assert iszero(tsimplify(b_iso - 4 / (3 * k₀ * ell_E(1 - η^2)))) "b_iso ≠ 4/(3k₀ℰη)"
println("  b_iso = ", b_iso)   #jl

# ### Transversely isotropic conductor aligned with the crack normal
#
# ``\boldsymbol{K}_0 = \mathrm{diag}(k_t,k_t,k_n)`` in the crack frame gives
# ``\mathrm{adj}\boldsymbol{K}_0 = \mathrm{diag}(k_tk_n,k_tk_n,k_t^{2})``, hence
# ``\lambda_1 = k_tk_n`` and ``\eta' = \eta``: the effective conductivity is the
# **geometric mean** ``\sqrt{k_tk_n}``, and the aspect ratio is untouched.

@syms k_t::positive k_n::positive

ti_subs = Dict(
    K₁₁ => k_t, K₂₂ => k_t, K₃₃ => k_n, K₁₂ => Sym(0), K₁₃ => Sym(0), K₂₃ => Sym(0)
)

b_ti = let (λ₁, λ₂) = diagonal_lambdas(subs_all(adjK, ti_subs), η)
    @assert iszero(tsimplify(λ₁ - k_t * k_n)) "λ₁ ≠ k_t k_n for an aligned TI conductor"
    @assert iszero(tsimplify(sqrt(λ₂ / λ₁) - η)) "η′ ≠ η for an aligned TI conductor"
    println("  aligned TI : λ₁ = ", λ₁, "  → effective conductivity √(k_t k_n)")   #jl
    tsimplify(cod_thermal(λ₁, η))
end

@assert iszero(tsimplify(subs(b_ti, Dict(k_t => k₀, k_n => k₀)) - b_iso)) "TI does not reduce to iso"
println("  b_TI  = ", b_ti)   #jl

# ### Ribbon crack
#
# The contour integral collapses to the single direction
# ``\underline{\xi}^{\star} = \underline{m}``, and
# ``\chi^{\mathcal{R}} = \pi/4`` replaces ``\chi^{\mathcal{E}} = 2/3``:
#
# ```math
# b^{\mathcal{R}} = \frac{\pi/4}{\hat{Q}^{\star}_{nn}(\underline{m})}
# = \frac{\pi}{2\sqrt{\det\bigl(\boldsymbol{K}_0|_{(\underline{m},\underline{n})}\bigr)}} ,
# ```
#
# so only the 2×2 transverse block of ``\boldsymbol{K}_0`` enters — the
# structure the shipped ribbon formula already has.

b_rib = let 𝛏m = 𝐞₂,
        am = tsimplify(𝐧 ⋅ 𝐊 ⋅ 𝐧),
        bm = tsimplify(𝐧 ⋅ 𝐊 ⋅ 𝛏m),
        cm = tsimplify(𝛏m ⋅ 𝐊 ⋅ 𝛏m)
    Qm = sqrt(am * cm - bm^2) / 2
    det_block = K₂₂ * K₃₃ - K₂₃^2
    @assert iszero(tsimplify(am * cm - bm^2 - det_block)) "the (m̲,n̲) block determinant is wrong"
    println("  ribbon : Q̂*ₙₙ(m̲) = ½√det(K₀|₍m,n₎) = ", tsimplify(Qm))   #jl
    tsimplify(PI / 4 / Qm)
end

println("  b_ribbon = ", b_rib)   #jl
println()                         #jl

# ## §6 Independent checks
#
# Three of them, none sharing any code with §§ 1–5.

println("="^78)                                  #jl
println("  § 6  INDEPENDENT CHECKS")             #jl
println("="^78)                                  #jl

# ### The textbook penny temperature jump
#
# An insulating circular crack of radius ``a`` under a remote normal flux opens
# a temperature jump ``[\![T]\!](r) = \frac{4\sigma_n a}{\pi k_0}
# \sqrt{1-r^{2}/a^{2}}``, whose average over the disc is
# ``\chi^{\mathcal{E}} = 2/3`` of the maximum. With the package normalization
# ``\langle[\![T]\!]\rangle = b\,\underline{b}\,\sigma_n`` and
# ``\underline{b} = a`` for a penny, that gives ``b = 8/(3\pi k_0)``.

let b_penny_text = 8 / (3 * PI * k₀),
        b_penny_here = tsimplify(subs(b_iso, η => Sym(1)))
    @assert iszero(tsimplify(b_penny_here - b_penny_text)) "penny b ≠ 8/(3πk₀)"
    println("  penny (η=1) : b = ", b_penny_here, "  = 8/(3πk₀), the textbook jump")   #jl
end

# ### The Hill-tensor flattening oracle
#
# Independent of everything above: build the order-2 Hill tensor of a flat
# ellipsoid, assemble ``\boldsymbol{\Lambda} = \boldsymbol{K}_0 -
# \boldsymbol{K}_0\boldsymbol{P}\boldsymbol{K}_0``, and take
# ``\boldsymbol{R} = \lim_{\omega\to0}\omega\,\boldsymbol{\Lambda}^{-1}`` with
# ``\omega = c/\underline{b}``. The limit must be
# ``\tfrac34\,b\,\underline{n}\otimes\underline{n}``.

"`R₃₃` from the flattening route, at finite flatness `ω = c/b`."
function R_from_hill(a, bmin, ω, k)
    P = hill_tensor(Ellipsoid(a, bmin, ω * bmin), TensISO{3}(k))
    Λ = TensISO{3}(k) - TensISO{3}(k) ⋅ P ⋅ TensISO{3}(k)
    return ω * inv([Λ[i, j] for i in 1:3, j in 1:3])[3, 3]
end

for ηv in (1.0, 0.5)
    b_here = Float64(N(subs(b_iso, Dict(k₀ => Sym(1), η => Sym(ηv)))))
    target = 0.75 * b_here
    vals = [R_from_hill(1.0, ηv, ω, 1.0) for ω in (1.0e-2, 1.0e-3)]
    @printf "  η = %.2f : (3/4)·b = %.6f   vs  ω-oracle %.6f → %.6f\n" ηv target vals[1] vals[2]   #jl
    @assert abs(vals[2] - target) / target < 5.0e-3 "the flattening oracle disagrees at η = $ηv"
end

# ### The shipped closed forms
#
# Since v0.4.0 the package evaluates exactly the formulas derived above, through
# the same adjugate route — so this is a genuine end-to-end check of
# [`cod_tensor`](@ref) against the symbolic chain, isotropic, aligned-TI and
# fully anisotropic alike. (Up to v0.3.2 the shipped values were smaller by
# ``4\pi/(3\eta)`` and ``\pi^{2}/4``; writing this derivation down is what
# surfaced that.)

let vals = Dict(k₀ => Sym(1))
    for ηv in (1.0, 0.6, 0.3)
        b_here = Float64(N(subs(subs(b_iso, vals), η => Sym(ηv))))
        crack = ηv == 1.0 ? PennyCrack(1.0) : EllipticCrack(1.0, ηv)
        b_code = cod_tensor(crack, TensISO{3}(1.0))
        @printf "  iso    η = %.2f : symbolic %.10f  vs cod_tensor %.10f\n" ηv b_here b_code   #jl
        @assert abs(b_here - b_code) < 1.0e-12 "iso b disagrees with cod_tensor at η = $ηv"
    end

    ## Aligned TI — the geometric-mean effective conductivity.
    b_ti_num = Float64(N(subs(b_ti, Dict(k_t => Sym(1), k_n => Sym(4), η => Sym(1)))))
    b_ti_code = cod_tensor(PennyCrack(1.0), Tens(Matrix(Diagonal([1.0, 1.0, 4.0]))))
    @printf "  TI     penny   : symbolic %.10f  vs cod_tensor %.10f\n" b_ti_num b_ti_code   #jl
    @assert abs(b_ti_num - b_ti_code) < 1.0e-12 "aligned-TI b disagrees with cod_tensor"

    ## Ribbon, anisotropic transverse block.
    b_rib_num = Float64(
        N(subs(b_rib, Dict(K₂₂ => Sym(4), K₃₃ => Sym(3) / 2, K₂₃ => Sym(0))))
    )
    b_rib_code = cod_tensor(RibbonCrack(0.8), Tens(Matrix(Diagonal([3.0, 4.0, 1.5]))))
    @printf "  ribbon aniso   : symbolic %.10f  vs cod_tensor %.10f\n" b_rib_num b_rib_code   #jl
    @assert abs(b_rib_num - b_rib_code) < 1.0e-12 "ribbon b disagrees with cod_tensor"
end

println()                                                                      #jl
println("="^78)                                                                #jl
println("  Transport chain verified symbolically at FULL anisotropy:")          #jl
println("  Green operator → Q̂*ₙₙ = ½√[(n̲∧ξ̲★)·adjK₀·(n̲∧ξ̲★)] → effective ellipse → b")   #jl
println("="^78)                                                                #jl
