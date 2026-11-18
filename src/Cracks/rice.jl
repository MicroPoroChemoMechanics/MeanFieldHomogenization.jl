# =============================================================================
#  rice.jl — contribution of a crack from its intensity factors integrated
#  along a growth path (Rice's internal variables).
#
#  A crack grown at fixed remote stress 𝚺 from a nucleus to its current shape
#  releases, front point by front point, the energy release rate
#  G = (1/π) 𝐊·𝐁ᴿ(ν, n)·𝐊 (SifAniso form, 𝐁ᴿ the COD tensor of the tangent
#  ribbon); the total is the complementary energy ½ 𝚺:V Δ𝕊:𝚺 of the crack.
#  With 𝐊 = 𝐌·(𝚺·n):
#
#     V Δ𝕊 = (2/π) ∫ dλ ∫_L n ⊗ˢ (𝐌ᵀ·𝐁ᴿ·𝐌) ⊗ˢ n dl .
#
#  Derivation: §1–2 of docs/src/theory/rice_crack_contributions.md.
# =============================================================================

# Size exponent of the density of a shape family: ε = N ℓ³ for a crack bounded in
# its plane, ε = N ℓ² for a crack unbounded along one direction.
_rice_size_power(::Type{<:CrackShape}) = 3
_rice_size_power(::Type{Ribbon}) = 2
_rice_size_power(S::Type) = _unsupported_crack_shape(S, "growth-path")

# (2/π) / (density factor ⋅ ℓᵈ): turns ∫∫ (…) dl dλ into the size-independent
# contribution of the family, for which Δ𝕊 = density factor ⋅ ε ⋅ ℍ.
_rice_prefactor(shape::Type, size, ::Type{T}) where {T} =
    2 / (T(π) * _crack_density_factor(shape, T) * size^_rice_size_power(shape))

# ∫_{λ₀}^{λ₁} f(λ) dλ over u ∈ [0, 1], λ = λ₀ + (λ₁ − λ₀) u: the limits a
# derivative may flow through (a dual radius, a dual angle) enter the integrand,
# and QuadGK builds its rule on plain floats.
function _growth_integral(f, λ₀, λ₁, reltol)
    # Adaptive quadrature: no symbolic counterpart.
    is_hard_numeric(promote_type(typeof(λ₀), typeof(λ₁))) || throw(
        ArgumentError(
            "the growth integral of Rice's formula is evaluated by adaptive quadrature: " *
                "its limits must be numbers, got $(typeof(λ₀)) and $(typeof(λ₁))"
        )
    )
    Δ = λ₁ - λ₀
    R = TensND.primal_type(promote_type(typeof(λ₀), typeof(λ₁)))
    return quadgk(u -> Δ * f(λ₀ + Δ * u), zero(R), one(R); rtol = reltol)[1]
end

# The frame (τ, ν, n) of the ribbon tangent to the front at a point, as a TensND
# basis, from the components `ν` and `n` read in the orthonormal basis `ℬ`: the
# columns of the frame are P⋅τ, P⋅ν, P⋅n with P the vectors of ℬ.
function _tangent_ribbon_basis(ν, n, ℬ::TensND.AbstractBasis{3, Tb}) where {Tb}
    T = promote_type(eltype(ν), eltype(n), Tb)
    P = SMatrix{3, 3, T}(TensND.vecbasis(ℬ, Val(:cov)))
    F = P * hcat(n × ν, ν, n)
    # orthonormal by construction; `Val(:typed)` keeps the basis type independent
    # of its value, hence the computation inferable
    return TensND.RotatedBasis(F, Val(3), Val(:typed))
end

# 𝐁ᴿ(ν, n) of the matrix `C₀`, as a static matrix of components in `ℬ`.
function _tangent_ribbon_cod(ν, n, ℬ, C₀::TensND.AbstractTens{4, 3}; reltol = 1.0e-6)
    T = promote_type(eltype(ν), eltype(n))
    ribbon = RibbonCrack(one(T), _tangent_ribbon_basis(ν, n, ℬ))
    return SMatrix{3, 3}(TensND.components(cod_tensor(ribbon, C₀; reltol), ℬ))
end

# The conduction twin: the scalar bᴿ of the tangent ribbon.
function _tangent_ribbon_cod(ν, n, ℬ, K₀::TensND.AbstractTens{2, 3}; reltol = 1.0e-6)
    T = promote_type(eltype(ν), eltype(n))
    return cod_tensor(RibbonCrack(one(T), _tangent_ribbon_basis(ν, n, ℬ)), K₀)
end

"""
    rice_cod_tensor(front, λ₀, λ₁, C₀, shape=Penny; size,
                    basis=CanonicalBasis{3,Float64}(), reltol=1e-8) -> Tens{2,3}
    rice_cod_tensor(front, λ₀, λ₁, K₀, shape=Penny; size, basis, reltol) -> Real

COD tensor ``\\boldsymbol{B}`` (elasticity, `C₀` of order 4) or thermal COD scalar
``b`` (conduction, `K₀` of order 2) of a **flat** crack, obtained by integrating
its intensity factors along a growth path.

The crack grows with the parameter ``\\lambda`` from ``\\lambda_0`` (a nucleus, or
any shape whose contribution vanishes) to ``\\lambda_1`` (its current shape).
`front(λ)` returns the points of the front at stage ``\\lambda``, each as a tuple
`(w, ν, M)`:

- `w ≥ 0`: the weight of the point, such that ``\\sum w\\,f \\approx
  \\int_L f\\,v_\\nu\\,\\mathrm{d}l`` with ``v_\\nu`` the advance of the front per unit
  ``\\lambda`` (the area swept per unit ``\\lambda`` is ``\\sum w``);
- `ν`: the outer unit normal to the front, in the crack plane;
- `M`: in elasticity, the ``3\\times 3`` matrix ``\\boldsymbol{M}`` such that the
  stress intensity factor vector is ``\\underline{K} = \\boldsymbol{M}\\cdot
  (\\boldsymbol{\\Sigma}\\cdot\\underline{n})``; in conduction, the scalar ``m``
  such that the flux intensity factor is ``K_T = m\\,(\\underline{n}\\cdot
  \\boldsymbol{\\sigma}^\\infty)``.

Vectors and matrices are components in the orthonormal `basis`, whose third vector
is the crack normal ``\\underline{n}``. With the energy release rate
``G = \\underline{K}\\cdot\\boldsymbol{B}^{\\mathcal R}(\\underline{\\nu},
\\underline{n})\\cdot\\underline{K}/\\pi``, ``\\boldsymbol{B}^{\\mathcal R}`` the COD
tensor of the ribbon tangent to the front [barthelemyIJSS2026](@cite), Rice's
formula reads [rice1975, kachanov2012](@cite)

```math
\\boldsymbol{B} = \\frac{2}{\\pi\\,f\\,k\\,\\ell^{d}}
\\int_{\\lambda_0}^{\\lambda_1}\\!\\int_L
\\boldsymbol{M}^{\\!T}\\cdot\\boldsymbol{B}^{\\mathcal R}\\cdot\\boldsymbol{M}
\\,v_\\nu\\,\\mathrm{d}l\\,\\mathrm{d}\\lambda ,
```

where ``\\ell`` = `size` is the length that defines the density of the family
(``\\varepsilon = N\\ell^3``, ``d = 3``, for `shape = Penny`; ``\\varepsilon =
N\\ell^2``, ``d = 2``, for `shape = Ribbon`), ``f`` = [`crack_density_factor`](@ref)
and ``k`` the factor of [`compliance_from_cod`](@ref). The result is the
``\\boldsymbol{B}`` of [`cod_tensor`](@ref): ``\\mathbb H = k\\,\\underline{n}
\\stackrel{s}{\\otimes}\\boldsymbol{B}\\stackrel{s}{\\otimes}\\underline{n}``.

The matrix enters only through ``\\boldsymbol{B}^{\\mathcal R}``, computed by
[`cod_tensor`](@ref) for any anisotropy; the intensity factors are the user's.
The ``\\lambda`` integral is adaptive (QuadGK), and `reltol` is its relative
tolerance as well as that of the numerical ``\\boldsymbol{B}^{\\mathcal R}`` of an
anisotropic matrix.

See [Contributions from intensity factors](@ref th-rice-contributions).
"""
function rice_cod_tensor(
        front, λ₀, λ₁, C₀::TensND.AbstractTens{4, 3}, shape::Type{S} = Penny;
        size, basis::TensND.AbstractBasis{3} = TensND.CanonicalBasis{3, Float64}(),
        reltol = 1.0e-8
    ) where {S}
    n = SVector(0, 0, 1)
    integrand(λ) = sum(
        w * (transpose(SMatrix{3, 3}(M)) * _tangent_ribbon_cod(SVector{3}(ν), n, basis, C₀; reltol) * SMatrix{3, 3}(M))
            for (w, ν, M) in front(λ)
    )
    I = _growth_integral(integrand, λ₀, λ₁, reltol)
    T = eltype(I)
    B = _rice_prefactor(shape, size, T) / _H_over_B_factor(shape, T) .* (I .+ transpose(I)) ./ 2
    return TensND.Tens(TensND.Tensors.SymmetricTensor{2, 3}(B), basis)
end

function rice_cod_tensor(
        front, λ₀, λ₁, K₀::TensND.AbstractTens{2, 3}, shape::Type{S} = Penny;
        size, basis::TensND.AbstractBasis{3} = TensND.CanonicalBasis{3, Float64}(),
        reltol = 1.0e-8
    ) where {S}
    n = SVector(0, 0, 1)
    integrand(λ) = sum(w * _tangent_ribbon_cod(SVector{3}(ν), n, basis, K₀) * m^2 for (w, ν, m) in front(λ))
    I = _growth_integral(integrand, λ₀, λ₁, reltol)
    T = typeof(I)
    return _rice_prefactor(shape, size, T) / _H_over_B_factor(shape, T) * I
end

"""
    _rice_compliance(front, λ₀, λ₁, C₀, shape; size, basis, reltol) -> Tens{4,3}
    _rice_compliance(front, λ₀, λ₁, K₀, shape; size, basis, reltol) -> Tens{2,3}

Rice's formula for a crack that is not flat. The intensity factors at a point of
a curved crack depend on the whole remote stress, not only on the traction on
the plane tangent at the front, so each point of `front(λ)` carries the
third-order influence array `𝒦`, ``K_q = \\mathcal K_{qij}\\Sigma_{ij}`` (symmetric
in ``ij``), with its normal: `(w, ν, n, 𝒦)`. In conduction it carries the vector
`k`, ``K_T = \\underline{k}\\cdot\\boldsymbol{\\sigma}^\\infty``: `(w, ν, n, k)`.
The contribution is assembled with no COD tensor in between:

```math
\\mathbb H = \\frac{2}{\\pi f \\ell^d}\\int\\!\\!\\int
\\mathcal K^{\\!T}\\cdot\\boldsymbol{B}^{\\mathcal R}(\\underline{\\nu},
\\underline{n})\\cdot\\mathcal K\\,v_\\nu\\,\\mathrm{d}l\\,\\mathrm{d}\\lambda ,
\\qquad
\\boldsymbol{R} = \\frac{2}{\\pi f \\ell^d}\\int\\!\\!\\int
b^{\\mathcal R}\\,\\underline{k}\\otimes\\underline{k}\\,v_\\nu\\,\\mathrm{d}l\\,\\mathrm{d}\\lambda .
```

Used by `method = :rice` of [`CircularArcCrack`](@ref).
"""
function _rice_compliance(
        front, λ₀, λ₁, C₀::TensND.AbstractTens{4, 3}, shape::Type{S};
        size, basis::TensND.AbstractBasis{3}, reltol = 1.0e-8
    ) where {S}
    integrand(λ) = sum(
        w * _contract_KBK(SArray{Tuple{3, 3, 3}}(𝒦), _tangent_ribbon_cod(SVector{3}(ν), SVector{3}(n), basis, C₀; reltol))
            for (w, ν, n, 𝒦) in front(λ)
    )
    I = _growth_integral(integrand, λ₀, λ₁, reltol)
    T = eltype(I)
    H = _rice_prefactor(shape, size, T) .* I
    return TensND.Tens(TensND.Tensors.SymmetricTensor{4, 3, T}((i, j, k, l) -> H[i, j, k, l]), basis)
end

function _rice_compliance(
        front, λ₀, λ₁, K₀::TensND.AbstractTens{2, 3}, shape::Type{S};
        size, basis::TensND.AbstractBasis{3}, reltol = 1.0e-8
    ) where {S}
    integrand(λ) = sum(
        w * _tangent_ribbon_cod(SVector{3}(ν), SVector{3}(n), basis, K₀) * (SVector{3}(k) * transpose(SVector{3}(k)))
            for (w, ν, n, k) in front(λ)
    )
    I = _growth_integral(integrand, λ₀, λ₁, reltol)
    T = eltype(I)
    R = _rice_prefactor(shape, size, T) .* (I .+ transpose(I)) ./ 2
    return TensND.Tens(TensND.Tensors.SymmetricTensor{2, 3}(R), basis)
end

# (𝒦ᵀ⋅𝐁⋅𝒦)ᵢⱼₖₗ = Σ_qr 𝒦_qij B_qr 𝒦_rkl, with B symmetrized.
function _contract_KBK(𝒦::SArray{Tuple{3, 3, 3}}, B::SMatrix{3, 3})
    Bs = (B + transpose(B)) / 2
    T = promote_type(eltype(𝒦), eltype(B))
    return SArray{Tuple{3, 3, 3, 3}, T}(
        ntuple(Val(81)) do idx
            i, j, k, l = Tuple(CartesianIndices((3, 3, 3, 3))[idx])
            sum(𝒦[q, i, j] * Bs[q, r] * 𝒦[r, k, l] for q in 1:3, r in 1:3)
        end
    )
end
