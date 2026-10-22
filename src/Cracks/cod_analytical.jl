# =============================================================================
#  cod_analytical.jl — closed-form COD tensors.
#  Moduli extractors live in Core.moduli.jl; this file only contains the
#  COD formulas.
# =============================================================================

"""
    _elliptic_CS(η) -> (𝒞, 𝒮, ℰ)

Angular integrals (paper eq. 1532):
``\\mathcal{C}_\\eta = \\int_0^{\\pi/2}\\cos^2\\!\\vartheta\\,/
\\sqrt{\\cos^2\\!\\vartheta+\\eta^2\\sin^2\\!\\vartheta}\\;\\mathrm{d}\\vartheta``,
likewise ``\\mathcal{S}_\\eta`` with ``\\sin^2\\!\\vartheta``, and
``\\mathcal{E}_\\eta`` the complete integral of the second kind.

The circular limit ``\\eta = 1`` is a removable ``0/0``. For a number type
the closed forms ``\\mathcal{S} = (\\mathcal{K}-\\mathcal{E})/k^2`` and
``\\mathcal{C} = \\mathcal{K}-\\mathcal{S}`` cancel as ``k^2 \\to 0`` (a relative
error near ``\\epsilon/k^2``), so below ``k^2 = 0.1`` they are replaced by the
series of the complete integrals,

```math
\\mathcal{K} = \\frac{\\pi}{2}\\sum_{n\\ge0} a_n k^{2n},\\quad
\\mathcal{S} = \\frac{\\pi}{2}\\sum_{n\\ge1}\\frac{2n}{2n-1}\\,a_n k^{2n-2},\\quad
a_n = \\Big(\\frac{(1/2)_n}{n!}\\Big)^2,
```

with ``\\mathcal{E} = \\mathcal{K} - k^2\\mathcal{S}``: exact at the circle, and
differentiable through it, where a dual ``k^2`` of value zero carries the
derivative of an elliptical crack's gap. The guard is deliberately **not**
restricted to `T <: Real`: on a symbolic element type `iszero(k²)` is `false`
for a free symbol (which is what we want — the elliptic branch is the general
answer) and `true` for an exact `Sym(1)`, where the general branch would
otherwise return `NaN`. Restricting it to `Real` made
`cod_tensor(PennyCrack(one(Sym)), …)` silently produce `NaN`.
"""
@inline function _elliptic_CS(η::T) where {T <: Number}
    η² = η^2
    k² = one(T) - η²
    if is_hard_numeric(T) && abs(k²) ≤ 0.1
        return _elliptic_CS_series(k²)
    elseif !is_hard_numeric(T) && isequal(k², zero(k²))
        q = T(π) / T(4)
        return q, q, T(π) / T(2)
    else
        𝒦 = ell_K(k²)
        ℰ = ell_E(k²)
        𝒞 = (ℰ - η² * 𝒦) / k²
        𝒮 = (𝒦 - ℰ) / k²
        return 𝒞, 𝒮, ℰ
    end
end

# The aspect ratio in the arithmetic of the moduli when that arithmetic is
# symbolic, so that the factors of the circle stay exact (π/4, π/2) instead of
# entering a symbolic COD tensor as Float64 numbers; a numerical ratio is kept as
# it is, the Float64 elliptic integrals being the fast path.
_ratio_in(::Type{T}, η) where {T} = _ratio_in(Val(is_hard_numeric(T)), T, η)
_ratio_in(::Val{true}, ::Type, η) = η
_ratio_in(::Val{false}, ::Type{T}, η) where {T} = T(η)
# The circle given as `1.0` enters as the integer it is, so that its closed form
# is exact; any other float stays the float it was given as.
_ratio_in(::Val{false}, ::Type{T}, η::AbstractFloat) where {T} = isinteger(η) ? T(Int(η)) : T(η)

# 𝒞, 𝒮, ℰ near the circle, |k²| ≤ 0.1: 40 terms leave a truncation below
# 0.1⁴⁰. The coefficients aₙ = ((½)ₙ/n!)² follow from aₙ = aₙ₋₁ ((2n − 1)/(2n))².
@inline function _elliptic_CS_series(k²)
    F = typeof(float(MFH_Core.primal(k²)))
    a = one(F)
    𝒦 = one(k²)
    𝒮 = zero(k²)
    kp = one(k²)                                  # k^(2n−2)
    for n in 1:40
        a = a * ((2n - 1) / (2n))^2
        𝒮 += (2n / (2n - 1)) * a * kp
        kp *= k²
        𝒦 += a * kp
    end
    h = F(π) / 2
    return h * (𝒦 - 𝒮), h * 𝒮, h * (𝒦 - k² * 𝒮)
end

# =============================================================================
#  ISO matrix
# =============================================================================

# The COD tensor diag(Bℓℓ, Bmm, Bnn) in the crack frame, built as the symmetric
# Tensors.jl storage it ends up in, so that its dimension is in its type: built
# from `Diagonal([…])`, a vector sized at run time, it was not, and every caller
# of a closed-form `cod_tensor` received a tensor of unknown dimension.
function _diagonal_cod(Bll, Bmm, Bnn, basis::TensND.AbstractBasis{3})
    T = promote_type(typeof(Bll), typeof(Bmm), typeof(Bnn))
    z = zero(T)
    return TensND.Tens(TensND.Tensors.SymmetricTensor{2, 3, T}((Bll, z, z, Bmm, z, Bnn)), basis)
end

"""
    _cod_iso_ellipse(c::EllipticCrack, E, ν) -> Tens{2,3}

Closed-form COD tensor ``\\boldsymbol{B}`` of an elliptic crack of aspect
ratio ``\\eta = b/a`` in an isotropic matrix ``(E,\\nu)``:

```math
\\begin{aligned}
B_{\\ell\\ell} &= \\frac{8\\,(1-\\nu^{2})}{3E}\\,
          \\frac{1-\\eta^{2}}
               {\\bigl(1-\\nu-\\eta^{2}\\bigr)\\mathcal{E}_\\eta + \\nu\\,\\eta^{2}\\,\\mathcal{K}_\\eta},\\\\
B_{mm} &= \\frac{8\\,(1-\\nu^{2})}{3E}\\,
          \\frac{1-\\eta^{2}}
               {\\bigl(1-(1-\\nu)\\eta^{2}\\bigr)\\mathcal{E}_\\eta - \\nu\\,\\eta^{2}\\,\\mathcal{K}_\\eta},\\\\
B_{nn} &= \\frac{8\\,(1-\\nu^{2})}{3E}\\,\\frac{1}{\\mathcal{E}_\\eta},
\\end{aligned}
```

with ``\\mathcal{K}_\\eta = \\mathcal{K}(\\sqrt{1-\\eta^{2}})`` and
``\\mathcal{E}_\\eta = \\mathcal{E}(\\sqrt{1-\\eta^{2}})`` the complete
elliptic integrals of first and second kind
[abramowitz1972](@cite). Circular penny
limit ``\\eta=1``: ``B_{nn} = 16(1-\\nu^{2})/(3\\pi E)``,
``B_{mm}=B_{\\ell\\ell}=B_{nn}/(1-\\nu/2)``.
"""
function _cod_iso_ellipse(c::EllipticCrack, E::Number, ν::Number)
    T = promote_type(typeof(E), typeof(ν))
    η = _ratio_in(T, aspect_ratio(c))
    𝒞, 𝒮, ℰ = _elliptic_CS(η)
    χ = 8 * (one(T) - ν^2) / (3 * E)
    Bll = χ / ((one(T) - ν) * 𝒞 + η^2 * 𝒮)
    Bmm = χ / ((one(T) - ν) * η^2 * 𝒮 + 𝒞)
    Bnn = χ / ℰ
    return _diagonal_cod(Bll, Bmm, Bnn, crack_basis(c))
end

"""
    _cod_iso_ribbon(c::RibbonCrack, E, ν) -> Tens{2,3}

Closed-form COD tensor of a ribbon (tunnel) crack in an isotropic
matrix.  Ribbon limit of the elliptic closed form
[kachanov1993, sevostianov2002](@cite).
"""
function _cod_iso_ribbon(c::RibbonCrack, E::Number, ν::Number)
    T = promote_type(typeof(E), typeof(ν))
    χ = T(π) * (one(T) - ν^2) / E
    Bll = χ / (one(T) - ν)
    return _diagonal_cod(Bll, χ, χ, crack_basis(c))
end

# =============================================================================
#  TI matrix (axis ≡ n̂)
# =============================================================================

@inline function _ti_sigma_gamma(E::Number, H::Number, ν₁::Number, ν₂::Number, Γ::Number)
    T = promote_type(typeof(E), typeof(H), typeof(ν₁), typeof(ν₂), typeof(Γ))
    return sqrt(T(2)) * sqrt(
        (one(T) - Γ * ν₂) / (Γ * (one(T) - ν₁)) +
            sqrt((one(T) - H * ν₂^2) / (H * (one(T) - ν₁^2))),
    )
end

"""
    _cod_ti_ellipse(c, E, H, ν₁, ν₂, Γ) -> Tens{2,3}

Closed-form COD tensor of an elliptic crack in a transversely
isotropic matrix whose TI axis is aligned with the crack normal
``\\underline{n}``. Expressions are given in the engineering
parameterization ``(E,\\nu_{1},\\nu_{2},H,\\Gamma)`` of
[hoenig1978, kanaun2009, barthelemyIJES2021](@cite); the auxiliary scalar
``\\sigma_\\gamma`` is defined in `_ti_sigma_gamma`. Reduces to the
isotropic case for ``\\nu_{1}=\\nu_{2}=\\nu``, ``H=\\Gamma=1``.
"""
function _cod_ti_ellipse(c::EllipticCrack, E::Number, H::Number, ν₁::Number, ν₂::Number, Γ::Number)
    T = promote_type(typeof(E), typeof(H), typeof(ν₁), typeof(ν₂), typeof(Γ))
    η = _ratio_in(T, aspect_ratio(c))
    σᵞ = _ti_sigma_gamma(E, H, ν₁, ν₂, Γ)
    𝒞, 𝒮, ℰ = _elliptic_CS(η)
    χ = 4 * σᵞ * (one(T) - ν₁^2) / (3 * E)
    β = σᵞ / 2 * sqrt(Γ) * (one(T) - ν₁)
    Bll = χ / (β * 𝒞 + η^2 * 𝒮)
    Bmm = χ / (β * η^2 * 𝒮 + 𝒞)
    Bnn = χ * sqrt((one(T) - H * ν₂^2) / (H * (one(T) - ν₁^2))) / ℰ
    return _diagonal_cod(Bll, Bmm, Bnn, crack_basis(c))
end

# NB: the Penny case (η = 1) is handled by the generic elliptic formula
# above; `_elliptic_CS(1)` returns the limit values `(π/4, π/4, π/2)` and
# the closed form is regular at η = 1. A previous specialized
# `_cod_ti_ellipse(::EllipticCrack{T, Penny}, …)` had erroneous prefactors
# (it did not reduce to the isotropic Penny formula
# `Bnn_iso = 16(1-ν²)/(3πE)` in the limit ν₁=ν₂=ν, H=Γ=1) and was removed.

"""
    _cod_ti_ribbon(c, E, H, ν₁, ν₂, Γ) -> Tens{2,3}

Closed-form COD tensor of a ribbon crack in an aligned TI matrix,
ribbon limit of the elliptic TI closed form
[hoenig1978, barthelemyIJES2021](@cite).
"""
function _cod_ti_ribbon(c::RibbonCrack, E::Number, H::Number, ν₁::Number, ν₂::Number, Γ::Number)
    T = promote_type(typeof(E), typeof(H), typeof(ν₁), typeof(ν₂), typeof(Γ))
    σᵞ = _ti_sigma_gamma(E, H, ν₁, ν₂, Γ)
    χ = T(π) * (one(T) - ν₁^2) / E
    Bnn = χ * sqrt((one(T) - H * ν₂^2) / (H * (one(T) - ν₁^2))) * σᵞ / 2
    Bmm = χ * σᵞ / 2
    Bll = χ / (sqrt(Γ) * (one(T) - ν₁))
    return _diagonal_cod(Bll, Bmm, Bnn, crack_basis(c))
end
