# =============================================================================
#  cod_analytical.jl — closed-form COD tensors.
#  Moduli extractors live in Core.moduli.jl; this file only contains the
#  COD formulas.
# =============================================================================

"""
    _elliptic_CS(η) -> (𝒞, 𝒮, ℰ)

Angular integrals (paper eq. 1532):
``\\mathcal C_\\eta = \\int_0^{\\pi/2}\\cos^2\\!\\phi\\,/
\\sqrt{\\cos^2\\!\\phi+\\eta^2\\sin^2\\!\\phi}\\;\\mathrm d\\phi``,
likewise ``\\mathcal S_\\eta`` with ``\\sin^2\\!\\phi``, and
``\\mathcal E_\\eta`` the complete integral of the second kind.

The circular limit ``\\eta = 1`` is a removable ``0/0`` and must be taken
by the shortcut below. The guard is deliberately **not** restricted to
`T <: Real`: on a symbolic element type `iszero(k²)` is `false` for a free
symbol (which is what we want — the elliptic branch is the general answer)
and `true` for an exact `Sym(1)`, where the general branch would otherwise
return `NaN`. Restricting it to `Real` made `cod_tensor(PennyCrack(one(Sym)), …)`
silently produce `NaN`.
"""
@inline function _elliptic_CS(η::T) where {T <: Number}
    η² = η^2
    k² = one(T) - η²
    if is_hard_numeric(T) ? iszero(k²) : isequal(k², zero(k²))
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

# =============================================================================
#  ISO matrix
# =============================================================================

"""
    _cod_iso_ellipse(c::EllipticCrack, E, ν) -> Tens{2,3}

Closed-form COD tensor ``\\mathbf B`` of an elliptic crack of aspect
ratio ``\\eta = b/a`` in an isotropic matrix ``(E,\\nu)``:

```
B_ℓℓ = 8(1−ν²)/(3E) · (1−η²) / ((1−ν−η²) 𝓔_η + ν η² 𝓚_η)
B_mm = 8(1−ν²)/(3E) · (1−η²) / ((1−(1−ν)η²) 𝓔_η − ν η² 𝓚_η)
B_nn = 8(1−ν²)/(3E) · 1/𝓔_η
```

with ``\\mathcal K_\\eta = \\mathcal K(\\sqrt{1-\\eta^{2}})`` and
``\\mathcal E_\\eta = \\mathcal E(\\sqrt{1-\\eta^{2}})`` the complete
elliptic integrals of first and second kind
([abramowitz1972](@cite)). Circular penny
limit ``\\eta=1``: ``B_{nn} = 16(1-\\nu^{2})/(3\\pi E)``,
``B_{mm}=B_{\\ell\\ell}=B_{nn}/(1-\\nu/2)``.
"""
function _cod_iso_ellipse(c::EllipticCrack, E::Number, ν::Number)
    T = promote_type(typeof(E), typeof(ν))
    η = aspect_ratio(c)
    𝒞, 𝒮, ℰ = _elliptic_CS(η)
    χ = 8 * (one(T) - ν^2) / (3 * E)
    Bll = χ / ((one(T) - ν) * 𝒞 + η^2 * 𝒮)
    Bmm = χ / ((one(T) - ν) * η^2 * 𝒮 + 𝒞)
    Bnn = χ / ℰ
    return TensND.Tens(Diagonal([Bll, Bmm, Bnn]), crack_basis(c))
end

"""
    _cod_iso_ribbon(c::RibbonCrack, E, ν) -> Tens{2,3}

Closed-form COD tensor of a ribbon (tunnel) crack in an isotropic
matrix.  Ribbon limit of the elliptic closed form
(see [kachanov1993](@cite),
 [sevostianov2002](@cite)).
"""
function _cod_iso_ribbon(c::RibbonCrack, E::Number, ν::Number)
    T = promote_type(typeof(E), typeof(ν))
    χ = T(π) * (one(T) - ν^2) / E
    Bll = χ / (one(T) - ν)
    return TensND.Tens(Diagonal([Bll, χ, χ]), crack_basis(c))
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
``\\hat{\\mathbf n}``. Expressions are given in the engineering
parameterization ``(E,\\nu_{1},\\nu_{2},H,\\Gamma)`` of
[hoenig1978](@cite),
[kanaun2009](@cite),
[barthelemyIJES2021](@cite); the auxiliary scalar
``\\sigma_\\gamma`` is defined in `_ti_sigma_gamma`. Reduces to the
isotropic case for ``\\nu_{1}=\\nu_{2}=\\nu``, ``H=\\Gamma=1``.
"""
function _cod_ti_ellipse(c::EllipticCrack, E::Number, H::Number, ν₁::Number, ν₂::Number, Γ::Number)
    T = promote_type(typeof(E), typeof(H), typeof(ν₁), typeof(ν₂), typeof(Γ))
    η = aspect_ratio(c)
    σᵞ = _ti_sigma_gamma(E, H, ν₁, ν₂, Γ)
    𝒞, 𝒮, ℰ = _elliptic_CS(η)
    χ = 4 * σᵞ * (one(T) - ν₁^2) / (3 * E)
    β = σᵞ / 2 * sqrt(Γ) * (one(T) - ν₁)
    Bll = χ / (β * 𝒞 + η^2 * 𝒮)
    Bmm = χ / (β * η^2 * 𝒮 + 𝒞)
    Bnn = χ * sqrt((one(T) - H * ν₂^2) / (H * (one(T) - ν₁^2))) / ℰ
    return TensND.Tens(Diagonal([Bll, Bmm, Bnn]), crack_basis(c))
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
([hoenig1978](@cite),
 [barthelemyIJES2021](@cite)).
"""
function _cod_ti_ribbon(c::RibbonCrack, E::Number, H::Number, ν₁::Number, ν₂::Number, Γ::Number)
    T = promote_type(typeof(E), typeof(H), typeof(ν₁), typeof(ν₂), typeof(Γ))
    σᵞ = _ti_sigma_gamma(E, H, ν₁, ν₂, Γ)
    χ = T(π) * (one(T) - ν₁^2) / E
    Bnn = χ * sqrt((one(T) - H * ν₂^2) / (H * (one(T) - ν₁^2))) * σᵞ / 2
    Bmm = χ * σᵞ / 2
    Bll = χ / (sqrt(Γ) * (one(T) - ν₁))
    return TensND.Tens(Diagonal([Bll, Bmm, Bnn]), crack_basis(c))
end
