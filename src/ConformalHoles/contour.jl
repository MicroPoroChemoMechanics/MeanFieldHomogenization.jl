# =============================================================================
#  contour.jl — the stress along the contour of a hole of the plane.
#
#  The contour is free: the stress there has one component, on the tangent
#  (the hoop stress). In an isotropic matrix it is 4 Re[φ′(σ)/ω′(σ)], the sum
#  of the two normal stresses, with the potential φ of `hole.jl`; it does not
#  depend on the moduli. For an elliptic hole in an anisotropic matrix it
#  follows from the potentials of `anisotropic.jl`.
# =============================================================================

"""
    contour_stress(hole::ConformalHole, Σ, θ)
    contour_stress(hole::ConformalHole, Σ, θ, C₀)

Stress along the contour of a free hole of the plane at the point
``z = \\omega(\\mathrm{e}^{\\mathrm{i}\\theta})``, under the remote stress `Σ`, a
tensor of order 2 of the plane: the normal stress on the facet normal to the
tangent, the only component that the free surface leaves. `θ` is an angle or a
collection of angles, for which the potentials are computed once.

Without a stiffness the matrix is isotropic, and its moduli do not enter:
with the potential ``\\varphi`` of the hole,

```math
\\sigma_{tt}(\\theta) = 4\\operatorname{Re}\\frac{\\varphi'(\\sigma)}{\\omega'(\\sigma)},
\\qquad \\sigma = \\mathrm{e}^{\\mathrm{i}\\theta} .
```

With the stiffness `C₀` of an anisotropic matrix of the plane the hole is an
ellipse, the one case solved in closed form
([The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse)).

# Examples
The ellipse of semi-axes ``a = 1.5`` and ``b = 0.5`` pulled across its long
axis: the stress at its tip is ``1 + 2a/b`` times the remote one.
```jldoctest
julia> using TensND

julia> contour_stress(ConformalHole((0.5,)), Tens([0.0 0.0; 0.0 1.0]), 0.0)
7.0
```
"""
contour_stress(h::ConformalHole, Σ::TensND.AbstractTens{2, 2}, θ) = _evaluate(_isotropic_contour(h, _remote_stress(Σ)), θ)
contour_stress(h::ConformalHole, Σ::TensND.AbstractTens{2, 2}, θ, ::TensND.TensISO{4, 2}) = contour_stress(h, Σ, θ)
function contour_stress(h::ConformalHole, Σ::TensND.AbstractTens{2, 2}, θ, C₀::TensND.AbstractTens{4, 2})
    _is_ellipse(h) || throw(
        ArgumentError(
            "the stress along the contour of a hole in an anisotropic matrix is known for the ellipse, " *
                "whose map has one term; this map has $(length(h.coefficients))"
        )
    )
    return _evaluate(_AnisotropicContour(first(h.coefficients), _characteristic_pair(C₀)..., _remote_stress(Σ)), θ)
end

# (Σ₁₁, Σ₂₂, Σ₁₂) in the canonical axes; the symmetric part is the stress.
function _remote_stress(Σ::TensND.AbstractTens{2, 2})
    S = TensND.canonical_components(Σ)
    return (S[1, 1], S[2, 2], (S[1, 2] + S[2, 1]) / 2)
end

_evaluate(contour, θ::Number) = contour(θ)
_evaluate(contour, θ) = map(contour, θ)

# What does not depend on the point: the map, and the potential φ = Γ ζ + Σ bₖ ζ⁻ᵏ.
struct _IsotropicContour{T}
    m::Vector{Complex{T}}
    Γ::T
    b::Vector{Complex{T}}
end

_isotropic_contour(h::ConformalHole, Σ) = ((m, Γ, _, b) = _loaded_potential(h, Σ); _IsotropicContour(m, Γ, b))

# The potential Φ = Γ ζ + Σ bₖ ζ⁻ᵏ of the hole under the remote stress
# (Σ₁₁, Σ₂₂, Σ₁₂), a combination of those of the three unit loads, with the
# constant Γ′ of the second potential and the map at the same element type.
function _loaded_potential(h::ConformalHole, Σ)
    a, loads = _potential_coefficients(h.coefficients)
    weights = (Σ[1], Σ[2], sqrt(2 * one(Σ[3])) * Σ[3])
    return (
        h.coefficients .* one(eltype(weights)),
        sum(w * Γ for (w, (Γ, _)) in zip(weights, loads)), sum(w * Γ′ for (w, (_, Γ′)) in zip(weights, loads)),
        [sum(weights[q] * a[k, q] for q in 1:3) for k in axes(a, 1)],
    )
end

function (c::_IsotropicContour)(θ)
    sθ, cθ = sincos(θ)
    σ⁻¹ = complex(cθ, -sθ)
    φ′ = c.Γ - sum(k * c.b[k] * σ⁻¹^(k + 1) for k in eachindex(c.b))
    ω′ = 1 - sum(k * c.m[k] * σ⁻¹^(k + 1) for k in eachindex(c.m))
    return 4 * real(_cdiv(φ′, ω′))
end

struct _AnisotropicContour{M <: Complex, P <: Complex, S <: Tuple}
    m::M
    s::P
    p::P
    Σ::S
end

function (c::_AnisotropicContour)(θ)
    (σ₁₁, σ₂₂, σ₁₂), (t₁, t₂) = _anisotropic_ellipse_stress(c.m, c.s, c.p, c.Σ, θ)
    return (σ₁₁ * t₁^2 + 2 * σ₁₂ * (t₁ * t₂) + σ₂₂ * t₂^2) / (t₁^2 + t₂^2)
end
