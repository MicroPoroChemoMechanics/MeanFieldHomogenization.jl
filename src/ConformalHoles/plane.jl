# =============================================================================
#  plane.jl — a `ConformalHole` in a two-dimensional cell: a hole of the plane.
#
#  The reference is an isotropic tensor of the plane, the stiffness
#  2k 𝕁 + 2μ 𝕂 or the conductivity. The compliance of the hole computed with
#  the map (`hole.jl`) is the whole of the problem: no axial nor antiplane
#  block. The frame of the hole is not read: the axes are those of the map.
#  An elliptic hole also takes an anisotropic stiffness or conductivity
#  (`anisotropic.jl`).
# =============================================================================

"""
    Core.gradient_gradient_concentration(hole::ConformalHole, P₁, P₀::AbstractTens{order, 2})

The same tensors in a two-dimensional problem, whose reference `P₀` is an
isotropic tensor of the plane: the stiffness ``2k\\,\\mathbb J + 2\\mu\\,\\mathbb K``
(`TensISO{4,2}`), for which the mean strain of the hole per unit remote strain
is ``2\\,(1/\\mu + 1/k)`` times the matrix of `_hole_compliance`, contracted
with `P₀`; or the conductivity (`TensISO{2,2}`), which does not enter. The
result is written in the axes of the map, ``z = x_1 + \\mathrm{i}\\,x_2``.
Plane strain and plane stress differ by the modulus ``k`` of `P₀` only.

An elliptic hole, whose map has its first term only, takes any stiffness of
the plane: its complex potentials in an anisotropic matrix are known in closed
form ([The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse)).
A symbolic stiffness is then a `TensISO`, a `TensCubic` or a `TensOrtho`.
"""
Core.gradient_gradient_concentration(
    h::ConformalHole, ::TensND.AbstractTens{4, 2}, C₀::TensND.AbstractTens{4, 2}; kw...
) = _plane_concentration(h, C₀)

function _plane_concentration(h::ConformalHole, C₀::TensND.TensISO{4, 2})
    α, β = TensND.get_data(C₀)                            # 2k and 2μ
    a, b, o = (α + β) / 2, (α - β) / 2, zero(α)
    C = SMatrix{3, 3}(a, b, o, b, a, o, o, o, β)
    return TensND.from_mandel(((4 * (inv(α) + inv(β))) * h.compliance) * C)
end
# A matrix of another type: the ellipse has its solution whatever the
# anisotropy, which holds at isotropy too; another hole needs a matrix that is
# isotropic, to the tolerance of `_plane_isotropic`, whose result is asserted:
# the projection it calls is not inferred for every class.
_plane_concentration(h::ConformalHole, C₀::TensND.AbstractTens{4, 2}) =
    _is_ellipse(h) ? TensND.from_mandel(_anisotropic_ellipse_concentration(first(h.coefficients), C₀)) :
    _plane_concentration(h, _plane_isotropic(C₀)::TensND.TensISO{4, 2, eltype(C₀), 2})

# The map of an ellipse has its first term only.
_is_ellipse(h::ConformalHole) = all(iszero, @view h.coefficients[2:end])

Core.gradient_gradient_concentration(
    h::ConformalHole, ::TensND.AbstractTens{2, 2}, K₀::TensND.AbstractTens{2, 2}; kw...
) = _plane_concentration(h, K₀)

function _plane_concentration(h::ConformalHole, ::TensND.TensISO{2, 2})
    S = _plane_gradient_concentration(h.coefficients)
    return TensND.from_mandel(SVector{3}(S[1, 1], S[2, 2], sqrt(2 * one(eltype(S))) * S[2, 1]))
end
# The insulating ellipse in a conductivity of another type has its solution
# whatever the anisotropy; its concentration tensor is not symmetric then.
# Another hole needs an isotropic conductivity.
_plane_concentration(h::ConformalHole, K₀::TensND.AbstractTens{2, 2}) =
    TensND.Tens(Tensor{2, 2}(_plane_gradient(h, K₀)))
_plane_gradient(h::ConformalHole, K₀) =
    _is_ellipse(h) ? _anisotropic_ellipse_gradient(first(h.coefficients), _symmetric_conductivity(K₀)) :
    (_plane_isotropic(K₀); _plane_gradient_concentration(h.coefficients))

# The conductivity the complex representation takes: symmetric. Symbols do not
# compare, and are taken as such.
_symmetric_conductivity(K₀) = _symmetric_conductivity(Val(is_hard_numeric(eltype(K₀))), K₀)
_symmetric_conductivity(::Val{false}, K₀) = K₀
function _symmetric_conductivity(::Val{true}, K₀)
    K = TensND.primal.(TensND.canonical_components(K₀))
    abs(K[1, 2] - K[2, 1]) ≤ 1.0e-10 * maximum(abs, K) || throw(
        ArgumentError(
            "the elliptic hole in an anisotropic matrix of the plane takes a symmetric conductivity; " *
                "got K₁₂ = $(K[1, 2]) and K₂₁ = $(K[2, 1])"
        )
    )
    return K₀
end

function Core.flux_gradient_concentration(
        h::ConformalHole{T}, ::TensND.AbstractTens{4, 2}, C₀::TensND.AbstractTens{4, 2}; kw...
    ) where {T}
    z = zero(promote_type(T, eltype(C₀)))
    return TensND.TensISO{2}(z, z)
end
function Core.flux_gradient_concentration(
        h::ConformalHole{T}, ::TensND.AbstractTens{2, 2}, K₀::TensND.AbstractTens{2, 2}; kw...
    ) where {T}
    return TensND.TensISO{2}(zero(promote_type(T, eltype(K₀))))
end
