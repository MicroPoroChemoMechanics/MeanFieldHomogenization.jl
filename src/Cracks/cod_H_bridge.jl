# =============================================================================
#  cod_H_bridge.jl — bridge between the crack compliance contribution
#  tensor H and the COD tensor B.
#
#  The factorization H = k (n̂ ⊗ˢ B ⊗ˢ n̂), with B and the flatness both
#  normalized by the half-width b (barthelemyIJSS2026), depends on the shape:
#     k = 3/4  for an elliptic (including penny) crack
#     k = 2/π  for a ribbon (tunnel) crack
#  The derivation is §3 of docs/src/theory/cod_tensors.md.
#  The factor dispatches on the shape family, as the rest of the crack algebra
#  (`compliance.jl`), so that a user-defined crack declaring its `shape_trait`
#  inherits it.
# =============================================================================

_H_over_B_factor(crack::MFH_Core.AbstractCrack, ::Type{T}) where {T} =
    _H_over_B_factor(MFH_Core.shape_trait(crack), T)
_H_over_B_factor(::Type{<:CrackShape}, ::Type{T}) where {T} = 3 * one(T) / 4
_H_over_B_factor(::Type{Ribbon}, ::Type{T}) where {T} = 2 * one(T) / T(π)
_H_over_B_factor(S::Type, ::Type) = _unsupported_crack_shape(S, "compliance")

# H = k n ⊗ˢ B ⊗ˢ n, with n the third vector of ℬ. In an orthonormal ℬ, the
# Kelvin–Mandel matrix of n ⊗ˢ B ⊗ˢ n is the out-of-plane block of B read in ℬ:
# the product of the components went through 81 of them and checked their
# symmetries back, most of the cost of a crack in a Mori–Tanaka scheme.
function _H_from_B(B, k, ℬ::TensND.OrthonormalBasis{3})
    X = SMatrix{3, 3}(TensND.components(B, ℬ))
    return k * TensND.from_mandel(mandel_op_embed(compliance_op_block(X)), ℬ)
end
function _H_from_B(B, k, ℬ::TensND.AbstractBasis)
    n̂ = TensND.tens_basis(ℬ, 3)
    return k * (n̂ ⊗ˢ TensND.express(B, ℬ) ⊗ˢ n̂)
end

# Its inverse: in ℬ, where n = e₃, (n ⊗ˢ B ⊗ˢ n)_{i3j3} = B_ij (1 + δ_i3)(1 + δ_j3) / 4,
# and these components of H determine B.
function _B_from_H(H, k, ℬ::TensND.AbstractBasis)
    newH = TensND.express(H, ℬ)
    return TensND.Tens(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) -> 4 * newH[i, 3, j, 3] / (k * (1 + (i == 3)) * (1 + (j == 3)))
        ),
        ℬ,
    )
end

"""
    cod_from_compliance(H, crack, ℬ=get_basis(H)) -> Tens{2,3}

Extract the size-independent COD tensor ``\\boldsymbol{B}`` from the crack
compliance contribution tensor ``\\mathbb{H}`` using

```math
\\mathbb{H} = k\\,\\underline{n}\\stackrel{s}{\\otimes}\\boldsymbol{B}\\stackrel{s}{\\otimes}\\underline{n},
```

with ``k = 3/4`` for an elliptic crack and ``k = 2/\\pi`` for a ribbon, and
``\\underline{n}`` the third vector of `ℬ`, the crack normal when `H` comes from
[`inverse_property_contribution`](@ref). The exact inverse of
[`compliance_from_cod`](@ref).

Dispatches on the shape family of the crack. ``\boldsymbol{B}`` is normalized by
the half-width ``b`` [barthelemyIJSS2026](@cite), and the factor ``k`` is derived
in [From B to the compliance H](@ref th-cod-compliance) [kachanov1992](@cite).
"""
cod_from_compliance(H, crack::MFH_Core.AbstractCrack, ℬ::TensND.AbstractBasis = TensND.get_basis(H)) =
    _B_from_H(H, _H_over_B_factor(crack, eltype(H)), ℬ)

"""
    cod_from_compliance(H, ℬ=get_basis(H)) -> Tens{2,3}

Elliptic / 3D default: identical to `cod_from_compliance(H, EllipticCrack-like, ℬ)`
with the Kachanov factor ``k = 3/4``.  Kept for back-compatibility with
code that does not carry the crack object.  For a ribbon, pass the
`RibbonCrack` explicitly.
"""
cod_from_compliance(H, ℬ::TensND.AbstractBasis = TensND.get_basis(H)) =
    _B_from_H(H, _H_over_B_factor(EllipticShape, eltype(H)), ℬ)

const BfromH = cod_from_compliance

"""
    compliance_from_cod(B, crack, ℬ=get_basis(B)) -> Tens{4,3}

``\\mathbb{H} = k\\,\\underline{n}\\stackrel{s}{\\otimes}\\boldsymbol{B}\\stackrel{s}{\\otimes}\\underline{n}``,
with the crack-shape-dependent factor ``k`` and ``\\underline{n}`` the third
vector of `ℬ`: what [`inverse_property_contribution`](@ref) returns from
[`cod_tensor`](@ref), and the exact inverse of [`cod_from_compliance`](@ref).
"""
compliance_from_cod(B, crack::MFH_Core.AbstractCrack, ℬ::TensND.AbstractBasis = TensND.get_basis(B)) =
    _H_from_B(B, _H_over_B_factor(crack, eltype(B)), ℬ)

"""
    compliance_from_cod(B, ℬ=get_basis(B)) -> Tens{4,3}

Elliptic / 3D default: identical to `compliance_from_cod(B, EllipticCrack-like, ℬ)`.
Kept for back-compatibility.
"""
compliance_from_cod(B, ℬ::TensND.AbstractBasis = TensND.get_basis(B)) =
    _H_from_B(B, _H_over_B_factor(EllipticShape, eltype(B)), ℬ)

const HfromB = compliance_from_cod
