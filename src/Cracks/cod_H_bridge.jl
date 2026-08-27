# =============================================================================
#  cod_H_bridge.jl — bridge between the crack compliance contribution
#  tensor H and the COD tensor B.
#
#  The factorization H = k (n̂ ⊗ˢ B ⊗ˢ n̂) depends on the crack shape:
#     k = 3/4  for an elliptic (including penny) crack (Kachanov / Echoes)
#     k = 2/π  for a ribbon (tunnel) crack
#  The inverse mapping therefore dispatches on the crack type.
# =============================================================================

# -----------------------------------------------------------------------------
# Factor tables — keep the geometric k as a scalar helper for clarity.
# -----------------------------------------------------------------------------
@inline _H_over_B_factor(::EllipticCrack, T) = T(3) / T(4)
@inline _H_over_B_factor(::RibbonCrack, T) = T(2) / T(π)

"""
    cod_from_compliance(H, crack, ℬ=get_basis(H)) -> Tens{2,3}

Extract the size-independent COD tensor ``\\mathbf B`` from the crack
compliance contribution tensor ``\\mathbb H`` using

```
H = k (n̂ ⊗ˢ B ⊗ˢ n̂),   k = 3/4  (elliptic)  or  k = 2/π  (ribbon).
```

Dispatches on the crack type
([kachanov1992](@cite),
 [barthelemyIJES2021](@cite)).

!!! note
    The pair [`compliance_from_cod`](@ref) / [`cod_from_compliance`](@ref)
    only preserves the ``B_{nn}`` component exactly; off-diagonal
    components involve Kelvin–Mandel pre-factors that are not inverted
    by the pair.
"""
function cod_from_compliance(
        H, crack::MFH_Core.AbstractCrack,
        ℬ::TensND.AbstractBasis = TensND.get_basis(H),
    )
    T = eltype(H)
    k = _H_over_B_factor(crack, T)
    newH = TensND.change_tens(H, ℬ)
    return TensND.Tens(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) ->
            newH[i, 3, j, 3] /
                (
                k * (
                    one(T) + MFH_Core._δ(i, 3, T) + MFH_Core._δ(j, 3, T) +
                        MFH_Core._δ(i, 3, T) * MFH_Core._δ(j, 3, T)
                ) / T(4)
            )
        ),
        ℬ,
    )
end

"""
    cod_from_compliance(H, ℬ=get_basis(H)) -> Tens{2,3}

Elliptic / 3D default: identical to `cod_from_compliance(H, EllipticCrack-like, ℬ)`
with the Kachanov factor ``k = 3/4``.  Kept for back-compatibility with
code that does not carry the crack object.  For a ribbon, pass the
`RibbonCrack` explicitly.
"""
function cod_from_compliance(H, ℬ::TensND.AbstractBasis = TensND.get_basis(H))
    T = eltype(H)
    newH = TensND.change_tens(H, ℬ)
    return TensND.Tens(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) ->
            16 * newH[i, 3, j, 3] /
                (
                3 * (
                    one(T) + MFH_Core._δ(i, 3, T) + MFH_Core._δ(j, 3, T) +
                        MFH_Core._δ(i, 3, T) * MFH_Core._δ(j, 3, T)
                )
            )
        ),
        ℬ,
    )
end

const BfromH = cod_from_compliance

"""
    compliance_from_cod(B, crack, ℬ=get_basis(B)) -> Tens{4,3}

Inverse of [`cod_from_compliance`](@ref).  Reconstruct ``\\mathbb H``
from ``\\mathbf B`` with the crack-shape-dependent factor ``k``.
"""
function compliance_from_cod(
        B, crack::MFH_Core.AbstractCrack,
        ℬ::TensND.AbstractBasis = TensND.get_basis(B),
    )
    T = eltype(B)
    k = _H_over_B_factor(crack, T)
    newB = TensND.change_tens(B, ℬ)
    data = zeros(T, 3, 3, 3, 3)
    @inbounds for i in 1:3, kidx in 1:3
        j = 3
        l = 3
        v = newB[i, kidx] * k *
            (
            one(T) + MFH_Core._δ(i, 3, T) + MFH_Core._δ(kidx, 3, T) +
                MFH_Core._δ(i, 3, T) * MFH_Core._δ(kidx, 3, T)
        ) / T(4)
        data[i, j, kidx, l] = v
    end
    sym = zeros(T, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, kidx in 1:3, l in 1:3
        sym[i, j, kidx, l] = (
            data[i, j, kidx, l] + data[j, i, kidx, l] +
                data[i, j, l, kidx] + data[j, i, l, kidx] +
                data[kidx, l, i, j] + data[l, kidx, i, j] +
                data[kidx, l, j, i] + data[l, kidx, j, i]
        ) / 8
    end
    return TensND.Tens(sym, ℬ)
end

"""
    compliance_from_cod(B, ℬ=get_basis(B)) -> Tens{4,3}

Elliptic / 3D default: identical to `compliance_from_cod(B, EllipticCrack-like, ℬ)`.
Kept for back-compatibility.
"""
function compliance_from_cod(B, ℬ::TensND.AbstractBasis = TensND.get_basis(B))
    T = eltype(B)
    newB = TensND.change_tens(B, ℬ)
    data = zeros(T, 3, 3, 3, 3)
    @inbounds for i in 1:3, k in 1:3
        j = 3
        l = 3
        v = newB[i, k] * 3 *
            (
            one(T) + MFH_Core._δ(i, 3, T) + MFH_Core._δ(k, 3, T) +
                MFH_Core._δ(i, 3, T) * MFH_Core._δ(k, 3, T)
        ) /
            16
        data[i, j, k, l] = v
    end
    sym = zeros(T, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        sym[i, j, k, l] = (
            data[i, j, k, l] + data[j, i, k, l] +
                data[i, j, l, k] + data[j, i, l, k] +
                data[k, l, i, j] + data[l, k, i, j] +
                data[k, l, j, i] + data[l, k, j, i]
        ) / 8
    end
    return TensND.Tens(sym, ℬ)
end

const HfromB = compliance_from_cod
