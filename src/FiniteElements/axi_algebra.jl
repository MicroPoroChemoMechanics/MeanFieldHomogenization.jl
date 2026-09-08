# =============================================================================
#  axi_algebra.jl — the backend-independent algebra of the axisymmetric solve.
#
#  Everything here is plain linear algebra on Kelvin-Mandel matrices: the
#  fixed point of the corrected boundary condition, the rotations between the
#  inclusion's local frame and the global one, the transverse-isotropy guards
#  on the constituents and the isotropy guard on the reference medium.
#
#  No finite-element library appears in this file, and none should: it is
#  shared verbatim by every backend.
# =============================================================================

"""
    _axi_correct(A_E, B_E, A_p, B_p, P0) -> (A, B, X)

The linear fixed point of the corrected boundary condition, on one modal
block: `X = (𝕀 − (𝔹ᵖ − P₀·𝔸ᵖ))⁻¹ (𝔹ᴱ − P₀·𝔸ᴱ)`, then `𝔸 = 𝔸ᴱ + 𝔸ᵖ X` and
`𝔹 = 𝔹ᴱ + 𝔹ᵖ X`.
"""
function _axi_correct(A_E, B_E, A_p, B_p, P0)
    Π_E = B_E - P0 * A_E
    Π_p = B_p - P0 * A_p
    X = (LinearAlgebra.I - Π_p) \ Π_E
    return A_E + A_p * X, B_E + B_p * X, X
end

# ─── Material matrices in the local frame ────────────────────────────────────

"""
    _fe_frame(incl) -> 3×3

Rotation matrix whose columns are the inclusion's local axes in global
coordinates. Shared by every finite-element inclusion: each one solves in its
own frame and rotates the result out.
"""
_fe_frame(incl) = hcat(Core._frame_columns(Core.inclusion_basis(incl))...)

"Components of a 4th-order tensor in the local frame, as a 3×3×3×3 array."
function _to_local4(P::TensND.AbstractTens{4, 3}, R)
    G = Core._C_array(P)
    L = zeros(Float64, 3, 3, 3, 3)
    @inbounds for p in 1:3, q in 1:3, r in 1:3, s in 1:3
        acc = 0.0
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            acc += R[i, p] * R[j, q] * R[k, r] * R[l, s] * G[i, j, k, l]
        end
        L[p, q, r, s] = acc
    end
    return L
end

"Components of a 2nd-order tensor in the local frame, as a 3×3 matrix."
_to_local2(P::TensND.AbstractTens{2, 3}, R) = R' * TensND.components_canon(P) * R

"Rebuild a global-frame `Tens{4,3}` from local-frame Kelvin-Mandel components."
function _from_local66(M66::AbstractMatrix{Float64}, R)
    L = Core.array_from_mandel66(M66)
    G = zeros(Float64, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        acc = 0.0
        for p in 1:3, q in 1:3, r in 1:3, s in 1:3
            acc += R[i, p] * R[j, q] * R[k, r] * R[l, s] * L[p, q, r, s]
        end
        G[i, j, k, l] = acc
    end
    return TensND.TensCanonical(
        Tensors.Tensor{4, 3}((i, j, k, l) -> G[i, j, k, l])
    )
end

_from_local33(M::AbstractMatrix{Float64}, R) = TensND.TensCanonical(R * M * R')

"""
    _axi_check_ti(M66, what) -> M66

Refuse a constituent whose local-frame components are not transversely
isotropic about the symmetry axis: any other anisotropy couples the Fourier
modes, and the whole point of the axisymmetric formulation is that they do not.
"""
function _axi_check_ti(M66::AbstractMatrix{Float64}, what::AbstractString; rtol = 1.0e-8)
    ref = M66
    scale = maximum(abs, ref)
    # A TI tensor about e₃ has no (11,13), (11,23), (11,12), (33,·shear) coupling
    # and satisfies M₁₁ = M₂₂, M₁₃ = M₂₃, M₄₄ = M₅₅, M₆₆ = M₁₁ − M₁₂.
    bad = max(
        abs(ref[1, 1] - ref[2, 2]), abs(ref[1, 3] - ref[2, 3]),
        abs(ref[4, 4] - ref[5, 5]), abs(ref[6, 6] - (ref[1, 1] - ref[1, 2])),
        maximum(abs, ref[1:3, 4:6]), maximum(abs, ref[4:6, 1:3]),
        abs(ref[4, 5]), abs(ref[4, 6]), abs(ref[5, 6]),
    )
    bad ≤ rtol * scale || throw(
        ArgumentError(
            "$what must be transversely isotropic about the inclusion's " *
                "symmetry axis — an axisymmetric Fourier formulation cannot " *
                "represent any other anisotropy, because the modes would couple. " *
                "Deviation: $(round(bad / scale * 100, sigdigits = 3)) %."
        )
    )
    return ref
end

function _axi_check_ti2(M::AbstractMatrix{Float64}, what::AbstractString; rtol = 1.0e-8)
    scale = maximum(abs, M)
    bad = max(
        abs(M[1, 1] - M[2, 2]), abs(M[1, 2]), abs(M[2, 1]),
        abs(M[1, 3]), abs(M[3, 1]), abs(M[2, 3]), abs(M[3, 2]),
    )
    bad ≤ rtol * scale || throw(
        ArgumentError(
            "$what must be transversely isotropic about the inclusion's " *
                "symmetry axis; deviation $(round(bad / scale * 100, sigdigits = 3)) %."
        )
    )
    return M
end

"""
    _fe_iso_moduli(P₀; rtol, what) -> (μ, ν)

Shear modulus and Poisson ratio of the reference medium, refusing anything
that is not isotropic in *content* — the corrected boundary condition uses the
closed-form Kelvin dipole field, whose anisotropic counterpart (Pan-Chou,
Barnett-Willis) is not implemented.

`rtol` is deliberately loose (0.01 %). The tensors this solver *returns* are
isotropic only to the discretization error — a few parts in a million — so an
iterative scheme fed back its own estimate can never present a reference
isotropic to machine precision. A tighter guard would refuse `SelfConsistent`
and `AsymmetricSelfConsistent` on a perfectly legitimate isotropic problem.
Genuine anisotropy, the case the guard exists for, is orders of magnitude
larger; `IsoSymmetrize` remains the answer for it.
"""
function _fe_iso_moduli(
        C₀::TensND.AbstractTens{4, 3}; rtol = 1.0e-4,
        what::AbstractString = "this inclusion",
    )
    C_iso = Core.isotropify(C₀)
    A, Aiso = Core._C_array(C₀), Core._C_array(C_iso)
    dev = maximum(abs, A .- Aiso)
    dev ≤ rtol * max(maximum(abs, Aiso), eps()) || throw(
        ArgumentError(
            "$what supports an isotropic reference medium only " *
                "(the corrected boundary condition uses the closed-form Kelvin " *
                "dipole field). The reference deviates from isotropy by " *
                "$(round(dev / maximum(abs, Aiso) * 100, sigdigits = 3)) %. Add " *
                "`symmetrize = IsoSymmetrize()` to the phase if this comes from an " *
                "iterative scheme."
        )
    )
    E, ν = Core.extract_iso_moduli(C_iso)
    return Float64(E / (2 * (1 + ν))), Float64(ν)
end

function _fe_iso_scalar(
        K₀::TensND.AbstractTens{2, 3}; rtol = 1.0e-4,
        what::AbstractString = "this inclusion",
    )
    M = TensND.components_canon(K₀)
    k = (M[1, 1] + M[2, 2] + M[3, 3]) / 3
    dev = maximum(abs, M .- k .* LinearAlgebra.I(3))
    dev ≤ rtol * max(abs(k), eps()) || throw(
        ArgumentError(
            "$what supports an isotropic reference medium only; " *
                "the reference conductivity deviates by " *
                "$(round(dev / abs(k) * 100, sigdigits = 3)) %."
        )
    )
    return Float64(k)
end

# ─── Memoization key ─────────────────────────────────────────────────────────
#
#  Rounded to 12 significant digits, and shared: an iterative scheme presents a
#  reference that differs in the last bits between iterations that are, for the
#  purposes of a finite-element solve, the same reference.

_fe_cache_key(P₀::TensND.AbstractTens{4, 3}) =
    map(x -> round(x, sigdigits = 12), Tuple(Core._C_array(P₀)))
_fe_cache_key(P₀::TensND.AbstractTens{2, 3}) =
    map(x -> round(x, sigdigits = 12), Tuple(TensND.components_canon(P₀)))
