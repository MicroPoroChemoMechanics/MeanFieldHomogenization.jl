# =============================================================================
#  rotational_average.jl — EXACT rotation-group averages of 2nd/4th-order
#  tensors with minor symmetry only (no major symmetry assumed).
#
#  Two operations, mirroring a runtime tensor symmetrization:
#
#  * `isotropify(t)`            — exact average over SO(3):
#        4th order → TensISO{4} (the isotropic subspace of minor-symmetric
#        tensors is 2-dimensional {𝕁, 𝕂}, even without major symmetry)
#        2nd order → TensISO{2}
#
#  * `transverse_isotropify(t, n)` — exact average over rotations about `n`:
#        4th order → TensTI{4,T,8} (full 8-dim commutant of the SO(2) action
#        on the Kelvin-Mandel space: Walpole W₁..W₆ + antisymmetric couplings
#        W₇ (m=1) and W₈ (m=2))
#        2nd order → TensTI{2,T,3} (a·nT + b·nₙ + c·w, w the in-plane
#        rotation generator w·p = n×p)
#
#  These are the operators to use on CONCENTRATION / CONTRIBUTION tensors
#  inside scheme kernels — such tensors generally lack major symmetry, and
#  the average preserves that (ℓ₃ ≠ ℓ₄, ℓ₇, ℓ₈).  For parameter extraction
#  or reporting, use the best-fit projections instead
#  (`Schemes.best_fit_ti`, `Schemes.best_fit_iso`).
#
#  Kelvin-Mandel convention (Tensors.jl): index order (11, 22, 33, 23, 13, 12)
#  with weight √2 on the shear slots.  The closed-form azimuthal average about
#  e₃ is the orthogonal projection onto the commutant algebra:
#      m=0 block {33, (11+22)/√2}      → kept in full (ℓ₁, ℓ₂, ℓ₃, ℓ₄)
#      m=1 doublet {23, 13}            → aI + bG → (ℓ₆, ℓ₇)
#      m=2 doublet {(11−22)/√2, 12}    → aI + bG → (ℓ₅, ℓ₈)
#  It is algebraically equivalent to the closed form hard-coded in echoes'
#  `transverse_isotropify_around_ez` (up to echoes' Mandel index convention).
# =============================================================================

# ── Kelvin-Mandel helpers (minor-symmetric, Tensors.jl index convention) ─────

const _MANDEL_IDX = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

"""
    mandel66_minor(arr::AbstractArray{T,4}) → Matrix{T} (6×6)

Kelvin-Mandel 6×6 matrix of a 4th-order array, minor-symmetrizing on read
(exactly, so tensors that are minor-symmetric up to round-off are cleanly
projected).  Index convention (11, 22, 33, 23, 13, 12), weights √2.
"""
function mandel66_minor(arr::AbstractArray{T, 4}) where {T}
    sq2 = sqrt(T(2))
    Λ(I) = I ≤ 3 ? one(T) : sq2
    M = Matrix{T}(undef, 6, 6)
    @inbounds for I in 1:6, J in 1:6
        (i, j) = _MANDEL_IDX[I]
        (k, l) = _MANDEL_IDX[J]
        v = (arr[i, j, k, l] + arr[j, i, k, l] + arr[i, j, l, k] + arr[j, i, l, k]) / 4
        M[I, J] = Λ(I) * Λ(J) * v
    end
    return M
end

"""
    array_from_mandel66(M::AbstractMatrix{T}) → Array{T,4}

Inverse of [`mandel66_minor`](@ref): rebuild the (exactly minor-symmetric)
3×3×3×3 array from a 6×6 Kelvin-Mandel matrix.
"""
function array_from_mandel66(M::AbstractMatrix{T}) where {T}
    sq2 = sqrt(T(2))
    Λ(I) = I ≤ 3 ? one(T) : sq2
    arr = Array{T, 4}(undef, 3, 3, 3, 3)
    @inbounds for I in 1:6, J in 1:6
        (i, j) = _MANDEL_IDX[I]
        (k, l) = _MANDEL_IDX[J]
        v = M[I, J] / (Λ(I) * Λ(J))
        arr[i, j, k, l] = v
        arr[j, i, k, l] = v
        arr[i, j, l, k] = v
        arr[j, i, l, k] = v
    end
    return arr
end

# ── Axis frame ───────────────────────────────────────────────────────────────

"""
    _axis_frame(n) → Matrix (3×3)

Return an orthonormal, right-handed frame `(e₁′, e₂′, n̂)` as the columns of a
3×3 matrix, with `n̂` the normalized axis.  The in-plane vectors are chosen
deterministically (from the canonical axis least aligned with `n`); the
azimuthal average is independent of that choice.
"""
function _axis_frame(n)
    T = promote_type(typeof(n[1]), typeof(n[2]), typeof(n[3]))
    nn = sqrt(n[1]^2 + n[2]^2 + n[3]^2)
    n̂ = (T(n[1] / nn), T(n[2] / nn), T(n[3] / nn))
    a1, a2, a3 = abs(n̂[1]), abs(n̂[2]), abs(n̂[3])
    h = a1 ≤ a2 ? (a1 ≤ a3 ? (one(T), zero(T), zero(T)) : (zero(T), zero(T), one(T))) :
        (a2 ≤ a3 ? (zero(T), one(T), zero(T)) : (zero(T), zero(T), one(T)))
    # u = h × n̂ (in-plane), v = n̂ × u  →  (u, v, n̂) right-handed
    u1 = h[2] * n̂[3] - h[3] * n̂[2]
    u2 = h[3] * n̂[1] - h[1] * n̂[3]
    u3 = h[1] * n̂[2] - h[2] * n̂[1]
    un = sqrt(u1^2 + u2^2 + u3^2)
    u1, u2, u3 = u1 / un, u2 / un, u3 / un
    v1 = n̂[2] * u3 - n̂[3] * u2
    v2 = n̂[3] * u1 - n̂[1] * u3
    v3 = n̂[1] * u2 - n̂[2] * u1
    R = Matrix{T}(undef, 3, 3)
    R[1, 1], R[2, 1], R[3, 1] = u1, u2, u3
    R[1, 2], R[2, 2], R[3, 2] = v1, v2, v3
    R[1, 3], R[2, 3], R[3, 3] = n̂[1], n̂[2], n̂[3]
    return R, n̂
end

# Components of a 4th-order array in the frame whose COLUMNS are given by R:
# arr′[a,b,c,d] = R[i,a] R[j,b] R[k,c] R[l,d] arr[i,j,k,l], via four successive
# single-index contractions (cost 4·3⁵ instead of 3⁸).
function _rotate4(arr::AbstractArray{TA, 4}, R::AbstractMatrix{TR}) where {TA, TR}
    T = promote_type(TA, TR)
    t1 = zeros(T, 3, 3, 3, 3)
    @inbounds for a in 1:3, j in 1:3, k in 1:3, l in 1:3
        s = zero(T)
        for i in 1:3
            s += R[i, a] * arr[i, j, k, l]
        end
        t1[a, j, k, l] = s
    end
    t2 = zeros(T, 3, 3, 3, 3)
    @inbounds for a in 1:3, b in 1:3, k in 1:3, l in 1:3
        s = zero(T)
        for j in 1:3
            s += R[j, b] * t1[a, j, k, l]
        end
        t2[a, b, k, l] = s
    end
    t1 .= 0
    @inbounds for a in 1:3, b in 1:3, c in 1:3, l in 1:3
        s = zero(T)
        for k in 1:3
            s += R[k, c] * t2[a, b, k, l]
        end
        t1[a, b, c, l] = s
    end
    t2 .= 0
    @inbounds for a in 1:3, b in 1:3, c in 1:3, d in 1:3
        s = zero(T)
        for l in 1:3
            s += R[l, d] * t1[a, b, c, l]
        end
        t2[a, b, c, d] = s
    end
    return t2
end

# ── Closed-form azimuthal average about e₃ (Kelvin-Mandel, Tensors order) ────

"""
    _ti_params_from_mandel_ez(M) → NTuple{8}

Coefficients `(ℓ₁, …, ℓ₈)` of the exact azimuthal average about `e₃` of a
minor-symmetric tensor given by its 6×6 Kelvin-Mandel matrix (Tensors.jl
index order 11,22,33,23,13,12).  Orthogonal projection onto the commutant
algebra; TensND Walpole convention (ℓ₃ ↔ C₃₃₁₁-side, ℓ₄ ↔ C₁₁₃₃-side).
"""
function _ti_params_from_mandel_ez(M::AbstractMatrix{T}) where {T}
    sq2 = sqrt(T(2))
    ℓ₁ = M[3, 3]
    ℓ₂ = (M[1, 1] + M[2, 2] + M[1, 2] + M[2, 1]) / 2
    ℓ₃ = (M[3, 1] + M[3, 2]) / sq2
    ℓ₄ = (M[1, 3] + M[2, 3]) / sq2
    ℓ₅ = (M[1, 1] + M[2, 2] - M[1, 2] - M[2, 1]) / 4 + M[6, 6] / 2
    ℓ₆ = (M[4, 4] + M[5, 5]) / 2
    ℓ₇ = (M[5, 4] - M[4, 5]) / 2
    ℓ₈ = (M[6, 1] - M[6, 2] - M[1, 6] + M[2, 6]) / (2 * sq2)
    return (ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆, ℓ₇, ℓ₈)
end

"""
    _ti_mandel_ez(params::NTuple{8,T}) → Matrix{T} (6×6)

Kelvin-Mandel matrix (Tensors order, axis e₃) of the axially-invariant tensor
with coefficients `(ℓ₁,…,ℓ₈)`.  Inverse of [`_ti_params_from_mandel_ez`](@ref)
on the commutant subspace.
"""
function _ti_mandel_ez(p::NTuple{8, T}) where {T}
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆, ℓ₇, ℓ₈ = p
    sq2 = sqrt(T(2))
    z = zero(T)
    M = zeros(T, 6, 6)
    M[1, 1] = M[2, 2] = (ℓ₂ + ℓ₅) / 2
    M[1, 2] = M[2, 1] = (ℓ₂ - ℓ₅) / 2
    M[3, 3] = ℓ₁
    M[3, 1] = M[3, 2] = ℓ₃ / sq2
    M[1, 3] = M[2, 3] = ℓ₄ / sq2
    M[4, 4] = M[5, 5] = ℓ₆
    M[4, 5] = -ℓ₇
    M[5, 4] = ℓ₇
    M[6, 6] = ℓ₅
    M[6, 1] = ℓ₈ / sq2
    M[6, 2] = -ℓ₈ / sq2
    M[1, 6] = -ℓ₈ / sq2
    M[2, 6] = ℓ₈ / sq2
    return M
end

# ── Public API — 4th order ───────────────────────────────────────────────────

# `isotropify` EXTENDS TensND's rather than declaring a rival binding. TensND
# exports one too — it takes a raw array and computes the same average — and
# `Core` is `using TensND`, so two independent functions of that name both
# reach the top-level module through `using`. Julia then refuses to bind the
# name at all, and `MeanFieldHomogenization.isotropify` was an `UndefVarError`
# despite being exported. Adding methods to the one function is the same
# arrangement `layer_count` and `layer_interface` already use across the
# layered morphologies. `transverse_isotropify` has no such twin and stands
# alone.
import TensND: isotropify

"""
    isotropify(t::TensND.AbstractTens{4,3}) → TensND.TensISO{4}

Exact average of `t` over SO(3):
`α = T_iijj/3`, `β = (T_ijij − α)/5` → `α𝕁 + β𝕂`.
Valid for minor-symmetric tensors with or without major symmetry (the
isotropic subspace is {𝕁, 𝕂} in both cases).
"""
function isotropify(t::TensND.AbstractTens{4, 3})
    arr = TensND.get_array(t)
    α = sum(arr[i, i, j, j] for i in 1:3, j in 1:3) / 3
    full_trace = sum(arr[i, j, i, j] for i in 1:3, j in 1:3)
    β = (full_trace - α) / 5
    return TensND.TensISO{3}(α, β)
end

"""
    isotropify(t::TensND.AbstractTens{2,3}) → TensND.TensISO{2}

Exact SO(3) average of a 2nd-order tensor: `(tr t / 3) 𝟏`.
"""
function isotropify(t::TensND.AbstractTens{2, 3})
    arr = TensND.get_array(t)
    λ = (arr[1, 1] + arr[2, 2] + arr[3, 3]) / 3
    return TensND.TensISO{3}(λ)
end

"""
    transverse_isotropify(t::TensND.AbstractTens{4,3}, n) → TensND.TensTI{4,T,8}

Exact average of `t` over all rotations about the axis `n`
(`(1/2π)∫ R_φ ⋆ t dφ`), for a minor-symmetric `t` with or without major
symmetry.  The result lives in the full 8-dimensional axially-invariant
space: the non-major-symmetric components (ℓ₃ ≠ ℓ₄) and the antisymmetric
azimuthal couplings (ℓ₇, ℓ₈) are preserved — they are dropped by naive
symmetric TI projections but present in e.g. averaged strain-concentration
tensors.
"""
function transverse_isotropify(t::TensND.AbstractTens{4, 3}, n)
    R, n̂ = _axis_frame(n)
    # Structured fast paths (already axially invariant about n̂)
    if t isa TensND.TensISO{4, 3}
        return TensND._lift_walpole_N8(TensND.fromISO(t, n̂))
    elseif t isa TensND.TensTI{4} && TensND.axis(t) == n̂
        return TensND._lift_walpole_N8(t)
    end
    arr = TensND.get_array(t)
    arr_ez = _rotate4(arr, R)
    p = _ti_params_from_mandel_ez(mandel66_minor(arr_ez))
    return TensND.TensTI{4}(p..., n̂)
end

"""
    transverse_isotropify(t::TensND.AbstractTens{2,3}, n) → TensND.TensTI{2,T,3}

Exact azimuthal average of a 2nd-order tensor about `n`:
`a·nT + b·nₙ + c·w` with `b = n̂ᵀ t n̂`, `a = (tr t − b)/2` and
`c = (w : t)/2` (`w` the in-plane rotation generator `w·p = n̂ × p`).
The antisymmetric in-plane part `c` is preserved (a symmetric TI
parametrization would silently drop it).
"""
function transverse_isotropify(t::TensND.AbstractTens{2, 3}, n)
    arr = TensND.get_array(t)
    T0 = eltype(arr)
    nn = sqrt(n[1]^2 + n[2]^2 + n[3]^2)
    T = promote_type(T0, typeof(nn))
    n̂ = (T(n[1] / nn), T(n[2] / nn), T(n[3] / nn))
    b = zero(T)
    @inbounds for i in 1:3, j in 1:3
        b += n̂[i] * arr[i, j] * n̂[j]
    end
    a = ((arr[1, 1] + arr[2, 2] + arr[3, 3]) - b) / 2
    # c = (1/2) Σ w[i,j] t[i,j],  w[i,j] = ε[i,k,j] n̂[k]
    c = (
        n̂[1] * (arr[3, 2] - arr[2, 3]) +
            n̂[2] * (arr[1, 3] - arr[3, 1]) +
            n̂[3] * (arr[2, 1] - arr[1, 2])
    ) / 2
    return TensND.TensTI{2}(a, b, c, n̂)
end

# ── ALV block helper — azimuthal average of a 6×6 Kelvin-Mandel block ────────

"""
    ti_average_mandel66(M::AbstractMatrix, n) → Matrix (6×6)

Exact azimuthal average about `n` of the minor-symmetric tensor whose 6×6
Kelvin-Mandel matrix (Tensors.jl order) is `M`, returned as a 6×6 matrix.
Used block-wise on ALV Volterra matrices (echoes' `visco_transverse_isotropify`
counterpart).
"""
function ti_average_mandel66(M::AbstractMatrix, n)
    R, n̂ = _axis_frame(n)
    arr_ez = _rotate4(array_from_mandel66(M), R)
    p = _ti_params_from_mandel_ez(mandel66_minor(arr_ez))
    M_ez = _ti_mandel_ez(p)
    # rotate back to the canonical frame: columns of R are (e₁′, e₂′, n̂), so
    # the inverse basis change uses Rᵀ.
    Rt = permutedims(R)
    return mandel66_minor(_rotate4(array_from_mandel66(M_ez), Rt))
end

"""
    iso_average_mandel66(M::AbstractMatrix{T}) → (α, β)

Exact SO(3) average of the minor-symmetric tensor whose Kelvin-Mandel matrix
is `M`: `α = (1/3)Σ_{i,j≤3} M[i,j]`, `β = (tr M − α)/5` (echoes'
`isotropify` closed form).  Returns the `(α, β)` coefficients of `α𝕁 + β𝕂`.
"""
function iso_average_mandel66(M::AbstractMatrix{T}) where {T}
    α = (M[1, 1] + M[1, 2] + M[1, 3] + M[2, 1] + M[2, 2] + M[2, 3] + M[3, 1] + M[3, 2] + M[3, 3]) / 3
    β = (M[1, 1] + M[2, 2] + M[3, 3] + M[4, 4] + M[5, 5] + M[6, 6] - α) / 5
    return (α, β)
end
