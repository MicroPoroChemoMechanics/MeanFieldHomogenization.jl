# =============================================================================
#  ortho_schemes_alv.jl — orthotropic-symmetry fast paths for ALV schemes.
#
#  When **every** ALV block matrix in a homogenization pipeline is an
#  ortho 4-tensor with the **same canonical material frame**
#  `(e₁, e₂, e₃)` (each `(6×6)` Mandel block is parametrized by 9 entries
#  of a full 3×3 "normal" block plus 3 entries of a shear-diagonal block),
#  the algebra collapses to:
#
#    * a `(3n)×(3n)` block-Volterra problem on the 3×3 normal subspace,
#    * three `n×n` scalar Volterra problems on the diagonal shears
#      `(o₄, o₅, o₆)`.
#
#  Storage:    12 · n²   entries vs   36 · n²    for the full `(6n×6n)` form
#              (3× cheaper)
#  Inverse:   `(3n)×(3n)` block-LU + 3 scalar inverses ≪ `(6n)×(6n)` LU
#  Product:   9 n×n GEMMs (normal) + 3 n×n GEMMs (shears)
#             vs the full 6n×6n GEMM
#
#  Inclusion ladder: iso ⊂ TI ⊂ ortho ⊂ generic aniso.  When all phases
#  are iso, the iso fast path (`iso_schemes_alv.jl`) is used; when all
#  are TI with the same axis, the TI fast path; ortho is selected
#  automatically when at least one phase breaks the TI in-plane
#  rotation symmetry but every phase shares the canonical material frame
#  `(e₁, e₂, e₃)`.
# =============================================================================

# ── Ortho 12-tuple primitives ───────────────────────────────────────────────

"""
    _ortho_identity(n, T) -> NTuple{12, Matrix{T}}

Ortho form of the ``6n\\times 6n`` block-diagonal identity matrix.  The 9
normal entries form the ``3\\times 3`` identity (``o_1 = o_5 = o_9 = \\mathbb 1``, others zero)
and the three shears are each ``\\mathbb 1``.
"""
@inline function _ortho_identity(n::Int, T::Type)
    Iₙ = Matrix{T}(LinearAlgebra.I, n, n)
    Zₙ = zeros(T, n, n)
    return (
        Iₙ, copy(Zₙ), copy(Zₙ),
        copy(Zₙ), copy(Iₙ), copy(Zₙ),
        copy(Zₙ), copy(Zₙ), copy(Iₙ),
        copy(Iₙ), copy(Iₙ), copy(Iₙ),
    )
end

"""
    _ortho_add!(acc, c, a) -> acc

In-place ortho scalar AXPY: `acc[k] .+= c · a[k]` for ``k = 1, \\dots, 12``.
"""
@inline function _ortho_add!(
        acc::NTuple{12, <:Matrix}, c::Real,
        a::NTuple{12, <:Matrix}
    )
    @inbounds for k in 1:12
        @. acc[k] += c * a[k]
    end
    return acc
end

"""
    _ortho_prod(a, b) -> NTuple{12, Matrix}

Ortho Volterra product ``M_a\\circ M_b``.  The normal ``3\\times 3`` block follows the
standard ``3\\times 3`` matrix product on Volterra entries; the shears multiply
component-wise.
"""
@inline function _ortho_prod(a::NTuple{12, <:Matrix}, b::NTuple{12, <:Matrix})
    # Normal 3×3 block product: c_{ij} = Σ_k a_{ik} · b_{kj}
    c1 = a[1] * b[1] + a[2] * b[4] + a[3] * b[7]
    c2 = a[1] * b[2] + a[2] * b[5] + a[3] * b[8]
    c3 = a[1] * b[3] + a[2] * b[6] + a[3] * b[9]
    c4 = a[4] * b[1] + a[5] * b[4] + a[6] * b[7]
    c5 = a[4] * b[2] + a[5] * b[5] + a[6] * b[8]
    c6 = a[4] * b[3] + a[5] * b[6] + a[6] * b[9]
    c7 = a[7] * b[1] + a[8] * b[4] + a[9] * b[7]
    c8 = a[7] * b[2] + a[8] * b[5] + a[9] * b[8]
    c9 = a[7] * b[3] + a[8] * b[6] + a[9] * b[9]
    # Shear diagonal: independent scalar products
    c10 = a[10] * b[10]
    c11 = a[11] * b[11]
    c12 = a[12] * b[12]
    return (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12)
end

"""
    _ortho_prod!(out, a, b) -> out

In-place ortho Volterra product.  27 `mul!` for the normal block + 3
for the shears.  No allocation when `out`'s components are sized
correctly.
"""
@inline function _ortho_prod!(
        out::NTuple{12, <:Matrix},
        a::NTuple{12, <:Matrix},
        b::NTuple{12, <:Matrix}
    )
    T = eltype(out[1])
    one_T = one(T)
    # Normal 3×3 block c_{ij} = Σ_k a_{ik} · b_{kj}
    @inbounds for ii in 1:3, jj in 1:3
        idx_c = 3 * (ii - 1) + jj            # (ii, jj) → flat index 1..9
        idx_a1 = 3 * (ii - 1) + 1
        idx_b1 = 0 * 3 + jj                   # k = 1 → (1, jj) flat = jj
        mul!(out[idx_c], a[idx_a1], b[idx_b1])
        for k in 2:3
            idx_a = 3 * (ii - 1) + k         # (ii, k)
            idx_b = 3 * (k - 1) + jj         # (k, jj)
            mul!(out[idx_c], a[idx_a], b[idx_b], one_T, one_T)
        end
    end
    # Shears
    mul!(out[10], a[10], b[10])
    mul!(out[11], a[11], b[11])
    mul!(out[12], a[12], b[12])
    return out
end

# ── Pack / unpack the normal 3×3 part as a (3n)×(3n) block-Volterra ───────
#
# Time-interleaved layout: block (i, j) of size 3×3 in the packed matrix
# corresponds to the normal sub-block of M[block(i), block(j)].  Because
# every component is lower-triangular in (i, j) (Volterra causality),
# the packed matrix is block-lower-triangular with `block_size = 3`,
# ready for `volterra_inverse(_; block_size = 3)` and
# `volterra_left_divide(_, _; block_size = 3)`.

@inline function _ortho_pack_normal(a::NTuple{12, <:Matrix})
    n = size(a[1], 1)
    T = promote_type(
        eltype(a[1]), eltype(a[2]), eltype(a[3]),
        eltype(a[4]), eltype(a[5]), eltype(a[6]),
        eltype(a[7]), eltype(a[8]), eltype(a[9])
    )
    M = zeros(T, 3 * n, 3 * n)
    @inbounds for i in 1:n, j in 1:n
        r = 3 * (i - 1)
        c = 3 * (j - 1)
        M[r + 1, c + 1] = a[1][i, j]
        M[r + 1, c + 2] = a[2][i, j]
        M[r + 1, c + 3] = a[3][i, j]
        M[r + 2, c + 1] = a[4][i, j]
        M[r + 2, c + 2] = a[5][i, j]
        M[r + 2, c + 3] = a[6][i, j]
        M[r + 3, c + 1] = a[7][i, j]
        M[r + 3, c + 2] = a[8][i, j]
        M[r + 3, c + 3] = a[9][i, j]
    end
    return M
end

@inline function _ortho_unpack_normal(M::AbstractMatrix, n::Int)
    T = eltype(M)
    o = ntuple(_ -> zeros(T, n, n), 9)
    @inbounds for i in 1:n, j in 1:n
        r = 3 * (i - 1)
        c = 3 * (j - 1)
        o[1][i, j] = M[r + 1, c + 1]
        o[2][i, j] = M[r + 1, c + 2]
        o[3][i, j] = M[r + 1, c + 3]
        o[4][i, j] = M[r + 2, c + 1]
        o[5][i, j] = M[r + 2, c + 2]
        o[6][i, j] = M[r + 2, c + 3]
        o[7][i, j] = M[r + 3, c + 1]
        o[8][i, j] = M[r + 3, c + 2]
        o[9][i, j] = M[r + 3, c + 3]
    end
    return o
end

"""
    _ortho_inv(a) -> NTuple{12, Matrix}

Ortho Volterra inverse.  Inverts the ``3\\times 3`` normal block via a ``3n\\times 3n``
block-Volterra inverse and the three scalar shears independently.
"""
function _ortho_inv(a::NTuple{12, <:Matrix})
    n = size(a[1], 1)
    M_pack = _ortho_pack_normal(a)
    M_inv = volterra_inverse(M_pack; block_size = 3)
    o_norm = _ortho_unpack_normal(M_inv, n)
    o10 = volterra_inverse(a[10]; block_size = 1)
    o11 = volterra_inverse(a[11]; block_size = 1)
    o12 = volterra_inverse(a[12]; block_size = 1)
    return (
        o_norm[1], o_norm[2], o_norm[3],
        o_norm[4], o_norm[5], o_norm[6],
        o_norm[7], o_norm[8], o_norm[9],
        o10, o11, o12,
    )
end

"""
    _ortho_left_divide(S, M) -> NTuple{12, Matrix}

Ortho form of the Volterra left-divide ``T = S^{-\\circ}\\circ M``.  Solves the
normal ``3\\times 3`` part as a ``3n\\times 3n`` block-Volterra system and the three
scalar shears independently.
"""
function _ortho_left_divide(
        S::NTuple{12, <:Matrix},
        M::NTuple{12, <:Matrix}
    )
    n = size(S[1], 1)
    S_pack = _ortho_pack_normal(S)
    M_pack = _ortho_pack_normal(M)
    T_pack = volterra_left_divide(S_pack, M_pack; block_size = 3)
    o_norm = _ortho_unpack_normal(T_pack, n)
    o10 = volterra_left_divide(S[10], M[10]; block_size = 1)
    o11 = volterra_left_divide(S[11], M[11]; block_size = 1)
    o12 = volterra_left_divide(S[12], M[12]; block_size = 1)
    return (
        o_norm[1], o_norm[2], o_norm[3],
        o_norm[4], o_norm[5], o_norm[6],
        o_norm[7], o_norm[8], o_norm[9],
        o10, o11, o12,
    )
end

# ── Ortho form detection and conversion ────────────────────────────────────

"""
    _is_ortho_block(M; tol = 1e-12) -> Bool

Heuristic: return `true` if the ``6n\\times 6n`` block matrix `M` is in
ortho form (each ``6\\times 6`` block is an ortho 4-tensor with canonical axes).
The check verifies that every entry outside the orthotropic Mandel
support pattern is below ``\\mathrm{tol}\\cdot\\max|M|`` in absolute value.

Allocation-free: scans each block once and tests for off-pattern
entries directly, without building intermediate parameter matrices.
"""
function _is_ortho_block(M::AbstractMatrix; tol::Real = 1.0e-12)
    sz = size(M, 1)
    sz == size(M, 2) || return false
    sz % 6 == 0 || return false
    n = sz ÷ 6
    iszero(n) && return true
    scale = max(maximum(abs, M), one(real(eltype(M))))
    abstol = tol * scale
    @inbounds for i in 1:n, j in 1:n
        r = 6 * (i - 1)
        c = 6 * (j - 1)
        # Ortho support: the 3×3 normal block (rows/cols 1..3) and the
        # 3×3 shear-diagonal block (rows/cols 4..6 only on the diagonal).
        # Everything else must be zero up to abstol.
        # Cross blocks (1..3 × 4..6 and 4..6 × 1..3): zero.
        for k in 1:3, l in 4:6
            abs(M[r + k, c + l]) ≤ abstol || return false
        end
        for k in 4:6, l in 1:3
            abs(M[r + k, c + l]) ≤ abstol || return false
        end
        # Lower-right 3×3 block: only diagonal entries allowed.
        for k in 4:6, l in 4:6
            if k != l
                abs(M[r + k, c + l]) ≤ abstol || return false
            end
        end
    end
    return true
end

"""
    _ortho_pair(M; axes) -> NTuple{12, Matrix}

Extract the 12 ortho parameter matrices from a ``6n\\times 6n`` block matrix.
Wrapper around [`ortho_params_from_blocks`](@ref).
"""
@inline _ortho_pair(
    M::AbstractMatrix;
    axes::NTuple{3, NTuple{3}} = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
) =
    ortho_params_from_blocks(M; axes = axes)

"""
    _ortho_blocks(o; axes) -> Matrix

Reassemble a ``6n\\times 6n`` ortho block matrix from the 12-tuple of Volterra
parameter matrices.  Wrapper around [`ortho_blocks_from_params`](@ref).
"""
@inline _ortho_blocks(
    o::NTuple{12, <:AbstractMatrix};
    axes::NTuple{3, NTuple{3}} = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
) =
    ortho_blocks_from_params(o; axes = axes)

"""
    _iso_to_ortho(αβ::Tuple) -> NTuple{12, Matrix}

Convert an iso ``(\\alpha, \\beta)`` pair into the ortho 12-tuple in the canonical
material frame.  Iso ``\\alpha\\,\\mathbb{J} + \\beta\\,\\mathbb{K}`` corresponds to:
  * normal block (``3\\times 3``): diag ``= (\\alpha + 2\\beta)/3``, off-diag ``= (\\alpha - \\beta)/3``
  * shear diagonal:    ``(\\beta, \\beta, \\beta)``.
"""
function _iso_to_ortho(αβ::Tuple)
    α, β = αβ
    diag = α ./ 3 .+ (2 // 3) .* β
    off = (α .- β) ./ 3
    return (
        copy(diag), copy(off), copy(off),
        copy(off), copy(diag), copy(off),
        copy(off), copy(off), copy(diag),
        copy(β), copy(β), copy(β),
    )
end

"""
    _ti_to_ortho(ℓ::NTuple{6}) -> NTuple{12, Matrix}

Convert a TI 6-tuple ``(\\ell_1, \\ell_2, \\ell_3, \\ell_4, \\ell_5, \\ell_6)`` (axis ``\\underline{e}_3``) into the ortho
12-tuple in the canonical material frame.  Cross-check helper used by
the inclusion-ladder tests.
"""
function _ti_to_ortho(ℓ::NTuple{6, <:AbstractMatrix})
    s2 = sqrt(2)
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆ = ℓ
    diag_in = (ℓ₂ .+ ℓ₅) ./ 2
    off_in = (ℓ₂ .- ℓ₅) ./ 2
    o3 = ℓ₄ ./ s2          # M[1,3]
    o7 = ℓ₃ ./ s2          # M[3,1]
    o6 = ℓ₄ ./ s2          # M[2,3]
    o8 = ℓ₃ ./ s2          # M[3,2]
    return (
        copy(diag_in), copy(off_in), copy(o3),
        copy(off_in), copy(diag_in), copy(o6),
        copy(o7), copy(o8), copy(ℓ₁),
        copy(ℓ₆), copy(ℓ₆), copy(ℓ₅),
    )
end

# ── Ortho scheme implementations ────────────────────────────────────────────

"""
    voigt_alv_ortho(o_phases, fractions) -> NTuple{12, Matrix}

Ortho-form Voigt bound: ``o_k^{\\mathrm{hom}} = \\sum_i f_i\\,o_{k,i}`` for each component ``k``.
"""
function voigt_alv_ortho(o_phases::AbstractVector, fractions::AbstractVector)
    length(o_phases) == length(fractions) ||
        throw(ArgumentError("voigt_alv_ortho: phase counts mismatch"))
    isempty(o_phases) && throw(ArgumentError("voigt_alv_ortho: at least one phase required"))
    n = size(o_phases[1][1], 1)
    T = eltype(fractions)
    for o in o_phases
        T = promote_type(T, eltype(o[1]))
    end
    eff = ntuple(_ -> zeros(T, n, n), 12)
    @inbounds for r in eachindex(o_phases)
        _ortho_add!(eff, fractions[r], o_phases[r])
    end
    return eff
end

"""
    reuss_alv_ortho(o_phases, fractions) -> NTuple{12, Matrix}

Ortho-form Reuss bound: invert per-phase compliances in ortho form,
average, invert back.
"""
function reuss_alv_ortho(o_phases::AbstractVector, fractions::AbstractVector)
    length(o_phases) == length(fractions) ||
        throw(ArgumentError("reuss_alv_ortho: phase counts mismatch"))
    o_inv_phases = [_ortho_inv(o) for o in o_phases]
    o_inv_eff = voigt_alv_ortho(o_inv_phases, fractions)
    return _ortho_inv(o_inv_eff)
end

"""
    dilute_concentration_alv_ortho(o_E, o_0, o_P) -> NTuple{12, Matrix}

Ortho-form dilute concentration
``\\widetilde{\\mathbb{A}}^{\\mathrm{dil}} = (H\\,\\mathbb{I} + \\widetilde{\\mathbb{P}}\\circ\\Delta\\widetilde{\\mathbb{C}})^{-\\circ}``,
``\\Delta\\widetilde{\\mathbb{C}} = \\widetilde{\\mathbb{C}}^{\\mathcal{E}} - \\widetilde{\\mathbb{C}}_0``.
"""
function dilute_concentration_alv_ortho(
        o_E::NTuple{12, <:Matrix},
        o_0::NTuple{12, <:Matrix},
        o_P::NTuple{12, <:Matrix}
    )
    n = size(o_E[1], 1)
    T = promote_type(eltype(o_E[1]), eltype(o_0[1]), eltype(o_P[1]))
    Δ = ntuple(k -> o_E[k] .- o_0[k], 12)
    PΔ = _ortho_prod(o_P, Δ)
    Id = _ortho_identity(n, T)
    sum_o = ntuple(k -> Id[k] .+ PΔ[k], 12)
    return _ortho_inv(sum_o)
end

"""
    dilute_contribution_alv_ortho(o_E, o_0, o_P) -> NTuple{12, Matrix}

Ortho-form dilute contribution ``\\widetilde{\\mathbb{N}} = \\Delta\\widetilde{\\mathbb{C}}\\circ\\widetilde{\\mathbb{A}}^{\\mathrm{dil}}``.
"""
function dilute_contribution_alv_ortho(
        o_E::NTuple{12, <:Matrix},
        o_0::NTuple{12, <:Matrix},
        o_P::NTuple{12, <:Matrix}
    )
    A_dil = dilute_concentration_alv_ortho(o_E, o_0, o_P)
    Δ = ntuple(k -> o_E[k] .- o_0[k], 12)
    return _ortho_prod(Δ, A_dil)
end

"""
    dilute_alv_ortho(o_0, contribs_ortho, fractions) -> NTuple{12, Matrix}

Ortho-form Dilute scheme: ``\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0 + \\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i``.
"""
function dilute_alv_ortho(
        o_0::NTuple{12, <:Matrix},
        contribs_ortho::AbstractVector,
        fractions::AbstractVector
    )
    length(contribs_ortho) == length(fractions) ||
        throw(ArgumentError("dilute_alv_ortho: phase counts mismatch"))
    out = ntuple(k -> copy(o_0[k]), 12)
    @inbounds for r in eachindex(contribs_ortho)
        _ortho_add!(out, fractions[r], contribs_ortho[r])
    end
    return out
end

"""
    dilute_dual_alv_ortho(o_0, contribs_compliance_ortho, fractions)
        -> NTuple{12, Matrix}

Ortho-form DiluteDual: invert to compliance ortho form, average, invert
back.
"""
function dilute_dual_alv_ortho(
        o_0::NTuple{12, <:Matrix},
        contribs_compliance_ortho::AbstractVector,
        fractions::AbstractVector
    )
    o_J_0 = _ortho_inv(o_0)
    o_J_eff = dilute_alv_ortho(o_J_0, contribs_compliance_ortho, fractions)
    return _ortho_inv(o_J_eff)
end

"""
    mori_tanaka_alv_ortho(o_0, A_duts_ortho, contribs_ortho, fractions, f_M)
        -> NTuple{12, Matrix}

Ortho-form Mori-Tanaka:

```math
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0
  + \\Bigl(\\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i\\Bigr)\\circ
    \\Bigl(f_0\\,H\\,\\mathbb{I} + \\sum_j f_j\\,\\widetilde{\\mathbb{A}}_j^{\\mathrm{dil}}\\Bigr)^{-\\circ},
```

all in ortho form.
"""
function mori_tanaka_alv_ortho(
        o_0::NTuple{12, <:Matrix},
        A_duts_ortho::AbstractVector,
        contribs_ortho::AbstractVector,
        fractions::AbstractVector, f_M::Real
    )
    length(A_duts_ortho) == length(contribs_ortho) == length(fractions) ||
        throw(ArgumentError("mori_tanaka_alv_ortho: phase counts mismatch"))
    n = size(o_0[1], 1)
    T = promote_type(eltype(o_0[1]), eltype(fractions), typeof(f_M))
    for o in A_duts_ortho
        T = promote_type(T, eltype(o[1]))
    end
    for o in contribs_ortho
        T = promote_type(T, eltype(o[1]))
    end
    Id = _ortho_identity(n, T)
    num = ntuple(_ -> zeros(T, n, n), 12)
    den = ntuple(k -> T(f_M) .* Id[k], 12)
    @inbounds for r in eachindex(A_duts_ortho)
        _ortho_add!(num, fractions[r], contribs_ortho[r])
        _ortho_add!(den, fractions[r], A_duts_ortho[r])
    end
    factor = _ortho_left_divide(den, num)
    return ntuple(k -> o_0[k] .+ factor[k], 12)
end

"""
    maxwell_alv_ortho(o_0, contribs_ortho, fractions, o_H_0) -> NTuple{12, Matrix}

Ortho-form Maxwell scheme.
"""
function maxwell_alv_ortho(
        o_0::NTuple{12, <:Matrix},
        contribs_ortho::AbstractVector,
        fractions::AbstractVector,
        o_H_0::NTuple{12, <:Matrix}
    )
    length(contribs_ortho) == length(fractions) ||
        throw(ArgumentError("maxwell_alv_ortho: phase counts mismatch"))
    n = size(o_0[1], 1)
    T = promote_type(eltype(o_0[1]), eltype(fractions), eltype(o_H_0[1]))
    for o in contribs_ortho
        T = promote_type(T, eltype(o[1]))
    end
    Id = _ortho_identity(n, T)
    Σ = ntuple(_ -> zeros(T, n, n), 12)
    @inbounds for r in eachindex(contribs_ortho)
        _ortho_add!(Σ, fractions[r], contribs_ortho[r])
    end
    HΣ = _ortho_prod(o_H_0, Σ)
    inv_arg = ntuple(k -> Id[k] .- HΣ[k], 12)
    factor = _ortho_prod(Σ, _ortho_inv(inv_arg))
    return ntuple(k -> o_0[k] .+ factor[k], 12)
end
