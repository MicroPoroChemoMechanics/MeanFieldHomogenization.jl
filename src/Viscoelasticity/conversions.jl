# =============================================================================
#  conversions.jl — convert between a `(6n×6n)` block matrix and per-symmetry
#  scalar parameter matrices `n×n`.
#
#  An isotropic 4-tensor in Mandel form is described by two scalars
#  `(α, β)` (with `α = 3K`, `β = 2μ`).  An iso block matrix in ALV is
#  fully described by the two scalar Volterra matrices `α(t,t')` and
#  `β(t,t')` of size `n×n`.
#
#  Conversions are needed at two places :
#   * extracting the iso parameters from a 6n×6n viscoelastic block
#     matrix in order to apply the time-space decoupling formula
#     for the Hill kernel ;
#   * reassembling a 6n×6n matrix after applying a scalar Volterra
#     operation (e.g. a Volterra inverse) on each component.
# =============================================================================

# The entries `(a, b)` of every 6×6 block of a 6n×6n block matrix, as an n×n
# matrix: rows a, a + 6, … and columns b, b + 6, ….
_slots(M::AbstractMatrix, a::Int, b::Int) = view(M, a:6:size(M, 1), b:6:size(M, 2))

"""
    iso_params_from_blocks(M) -> (α::Matrix, β::Matrix)

Decompose a ``6n\\times 6n`` block matrix whose every ``6\\times 6`` block is an
**isotropic** 4-tensor in Mandel form into the two scalar parameter
matrices `α` and `β`, both of size ``n\\times n``.  The block layout is

```math
\\mathbf{M}_{ij} = \\alpha_{ij}\\,\\mathrm{Mat}(\\mathbb{J}) + \\beta_{ij}\\,\\mathrm{Mat}(\\mathbb{K})
```

where ``\\mathrm{Mat}(\\mathbb{J}) = \\tfrac{1}{3}\\,\\mathbf{e}_1\\mathbf{e}_1^{\\!T}`` (``\\mathbf{e}_1`` the Mandel-1 unit) augmented
with the symmetric off-diagonal part ``\\tfrac{1}{3}`` for the upper ``3\\times 3`` block,
and ``\\mathrm{Mat}(\\mathbb{K}) = \\mathrm{Mat}(\\mathbb{I}) - \\mathrm{Mat}(\\mathbb{J})``.

``\\alpha_{ij}`` and ``\\beta_{ij}`` are extracted from the diagonal entry ``(\\mathbf{M}_{ij})_{11}``
and the (4,4) Mandel-shear entry of each block:

```math
\\alpha_{ij} = (\\mathbf{M}_{ij})_{11} + 2\\,(\\mathbf{M}_{ij})_{12},
\\qquad
\\beta_{ij} = (\\mathbf{M}_{ij})_{44} .
```
"""
function iso_params_from_blocks(M::AbstractMatrix)
    sz = size(M, 1)
    sz == size(M, 2) || throw(ArgumentError("iso_params_from_blocks: M must be square"))
    sz % 6 == 0 || throw(ArgumentError("iso_params_from_blocks: size $(sz) not divisible by 6"))
    # α = M[1,1] + 2 M[1,2] (= 3K), β = M[4,4] (= 2μ in Mandel).
    return _slots(M, 1, 1) .+ 2 .* _slots(M, 1, 2), copy(_slots(M, 4, 4))
end

"""
    iso_blocks_from_params(α::Matrix, β::Matrix) -> Matrix

Inverse of [`iso_params_from_blocks`](@ref): build a ``6n\\times 6n`` block
matrix whose every block is the iso 4-tensor ``\\alpha_{ij}\\,\\mathbb{J} + \\beta_{ij}\\,\\mathbb{K}``
in Mandel form.

Both `α` and `β` must be ``n\\times n`` and have a common element type.

Note: a `kron(α, 𝕁_M) + kron(β, 𝕂_M)` formulation was experimented
(P3.3.1) and reverted — the two extra ``6n\\times 6n`` intermediate
allocations made it ~3× slower than this single-pass scalar loop,
which writes each entry exactly once.
"""
function iso_blocks_from_params(α::AbstractMatrix, β::AbstractMatrix)
    size(α) == size(β) ||
        throw(ArgumentError("iso_blocks_from_params: α and β must have same size"))
    n = size(α, 1)
    n == size(α, 2) ||
        throw(ArgumentError("iso_blocks_from_params: α must be square"))
    # The type of a ratio: integer parameters give a float block (the entries
    # are divided), a dual or symbolic one is kept (`float(Sym)` is Float64).
    T = typeof(one(promote_type(eltype(α), eltype(β))) / 2)
    return _add_iso_blocks!(zeros(T, 6 * n, 6 * n), one(T), α, β)
end

# M += w · (the 6n×6n matrix of the isotropic blocks α, β), in place: each entry
# is written once, and the weight multiplies the entry as it is, so w = 1 into
# zeros gives the entries exactly.
function _add_iso_blocks!(M::AbstractMatrix, w, α::AbstractMatrix, β::AbstractMatrix)
    T = eltype(M)
    @inbounds for j in axes(α, 2), i in axes(α, 1)
        a = T(α[i, j])
        b = T(β[i, j])
        r, c = 6 * (i - 1), 6 * (j - 1)
        diag_top = (a + 2b) / 3
        offdiag_top = (a - b) / 3
        for l in 1:3, k in 1:3
            M[r + k, c + l] += w * ((k == l) ? diag_top : offdiag_top)
        end
        for k in 4:6
            M[r + k, c + k] += w * b
        end
    end
    return M
end

# =============================================================================
#  TI (transversely isotropic) — Walpole basis with axis n = e_z (canonical)
# =============================================================================
#
#  A TI 4-tensor in the Walpole basis is described by 6 scalars
#  `(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)`.  In the canonical axis n = e₃ the
#  6×6 Mandel block has the structure (zero entries omitted):
#
#      M[1,1] = M[2,2] = (ℓ₂ + ℓ₅)/2
#      M[1,2] = M[2,1] = (ℓ₂ − ℓ₅)/2
#      M[3,3] = ℓ₁
#      M[1,3] = M[2,3] = ℓ₄/√2     (W₄ = (nT⊗nₙ)/√2 → column 3)
#      M[3,1] = M[3,2] = ℓ₃/√2     (W₃ = (nₙ⊗nT)/√2 → row 3)
#      M[4,4] = M[5,5] = ℓ₆        (out-of-plane shear)
#      M[6,6] = ℓ₅                 (in-plane shear)
#
#  Major symmetry of the elastic tensor implies ℓ₃ = ℓ₄.  In ALV the
#  Volterra product of two major-symmetric TI tensors is generally NOT
#  major-symmetric (because the synthetic Walpole 2×2 matrices may not
#  commute when the time grid is non-uniform), so the 6-parameter form
#  is used internally even when inputs are major-symmetric.
#
#  REPRESENTABILITY LIMIT — this 6-parameter TI block form captures the
#  ℓ₃ ≠ ℓ₄ major-asymmetry but NOT the two antisymmetric azimuthal
#  couplings (ℓ₇, ℓ₈ of the full 8-dim axially-invariant space; the t13 /
#  t16 patterns of `Core.transverse_isotropify`).  Those appear when a
#  concentration tensor is azimuthally averaged block-by-block on the ALV
#  side (`homogenize_alv._ti_project_blocks`): such a result is kept as a
#  FULL 6n×6n matrix, not routed through this 6-parameter closure.  The
#  6-parameter form here is the fast path for the dedicated structured
#  TI ALV pipeline (`ti_schemes_alv.jl`), where all operands are genuine
#  TI(ez) stiffnesses/compliances (ℓ₇ = ℓ₈ = 0).
# =============================================================================

# The (row, column) slots of the 12 orthotropic parameters in a 6×6 block: the
# normal 3×3 block row-major, then the shear diagonal.
const _ORTHO_SLOTS = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6))

# ── Isotropy of a block matrix ──────────────────────────────────────────────

"""
    _is_iso_block(M::AbstractMatrix; tol = 0) -> Bool

Heuristic: return `true` if the ``6n\\times 6n`` block matrix `M` is in iso
form (each ``6\\times 6`` block is an iso 4-tensor in Mandel form).  The check
verifies the canonical iso pattern ``(\\mathbf{M}_{ij})_{kk} = (\\alpha + 2\\beta)/3``
(diag), ``(\\alpha - \\beta)/3`` (off-diag in the 1..3 block), ``\\beta`` on the 4..6
diagonal, and zero elsewhere — to within absolute tolerance `tol`
times the maximum modulus of `M`.

Used by `homogenize_alv` to dispatch to the iso fast path automatically.

Allocation-free: extracts ``(\\alpha, \\beta)`` from the canonical block entries
on the fly without building the parameter matrices.
"""
function _is_iso_block(M::AbstractMatrix; tol::Real = 1.0e-12)
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
        # Extract (α, β) from the canonical entries of block (i, j).
        a = M[r + 1, c + 1] + 2 * M[r + 1, c + 2]
        b = M[r + 4, c + 4]
        diag_top = (a + 2b) / 3
        offdiag_top = (a - b) / 3
        # Upper-left 3×3 block: diag = (α+2β)/3, off-diag = (α-β)/3.
        for k in 1:3, l in 1:3
            expected = (k == l) ? diag_top : offdiag_top
            abs(M[r + k, c + l] - expected) ≤ abstol || return false
        end
        # Cross blocks (1..3 × 4..6 and 4..6 × 1..3) must be zero.
        for k in 1:3, l in 4:6
            abs(M[r + k, c + l]) ≤ abstol || return false
        end
        for k in 4:6, l in 1:3
            abs(M[r + k, c + l]) ≤ abstol || return false
        end
        # Lower-right 3×3 block: diagonal = β, off-diag = 0.
        for k in 4:6, l in 4:6
            expected = (k == l) ? b : zero(b)
            abs(M[r + k, c + l] - expected) ≤ abstol || return false
        end
    end
    return true
end
