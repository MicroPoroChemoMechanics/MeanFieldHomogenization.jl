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
    n = sz ÷ 6
    T = eltype(M)
    α = zeros(T, n, n)
    β = zeros(T, n, n)
    @inbounds for i in 1:n, j in 1:n
        r1 = 6 * (i - 1) + 1
        c1 = 6 * (j - 1) + 1
        c2 = 6 * (j - 1) + 2
        # α = M[1,1] + 2 M[1,2] (= 3K), β = M[4,4] (= 2μ in Mandel).
        α[i, j] = M[r1, c1] + 2 * M[r1, c2]
        β[i, j] = M[r1 + 3, c1 + 3]
    end
    return α, β
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
    T = promote_type(eltype(α), eltype(β))
    M = zeros(T, 6 * n, 6 * n)
    @inbounds for i in 1:n, j in 1:n
        a = T(α[i, j])
        b = T(β[i, j])
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        diag_top = (a + 2b) / 3
        offdiag_top = (a - b) / 3
        for k in 1:3, l in 1:3
            M[rows[k], cols[l]] = (k == l) ? diag_top : offdiag_top
        end
        for k in 4:6
            M[rows[k], cols[k]] = b
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

const _SQRT2_ALV = sqrt(2)

"""
    ti_params_from_blocks(M; axis = (0, 0, 1))
        -> NTuple{6, Matrix{T}}

Decompose a ``6n\\times 6n`` block matrix whose every ``6\\times 6`` block is a TI
4-tensor with axis ``\\underline{n}`` (only ``\\underline{n} = \\underline{e}_3`` is currently supported) into
the six scalar Walpole parameter matrices ``(\\ell_1, \\ell_2, \\ell_3, \\ell_4, \\ell_5, \\ell_6)``,
each of size ``n\\times n``.
"""
function ti_params_from_blocks(
        M::AbstractMatrix;
        axis::NTuple{3} = (0.0, 0.0, 1.0)
    )
    axis == (0.0, 0.0, 1.0) ||
        throw(ArgumentError("ti_params_from_blocks: only axis = e₃ is supported"))
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("ti_params_from_blocks: M must be square"))
    sz % 6 == 0 ||
        throw(ArgumentError("ti_params_from_blocks: size $(sz) not divisible by 6"))
    n = sz ÷ 6
    T = eltype(M)
    ℓ₁ = zeros(T, n, n); ℓ₂ = zeros(T, n, n)
    ℓ₃ = zeros(T, n, n); ℓ₄ = zeros(T, n, n)
    ℓ₅ = zeros(T, n, n); ℓ₆ = zeros(T, n, n)
    s2 = T(_SQRT2_ALV)
    @inbounds for i in 1:n, j in 1:n
        r = 6 * (i - 1)
        c = 6 * (j - 1)
        ℓ₁[i, j] = M[r + 3, c + 3]
        ℓ₂[i, j] = M[r + 1, c + 1] + M[r + 1, c + 2]
        ℓ₅[i, j] = M[r + 1, c + 1] - M[r + 1, c + 2]
        ℓ₃[i, j] = s2 * M[r + 3, c + 1]
        ℓ₄[i, j] = s2 * M[r + 1, c + 3]
        ℓ₆[i, j] = M[r + 4, c + 4]
    end
    return (ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)
end

"""
    ti_blocks_from_params(ℓ::NTuple{6, AbstractMatrix}; axis = (0, 0, 1))
        -> Matrix{T}

Inverse of [`ti_params_from_blocks`](@ref): rebuild a ``6n\\times 6n`` block
matrix whose every ``6\\times 6`` block has the TI Mandel structure with axis
``\\underline{n} = \\underline{e}_3`` and Walpole coefficients ``((\\ell_1)_{ij}, \\dots, (\\ell_6)_{ij})``.
"""
function ti_blocks_from_params(
        ℓ::NTuple{6, <:AbstractMatrix};
        axis::NTuple{3} = (0.0, 0.0, 1.0)
    )
    axis == (0.0, 0.0, 1.0) ||
        throw(ArgumentError("ti_blocks_from_params: only axis = e₃ is supported"))
    n = size(ℓ[1], 1)
    @inbounds for k in 1:6
        size(ℓ[k]) == (n, n) ||
            throw(ArgumentError("ti_blocks_from_params: all ℓᵢ must be n×n"))
    end
    T = promote_type(map(eltype, ℓ)...)
    M = zeros(T, 6 * n, 6 * n)
    s2 = T(_SQRT2_ALV)
    @inbounds for i in 1:n, j in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        ℓ₁_ij = T(ℓ[1][i, j]); ℓ₂_ij = T(ℓ[2][i, j])
        ℓ₃_ij = T(ℓ[3][i, j]); ℓ₄_ij = T(ℓ[4][i, j])
        ℓ₅_ij = T(ℓ[5][i, j]); ℓ₆_ij = T(ℓ[6][i, j])
        block_diag_in = (ℓ₂_ij + ℓ₅_ij) / 2
        block_off_in = (ℓ₂_ij - ℓ₅_ij) / 2
        # In-plane block (1:2, 1:2)
        M[rows[1], cols[1]] = block_diag_in
        M[rows[1], cols[2]] = block_off_in
        M[rows[2], cols[1]] = block_off_in
        M[rows[2], cols[2]] = block_diag_in
        # Axial-axial diagonal
        M[rows[3], cols[3]] = ℓ₁_ij
        # Axial-transverse coupling (W₃: row 3, W₄: col 3)
        M[rows[3], cols[1]] = ℓ₃_ij / s2
        M[rows[3], cols[2]] = ℓ₃_ij / s2
        M[rows[1], cols[3]] = ℓ₄_ij / s2
        M[rows[2], cols[3]] = ℓ₄_ij / s2
        # Out-of-plane shears (Mandel 4 = 23, 5 = 13)
        M[rows[4], cols[4]] = ℓ₆_ij
        M[rows[5], cols[5]] = ℓ₆_ij
        # In-plane shear (Mandel 6 = 12)
        M[rows[6], cols[6]] = ℓ₅_ij
    end
    return M
end

# =============================================================================
#  ORTHO (orthotropic) — material-frame Mandel block, axes (e₁, e₂, e₃)
# =============================================================================
#
#  An orthotropic 4-tensor in its material frame `(e₁, e₂, e₃)` has the
#  block-diagonal Mandel structure (see `TensND.TensOrtho`):
#
#      M_block = [[A_norm,  0       ];
#                 [0      ,  D_shear ]]
#
#  where `A_norm` is the (3 × 3) "normal" block carrying the elastic
#  constants `(C₁₁, C₁₂, C₁₃, C₂₂, C₂₃, C₃₃)` (major-symmetric for an
#  elastic stiffness, but **not** preserved by a Volterra product of two
#  ortho ALV operators — the closure subspace is the full 3×3) and
#  `D_shear = diag(2C₄₄, 2C₅₅, 2C₆₆)` is the (3 × 3) shear-diagonal block
#  in Mandel form (factors of 2 absorb the engineering-strain
#  convention).
#
#  In ALV the Volterra closure is therefore parametrized by 12 scalar
#  Volterra `n × n` matrices :
#    * 9 entries of the full unsymmetric normal block `(o[1..9])` —
#      laid out row-major in a `(3 × 3)` array `(o₁₁, o₁₂, o₁₃,
#                                                o₂₁, o₂₂, o₂₃,
#                                                o₃₁, o₃₂, o₃₃)`,
#    * 3 entries of the shear diagonal `(o[10..12]) = (o₄, o₅, o₆)`
#      corresponding to the Mandel-position-4 / -5 / -6 diagonal.
#
#  Every ortho 4-tensor with the same canonical material frame
#  `(e₁, e₂, e₃)` is closed under Volterra product, inverse, and
#  left-divide.  Iso ⊂ TI ⊂ ortho ⊂ generic aniso.
# =============================================================================

"""
    ortho_params_from_blocks(M; axes = ((1,0,0),(0,1,0),(0,0,1)))
        -> NTuple{12, Matrix{T}}

Decompose a ``6n\\times 6n`` block matrix whose every ``6\\times 6`` block is an ortho
4-tensor with material frame ``(\\underline{e}_1, \\underline{e}_2, \\underline{e}_3)`` (only the canonical frame is
currently supported) into the 12 scalar parameter matrices

```math
(o_{11}, o_{12}, o_{13}, o_{21}, o_{22}, o_{23}, o_{31}, o_{32}, o_{33}, o_4, o_5, o_6),
```

each of size ``n\\times n``, where the ``3\\times 3`` normal block of the Mandel form is
stored row-major in entries 1..9 and the 3 shear-diagonal Mandel
entries ``(M_{44}, M_{55}, M_{66})`` map to ``(o_4, o_5, o_6)``.

Note that closure under Volterra product does **not** preserve major
symmetry (``o_{12} \\ne o_{21}`` in general), so the full 9-entry normal block is
needed even for major-symmetric inputs.
"""
function ortho_params_from_blocks(
        M::AbstractMatrix;
        axes::NTuple{3, NTuple{3}} = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) ||
        throw(ArgumentError("ortho_params_from_blocks: only canonical axes are supported"))
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("ortho_params_from_blocks: M must be square"))
    sz % 6 == 0 ||
        throw(ArgumentError("ortho_params_from_blocks: size $(sz) not divisible by 6"))
    n = sz ÷ 6
    T = eltype(M)
    o = ntuple(_ -> zeros(T, n, n), 12)
    @inbounds for i in 1:n, j in 1:n
        r = 6 * (i - 1)
        c = 6 * (j - 1)
        o[1][i, j] = M[r + 1, c + 1]
        o[2][i, j] = M[r + 1, c + 2]
        o[3][i, j] = M[r + 1, c + 3]
        o[4][i, j] = M[r + 2, c + 1]
        o[5][i, j] = M[r + 2, c + 2]
        o[6][i, j] = M[r + 2, c + 3]
        o[7][i, j] = M[r + 3, c + 1]
        o[8][i, j] = M[r + 3, c + 2]
        o[9][i, j] = M[r + 3, c + 3]
        o[10][i, j] = M[r + 4, c + 4]
        o[11][i, j] = M[r + 5, c + 5]
        o[12][i, j] = M[r + 6, c + 6]
    end
    return o
end

"""
    ortho_blocks_from_params(o::NTuple{12, AbstractMatrix};
                              axes = ((1,0,0),(0,1,0),(0,0,1)))
        -> Matrix{T}

Inverse of [`ortho_params_from_blocks`](@ref): rebuild a ``6n\\times 6n`` block
matrix whose every ``6\\times 6`` block has the ortho Mandel structure with axes
``(\\underline{e}_1, \\underline{e}_2, \\underline{e}_3)``.  All off-block-diagonal entries are zero (block 1..3 ↔
block 4..6 coupling is forbidden by orthotropic symmetry).
"""
function ortho_blocks_from_params(
        o::NTuple{12, <:AbstractMatrix};
        axes::NTuple{3, NTuple{3}} = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) ||
        throw(ArgumentError("ortho_blocks_from_params: only canonical axes are supported"))
    n = size(o[1], 1)
    @inbounds for k in 1:12
        size(o[k]) == (n, n) ||
            throw(ArgumentError("ortho_blocks_from_params: all components must be n×n"))
    end
    T = promote_type(map(eltype, o)...)
    M = zeros(T, 6 * n, 6 * n)
    @inbounds for i in 1:n, j in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        # Normal 3×3 block (full, no symmetry assumed)
        M[rows[1], cols[1]] = T(o[1][i, j])
        M[rows[1], cols[2]] = T(o[2][i, j])
        M[rows[1], cols[3]] = T(o[3][i, j])
        M[rows[2], cols[1]] = T(o[4][i, j])
        M[rows[2], cols[2]] = T(o[5][i, j])
        M[rows[2], cols[3]] = T(o[6][i, j])
        M[rows[3], cols[1]] = T(o[7][i, j])
        M[rows[3], cols[2]] = T(o[8][i, j])
        M[rows[3], cols[3]] = T(o[9][i, j])
        # Shear diagonal (Mandel 4, 5, 6)
        M[rows[4], cols[4]] = T(o[10][i, j])
        M[rows[5], cols[5]] = T(o[11][i, j])
        M[rows[6], cols[6]] = T(o[12][i, j])
    end
    return M
end
