# =============================================================================
#  volterra_inverse.jl — block-triangular forward-substitution inverse
#  for discrete Volterra operators.
#
#  A discrete relaxation kernel `R̃` (built by `trapezoidal_matrix`) is a
#  lower-block-triangular matrix of size `(B·n) × (B·n)` with `B = 1` or
#  `B = 6`.  Its Volterra inverse `J̃ = R̃^{-vol}` satisfies
#       R̃ * J̃ = J̃ * R̃ = H 𝟙
#  where the right-hand side is the discrete Heaviside identity (block-
#  diagonal `H_{ii} = I_B`, off-diagonal `0`).  Because `R̃` is lower-
#  block-triangular, `J̃` is too, and we can compute it block-column by
#  block-column via forward substitution :
#       J̃_{ii} = R̃_{ii}^{-1}
#       J̃_{ij} = -R̃_{ii}^{-1} * Σ_{k = j+1}^{i} R̃_{ik} * J̃_{kj}     (j < i)
#
#  This is the Julia counterpart of a reference `get_inv_mat()`,
#  but uses forward-substitution instead of a generic LU and is therefore
#  ~2× faster on the lower-triangular structure.
# =============================================================================

"""
    volterra_inverse(M::AbstractMatrix; block_size::Int = 6) -> Matrix

Compute the Volterra inverse of a lower-block-triangular matrix
`M` of size ``(Bn)\\times(Bn)`` with ``B`` = `block_size`.  The result is also
lower-block-triangular and satisfies `M * volterra_inverse(M)` ``= H\\,\\mathbb 1``
(block-diagonal identity).

`block_size` must be a positive divisor of `size(M, 1)` and `size(M, 2)`,
typically `1` (scalar Volterra kernel) or `6` (4-tensor in Mandel form).

The cost is ``O(B^3 n^2)`` flops via block forward-substitution; each
diagonal block is inverted with the dense `inv(...)` of stdlib (so
``B = 6`` costs only a few hundred flops per block).
"""
function volterra_inverse(M::AbstractMatrix; block_size::Int = 6)
    B = block_size
    B ≥ 1 || throw(ArgumentError("volterra_inverse: block_size must be ≥ 1"))
    sz = size(M, 1)
    sz == size(M, 2) || throw(ArgumentError("volterra_inverse: M must be square"))
    sz % B == 0 || throw(
        ArgumentError(
            "volterra_inverse: matrix size $(sz) not divisible by block_size $(B)"
        )
    )
    n = sz ÷ B
    T = eltype(M)

    # ── Fast path: BlasFloat scalar Volterra → LAPACK trtri ────────────────
    if B == 1 && T <: LinearAlgebra.BlasFloat && M isa StridedMatrix && n ≥ 64
        return Matrix(inv(LowerTriangular(M)))
    end

    inv_M = zeros(T, sz, sz)
    if B == 1
        _volterra_forward_scalar!(inv_M, M, n)
    else
        _volterra_forward_block!(inv_M, M, n, B)
    end
    return inv_M
end

"""
    volterra_inverse!(out::AbstractMatrix, M::AbstractMatrix; block_size = 6) -> out

In-place version of [`volterra_inverse`](@ref).  Writes the Volterra
inverse of `M` directly into the pre-allocated `out` (which must have
the same size as `M`).  Returns `out`.
"""
function volterra_inverse!(
        out::AbstractMatrix, M::AbstractMatrix;
        block_size::Int = 6
    )
    B = block_size
    B ≥ 1 || throw(ArgumentError("volterra_inverse!: block_size must be ≥ 1"))
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("volterra_inverse!: M must be square"))
    size(out) == size(M) ||
        throw(DimensionMismatch("volterra_inverse!: out and M size mismatch"))
    sz % B == 0 ||
        throw(ArgumentError("volterra_inverse!: size $(sz) not divisible by $(B)"))
    n = sz ÷ B
    T = eltype(M)

    fill!(out, zero(eltype(out)))
    # Fast path: BlasFloat scalar Volterra → LAPACK trtri (LowerTriangular).
    if B == 1 && T <: LinearAlgebra.BlasFloat && M isa StridedMatrix && n ≥ 64
        copyto!(out, M)
        LinearAlgebra.LAPACK.trtri!('L', 'N', out)
        # `trtri!` leaves the upper triangle untouched; zero it.
        @inbounds for j in 2:sz, i in 1:(j - 1)
            out[i, j] = zero(eltype(out))
        end
        return out
    end

    if B == 1
        _volterra_forward_scalar!(out, M, n)
    else
        _volterra_forward_block!(out, M, n, B)
    end
    return out
end

# ── Scalar (1×1 block) forward substitution ─────────────────────────────────
# Solve `J̃ = M^{-vol}` block-column by block-column.

@inline function _volterra_forward_scalar!(
        inv_M::AbstractMatrix, M::AbstractMatrix,
        n::Int
    )
    @inbounds for j in 1:n
        # Diagonal: `inv_M[j, j] = 1 / M[j, j]`.
        diag_jj = M[j, j]
        iszero(diag_jj) && throw(SingularException(j))
        inv_M[j, j] = inv(diag_jj)
        # Off-diagonal entries below row `j` in column `j`.
        for i in (j + 1):n
            acc = zero(eltype(inv_M))
            for k in j:(i - 1)
                acc += M[i, k] * inv_M[k, j]
            end
            diag_ii = M[i, i]
            iszero(diag_ii) && throw(SingularException(i))
            inv_M[i, j] = -acc / diag_ii
        end
    end
    return inv_M
end

# ── Block forward substitution (B×B per block, generic B) ───────────────────

@inline function _volterra_forward_block!(
        inv_M::AbstractMatrix, M::AbstractMatrix,
        n::Int, B::Int
    )
    T = eltype(inv_M)
    # Pre-invert the diagonal blocks once.  We need them for both the
    # diagonal placement and the forward-substitution division below.
    diag_inv = Vector{Matrix{T}}(undef, n)
    @inbounds for i in 1:n
        rows = ((i - 1) * B + 1):(i * B)
        diag_inv[i] = inv(Matrix(view(M, rows, rows)))
    end

    @inbounds for j in 1:n
        col_block = ((j - 1) * B + 1):(j * B)

        # Diagonal block.
        rows_j = col_block
        inv_M[rows_j, col_block] = diag_inv[j]

        # Off-diagonal blocks below row `j`.
        for i in (j + 1):n
            rows_i = ((i - 1) * B + 1):(i * B)
            acc = zeros(T, B, B)
            for k in j:(i - 1)
                rows_k = ((k - 1) * B + 1):(k * B)
                M_ik = view(M, rows_i, rows_k)
                inv_kj = view(inv_M, rows_k, col_block)
                # acc += M_ik * inv_kj
                mul!(acc, M_ik, inv_kj, one(T), one(T))
            end
            inv_M[rows_i, col_block] = -diag_inv[i] * acc
        end
    end
    return inv_M
end

# ── Volterra product (block matrix multiplication, lower-triangular safe) ──

"""
    volterra_product(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Discrete Volterra product ``A\\circ B``: a regular matrix multiplication of
two lower-block-triangular matrices, returned as a fresh `Matrix`.
The result is again lower-block-triangular by construction.

Equivalent to `A * B`, exposed as a named function to make the
viscoelastic algebra explicit at call sites.
"""
volterra_product(A::AbstractMatrix, B::AbstractMatrix) = A * B

# ── Volterra divide: T = M · S^{-vol}, computed without forming inv(S) ─────
#
# Algorithmically equivalent to `M * volterra_inverse(S; ...)` but
# avoids the intermediate `inv(S)` whose entries blow up when `S` has
# tiny diagonal (e.g. soft-phase ALV moduli).  Direct forward
# substitution on the linear system `T · S = M`:
#     T[i, j] = (M[i, j] - Σ_{k=j+1..i} T[i, k] · S[k, j]) / S[j, j]
# Each `T[i, j]` is a ratio of similar-magnitude numbers and stays
# well-conditioned even when `S[j, j]` is tiny (the numerator scales
# accordingly).

"""
    volterra_divide(M, S; block_size = 1) -> Matrix

Compute ``T = M\\circ S^{-\\circ}`` (Volterra-divide) by direct block forward
substitution on the linear system ``T\\circ S = M``.  Numerically stable
where `M * volterra_inverse(S)` would lose precision through a huge
intermediate inverse, e.g. for soft-phase moduli (``k, \\mu \\to 0``) or
step-activated `ViscoLaw` kernels in multi-layer ALV recurrences.

`block_size` must divide `size(M, 1)`; typical values are `1`
(scalar Volterra) and `6` (4-tensor Mandel).  The result is
lower-block-triangular if both `M` and `S` are.
"""
function volterra_divide(
        M::AbstractMatrix, S::AbstractMatrix;
        block_size::Int = 1
    )
    B = block_size
    sz = size(M, 1)
    sz == size(M, 2) == size(S, 1) == size(S, 2) ||
        throw(ArgumentError("volterra_divide: M and S must be square of the same size"))
    sz % B == 0 || throw(
        ArgumentError(
            "volterra_divide: matrix size $(sz) not divisible by block_size $(B)"
        )
    )
    n = sz ÷ B
    T = promote_type(eltype(M), eltype(S))

    # ── Fast path: BlasFloat scalar Volterra → BLAS trsm ───────────────────
    if B == 1 && T <: LinearAlgebra.BlasFloat &&
            M isa StridedMatrix && S isa StridedMatrix && n ≥ 64
        Mt = eltype(M) === T ? M : convert(Matrix{T}, M)
        St = eltype(S) === T ? S : convert(Matrix{T}, S)
        return Mt / LowerTriangular(St)
    end

    out = zeros(T, sz, sz)
    if B == 1
        _volterra_divide_scalar!(out, M, S, n)
    else
        _volterra_divide_block!(out, M, S, n, B)
    end
    return out
end

# Iterate columns from right (j = n) to left (j = 1) so that when
# computing T[i, j] we already know T[i, k] for k > j.
@inline function _volterra_divide_scalar!(
        out::AbstractMatrix, M::AbstractMatrix,
        S::AbstractMatrix, n::Int
    )
    @inbounds for j in n:-1:1
        diag = S[j, j]
        iszero(diag) && throw(SingularException(j))
        for i in j:n
            acc = M[i, j]
            for k in (j + 1):i
                acc -= out[i, k] * S[k, j]
            end
            out[i, j] = acc / diag
        end
    end
    return out
end

@inline function _volterra_divide_block!(
        out::AbstractMatrix, M::AbstractMatrix,
        S::AbstractMatrix, n::Int, B::Int
    )
    T = eltype(out)
    diag_inv = Vector{Matrix{T}}(undef, n)
    @inbounds for i in 1:n
        rows = ((i - 1) * B + 1):(i * B)
        diag_inv[i] = inv(Matrix(view(S, rows, rows)))
    end
    @inbounds for j in n:-1:1
        col_block = ((j - 1) * B + 1):(j * B)
        for i in j:n
            row_block = ((i - 1) * B + 1):(i * B)
            acc = Matrix(view(M, row_block, col_block))
            for k in (j + 1):i
                k_block = ((k - 1) * B + 1):(k * B)
                T_ik = view(out, row_block, k_block)
                S_kj = view(S, k_block, col_block)
                mul!(acc, T_ik, S_kj, -one(T), one(T))
            end
            out[row_block, col_block] = acc * diag_inv[j]
        end
    end
    return out
end

# ── Volterra LEFT divide: T = S^{-vol} ∘ M, computed without forming inv(S) ──
#
# Forward substitution on the linear system `S · T = M`, processing rows
# from i = 1 to n.  Equivalent to `volterra_inverse(S) * M` but avoids
# the intermediate dense inverse and stays stable for soft-phase moduli.
#
# This is the form needed by the Hervé–Zaoui closed-form interface
# transitions, where the algebra is `T = M_b^{-1} · M_a` (with the
# Volterra inverse on the LEFT of the numerator).  In a non-commutative
# Volterra algebra (non-uniform time grid, multiple Maxwell time
# constants), the right and left divides differ.
#
# Algorithm (S, M, T all lower-block-triangular of the same size):
#     T[i, j] = S[i,i]^{-1} · (M[i, j] - Σ_{k=j..i-1} S[i, k] · T[k, j])
# iterating j over columns and i over rows i ≥ j.

"""
    volterra_left_divide(S, M; block_size = 1) -> Matrix

Compute ``T = S^{-\\circ}\\circ M`` (Volterra LEFT-divide) by direct forward
substitution on the linear system ``S\\circ T = M``.

Use this rather than [`volterra_divide`](@ref) when the closed-form
algebra requires the inverse on the LEFT of the numerator (e.g. the
Hervé–Zaoui bulk/shear transition matrices, where
``\\mathbf{T} = \\mathbf{M}_b^{-1}\\,\\mathbf{M}_a``).  On a non-uniform time grid Volterra trapezoidal
matrices do **not** form a commutative algebra, so the order matters
and right-vs-left divides give different results.

`block_size` must divide `size(M, 1)`; typical values are `1` (scalar
Volterra) and `6` (4-tensor Mandel).  Both arguments must be lower-
block-triangular of the same size; the result is lower-block-triangular.
"""
function volterra_left_divide(
        S::AbstractMatrix, M::AbstractMatrix;
        block_size::Int = 1
    )
    B = block_size
    sz = size(M, 1)
    sz == size(M, 2) == size(S, 1) == size(S, 2) ||
        throw(ArgumentError("volterra_left_divide: M and S must be square of the same size"))
    sz % B == 0 || throw(
        ArgumentError(
            "volterra_left_divide: matrix size $(sz) not divisible by block_size $(B)"
        )
    )
    n = sz ÷ B
    T = promote_type(eltype(M), eltype(S))

    # ── Fast path: BlasFloat scalar Volterra → BLAS trsm ───────────────────
    if B == 1 && T <: LinearAlgebra.BlasFloat &&
            M isa StridedMatrix && S isa StridedMatrix && n ≥ 64
        Mt = eltype(M) === T ? M : convert(Matrix{T}, M)
        St = eltype(S) === T ? S : convert(Matrix{T}, S)
        return LowerTriangular(St) \ Mt
    end

    out = zeros(T, sz, sz)
    if B == 1
        _volterra_left_divide_scalar!(out, M, S, n)
    else
        _volterra_left_divide_block!(out, M, S, n, B)
    end
    return out
end

# Iterate columns j = 1..n from left to right, then for each column
# walk rows i = j..n via standard forward substitution.
@inline function _volterra_left_divide_scalar!(
        out::AbstractMatrix,
        M::AbstractMatrix,
        S::AbstractMatrix, n::Int
    )
    @inbounds for j in 1:n
        for i in j:n
            diag_i = S[i, i]
            iszero(diag_i) && throw(SingularException(i))
            acc = M[i, j]
            for k in j:(i - 1)
                acc -= S[i, k] * out[k, j]
            end
            out[i, j] = acc / diag_i
        end
    end
    return out
end

@inline function _volterra_left_divide_block!(
        out::AbstractMatrix,
        M::AbstractMatrix,
        S::AbstractMatrix,
        n::Int, B::Int
    )
    T = eltype(out)
    diag_inv = Vector{Matrix{T}}(undef, n)
    @inbounds for i in 1:n
        rows = ((i - 1) * B + 1):(i * B)
        diag_inv[i] = inv(Matrix(view(S, rows, rows)))
    end
    @inbounds for j in 1:n
        col_block = ((j - 1) * B + 1):(j * B)
        for i in j:n
            row_block = ((i - 1) * B + 1):(i * B)
            acc = Matrix(view(M, row_block, col_block))
            for k in j:(i - 1)
                k_block = ((k - 1) * B + 1):(k * B)
                S_ik = view(S, row_block, k_block)
                T_kj = view(out, k_block, col_block)
                mul!(acc, S_ik, T_kj, -one(T), one(T))
            end
            out[row_block, col_block] = diag_inv[i] * acc
        end
    end
    return out
end
