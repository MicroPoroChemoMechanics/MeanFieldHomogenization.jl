# =============================================================================
#  trapezoidal.jl — discrete representation of a Stieltjes / Volterra
#  integral on a time grid via the trapezoidal rule of sanahuja2013.
#
#  Given a kernel `f(t, t')` (scalar, tensor, or 6×6 matrix), build the
#  lower-block-triangular matrix `M` of size `(B·n) × (B·n)` such that
#  the discrete relation `y = M · x` approximates the Stieltjes integral
#       y(t_i) = ∫_{t_0}^{t_i} f(t_i, τ) dx(τ)
#  on the time grid `t_0 < t_1 < … < t_{n-1}` (ECHOES `visco_law.cpp:80–103`).
#
#  Block-row index `i = 1..n` corresponds to time `t_{i-1}` (1-based).
#  Block-column index `j = 1..n` to the trapezoidal weight at time
#  `t_{j-1}`.  The block matrix layout (1-based) is:
#       M[1, 1]   = f(t_0, t_0)
#       M[i, i]   = 0.5 (f(t_{i-1}, t_{i-2}) + f(t_{i-1}, t_{i-1}))   (i ≥ 2)
#       M[i, 1]   = 0.5 (f(t_{i-1}, t_0)     - f(t_{i-1}, t_1))       (i ≥ 2)
#       M[i, j]   = 0.5 (f(t_{i-1}, t_{j-2}) - f(t_{i-1}, t_j))       (1 < j < i)
#       M[i, j]   = 0                                                  (j > i)
# =============================================================================

"""
    trapezoidal_matrix(law::AbstractViscoLaw, times::AbstractVector{<:Real}) -> Matrix

Build the discrete ``(Bn)\\times(Bn)`` lower-block-triangular matrix
representing the Stieltjes integral
``y(t_i) = \\int_{t_0}^{t_i} f(t_i, \\tau)\\,\\mathrm{d}x(\\tau)``
on the time grid `times` ``= (t_0, \\dots, t_{n-1})`` using the trapezoidal
rule of [sanahuja2013](@citet).  ``B = 1`` for scalar-valued kernels and ``B = 6``
for 4-tensor- or ``6\\times 6``-matrix-valued kernels (Mandel form).

The exact block layout is described at the top of `trapezoidal.jl`.
The output is type-stable in the element type returned by
`visco_eval(law, ...)`.
"""
function trapezoidal_matrix(law::AbstractViscoLaw, times::AbstractVector{<:Real})
    n = length(times)
    n ≥ 1 || throw(ArgumentError("trapezoidal_matrix: times must be non-empty"))
    sample = visco_eval(law, times[1], times[1])
    return _trapezoidal_dispatch(law, times, sample)
end

# Scalar specialization: M is `n × n`.
function _trapezoidal_dispatch(
        law::AbstractViscoLaw, times::AbstractVector{<:Real},
        ::Number
    )
    n = length(times)
    T = typeof(visco_eval(law, times[1], times[1]))
    M = zeros(T, n, n)
    _fill_trapezoidal_scalar!(M, law, times)
    return M
end

# Tensor specialization (TensND.AbstractTens{4,3}): M is `6n × 6n`.
function _trapezoidal_dispatch(
        law::AbstractViscoLaw, times::AbstractVector{<:Real},
        ::TensND.AbstractTens{4, 3}
    )
    n = length(times)
    sample = visco_eval(law, times[1], times[1])
    T = eltype(sample)
    M = zeros(T, 6 * n, 6 * n)
    _fill_trapezoidal_tensor!(M, law, times)
    return M
end

# Order-2 tensor specialization (TensND.AbstractTens{2,3}): M is `3n × 3n`.
function _trapezoidal_dispatch(
        law::AbstractViscoLaw, times::AbstractVector{<:Real},
        ::TensND.AbstractTens{2, 3}
    )
    n = length(times)
    sample = visco_eval(law, times[1], times[1])
    T = eltype(sample)
    M = zeros(T, 3 * n, 3 * n)
    _fill_trapezoidal_order2_tens!(M, law, times)
    return M
end

# Matrix specialization: dispatches by size (3×3 → 3n×3n, 6×6 → 6n×6n).
function _trapezoidal_dispatch(
        law::AbstractViscoLaw, times::AbstractVector{<:Real},
        sample::AbstractMatrix
    )
    n = length(times)
    T = eltype(sample)
    if size(sample) == (3, 3)
        M = zeros(T, 3 * n, 3 * n)
        _fill_trapezoidal_order2_mat!(M, law, times)
        return M
    elseif size(sample) == (6, 6)
        M = zeros(T, 6 * n, 6 * n)
        _fill_trapezoidal_mandel!(M, law, times)
        return M
    else
        throw(ArgumentError("trapezoidal_matrix: only 3×3 (order-2) or 6×6 (order-4 Mandel) matrices are supported"))
    end
end

# ── Scalar case ─────────────────────────────────────────────────────────────

@inline function _fill_trapezoidal_scalar!(
        M::AbstractMatrix, law::AbstractViscoLaw,
        times::AbstractVector
    )
    n = length(times)
    n == 0 && return M
    T = eltype(M)
    # Per-row cache: cache[k] = visco_eval(law, times[i], times[k]) for k = 1..i.
    # Halves the number of `visco_eval` calls (n(n+1)/2 instead of ~n(n+1)).
    # Note: threading the outer loop with `Threads.@threads :static` was
    # experimented and reverted — the macro overhead exceeds the gain for
    # cheap iso ViscoLaws (a 50% slowdown was observed at n = 200 even
    # with a single thread).  Re-enable behind an opt-in kwarg if/when
    # heavy kernels (Mittag-Leffler, numerical integral) are dominant.
    cache = Vector{T}(undef, n)
    @inbounds begin
        cache[1] = visco_eval(law, times[1], times[1])
        M[1, 1] = cache[1]
        for i in 2:n
            for k in 1:i
                cache[k] = visco_eval(law, times[i], times[k])
            end
            M[i, i] = (cache[i - 1] + cache[i]) / 2
            M[i, 1] = (cache[1] - cache[2]) / 2
            for j in 2:(i - 1)
                M[i, j] = (cache[j - 1] - cache[j + 1]) / 2
            end
        end
    end
    return M
end

# ── Tensor / Mandel case (block-by-block) ───────────────────────────────────

# Convert a sample (4-tensor or 6×6 matrix) to a 6×6 Mandel array,
# preserving element type.
@inline _to_mandel(sample::AbstractMatrix) = sample
@inline function _to_mandel(sample::TensND.AbstractTens{4, 3})
    return TensND.mandel(TensND.canonical_components(sample); symmetrize = true)
end

# The block cases, B×B blocks: 6 for a 4-tensor or a 6×6 Mandel matrix, 3 for a
# 2-tensor or a 3×3 matrix. `convert_sample` turns a kernel value into its B×B
# block.
@inline _fill_trapezoidal_blocks!(
    M::AbstractMatrix, law::AbstractViscoLaw, times::AbstractVector, ::Val{B}, convert_sample::F
) where {B, F} = (_fill_trapezoidal_classes!((M,), law, times, (Val(B),), x -> (convert_sample(x),)); M)

# Several operators from one law at once, one per block of a symmetry class:
# `blocks_of` turns a kernel value into the tuple of its blocks, of sizes
# `sizes` (`Val`s), and block k of every value goes into `Ms[k]`. The law is
# evaluated once per pair of instants. A per-row cache halves the number of
# `visco_eval` calls; the weights are the layout at the top of this file.
# (Threading reverted: see note in `_fill_trapezoidal_scalar!`.)
function _fill_trapezoidal_classes!(
        Ms::Tuple, law::AbstractViscoLaw, times::AbstractVector, sizes::Tuple, blocks_of::F
    ) where {F}
    n = length(times)
    n == 0 && return Ms
    half = inv(eltype(first(Ms))(2))
    # `map` over the tuples, unrolled: the blocks have different sizes, so a
    # zipped loop would not infer.
    put!(i, j, a, b, op::O) where {O} = (map((M, x, y, s) -> _put_block!(M, i, j, x, y, op, half, s), Ms, a, b, sizes); nothing)
    b₁ = blocks_of(visco_eval(law, times[1], times[1]))
    cache = Vector{typeof(b₁)}(undef, n)
    cache[1] = b₁
    put!(1, 1, b₁, b₁, +)
    @inbounds for i in 2:n
        for k in 1:i
            cache[k] = blocks_of(visco_eval(law, times[i], times[k]))
        end
        put!(i, 1, cache[1], cache[2], -)
        for j in 2:(i - 1)
            put!(i, j, cache[j - 1], cache[j + 1], -)
        end
        put!(i, i, cache[i - 1], cache[i], +)
    end
    return Ms
end

# Block (i, j) of `M` set to `op(a, b) / 2`, by a loop over its B×B entries:
# measured faster than a broadcast over views (1.3 times at B = 3). `(a + a)/2`
# writes the first block exactly.
@inline function _put_block!(M, i, j, a, b, op::F, half, ::Val{B}) where {F, B}
    r, c = B * (i - 1), B * (j - 1)
    @inbounds for l in 1:B, k in 1:B
        M[r + k, c + l] = op(a[k, l], b[k, l]) * half
    end
    return M
end

_fill_trapezoidal_tensor!(M::AbstractMatrix, law::AbstractViscoLaw, times::AbstractVector) =
    _fill_trapezoidal_blocks!(M, law, times, Val(6), _to_mandel)
_fill_trapezoidal_mandel!(M::AbstractMatrix, law::AbstractViscoLaw, times::AbstractVector) =
    _fill_trapezoidal_blocks!(M, law, times, Val(6), identity)
