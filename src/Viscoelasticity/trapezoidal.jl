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
    trapezoidal_matrix(law::ViscoLaw, times::AbstractVector{<:Real}) -> Matrix

Build the discrete `(B·n) × (B·n)` lower-block-triangular matrix
representing the Stieltjes integral
``y(t_i) = ∫_{t_0}^{t_i} f(t_i, τ) \\, dx(τ)``
on the time grid `times = (t_0, …, t_{n-1})` using the trapezoidal
rule of [sanahuja2013](@cite).  `B = 1` for scalar-valued kernels and `B = 6`
for 4-tensor- or `6×6`-matrix-valued kernels (Mandel form).

The exact block layout is described at the top of `trapezoidal.jl`.
The output is type-stable in the element type returned by
`visco_eval(law, ...)`.
"""
function trapezoidal_matrix(law::ViscoLaw, times::AbstractVector{<:Real})
    n = length(times)
    n ≥ 1 || throw(ArgumentError("trapezoidal_matrix: times must be non-empty"))
    sample = visco_eval(law, times[1], times[1])
    return _trapezoidal_dispatch(law, times, sample)
end

# Scalar specialization: M is `n × n`.
function _trapezoidal_dispatch(
        law::ViscoLaw, times::AbstractVector{<:Real},
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
        law::ViscoLaw, times::AbstractVector{<:Real},
        ::TensND.AbstractTens{4, 3}
    )
    n = length(times)
    sample = visco_eval(law, times[1], times[1])
    T = eltype(TensND.get_array(sample))
    M = zeros(T, 6 * n, 6 * n)
    _fill_trapezoidal_tensor!(M, law, times)
    return M
end

# Order-2 tensor specialization (TensND.AbstractTens{2,3}): M is `3n × 3n`.
function _trapezoidal_dispatch(
        law::ViscoLaw, times::AbstractVector{<:Real},
        ::TensND.AbstractTens{2, 3}
    )
    n = length(times)
    sample = visco_eval(law, times[1], times[1])
    T = eltype(TensND.get_array(sample))
    M = zeros(T, 3 * n, 3 * n)
    _fill_trapezoidal_order2_tens!(M, law, times)
    return M
end

# Matrix specialization: dispatches by size (3×3 → 3n×3n, 6×6 → 6n×6n).
function _trapezoidal_dispatch(
        law::ViscoLaw, times::AbstractVector{<:Real},
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
        M::AbstractMatrix, law::ViscoLaw,
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
    return _tens_to_mandel66(sample)
end

# Convert a TensND 4-tensor in {3,3,3,3} array form to a 6×6 Mandel matrix.
# Mandel convention: σ_4 = √2 σ_23, σ_5 = √2 σ_13, σ_6 = √2 σ_12.
function _tens_to_mandel66(C::TensND.AbstractTens{4, 3})
    arr = TensND.get_array(C)
    T = eltype(arr)
    M = zeros(T, 6, 6)
    sq2 = sqrt(T(2))
    voigt = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
    for I in 1:6, J in 1:6
        i, j = voigt[I]
        k, l = voigt[J]
        scale = (I ≥ 4 ? sq2 : one(T)) * (J ≥ 4 ? sq2 : one(T))
        M[I, J] = arr[i, j, k, l] * scale
    end
    return M
end

# Fill the (6×6)-block-structured matrix from a 4-tensor kernel.
# Per-row cache halves the number of `visco_eval` + `_to_mandel` calls.
# (Threading reverted: see note in `_fill_trapezoidal_scalar!`.)
@inline function _fill_trapezoidal_tensor!(
        M::AbstractMatrix, law::ViscoLaw,
        times::AbstractVector
    )
    n = length(times)
    n == 0 && return M
    T = eltype(M)
    cache = Vector{Matrix{T}}(undef, n)
    @inbounds begin
        cache[1] = _to_mandel(visco_eval(law, times[1], times[1]))
        _set_block!(M, 1, 1, cache[1])
        for i in 2:n
            for k in 1:i
                cache[k] = _to_mandel(visco_eval(law, times[i], times[k]))
            end
            _fill_row_blocks_from_cache!(M, cache, i, T)
        end
    end
    return M
end

@inline function _fill_trapezoidal_mandel!(
        M::AbstractMatrix, law::ViscoLaw,
        times::AbstractVector
    )
    n = length(times)
    n == 0 && return M
    T = eltype(M)
    cache = Vector{Matrix{T}}(undef, n)
    @inbounds begin
        cache[1] = copy(visco_eval(law, times[1], times[1]))
        _set_block!(M, 1, 1, cache[1])
        for i in 2:n
            for k in 1:i
                cache[k] = visco_eval(law, times[i], times[k])
            end
            _fill_row_blocks_from_cache!(M, cache, i, T)
        end
    end
    return M
end

# Place row `i` blocks into `M` from the cache `cache[1..i]` of
# `_to_mandel(visco_eval(law, times[i], times[k]))`.  Writes blocks
# directly entry-by-entry, no temporary `(a ± b) / 2` allocation.
@inline function _fill_row_blocks_from_cache!(
        M::AbstractMatrix,
        cache::Vector{<:AbstractMatrix},
        i::Int, ::Type{T}
    ) where {T}
    half = inv(T(2))
    @inbounds begin
        # j = 1 : (cache[1] - cache[2]) / 2
        c1, c2 = cache[1], cache[2]
        for kk in 1:6, ll in 1:6
            M[6 * (i - 1) + kk, ll] = (c1[kk, ll] - c2[kk, ll]) * half
        end
        # 1 < j < i : (cache[j-1] - cache[j+1]) / 2
        for j in 2:(i - 1)
            cm, cp = cache[j - 1], cache[j + 1]
            for kk in 1:6, ll in 1:6
                M[6 * (i - 1) + kk, 6 * (j - 1) + ll] =
                    (cm[kk, ll] - cp[kk, ll]) * half
            end
        end
        # j = i : (cache[i-1] + cache[i]) / 2  (note + sign)
        cm, cp = cache[i - 1], cache[i]
        for kk in 1:6, ll in 1:6
            M[6 * (i - 1) + kk, 6 * (i - 1) + ll] =
                (cm[kk, ll] + cp[kk, ll]) * half
        end
    end
    return M
end

# Place a 6×6 block into the (i, j)-th block of the 6n×6n matrix.
@inline function _set_block!(M::AbstractMatrix, i::Int, j::Int, block::AbstractMatrix)
    rows = (6 * (i - 1) + 1):(6 * i)
    cols = (6 * (j - 1) + 1):(6 * j)
    @inbounds M[rows, cols] = block
    return M
end
