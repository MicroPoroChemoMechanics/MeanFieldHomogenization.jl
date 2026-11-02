# =============================================================================
#  trapezoidal.jl — discrete representation of a Stieltjes / Volterra
#  integral on a time grid via the trapezoidal rule of sanahuja2013.
#
#  Given a kernel `f(t, t')` (scalar, tensor, or 6×6 matrix), build the
#  lower-block-triangular matrix `M` of size `(B·n) × (B·n)` such that
#  the discrete relation `y = M · x` approximates the Stieltjes integral
#       y(t_i) = ∫_0^{t_i} f(t_i, τ) dx(τ)
#  on the time grid `t_1 ≤ t_2 ≤ … ≤ t_n` (ECHOES `visco_law.cpp:80–103`, which
#  evaluates the law at the instants themselves),
#  with fₖ = f(t_i, t_k) on row i:
#       M[1, 1]   = f₁                                   (the jump from 0 at t_1)
#       M[i, i]   = (fᵢ₋₁ + fᵢ)/2                         (i ≥ 2)
#       M[i, 1]   = (f₁ - f₂)/2                           (i ≥ 2)
#       M[i, j]   = (fⱼ₋₁ - fⱼ₊₁)/2                       (1 < j < i)
#       M[i, j]   = 0                                     (j > i)
#  An instant given twice, t_{k-1} = t_k, is a jump: its first node is the
#  left limit t⁻ (the law is evaluated just before t), and the interval of
#  zero length between the two nodes is weighted by f(t_i, t⁺) instead of a
#  mean (`_trapezoidal_weight`), which is the Stieltjes integral of a jump.
# =============================================================================

"""
    trapezoidal_matrix(law::AbstractViscoLaw, times::AbstractVector{<:Real}) -> Matrix

Build the discrete ``(Bn)\\times(Bn)`` lower-block-triangular matrix
representing the Stieltjes integral
``y(t_i) = \\int_0^{t_i} f(t_i, \\tau)\\,\\mathrm{d}x(\\tau)``
on the time grid `times` ``= (t_1, \\dots, t_n)`` using the trapezoidal
rule of [sanahuja2013](@citet).  ``B = 1`` for scalar-valued kernels and ``B = 6``
for 4-tensor- or ``6\\times 6``-matrix-valued kernels (Mandel form).

The instants are in increasing order; one given twice is a jump, as in [`TimeGrid`](@ref). The exact
block layout is described at the top of `trapezoidal.jl`. The output is type-stable in
the element type returned by `visco_eval(law, ...)`.
"""
function trapezoidal_matrix(law::AbstractViscoLaw, times::AbstractVector{<:Real})
    TimeGrid(times)   # in increasing order
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

# ── Doubled instants and evaluation instants ────────────────────────────────

# A grid may hold an instant twice, t_{k-1} = t_k: the first node stands for the left limit
# t⁻ and the second for t⁺, so that a history or a kernel jumps there. The interval between
# them has zero length; node 1 ends the "interval" of the jump from zero at the first instant.
@inline _is_jump(times, k) = k == 1 || times[k] == times[k - 1]

# The instants at which a law or a history is evaluated at the nodes: the instant itself,
# except at the first node of a doubled instant, evaluated just before it (its left limit).
# A function with a jump there must be right-continuous, `t ≥ a ? … : …`.
function _evaluation_instants(times::AbstractVector)
    τ = float.(times)
    for k in 2:length(times)
        _is_jump(times, k) && (τ[k - 1] = _left_of(τ[k - 1]))
    end
    return τ
end
_left_of(t::AbstractFloat) = prevfloat(t)
_left_of(t::ForwardDiff.Dual{T}) where {T} = ForwardDiff.Dual{T}(_left_of(ForwardDiff.value(t)), ForwardDiff.partials(t))

# ── The weights ─────────────────────────────────────────────────────────────

# On row i, with fₖ = f(tᵢ, τₖ), the response is y_i = Σₖ wₖ (xₖ - xₖ₋₁) (x₀ = 0), the
# interval ending at node k weighted by wₖ = (fₖ₋₁ + fₖ)/2, or by fₖ when it has zero length
# (a jump, integrated against the kernel just after it). The coefficient of xⱼ is
# wⱼ - wⱼ₊₁ (wᵢ on the diagonal), written below without forming the wₖ, so that a grid
# without doubled instants gives the classical weights to the last bit. `f` holds tuples of
# blocks; every case is a map over them.
@inline function _trapezoidal_weight(f, times, i, j, half)
    left = _is_jump(times, j)
    j == i && return left ? f[i] : map((a, b) -> (a + b) * half, f[i - 1], f[i])
    right = _is_jump(times, j + 1)
    left && right && return map(-, f[j], f[j + 1])
    left && return map((a, c) -> (a - c) * half, f[j], f[j + 1])
    right && return map((a, b, c) -> (a + b) * half - c, f[j - 1], f[j], f[j + 1])
    return map((a, c) -> (a - c) * half, f[j - 1], f[j + 1])
end

# ── Filling the operators ───────────────────────────────────────────────────

# Several operators at once, one per block of a symmetry class: `value(i, k)` is the kernel
# at the pair of nodes (i, k), `blocks_of` turns it into the tuple of its blocks, of sizes
# `sizes` (`Val`s), and block m of every value goes into `Ms[m]`. A per-row cache evaluates
# each pair once. Threading the outer loop was tried and reverted: its overhead exceeded the
# gain for cheap isotropic laws (50 % slower at n = 200, even on one thread).
function _fill_trapezoidal_classes!(
        Ms::Tuple, value::V, times::AbstractVector, sizes::Tuple, blocks_of::F
    ) where {V, F}
    n = length(times)
    n == 0 && return Ms
    half = inv(eltype(first(Ms))(2))
    b₁ = blocks_of(value(1, 1))
    cache = Vector{typeof(b₁)}(undef, n)
    @inbounds for i in 1:n
        for k in 1:i
            cache[k] = blocks_of(value(i, k))
        end
        for j in 1:i
            # `map` over the tuples, unrolled: the blocks have different sizes.
            map((M, x, s) -> _put_block!(M, i, j, x, s), Ms, _trapezoidal_weight(cache, times, i, j, half), sizes)
        end
    end
    return Ms
end

# The law at the evaluation instants of the nodes.
_node_values(law::AbstractViscoLaw, times::AbstractVector) =
    (τ = _evaluation_instants(times); (i, k) -> visco_eval(law, τ[i], τ[k]))

# One operator of blocks B×B: 1 for a scalar kernel, 6 for a 4-tensor or a 6×6 Mandel matrix,
# 3 for a 2-tensor or a 3×3 matrix. `convert_sample` turns a kernel value into its block.
_fill_trapezoidal_scalar!(M::AbstractMatrix, law::AbstractViscoLaw, times::AbstractVector) =
    _fill_trapezoidal_blocks!(M, law, times, Val(1), x -> SMatrix{1, 1}(x))
@inline _fill_trapezoidal_blocks!(
    M::AbstractMatrix, law::AbstractViscoLaw, times::AbstractVector, ::Val{B}, convert_sample::F
) where {B, F} = (_fill_trapezoidal_classes!((M,), _node_values(law, times), times, (Val(B),), x -> (convert_sample(x),)); M)

# Block (i, j) of `M` set to the B×B value `x`, entry by entry: measured faster than a
# broadcast over views (1.3 times at B = 3).
@inline function _put_block!(M, i, j, x, ::Val{B}) where {B}
    r, c = B * (i - 1), B * (j - 1)
    @inbounds for l in 1:B, k in 1:B
        M[r + k, c + l] = x[k, l]
    end
    return M
end

# Convert a sample (4-tensor or 6×6 matrix) to a 6×6 Mandel array,
# preserving element type.
@inline _to_mandel(sample::AbstractMatrix) = sample
@inline function _to_mandel(sample::TensND.AbstractTens{4, 3})
    return TensND.mandel(TensND.canonical_components(sample); symmetrize = true)
end

_fill_trapezoidal_tensor!(M::AbstractMatrix, law::AbstractViscoLaw, times::AbstractVector) =
    _fill_trapezoidal_blocks!(M, law, times, Val(6), _to_mandel)
_fill_trapezoidal_mandel!(M::AbstractMatrix, law::AbstractViscoLaw, times::AbstractVector) =
    _fill_trapezoidal_blocks!(M, law, times, Val(6), identity)
