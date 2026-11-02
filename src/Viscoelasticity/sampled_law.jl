# =============================================================================
#  sampled_law.jl — a law known by its kernel at the instants of a time grid,
#  and the two kernel transforms of a phase that appears or disappears.
#
#  `homogenize_alv` returns the relaxation OPERATOR of a composite on a grid,
#  the trapezoidal block matrix of `trapezoidal.jl`. To use that composite as a
#  phase of the next scale, its KERNEL is needed at the grid instants: each row
#  of the trapezoidal matrix is an invertible linear map of the kernel values
#  on that row, inverted here by sums and differences only.
# =============================================================================

"""
    SampledViscoLaw(times, samples; mode = :relaxation)

A viscoelastic law known at the instants of the grid `times`: `samples[i, j]` is its
kernel at `(times[i], times[j])` for `j ≤ i`, a scalar, a tensor or a block (a ``6\\times 6``
Mandel matrix at order 4, a ``3\\times 3`` matrix at order 2). At an instant given twice
(a jump, see [`TimeGrid`](@ref)) the first sample is the left limit.

`law(t, t′)` is exact at the instants of the grid and interpolated between them, as the
trapezoidal rule integrates it (see [`operator_law`](@ref)); ``t`` and ``t'`` lie in
``[t_1, t_n]``, and the kernel vanishes for ``t' > t``.

It is what [`operator_law`](@ref) returns, and what lets the composite of one scale be a phase
of the next in [`homogenize_alv`](@ref), on the same grid.
"""
struct SampledViscoLaw{V <: AbstractVector{<:Real}, S <: AbstractMatrix} <: AbstractViscoLaw
    times::V
    samples::S
    mode::Symbol
    function SampledViscoLaw(times::V, samples::S; mode::Symbol = :relaxation) where {V <: AbstractVector{<:Real}, S <: AbstractMatrix}
        n = length(times)
        size(samples) == (n, n) || throw(
            DimensionMismatch("SampledViscoLaw: $(size(samples)) samples for a grid of $n instants; expected ($n, $n)")
        )
        return new{V, S}(times, samples, _canonical_mode(mode))
    end
end

# A phase property in aging viscoelasticity: a law; its relaxation operator on the time
# grid — a dense block matrix or a `VolterraOperator` — as one scale hands it to the
# next; or an elastic (conduction, at order 2) tensor, whose kernel is C H(t - t').
const ALVProperty = Union{AbstractViscoLaw, AbstractMatrix, TensND.AbstractTens}

visco_eval(law::SampledViscoLaw, t, t_p) =
    mapreduce(((i, j, w),) -> w * law.samples[i, j], +, _kernel_stencil(law.times, t, t_p))

(law::SampledViscoLaw)(t, t_p) = visco_eval(law, t, t_p)

# ── Reading a kernel between the instants ───────────────────────────────────
#
# The trapezoidal rule integrates exactly a kernel f(tᵢ, ·) linear between the nodes, against
# a history linear between them, and a jump at a doubled instant against f(tᵢ, t⁺). That
# piecewise linear kernel, right-continuous at the doubled instants, is the one the operator
# holds: it is what the kernel is between the nodes in t′. Between rows (in t) the operator
# says nothing; the kernel is taken linear there too. On the cell [tₐ, tₐ₊₁] × [t_b, t_b₊₁]
# below the diagonal this is the bilinear interpolation of its four corners; on a cell the
# diagonal cuts (a = b), the linear one on the triangle t′ ≤ t of its three corners
# (a, a), (a+1, a), (a+1, a+1), the fourth corner lying above the diagonal. Both agree on
# the edges, so that the kernel is continuous below the diagonal away from the doubled
# instants. Locating the cell is a binary search: `searchsortedlast` returns the second node
# of a doubled instant, the right limit.

# The cell that holds `t`: its first node a (tₐ ≤ t < tₐ₊₁, or a = n at the last instant), the
# node after it, and the fraction of the cell at `t`.
function _grid_cell(times, t)
    is_hard_numeric(typeof(t)) || throw(
        ArgumentError(
            "the kernel is read between the instants by comparing them, which a $(typeof(t)) cannot do; " *
                "give numbers, or build the operator from a law, which takes any instant"
        )
    )
    n = length(times)
    first(times) ≤ t ≤ last(times) || throw(
        DomainError(t, "the instant lies outside the grid [$(first(times)), $(last(times))]; nothing is extrapolated")
    )
    a = searchsortedlast(times, t)
    a == n && return n, n, (t - times[n]) / one(eltype(times))
    return a, a + 1, (t - times[a]) / (times[a + 1] - times[a])
end

# The kernel at (t, t′) as Σ w f(i, j) over at most four nodes (i, j), returned as four
# (i, j, w): a fixed shape, whatever the case, so that the caller's sum is inferred.
function _kernel_stencil(times, t, t′)
    a, a₊, θ = _grid_cell(times, t)
    b, b₊, φ = _grid_cell(times, t′)
    o, l = zero(θ * φ), one(θ * φ)
    t′ > t && return ((1, 1, o), (1, 1, o), (1, 1, o), (1, 1, o))
    a == b && return ((a, a, l - θ), (a₊, a, θ - φ), (a₊, a₊, φ), (a₊, a₊, o))
    return ((a, b, (l - θ) * (l - φ)), (a, b₊, (l - θ) * φ), (a₊, b, θ * (l - φ)), (a₊, b₊, θ * φ))
end

"""
    operator_law(R, times; block_size = 6, mode = :relaxation) -> SampledViscoLaw

The law whose trapezoidal operator on `times` is `R`, a lower block-triangular
``(Bn)\\times(Bn)`` matrix with `B = block_size` (`6` for a 4-tensor in Mandel form, `3` for a
2-tensor, `1` for a scalar): typically the relaxation operator that [`homogenize_alv`](@ref)
returns for a composite, so that this composite becomes a phase of the next scale.

Its kernel is recovered row by row: on row ``i``, with ``f_k = f(t_i, t_k)``, the sum of
the blocks ``j \\ge k`` is the weight of the interval that ends at ``t_k``,
``\\tfrac12(f_{k-1}+f_k)``, or ``f_k`` when that interval has zero length (``k = 1``, or an
instant given twice); these sums give the ``f_k`` one after the other, exactly.
[`trapezoidal_matrix`](@ref) of the result gives `R` back.

The result is a function of ``(t, t')``: exact at the instants, and between them the kernel
that the trapezoidal rule integrates exactly, linear in ``t'`` between the instants and
right-continuous at a doubled one, linear in ``t`` too (bilinear on a cell of the grid,
linear on the triangle ``t' \\le t`` of a cell the diagonal cuts). Finding the cell costs a
binary search, ``O(\\log n)``, and the value four samples. Recovering the samples costs
``O(n^2)`` once: for many evaluations, `law = operator_law(R)` once and `law(t, t′)` then; a
single `R(t, t′)` recovers only the two rows it needs, ``O(n)``.
"""
Base.@constprop :aggressive function operator_law(R::AbstractMatrix, times::AbstractVector{<:Real}; block_size::Int = 6, mode::Symbol = :relaxation)
    # The block size sets the type of the samples (a scalar or a block): propagated as a
    # constant, a literal `block_size` (the default included) is resolved at compile time.
    n, B = length(times), block_size
    size(R) == (B * n, B * n) || throw(
        DimensionMismatch("operator_law: a $(size(R)) operator for $n instants and blocks of size $B; expected ($(B * n), $(B * n))")
    )
    TimeGrid(times)   # in increasing order
    return SampledViscoLaw(times, _kernel_samples((i, j) -> _operator_block(R, i, j, Val(B)), times); mode)
end

# The block (i, j) of the operator: a view for blocks, the entry for scalars.
_operator_block(R, i, j, ::Val{1}) = R[i, j]
_operator_block(R, i, j, ::Val{B}) where {B} = R[(B * (i - 1) + 1):(B * i), (B * (j - 1) + 1):(B * j)]

# The kernel samples f(tᵢ, tₖ), k ≤ i, from the blocks `M(i, k)` of a trapezoidal operator.
function _kernel_samples(M::F, times) where {F}
    n = length(times)
    z = zero(M(1, 1))
    S = Matrix{typeof(z)}(undef, n, n)
    for i in 1:n
        _kernel_row!(view(S, i, :), M, times, i)
        S[i, (i + 1):n] .= Ref(z)       # the kernel vanishes for t' > t
    end
    return S
end

# Row i of the kernel, into f[1:i]: the suffix sums Sₖ = Σ_{j ≥ k} M(i, j) are the weights wₖ of
# the intervals ending at node k (`_trapezoidal_weight`), wₖ = fₖ at a jump (k = 1 or a doubled
# instant) and (fₖ₋₁ + fₖ)/2 otherwise, solved for fₖ from k = 1 on.
function _kernel_row!(f, M::F, times, i) where {F}
    f[i] = M(i, i)
    for k in (i - 1):-1:1
        f[k] = f[k + 1] + M(i, k)
    end
    for k in 2:i
        _is_jump(times, k) || (f[k] = 2 * f[k] - f[k - 1])
    end
    return f
end

"""
    precipitated(law, t_s) -> ViscoLaw

The relaxation law of a phase that precipitates **stress-free** at ``t_s``: a strain applied
before ``t_s`` loads nothing, so the kernel is ``H(t'-t_s)\\,R(t,t')``. The phase of hydrates
filling a pore at ``t_s`` in a hydrating paste, for instance.
"""
precipitated(law::AbstractViscoLaw, t_s::Real) = _relaxation_transform(law, _precipitation(t_s), "precipitated")

"""
    dissolved(law, t_d) -> ViscoLaw

The relaxation law of a phase that **dissolves** at ``t_d``: it carries no stress afterwards,
so the kernel is ``H(t_d-t)\\,R(t,t')``. An unhydrated grain consumed by hydration at ``t_d``.
"""
dissolved(law::AbstractViscoLaw, t_d::Real) = _relaxation_transform(law, _dissolution(t_d), "dissolved")

# When the phase carries stress: from t_s on in the loading time t′ (the left limit of a doubled
# t_s excluded, its right limit included), and before t_d in the current time t.
_precipitation(t_s) = (t, t_p) -> t_p ≥ t_s
_dissolution(t_d) = (t, t_p) -> t < t_d

# Both transforms switch a relaxation kernel on or off; a creep kernel has no such meaning.
function _relaxation_transform(law::AbstractViscoLaw, alive::F, name) where {F}
    visco_mode(law) === :relaxation || throw(
        ArgumentError("$name acts on a relaxation kernel; this law is in :$(visco_mode(law)) mode")
    )
    return ViscoLaw(
        function (t, t_p)
            v = visco_eval(law, t, t_p)
            return alive(t, t_p) ? v : zero(v)
        end, :relaxation
    )
end
