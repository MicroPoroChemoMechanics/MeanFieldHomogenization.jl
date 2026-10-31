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

A viscoelastic law known only at the instants of the grid `times`: `samples[i, j]` is its
kernel at `(times[i], times[j])` for `j ≤ i`, a scalar or a block (a ``6\\times 6`` Mandel
matrix at order 4, a ``3\\times 3`` matrix at order 2). Evaluating it off the grid is an error.

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
        mode in VALID_VISCO_MODES || throw(
            ArgumentError("SampledViscoLaw mode must be one of $(VALID_VISCO_MODES); got :$(mode)")
        )
        return new{V, S}(times, samples, mode)
    end
end

# A phase property in aging viscoelasticity: a law, or its relaxation operator on the time
# grid — a dense block matrix or a compact `AbstractALVKernel` — as one scale hands it to the
# next.
const ALVProperty = Union{AbstractViscoLaw, AbstractMatrix}

visco_eval(law::SampledViscoLaw, t, t_p) =
    law.samples[_grid_index(law.times, t), _grid_index(law.times, t_p)]

(law::SampledViscoLaw)(t, t_p) = visco_eval(law, t, t_p)

# The index of `t` in the grid; an instant off the grid has no sample.
function _grid_index(times::AbstractVector, t)
    i = searchsortedfirst(times, t)
    (i ≤ length(times) && times[i] == t) || throw(
        ArgumentError("SampledViscoLaw: t = $t is not an instant of its grid; evaluate it on the grid it was sampled on")
    )
    return i
end

"""
    operator_law(R, times; block_size = 6, mode = :relaxation) -> SampledViscoLaw

The law whose trapezoidal operator on `times` is `R`, a lower block-triangular
``(Bn)\\times(Bn)`` matrix with `B = block_size` (`6` for a 4-tensor in Mandel form, `3` for a
2-tensor, `1` for a scalar): typically the relaxation operator that [`homogenize_alv`](@ref)
returns for a composite, so that this composite becomes a phase of the next scale.

Its kernel is recovered row by row: on row ``i``, the operator holds
``\\tfrac12(f_{j-1}-f_{j+1})`` for ``j<i`` and ``\\tfrac12(f_{i-1}+f_i)`` on the diagonal, with
``f_j = f(t_i, t_j)``, a system solved exactly by sums and differences.
[`trapezoidal_matrix`](@ref) of the result gives `R` back.
"""
Base.@constprop :aggressive function operator_law(R::AbstractMatrix, times::AbstractVector{<:Real}; block_size::Int = 6, mode::Symbol = :relaxation)
    # The block size sets the type of the samples (a scalar or a block): propagated as a
    # constant, a literal `block_size` (the default included) is resolved at compile time.
    n, B = length(times), block_size
    size(R) == (B * n, B * n) || throw(
        DimensionMismatch("operator_law: a $(size(R)) operator for $n instants and blocks of size $B; expected ($(B * n), $(B * n))")
    )
    return SampledViscoLaw(times, _kernel_samples(R, n, Val(B)); mode)
end

# The block (i, j) of the operator: a view for blocks, the entry for scalars.
_operator_block(R, i, j, ::Val{1}) = R[i, j]
_operator_block(R, i, j, ::Val{B}) where {B} = R[(B * (i - 1) + 1):(B * i), (B * (j - 1) + 1):(B * j)]

# Row i: f_k = f_1 - c_k with c_1 = 0, c_2 = 2M_1, c_{k+1} = c_{k-1} + 2M_k, and the diagonal
# equation f_{i-1} + f_i = 2M_i fixes f_1 = M_i + (c_{i-1} + c_i)/2.
function _kernel_samples(R::AbstractMatrix, n::Int, b::Val)
    M(i, j) = _operator_block(R, i, j, b)
    F = Matrix{typeof(M(1, 1))}(undef, n, n)
    z = zero(M(1, 1))
    for i in 1:n
        c = Vector{typeof(z)}(undef, i)
        c[1] = z
        i ≥ 2 && (c[2] = 2 * M(i, 1))
        for k in 2:(i - 1)
            c[k + 1] = c[k - 1] + 2 * M(i, k)
        end
        f1 = i == 1 ? M(1, 1) : M(i, i) + (c[i - 1] + c[i]) / 2
        F[i, 1:i] .= Ref(f1) .- c
        F[i, (i + 1):n] .= Ref(z)       # the kernel vanishes for t' > t
    end
    return F
end

"""
    precipitated(law, t_s) -> ViscoLaw

The relaxation law of a phase that precipitates **stress-free** at ``t_s``: a strain applied
before ``t_s`` loads nothing, so the kernel is ``H(t'-t_s)\\,R(t,t')``. The phase of hydrates
filling a pore at ``t_s`` in a hydrating paste, for instance.
"""
precipitated(law::AbstractViscoLaw, t_s::Real) = _relaxation_transform(law, (t, t_p) -> t_p ≥ t_s, "precipitated")

"""
    dissolved(law, t_d) -> ViscoLaw

The relaxation law of a phase that **dissolves** at ``t_d``: it carries no stress afterwards,
so the kernel is ``H(t_d-t)\\,R(t,t')``. An unhydrated grain consumed by hydration at ``t_d``.
"""
dissolved(law::AbstractViscoLaw, t_d::Real) = _relaxation_transform(law, (t, t_p) -> t < t_d, "dissolved")

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
