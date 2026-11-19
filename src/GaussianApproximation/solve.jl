# =============================================================================
#  solve.jl — the linear system of a Gaussian approximation.
#
#  The unknowns are carried by the nodes inside the region only; the matrix is
#  the convolution of `convolution.jl` restricted to them, plus a block on each
#  node. It is never formed: a Krylov method needs its product alone.
#
#  Derivatives follow from the implicit-function theorem. With A(p) w = r,
#      A ∂w = ∂r - (∂A) w,
#  each partial derivative is one more solve with the same matrix. The FFT only
#  ever sees plain floating-point arrays, whatever is differentiated.
# =============================================================================

"""
    SampledKernel{m, T}

Kernel of a Gaussian system: the `samples` of its symmetric `m × m` block at the
differences of nodes (see [`sample_kernel`](@ref)) and a block `onsite` added on
every node (an interface stiffness; zero by default).
"""
struct SampledKernel{m, T <: Number, d, N, L}
    samples::NTuple{N, Array{T, d}}
    onsite::SMatrix{m, m, T, L}
end
SampledKernel(samples::Arrays{T, d}, ::Val{m}) where {T, d, m} =
    SampledKernel(samples, zero(SMatrix{m, m, T}))
function SampledKernel(samples::Arrays{Ts, d}, onsite::SMatrix{m, m, To, L}) where {Ts, d, m, To, L}
    T = promote_type(Ts, To)
    return SampledKernel{m, T, d, length(samples), L}(
        map(s -> convert(Array{T, d}, s), samples), SMatrix{m, m, T, L}(onsite)
    )
end

# The kernel at zero distance, the center of each sample array.
_self_block(K::SampledKernel{m, T, d, N, L}) where {m, T, d, N, L} =
    SMatrix{m, m, T, L}(ntuple(k -> (s = K.samples[_pair(fldmod1(k, m)...)]; s[(length(s) + 1) ÷ 2]), Val(L)))

# The matrix of the system: scatter the unknowns on the grid, convolve, gather,
# and add on node j the block `spring[j] * self + onsite`, where `self` is the
# kernel at zero distance and the springs hold the nodes of an edge
# (`edge_rule.jl`).
struct MaskedConvolution{d, m, L, C <: GridConvolution{d, m}}
    conv::C
    inside::Vector{CartesianIndex{d}}
    blocks::Vector{SMatrix{m, m, Float64, L}}
    X::NTuple{m, Array{Float64, d}}
    Y::NTuple{m, Array{Float64, d}}
end

function MaskedConvolution(K::SampledKernel{m, Float64}, inside, spring) where {m}
    conv = GridConvolution(K.samples, Val(m))
    self = _self_block(K)
    return MaskedConvolution(
        conv, inside, [s * self + K.onsite for s in spring],
        ntuple(_ -> zeros(conv.dims), Val(m)), ntuple(_ -> zeros(conv.dims), Val(m))
    )
end

Base.eltype(::MaskedConvolution) = Float64
Base.size(A::MaskedConvolution{d, m}) where {d, m} = (m * length(A.inside), m * length(A.inside))
Base.size(A::MaskedConvolution, i::Integer) = size(A)[i]

function LinearAlgebra.mul!(y::AbstractVector{Float64}, A::MaskedConvolution{d, m}, x::AbstractVector{Float64}) where {d, m}
    N = length(A.inside)
    for (q, X) in enumerate(A.X)
        fill!(X, 0)
        @views X[A.inside] .= x[((q - 1) * N + 1):(q * N)]
    end
    convolve!(A.Y, A.conv, A.X)
    for (p, Y) in enumerate(A.Y)
        @views y[((p - 1) * N + 1):(p * N)] .= Y[A.inside]
    end
    return _add_blocks!(y, A.blocks, x)
end

# y[node j] += blocks[j] * x[node j], the unknowns stored component after component.
function _add_blocks!(y, blocks::Vector{<:SMatrix{m, m}}, x) where {m}
    N = length(blocks)
    for (j, B) in enumerate(blocks), p in 1:m, q in 1:m
        y[(p - 1) * N + j] += B[p, q] * x[(q - 1) * N + j]
    end
    return y
end

# Jacobi preconditioner: the inverse of the blocks of the diagonal.
struct BlockJacobi{m, L}
    inverses::Vector{SMatrix{m, m, Float64, L}}
end
BlockJacobi(A::MaskedConvolution, self) = BlockJacobi([inv(self + B) for B in A.blocks])
Base.eltype(::BlockJacobi) = Float64
Base.size(M::BlockJacobi{m}) where {m} = (m * length(M.inverses), m * length(M.inverses))
Base.size(M::BlockJacobi, i::Integer) = size(M)[i]
LinearAlgebra.mul!(y::AbstractVector{Float64}, M::BlockJacobi, x::AbstractVector{Float64}) =
    _add_blocks!(fill!(y, 0), M.inverses, x)

"""
    masked_product(kernel::SampledKernel, inside, spring, W) -> Matrix

Product of the matrix of the Gaussian system (the `kernel` restricted to the
nodes `inside`, held by the springs `spring`) by each column of `W`.
"""
masked_product(K::SampledKernel, inside, spring, W::AbstractMatrix) =
    _masked_product(_common_eltype(K, W)..., inside, spring)

"""
    gaussian_solve(kernel::SampledKernel, inside, spring, R; symmetric=true, rtol=1e-10, itmax=0) -> Matrix

Solution `W` of the Gaussian system for each column of the right-hand sides `R`
(`m` components per inside node, component after component), the node `j` held
by the spring `spring[j]` times the kernel at zero distance. The kernel may
carry `ForwardDiff` duals, nested or not: the derivatives of `W` are then exact
for the discrete system, by the implicit-function theorem.

`symmetric = true` (a positive kernel, as that of a crack) uses the conjugate
gradient; `false` uses GMRES. `itmax = 0` leaves the limit to the solver.
"""
gaussian_solve(K::SampledKernel, inside, spring, R::AbstractMatrix; kw...) =
    _gaussian_solve(_common_eltype(K, R)..., inside, spring; kw...)

# Kernel and right-hand sides brought to one number type.
function _common_eltype(K::SampledKernel{m, Tk, d, N, L}, R::AbstractMatrix{Tr}) where {m, Tk, d, N, L, Tr}
    T = promote_type(Tk, Tr)
    return SampledKernel{m, T, d, N, L}(map(s -> convert(Array{T, d}, s), K.samples), SMatrix{m, m, T, L}(K.onsite)),
        convert(Matrix{T}, R)
end

# The FFT plans of `GridConvolution` are not inferred from the size of the grid:
# the matrix is built here and handed to a function that is compiled for its
# type, and the result is asserted.
function _masked_product(K::SampledKernel{m, Float64}, W::Matrix{Float64}, inside, spring) where {m}
    return _product_columns(MaskedConvolution(K, inside, spring), W)::Matrix{Float64}
end
_product_columns(A, W) = reduce(hcat, [mul!(similar(w, Float64), A, w) for w in eachcol(W)])

function _gaussian_solve(
        K::SampledKernel{m, Float64}, R::Matrix{Float64}, inside, spring;
        symmetric::Bool = true, rtol::Float64 = 1.0e-10, itmax::Int = 0
    ) where {m}
    A = MaskedConvolution(K, inside, spring)
    M = BlockJacobi(A, _self_block(K))
    # `symmetric` is a run-time Boolean: a static branch per value
    return (
        symmetric ? _solve_columns(A, M, R, Val(true), rtol, itmax) :
            _solve_columns(A, M, R, Val(false), rtol, itmax)
    )::Matrix{Float64}
end
_solve_columns(A, M, R, method::Val, rtol, itmax) =
    reduce(hcat, [_krylov(A, M, collect(r), method, rtol, itmax) for r in eachcol(R)])

_krylov(A, M, r, ::Val{true}, rtol, itmax) = _converged(Krylov.cg(A, r; M, atol = 0.0, rtol, itmax)...)
_krylov(A, M, r, ::Val{false}, rtol, itmax) = _converged(Krylov.gmres(A, r; M, atol = 0.0, rtol, itmax)...)

# The iteration limit reached before the tolerance: never a silently wrong field.
_converged(w, stats) = stats.solved ? w : error(
        "the Krylov solve of the Gaussian system stopped before its tolerance " *
        "($(stats.niter) iterations, status \"$(stats.status)\"). Raise `itmax` or loosen `rtol`."
    )

# One layer of duals: (A₀ + Σ Aᵢ εᵢ)(W₀ + Σ Wᵢ εᵢ), and the solve that inverts it.
# The value type V may itself be a dual: both recurse down to Float64.
_values(a::AbstractArray) = ForwardDiff.value.(a)
_partials(a::AbstractArray, i) = ForwardDiff.partials.(a, i)
_values(K::SampledKernel) = SampledKernel(map(_values, K.samples), _values(K.onsite))
_partials(K::SampledKernel, i) = SampledKernel(map(s -> _partials(s, i), K.samples), _partials(K.onsite, i))
_duals(::Type{D}, v, p::NTuple{P}) where {D, P} =
    [D(v[j], ForwardDiff.Partials(ntuple(i -> p[i][j], Val(P)))) for j in CartesianIndices(v)]

function _masked_product(
        K::SampledKernel{m, D}, W::Matrix{D}, inside, spring
    ) where {m, Tg, V, P, D <: ForwardDiff.Dual{Tg, V, P}}
    K₀, W₀ = _values(K), _values(W)
    parts = ntuple(Val(P)) do i
        _masked_product(K₀, _partials(W, i), inside, spring) + _masked_product(_partials(K, i), W₀, inside, spring)
    end
    return _duals(D, _masked_product(K₀, W₀, inside, spring), parts)
end

function _gaussian_solve(
        K::SampledKernel{m, D}, R::Matrix{D}, inside, spring; kw...
    ) where {m, Tg, V, P, D <: ForwardDiff.Dual{Tg, V, P}}
    K₀ = _values(K)
    W₀ = _gaussian_solve(K₀, _values(R), inside, spring; kw...)
    parts = ntuple(Val(P)) do i
        _gaussian_solve(K₀, _partials(R, i) - _masked_product(_partials(K, i), W₀, inside, spring), inside, spring; kw...)
    end
    return _duals(D, W₀, parts)
end

# An iterative solve stops on a comparison, which a symbolic number cannot decide.
_gaussian_solve(::SampledKernel{m, T}, ::Matrix{T}, inside, spring; kw...) where {m, T} = throw(
    ArgumentError(
        "the Gaussian system is solved iteratively, for floating-point numbers and " *
            "ForwardDiff duals only; got the number type $T."
    )
)
