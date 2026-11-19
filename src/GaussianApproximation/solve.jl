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

Matrix of a Gaussian system on the nodes it keeps: the `samples` of the
symmetric `m × m` block of its kernel at the differences of nodes (see
[`sample_kernel`](@ref)), and one more `m × m` block per node, added on the
diagonal (`blocks[j]` on the `j`-th node kept): the spring of an edge node, an
interface stiffness, the compliance of a contrast.
"""
struct SampledKernel{m, T <: Number, d, N, L}
    samples::NTuple{N, Array{T, d}}
    blocks::Vector{SMatrix{m, m, T, L}}
    # An inner constructor, so that no default outer one is generated: its
    # `NTuple{N, Array{T, d}}` argument leaves `d` unbound for an empty tuple.
    SampledKernel{m, T, d, N, L}(samples, blocks) where {m, T <: Number, d, N, L} =
        new{m, T, d, N, L}(samples, blocks)
end
function SampledKernel(samples::Arrays{Ts, d}, blocks::Vector{SMatrix{m, m, Tb, L}}) where {Ts, d, m, Tb, L}
    T = promote_type(Ts, Tb)
    return SampledKernel{m, T, d, length(samples), L}(
        map(s -> convert(Array{T, d}, s), samples), convert(Vector{SMatrix{m, m, T, L}}, blocks)
    )
end

"""
    self_block(samples, Val(m)) -> SMatrix

The kernel at zero distance, read at the center of its `samples`.
"""
self_block(samples::Arrays{T, d}, ::Val{m}) where {T, d, m} =
    SMatrix{m, m, T}(ntuple(k -> (s = samples[_pair(fldmod1(k, m)...)]; s[(length(s) + 1) ÷ 2]), Val(m * m)))

# The matrix of the system: scatter the unknowns on the grid, convolve, gather,
# and add the block of each node.
struct MaskedConvolution{d, m, L, C <: GridConvolution{d, m}}
    conv::C
    inside::Vector{CartesianIndex{d}}
    blocks::Vector{SMatrix{m, m, Float64, L}}
    X::NTuple{m, Array{Float64, d}}
    Y::NTuple{m, Array{Float64, d}}
end

function MaskedConvolution(K::SampledKernel{m, Float64}, inside) where {m}
    length(inside) == length(K.blocks) || throw(
        DimensionMismatch("the kernel holds $(length(K.blocks)) node blocks for $(length(inside)) nodes")
    )
    conv = GridConvolution(K.samples, Val(m))
    return MaskedConvolution(
        conv, inside, K.blocks, ntuple(_ -> zeros(conv.dims), Val(m)), ntuple(_ -> zeros(conv.dims), Val(m))
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
BlockJacobi(K::SampledKernel{m, Float64}) where {m} =
    (self = self_block(K.samples, Val(m)); BlockJacobi([inv(self + B) for B in K.blocks]))
Base.eltype(::BlockJacobi) = Float64
Base.size(M::BlockJacobi{m}) where {m} = (m * length(M.inverses), m * length(M.inverses))
Base.size(M::BlockJacobi, i::Integer) = size(M)[i]
LinearAlgebra.mul!(y::AbstractVector{Float64}, M::BlockJacobi, x::AbstractVector{Float64}) =
    _add_blocks!(fill!(y, 0), M.inverses, x)

"""
    masked_product(kernel::SampledKernel, inside, W) -> Matrix

Product of the matrix of the Gaussian system (the `kernel` restricted to the
nodes `inside`) by each column of `W`.
"""
masked_product(K::SampledKernel, inside, W::AbstractMatrix) = _masked_product(_common_eltype(K, W)..., inside)

"""
    gaussian_solve(kernel::SampledKernel, inside, R; method=:cg, rtol=1e-10, itmax=0) -> Matrix

Solution `W` of the Gaussian system for each column of the right-hand sides `R`
(`m` components per node kept, component after component). The kernel may carry
`ForwardDiff` duals, nested or not: the derivatives of `W` are then exact for
the discrete system, by the implicit-function theorem.

`method` is `:cg` (conjugate gradient with a block-Jacobi preconditioner, for a
positive definite matrix, as that of a crack), `:minres` (symmetric, of any
sign) or `:gmres`. `itmax = 0` leaves the limit to the solver.
"""
gaussian_solve(K::SampledKernel, inside, R::AbstractMatrix; kw...) =
    _gaussian_solve(_common_eltype(K, R)..., inside; kw...)

# Kernel and right-hand sides brought to one number type.
function _common_eltype(K::SampledKernel{m, Tk, d, N, L}, R::AbstractMatrix{Tr}) where {m, Tk, d, N, L, Tr}
    T = promote_type(Tk, Tr)
    return SampledKernel{m, T, d, N, L}(
            map(s -> convert(Array{T, d}, s), K.samples), convert(Vector{SMatrix{m, m, T, L}}, K.blocks)
        ), convert(Matrix{T}, R)
end

# The FFT plans of `GridConvolution` are not inferred from the size of the grid:
# the matrix is built here and handed to a function that is compiled for its
# type, and the result is asserted.
function _masked_product(K::SampledKernel{m, Float64}, W::Matrix{Float64}, inside) where {m}
    return _product_columns(MaskedConvolution(K, inside), W)::Matrix{Float64}
end
_product_columns(A, W) = reduce(hcat, [mul!(similar(w, Float64), A, w) for w in eachcol(W)])

function _gaussian_solve(
        K::SampledKernel{m, Float64}, R::Matrix{Float64}, inside;
        method::Symbol = :cg, rtol::Float64 = 1.0e-10, itmax::Int = 0
    ) where {m}
    A = MaskedConvolution(K, inside)
    # `method` is a run-time Symbol: a static branch per value
    return (
        method === :cg ? _solve_columns(A, BlockJacobi(K), R, Val(:cg), rtol, itmax) :
            method === :minres ? _solve_columns(A, I, R, Val(:minres), rtol, itmax) :
            method === :gmres ? _solve_columns(A, BlockJacobi(K), R, Val(:gmres), rtol, itmax) :
            throw(ArgumentError("unknown method :$method for the Gaussian system; use :cg, :minres or :gmres"))
    )::Matrix{Float64}
end
_solve_columns(A, M, R, method::Val, rtol, itmax) =
    reduce(hcat, [_krylov(A, M, collect(r), method, rtol, itmax) for r in eachcol(R)])

_krylov(A, M, r, ::Val{:cg}, rtol, itmax) = _converged(Krylov.cg(A, r; M, atol = 0.0, rtol, itmax)...)
_krylov(A, M, r, ::Val{:minres}, rtol, itmax) = _converged(Krylov.minres(A, r; M, atol = 0.0, rtol, itmax)...)
_krylov(A, M, r, ::Val{:gmres}, rtol, itmax) = _converged(Krylov.gmres(A, r; M, atol = 0.0, rtol, itmax)...)

# The iteration limit reached before the tolerance: never a silently wrong field.
_converged(w, stats) = stats.solved ? w : error(
        "the Krylov solve of the Gaussian system stopped before its tolerance " *
        "($(stats.niter) iterations, status \"$(stats.status)\"). Raise `itmax` or loosen `rtol`."
    )

# One layer of duals: (A₀ + Σ Aᵢ εᵢ)(W₀ + Σ Wᵢ εᵢ), and the solve that inverts it.
# The value type V may itself be a dual: both recurse down to Float64.
_values(a::AbstractArray) = ForwardDiff.value.(a)
_partials(a::AbstractArray, i) = ForwardDiff.partials.(a, i)
_values(K::SampledKernel) = SampledKernel(map(_values, K.samples), map(_values, K.blocks))
_partials(K::SampledKernel, i) = SampledKernel(map(s -> _partials(s, i), K.samples), map(b -> _partials(b, i), K.blocks))
_duals(::Type{D}, v, p::NTuple{P}) where {D, P} =
    [D(v[j], ForwardDiff.Partials(ntuple(i -> p[i][j], Val(P)))) for j in CartesianIndices(v)]

function _masked_product(
        K::SampledKernel{m, D}, W::Matrix{D}, inside
    ) where {m, Tg, V, P, D <: ForwardDiff.Dual{Tg, V, P}}
    K₀, W₀ = _values(K), _values(W)
    parts = ntuple(Val(P)) do i
        _masked_product(K₀, _partials(W, i), inside) + _masked_product(_partials(K, i), W₀, inside)
    end
    return _duals(D, _masked_product(K₀, W₀, inside), parts)
end

function _gaussian_solve(
        K::SampledKernel{m, D}, R::Matrix{D}, inside; kw...
    ) where {m, Tg, V, P, D <: ForwardDiff.Dual{Tg, V, P}}
    K₀ = _values(K)
    W₀ = _gaussian_solve(K₀, _values(R), inside; kw...)
    parts = ntuple(Val(P)) do i
        _gaussian_solve(K₀, _partials(R, i) - _masked_product(_partials(K, i), W₀, inside), inside; kw...)
    end
    return _duals(D, W₀, parts)
end

# An iterative solve stops on a comparison, which a symbolic number cannot decide.
_gaussian_solve(::SampledKernel{m, T}, ::Matrix{T}, inside; kw...) where {m, T} = throw(
    ArgumentError(
        "the Gaussian system is solved iteratively, for floating-point numbers and " *
            "ForwardDiff duals only; got the number type $T."
    )
)
