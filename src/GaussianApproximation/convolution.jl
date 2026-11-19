# =============================================================================
#  convolution.jl — the product by the matrix of a Gaussian approximation.
#
#  With the nodes on a regular grid, entry (I, J) of the matrix depends on I - J
#  only: the product is a discrete convolution of the kernel sampled at the
#  differences of nodes, K[Δ] = K(Δ h), with the unknown fields. It is evaluated
#  by FFT on arrays padded with zeros to at least 2n - 1 per direction, which
#  makes the periodic convolution of the transform equal to the plain one: the
#  sum over the nodes is exact, nothing is periodized.
#
#  The kernel is a symmetric block of m × m components, even in Δ; its spectra
#  are then real and only the m(m+1)/2 upper ones are kept.
# =============================================================================

# The sampled components of a kernel, or the fields of a product: at least one
# array, all of one type (an empty tuple would leave that type undetermined).
const Arrays{T, d} = Tuple{Array{T, d}, Vararg{Array{T, d}}}

# Position of the pair (p, q), p ≤ q, in the upper triangle stored by columns.
@inline _pair(p, q) = p ≤ q ? p + (q * (q - 1)) ÷ 2 : q + (p * (p - 1)) ÷ 2

"""
    sample_kernel(kernel, dims, Val(m)) -> NTuple{m(m+1)/2, Array}

Values of the symmetric `m × m` block `kernel(Δ)` at every difference of nodes
`Δ` (a tuple of integers) of a grid of size `dims`: one array of size `2n - 1`
per direction and per component of the upper triangle, the difference `Δ` stored
at index `Δ + n`. The kernel is even: it is evaluated on half of the differences.
"""
function sample_kernel(kernel::F, dims::NTuple{d, Int}, vm::Val{m}) where {F, d, m}
    Δs = CartesianIndices(map(n -> (1 - n):(n - 1), dims))
    off = CartesianIndex(dims)
    T = eltype(kernel(Tuple(first(Δs))))
    samples = ntuple(_ -> Array{T, d}(undef, size(Δs)), Val((m * (m + 1)) ÷ 2))
    # the first half of the differences, in the order of the indices, and the center
    for Δ in Iterators.take(Δs, (length(Δs) + 1) ÷ 2)
        B = kernel(Tuple(Δ))
        for q in 1:m, p in 1:q
            samples[_pair(p, q)][Δ + off] = samples[_pair(p, q)][off - Δ] = B[p, q]
        end
    end
    return samples
end

"""
    GridConvolution{d, m}

Product of the fields carried by the nodes of a grid by a sampled kernel (see
[`sample_kernel`](@ref)), by FFT on zero-padded arrays. Built once per kernel,
applied by [`convolve!`](@ref).
"""
struct GridConvolution{d, m, PF, PB}
    dims::NTuple{d, Int}
    spectra::Vector{Array{Float64, d}}
    forward::PF
    backward::PB
    pad::Array{ComplexF64, d}
    work::Vector{Array{ComplexF64, d}}
    acc::Array{ComplexF64, d}
end

function GridConvolution(samples::Arrays{Float64, d}, ::Val{m}) where {d, m}
    length(samples) == (m * (m + 1)) ÷ 2 || throw(
        DimensionMismatch("a symmetric $m×$m block has $((m * (m + 1)) ÷ 2) components, got $(length(samples))")
    )
    dims = map(s -> (s + 1) ÷ 2, size(first(samples)))
    P = _padded_dims(dims)
    pad, acc = zeros(ComplexF64, P), zeros(ComplexF64, P)
    # the single seam to the FFT: FFTA.jl, pure Julia, planned once per grid
    forward, backward = FFTA.plan_fft(pad), FFTA.plan_bfft(pad)
    spectra = map(collect(samples)) do K
        fill!(pad, 0)
        for Δ in CartesianIndices(map(n -> (1 - n):(n - 1), dims))
            # the difference Δ sits at Δ modulo the padded size
            pad[CartesianIndex(mod.(Tuple(Δ), P) .+ 1)] = K[Δ + CartesianIndex(dims)]
        end
        # an even real kernel has a real spectrum
        real(mul!(acc, forward, pad))
    end
    return GridConvolution{d, m, typeof(forward), typeof(backward)}(
        dims, spectra, forward, backward, pad, [zeros(ComplexF64, P) for _ in 1:m], acc
    )
end

"""
    convolve!(Y, C::GridConvolution, X) -> Y

`Y[p][I] = Σ_q Σ_J K_pq(I - J) X[q][J]` over the nodes `J` of the grid: `X` and
`Y` are tuples of `m` arrays of the size of the grid.
"""
function convolve!(Y::NTuple{m, Array{Float64, d}}, C::GridConvolution{d, m}, X::NTuple{m, Array{Float64, d}}) where {d, m}
    R = CartesianIndices(C.dims)
    for (w, x) in zip(C.work, X)
        fill!(C.pad, 0)
        copyto!(C.pad, R, x, R)
        mul!(w, C.forward, C.pad)
    end
    scale = inv(length(C.acc))
    for (p, y) in enumerate(Y)
        fill!(C.acc, 0)
        # accumulation into the buffer: a sum of broadcasts would allocate one array per term
        for q in 1:m
            C.acc .+= C.spectra[_pair(p, q)] .* C.work[q]
        end
        mul!(C.pad, C.backward, C.acc)
        @views y .= real.(C.pad[R]) .* scale
    end
    return Y
end

"""
    convolve_direct(samples, X, Val(m)) -> NTuple{m, Array}

The same product as [`convolve!`](@ref) by the plain double sum over the nodes:
quadratic in the number of nodes, generic in the number type, and the reference
the FFT product is tested against.
"""
function convolve_direct(samples::Arrays{Tk, d}, X::Arrays{Tx, d}, ::Val{m}) where {Tk, Tx, d, m}
    R = CartesianIndices(size(first(X)))
    off = CartesianIndex(size(first(X)))
    T = promote_type(Tk, Tx)
    return ntuple(Val(m)) do p
        [sum(samples[_pair(p, q)][I - J + off] * X[q][J] for q in 1:m, J in R; init = zero(T)) for I in R]
    end
end
