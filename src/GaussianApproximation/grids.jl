# =============================================================================
#  grids.jl — the regular grid of nodes that carries the Gaussians.
# =============================================================================

"""
    GaussianGrid{d, T}

Regular grid of nodes of step `h` in dimension `d`, with the shape parameter `H`
of the Gaussians it carries. Node `I` (a `CartesianIndex`) is the center of cell
`I` of the box of lower corner `lo`: ``\\underline{x}_I = \\underline{lo} + (I - \\tfrac12)\\,h``.
"""
struct GaussianGrid{d, T <: Number}
    lo::SVector{d, T}
    h::T
    dims::NTuple{d, Int}
    H::T
    function GaussianGrid{d, T}(lo, h, dims, H) where {d, T <: Number}
        _check_positive(h, "grid step h")
        _check_positive(H, "shape parameter H")
        all(>(0), dims) || throw(ArgumentError("a grid needs at least one node per direction, got $dims"))
        return new{d, T}(lo, h, dims, H)
    end
end

# A comparison a symbolic value cannot decide: checked on hard numbers only.
function _check_positive(x, what)
    is_hard_numeric(typeof(x)) || return nothing
    x > 0 || throw(ArgumentError("the $what must be positive, got $x"))
    return nothing
end

"""
    GaussianGrid(lo, hi, n; H=2.0)

Grid covering the box of corners `lo` and `hi` with `n` nodes along its longest
side; the other sides receive as many nodes of the same step as they need. `H` is
the shape parameter of the Gaussians: the larger, the smaller the
[`saturation_error`](@ref) and the wider the Gaussians.
"""
function GaussianGrid(lo, hi, n::Integer; H = 2.0)
    d = length(lo)
    length(hi) == d || throw(DimensionMismatch("the corners have lengths $d and $(length(hi))"))
    n > 0 || throw(ArgumentError("the number of nodes n must be positive, got $n"))
    T = TensND.float_like(promote_type(eltype(lo), eltype(hi), typeof(H)))
    a = SVector{d, T}(lo)
    L = SVector{d, T}(hi) - a
    all(>(0), L) || throw(ArgumentError("the box must have positive sides, got lo = $lo, hi = $hi"))
    h = maximum(L) / n
    # a side that is a multiple of h up to rounding gets exactly that many nodes
    dims = ntuple(i -> max(1, ceil(Int, L[i] / h - 1.0e-9)), Val(d))
    return GaussianGrid{d, T}(a, h, dims, T(H))
end

"Position of node `I` of the grid."
node(g::GaussianGrid{d, T}, I::CartesianIndex{d}) where {d, T} =
    g.lo + (SVector{d, Int}(Tuple(I)) .- one(T) / 2) * g.h

"Iterator over the positions of the nodes of the grid, in the order of its indices."
nodes(g::GaussianGrid) = (node(g, I) for I in CartesianIndices(g.dims))

Base.show(io::IO, g::GaussianGrid{d}) where {d} =
    print(io, "GaussianGrid{", d, "} (", join(g.dims, "×"), " nodes, h=", g.h, ", H=", g.H, ")")

# Size of the zero-padded arrays on which a convolution over the grid is exact:
# at least 2n - 1 per direction, rounded up to a product of small primes.
_padded_dims(dims::NTuple{d, Int}) where {d} = map(n -> nextprod((2, 3, 5), 2n - 1), dims)

"""
    convolution_bytes(dims, components, fields) -> Int

Memory, in bytes, that a solve on a grid of size `dims` needs for a kernel of
`components` independent components acting on `fields` unknown fields: the
spectra of the kernel on the padded grid and the complex work arrays of the
product.
"""
function convolution_bytes(dims::NTuple{d, Int}, components::Integer, fields::Integer) where {d}
    P = prod(_padded_dims(dims))
    return P * (8 * components + 16 * (fields + 1))
end

# Refuse before allocating: the estimate against a budget the caller sets.
function _check_memory(dims, components, fields, memory_limit)
    need = convolution_bytes(dims, components, fields)
    need ≤ memory_limit || throw(
        ArgumentError(
            "a grid of $(join(dims, "×")) nodes needs about $(round(need / 2^30; digits = 2)) GiB, " *
                "over the limit of $(round(memory_limit / 2^30; digits = 2)) GiB. " *
                "Reduce the number of nodes `n`, or raise `memory_limit` (in bytes)."
        )
    )
    return nothing
end
