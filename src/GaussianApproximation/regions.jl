# =============================================================================
#  regions.jl — describing a shape without a mesh.
#
#  The Gaussian approximation only asks one question to a shape: is this node
#  inside? A region is therefore anything that answers `x ∈ region` and knows a
#  box that contains it. Regions come from a catalog, from set operations on
#  other regions, from a function, or from an image.
# =============================================================================

"""
    AbstractRegion{d}

A bounded set of points of dimension `d` (2 for the plane of a crack, 3 for a
heterogeneity). A region answers `x ∈ region` and gives its
[`bounding_box`](@ref). Regions combine with `∪`, `∩` and `setdiff`.
"""
abstract type AbstractRegion{d} end

"""
    bounding_box(region) -> (lo, hi)

Corners of a box, with sides along the axes, that contains `region`.
"""
function bounding_box end

"""
    measure(region)

Area (`d = 2`) or volume (`d = 3`) of `region` when it is known in closed form,
`nothing` otherwise: the measure is then counted on the grid.
"""
measure(::AbstractRegion) = nothing

# ─── Position of a catalog shape in its plane ────────────────────────────────

# Center and rotation of a planar shape; `c`, `s` are the cosine and sine of the
# angle between the first axis of the shape and the first axis of the plane.
struct _Frame2{T <: Number}
    center::SVector{2, T}
    c::T
    s::T
end

function _Frame2(::Type{T}, center, angle) where {T}
    length(center) == 2 || throw(DimensionMismatch("a center in the plane has 2 coordinates, got $(length(center))"))
    s, c = sincos(T(angle))
    return _Frame2{T}(SVector{2, T}(center), c, s)
end

# Coordinates of `x` in the axes of the shape.
@inline function _local(f::_Frame2, x)
    δ = SVector{2}(x) - f.center
    return SVector(f.c * δ[1] + f.s * δ[2], -f.s * δ[1] + f.c * δ[2])
end

# Bounding box of the rotated rectangle of half-sides (a, b).
function _box(f::_Frame2, a, b)
    e = SVector(abs(f.c) * a + abs(f.s) * b, abs(f.s) * a + abs(f.c) * b)
    return (f.center - e, f.center + e)
end

_shape_type(xs...; center, angle) =
    TensND.float_like(promote_type(map(typeof, xs)..., eltype(center), typeof(angle)))

# ─── Catalog, in the plane ───────────────────────────────────────────────────

"""
    Ellipse(a, b; center=(0, 0), angle=0)

Ellipse of semi-axes `a` (along the direction at `angle` from the first axis) and
`b`.
"""
struct Ellipse{T <: Number} <: AbstractRegion{2}
    a::T
    b::T
    frame::_Frame2{T}
end
function Ellipse(a, b; center = (0, 0), angle = 0)
    T = _shape_type(a, b; center, angle)
    _check_positive(a, "semi-axis a")
    _check_positive(b, "semi-axis b")
    return Ellipse{T}(T(a), T(b), _Frame2(T, center, angle))
end

"""
    Disk(a; center=(0, 0))

Disk of radius `a`: `Ellipse(a, a; center)`.
"""
Disk(a; center = (0, 0)) = Ellipse(a, a; center)

Base.in(x, r::Ellipse) = (ξ = _local(r.frame, x); (ξ[1] / r.a)^2 + (ξ[2] / r.b)^2 ≤ 1)
bounding_box(r::Ellipse) = _box(r.frame, r.a, r.b)
measure(r::Ellipse{T}) where {T} = T(π) * r.a * r.b

"""
    Superellipse(a, b, p; center=(0, 0), angle=0)

Region ``|x/a|^{2p} + |y/b|^{2p} \\le 1``: an ellipse for ``p = 1``, a rectangle
as ``p \\to \\infty``, a four-pointed star for ``p < 1/2``.
"""
struct Superellipse{T <: Number} <: AbstractRegion{2}
    a::T
    b::T
    p::T
    frame::_Frame2{T}
end
function Superellipse(a, b, p; center = (0, 0), angle = 0)
    T = _shape_type(a, b, p; center, angle)
    _check_positive(a, "semi-axis a")
    _check_positive(b, "semi-axis b")
    _check_positive(p, "exponent p")
    return Superellipse{T}(T(a), T(b), T(p), _Frame2(T, center, angle))
end

Base.in(x, r::Superellipse) =
    (ξ = _local(r.frame, x); abs(ξ[1] / r.a)^(2 * r.p) + abs(ξ[2] / r.b)^(2 * r.p) ≤ 1)
bounding_box(r::Superellipse) = _box(r.frame, r.a, r.b)
measure(r::Superellipse) = 4 * r.a * r.b * gamma(1 + 1 / (2 * r.p))^2 / gamma(1 + 1 / r.p)

"""
    Rectangle(a, b; center=(0, 0), angle=0)

Rectangle of half-sides `a` and `b` (sides `2a` and `2b`).
"""
struct Rectangle{T <: Number} <: AbstractRegion{2}
    a::T
    b::T
    frame::_Frame2{T}
end
function Rectangle(a, b; center = (0, 0), angle = 0)
    T = _shape_type(a, b; center, angle)
    _check_positive(a, "half-side a")
    _check_positive(b, "half-side b")
    return Rectangle{T}(T(a), T(b), _Frame2(T, center, angle))
end

Base.in(x, r::Rectangle) = (ξ = _local(r.frame, x); abs(ξ[1]) ≤ r.a && abs(ξ[2]) ≤ r.b)
bounding_box(r::Rectangle) = _box(r.frame, r.a, r.b)
measure(r::Rectangle) = 4 * r.a * r.b

"""
    StarShape(ρ, ρmax; center=(0, 0))

Region bounded by the polar curve ``r = \\rho(\\varphi)`` around `center`, with
`ρmax` an upper bound of ``\\rho``.

```julia
StarShape(φ -> 1 + 0.3cos(5φ), 1.3)
```
"""
struct StarShape{T <: Number, F} <: AbstractRegion{2}
    ρ::F
    ρmax::T
    center::SVector{2, T}
end
function StarShape(ρ::F, ρmax; center = (0, 0)) where {F}
    T = TensND.float_like(promote_type(typeof(ρmax), eltype(center)))
    _check_positive(ρmax, "bound ρmax")
    return StarShape{T, F}(ρ, T(ρmax), SVector{2, T}(center))
end

function Base.in(x, r::StarShape)
    δ = SVector{2}(x) - r.center
    return hypot(δ[1], δ[2]) ≤ r.ρ(atan(δ[2], δ[1]))
end
bounding_box(r::StarShape) = (r.center .- r.ρmax, r.center .+ r.ρmax)

"""
    Polygon(points)

Simple polygon given by its vertices, in order (either orientation); the last
vertex is joined to the first.

```julia
Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])   # an L
```
"""
struct Polygon{T <: Number} <: AbstractRegion{2}
    vertices::Vector{SVector{2, T}}
end
function Polygon(points)
    length(points) ≥ 3 || throw(ArgumentError("a polygon needs at least 3 vertices, got $(length(points))"))
    T = TensND.float_like(mapreduce(eltype, promote_type, points))
    return Polygon{T}([SVector{2, T}(p) for p in points])
end

# Even-odd rule: a point is inside when a half-line from it crosses the boundary
# an odd number of times.
function Base.in(x, r::Polygon)
    ξ = SVector{2}(x)
    v = r.vertices
    return isodd(
        count(zip(v, Iterators.flatten((Iterators.drop(v, 1), (first(v),))))) do (p, q)
            (p[2] > ξ[2]) != (q[2] > ξ[2]) &&
                ξ[1] < p[1] + (ξ[2] - p[2]) * (q[1] - p[1]) / (q[2] - p[2])
        end
    )
end
bounding_box(r::Polygon) = (reduce((a, b) -> min.(a, b), r.vertices), reduce((a, b) -> max.(a, b), r.vertices))
# Shoelace formula.
measure(r::Polygon) = (
    v = r.vertices;
    abs(sum(p[1] * q[2] - q[1] * p[2] for (p, q) in zip(v, Iterators.flatten((Iterators.drop(v, 1), (first(v),)))))) / 2
)

# ─── Any function, any image ─────────────────────────────────────────────────

"""
    Region(inside, lo, hi; measure=nothing)

Region defined by any function: `inside(x)` returns `true` for the points of the
region, all contained in the box of corners `lo` and `hi`. The dimension is the
length of the corners. `measure` is its area or volume when it is known.

```julia
Region(x -> x[1]^4 + x[2]^4 ≤ 1 && x[1] + x[2] < 1.2, (-1, -1), (1, 1))
```
"""
struct Region{d, T <: Number, F, M <: Union{Nothing, T}} <: AbstractRegion{d}
    inside::F
    lo::SVector{d, T}
    hi::SVector{d, T}
    measure::M
end
function Region(inside::F, lo, hi; measure = nothing) where {F}
    d = length(lo)
    length(hi) == d || throw(DimensionMismatch("the corners have lengths $d and $(length(hi))"))
    T = TensND.float_like(promote_type(eltype(lo), eltype(hi)))
    M = _measure_as(T, measure)
    return Region{d, T, F, typeof(M)}(inside, SVector{d, T}(lo), SVector{d, T}(hi), M)
end
_measure_as(::Type{T}, ::Nothing) where {T} = nothing
_measure_as(::Type{T}, measure) where {T} = T(measure)

Base.in(x, r::Region{d}) where {d} = r.inside(SVector{d}(x))::Bool
bounding_box(r::Region) = (r.lo, r.hi)
measure(r::Region) = r.measure

# ─── Regions of space from the shapes of the package ─────────────────────────

"""
    Region(shape)

The region of space occupied by a shape of the package, in the frame of the
shape for a [`Supersphere`](@ref) or a [`Superspheroid`](@ref), with its
orientation for an [`Ellipsoid`](@ref).
"""
function Region(e::Elasticity.Ellipsoid{3})
    a = SVector{3}(e.semi_axes)
    P = SMatrix{3, 3}(TensND.vecbasis(e.basis, Val(:cov)))
    half = sqrt.((P .^ 2) * (a .^ 2))               # half-extents of the box along the axes
    return Region(x -> sum(abs2, (transpose(P) * x) ./ a) ≤ 1, -half, half; measure = 4 * π * prod(a) / 3)
end
function Region(s::Superspheres.AbstractSuperShape)
    half = _half_extents(s)
    return Region(x -> Superspheres.level_set(s, x) ≤ 0, -half, half; measure = Superspheres.shape_volume(s))
end
_half_extents(s::Superspheres.Supersphere) = SVector(s.a, s.a, s.a)
_half_extents(s::Superspheres.Superspheroid) = SVector(s.a, s.a, s.c)

"""
    Torus(R, a; center=(0, 0, 0))

Torus of axis ``\\underline{e}_3``: the points at a distance less than `a` from
the circle of radius `R` (``a < R``).
"""
function Torus(R, a; center = (0, 0, 0))
    is_hard_numeric(promote_type(typeof(R), typeof(a))) && !(0 < a < R) &&
        throw(ArgumentError("a torus needs 0 < a < R (tube and circle radii), got R = $R, a = $a"))
    c = SVector{3}(center)
    half = SVector(R + a, R + a, a)
    return Region(
        x -> (y = x - c; (hypot(y[1], y[2]) - R)^2 + y[3]^2 ≤ a^2), c - half, c + half;
        measure = 2 * π^2 * R * a^2
    )
end

"""
    PixelRegion(mask, h; lo=zeros)

Region read from an image or a voxel array: `mask[I]` is `true` when the cell
`I`, of side `h` and lower corner ``\\underline{lo} + (I - 1)\\,h``, belongs to
the region. A 2D `mask` describes a crack contour (a segmented micrograph), a 3D
one a heterogeneity (a tomography).
"""
struct PixelRegion{d, T <: Number, M <: AbstractArray{Bool, d}} <: AbstractRegion{d}
    mask::M
    h::T
    lo::SVector{d, T}
end
function PixelRegion(mask::AbstractArray{Bool, d}, h; lo = ntuple(_ -> 0, Val(d))) where {d}
    T = TensND.float_like(promote_type(typeof(h), eltype(lo)))
    _check_positive(h, "pixel size h")
    any(mask) || throw(ArgumentError("the mask of a `PixelRegion` has no cell set to `true`"))
    return PixelRegion{d, T, typeof(mask)}(mask, T(h), SVector{d, T}(lo))
end

function Base.in(x, r::PixelRegion{d}) where {d}
    I = CartesianIndex(ntuple(i -> floor(Int, (x[i] - r.lo[i]) / r.h) + 1, Val(d)))
    return checkbounds(Bool, r.mask, I) && r.mask[I]
end
bounding_box(r::PixelRegion{d}) where {d} = (r.lo, r.lo + SVector{d, Int}(size(r.mask)) * r.h)
measure(r::PixelRegion{d}) where {d} = count(r.mask) * r.h^d

# ─── Set operations ──────────────────────────────────────────────────────────

"Union of two regions, built by `a ∪ b`."
struct UnionRegion{d, A <: AbstractRegion{d}, B <: AbstractRegion{d}} <: AbstractRegion{d}
    a::A
    b::B
end
"Intersection of two regions, built by `a ∩ b`."
struct IntersectionRegion{d, A <: AbstractRegion{d}, B <: AbstractRegion{d}} <: AbstractRegion{d}
    a::A
    b::B
end
"Region `a` without the points of `b`, built by `setdiff(a, b)`."
struct DifferenceRegion{d, A <: AbstractRegion{d}, B <: AbstractRegion{d}} <: AbstractRegion{d}
    a::A
    b::B
end

Base.union(a::A, b::B) where {d, A <: AbstractRegion{d}, B <: AbstractRegion{d}} = UnionRegion{d, A, B}(a, b)
Base.intersect(a::A, b::B) where {d, A <: AbstractRegion{d}, B <: AbstractRegion{d}} = IntersectionRegion{d, A, B}(a, b)
Base.setdiff(a::A, b::B) where {d, A <: AbstractRegion{d}, B <: AbstractRegion{d}} = DifferenceRegion{d, A, B}(a, b)

Base.in(x, r::UnionRegion) = x ∈ r.a || x ∈ r.b
Base.in(x, r::IntersectionRegion) = x ∈ r.a && x ∈ r.b
Base.in(x, r::DifferenceRegion) = x ∈ r.a && !(x ∈ r.b)

function bounding_box(r::UnionRegion)
    (la, ha), (lb, hb) = bounding_box(r.a), bounding_box(r.b)
    return (min.(la, lb), max.(ha, hb))
end
function bounding_box(r::IntersectionRegion)
    (la, ha), (lb, hb) = bounding_box(r.a), bounding_box(r.b)
    lo, hi = max.(la, lb), min.(ha, hb)
    all(lo .< hi) || throw(ArgumentError("the two regions of an intersection do not overlap"))
    return (lo, hi)
end
bounding_box(r::DifferenceRegion) = bounding_box(r.a)

"""
    Annulus(a, c; center=(0, 0))

Ring between the radii `c` and `a` (``c < a``): `setdiff(Disk(a), Disk(c))`.
"""
function Annulus(a, c; center = (0, 0))
    is_hard_numeric(promote_type(typeof(a), typeof(c))) && !(0 < c < a) &&
        throw(ArgumentError("an annulus needs 0 < c < a (inner and outer radii), got a = $a, c = $c"))
    return setdiff(Disk(a; center), Disk(c; center))
end

# ─── A region seen by a grid ─────────────────────────────────────────────────

"""
    GaussianGrid(region, n; H=2.0)

Grid on the [`bounding_box`](@ref) of `region`, with `n` nodes along its longest
side.
"""
GaussianGrid(region::AbstractRegion, n::Integer; H = 2.0) = GaussianGrid(bounding_box(region)..., n; H)

"""
    inside_nodes(region, grid) -> Vector{CartesianIndex}

Indices of the nodes of `grid` that belong to `region`.
"""
function inside_nodes(region::AbstractRegion{d}, grid::GaussianGrid{d}) where {d}
    inside = [I for I in CartesianIndices(grid.dims) if node(grid, I) ∈ region]
    isempty(inside) && throw(
        ArgumentError(
            "no node of the grid ($(join(grid.dims, "×")), h = $(grid.h)) falls in the region. " *
                "Increase the number of nodes `n`."
        )
    )
    return inside
end
