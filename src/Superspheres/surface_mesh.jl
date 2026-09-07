# =============================================================================
#  surface_mesh.jl — a triangulated surface for a supershape.
#
#  Subdivided octahedron → radial map → tangential relaxation.
#
#  ## Why the octahedron, and not an icosphere or a (θ, φ) grid
#
#  No CAD kernel can represent `|x|^{2p} + |y|^{2p} + |z|^{2p} = a^{2p}`, so the
#  surface has to be discretized in Julia and handed to the mesher as a
#  *discrete* entity.  What it is discretized *from* then matters, and starting
#  from a subdivided octahedron buys three things at once:
#
#    * the face edges lie **exactly** in the coordinate planes, so an octant is
#      native — cubic symmetry can be exploited with no tolerance offset;
#    * for a concave `p < 1/2` those same planes are where the surface
#      **creases**, so mesh edges land on the creases by construction instead of
#      straddling them;
#    * there is no polar degeneracy, as a (θ, φ) grid has at both poles.
#
#  And the octahedron *is* the `p = 1/2` supersphere, so the construction is
#  self-consistent: at that value the radial map is the identity and the mesh is
#  exact.  That makes `p = 1/2` a bit-exact oracle rather than a converging one,
#  and it is the test worth keeping.
#
#  ## This file is deliberately not type-generic
#
#  Everything else about these shapes carries `T <: Number` and differentiates.
#  A mesh does not: it exists to be handed to a mesher, which wants floating
#  point, and the finite-element solve downstream is `Float64` besides.  So
#  `TriSurface{T <: Real}`, and the genericity lives in `shapes.jl` where it
#  earns its keep.
# =============================================================================

"""
    TriSurface{T}

A triangulated surface. `nodes` are points, `tris` are 1-based node triples
oriented so that ``(b-a) \\times (c-a)`` points **outward**.

`constraint[i]` records what a node may do under relaxation, and it is the field
that makes octant symmetry usable:

| value | meaning |
|:--|:--|
| `0` | free |
| `1`, `2`, `3` | confined to the plane ``x = 0``, ``y = 0``, ``z = 0`` |
| `4` | pinned — an axis vertex of the shape, shared by two planes |

A node on a plane has the corresponding coordinate **exactly** zero, and stays
that way through every operation here. Nothing rounds it back.
"""
struct TriSurface{T <: Real}
    nodes::Vector{NTuple{3, T}}
    tris::Vector{NTuple{3, Int}}
    constraint::Vector{Int8}
end

"""
    node_count(surf) -> Int
    triangle_count(surf) -> Int

Sizes of a [`TriSurface`](@ref). Spelled out rather than called `nnodes` and
`ntris`, which would collide with `Ferrite`'s own names for anyone who loads
both.
"""
node_count(s::TriSurface) = length(s.nodes)

"See [`node_count`](@ref)."
triangle_count(s::TriSurface) = length(s.tris)

Base.eltype(::TriSurface{T}) where {T} = T

function Base.show(io::IO, s::TriSurface{T}) where {T}
    return print(
        io, "TriSurface{", T, "}(", node_count(s), " nodes, ",
        triangle_count(s), " triangles)"
    )
end

# ─── Small vector helpers, on tuples, to keep the inner loops allocation-free ─

_norm3(v) = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
_unit3(v) = (n = _norm3(v); (v[1] / n, v[2] / n, v[3] / n))
_dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
_cross3(a, b) = (
    a[2] * b[3] - a[3] * b[2],
    a[3] * b[1] - a[1] * b[3],
    a[1] * b[2] - a[2] * b[1],
)

# ─── The base subdivision ────────────────────────────────────────────────────

"""
    octant_patch(level; T = Float64) -> TriSurface

The ``(+,+,+)`` face of the unit octahedron, subdivided `level` times into
``4^{\\text{level}}`` triangles and projected onto the unit sphere.

Nodes come from the barycentric lattice ``(i, j, k)`` with
``i + j + k = N = 2^{\\text{level}}``, normalized. A node with ``k = 0``
therefore has ``z`` **exactly** zero — not zero to a tolerance — which is what
makes the coordinate planes exact symmetry planes of the mesh and an octant
computation legitimate.
"""
function octant_patch(level::Integer; T::Type{<:Real} = Float64)
    level ≥ 0 || throw(ArgumentError("level must be ≥ 0, got $level"))
    N = 2^level
    idx = Dict{Tuple{Int, Int}, Int}()
    nodes = NTuple{3, T}[]
    constraint = Int8[]
    for i in 0:N, j in 0:(N - i)
        k = N - i - j
        push!(nodes, _unit3((T(i), T(j), T(k))))
        # One vanishing barycentric coordinate puts the node on a coordinate
        # plane; two put it on an axis, where two planes meet and it is pinned.
        nz = (i == 0) + (j == 0) + (k == 0)
        c = if nz ≥ 2
            Int8(4)
        elseif i == 0
            Int8(1)
        elseif j == 0
            Int8(2)
        elseif k == 0
            Int8(3)
        else
            Int8(0)
        end
        push!(constraint, c)
        idx[(i, j)] = length(nodes)
    end
    tris = NTuple{3, Int}[]
    for i in 0:(N - 1), j in 0:(N - 1 - i)
        a, b, c = idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]
        push!(tris, _outward(nodes, (a, b, c)))
        if i + j < N - 1
            d = idx[(i + 1, j + 1)]
            push!(tris, _outward(nodes, (b, d, c)))
        end
    end
    return TriSurface{T}(nodes, tris, constraint)
end

# Orient so that (b-a) × (c-a) points away from the origin. Every surface here
# is star-shaped about the origin, so the centroid is a valid outward reference
# — which is the same property that makes the radial map single-valued.
function _outward(nodes, t)
    a, b, c = nodes[t[1]], nodes[t[2]], nodes[t[3]]
    nrm = _cross3(
        (b[1] - a[1], b[2] - a[2], b[3] - a[3]),
        (c[1] - a[1], c[2] - a[2], c[3] - a[3]),
    )
    ctr = ((a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3, (a[3] + b[3] + c[3]) / 3)
    return _dot3(nrm, ctr) ≥ 0 ? t : (t[1], t[3], t[2])
end

"""
    unit_octahedron(level; T = Float64) -> TriSurface

The whole subdivided octahedron projected onto the unit sphere, obtained by
reflecting [`octant_patch`](@ref) through the three coordinate planes and
merging the shared nodes.

Nodes on a plane merge **exactly**, because the reflection maps their vanishing
coordinate to itself. One subtlety makes that true in floating point rather than
almost true: `-1.0 * 0.0` is `-0.0`, which compares equal to `0.0` but hashes
differently, so the merge key adds `0.0` to normalize the sign of zero. Without
that the three planes carry doubled nodes and the surface is not closed.
"""
function unit_octahedron(level::Integer; T::Type{<:Real} = Float64)
    patch = octant_patch(level; T)
    nodes = NTuple{3, T}[]
    constraint = Int8[]
    tris = NTuple{3, Int}[]
    key(v) = (
        round(Float64(v[1]); digits = 12) + 0.0,
        round(Float64(v[2]); digits = 12) + 0.0,
        round(Float64(v[3]); digits = 12) + 0.0,
    )
    seen = Dict{NTuple{3, Float64}, Int}()
    for sx in (1, -1), sy in (1, -1), sz in (1, -1)
        s = (T(sx), T(sy), T(sz))
        local_id = Vector{Int}(undef, node_count(patch))
        for (n, v) in enumerate(patch.nodes)
            w = (s[1] * v[1], s[2] * v[2], s[3] * v[3])
            k = key(w)
            id = get(seen, k, 0)
            if id == 0
                push!(nodes, w)
                push!(constraint, patch.constraint[n])
                id = length(nodes)
                seen[k] = id
            end
            local_id[n] = id
        end
        flip = (sx * sy * sz) < 0        # an odd number of reflections reverses winding
        for t in patch.tris
            a, b, c = local_id[t[1]], local_id[t[2]], local_id[t[3]]
            push!(tris, flip ? (a, c, b) : (a, b, c))
        end
    end
    return TriSurface{T}(nodes, tris, constraint)
end

# ─── The radial map ──────────────────────────────────────────────────────────

"""
    project_to_shape!(surf, shape) -> surf

Move every node radially onto the surface of `shape`.

Valid for **every** ``p > 0``, concave shapes included, because these bodies are
star-shaped about their center and [`radial_distance`](@ref) is therefore
single-valued in a direction. A projection along the normal, or any iterative
snapping, would have to worry about folding; this one cannot fold.
"""
function project_to_shape!(surf::TriSurface, shape::AbstractSuperShape)
    @inbounds for i in eachindex(surf.nodes)
        surf.nodes[i] = surface_point(shape, _unit3(surf.nodes[i]))
    end
    return surf
end

"""
    shape_surface(shape, level; octant = false, relax = 0, kwargs...) -> TriSurface

The discretized surface of `shape`: a subdivided octahedron — or a single octant
patch, when `octant = true` — radially mapped, and relaxed `relax` times.
`kwargs` are forwarded to [`relax_surface!`](@ref).
"""
function shape_surface(
        shape::AbstractSuperShape, level::Integer;
        octant::Bool = false, relax::Integer = 0, kwargs...,
    )
    surf = octant ? octant_patch(level) : unit_octahedron(level)
    project_to_shape!(surf, shape)
    relax > 0 && relax_surface!(surf, shape; iterations = relax, kwargs...)
    return surf
end

# ─── Tangential relaxation ───────────────────────────────────────────────────

"""
    relax_surface!(surf, shape; iterations = 20, step = 0.5, density = nothing) -> surf

Equalize the mesh by a **tangential** umbrella sweep followed by a radial
re-projection, `iterations` times.

The raw radial map is far from uniform: it compresses the mesh near the axis
vertices of a concave shape and stretches it near the body diagonal.

**What the sweep buys, and what it costs.** It equalizes *edge lengths*, and it
does so at the expense of the *minimum angle* — the two are not the same
quality and cannot both be maximized. Measured at level 3, 30 iterations:

| shape | ``h_{\\max}/h_{\\min}`` | smallest angle |
|:--|:--|:--|
| `Supersphere(1, 0.30)` | `4.38 → 3.41` | `19.8° → 16.9°` |
| `Supersphere(1, 0.25)` | `8.07 → 6.09` | `12.7° → 9.4°` |
| `Superspheroid(1, 2.5, 0.35)` | `7.96 → 4.92` | `15.0° → 11.7°` |

So use it when element *size* uniformity is what matters — which is the usual
case, a mesher sizing its tetrahedra from the boundary edges — and leave it off
(`relax = 0`, the default) when the smallest angle is the binding constraint.

Two details separate a sweep that helps from one that wrecks the mesh, and both
were established by getting them wrong first.

* **The displacement is projected onto the tangent plane before the node
  moves.** Its normal component is destroyed by the radial re-projection
  anyway, so keeping it only adds a drift that fights the curvature.
* **A node confined to a coordinate plane relaxes only against its neighbors on
  that same plane**, as a one-dimensional chain along the boundary curve.
  Averaging it against interior neighbors drags it along the curve unevenly and
  destroys an already-perfect mesh. At ``p = 1/2``, where the radial map is the
  identity and the raw mesh *is* the exact barycentric lattice, that mistake
  turns 60° angles into 30° ones.

With both in place the sweep is a **fixed point** of the ``p = 1/2`` lattice,
which is the property to keep and is asserted by the tests.

`density(x) -> ρ` biases the average toward high ``ρ``, so nodes cluster where
``ρ`` is large; pass ``ρ = 1/h`` for a target size field ``h``. The default,
`nothing`, equalizes edge lengths.
"""
function relax_surface!(
        surf::TriSurface{T}, shape::AbstractSuperShape;
        iterations::Integer = 20, step::Real = 0.5, density = nothing,
    ) where {T}
    0 < step ≤ 1 || throw(ArgumentError("step must lie in (0, 1], got $step"))
    adj = _adjacency(surf)
    buf = similar(surf.nodes)
    for _ in 1:iterations
        @inbounds for i in eachindex(surf.nodes)
            c = surf.constraint[i]
            x = surf.nodes[i]
            if c == 4
                buf[i] = x
                continue
            end
            nbrs = if c == 0
                adj[i]
            else
                filter(j -> surf.constraint[j] == c || surf.constraint[j] == 4, adj[i])
            end
            if isempty(nbrs)
                buf[i] = x
                continue
            end
            sx = sy = sz = zero(T)
            sw = zero(T)
            for j in nbrs
                v = surf.nodes[j]
                w = density === nothing ? one(T) : T(density(v))
                sx += w * v[1]
                sy += w * v[2]
                sz += w * v[3]
                sw += w
            end
            if !(sw > 0)
                buf[i] = x
                continue
            end
            d = (sx / sw - x[1], sy / sw - x[2], sz / sw - x[3])
            if c == 0
                nu = outward_normal(shape, x)
                dn = _dot3(d, nu)
                d = (d[1] - dn * nu[1], d[2] - dn * nu[2], d[3] - dn * nu[3])
            else
                # Confined to the plane x_c = 0. The surface cut by that plane
                # is a curve, so a radial re-projection *within* the plane is
                # already exact and no tangential correction is called for —
                # only the out-of-plane component must be killed, exactly, so
                # the node's vanishing coordinate stays a hard zero.
                d = c == 1 ? (zero(T), d[2], d[3]) :
                    c == 2 ? (d[1], zero(T), d[3]) : (d[1], d[2], zero(T))
            end
            m = (x[1] + step * d[1], x[2] + step * d[2], x[3] + step * d[3])
            buf[i] = _norm3(m) > 0 ? surface_point(shape, _unit3(m)) : x
        end
        copyto!(surf.nodes, buf)
    end
    return surf
end

function _adjacency(surf::TriSurface)
    adj = [Set{Int}() for _ in 1:node_count(surf)]
    for (a, b, c) in surf.tris
        push!(adj[a], b)
        push!(adj[a], c)
        push!(adj[b], a)
        push!(adj[b], c)
        push!(adj[c], a)
        push!(adj[c], b)
    end
    return [collect(s) for s in adj]
end

# ─── Diagnostics — this is what the closed forms in shapes.jl are for ────────

function _tri_area(surf::TriSurface, t)
    a, b, c = surf.nodes[t[1]], surf.nodes[t[2]], surf.nodes[t[3]]
    n = _cross3(
        (b[1] - a[1], b[2] - a[2], b[3] - a[3]),
        (c[1] - a[1], c[2] - a[2], c[3] - a[3]),
    )
    return _norm3(n) / 2
end

"""
    mesh_area(surf)

Sum of the triangle areas.

Deliberately **not** compared against a closed form: a triangulation of a curved
surface always under-resolves it, and unlike the volume there is no exact area
for a supersphere to compare with anyway. Used for self-convergence only.
"""
mesh_area(surf::TriSurface) = sum(t -> _tri_area(surf, t), surf.tris; init = 0.0)

"""
    mesh_volume(surf)

Volume enclosed by the surface, ``\\frac16 \\sum \\underline v_1 \\cdot
(\\underline v_2 \\times \\underline v_3)`` over outward-oriented triangles.

For an **octant patch** this is the volume of the cone from the origin over the
patch, which is exactly one eighth of the body — the three flat coordinate faces
pass through the origin and contribute nothing to the divergence integral. So
`8 * mesh_volume(patch)` compares directly with
[`shape_volume`](@ref), and that identity is what says the octant and the full
cell describe the same object.
"""
function mesh_volume(surf::TriSurface)
    acc = 0.0
    for t in surf.tris
        a, b, c = surf.nodes[t[1]], surf.nodes[t[2]], surf.nodes[t[3]]
        acc += _dot3(a, _cross3(b, c))
    end
    return acc / 6
end

"""
    edge_lengths(surf) -> Vector{Float64}

Every distinct edge length, for a quality report.
"""
function edge_lengths(surf::TriSurface)
    seen = Set{Tuple{Int, Int}}()
    out = Float64[]
    for (a, b, c) in surf.tris, (i, j) in ((a, b), (b, c), (c, a))
        k = minmax(i, j)
        k in seen && continue
        push!(seen, k)
        p, q = surf.nodes[i], surf.nodes[j]
        push!(out, _norm3((p[1] - q[1], p[2] - q[2], p[3] - q[3])))
    end
    return out
end

"""
    mesh_quality(surf) -> NamedTuple

`(; nodes, triangles, area, volume, hmin, hmax, hmean, ratio, min_angle)`.

`ratio = hmax/hmin` and `min_angle`, in degrees, are the two numbers that decide
whether a mesher will accept the surface as a volume boundary — a sliver on the
boundary propagates into the tetrahedra and produces a negative Jacobian
somewhere inside.
"""
function mesh_quality(surf::TriSurface)
    h = edge_lengths(surf)
    amin = 180.0
    for t in surf.tris
        p = (surf.nodes[t[1]], surf.nodes[t[2]], surf.nodes[t[3]])
        for k in 1:3
            u = p[mod1(k + 1, 3)] .- p[k]
            v = p[mod1(k + 2, 3)] .- p[k]
            cosang = _dot3(u, v) / (_norm3(u) * _norm3(v))
            amin = min(amin, acosd(clamp(cosang, -1.0, 1.0)))
        end
    end
    return (;
        nodes = node_count(surf), triangles = triangle_count(surf),
        area = mesh_area(surf), volume = mesh_volume(surf),
        hmin = minimum(h), hmax = maximum(h), hmean = sum(h) / length(h),
        ratio = maximum(h) / minimum(h), min_angle = amin,
    )
end
