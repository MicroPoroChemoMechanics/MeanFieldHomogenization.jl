# =============================================================================
#  cell_gmsh_geometry.jl — the truncated Eshelby cell around a supershape.
#
#  The third cell family, after the flat crack and the axisymmetric core-shell
#  sphere: a full three-dimensional shell of matrix between a non-ellipsoidal
#  inclusion surface and an outer sphere.
#
#  ## Why the surface is *discrete*, and why there is no `.geo` file
#
#  No CAD kernel represents `|x|^{2p} + |y|^{2p} + |z|^{2p} = a^{2p}`.  So the
#  inclusion surface is built analytically by `Superspheres.shape_surface` and
#  handed to gmsh as a **discrete** entity — nodes and triangles, no
#  parametrization.  gmsh then meshes the volume between it and the outer
#  boundary and leaves the given surface meshes untouched, which is exactly what
#  is wanted: the boundary is ours, the interior is gmsh's.
#
#  ## Size control is radial, and that is not laziness
#
#  The interior size comes from a purely radial `MathEval` background field
#  rather than a `Distance` field. A `Distance` field would have to sample the
#  discrete surface; the cell is star-shaped about the origin, so a radial law is
#  *exact* and free:
#
#      h(r) = h_in + (h_out - h_in) · clamp((r - a)/(R - a), 0, 1)^grading
#
#  The two surface meshes fix the size on the two boundaries, so the field only
#  governs the interior. A large ratio between the surface resolutions is
#  therefore not a constraint on the interior grading.
#
#  The `gmsh` module is passed as an argument rather than imported, exactly as in
#  `crack_gmsh_geometry.jl` and `axi_gmsh_geometry.jl`: it is reached as
#  `Gmsh.gmsh` or as `GridapGmsh.gmsh` depending on the backend, both sitting on
#  the same `gmsh_jll`. The caller owns `initialize` and `finalize`.
# =============================================================================

# Surface, volume and physical-group tags. The `CELL_` prefix keeps them apart
# from the crack family's `TAG_*` and the axisymmetric family's `AXI_TAG_*`,
# which live in this same module.
const CELL_TAG_INCLUSION = 1
const CELL_TAG_OUTER = 2
const CELL_TAG_MATRIX = 1
const CELL_SET_INCLUSION = "inclusion"
const CELL_SET_OUTER = "outer"
const CELL_SET_MATRIX = "matrix"

"""
    FECellMeshOptions(; kwargs...)

Geometry and resolution of the truncated cell around a supershape. Every size is
a knob: the machine budget is a hard constraint, and the useful resolution is
not known before a convergence study.

| Keyword | Default | Meaning |
|:--|:--|:--|
| `radius_ratio` | `4.0` | outer radius as a multiple of the shape's **bounding** radius |
| `level` | `4` | subdivision level of the inclusion surface: ``8 \\cdot 4^{\\text{level}}`` triangles |
| `outer_level` | `3` | subdivision level of the outer sphere |
| `relax` | `60` | tangential relaxation sweeps on the inclusion surface |
| `h_in`, `h_out` | `nothing` | interior element size at each boundary; derived from the surface meshes when `nothing` |
| `grading` | `1.5` | exponent of the radial size law |
| `order` | `2` | element order; `2` is required for a curved boundary |
| `algorithm3d` | `1` | gmsh `Mesh.Algorithm3D`: `1` Delaunay (robust with a fixed boundary), `10` HXT (faster) |
| `optimize` | `true` | run gmsh's tetrahedron optimizer |
| `max_dofs` | `200_000` | refuse to factorize above this |
| `min_free_gb` | `6.0` | refuse to start a factorization with less than this available |
| `verbose` | `false` | let gmsh talk |

Three of these deserve more than a line.

**`outer_level` is not a free knob.** At `outer_level = 2` the outer boundary
carries 128 curved triangles, which does not resolve the ``1/r^2`` dipole term
of the corrected boundary condition: the exact spherical-pore gate then stalls
at ``-2.6\\times10^{-4}``. Level 3 takes it to ``-6.9\\times10^{-6}``.

**`radius_ratio` multiplies the bounding radius, not `a`.** For an elongated
superspheroid the two differ by the aspect ratio, and using `a` lets the outer
boundary come within ``1.2c`` of the body — where the exact spheroid gate falls
from ``10^{-5}`` to ``5.7\\times10^{-3}``. For a supersphere with ``p \\le 1``
the bounding radius **is** ``a``, so the literature's ``R/a`` convention is
preserved exactly where the literature uses it.

**`max_dofs` and `min_free_gb` are not performance knobs.** See
[`fe_available_gb`](@ref): they exist because an out-of-memory kill takes the
session with it.
"""
Base.@kwdef struct FECellMeshOptions
    radius_ratio::Float64 = 4.0
    level::Int = 4
    outer_level::Int = 3
    relax::Int = 60
    h_in::Union{Nothing, Float64} = nothing
    h_out::Union{Nothing, Float64} = nothing
    grading::Float64 = 1.5
    order::Int = 2
    algorithm3d::Int = 1
    optimize::Bool = true
    max_dofs::Int = 200_000
    min_free_gb::Float64 = 6.0
    verbose::Bool = false
end

"""
    _cell_outer_radius(shape, opts) -> R

`radius_ratio` times the shape's [`bounding_radius`](@ref MeanFieldHomogenization.Superspheres.bounding_radius) — see
[`FECellMeshOptions`](@ref) for why it is not `a`.
"""
_cell_outer_radius(shape::AbstractSuperShape, opts::FECellMeshOptions) =
    opts.radius_ratio * bounding_radius(shape)

"""
    _build_gmsh_cell_model(gmsh, shape, opts) -> NamedTuple

Build and mesh the cell in the **already initialized** gmsh session. Returns the
two surfaces actually used, the radius, the applied sizes and the snapping
report.

The meshed domain is the matrix **shell** between the inclusion surface and the
outer sphere: a pore is not meshed at all, its boundary being naturally
flux-free in conduction and traction-free in elasticity. That is why the volume
is built from two surface loops with the outer one first — the inclusion is a
hole.

`cavity_volume` is the volume of the body the solve **actually sees**, measured
on the curved boundary once it has been snapped. Normalizing a localization
tensor by the *exact* volume while solving on the meshed one would fold a
geometry error into the answer with nothing to reveal it; reporting both is what
makes that error visible instead.
"""
function _build_gmsh_cell_model(
        gmsh, shape::AbstractSuperShape, opts::FECellMeshOptions = FECellMeshOptions()
    )
    opts.order ≥ 1 || throw(ArgumentError("order must be ≥ 1, got $(opts.order)"))
    R = _cell_outer_radius(shape, opts)
    inner = shape_surface(shape, opts.level; relax = opts.relax)
    outer = unit_octahedron(opts.outer_level)
    for i in eachindex(outer.nodes)
        n = outer.nodes[i]
        outer.nodes[i] = (R * n[1], R * n[2], R * n[3])
    end

    hin = opts.h_in === nothing ? mesh_quality(inner).hmean : opts.h_in
    hout = opts.h_out === nothing ? mesh_quality(outer).hmean : opts.h_out

    gmsh.model.add("mfh_supershape_cell")
    _add_discrete_surface!(gmsh, inner, CELL_TAG_INCLUSION, 0)
    _add_discrete_surface!(gmsh, outer, CELL_TAG_OUTER, node_count(inner))

    gmsh.model.mesh.createTopology()
    slo = gmsh.model.geo.addSurfaceLoop([CELL_TAG_OUTER])
    sli = gmsh.model.geo.addSurfaceLoop([CELL_TAG_INCLUSION])
    gmsh.model.geo.addVolume([slo, sli], CELL_TAG_MATRIX)
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(
        2, [CELL_TAG_INCLUSION], CELL_TAG_INCLUSION, CELL_SET_INCLUSION
    )
    gmsh.model.addPhysicalGroup(2, [CELL_TAG_OUTER], CELL_TAG_OUTER, CELL_SET_OUTER)
    gmsh.model.addPhysicalGroup(3, [CELL_TAG_MATRIX], CELL_TAG_MATRIX, CELL_SET_MATRIX)

    _set_cell_size_field!(gmsh, shape.a, R, hin, hout, opts.grading)
    gmsh.option.setNumber("Mesh.Algorithm3D", opts.algorithm3d)
    gmsh.option.setNumber("Mesh.Optimize", opts.optimize ? 1 : 0)
    gmsh.model.mesh.generate(3)

    snap = nothing
    if opts.order > 1
        gmsh.model.mesh.setOrder(opts.order)
        snap = _snap_cell_surface_to_shape!(gmsh, shape, CELL_TAG_INCLUSION)
        # The outer boundary is a subdivided octahedron, so without this it is a
        # coarse polyhedron and `radius_ratio` would not name a radius. The
        # corrected boundary condition itself does not care — it evaluates the
        # dipole field at whatever position each boundary node actually has —
        # but an R/a sweep has to mean what it says.
        _snap_cell_surface_to_sphere!(gmsh, R, CELL_TAG_OUTER)
    end

    cavity_volume = if opts.order > 1
        fe_cell_curved_volume(gmsh, CELL_TAG_INCLUSION)
    else
        mesh_volume(inner)
    end

    return (; R, inner, outer, h_in = hin, h_out = hout, snap, cavity_volume)
end

function _add_discrete_surface!(gmsh, surf::TriSurface, tag::Integer, tag_offset::Integer)
    gmsh.model.addDiscreteEntity(2, tag)
    n = node_count(surf)
    ntags = collect((tag_offset + 1):(tag_offset + n))
    coords = Vector{Float64}(undef, 3n)
    @inbounds for (i, v) in enumerate(surf.nodes)
        coords[3i - 2] = v[1]
        coords[3i - 1] = v[2]
        coords[3i] = v[3]
    end
    gmsh.model.mesh.addNodes(2, tag, ntags, coords)
    conn = Vector{Int}(undef, 3 * triangle_count(surf))
    @inbounds for (i, t) in enumerate(surf.tris)
        conn[3i - 2] = tag_offset + t[1]
        conn[3i - 1] = tag_offset + t[2]
        conn[3i] = tag_offset + t[3]
    end
    # Element type 2 is the 3-node triangle; element tags are left to gmsh.
    gmsh.model.mesh.addElementsByType(tag, 2, Int[], conn)
    return tag
end

function _set_cell_size_field!(gmsh, a, R, hin, hout, grading)
    # The surface meshes already fix the size on both boundaries, so switching
    # the default mechanisms off makes the field the single authority in
    # between, which is the only way the grading is predictable.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    f = gmsh.model.mesh.field.add("MathEval")
    # Every interpolated number is parenthesized. A *negative* coefficient is
    # legitimate — it means the outer surface carries finer elements than the
    # inclusion, which happens at a small `radius_ratio` with a fine
    # `outer_level` — but unparenthesized it produces `... + -0.0013 * ...`,
    # which gmsh's `mathex` parser rejects with `Invalid expression` and an
    # **abort**, not a Julia exception. Found at `level = 2, radius_ratio = 2`.
    expr = "($(hin)) + ($(hout - hin))*(Min(Max((Sqrt(x*x+y*y+z*z)-($(a)))/($(R - a)),0),1))^($(grading))"
    gmsh.model.mesh.field.setString(f, "F", expr)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    return f
end

"""
    fe_cell_size_estimate(shape, opts) -> NamedTuple

Predict `(; ntets, nnodes_p1, nnodes_p2, dofs_scalar, dofs_vector, …)`
**without meshing**, by integrating the radial size law:

```math
N_{\\text{tets}} \\approx \\int \\frac{4\\pi r^2\\,\\mathrm dr}{h(r)^3/(6\\sqrt2)} .
```

The point is to find out that a set of options will exhaust the machine *before*
gmsh spends ten minutes proving it. It is an estimate, not a measurement, and it
**over-predicts by about a factor of three**: at ``p = 0.3``, level 3,
`radius_ratio = 3`, it announces 184 000 vector degrees of freedom where the
mesh actually carries 61 400. That direction is the desired failure mode for
something whose job is to refuse a solve — but the factor is stated rather than
called "of order one", so that nobody sizes a study from it.
"""
function fe_cell_size_estimate(
        shape::AbstractSuperShape, opts::FECellMeshOptions = FECellMeshOptions()
    )
    R = _cell_outer_radius(shape, opts)
    inner = shape_surface(shape, opts.level; relax = opts.relax)
    outer = unit_octahedron(opts.outer_level)
    hin = opts.h_in === nothing ? mesh_quality(inner).hmean : opts.h_in
    # `outer` is unscaled here, unlike in `_build_gmsh_cell_model` where its
    # nodes have already been multiplied by R.
    hout = opts.h_out === nothing ? mesh_quality(outer).hmean * R : opts.h_out
    a = shape.a
    n = 20_000
    vtet(h) = h^3 / (6 * sqrt(2.0))
    acc = 0.0
    r0 = equivalent_sphere_radius(shape)
    for k in 1:n
        r = r0 + (R - r0) * (k - 0.5) / n
        h = hin + (hout - hin) * clamp((r - a) / (R - a), 0, 1)^opts.grading
        acc += 4π * r^2 * ((R - r0) / n) / vtet(h)
    end
    ntets = round(Int, acc)
    np1 = round(Int, ntets / 5.5)              # about 5.5 tetrahedra per node in 3-D
    np2 = np1 + round(Int, 1.15 * ntets)       # one node per edge, about 7 edges per node / 2
    nn = opts.order > 1 ? np2 : np1
    return (;
        ntets, nnodes_p1 = np1, nnodes_p2 = np2,
        dofs_scalar = nn, dofs_vector = 3nn,
        R, h_in = hin, h_out = hout,
    )
end

# ─── The curved boundary ─────────────────────────────────────────────────────

"""
    _snap_cell_surface_to_shape!(gmsh, shape, tag; tol, passes) -> NamedTuple

Move the mid-edge nodes of a second-order surface onto the exact shape, backing
off wherever that would invert a neighboring element.

`gmsh.model.mesh.setOrder(2)` places every mid-edge node at the midpoint of the
straight segment. It cannot do better: the surface was handed to it as a
discrete entity, so it has no idea what surface the node is supposed to be on.
Projecting those nodes radially — exact and single-valued here, the body being
star-shaped — turns the boundary into a genuine second-order approximation and
takes the geometry error from ``O(h^2)`` to ``O(h^3)``. Measured on this family
it is worth about a factor of 16 in cost at equal geometric accuracy.

## Why it is safeguarded

The displacement is ``O(h^2)``, so asymptotically nothing can be inverted. That
argument fails on a *concave* shape, where the curvature near the conical points
on the axes is not small against ``h`` — and, unlike an ordinary
under-resolution, it never becomes small, because at a conical point the
curvature is unbounded.

So each node carries a blend factor. All are snapped fully, the element
qualities are read back, and any node belonging to an element below `tol` has
its factor halved; repeat. `limited` counts the nodes that had to back off and
`min_blend` says how far the worst one did.

**Refinement does not remove it, and it is not meant to.** Measured on
``p = 0.35``, `radius_ratio = 3`:

| level | tetrahedra | `limited` | fraction | `min_blend` |
|--:|--:|--:|--:|--:|
| 2 | 5 604 | 6 / 258 | 2.3 % | `0.5` |
| 3 | 21 736 | 37 / 1026 | 3.6 % | `0.25` |
| 4 | 102 641 | 73 / 4098 | 1.8 % | `0.125` |

The count grows and the worst blend halves at every level, because the same
non-differentiable feature is being approached by ever-smaller elements whose
curvature relative to it stays infinite. What is stable is the **fraction**, a
few percent, and that is the number to watch: a large fraction means the level
really is too coarse for that ``p``, a few percent means the conical points are
being handled as they must be.

What separates a shape that needs the safeguard from one that does not is
unbounded curvature, **not** sharpness. At level 3 the sphere, the mildly
concave ``p = 0.45``, and the convex ``p = 2`` all report `limited = 0`; only
``p = 0.35`` limits. And the octahedron, ``p = 1/2``, reports `maxd = 1.2e-16`:
it has edges and vertices everywhere, yet there is nothing to snap at all,
because its faces are flat and the mid-edge nodes are already exactly on the
surface. Edges are harmless; conical points are not.

The nodes are shared with the adjacent tetrahedra, so `setNode` moves the volume
mesh along with the surface. That is the point, not a side effect.
"""
function _snap_cell_surface_to_shape!(
        gmsh, shape::AbstractSuperShape, tag::Integer;
        quality::AbstractString = "minSICN", tol::Real = 0.0, passes::Integer = 12,
    )
    ntags, coords, _ = gmsh.model.mesh.getNodes(2, tag, true, false)
    n = length(ntags)
    orig = Vector{NTuple{3, Float64}}(undef, n)
    target = Vector{NTuple{3, Float64}}(undef, n)
    for i in 1:n
        x = (coords[3i - 2], coords[3i - 1], coords[3i])
        orig[i] = x
        r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
        target[i] = r > 0 ? surface_point(shape, (x[1] / r, x[2] / r, x[3] / r)) : x
    end
    slot = Dict{Int, Int}(Int(t) => i for (i, t) in enumerate(ntags))
    lam = ones(Float64, n)
    _apply_cell_blend!(gmsh, ntags, orig, target, lam, 1:n)

    etypes, etags, enodes = gmsh.model.mesh.getElements(3, CELL_TAG_MATRIX)
    limited = 0
    for _ in 1:passes
        bad = Int[]
        for (k, et) in enumerate(etypes)
            nper = et == 11 ? 10 : et == 4 ? 4 : 0
            nper == 0 && continue
            q = gmsh.model.mesh.getElementQualities(etags[k], quality)
            conn = enodes[k]
            for (e, qe) in enumerate(q)
                qe > tol && continue
                o = nper * (e - 1)
                for j in 1:nper
                    i = get(slot, Int(conn[o + j]), 0)
                    i == 0 || push!(bad, i)
                end
            end
        end
        isempty(bad) && break
        unique!(bad)
        for i in bad
            lam[i] /= 2
        end
        limited = count(<(1.0), lam)
        _apply_cell_blend!(gmsh, ntags, orig, target, lam, bad)
    end

    maxd = 0.0
    for i in 1:n
        d = sqrt(sum(abs2, lam[i] .* (target[i] .- orig[i])))
        maxd = max(maxd, d)
    end
    return (; moved = n, total = n, maxd, limited, min_blend = minimum(lam))
end

function _apply_cell_blend!(gmsh, ntags, orig, target, lam, idx)
    buf = zeros(Float64, 3)
    empt = Float64[]
    for i in idx
        l = lam[i]
        for k in 1:3
            buf[k] = orig[i][k] + l * (target[i][k] - orig[i][k])
        end
        gmsh.model.mesh.setNode(ntags[i], buf, empt)
    end
    return nothing
end

"""
    _snap_cell_surface_to_sphere!(gmsh, R, tag) -> NamedTuple

The counterpart of [`_snap_cell_surface_to_shape!`](@ref) for the outer
boundary: push every node of a second-order surface onto the sphere of radius
`R`. No safeguard is needed — the outer mesh is uniform and far from anything
sharp.
"""
function _snap_cell_surface_to_sphere!(gmsh, R::Real, tag::Integer)
    ntags, coords, _ = gmsh.model.mesh.getNodes(2, tag, true, false)
    moved = 0
    maxd = 0.0
    buf = zeros(Float64, 3)
    empt = Float64[]
    for (i, t) in enumerate(ntags)
        x = (coords[3i - 2], coords[3i - 1], coords[3i])
        r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
        r > 0 || continue
        d = abs(R - r)
        d > 1.0e-14 || continue
        buf[1] = R * x[1] / r
        buf[2] = R * x[2] / r
        buf[3] = R * x[3] / r
        gmsh.model.mesh.setNode(t, buf, empt)
        moved += 1
        maxd = max(maxd, d)
    end
    return (; moved, total = length(ntags), maxd)
end

# Quadratic (6-node) triangle shape functions, in gmsh's node ordering
# (v1, v2, v3, e12, e23, e31).
function _p2tri(ξ, η)
    l1 = 1 - ξ - η
    l2 = ξ
    l3 = η
    return (
        l1 * (2l1 - 1), l2 * (2l2 - 1), l3 * (2l3 - 1),
        4l1 * l2, 4l2 * l3, 4l3 * l1,
    )
end

_cdot(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
_ccross(a, b) = (
    a[2] * b[3] - a[3] * b[2],
    a[3] * b[1] - a[1] * b[3],
    a[1] * b[2] - a[2] * b[1],
)

"""
    fe_cell_curved_volume(gmsh, tag; sub = 8) -> V

Volume enclosed by a **second-order** surface,
``\\frac13\\int \\underline x \\cdot \\underline n\\,\\mathrm dS``, evaluated by
splitting every 6-node triangle into ``\\texttt{sub}^2`` flat sub-triangles
through the quadratic map and summing the signed cones from the origin.

This is what measures the gain from snapping the boundary: the flat
[`mesh_volume`](@ref MeanFieldHomogenization.Superspheres.mesh_volume) cannot see a curved boundary at all, and would report the
same number before and after.
"""
function fe_cell_curved_volume(gmsh, tag::Integer; sub::Integer = 8)
    etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
    j = findfirst(==(9), etypes)          # 9 = 6-node second-order triangle
    j === nothing && throw(
        ArgumentError("surface $tag carries no 6-node triangles; was `setOrder(2)` run?")
    )
    conn = enodes[j]
    ntags, coords, _ = gmsh.model.mesh.getNodes(2, tag, true, false)
    pos = Dict{Int, NTuple{3, Float64}}()
    for (i, t) in enumerate(ntags)
        pos[Int(t)] = (coords[3i - 2], coords[3i - 1], coords[3i])
    end
    acc = 0.0
    nel = length(conn) ÷ 6
    for e in 1:nel
        p = ntuple(k -> pos[Int(conn[6(e - 1) + k])], 6)
        at(ξ, η) = begin
            N = _p2tri(ξ, η)
            (
                sum(N[k] * p[k][1] for k in 1:6),
                sum(N[k] * p[k][2] for k in 1:6),
                sum(N[k] * p[k][3] for k in 1:6),
            )
        end
        h = 1 / sub
        for i in 0:(sub - 1), jj in 0:(sub - 1 - i)
            a = at(i * h, jj * h)
            b = at((i + 1) * h, jj * h)
            c = at(i * h, (jj + 1) * h)
            acc += _cdot(a, _ccross(b, c))
            if i + jj < sub - 1
                d = at((i + 1) * h, (jj + 1) * h)
                acc += _cdot(b, _ccross(d, c))
            end
        end
    end
    return acc / 6
end

"""
    fe_cell_meshed_volume(gmsh, tag = CELL_TAG_MATRIX) -> V

Volume of the meshed region, summed over its tetrahedra from their four corner
nodes.

`gmsh.model.occ.getMass` is not an option: the volume lives in the `geo` kernel
and is bounded by *discrete* surfaces, which OCC knows nothing about. Straight
tetrahedra also mean this slightly under-reports a second-order mesh with a
curved boundary, by the same ``O(h^3)`` that snapping buys back on the surface —
so use it as a check on the **topology**, not as a precision measurement.
"""
function fe_cell_meshed_volume(gmsh, tag::Integer = CELL_TAG_MATRIX)
    etypes, _, enodes = gmsh.model.mesh.getElements(3, tag)
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    pos = Dict{Int, NTuple{3, Float64}}()
    for (i, t) in enumerate(ntags)
        pos[Int(t)] = (coords[3i - 2], coords[3i - 1], coords[3i])
    end
    acc = 0.0
    for (k, et) in enumerate(etypes)
        nper = et == 4 ? 4 : et == 11 ? 10 : 0     # 4-node tet, 10-node tet
        nper == 0 && continue
        conn = enodes[k]
        for e in 1:(length(conn) ÷ nper)
            o = nper * (e - 1)
            a, b, c, d = ntuple(i -> pos[Int(conn[o + i])], 4)
            u = (b[1] - a[1], b[2] - a[2], b[3] - a[3])
            v = (c[1] - a[1], c[2] - a[2], c[3] - a[3])
            w = (d[1] - a[1], d[2] - a[2], d[3] - a[3])
            acc += abs(_cdot(u, _ccross(v, w))) / 6
        end
    end
    return acc
end
