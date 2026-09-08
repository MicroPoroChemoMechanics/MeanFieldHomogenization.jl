# =============================================================================
#  cell_octant.jl — one eighth of the truncated cell, and why that is exact.
#
#  A supershape whose three coordinate planes are mirror planes sits in a cell
#  -- shape, concentric outer sphere, isotropic reference -- invariant under the
#  group `{diag(±1,±1,±1)}` of order 8. Every Kelvin load case is an eigenvector
#  of that group, and `cell_driver.jl` shows the dipole correction transforms by
#  the same law as the remote field. So one octant, with a symmetry or
#  antisymmetry condition on each plane, carries the whole answer.
#
#  ## The shape of the domain changes, not just its size
#
#  The full cell is a *shell*: `addVolume([outer_loop, inner_loop])`, the second
#  loop being a hole. An octant is simply connected, and its boundary is a
#  single closed surface in **five** pieces -- two curved caps and three flat
#  faces. So it is one surface loop, and the flat faces have to be built.
#
#  ## Three things that come for free, and they are what make this cheap
#
#  * **The caps already exist.** `octant_patch` was written for this, its face
#    edges lying exactly in the coordinate planes, and `shape_surface(shape,
#    level; octant = true)` already returns the inner one.
#  * **Snapping preserves the planes exactly.** `_snap_cell_surface_to_shape!`
#    scales a node radially, and `radial_distance * 0.0 == 0.0` in floating
#    point. A node of a plane stays in its plane, bit for bit, through the
#    second-order promotion and both snapping passes -- mid-edge nodes included,
#    being midpoints of coplanar pairs. No new snapping code, and a free test.
#  * **`fe_cell_curved_volume` already returns `V/8`.** It integrates
#    `⅓∮ x·n dS` over the cap alone; the flat faces pass through the origin and
#    contribute nothing. That is the identity `mesh_volume` documents.
#
#  ## What has to be built, and the trap that disappears
#
#  The three flat faces. Each is bounded by four chains: the inner cap's trace
#  in that plane, a radial segment out along one axis, the outer cap's trace,
#  and a radial segment back.
#
#  The obvious construction -- a structured polar grid between the two traces --
#  needs the two to carry the *same number of nodes*, which would force
#  `level == outer_level` and throw away the whole point: `outer_level` is a
#  floor set by resolving the `1/r²` dipole term, while `level` is the
#  convergence knob. Meshing each flat face as an **unstructured** planar
#  region instead makes the question vanish rather than answering it: a Delaunay
#  mesher takes a contour whose four pieces have different densities without
#  complaint, and the same radial size field governs the transition.
#
#  The faces are meshed in an auxiliary gmsh model whose boundary nodes are
#  pinned -- one `setTransfiniteCurve(·, 2)` per segment -- so the contour comes
#  back exactly as it went in, and the caps and faces share nodes rather than
#  two discretizations of one curve.
# =============================================================================

# Tags. Surfaces 1-2 are the caps, 3-5 the planes; curves and points live in
# their own dimensional tag spaces but are numbered apart for legible gmsh
# messages.
const CELL_TAG_PLANE = (3, 4, 5)
const CELL_SET_PLANE = ("plane_x", "plane_y", "plane_z")
const _OCT_CURVE_TAGS = 11:19
const _OCT_POINT_TAGS = 21:26

"""
    OctantShell

The closed boundary of one octant of a cell, as a single deduplicated node list
with five triangulated faces, nine ordered curves and six corners.

Assembling it in Julia rather than welding it in gmsh is deliberate: no node is
created twice, so watertightness is a property of the data structure and can be
asserted **before** any mesher runs. `_octant_closure_defect` is that assertion.

Indexing, fixed once here and relied on throughout:

| | |
|:--|:--|
| `faces[1]` | the inclusion cap, oriented **inward** (it bounds a cavity) |
| `faces[2]` | the outer cap, oriented outward |
| `faces[2+k]` | the flat face in the plane ``x_k = 0``, normal ``-e_k`` |
| `curves[k]` | inner cap trace in the plane ``x_k = 0`` |
| `curves[3+k]` | outer cap trace in the same plane |
| `curves[6+k]` | radial segment along axis `k` |
| `corners[k]` | inner axis vertex on axis `k` |
| `corners[3+k]` | outer axis vertex on axis `k` |
"""
struct OctantShell
    nodes::Vector{NTuple{3, Float64}}
    faces::NTuple{5, Vector{NTuple{3, Int}}}
    curves::NTuple{9, Vector{Int}}
    corners::NTuple{6, Int}
end

"""
    _octant_radial_nodes(r0, R, hin, hout, grading) -> Vector{Float64}

Radii of the nodes of one radial segment, from `r0` to `R`, spaced by the same
law `_set_cell_size_field!` imposes in the interior.

Computed once and imposed on **both** flat faces that share the segment. Letting
gmsh choose it independently in two meshes would give two discretizations of one
curve, and the 3-D mesher would still produce *something* -- a leak that does
not announce itself.
"""
function _octant_radial_nodes(r0::Real, R::Real, hin::Real, hout::Real, grading::Real)
    h(r) = hin + (hout - hin) * clamp((r - r0) / (R - r0), 0, 1)^grading
    # Number of intervals from the integral of dr/h(r), then equal steps in that
    # parameter so the spacing follows the law rather than being uniform.
    nsub = 512
    rs = range(r0, R; length = nsub + 1)
    s = zeros(nsub + 1)
    for i in 1:nsub
        s[i + 1] = s[i] + (rs[i + 1] - rs[i]) * 0.5 * (inv(h(rs[i])) + inv(h(rs[i + 1])))
    end
    n = max(1, round(Int, s[end]))
    out = Vector{Float64}(undef, n + 1)
    out[1], out[end] = r0, R
    j = 1
    for i in 1:(n - 1)
        target = s[end] * i / n
        while j < nsub && s[j + 1] < target
            j += 1
        end
        t = (target - s[j]) / (s[j + 1] - s[j])
        out[i + 1] = rs[j] + t * (rs[j + 1] - rs[j])
    end
    return out
end

"""
    _octant_closure_defect(shell) -> Float64

Relative defect of the divergence identity over the shell's five faces.

``\\tfrac16 \\sum_{\\text{tri}} a\\cdot(b\\times c)`` is the enclosed volume when
the boundary is closed and consistently oriented outward, and the flat faces
pass through the origin so they contribute nothing. It must therefore equal the
volume between the two caps. A flipped face, a duplicated node or a chain
traversed backwards all show up here — before meshing, in milliseconds.
"""
function _octant_closure_defect(shell::OctantShell)
    v = 0.0
    for f in shell.faces, (i, j, k) in f
        a, b, c = shell.nodes[i], shell.nodes[j], shell.nodes[k]
        v += (
            a[1] * (b[2] * c[3] - b[3] * c[2]) -
                a[2] * (b[1] * c[3] - b[3] * c[1]) +
                a[3] * (b[1] * c[2] - b[2] * c[1])
        )
    end
    v /= 6
    # The same volume, read off the two caps independently.
    cap(f, sgn) = begin
        s = 0.0
        for (i, j, k) in f
            a, b, c = shell.nodes[i], shell.nodes[j], shell.nodes[k]
            s += (
                a[1] * (b[2] * c[3] - b[3] * c[2]) -
                    a[2] * (b[1] * c[3] - b[3] * c[1]) +
                    a[3] * (b[1] * c[2] - b[2] * c[1])
            )
        end
        return sgn * s / 6
    end
    ref = cap(shell.faces[2], 1) + cap(shell.faces[1], 1)
    return abs(v - ref) / max(abs(ref), eps())
end

# Orientation helper: flip a triangle list so that its outward normal agrees
# with `want(centroid)`.
function _orient!(tris, nodes, want)
    for (n, (i, j, k)) in enumerate(tris)
        a, b, c = nodes[i], nodes[j], nodes[k]
        u = (b[1] - a[1], b[2] - a[2], b[3] - a[3])
        v = (c[1] - a[1], c[2] - a[2], c[3] - a[3])
        nx = u[2] * v[3] - u[3] * v[2]
        ny = u[3] * v[1] - u[1] * v[3]
        nz = u[1] * v[2] - u[2] * v[1]
        g = ((a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3, (a[3] + b[3] + c[3]) / 3)
        w = want(g)
        if nx * w[1] + ny * w[2] + nz * w[3] < 0
            tris[n] = (i, k, j)
        end
    end
    return tris
end

"""
    _mesh_octant_plane(gmsh, loop, hs, field) -> (interior, tris)

Mesh one flat face, given its closed boundary polyline `loop` (3-D points, first
not repeated at the end) and the target size `hs` at each of them.

Done in a throw-away model so the main one stays untouched, and with every
boundary segment made transfinite with two nodes: gmsh must return the contour
exactly as given, or the face and the cap it meets would no longer share nodes.
`tris` indexes `[loop; interior]`.
"""
function _mesh_octant_plane(gmsh, loop::Vector{NTuple{3, Float64}}, hs::Vector{Float64}, field)
    name = "mfh_octant_face_" * string(hash((loop, hs)); base = 16)
    gmsh.model.add(name)
    geo = gmsh.model.geo
    n = length(loop)
    pts = [geo.addPoint(loop[i][1], loop[i][2], loop[i][3], hs[i]) for i in 1:n]
    lines = [geo.addLine(pts[i], pts[mod1(i + 1, n)]) for i in 1:n]
    cl = geo.addCurveLoop(lines)
    surf = geo.addPlaneSurface([cl])
    geo.synchronize()
    for l in lines
        gmsh.model.mesh.setTransfiniteCurve(l, 2)
    end
    field(gmsh)
    gmsh.model.mesh.generate(2)

    # The corner nodes, in loop order, so the mapping back is by construction.
    corner_tag = Vector{Int}(undef, n)
    for i in 1:n
        t, _, _ = gmsh.model.mesh.getNodes(0, pts[i])
        corner_tag[i] = Int(t[1])
    end
    idx = Dict{Int, Int}(corner_tag[i] => i for i in 1:n)

    interior = NTuple{3, Float64}[]
    ntags, coords, _ = gmsh.model.mesh.getNodes(2, surf, true, false)
    for (a, t) in enumerate(ntags)
        haskey(idx, Int(t)) && continue
        push!(interior, (coords[3a - 2], coords[3a - 1], coords[3a]))
        idx[Int(t)] = n + length(interior)
    end

    tris = NTuple{3, Int}[]
    etags, econn = gmsh.model.mesh.getElementsByType(2, surf)
    for e in 1:length(etags)
        push!(
            tris,
            (
                idx[Int(econn[3e - 2])], idx[Int(econn[3e - 1])], idx[Int(econn[3e])],
            )
        )
    end

    gmsh.model.remove()
    return interior, tris
end

"""
    _octant_shell(gmsh, shape, R, opts) -> (; shell, inner, outer)

Assemble the five-faced boundary of one octant: the two caps as they come from
`Superspheres`, the three flat faces meshed between them, and a single
deduplicated node list in which every shared node appears once.
"""
function _octant_shell(gmsh, shape::AbstractSuperShape, R::Real, opts::FECellMeshOptions)
    inner = shape_surface(shape, opts.level; octant = true, relax = opts.relax)
    outer = octant_patch(opts.outer_level)
    for i in eachindex(outer.nodes)
        v = outer.nodes[i]
        outer.nodes[i] = (R * v[1], R * v[2], R * v[3])
    end

    hin = opts.h_in === nothing ? mesh_quality(inner).hmean : opts.h_in
    hout = opts.h_out === nothing ? mesh_quality(outer).hmean : opts.h_out
    a = shape.a
    hlaw(r) = hin + (hout - hin) * clamp((r - a) / (R - a), 0, 1)^opts.grading
    field = g -> _set_cell_size_field!(g, a, R, hin, hout, opts.grading)

    ch_in, ch_out = boundary_chains(inner), boundary_chains(outer)
    co_in, co_out = patch_corners(inner), patch_corners(outer)

    # One global node list. The caps go in whole; the flat faces contribute
    # only their interiors, their boundaries being nodes already present.
    nodes = NTuple{3, Float64}[]
    off_in = 0
    append!(nodes, inner.nodes)
    off_out = length(nodes)
    append!(nodes, outer.nodes)

    faces_in = [(t[1] + off_in, t[3] + off_in, t[2] + off_in) for t in inner.tris]
    faces_out = [(t[1] + off_out, t[2] + off_out, t[3] + off_out) for t in outer.tris]

    # The three radial segments, computed once and shared by two faces each.
    radial = ntuple(3) do k
        r0 = norm3(inner.nodes[co_in[k]])
        rs = _octant_radial_nodes(r0, R, hlaw(r0), hlaw(R), opts.grading)
        ids = Vector{Int}(undef, length(rs))
        ids[1] = co_in[k] + off_in
        ids[end] = co_out[k] + off_out
        for i in 2:(length(rs) - 1)
            push!(nodes, ntuple(l -> l == k ? rs[i] : 0.0, 3))
            ids[i] = length(nodes)
        end
        ids
    end

    planes = Vector{NTuple{3, Int}}[]
    for k in 1:3
        others = [l for l in 1:3 if l != k]
        a2, b2 = others[1], others[2]
        cin = [i + off_in for i in ch_in[k]]                       # corner a2 → b2
        cout = [i + off_out for i in ch_out[k]]                    # corner a2 → b2
        loop_ids = vcat(
            cin[1:(end - 1)],                                       # a2 → b2, inner
            radial[b2][1:(end - 1)],                                # inner b2 → outer b2
            reverse(cout)[1:(end - 1)],                             # outer b2 → a2
            reverse(radial[a2])[1:(end - 1)],                       # outer a2 → inner a2
        )
        loop = [nodes[i] for i in loop_ids]
        hs = [hlaw(norm3(p)) for p in loop]
        interior, tris = _mesh_octant_plane(gmsh, loop, hs, field)
        base = length(nodes)
        append!(nodes, interior)
        gid(i) = i ≤ length(loop_ids) ? loop_ids[i] : base + (i - length(loop_ids))
        push!(planes, [(gid(t[1]), gid(t[2]), gid(t[3])) for t in tris])
    end

    faces = (faces_in, faces_out, planes[1], planes[2], planes[3])
    _orient!(faces[1], nodes, g -> (-g[1], -g[2], -g[3]))          # into the cavity
    _orient!(faces[2], nodes, g -> g)                              # away from the origin
    for k in 1:3
        _orient!(faces[2 + k], nodes, _ -> ntuple(l -> l == k ? -1.0 : 0.0, 3))
    end

    curves = (
        [i + off_in for i in ch_in[1]], [i + off_in for i in ch_in[2]],
        [i + off_in for i in ch_in[3]],
        [i + off_out for i in ch_out[1]], [i + off_out for i in ch_out[2]],
        [i + off_out for i in ch_out[3]],
        radial[1], radial[2], radial[3],
    )
    corners = (
        co_in[1] + off_in, co_in[2] + off_in, co_in[3] + off_in,
        co_out[1] + off_out, co_out[2] + off_out, co_out[3] + off_out,
    )
    return (; shell = OctantShell(nodes, faces, curves, corners), inner, outer)
end

norm3(v) = sqrt(v[1]^2 + v[2]^2 + v[3]^2)

"""
    _add_octant_brep!(gmsh, shell)

Declare the shell to gmsh as an explicit boundary representation: six point
entities, nine curves, five surfaces, one volume.

`addDiscreteEntity` takes the bounding entities, so the topology is *stated*
rather than recovered. That is what makes the construction exact: every node is
added once, to the single entity that owns it, and nothing is welded by
coordinate tolerance. A node shared by two faces is a curve node, and a node
shared by two curves is a point.
"""
function _add_octant_brep!(gmsh, shell::OctantShell)
    nn = length(shell.nodes)
    owner = zeros(Int, nn)                    # 0 unassigned, else an entity slot
    POINT(i) = i                              # 1:6
    CURVE(i) = 6 + i                          # 7:15
    SURF(i) = 15 + i                          # 16:20

    for (i, c) in enumerate(shell.corners)
        owner[c] = POINT(i)
    end
    for (i, ch) in enumerate(shell.curves), n in ch[2:(end - 1)]
        owner[n] = CURVE(i)
    end
    for (i, f) in enumerate(shell.faces), t in f, n in t
        owner[n] == 0 && (owner[n] = SURF(i))
    end
    any(iszero, owner) && throw(
        ArgumentError("octant shell carries $(count(iszero, owner)) unclassified nodes")
    )

    coords(ids) = begin
        c = Vector{Float64}(undef, 3 * length(ids))
        for (a, i) in enumerate(ids)
            v = shell.nodes[i]
            c[3a - 2], c[3a - 1], c[3a] = v[1], v[2], v[3]
        end
        c
    end
    owned(slot) = findall(==(slot), owner)

    for i in 1:6
        tag = _OCT_POINT_TAGS[i]
        gmsh.model.addDiscreteEntity(0, tag)
        gmsh.model.mesh.addNodes(0, tag, [shell.corners[i]], coords([shell.corners[i]]))
    end

    # Which corners bound each curve, by construction of `_octant_shell`.
    curve_ends = (
        (2, 3), (1, 3), (1, 2),                       # inner traces, in plane k
        (5, 6), (4, 6), (4, 5),                       # outer traces
        (1, 4), (2, 5), (3, 6),                       # radial segments
    )
    for i in 1:9
        tag = _OCT_CURVE_TAGS[i]
        b = collect(_OCT_POINT_TAGS[j] for j in curve_ends[i])
        gmsh.model.addDiscreteEntity(1, tag, b)
        ids = owned(CURVE(i))
        isempty(ids) || gmsh.model.mesh.addNodes(1, tag, ids, coords(ids))
        ch = shell.curves[i]
        conn = Int[]
        for j in 1:(length(ch) - 1)
            push!(conn, ch[j], ch[j + 1])
        end
        gmsh.model.mesh.addElementsByType(tag, 1, Int[], conn)   # type 1: 2-node line
    end

    surf_curves = (
        (1, 2, 3), (4, 5, 6),
        (1, 4, 8, 9), (2, 5, 7, 9), (3, 6, 7, 8),
    )
    stags = (CELL_TAG_INCLUSION, CELL_TAG_OUTER, CELL_TAG_PLANE...)
    for i in 1:5
        tag = stags[i]
        b = collect(_OCT_CURVE_TAGS[j] for j in surf_curves[i])
        gmsh.model.addDiscreteEntity(2, tag, b)
        ids = owned(SURF(i))
        isempty(ids) || gmsh.model.mesh.addNodes(2, tag, ids, coords(ids))
        conn = Int[]
        for t in shell.faces[i]
            push!(conn, t[1], t[2], t[3])
        end
        gmsh.model.mesh.addElementsByType(tag, 2, Int[], conn)   # type 2: triangle
    end

    # The volume goes through the built-in kernel, exactly as the full cell does
    # (`cell_gmsh_geometry.jl`): `createTopology` exports the discrete surfaces
    # into it so their tags can enter a surface loop, and the 3-D algorithm only
    # fills a `geo` volume. One loop of five surfaces here, against two loops
    # (an outer and a hole) there — that is the whole structural difference.
    gmsh.model.mesh.createTopology()
    sl = gmsh.model.geo.addSurfaceLoop(collect(stags))
    gmsh.model.geo.addVolume([sl], CELL_TAG_MATRIX)
    gmsh.model.geo.synchronize()
    return nothing
end

"""
    _build_gmsh_cell_model_octant(gmsh, shape, R, opts) -> NamedTuple

The octant counterpart of `_build_gmsh_cell_model`, returning the same fields so
that everything downstream is indifferent to which one ran.

`cavity_volume` is the volume of the **whole** body, not of the eighth that was
meshed. It has two consumers pulling in opposite directions: the dipole moment
`𝔽 = ±V ℂ₀` needs the total volume, since the dipole is that of the entire
inclusion, while the surface average needs the meshed eighth. Returning the
total and dividing in exactly one place — the average — keeps the trap visible.
"""
function _build_gmsh_cell_model_octant(
        gmsh, shape::AbstractSuperShape, R::Real, opts::FECellMeshOptions
    )
    has_coordinate_mirrors(shape) || throw(
        ArgumentError(
            "`octant = true` requires the three coordinate planes to be mirror " *
                "planes of $(nameof(typeof(shape))), which `has_coordinate_mirrors` " *
                "does not confirm. Add a method if the shape has them — and run " *
                "`check_coordinate_mirrors` on it first. There is deliberately no " *
                "fallback to the full cell: silently meshing eight times what was " *
                "asked for is worse than refusing."
        )
    )

    built = _octant_shell(gmsh, shape, R, opts)
    shell, inner, outer = built.shell, built.inner, built.outer
    defect = _octant_closure_defect(shell)
    defect < 1.0e-9 || throw(
        ErrorException(
            "octant shell is not watertight or not consistently oriented: " *
                "divergence defect $(defect). This is a bug in `_octant_shell`, " *
                "caught before meshing."
        )
    )

    hin = opts.h_in === nothing ? mesh_quality(inner).hmean : opts.h_in
    hout = opts.h_out === nothing ? mesh_quality(outer).hmean : opts.h_out

    gmsh.model.add("mfh_supershape_cell_octant")
    _add_octant_brep!(gmsh, shell)
    gmsh.model.addPhysicalGroup(
        2, [CELL_TAG_INCLUSION], CELL_TAG_INCLUSION, CELL_SET_INCLUSION
    )
    gmsh.model.addPhysicalGroup(2, [CELL_TAG_OUTER], CELL_TAG_OUTER, CELL_SET_OUTER)
    for k in 1:3
        gmsh.model.addPhysicalGroup(2, [CELL_TAG_PLANE[k]], CELL_TAG_PLANE[k], CELL_SET_PLANE[k])
    end
    gmsh.model.addPhysicalGroup(3, [CELL_TAG_MATRIX], CELL_TAG_MATRIX, CELL_SET_MATRIX)

    _set_cell_size_field!(gmsh, shape.a, R, hin, hout, opts.grading)
    gmsh.option.setNumber("Mesh.Algorithm3D", opts.algorithm3d)
    gmsh.option.setNumber("Mesh.Optimize", opts.optimize ? 1 : 0)
    gmsh.model.mesh.generate(3)

    snap = nothing
    if opts.order > 1
        gmsh.model.mesh.setOrder(opts.order)
        # Both snappings scale a node radially, so a node with `x_k == 0.0`
        # keeps it exactly: the flat faces stay flat with no extra care.
        snap = _snap_cell_surface_to_shape!(gmsh, shape, CELL_TAG_INCLUSION)
        _snap_cell_surface_to_sphere!(gmsh, R, CELL_TAG_OUTER)
    end

    # `fe_cell_curved_volume` integrates ⅓∮x·n over the inclusion surface as the
    # mesh orients it. The octant orients all five faces **out of the matrix**,
    # which for the cavity wall means *into* the cavity, so the integral comes
    # back negative — the full cell, whose two-loop structure carries the
    # topology instead, leaves that wall pointing away from the origin. Hence
    # the sign, stated rather than absorbed into an `abs`.
    cavity_volume = 8 * (
        opts.order > 1 ? -fe_cell_curved_volume(gmsh, CELL_TAG_INCLUSION) :
            mesh_volume(inner)
    )

    return (;
        R, inner, outer, h_in = hin, h_out = hout, snap, cavity_volume,
        octant = true, closure_defect = defect,
    )
end
