# =============================================================================
#  mesh.jl — the elliptical-crack-in-a-ball mesh.
#
#  Ported from the gmsh model of the `SifAniso` study (FEniCSx implementation
#  of the same corrected finite Eshelby cell), including the two traps that
#  cost real debugging time there and are re-checked here:
#
#   * the crack disc is `embed`ed in the ball, never `fragment`ed — fragment
#     would cut the ball into two half-balls, i.e. two disjoint volumes;
#   * the `Crack` plugin duplicates the nodes of the crack *front* as well as
#     those of the lips, in spite of `OpenBoundaryPhysicalGroup` (still true in
#     gmsh 4.15). Left alone, the crack is effectively half an element larger
#     than asked for and the opening is overestimated by 10-20 %. The front is
#     therefore welded back together after import — see `weld_crack_front`.
# =============================================================================

const TAG_MATRIX = 100
const TAG_SPHERE_EXT = 200
const TAG_CRACK = 300
const TAG_CRACK_FRONT = 400
const TAG_OUTER_EDGES = 210
const TAG_OUTER_PTS = 220

const SET_OUTER = "outer"
const SET_CRACK = "crack"
const SET_OUTER_EDGES = "outer_edges"
const SET_OUTER_PTS = "outer_pts"

"""
    _build_gmsh_crack_model(gmsh, a, b, R, htipdiv)

Populate the current gmsh session with the crack-in-a-ball model: a ball of
radius `R` centered on an elliptical crack of semi-axes `(a, b)` lying in the
`z = 0` plane, refined to `min(a,b)/htipdiv` in a torus hugging the crack front
and coarsening to `R/3` at the outer boundary.

The gmsh **module** is passed in rather than imported, for the reason given in
[`_build_gmsh_axi_model`](@ref): it is a weak dependency reached by a different
route from each backend. The caller owns `gmsh.initialize()` /
`gmsh.finalize()`.
"""
function _build_gmsh_crack_model(gmsh, a::Float64, b::Float64, R::Float64, htipdiv::Float64)
    gmsh.model.add("mfh_crack_in_ball")

    bmin = min(a, b)
    h_tip = bmin / htipdiv
    h_crack = bmin / 2
    h_far = R / 3
    r_flat = 0.4bmin          # radius of the fully-refined disc around the front
    r_tip = 0.75bmin           # end of the tip → crack-zone transition
    r_dense = min(0.4R, 3.0bmin)

    sphere = gmsh.model.occ.addSphere(0, 0, 0, R)
    disk = gmsh.model.occ.addDisk(0, 0, 0, a, b)
    gmsh.model.occ.synchronize()

    front = [abs(t) for (_, t) in gmsh.model.getBoundary([(2, disk)], false, false, false)]
    outer = [abs(t) for (_, t) in gmsh.model.getBoundary([(3, sphere)], false, false, false)]

    # `embed` constrains the mesher to place nodes on the disc while leaving the
    # ball a single volume; `occ.fragment` would split it in two.
    gmsh.model.mesh.embed(2, [disk], 3, sphere)

    gmsh.model.addPhysicalGroup(3, [sphere], TAG_MATRIX, "matrix")
    gmsh.model.addPhysicalGroup(2, outer, TAG_SPHERE_EXT, SET_OUTER)
    gmsh.model.addPhysicalGroup(2, [disk], TAG_CRACK, SET_CRACK)
    gmsh.model.addPhysicalGroup(1, front, TAG_CRACK_FRONT, "front")

    # The seam curves and poles of the OCC sphere.  A physical group has a
    # dimension, so the surface group above does not carry them, and a backend
    # that reads boundary conditions off entity labels rather than off mesh
    # facets would leave those dofs free — eleven nodes here, enough to skew the
    # opening without ever looking like a bug.
    e_out = gmsh.model.getBoundary([(2, s) for s in outer], false, false, false)
    p_out = gmsh.model.getBoundary(e_out, false, false, false)
    isempty(e_out) || gmsh.model.addPhysicalGroup(
        1, unique(abs(t) for (_, t) in e_out), TAG_OUTER_EDGES, SET_OUTER_EDGES
    )
    isempty(p_out) || gmsh.model.addPhysicalGroup(
        0, unique(abs(t) for (_, t) in p_out), TAG_OUTER_PTS, SET_OUTER_PTS
    )

    # Size field 1: distance to the crack front.  `SizeMax` must be `h_far`,
    # not `h_crack`, otherwise the `Min` combination clamps the whole domain.
    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "CurvesList", Float64.(front))
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 200)
    f2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f2, "InField", f1)
    gmsh.model.mesh.field.setNumber(f2, "SizeMin", h_tip)
    gmsh.model.mesh.field.setNumber(f2, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f2, "DistMin", r_flat)
    gmsh.model.mesh.field.setNumber(f2, "DistMax", r_tip)

    # Size field 2: radial coarsening.  A `MathEval` distance rather than a
    # `Distance`-to-a-point field, so that no stray geometric point is added —
    # it would survive as a node with no cell attached, i.e. a zero row in the
    # stiffness matrix.
    f3 = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f3, "F", "Sqrt(x*x + y*y + z*z)")
    f4 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f4, "InField", f3)
    gmsh.model.mesh.field.setNumber(f4, "SizeMin", h_crack)
    gmsh.model.mesh.field.setNumber(f4, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f4, "DistMin", r_dense)
    gmsh.model.mesh.field.setNumber(f4, "DistMax", R)

    fmin = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", Float64[f2, f4])
    gmsh.model.mesh.field.setAsBackgroundMesh(fmin)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)      # HXT

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Gmsh")

    # Split the lips: duplicate the nodes of the crack surface.
    gmsh.plugin.setNumber("Crack", "Dimension", 2)
    gmsh.plugin.setNumber("Crack", "PhysicalGroup", TAG_CRACK)
    gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", TAG_CRACK_FRONT)
    gmsh.plugin.setNumber("Crack", "NormalX", 0.0)
    gmsh.plugin.setNumber("Crack", "NormalY", 0.0)
    gmsh.plugin.setNumber("Crack", "NormalZ", 1.0)
    gmsh.plugin.run("Crack")
    return nothing
end


"""
    _weld_msh_crack_front(path, a, b; ell_tol, z_tol) -> Int

Merge the duplicated nodes of the crack **front** in a written MSH 4.1 file,
in place, and return how many pairs were welded.

The `Crack` plugin duplicates the nodes of the front along with those of the
lips, in spite of `OpenBoundaryPhysicalGroup` (still true in gmsh 4.15). Left
alone the crack is effectively half an element longer than asked for, and the
opening comes out 10-20 % too large. The lips must stay split — that
discontinuity *is* the crack — so a blanket `removeDuplicateNodes` is not an
option: only nodes on the ellipse ``(x/a)^2 + (y/b)^2 = 1``, ``z = 0`` are merged.

Working on the file rather than on the live gmsh model is what makes this
shared: a node merge is a renumbering of the element connectivity, and every
backend reads the same file. Nodes left unreferenced are harmless — both mesh
readers drop them.
"""
function _weld_msh_crack_front(
        path::AbstractString, a::Real, b::Real;
        ell_tol = 1.0e-6, z_tol = 1.0e-9
    )
    lines = readlines(path)
    ints(l) = parse.(Int, split(l))

    # ── $Nodes: tag → coordinates, and the remap of the front pairs ──────────
    i = findfirst(==("\$Nodes"), lines)
    i === nothing && throw(ArgumentError("no \$Nodes section in $path"))
    nblocks = ints(lines[i + 1])[1]
    remap = Dict{Int, Int}()
    seen = Dict{NTuple{3, Float64}, Int}()
    p = i + 2
    for _ in 1:nblocks
        n = ints(lines[p])[4]
        tags = [ints(lines[p + k])[1] for k in 1:n]
        for k in 1:n
            x, y, z = parse.(Float64, split(lines[p + n + k]))
            (abs(z) < z_tol && abs((x / a)^2 + (y / b)^2 - 1) < ell_tol) || continue
            key = (x, y, z)
            j = get(seen, key, 0)
            j == 0 ? (seen[key] = tags[k]) : (remap[tags[k]] = j)
        end
        p += 1 + 2n
    end
    isempty(remap) && return 0

    # ── $Elements: renumber, leaving the block headers and element tags alone ─
    i = findfirst(==("\$Elements"), lines)
    i === nothing && throw(ArgumentError("no \$Elements section in $path"))
    nblocks = ints(lines[i + 1])[1]
    p = i + 2
    for _ in 1:nblocks
        n = ints(lines[p])[4]
        for k in 1:n
            v = ints(lines[p + k])
            any(t -> haskey(remap, t), @view v[2:end]) || continue
            for q in 2:length(v)
                v[q] = get(remap, v[q], v[q])
            end
            lines[p + k] = join(v, " ")
        end
        p += 1 + n
    end

    # ── $Nodes: drop the nodes that are now unreferenced ─────────────────────
    #
    #  Not cosmetic. Gridap builds its topology on the assumption that every
    #  vertex belongs to some cell, and a node left behind by the weld belongs
    #  to none: `TestFESpace` then indexes an owner array at 0 and throws.
    i = findfirst(==("\$Nodes"), lines)
    nblocks = ints(lines[i + 1])[1]
    p = i + 2
    kept = String[]
    total, lo, hi = 0, typemax(Int), 0
    for _ in 1:nblocks
        h = ints(lines[p])
        n = h[4]
        tags = [ints(lines[p + k])[1] for k in 1:n]
        keep = [k for k in 1:n if !haskey(remap, tags[k])]
        push!(kept, join((h[1], h[2], h[3], length(keep)), " "))
        append!(kept, string(tags[k]) for k in keep)
        append!(kept, lines[p + n + k] for k in keep)
        total += length(keep)
        for k in keep
            lo = min(lo, tags[k])
            hi = max(hi, tags[k])
        end
        p += 1 + 2n
    end
    lines = vcat(
        lines[1:i], join((nblocks, total, lo, hi), " "), kept, lines[p:end]
    )

    open(path, "w") do io
        for l in lines
            println(io, l)
        end
    end
    return length(remap)
end
