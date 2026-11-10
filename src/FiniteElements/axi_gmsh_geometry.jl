# =============================================================================
#  axi_mesh.jl — the meridian half-plane of the sphere with an off-center core.
#
#  The geometry is a solid of revolution, so the mesh is *two-dimensional*: the
#  half-plane `ρ ≥ 0` of the cylindrical coordinates `(ρ, θ, z)`, with the
#  symmetry axis `ρ = 0` as part of the boundary.  A node of this mesh stands
#  for a whole circle of the three-dimensional body.
#
#  Three regions, all half-discs centered on the axis:
#
#      core    radius a_c = a·w^(1/3), centered at z = d = α·(a − a_c)
#      shell   the rest of the inclusion (radius a, centered at the origin)
#      matrix  the rest of the cell (radius R = radius_ratio·a)
#
#  Built with the *built-in* gmsh kernel from explicit points and circle arcs.
#  OCC booleans would work too, but the explicit construction makes the region
#  and boundary tags unambiguous, which matters here because the axis is a
#  boundary of the computational domain without being a boundary of the body.
# =============================================================================

const AXI_TAG_CORE = 1
const AXI_TAG_SHELL = 2
const AXI_TAG_MATRIX = 3
const AXI_TAG_OUTER = 10
const AXI_TAG_AXIS = 11
const AXI_TAG_OUTER_PTS = 20
const AXI_TAG_AXIS_PTS = 21

const AXI_SET_CORE = "core"
const AXI_SET_SHELL = "shell"
const AXI_SET_MATRIX = "matrix"
const AXI_SET_OUTER = "outer"
const AXI_SET_AXIS = "axis"
const AXI_SET_OUTER_PTS = "outer_pts"
const AXI_SET_AXIS_PTS = "axis_pts"

"""
    _build_gmsh_axi_model(gmsh, a, a_core, d, R, h_in, h_out)

Populate the current gmsh session with the meridian half-plane of a sphere of
radius `a` holding a core of radius `a_core` centered at `z = d`, itself
embedded in a ball of matrix of radius `R`.

Element size is `h_in` on the inclusion and its core and `h_out` on the outer
boundary, gmsh interpolating in between.

The gmsh **module** is passed in rather than imported: gmsh is a weak
dependency, and the two finite-element backends reach it by different approaches —
`Gmsh.gmsh` for Ferrite, `GridapGmsh.gmsh` for Gridap, both over the same
`gmsh_jll`. Taking it as an argument keeps this geometry here in `src/`,
shared, instead of duplicating it in each extension.

The caller owns `gmsh.initialize()` / `gmsh.finalize()`.
"""
function _build_gmsh_axi_model(
        gmsh, a::Float64, a_core::Float64, d::Float64, R::Float64,
        h_in::Float64, h_out::Float64
    )
    gmsh.model.add("mfh_excentered_sphere_axi")
    geo = gmsh.model.geo

    # Centers of the three families of arcs.
    c_out = geo.addPoint(0.0, 0.0, 0.0, h_out)
    c_inc = geo.addPoint(0.0, 0.0, 0.0, h_in)
    c_cor = geo.addPoint(0.0, d, 0.0, h_in)

    # Axis points, from top to bottom.
    p_out_t = geo.addPoint(0.0, R, 0.0, h_out)
    p_inc_t = geo.addPoint(0.0, a, 0.0, h_in)
    p_cor_t = geo.addPoint(0.0, d + a_core, 0.0, h_in)
    p_cor_b = geo.addPoint(0.0, d - a_core, 0.0, h_in)
    p_inc_b = geo.addPoint(0.0, -a, 0.0, h_in)
    p_out_b = geo.addPoint(0.0, -R, 0.0, h_out)

    # Equator points.
    p_out_e = geo.addPoint(R, 0.0, 0.0, h_out)
    p_inc_e = geo.addPoint(a, 0.0, 0.0, h_in)
    p_cor_e = geo.addPoint(a_core, d, 0.0, h_in)

    # Axis segments, oriented downwards.
    l_mat_t = geo.addLine(p_out_t, p_inc_t)
    l_shl_t = geo.addLine(p_inc_t, p_cor_t)
    l_cor = geo.addLine(p_cor_t, p_cor_b)
    l_shl_b = geo.addLine(p_cor_b, p_inc_b)
    l_mat_b = geo.addLine(p_inc_b, p_out_b)

    # Arcs, oriented upwards (bottom → equator → top); each spans π/2 < π.
    a_out_1 = geo.addCircleArc(p_out_b, c_out, p_out_e)
    a_out_2 = geo.addCircleArc(p_out_e, c_out, p_out_t)
    a_inc_1 = geo.addCircleArc(p_inc_b, c_inc, p_inc_e)
    a_inc_2 = geo.addCircleArc(p_inc_e, c_inc, p_inc_t)
    a_cor_1 = geo.addCircleArc(p_cor_b, c_cor, p_cor_e)
    a_cor_2 = geo.addCircleArc(p_cor_e, c_cor, p_cor_t)

    s_core = geo.addPlaneSurface([geo.addCurveLoop([l_cor, a_cor_1, a_cor_2])])
    s_shell = geo.addPlaneSurface(
        [geo.addCurveLoop([l_shl_t, -a_cor_2, -a_cor_1, l_shl_b, a_inc_1, a_inc_2])]
    )
    s_matrix = geo.addPlaneSurface(
        [geo.addCurveLoop([l_mat_t, -a_inc_2, -a_inc_1, l_mat_b, a_out_1, a_out_2])]
    )

    geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [s_core], AXI_TAG_CORE, AXI_SET_CORE)
    gmsh.model.addPhysicalGroup(2, [s_shell], AXI_TAG_SHELL, AXI_SET_SHELL)
    gmsh.model.addPhysicalGroup(2, [s_matrix], AXI_TAG_MATRIX, AXI_SET_MATRIX)
    gmsh.model.addPhysicalGroup(1, [a_out_1, a_out_2], AXI_TAG_OUTER, AXI_SET_OUTER)
    gmsh.model.addPhysicalGroup(
        1, [l_mat_t, l_shl_t, l_cor, l_shl_b, l_mat_b], AXI_TAG_AXIS, AXI_SET_AXIS
    )

    # The endpoints of those curves, declared separately.  A physical group has
    # a dimension, so a curve group cannot carry its own bounding points, and a
    # backend that reads boundary conditions off *entity labels* rather than off
    # mesh facets would silently leave them free — three dofs on the outer
    # sphere, six on the axis.  Few enough to look like discretization error,
    # and enough to cost a whole order of convergence.
    gmsh.model.addPhysicalGroup(
        0, [p_out_b, p_out_e, p_out_t], AXI_TAG_OUTER_PTS, AXI_SET_OUTER_PTS
    )
    gmsh.model.addPhysicalGroup(
        0, [p_out_t, p_inc_t, p_cor_t, p_cor_b, p_inc_b, p_out_b],
        AXI_TAG_AXIS_PTS, AXI_SET_AXIS_PTS
    )

    gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    return nothing
end
