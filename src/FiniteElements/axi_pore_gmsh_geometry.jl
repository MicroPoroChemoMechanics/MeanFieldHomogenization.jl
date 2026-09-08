# =============================================================================
#  axi_pore_gmsh_geometry.jl — the meridian half-plane of a superspheroidal
#  cavity in a ball of matrix.
#
#  The axisymmetric counterpart of `cell_gmsh_geometry.jl`, and the same trade:
#  one region instead of three, because a cavity has no interior to mesh, and a
#  boundary that no CAD kernel represents.
#
#  ## Why a spline through sampled points, and not an arc
#
#  The meridian profile is the superellipse
#
#      (ρ/a)^m + (|z|/c)^m = 1 ,      m = 2p ,
#
#  which is a circle only at `m = 2`. It has a closed-form parametrization —
#
#      ρ = a t^{1/m} ,   z = c (1 − t)^{1/m} ,   t ∈ [0, 1] ,
#
#  which satisfies the level set identically, `t + (1 − t) = 1`, so no root find
#  and no quadrature enter the geometry. Sampling it and handing gmsh a spline
#  keeps the boundary *ours* while letting the second-order promotion put
#  mid-edge nodes on a curve rather than on a chord.
#
#  ## The two corners, and why they are endpoints
#
#  For `m < 1` the profile is not smooth at either end, and `outward_normal`
#  says so in its own words: "the equatorial plane is the crease; the two poles
#  are the conical points". So the profile is cut into **two** splines meeting at
#  the equator, and the poles are curve endpoints — a corner is never interior
#  to a spline, which is what would round it off.
#
#  The sampling clusters at both ends for the same reason. A uniform step in `t`
#  spreads points exactly where the curvature is unbounded.
#
#  ## The axis is a boundary of the domain without being a boundary of the body
#
#  Two segments here, `z ∈ [c, R]` and `z ∈ [−R, −c]`, against the five of the
#  core-shell model. Everything else about the axis is unchanged, including the
#  point groups a label-driven backend needs.
# =============================================================================

const AXI_TAG_INCLUSION = 12
const AXI_SET_INCLUSION = "inclusion"

"""
    _superspheroid_meridian(shape, n) -> Vector{NTuple{2, Float64}}

`n` points of the meridian profile from the equator `(a, 0)` to the pole
`(0, c)`, both included, as `(ρ, z)` pairs.

Exact by construction: the parametrization `ρ = a t^{1/m}`, `z = c(1−t)^{1/m}`
satisfies `(ρ/a)^m + (z/c)^m = t + (1−t) = 1` for every `t`, so the points lie
on the surface to round-off and `level_set` returns zero on each — which the
tests check rather than assume.

The step clusters at both ends, through `t = (1 − cos πs)/2`. Uniform in `t`
would put its points where the profile is flat and starve the two corners, whose
curvature is unbounded for a concave shape.
"""
function _superspheroid_meridian(shape::Superspheroid, n::Integer)
    n ≥ 3 || throw(ArgumentError("a meridian needs at least 3 points, got $n"))
    m = Float64(shape_exponent(shape))
    a, c = Float64(shape.a), Float64(shape.c)
    pts = Vector{NTuple{2, Float64}}(undef, n)
    for k in 1:n
        s = (k - 1) / (n - 1)
        t = (1 - cos(π * s)) / 2                     # 0 at the pole, 1 at the equator
        pts[n + 1 - k] = (a * t^(1 / m), c * (1 - t)^(1 / m))
    end
    # Pin the two corners exactly: `t^(1/m)` at `t = 0` is zero in exact
    # arithmetic and must be zero here too, or the pole drifts off the axis and
    # the axis condition stops applying to it.
    pts[1] = (a, 0.0)
    pts[end] = (0.0, c)
    return pts
end

"""
    _build_gmsh_axi_pore_model(gmsh, shape, R, h_in, h_out, nprofile) -> nothing

Build and mesh the meridian half-plane: one region of matrix between the
superspheroidal cavity and the outer circle of radius `R`.

Element size is `h_in` on the cavity wall and `h_out` on the outer boundary,
gmsh interpolating in between — per-point sizing, exactly as
`_build_gmsh_axi_model`, with no background field.

The gmsh **module** is passed in rather than imported, and the caller owns
`initialize` / `finalize`.
"""
function _build_gmsh_axi_pore_model(
        gmsh, shape::Superspheroid, R::Float64, h_in::Float64, h_out::Float64,
        nprofile::Integer = 61,
    )
    gmsh.model.add("mfh_axi_supershape_pore")
    geo = gmsh.model.geo
    c = Float64(shape.c)

    prof = _superspheroid_meridian(shape, nprofile)      # equator → pole

    # Center of the outer arcs.
    c_out = geo.addPoint(0.0, 0.0, 0.0, h_out)

    # Axis points, top to bottom. The cavity's poles are at ±c, not ±a.
    p_out_t = geo.addPoint(0.0, R, 0.0, h_out)
    p_inc_t = geo.addPoint(0.0, c, 0.0, h_in)
    p_inc_b = geo.addPoint(0.0, -c, 0.0, h_in)
    p_out_b = geo.addPoint(0.0, -R, 0.0, h_out)

    # Equator points.
    p_out_e = geo.addPoint(R, 0.0, 0.0, h_out)
    p_inc_e = geo.addPoint(prof[1][1], 0.0, 0.0, h_in)

    # Interior control points of the profile, upper and lower half. They are
    # spline controls, not mesh vertices: only the endpoints of a curve are
    # meshed, so a generous sampling costs nothing.
    up = [geo.addPoint(ρ, z, 0.0, h_in) for (ρ, z) in prof[2:(end - 1)]]
    lo = [geo.addPoint(ρ, -z, 0.0, h_in) for (ρ, z) in prof[2:(end - 1)]]

    # Axis segments, oriented downwards.
    l_ax_t = geo.addLine(p_out_t, p_inc_t)
    l_ax_b = geo.addLine(p_inc_b, p_out_b)

    # The profile, oriented upwards (bottom pole → equator → top pole), so the
    # loop below can mirror `_build_gmsh_axi_model` line for line.
    s_inc_1 = geo.addSpline([p_inc_b; reverse(lo); p_inc_e])
    s_inc_2 = geo.addSpline([p_inc_e; up; p_inc_t])

    # Outer arcs, upwards; each spans π/2 < π, which the built-in kernel needs.
    a_out_1 = geo.addCircleArc(p_out_b, c_out, p_out_e)
    a_out_2 = geo.addCircleArc(p_out_e, c_out, p_out_t)

    s_matrix = geo.addPlaneSurface(
        [geo.addCurveLoop([l_ax_t, -s_inc_2, -s_inc_1, l_ax_b, a_out_1, a_out_2])]
    )

    geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [s_matrix], AXI_TAG_MATRIX, AXI_SET_MATRIX)
    gmsh.model.addPhysicalGroup(1, [a_out_1, a_out_2], AXI_TAG_OUTER, AXI_SET_OUTER)
    gmsh.model.addPhysicalGroup(1, [l_ax_t, l_ax_b], AXI_TAG_AXIS, AXI_SET_AXIS)
    # Nothing is prescribed on this one, and that is what makes it a cavity: it
    # exists so the solve can integrate over it.
    gmsh.model.addPhysicalGroup(
        1, [s_inc_1, s_inc_2], AXI_TAG_INCLUSION, AXI_SET_INCLUSION
    )

    # The endpoints, for the same reason as in the core-shell model: a physical
    # group has a dimension, so a curve group cannot carry its own bounding
    # points, and a backend reading boundary conditions off entity labels would
    # leave them free.
    gmsh.model.addPhysicalGroup(
        0, [p_out_b, p_out_e, p_out_t], AXI_TAG_OUTER_PTS, AXI_SET_OUTER_PTS
    )
    gmsh.model.addPhysicalGroup(
        0, [p_out_t, p_inc_t, p_inc_b, p_out_b], AXI_TAG_AXIS_PTS, AXI_SET_AXIS_PTS
    )

    gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    return nothing
end
