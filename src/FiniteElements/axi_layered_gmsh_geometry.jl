# =============================================================================
#  axi_layered_gmsh_geometry.jl — the meridian half-plane of an `N`-layer
#  spheroid in a ball of matrix.
#
#  The generalization of `axi_gmsh_geometry.jl` from three regions to `N + 1`,
#  and from circles to ellipses. Everything else is the same trade: the body is
#  a solid of revolution, so the mesh is two-dimensional on the half-plane
#  `ρ ≥ 0` of `(ρ, θ, z)`, and a node stands for a whole circle of the
#  three-dimensional body.
#
#  ## Free semi-axes, and the confocal case as a slice of them
#
#  The layers are described by their **semi-axes**, per layer, ascending — the
#  same `(axis_radii, disk_radii)` pair `LayeredSpheroid` takes, in the same
#  order. Nothing here knows what "confocal" means, and that is deliberate:
#
#    * handed `confocal_layer_radii(...)`, the mesh is the body the analytic
#      solution solves, by construction rather than by transcription — which is
#      the only way a cross-check of that solution is worth anything;
#    * handed anything else, it meshes a nest of spheroids that no closed form
#      covers, which is what the numerical study is for.
#
#  Coaxial and concentric is all the axisymmetry needs; a layer of a *different*
#  aspect ratio from its neighbor is perfectly admissible, and is the case with
#  no analytic counterpart.
#
#  ## Why splines, and not ellipse arcs
#
#  The built-in kernel's `addEllipseArc` needs the major axis named and refuses
#  an arc of more than π/2 in some configurations, and "which axis is major"
#  flips between an oblate and a prolate layer. Sampling the profile and handing
#  gmsh a spline sidesteps both, uses the sampler the superspheroidal cavity
#  already has — an ellipse is that profile at `p = 1` — and puts the
#  second-order mid-edge nodes on a curve rather than on a chord.
#
#  ## The element size is bounded by the thinnest layer, not by `nradial`
#
#  With free semi-axes two consecutive boundaries can come arbitrarily close,
#  and an element spanning a whole layer leaves that layer unresolved while the
#  mesh looks perfectly reasonable. So the size on each boundary is capped by a
#  fraction of its distance to its neighbors, measured on the semi-axes. This is
#  the same failure the concave superspheroid's wedge exhibited, where an
#  unresolved gap left `R₃₃` 1.2 % wrong and converging in appearance only.
# =============================================================================

# Layer `ℓ` (core first) gets tag `AXI_TAG_LAYER0 + ℓ`, clear of the
# core/shell/matrix tags of the two-region model.
const AXI_TAG_LAYER0 = 100

"""
    axi_layer_set(ℓ) -> String

Physical-group name of layer `ℓ`, counted from the core. The matrix keeps
`AXI_SET_MATRIX`, so a driver reads `N + 1` regions from one list.
"""
axi_layer_set(ℓ::Integer) = "layer$(ℓ)"

# The shape a backend matches on to find the layer sets of a grid without being
# told how many there are. Kept beside the name it has to agree with.
const AXI_LAYER_SET_PATTERN = r"^layer([0-9]+)$"

"""
    _layer_size_caps(axis_radii, disk_radii, R, h_in) -> Vector{Float64}

Element size allowed on each layer boundary: `h_in`, capped at a third of the
distance to the nearest neighboring boundary. The distance is taken as the
smaller of the two semi-axis gaps, which is a lower bound on the true gap
between two nested coaxial ellipses and therefore the safe side.

The outermost boundary is capped against the matrix too — a thin shell is thin
whether its neighbor is another layer or the cell wall.
"""
function _layer_size_caps(
        axis_radii::AbstractVector{Float64}, disk_radii::AbstractVector{Float64},
        R::Float64, h_in::Float64
    )
    n = length(axis_radii)
    caps = fill(h_in, n)
    for ℓ in 1:n
        gap = Inf
        if ℓ > 1
            gap = min(
                gap, axis_radii[ℓ] - axis_radii[ℓ - 1],
                disk_radii[ℓ] - disk_radii[ℓ - 1],
            )
        end
        if ℓ < n
            gap = min(
                gap, axis_radii[ℓ + 1] - axis_radii[ℓ],
                disk_radii[ℓ + 1] - disk_radii[ℓ],
            )
        else
            gap = min(gap, R - axis_radii[ℓ], R - disk_radii[ℓ])
        end
        # The core has no inner neighbor, so what bounds it is its **own** size —
        # always, not only when it is the single layer. A small core inside a
        # large shell has a generous gap to its neighbor and would keep the full
        # `h_in`, which is set from the *outer* semi-axis: at
        # `disk_radii = (0.01, 1.0)` and `nradial = 16` that is one element of
        # 0.0625 for a core 0.01 across, so the core would not be meshed at all
        # while every gap-based bound looked satisfied.
        ℓ == 1 && (gap = min(gap, axis_radii[1], disk_radii[1]))
        # Floored as well as capped. `check_nested_spheroids` rules out a zero
        # gap, but a merely tiny one would ask gmsh for a mesh nobody wants —
        # and an unbounded element count is a worse failure than a coarse layer,
        # because it takes the machine with it. A layer thinner than `h_in/64`
        # is under-resolved on purpose, and `fe_axi_mesh_report`'s volume error
        # is what says so.
        caps[ℓ] = clamp(gap / 3, h_in / 64, h_in)
    end
    return caps
end

"""
    check_nested_spheroids(axis_radii, disk_radii, R = Inf)

Verify that the layers are strictly nested and inside the cell, and throw a
message naming the offending layer otherwise.

Two coaxial concentric ellipses are nested **if and only if** both semi-axes of
the inner one are no larger than the outer's, so the whole condition is two
comparisons per layer. It is worth checking rather than trusting: a violation
does not produce a wrong answer, it produces a self-intersecting geometry that
gmsh rejects from deep inside its own meshing pipeline, with a message that
names neither the layer nor the semi-axis.

`disk_radii` are the transverse semi-axes and `axis_radii` those along the
revolution axis — the argument order
[`LayeredSpheroid`](@ref MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid) uses.

Consecutive layers must be **strictly** ascending in both semi-axes. Two
coinciding boundaries pass every containment argument and carry no volume, but
they cannot be meshed: the element size is capped at a fraction of a layer's
thickness, so a zero-thickness layer asks gmsh for a zero-sized element. A
sphere is still a legitimate layer — that is `a == c` *within* one layer, which
is unrestricted, and it is the slice where `LayeredSphere` is the closed form.
"""
function check_nested_spheroids(
        axis_radii::AbstractVector{<:Real}, disk_radii::AbstractVector{<:Real},
        R::Real = Inf
    )
    n = length(axis_radii)
    n == length(disk_radii) || throw(
        DimensionMismatch(
            "a layered spheroid needs one axis semi-axis per disk semi-axis, " *
                "got $n and $(length(disk_radii))"
        )
    )
    n ≥ 1 || throw(ArgumentError("a layered spheroid needs at least one layer"))
    for ℓ in 1:n
        (axis_radii[ℓ] > 0 && disk_radii[ℓ] > 0) || throw(
            ArgumentError(
                "layer $ℓ has a non-positive semi-axis: " *
                    "axis $(axis_radii[ℓ]), disk $(disk_radii[ℓ])"
            )
        )
    end
    for ℓ in 1:(n - 1)
        (axis_radii[ℓ] < axis_radii[ℓ + 1] && disk_radii[ℓ] < disk_radii[ℓ + 1]) ||
            throw(
            ArgumentError(
                "layers $ℓ and $(ℓ + 1) are not strictly nested: layer $ℓ is " *
                    "(disk $(disk_radii[ℓ]), axis $(axis_radii[ℓ])) and layer " *
                    "$(ℓ + 1) is (disk $(disk_radii[ℓ + 1]), axis " *
                    "$(axis_radii[ℓ + 1])). Both semi-axes must **grow** outwards: " *
                    "a wider core inside a narrower shell is not a nest whatever " *
                    "the other semi-axis does, and two coinciding boundaries carry " *
                    "no volume and cannot be meshed — the element size is capped " *
                    "at a fraction of a layer's thickness, so a zero-thickness " *
                    "layer asks for a zero-sized element."
            )
        )
    end
    isfinite(R) || return nothing
    (R > axis_radii[n] && R > disk_radii[n]) || throw(
        ArgumentError(
            "the cell radius $R does not contain the outer layer " *
                "(disk $(disk_radii[n]), axis $(axis_radii[n])); raise " *
                "`radius_ratio`"
        )
    )
    return nothing
end

"""
    _ellipse_meridian(a, c, n) -> Vector{NTuple{2,Float64}}

`n` points of the quarter meridian of the ellipse of transverse semi-axis `a`
and axis semi-axis `c`, from the equator `(a, 0)` to the pole `(0, c)`, both
endpoints exact.

Goes through the superspheroid sampler at `p = 1`: an ellipse *is* that profile,
so there is one parametrization in the package rather than two that can drift.
"""
_ellipse_meridian(a::Real, c::Real, n::Integer) =
    _superspheroid_meridian(Superspheroid(Float64(a), Float64(c), 1.0), n)

"""
    _build_gmsh_axi_layered_model(gmsh, axis_radii, disk_radii, R, h_in, h_out,
                                  nprofile)

Populate the current gmsh session with the meridian half-plane of an `N`-layer
spheroid — layer `ℓ` bounded by the ellipse of semi-axes
`(disk_radii[ℓ], axis_radii[ℓ])`, ascending — inside a ball of matrix of radius
`R`.

Physical groups: `axi_layer_set(ℓ)` for each layer, `AXI_SET_MATRIX`,
`AXI_SET_OUTER`, `AXI_SET_AXIS`, and the point groups a label-driven backend
needs.

The gmsh module is passed in rather than imported, and the caller owns
`initialize`/`finalize`, for the reasons `_build_gmsh_axi_model` gives.
"""
function _build_gmsh_axi_layered_model(
        gmsh, axis_radii::AbstractVector{Float64},
        disk_radii::AbstractVector{Float64}, R::Float64,
        h_in::Float64, h_out::Float64, nprofile::Integer = 41,
    )
    check_nested_spheroids(axis_radii, disk_radii, R)
    n = length(axis_radii)
    gmsh.model.add("mfh_axi_layered_spheroid")
    geo = gmsh.model.geo
    caps = _layer_size_caps(axis_radii, disk_radii, R, h_in)

    c_out = geo.addPoint(0.0, 0.0, 0.0, h_out)
    p_out_t = geo.addPoint(0.0, R, 0.0, h_out)
    p_out_b = geo.addPoint(0.0, -R, 0.0, h_out)
    p_out_e = geo.addPoint(R, 0.0, 0.0, h_out)

    # Per layer: the two axis points, the equator point, and the two splines.
    p_top = Vector{Int}(undef, n)
    p_bot = Vector{Int}(undef, n)
    s_up = Vector{Int}(undef, n)
    s_lo = Vector{Int}(undef, n)
    for ℓ in 1:n
        a, c, h = disk_radii[ℓ], axis_radii[ℓ], caps[ℓ]
        p_top[ℓ] = geo.addPoint(0.0, c, 0.0, h)
        p_bot[ℓ] = geo.addPoint(0.0, -c, 0.0, h)
        p_eq = geo.addPoint(a, 0.0, 0.0, h)
        prof = _ellipse_meridian(a, c, nprofile)
        # Interior control points only: the endpoints are already placed, and a
        # spline control is not a mesh vertex, so sampling generously is free.
        up = [geo.addPoint(ρ, z, 0.0, h) for (ρ, z) in prof[2:(end - 1)]]
        lo = [geo.addPoint(ρ, -z, 0.0, h) for (ρ, z) in prof[2:(end - 1)]]
        # Oriented upwards, bottom pole → equator → top pole, so the loops below
        # mirror `_build_gmsh_axi_model` line for line.
        s_lo[ℓ] = geo.addSpline([p_bot[ℓ]; reverse(lo); p_eq])
        s_up[ℓ] = geo.addSpline([p_eq; up; p_top[ℓ]])
    end

    # Axis segments, oriented downwards: matrix, then one per layer boundary
    # going in, then the core, then back out.
    l_top = Vector{Int}(undef, n)     # from boundary ℓ+1 (or R) down to ℓ
    l_bot = Vector{Int}(undef, n)
    l_top[n] = geo.addLine(p_out_t, p_top[n])
    l_bot[n] = geo.addLine(p_bot[n], p_out_b)
    for ℓ in (n - 1):-1:1
        l_top[ℓ] = geo.addLine(p_top[ℓ + 1], p_top[ℓ])
        l_bot[ℓ] = geo.addLine(p_bot[ℓ], p_bot[ℓ + 1])
    end
    l_core = geo.addLine(p_top[1], p_bot[1])

    a_out_1 = geo.addCircleArc(p_out_b, c_out, p_out_e)
    a_out_2 = geo.addCircleArc(p_out_e, c_out, p_out_t)

    # The core is bounded by its own profile and the axis through it; every
    # other layer is an annulus between two profiles.
    s_layer = Vector{Int}(undef, n)
    s_layer[1] = geo.addPlaneSurface(
        [geo.addCurveLoop([l_core, s_lo[1], s_up[1]])]
    )
    for ℓ in 2:n
        s_layer[ℓ] = geo.addPlaneSurface(
            [
                geo.addCurveLoop(
                    [
                        l_top[ℓ - 1], -s_up[ℓ - 1], -s_lo[ℓ - 1], l_bot[ℓ - 1],
                        s_lo[ℓ], s_up[ℓ],
                    ]
                ),
            ]
        )
    end
    s_matrix = geo.addPlaneSurface(
        [
            geo.addCurveLoop(
                [l_top[n], -s_up[n], -s_lo[n], l_bot[n], a_out_1, a_out_2]
            ),
        ]
    )

    geo.synchronize()

    for ℓ in 1:n
        gmsh.model.addPhysicalGroup(
            2, [s_layer[ℓ]], AXI_TAG_LAYER0 + ℓ, axi_layer_set(ℓ)
        )
    end
    gmsh.model.addPhysicalGroup(2, [s_matrix], AXI_TAG_MATRIX, AXI_SET_MATRIX)
    gmsh.model.addPhysicalGroup(1, [a_out_1, a_out_2], AXI_TAG_OUTER, AXI_SET_OUTER)
    gmsh.model.addPhysicalGroup(
        1, [l_top..., l_core, l_bot...], AXI_TAG_AXIS, AXI_SET_AXIS
    )
    gmsh.model.addPhysicalGroup(
        0, [p_out_b, p_out_e, p_out_t], AXI_TAG_OUTER_PTS, AXI_SET_OUTER_PTS
    )
    # Every axis point, layer poles included: a physical group carries one
    # dimension, so the curve group above cannot own its endpoints, and a
    # backend reading boundary conditions off entity labels would leave them
    # free — few enough dofs to pass for discretization error, and enough to
    # cost an order of convergence.
    gmsh.model.addPhysicalGroup(
        0, [p_out_t, p_top..., p_bot..., p_out_b], AXI_TAG_AXIS_PTS, AXI_SET_AXIS_PTS
    )

    gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    return nothing
end
