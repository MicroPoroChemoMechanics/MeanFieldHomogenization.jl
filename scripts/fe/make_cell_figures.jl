# =============================================================================
#  make_cell_figures.jl — regenerate the static assets of the supershape-cavity
#  section of `docs/src/manual/fe_inclusions.md` and of
#  `docs/src/applications/concave_pores.md`.
#
#  Maintenance script, run **by hand**, never at documentation-build time. The
#  pages embed the committed PNGs, so a doc build stays free of `gmsh_jll`.
#
#      julia scripts/fe/make_cell_figures.jl              # everything
#      julia scripts/fe/make_cell_figures.jl mesh         # one section only
#
#  Outputs (committed):
#      docs/src/assets/fe/cell_mesh_3d.png
#      docs/src/assets/fe/axi_pore_mesh.png
#      docs/src/assets/fe/cell_results.md.in    (tables, pasted into the pages)
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots

import Ferrite, FerriteGmsh, Gmsh

gr()
default(;
    fontfamily = "sans-serif", framestyle = :box, grid = true, legendfontsize = 8,
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)

const OUT = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "fe"))
mkpath(OUT)

const SECTIONS = isempty(ARGS) ? nothing : Set(ARGS)
want(name) = SECTIONS === nothing || name in SECTIONS

const FE = MeanFieldHomogenization.FiniteElements
const νm, Em = 0.3, 1.0
const C₀ = iso_stiffness(Em / (3 * (1 - 2νm)), Em / (2 * (1 + νm)))

# ─── Reaching into the mesh ──────────────────────────────────────────────────
#
# A maintenance script may: the grid is an implementation detail, and drawing it
# is the one job that needs it.

"Grid of a supershape cell, and the shape it was built around."
function cell_grid(shape, opts)
    b = FE.FerriteBackend()
    g = FE.fe_cell_grid(b, shape, opts)
    return g.grid, g.built
end

"Append the closed polyline `pts` to `(X, Y, Z)`, NaN-separated."
function push_loop!(X, Y, Z, pts)
    for p in pts
        push!(X, p[1]); push!(Y, p[2]); push!(Z, p[3])
    end
    push!(X, pts[1][1]); push!(Y, pts[1][2]); push!(Z, pts[1][3])
    push!(X, NaN); push!(Y, NaN); push!(Z, NaN)
    return nothing
end

"""
Facet loops of a facet set, normalized by `a`.

The cell is meshed with **quadratic** tetrahedra — `order = 2` is mandatory,
the boundary having been curved on purpose — so a facet carries six nodes and
only the three corners are wanted for a wireframe.
"""
function facet_loops!(X, Y, Z, grid, name, a; keep = _ -> true)
    for fi in Ferrite.getfacetset(grid, name)
        cell = grid.cells[fi[1]]
        nodes = Ferrite.facets(cell)[fi[2]]
        pts = [Tuple(Ferrite.getnodes(grid, n).x ./ a) for n in nodes[1:3]]
        keep(pts) || continue
        push_loop!(X, Y, Z, pts)
    end
    return nothing
end

"""
Cut-away view of the full cell: the far half of the outer sphere, so the near
side does not veil the cavity, and the cavity wall in red.
"""
function figure_full(shape, opts; kw...)
    grid, built = cell_grid(shape, opts)
    a = shape.a
    Xo, Yo, Zo = Float64[], Float64[], Float64[]
    facet_loops!(
        Xo, Yo, Zo, grid, FE.CELL_SET_OUTER, a;
        keep = pts -> sum(p[2] for p in pts) / 3 ≤ 0,
    )
    Xi, Yi, Zi = Float64[], Float64[], Float64[]
    facet_loops!(Xi, Yi, Zi, grid, FE.CELL_SET_INCLUSION, a)

    L = opts.radius_ratio * bounding_radius(shape) / a * 1.02
    plt = plot(;
        legend = false, xlabel = "x / a", ylabel = "y / a", zlabel = "z / a",
        camera = (35, 30), xlims = (-L, L), ylims = (-L, L), zlims = (-L, L), kw...,
    )
    plot!(plt, Xo, Yo, Zo; lc = RGBA(0.42, 0.55, 0.72, 0.3), lw = 0.4)
    plot!(plt, Xi, Yi, Zi; lc = :crimson, lw = 0.6)
    return plt, built
end

"""
The same cell as an octant: the cavity wall in red, the outer eighth-sphere in
blue, and the three flat faces in gray — the part that had to be built, and the
part that makes the saving exact rather than approximate.
"""
function figure_octant(shape, opts; kw...)
    grid, built = cell_grid(shape, opts)
    a = shape.a
    Xo, Yo, Zo = Float64[], Float64[], Float64[]
    facet_loops!(Xo, Yo, Zo, grid, FE.CELL_SET_OUTER, a)
    Xi, Yi, Zi = Float64[], Float64[], Float64[]
    facet_loops!(Xi, Yi, Zi, grid, FE.CELL_SET_INCLUSION, a)
    Xp, Yp, Zp = Float64[], Float64[], Float64[]
    for k in 1:3
        facet_loops!(Xp, Yp, Zp, grid, FE.CELL_SET_PLANE[k], a)
    end

    L = opts.radius_ratio * bounding_radius(shape) / a * 1.02
    plt = plot(;
        legend = false, xlabel = "x / a", ylabel = "y / a", zlabel = "z / a",
        camera = (35, 30), xlims = (0, L), ylims = (0, L), zlims = (0, L), kw...,
    )
    plot!(plt, Xp, Yp, Zp; lc = RGBA(0.35, 0.35, 0.35, 0.55), lw = 0.3)
    plot!(plt, Xo, Yo, Zo; lc = RGBA(0.42, 0.55, 0.72, 0.45), lw = 0.4)
    plot!(plt, Xi, Yi, Zi; lc = :crimson, lw = 0.7)
    return plt, built
end

const report = IOBuffer()

# ─── Runs ────────────────────────────────────────────────────────────────────

const CONCAVE = Supersphere(1.0, 0.35)
const OPTS_FULL = FECellMeshOptions(;
    level = 3, outer_level = 3, radius_ratio = 3.0, relax = 20
)
const OPTS_OCT = FECellMeshOptions(;
    level = 3, outer_level = 3, radius_ratio = 3.0, relax = 20, octant = true
)

want("mesh") && let
    println("── figures: the cell, whole and in an octant ───────────────────")
    pf, bf = figure_full(
        CONCAVE, OPTS_FULL;
        title = "p = 0.35 — the cavity inside the shell", size = (460, 440),
    )
    po, bo = figure_octant(
        CONCAVE, OPTS_OCT;
        title = "the same cell, one octant", size = (460, 440),
    )
    savefig(
        plot(
            pf, po; layout = (1, 2), left_margin = 8Plots.mm,
            bottom_margin = 8Plots.mm, size = (980, 460),
        ),
        joinpath(OUT, "cell_mesh_3d.png"),
    )
    println("wrote ", joinpath(OUT, "cell_mesh_3d.png"))

    println(report, "<!-- Generated by scripts/fe/make_cell_figures.jl — do not edit by hand. -->\n")
    println(report, "### The cell of a concave supersphere (p = 0.35, level 3, R/a = 3)\n")
    println(report, "| | whole cell | octant |")
    println(report, "|---|---|---|")
    @printf report "| cavity volume | %.6f | %.6f |\n" bf.cavity_volume bo.cavity_volume
    @printf report "| exact volume | %.6f | %.6f |\n" shape_volume(CONCAVE) shape_volume(CONCAVE)
    @printf report "| snapping limited at | %d nodes | %d nodes |\n" bf.snap.limited bo.snap.limited
    @printf report "| shell closure defect | — | %.1e |\n" bo.closure_defect
    println(report)
end

want("cost") && let
    println("── the cost, and what it buys ─────────────────────────────────")
    println(report, "### Cost and accuracy, spherical pore (p = 1)\n")
    println(report, "| level | mode | vector dofs | conduction error | time |")
    println(report, "|---|---|---|---|---|")
    for lvl in (2, 3, 4), oct in (false, true)
        lvl == 4 && !oct && continue          # the full cell at level 4 is what the octant is for
        o = FECellMeshOptions(;
            level = lvl, outer_level = 3, radius_ratio = 3.0, relax = 20,
            octant = oct, max_dofs = 400_000, min_free_gb = 4.0,
        )
        pore = FESupershapePore(Supersphere(1.0, 1.0); opts = o)
        t = @elapsed A = gradient_gradient_loc(pore, TensISO{3}(1.0), TensISO{3}(1.0))
        err = abs(Matrix(get_array(A))[1, 1] - 1.5) / 1.5
        rep = fe_cell_mesh_report(pore)
        @printf report "| %d | %s | %d | %.2e | %.1f s |\n" lvl (
            oct ? "octant" : "whole"
        ) (3 * rep.nnodes) err t
        @printf "  level %d %-7s err %.2e  %.1f s\n" lvl (oct ? "octant" : "whole") err t
        flush(stdout)
    end
    println(report)
end

# ─── The axisymmetric cavity ─────────────────────────────────────────────────

"""
The meridian half-plane of a superspheroidal cavity: the triangle mesh, the
cavity wall in red, and the axis in blue.

Two-dimensional, so this is the whole computational domain and not a slice of
one — which is the point of the figure.
"""
function figure_axi(shape, opts; kw...)
    b = FE.FerriteBackend()
    grid = FE.fe_axi_grid(b, FEAxiSupershapePore(shape; opts))
    X, Y = Float64[], Float64[]
    for ci in 1:Ferrite.getncells(grid)
        c = Ferrite.getcoordinates(grid, ci)
        append!(X, [c[1][1], c[2][1], c[3][1], c[1][1], NaN])
        append!(Y, [c[1][2], c[2][2], c[3][2], c[1][2], NaN])
    end
    R = opts.radius_ratio * bounding_radius(shape)
    plt = plot(;
        aspect_ratio = 1, legend = false, xlabel = "ρ / a", ylabel = "z / a",
        xlims = (-0.05R, 1.05R), ylims = (-1.05R, 1.05R), kw...,
    )
    plot!(plt, X, Y; lc = :grey70, lw = 0.3)
    # The exact profile, both halves, from the closed form rather than the mesh.
    prof = FE._superspheroid_meridian(shape, 200)
    ρ = [q[1] for q in prof]
    z = [q[2] for q in prof]
    plot!(plt, vcat(ρ, reverse(ρ)), vcat(z, -reverse(z)); lc = :crimson, lw = 2)
    plot!(plt, [0.0, 0.0], [-R, R]; lc = :steelblue, lw = 2)
    return plt, grid
end

want("axi") && let
    println("── figure: the axisymmetric cavity ────────────────────────────")
    o = FEAxiMeshOptions(; nradial = 20, radius_ratio = 4.0)
    p1, g1 = figure_axi(
        Superspheroid(1.0, 1.0, 0.35), o;
        title = "p = 0.35, c/a = 1", size = (330, 440),
    )
    p2, g2 = figure_axi(
        Superspheroid(1.0, 0.4, 0.4), o;
        title = "p = 0.40, c/a = 0.4", size = (330, 440),
    )
    p3, _ = figure_axi(
        Superspheroid(1.0, 2.0, 0.8), o;
        title = "p = 0.80, c/a = 2", size = (330, 440),
    )
    # The fourth panel is the reason the mesher grades at all: the equatorial
    # crease, magnified. A uniform element there is 0.031a where the half-gap is
    # 0.0012a, and elements that straddle a gap twenty-five times thinner than
    # themselves put `R₃₃` 1.2 % off — not merely unconverged.
    p4, _ = figure_axi(
        Superspheroid(1.0, 0.4, 0.4), o;
        title = "the crease, magnified 12×", size = (330, 440),
        xlims = (0.78, 1.04), ylims = (-0.13, 0.13),
        xlabel = "ρ / a", ylabel = "z / a",
    )
    savefig(
        plot(
            p1, p2, p3, p4; layout = (1, 4), left_margin = 6Plots.mm,
            bottom_margin = 6Plots.mm, size = (1320, 440),
        ),
        joinpath(OUT, "axi_pore_mesh.png"),
    )
    println("wrote ", joinpath(OUT, "axi_pore_mesh.png"))

    println(report, "### The axisymmetric cell (nradial = 20, R/a = 4)\n")
    println(report, "| shape | cells | matrix volume, measured / exact |")
    println(report, "|---|---|---|")
    for (lbl, sh) in (
            ("p = 0.35, c/a = 1", Superspheroid(1.0, 1.0, 0.35)),
            ("p = 0.40, c/a = 0.4", Superspheroid(1.0, 0.4, 0.4)),
            ("p = 0.80, c/a = 2", Superspheroid(1.0, 2.0, 0.8)),
        )
        r = fe_axi_pore_mesh_report(FEAxiSupershapePore(sh; opts = o))
        @printf report "| %s | %d | %.4f / %.4f (%+.2f %%) |\n" lbl r.ncells (
            r.volume_matrix
        ) r.volume_matrix_exact (100 * r.volume_error)
    end
    println(report)
end

# ─── This package against Sevostianov et al. (2016), superimposed ────────────
#
#  Their Tables B.1 and B.4, all eighteen and seventeen rows, plotted as points
#  against this package's Fourier axisymmetric cell as a line. Their material
#  parameters are nowhere in the paper and are *inferred* from its own `p = 1`
#  row, where the body is an exact sphere: `E₀ = 1`, `ν₀ = 1/3`, `k₀ = 1`.
#
#  The dashed band on each panel is their **own** mesh uncertainty, Table B.3 —
#  the relative change between their two meshes. A curve inside that band agrees
#  with them as closely as they agree with themselves, which is the only
#  meaningful standard here.
want("sevostianov") && let
    println("── figure: against Sevostianov et al. (2016) ──────────────────")

    # Transcribed from the paper. Compliance: p => (H₁₁₁₁, H₁₁₂₂, H₁₁₃₃, H₃₃₃₃, H₁₃₁₃).
    theirH = [
        (0.20, 1.887796, -0.477459, -0.964299, 27.731870, 8.024717),
        (0.25, 1.808015, -0.419517, -0.842265, 12.354200, 3.654240),
        (0.30, 1.819960, -0.420280, -0.783590, 7.405500, 2.558693),
        (0.33, 1.840120, -0.421878, -0.739390, 5.895420, 1.901430),
        (0.35, 1.854860, -0.424255, -0.720632, 5.217026, 1.742868),
        (0.40, 1.894290, -0.434329, -0.681435, 4.065400, 1.506599),
        (0.45, 1.918003, -0.443068, -0.642262, 3.352013, 1.388400),
        (0.50, 1.937640, -0.451064, -0.611918, 2.916446, 1.323735),
        (0.55, 1.952870, -0.459580, -0.587291, 2.639410, 1.289338),
        (0.60, 1.963770, -0.466431, -0.568153, 2.456980, 1.269780),
        (0.65, 1.973077, -0.472413, -0.553477, 2.332502, 1.269780),
        (0.70, 1.979451, -0.477634, -0.541160, 2.241809, 1.258690),
        (0.75, 1.984812, -0.482264, -0.531049, 2.174690, 1.252894),
        (0.80, 1.989424, -0.486274, -0.522567, 2.122560, 1.251279),
        (0.85, 1.993319, -0.489852, -0.515269, 2.081938, 1.250837),
        (0.90, 1.996558, -0.492969, -0.508869, 2.049180, 1.250650),
        (0.95, 1.999258, -0.496250, -0.501454, 2.027340, 1.250050),
        (1.00, 2.001203, -0.498053, -0.498057, 2.001212, 1.249900),
    ]
    # Table B.3, their mesh-to-mesh change in percent, same column order.
    theirMesh = Dict(
        0.20 => (0.21, 0.46, 0.17, 0.41, 0.42), 0.25 => (0.07, 0.18, 1.32, 0.43, 0.33),
        0.30 => (0.33, 0.42, 2.50, 0.10, 0.83), 0.33 => (0.02, 0.06, 1.47, 0.44, 0.31),
        0.35 => (0.01, 0.04, 0.11, 0.40, 0.27), 0.40 => (0.00, 0.01, 0.13, 0.40, 0.16),
        0.45 => (0.00, 0.00, 0.11, 0.29, 0.07), 0.50 => (0.01, 0.00, 0.11, 0.21, 0.06),
        0.55 => (0.01, 0.01, 0.07, 0.16, 0.02), 0.60 => (0.01, 0.01, 0.05, 0.08, 0.02),
        0.65 => (0.02, 0.02, 0.04, 0.05, 0.91), 0.70 => (0.01, 0.01, 0.02, 0.03, 0.18),
        0.75 => (0.01, 0.01, 0.04, 0.04, 0.23), 0.80 => (0.01, 0.01, 0.02, 0.02, 0.22),
        0.85 => (0.01, 0.01, 0.02, 0.02, 0.11), 0.90 => (0.01, 0.01, 0.01, 0.01, 0.06),
        0.95 => (0.27, 0.03, 0.02, 0.02, 0.02), 1.00 => (0.00, 0.00, 0.00, 0.00, 0.04),
    )
    # Resistivity: p => (R₁₁, R₃₃).
    theirR = [
        (0.20, 2.024681, 15.1971865), (0.25, 1.715557, 6.631911),
        (0.30, 1.640000, 3.939221), (0.35, 1.548567, 2.823389),
        (0.40, 1.528998, 2.289616), (0.45, 1.511922, 1.999021),
        (0.50, 1.506300, 1.832213), (0.55, 1.505546, 1.728305),
        (0.60, 1.505895, 1.662567), (0.65, 1.507544, 1.616467),
        (0.70, 1.509126, 1.587973), (0.75, 1.509902, 1.559419),
        (0.80, 1.509923, 1.539122), (0.85, 1.508904, 1.523779),
        (0.90, 1.507043, 1.509388), (0.95, 1.507253, 1.505122),
        (1.00, 1.501244, 1.496256),
    ]

    # The inferred material, and the cell. `R/a = 6` and `nradial = 28`: what
    # remains after the dipole correction is truncation, so the cell radius is
    # the knob that matters and the mesh density is already past its plateau.
    νs, Es, ks = 1 / 3, 1.0, 1.0
    Cs = iso_stiffness(Es / (3 * (1 - 2νs)), Es / (2 * (1 + νs)))
    Ks = TensISO{3}(ks)
    opts = FEAxiMeshOptions(; nradial = 28, radius_ratio = 6.0)

    ps = sort(unique(vcat([r[1] for r in theirH], [r[1] for r in theirR])))
    mine = Dict{Float64, Any}()
    for p in ps
        pore = FEAxiSupershapePore(Superspheroid(1.0, 1.0, p); opts)
        H = Matrix(KM(compliance_contribution(pore, Cs, Cs)))
        R = Matrix(get_array(resistivity_contribution(pore, Ks, Ks)))
        mine[p] = (
            H = (H[1, 1], H[1, 2], H[1, 3], H[3, 3], H[5, 5] / 2),
            R = (R[1, 1], R[3, 3]),
        )
        @printf "  p = %.2f  H1111 = %8.4f  H3333 = %9.4f  R33 = %8.4f\n" p H[1, 1] H[3, 3] R[3, 3]
        flush(stdout)
    end

    # One panel: our line and their published points, and nothing else. Their
    # Table B.3 — the change between their two meshes — is *not* drawn on their
    # B.1 values: it is a separate quantity, and superimposing it as an error
    # bar would present a construction of ours as their own presentation. It
    # appears once, as its own curve, in the deviation panel.
    function panel(ttl, xs, theirs, ours; logy = false, leg = false, kw...)
        plt = plot(;
            title = ttl, xlabel = "p", legend = leg ? :best : false,
            titlefontsize = 9, yscale = logy ? :log10 : :identity, kw...,
        )
        # Markers on our curve too: it is nineteen finite-element solves at the
        # abscissae *they* tabulate, not an interpolant, and a bare line invites
        # the reader to assume otherwise.
        plot!(
            plt, xs, ours; lc = :crimson, lw = 1.5, marker = :diamond, ms = 3,
            msw = 0, mc = :crimson, label = "this package (FE)",
        )
        scatter!(
            plt, xs, theirs; mc = :steelblue, ms = 4.5, msw = 0.6, msc = :white,
            label = "Sevostianov et al. (2016)",
        )
        return plt
    end

    pH = [r[1] for r in theirH]
    pR = [r[1] for r in theirR]
    # `H` has the dimensions of a compliance and `R` of a resistivity, so what
    # is plotted is `E₀H` and `k₀R` — dimensionless, and the form in which the
    # paper reports them. `E₀ = k₀ = 1` here, so the numbers coincide with the
    # tables; the labels still have to say so.
    labels = ("E₀H₁₁₁₁", "E₀H₁₁₂₂", "E₀H₁₁₃₃", "E₀H₃₃₃₃", "E₀H₁₃₁₃")
    panels = Any[]
    for c in 1:5
        th = [r[c + 1] for r in theirH]
        ou = [mine[p].H[c] for p in pH]
        # `E₀H₃₃₃₃` and `E₀H₁₃₁₃` run over more than a decade, so those two are
        # plotted logarithmically; a linear axis there would hide everything
        # below `p = 0.4` under the two leftmost points.
        logy = c in (4, 5)
        push!(panels, panel(labels[c], pH, th, ou; logy, leg = c == 1))
    end
    devs = Any[]
    for c in 1:5
        th = [r[c + 1] for r in theirH]
        push!(devs, (labels[c], pH, [100 * (mine[p].H[c] - th[i]) / abs(th[i]) for (i, p) in enumerate(pH)]))
    end
    for (c, lbl) in ((1, "k₀R₁₁"), (2, "k₀R₃₃"))
        th = [r[c + 1] for r in theirR]
        ou = [mine[p].R[c] for p in pR]
        push!(panels, panel(lbl, pR, th, ou; logy = c == 2))
        push!(devs, (lbl, pR, [100 * (ou[i] - th[i]) / abs(th[i]) for i in eachindex(pR)]))
    end

    # The eighth panel: every deviation on one logarithmic axis, against the
    # largest change between *their* two meshes. Below that gray line, the two
    # calculations agree as closely as theirs agrees with itself.
    dp = plot(;
        title = "|deviation| from their tables", xlabel = "p",
        ylabel = "percent", yscale = :log10, legend = :topright,
        titlefontsize = 9, legendfontsize = 6, ylims = (1.0e-2, 1.0e2),
    )
    # Their Table B.3, plotted as itself: the largest relative change between
    # their two meshes at each `p`. It is their own resolution, published by
    # them, and it is the yardstick a deviation should be read against — but it
    # belongs on an axis of deviations, not on their tabulated values.
    plot!(
        dp, pH, [maximum(theirMesh[p]) for p in pH]; lc = :grey40, ls = :dash,
        lw = 2, label = "their Table B.3 (mesh a vs b)",
    )
    # The five compliance components share one color and one legend entry: the
    # panel's message is which quantity is the outlier, and eight named series
    # squeeze the axes without adding it.
    for (i, (lbl, xs, d)) in enumerate(devs)
        style = if lbl == "k₀R₁₁"
            (lc = :crimson, label = "k₀R₁₁")
        elseif lbl == "k₀R₃₃"
            (lc = :darkorange, label = "k₀R₃₃")
        else
            (lc = :steelblue, label = i == 1 ? "the five E₀H components" : "")
        end
        plot!(
            dp, xs, max.(abs.(d), 1.0e-2); lw = 1.5, marker = :circle, ms = 2.5,
            msw = 0, mc = style.lc, style...,
        )
    end
    push!(panels, dp)
    savefig(
        plot(
            panels...; layout = (2, 4), size = (1520, 700),
            left_margin = 7Plots.mm, bottom_margin = 7Plots.mm,
        ),
        joinpath(OUT, "axi_vs_sevostianov.png"),
    )
    println("wrote ", joinpath(OUT, "axi_vs_sevostianov.png"))

    # The table goes to stdout as well as to the report buffer, so a partial
    # run still shows its numbers instead of only writing a file it declines
    # to touch.
    tee(io, args...) = (println(io, args...); println(stdout, args...))
    println(report, "### Against Sevostianov et al. (2016), Tables B.1 and B.4\n")
    println(report, "Inferred material: `E₀ = 1`, `ν₀ = 1/3`, `k₀ = 1`. ")
    println(report, "`R/a = 6`, `nradial = 28`. Deviations in percent of their value; ")
    println(report, "the last column is the largest change between *their* two meshes.\n")
    tee(
        report,
        "| `p` | `H₁₁₁₁` | `H₁₁₂₂` | `H₁₁₃₃` | `H₃₃₃₃` | `H₁₃₁₃` | `k₀R₁₁` | `k₀R₃₃` | their mesh |",
    )
    tee(report, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in ps
        row = IOBuffer()
        @printf row "| %.2f |" p
        for c in 1:5
            if haskey(theirMesh, p)
                th = only(r[c + 1] for r in theirH if r[1] == p)
                @printf row " %+.2f |" (100 * (mine[p].H[c] - th) / abs(th))
            else
                print(row, " — |")
            end
        end
        for c in 1:2
            i = findfirst(r -> r[1] == p, theirR)
            if i === nothing
                print(row, " — |")
            else
                th = theirR[i][c + 1]
                @printf row " %+.2f |" (100 * (mine[p].R[c] - th) / abs(th))
            end
        end
        if haskey(theirMesh, p)
            @printf row " %.2f |" maximum(theirMesh[p])
        else
            print(row, " — |")
        end
        tee(report, String(take!(row)))
    end
    println(report)
end

# The report buffer is flushed last, once every section above has written to
# it — a partial run leaves the committed file alone rather than truncating it
# to whatever sections were asked for.
SECTIONS === nothing ?
    write(joinpath(OUT, "cell_results.md.in"), String(take!(report))) :
    @info "partial run ($(join(ARGS, ", "))) — cell_results.md.in left untouched"
