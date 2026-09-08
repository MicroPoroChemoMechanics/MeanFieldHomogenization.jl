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
#      docs/src/assets/fe/cell_mesh_octant.png
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

SECTIONS === nothing ?
    write(joinpath(OUT, "cell_results.md.in"), String(take!(report))) :
    @info "partial run ($(join(ARGS, ", "))) — cell_results.md.in left untouched"
