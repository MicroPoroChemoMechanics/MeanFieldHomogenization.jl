# =============================================================================
#  make_doc_figures.jl — regenerate the static assets of the documentation page
#  `docs/src/manual/fe_inclusions.md`.
#
#  Maintenance script, run **by hand**, never at documentation-build time: the
#  finite-element cases below mesh a ball and factorize up to a 2·10⁵-dof system
#  a dozen times over. The page embeds the committed PNGs and quotes the
#  numbers printed here, so a doc build stays free of `gmsh_jll` and of any
#  finite-element work.
#
#      julia scripts/fe/make_doc_figures.jl
#
#  Outputs (committed):
#      docs/src/assets/fe/mesh_crack_plane.png
#      docs/src/assets/fe/mesh_slice.png
#      docs/src/assets/fe/mesh_3d.png
#      docs/src/assets/fe/convergence.png
#      docs/src/assets/fe/dipole_scaling.png
#      docs/src/assets/fe/results.md      (tables, pasted into the page)
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

const νm, Em = 0.3, 1.0
const C₀ = iso_stiffness(Em / (3 * (1 - 2νm)), Em / (2 * (1 + νm)))

diag3(B) = diag(Matrix(get_array(B)))

# The plotting needs the discretization itself, which is an implementation
# detail rather than public API — this is a maintenance script, so it reaches in
# deliberately. Since the `src/FiniteElements` reorganization the mesh and the
# facet-set names belong to the package, and only the lip split is the backend's.
const FE = MeanFieldHomogenization.FiniteElements
const EXT = Base.get_extension(MeanFieldHomogenization, :MeanFieldHomogenizationFerriteExt)

"Grid of a crack, and the `+` lip facets of its crack plane."
function crack_mesh(crack)
    grid = FE._crack_setup(crack).grid
    lip_up, _ = EXT.split_crack_lips(grid)
    return grid, lip_up
end

# ─── Figure 1 — the mesh in the crack plane ──────────────────────────────────

"""
Triangulation of the upper lip, seen from `+z`. Shows the graded refinement:
elements shrink towards the elliptical front, where the displacement field has
a square-root singularity.
"""
function figure_crack_plane(crack; kw...)
    grid, lip_up = crack_mesh(crack)
    a, b = crack.a, crack.b
    plt = plot(;
        aspect_ratio = 1, xlabel = "x / a", ylabel = "y / a",
        title = "Crack plane z = 0 — upper lip", legend = false, kw...,
    )
    for fi in lip_up
        cell = grid.cells[fi[1]]
        nodes = Ferrite.facets(cell)[fi[2]]
        pts = [Ferrite.getnodes(grid, n).x for n in nodes]
        push!(pts, pts[1])
        plot!(
            plt, [p[1] / a for p in pts], [p[2] / a for p in pts];
            lc = :steelblue, lw = 0.35
        )
    end
    θ = range(0, 2π; length = 400)
    plot!(plt, cos.(θ), (b / a) .* sin.(θ); lc = :crimson, lw = 2)
    return plt
end

# ─── Figure 2 — a slice of the ball ──────────────────────────────────────────

"Polygon where the tetrahedron `p[1..4]` meets the plane `y = 0`."
function tet_slice(p)
    pts = NTuple{2, Float64}[]
    for (i, j) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
        yi, yj = p[i][2], p[j][2]
        (yi < 0) == (yj < 0) && continue
        t = yi / (yi - yj)
        push!(pts, (p[i][1] + t * (p[j][1] - p[i][1]), p[i][3] + t * (p[j][3] - p[i][3])))
    end
    length(pts) < 3 && return pts
    cx = sum(first, pts) / length(pts)
    cz = sum(last, pts) / length(pts)
    return sort(pts; by = q -> atan(q[2] - cz, q[1] - cx))
end

"""
Section `y = 0` of the ball of matrix: the radial grading, from `h_tip` at the
crack front out to `R/3` on the outer boundary.
"""
function figure_slice(crack; zoom = nothing, kw...)
    grid, _ = crack_mesh(crack)
    a = crack.a
    plt = plot(;
        aspect_ratio = 1, xlabel = "x / a", ylabel = "z / a",
        legend = false, kw...,
    )
    for cid in 1:Ferrite.getncells(grid)
        poly = tet_slice(Ferrite.getcoordinates(grid, cid))
        length(poly) < 3 && continue
        xs = [q[1] / a for q in poly]
        zs = [q[2] / a for q in poly]
        push!(xs, xs[1])
        push!(zs, zs[1])
        plot!(plt, xs, zs; lc = :grey35, lw = 0.25)
    end
    plot!(plt, [-1, 1], [0, 0]; lc = :crimson, lw = 2.5)
    zoom !== nothing && plot!(plt; xlims = (-zoom, zoom), ylims = (-zoom, zoom))
    return plt
end

# ─── Figure 3 — cut-away 3-D view ────────────────────────────────────────────

"Append the closed polyline `pts` to `(X, Y, Z)`, NaN-separated."
function push_loop!(X, Y, Z, pts)
    for p in pts
        push!(X, p[1])
        push!(Y, p[2])
        push!(Z, p[3])
    end
    push!(X, pts[1][1])
    push!(Y, pts[1][2])
    push!(Z, pts[1][3])
    push!(X, NaN)
    push!(Y, NaN)
    push!(Z, NaN)
    return nothing
end

"""
Cut-away perspective of the ball: the half `y ≤ 0` of the outer surface, the
`y = 0` cut face — which exposes the interior grading — and the crack lips in
red. Everything is drawn as a single NaN-separated polyline per color, which
keeps GR fast.
"""
function figure_3d(crack; half = nothing, outer = true, slice = true, kw...)
    grid, lip_up = crack_mesh(crack)
    a = crack.a
    inbox(pts) = half === nothing ||
        all(abs(p[1]) ≤ half && abs(p[2]) ≤ half && abs(p[3]) ≤ half for p in pts)

    # Outer sphere: the far half only, so the near side does not veil the crack.
    Xo, Yo, Zo = Float64[], Float64[], Float64[]
    if outer
        for fi in Ferrite.getfacetset(grid, FE.SET_OUTER)
            nodes = Ferrite.facets(grid.cells[fi[1]])[fi[2]]
            pts = [Ferrite.getnodes(grid, n).x ./ a for n in nodes]
            sum(p[2] for p in pts) / length(pts) ≤ 0 || continue
            push_loop!(Xo, Yo, Zo, pts)
        end
    end

    # Cut face y = 0, on the far side of the crack only (y ≤ 0 is behind it).
    Xc, Yc, Zc = Float64[], Float64[], Float64[]
    if slice
        for cid in 1:Ferrite.getncells(grid)
            poly = tet_slice(Ferrite.getcoordinates(grid, cid))
            length(poly) < 3 && continue
            pts3 = [(q[1] / a, 0.0, q[2] / a) for q in poly]
            inbox(pts3) || continue
            push_loop!(Xc, Yc, Zc, pts3)
        end
    end

    # The crack itself, in full: an ellipse floating in the ball.
    Xk, Yk, Zk = Float64[], Float64[], Float64[]
    for fi in lip_up
        nodes = Ferrite.facets(grid.cells[fi[1]])[fi[2]]
        pts = [Ferrite.getnodes(grid, n).x ./ a for n in nodes]
        push_loop!(Xk, Yk, Zk, pts)
    end

    L = half === nothing ? crack.mesh.radius_ratio * 1.02 : half
    plt = plot(;
        legend = false, xlabel = "x / a", ylabel = "y / a", zlabel = "z / a",
        camera = (35, 30), xlims = (-L, L), ylims = (-L, L), zlims = (-L, L), kw...,
    )
    outer && plot!(plt, Xo, Yo, Zo; lc = RGBA(0.42, 0.55, 0.72, 0.3), lw = 0.4)
    slice && plot!(plt, Xc, Yc, Zc; lc = RGBA(0.35, 0.35, 0.35, 0.55), lw = 0.3)
    plot!(plt, Xk, Yk, Zk; lc = :crimson, lw = 0.6)
    return plt
end

# ─── Runs ────────────────────────────────────────────────────────────────────

println("── figures: mesh ───────────────────────────────────────────────────")
mesh_crack = FEEllipticCrack(1.0, 0.25; htipdiv = 6.0)
rep = fe_mesh_report(mesh_crack)
@printf "  %d cells, %d dofs; lips %d/%d, area %.6f vs πab = %.6f\n" rep.ncells rep.ndofs rep.nfacets_up rep.nfacets_dn rep.area_up rep.area_exact

savefig(
    figure_crack_plane(mesh_crack; size = (620, 360)),
    joinpath(OUT, "mesh_crack_plane.png")
)

p_full = figure_slice(mesh_crack; title = "Section y = 0", size = (430, 400))
p_zoom = figure_slice(mesh_crack; zoom = 1.6, title = "zoom on the crack", size = (430, 400))
savefig(
    plot(p_full, p_zoom; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (900, 420)),
    joinpath(OUT, "mesh_slice.png")
)

savefig(
    plot(
        figure_3d(
            mesh_crack; slice = false,
            title = "the crack inside the ball of matrix", size = (460, 440)
        ),
        figure_3d(
            mesh_crack; half = 1.8, outer = false,
            title = "zoom: crack and the cut behind it", size = (460, 440)
        );
        layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (980, 460)
    ),
    joinpath(OUT, "mesh_3d.png")
)

println("── dipole scaling ──────────────────────────────────────────────────")
ratios = [3.0, 5.0, 7.0, 10.0]
Bu = Float64[]
for rr in ratios
    d = fe_cod_breakdown(FEEllipticCrack(1.0, 0.25; htipdiv = 6.0, radius_ratio = rr), C₀)
    push!(Bu, norm(d.B_u))
    @printf "  R/a = %4.1f   ‖B_u‖ = %.4e\n" rr norm(d.B_u)
    flush(stdout)
end
slope = (log(Bu[end]) - log(Bu[1])) / (log(ratios[end]) - log(ratios[1]))
@printf "  fitted slope on log-log = %.3f   (theory: -3)\n" slope

plt_dip = plot(
    ratios, Bu;
    xscale = :log10, yscale = :log10, marker = :circle, lw = 2, lc = :steelblue,
    label = "‖B_u‖ (finite elements)",
    xlabel = "R / a", ylabel = "‖B_u‖",
    title = "Weight of the boundary correction", size = (620, 400)
)
plot!(
    plt_dip, ratios, Bu[1] .* (ratios ./ ratios[1]) .^ (-3);
    ls = :dash, lw = 2, lc = :crimson, label = "slope −3, i.e. (a/R)³"
)
savefig(plt_dip, joinpath(OUT, "dipole_scaling.png"))

println("── convergence ─────────────────────────────────────────────────────")
function convergence(a, b, htipdivs)
    ana = diag3(cod_tensor(EllipticCrack(a, b), C₀))
    rows = Vector{Any}[]
    Bs = Vector{Float64}[]
    for hd in htipdivs
        c = FEEllipticCrack(a, b; htipdiv = hd)
        B = diag3(cod_tensor(c, C₀))
        push!(Bs, B)
        push!(rows, Any[hd, fe_mesh_report(c).ndofs, B, 100 .* (B .- ana) ./ ana])
        @printf "  b/%-3g  ndof=%7d  B=[%.5f %.5f %.5f]\n" hd rows[end][2] B[1] B[2] B[3]
        flush(stdout)
    end
    h1, h2 = 1 / htipdivs[end - 1], 1 / htipdivs[end]
    B0 = Bs[end] .+ (Bs[end] .- Bs[end - 1]) .* (h2 / (h1 - h2))
    return (; ana, rows, B0, err0 = 100 .* (B0 .- ana) ./ ana, htipdivs, Bs)
end

penny = convergence(1.0, 1.0, (4, 6, 9, 12))
ellip = convergence(1.0, 0.25, (6, 9, 12))

plt_cv = plot(;
    xlabel = "element size at the front,  h = b / htipdiv",
    ylabel = "relative error on B₃₃  (%)",
    title = "Convergence of the finite-element COD tensor",
    size = (640, 420), legend = :bottomright
)
for (nm, r, col) in (
        ("penny, b/a = 1", penny, :steelblue),
        ("elliptic, b/a = 1/4", ellip, :darkorange),
    )
    h = [1 / hd for hd in r.htipdivs]
    e = [abs(row[4][3]) for row in r.rows]
    plot!(plt_cv, h, e; marker = :circle, lw = 2, lc = col, mc = col, label = nm)
    plot!(
        plt_cv, [0.0, h[1]], [abs(r.err0[3]), abs(r.err0[3]) + (e[1] - abs(r.err0[3]))];
        ls = :dot, lw = 1.5, lc = col, label = ""
    )
    scatter!(
        plt_cv, [0.0], [abs(r.err0[3])]; mc = col, ms = 7, marker = :star5,
        label = "$nm — Richardson h→0"
    )
end
savefig(plt_cv, joinpath(OUT, "convergence.png"))

# ─── Numbers, for pasting into the page ──────────────────────────────────────

open(joinpath(OUT, "results.md"), "w") do io
    println(io, "<!-- Generated by scripts/fe/make_doc_figures.jl — do not edit by hand. -->\n")
    println(io, "### Mesh (a = 1, b = 0.25, R/a = 5, htipdiv = 6)\n")
    @printf io "- %d cells, %d nodes, %d dofs\n" rep.ncells rep.nnodes rep.ndofs
    @printf io "- lips: %d / %d facets, area %.6f each vs exact πab = %.6f (%.2f %%)\n\n" rep.nfacets_up rep.nfacets_dn rep.area_up rep.area_exact 100abs(rep.area_up - rep.area_exact) / rep.area_exact

    println(io, "### Weight of the boundary correction\n")
    println(io, "| R/a | ‖B_u‖ |")
    println(io, "|---|---|")
    for (rr, v) in zip(ratios, Bu)
        @printf io "| %.0f | %.3e |\n" rr v
    end
    @printf io "\nFitted log-log slope: **%.2f** (theory −3).\n\n" slope

    for (nm, r) in (
            ("Penny-shaped crack (b/a = 1)", penny),
            ("Elliptical crack (b/a = 1/4)", ellip),
        )
        println(io, "### $nm\n")
        @printf io "Closed form: `diag(B) = [%.5f, %.5f, %.5f]`\n\n" r.ana[1] r.ana[2] r.ana[3]
        println(io, "| h at the front | dofs | diag(B) | relative error (%) |")
        println(io, "|---|---|---|---|")
        for row in r.rows
            @printf io "| b/%g | %d | [%.5f, %.5f, %.5f] | [%+.2f, %+.2f, %+.2f] |\n" row[1] row[2] row[3][1] row[3][2] row[3][3] row[4][1] row[4][2] row[4][3]
        end
        @printf io "| **h → 0** (Richardson) | — | **[%.5f, %.5f, %.5f]** | **[%+.2f, %+.2f, %+.2f]** |\n\n" r.B0[1] r.B0[2] r.B0[3] r.err0[1] r.err0[2] r.err0[3]
    end
end

println("\nwrote figures and results.md to ", OUT)
