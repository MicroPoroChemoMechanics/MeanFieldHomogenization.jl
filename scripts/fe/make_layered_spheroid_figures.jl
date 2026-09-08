# =============================================================================
#  make_layered_spheroid_figures.jl — the static assets of
#  `docs/src/tutorials/axi_layered_spheroid.md`.
#
#  Maintenance script, run **by hand**, never at documentation-build time:
#  `docs/Project.toml` carries neither Ferrite nor gmsh, and the build is long
#  enough already. The page embeds the committed PNGs and the numbers this
#  script measures.
#
#      julia scripts/fe/make_layered_spheroid_figures.jl          # everything
#      julia scripts/fe/make_layered_spheroid_figures.jl mesh     # one section
#
#  Outputs (committed):
#      docs/src/assets/fe/layered_spheroid_mesh.png
#      docs/src/assets/fe/layered_spheroid_vs_analytic.png
#      docs/src/assets/fe/layered_spheroid_results.md.in
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
const report = IOBuffer()

_km(t) = Matrix(KM(TensND.change_tens(t, TensND.CanonicalBasis{3, Float64}())))
_ar(t) = Matrix(TensND.get_array(TensND.change_tens(t, TensND.CanonicalBasis{3, Float64}())))
Ciso(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))

# ─── The meridian mesh ───────────────────────────────────────────────────────

"""
One panel: the meridian half-plane of a layered spheroid, each layer's boundary
drawn from its own closed form rather than from the mesh, and the axis in blue.

Two-dimensional, so this is the whole computational domain and not a slice of
one — which is the point of the figure.
"""
function figure_layers(ar, dr, opts; zoom = 1.35, kw...)
    b = FE.FerriteBackend()
    n = length(ar)
    K = ntuple(_ -> TensISO{3}(1.0), n)
    incl = FEAxiLayeredSpheroid(ar, dr, K; opts)
    grid = FE.fe_axi_grid(b, incl)
    X, Y = Float64[], Float64[]
    for ci in 1:Ferrite.getncells(grid)
        c = Ferrite.getcoordinates(grid, ci)
        append!(X, [c[1][1], c[2][1], c[3][1], c[1][1], NaN])
        append!(Y, [c[1][2], c[2][2], c[3][2], c[1][2], NaN])
    end
    R = opts.radius_ratio * max(maximum(ar), maximum(dr))
    # Zoomed on the inclusion by default: at `R/a = 3` the layers occupy a third
    # of the panel and their boundaries are indistinguishable, which defeats the
    # purpose of showing them. `zoom = Inf` gives the whole cell.
    W = isfinite(zoom) ? zoom * max(maximum(ar), maximum(dr)) : R
    plt = plot(;
        aspect_ratio = 1, legend = false, xlabel = "ρ", ylabel = "z",
        xlims = (-0.05W, 1.05W), ylims = (-1.05W, 1.05W), titlefontsize = 9, kw...,
    )
    plot!(plt, X, Y; lc = :grey70, lw = 0.3)
    for ℓ in 1:n
        t = range(0, 2π; length = 400)
        plot!(plt, dr[ℓ] .* sin.(t), ar[ℓ] .* cos.(t); lc = :crimson, lw = 1.8)
    end
    plot!(plt, [0.0, 0.0], [-R, R]; lc = :steelblue, lw = 2)
    return plt, grid, incl
end

want("mesh") && let
    println("── figure: the meridian mesh ──────────────────────────────────")
    o = FEAxiMeshOptions(; nradial = 16, radius_ratio = 3.0)
    cases = (
        ("confocal prolate, 2 layers", confocal_layer_radii(2.0, 1.0, (0.3, 0.7))...),
        ("confocal oblate, 3 layers", confocal_layer_radii(0.5, 1.0, (0.25, 0.35, 0.4))...),
        # The panel that shows what the confocal family forbids: an oblate core
        # inside a prolate shell. Nested, axisymmetric, no closed form.
        ("free radii, aspect reverses", (0.4, 1.4), (0.9, 1.0)),
    )
    panels = Any[]
    println(report, "### The meridian mesh (nradial = 16, R/a = 3)\n")
    println(report, "| geometry | layers | cells | worst layer volume error |")
    println(report, "|---|---:|---:|---:|")
    for (tag, ar, dr) in cases
        plt, grid, incl = figure_layers(ar, dr, o; title = tag, size = (340, 460))
        push!(panels, plt)
        b = FE.FerriteBackend()
        ex = layer_volumes(incl)
        worst = maximum(
            abs(FE.fe_axi_region_volume(b, grid, axi_layer_set(ℓ)) - ex[ℓ]) / ex[ℓ]
            for ℓ in eachindex(ex)
        )
        @printf report "| %s | %d | %d | %.1e |\n" tag length(ar) Ferrite.getncells(grid) worst
        @printf "  %-30s %6d cells, worst volume %.1e\n" tag Ferrite.getncells(grid) worst
        flush(stdout)
    end
    println(report)
    # A fourth panel with the whole cell, so the reader sees what the three
    # zoomed ones are a detail of: the matrix reaches `R = 3` times the outer
    # semi-axis, and that is where the corrected boundary condition acts.
    ctx, _, _ = figure_layers(
        cases[1][2], cases[1][3], o;
        zoom = Inf, title = "the whole cell, R/a = 3", size = (340, 460),
    )
    push!(panels, ctx)
    savefig(
        plot(
            panels...; layout = (1, 4), size = (1360, 460),
            left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
        ),
        joinpath(OUT, "layered_spheroid_mesh.png"),
    )
    println("wrote ", joinpath(OUT, "layered_spheroid_mesh.png"))
end

# ─── Against the two exact families ──────────────────────────────────────────

want("validation") && let
    println("── figure: against the two exact families ─────────────────────")
    K₀ = TensISO{3}(1.0)

    # (a) Confocal, sweeping the outer aspect ratio from oblate to prolate
    #     through the sphere. The sphere is not in the confocal family, so it is
    #     approached rather than reached, and both branches of the analytic
    #     solution are exercised on one curve.
    # One sweep for both physics now. The elastic branch used to be excluded
    # near the sphere, where the confocal chart degenerates and its system's
    # columns spanned `q^{2n+1}`; equilibrating them moved the usable range from
    # `|1 - ω| ≥ 0.3` to `1e-3`, so the same abscissae serve both.
    ωs = [0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
    ωs_el = ωs
    fe11, fe33, an11, an33 = Float64[], Float64[], Float64[], Float64[]
    for ω in ωs
        ar, dr = confocal_layer_radii(ω, 1.0, (0.3, 0.7))
        K = (TensISO{3}(5.0), TensISO{3}(2.0))
        A = _ar(gradient_gradient_loc(
            FEAxiLayeredSpheroid(
                ar, dr, K; opts = FEAxiMeshOptions(; nradial = 14, radius_ratio = 6.0)
            ), K₀, K₀,
        ))
        E = _ar(gradient_gradient_loc(LayeredSpheroid(ar, dr, K), K₀, K₀))
        push!(fe11, A[1, 1]); push!(fe33, A[3, 3])
        push!(an11, E[1, 1]); push!(an33, E[3, 3])
        @printf "  confocal ω = %.2f   A11 %.6f / %.6f   A33 %.6f / %.6f\n" ω A[1, 1] E[1, 1] A[3, 3] E[3, 3]
        flush(stdout)
    end
    p1 = plot(;
        title = "confocal, 2 layers: A_∇∇ against LayeredSpheroid",
        xlabel = "outer aspect ratio ω", ylabel = "component",
        titlefontsize = 9, legend = :best,
    )
    plot!(p1, ωs, an11; lc = :steelblue, lw = 2, label = "A₁₁ analytic")
    scatter!(p1, ωs, fe11; mc = :steelblue, ms = 4, msw = 0, label = "A₁₁ finite elements")
    plot!(p1, ωs, an33; lc = :crimson, lw = 2, label = "A₃₃ analytic")
    scatter!(p1, ωs, fe33; mc = :crimson, ms = 4, msw = 0, marker = :diamond, label = "A₃₃ finite elements")

    # (b) Concentric spheres of free radii, against LayeredSphere: the slice
    #     that covers arbitrary radii without assuming anything confocal.
    fr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sf, sa = Float64[], Float64[]
    for r in fr
        K = (TensISO{3}(5.0), TensISO{3}(2.0))
        A = _ar(gradient_gradient_loc(
            FEAxiLayeredSpheroid(
                (r, 1.0), (r, 1.0), K;
                opts = FEAxiMeshOptions(; nradial = 16, radius_ratio = 6.0),
            ), K₀, K₀,
        ))
        E = _ar(gradient_gradient_loc(LayeredSphere((r, 1.0), K), K₀, K₀))
        push!(sf, A[1, 1]); push!(sa, E[1, 1])
        @printf "  spheres r = %.2f   A %.6f / %.6f\n" r A[1, 1] E[1, 1]
        flush(stdout)
    end
    p2 = plot(;
        title = "concentric spheres, free radii: against LayeredSphere",
        xlabel = "core radius / outer radius", ylabel = "A_∇∇",
        titlefontsize = 9, legend = :best,
    )
    plot!(p2, fr, sa; lc = :steelblue, lw = 2, label = "analytic")
    scatter!(p2, fr, sf; mc = :steelblue, ms = 4, msw = 0, label = "finite elements")

    # (c) The deviations, on one logarithmic axis.
    p3 = plot(;
        title = "|deviation| from the closed form", xlabel = "sweep parameter",
        ylabel = "percent", yscale = :log10, titlefontsize = 9, legend = :best,
    )
    plot!(
        p3, ωs, max.(100 .* abs.(fe11 .- an11) ./ abs.(an11), 1.0e-4);
        lc = :steelblue, lw = 1.5, marker = :circle, ms = 3, msw = 0, mc = :steelblue,
        label = "confocal A₁₁ (vs ω)",
    )
    plot!(
        p3, ωs, max.(100 .* abs.(fe33 .- an33) ./ abs.(an33), 1.0e-4);
        lc = :crimson, lw = 1.5, marker = :diamond, ms = 3, msw = 0,
        label = "confocal A₃₃ (vs ω)",
    )
    plot!(
        p3, fr, max.(100 .* abs.(sf .- sa) ./ abs.(sa), 1.0e-4);
        lc = :darkorange, lw = 1.5, marker = :square, ms = 3, msw = 0,
        label = "spheres (vs r)",
    )
    # ── The same two slices in elasticity ────────────────────────────────
    #
    #  Five constants instead of two. The *stress* side is compared on the
    #  sphere slice only: it is not derivable from the strain side for a
    #  heterogeneous inclusion, and `LayeredSpheroid` supplies none in
    #  elasticity — the generic that used to answer was wrong by 110 %.
    C₀ = Ciso(1.0, 0.25)
    Cs = (Ciso(4.0, 0.2), Ciso(1.5, 0.3))
    labels4 = ("A₁₁₁₁", "A₃₃₃₃", "A₁₃₁₃")
    idx4 = ((1, 1), (3, 3), (5, 5))
    feE = [Float64[] for _ in idx4]
    anE = [Float64[] for _ in idx4]
    for ω in ωs_el
        ar, dr = confocal_layer_radii(ω, 1.0, (0.3, 0.7))
        fe = FEAxiLayeredSpheroid(
            ar, dr, Cs; opts = FEAxiMeshOptions(; nradial = 12, radius_ratio = 6.0)
        )
        ana = LayeredSpheroid(ar, dr, Cs)
        A = _km(strain_strain_loc(fe, C₀, C₀))
        E = _km(strain_strain_loc(ana, C₀, C₀))
        for (k, (i, j)) in enumerate(idx4)
            push!(feE[k], A[i, j]); push!(anE[k], E[i, j])
        end
        # No stress-side curve here, and that is a finding rather than an
        # omission: `LayeredSpheroid` implements no elastic `stress_strain_loc`,
        # so the call used to fall through to the generic `ℂ₁ : 𝔸_εε` — which
        # presupposes a uniform stiffness a multilayer does not have, and was
        # wrong by 110 % at every aspect ratio. The generic now refuses instead,
        # so the stress side is compared on the sphere slice below, where
        # `LayeredSphere` supplies a real one.
        @printf "  elastic ω = %.2f   A1111 %.6f / %.6f\n" ω A[1, 1] E[1, 1]
        flush(stdout)
    end
    p4 = plot(;
        title = "confocal, 2 layers: A_eps-eps against LayeredSpheroid",
        xlabel = "outer aspect ratio ω", ylabel = "Kelvin-Mandel component",
        titlefontsize = 9, legend = :best,
    )
    for (k, col) in enumerate((:steelblue, :crimson, :seagreen))
        plot!(p4, ωs_el, anE[k]; lc = col, lw = 2, label = "$(labels4[k]) analytic")
        scatter!(
            p4, ωs_el, feE[k]; mc = col, ms = 4, msw = 0,
            marker = (:circle, :diamond, :square)[k], label = "$(labels4[k]) FE",
        )
    end

    # Spheres, elasticity, both sides of gate B.
    feA, anA, feB, anB = Float64[], Float64[], Float64[], Float64[]
    for r in fr
        fe = FEAxiLayeredSpheroid(
            (r, 1.0), (r, 1.0), Cs;
            opts = FEAxiMeshOptions(; nradial = 16, radius_ratio = 6.0),
        )
        ana = LayeredSphere((r, 1.0), Cs)
        push!(feA, _km(strain_strain_loc(fe, C₀, C₀))[1, 1])
        push!(anA, _km(strain_strain_loc(ana, C₀, C₀))[1, 1])
        push!(feB, _km(stress_strain_loc(fe, C₀, C₀))[1, 1])
        push!(anB, _km(stress_strain_loc(ana, C₀, C₀))[1, 1])
        @printf "  elastic spheres r = %.2f   A %.6f / %.6f   B %.6f / %.6f\n" r feA[end] anA[end] feB[end] anB[end]
        flush(stdout)
    end
    p5 = plot(;
        title = "spheres, free radii: both sides of gate B",
        xlabel = "core radius / outer radius", ylabel = "component (1,1)",
        titlefontsize = 9, legend = :best,
    )
    plot!(p5, fr, anA; lc = :steelblue, lw = 2, label = "A_eps-eps analytic")
    scatter!(p5, fr, feA; mc = :steelblue, ms = 4, msw = 0, label = "A_eps-eps FE")
    plot!(p5, fr, anB; lc = :crimson, lw = 2, label = "A_sig-eps analytic")
    scatter!(p5, fr, feB; mc = :crimson, ms = 4, msw = 0, marker = :diamond, label = "A_sig-eps FE")

    p6 = plot(;
        title = "|deviation|, elasticity", xlabel = "sweep parameter",
        ylabel = "percent", yscale = :log10, titlefontsize = 9, legend = :best,
    )
    for (k, col) in enumerate((:steelblue, :crimson, :seagreen))
        d = abs.(feE[k] .- anE[k]) ./ abs.(anE[k])
        plot!(
            p6, ωs_el, max.(100 .* d, 1.0e-4); lc = col, lw = 1.5,
            marker = (:circle, :diamond, :square)[k], ms = 3, msw = 0,
            label = "$(labels4[k]) (vs ω)",
        )
    end
    plot!(
        p6, fr, max.(100 .* abs.(feA .- anA) ./ abs.(anA), 1.0e-4);
        lc = :darkorange, lw = 1.5, marker = :utriangle, ms = 3, msw = 0,
        label = "spheres A_eps-eps (vs r)",
    )

    savefig(
        plot(
            p1, p2, p3, p4, p5, p6; layout = (2, 3), size = (1400, 850),
            left_margin = 7Plots.mm, bottom_margin = 8Plots.mm,
        ),
        joinpath(OUT, "layered_spheroid_vs_analytic.png"),
    )
    println("wrote ", joinpath(OUT, "layered_spheroid_vs_analytic.png"))

    # ── Why the elastic sweep stops short of the sphere ───────────────────
    #
    #  A one-layer confocal spheroid *is* a homogeneous spheroid, so its answer
    #  is the closed-form Eshelby result and the confocal machinery has to
    #  reproduce it exactly. This panel is where the elastic branch's near-sphere
    #  defect was measured, and it is now the record that column equilibration
    #  removed it: from a total loss at `|1 - ω| = 1e-3` to `4e-9`, machine
    #  precision down to `1e-2`, and a residual limit only at `1e-4` where the
    #  spread reaches `q^{19} ≈ 10³⁵` and even transport is at `8e-10`.
    println("── figure: the analytic elastic branch near the sphere ────────")
    ωn = [0.999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 1.05, 1.1, 1.2, 1.3, 1.6]
    de, dc = Float64[], Float64[]
    K1 = TensISO{3}(5.0); C1 = Ciso(4.0, 0.2)
    for ω in ωn
        ar, dr = confocal_layer_radii(ω, 1.0, (1.0,))
        an = _km(strain_strain_loc(LayeredSpheroid(ar, dr, (C1,)), C₀, C₀))
        ex = _km(strain_strain_loc(Spheroid(ω), C1, C₀))
        push!(de, norm(an - ex) / norm(ex))
        ant = _ar(gradient_gradient_loc(LayeredSpheroid(ar, dr, (K1,)), K₀, K₀))
        ext = _ar(gradient_gradient_loc(Spheroid(ω), K1, K₀))
        push!(dc, norm(ant - ext) / norm(ext))
        @printf "  ω = %.3f   elastic %.2e   transport %.2e\n" ω de[end] dc[end]
        flush(stdout)
    end
    pn = plot(;
        title = "one confocal layer against the closed-form spheroid",
        xlabel = "|1 - ω|  (distance from the sphere)", ylabel = "relative error",
        xscale = :log10, yscale = :log10, titlefontsize = 9,
        legend = :bottomright,
    )
    x = abs.(1 .- ωn)
    o = sortperm(x)
    plot!(
        pn, x[o], max.(de[o], 1.0e-17); lc = :crimson, lw = 2, marker = :circle,
        ms = 4, msw = 0, mc = :crimson, label = "elasticity",
    )
    plot!(
        pn, x[o], max.(dc[o], 1.0e-17); lc = :steelblue, lw = 2, marker = :diamond,
        ms = 4, msw = 0, mc = :steelblue, label = "transport, same chart",
    )
    savefig(
        plot(pn; size = (620, 460), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm),
        joinpath(OUT, "layered_spheroid_analytic_near_sphere.png"),
    )
    println("wrote ", joinpath(OUT, "layered_spheroid_analytic_near_sphere.png"))

    println(report, "### The analytic elastic branch near the sphere\n")
    println(
        report,
        "One confocal layer is a homogeneous spheroid, so the closed-form " *
            "Eshelby result is the reference and the confocal machinery must " *
            "reproduce it exactly. Transport does, everywhere. Elasticity does " *
            "not, once the chart degenerates (`focal → 0`, `q → ∞`).\n",
    )
    println(report, "| `ω` | elasticity | transport, same chart |")
    println(report, "|---|---:|---:|")
    for (i, ω) in enumerate(ωn)
        @printf report "| %.3f | %.1e | %.1e |\n" ω de[i] dc[i]
    end
    println(report)

    println(report, "### Against the closed forms\n")
    println(
        report,
        "Two layers. Conduction: `k₁/k₀ = 5`, `k₂/k₀ = 2`. Elasticity: " *
            "`E₁/E₀ = 4`, `ν₁ = 0.2`, `E₂/E₀ = 1.5`, `ν₂ = 0.3`, `ν₀ = 0.25`.\n",
    )
    println(report, "| slice | worst deviation | median deviation |")
    println(report, "|---|---:|---:|")
    d1 = abs.(fe11 .- an11) ./ abs.(an11)
    d3 = abs.(fe33 .- an33) ./ abs.(an33)
    ds = abs.(sf .- sa) ./ abs.(sa)
    @printf report "| confocal, `A₁₁`, ω ∈ [0.3, 3] | %.1e | %.1e |\n" maximum(d1) sort(d1)[length(d1) ÷ 2]
    @printf report "| confocal, `A₃₃`, ω ∈ [0.3, 3] | %.1e | %.1e |\n" maximum(d3) sort(d3)[length(d3) ÷ 2]
    @printf report "| spheres, free radii | %.1e | %.1e |\n" maximum(ds) sort(ds)[length(ds) ÷ 2]
    for (k, lbl) in enumerate(labels4)
        d = abs.(feE[k] .- anE[k]) ./ abs.(anE[k])
        @printf report "| confocal, `%s`, ω ∈ [0.3, 3] | %.1e | %.1e |\n" lbl maximum(d) sort(d)[length(d) ÷ 2]
    end
    println(
        report,
        "| confocal, `𝔸_σε` — no reference | — | — |",
    )
    dse = abs.(feA .- anA) ./ abs.(anA)
    @printf report "| spheres, elasticity, `𝔸_εε` | %.1e | %.1e |\n" maximum(dse) sort(dse)[length(dse) ÷ 2]
    dsb = abs.(feB .- anB) ./ abs.(anB)
    @printf report "| spheres, elasticity, `𝔸_σε` | %.1e | %.1e |\n" maximum(dsb) sort(dsb)[length(dsb) ÷ 2]
    println(report)
end

# The report is flushed last, once every section has written to it; a partial
# run leaves the committed file alone rather than truncating it.
SECTIONS === nothing ?
    write(joinpath(OUT, "layered_spheroid_results.md.in"), String(take!(report))) :
    @info "partial run ($(join(ARGS, ", "))) — the .md.in was left untouched"
