# =============================================================================
#  make_axi_figures.jl — regenerate the static assets of the documentation page
#  `docs/src/applications/recycled_aggregate.md`.
#
#  Maintenance script, run **by hand**, never at documentation-build time. The
#  page embeds the committed PNGs and quotes the numbers printed here, so a doc
#  build stays free of `gmsh_jll` and of any finite-element work.
#
#      julia scripts/fe/make_axi_figures.jl              # everything
#      julia scripts/fe/make_axi_figures.jl mesh         # one section only
#
#  Section names: schematic, mesh, convergence, contrast, conductivity, cost.
#  Passing any
#  of them regenerates only those figures and leaves `axi_results.md` alone,
#  which is what you want while iterating on a plot.
#
#  Outputs (committed):
#      docs/src/assets/fe/axi_schematic.png
#      docs/src/assets/fe/axi_mesh.png
#      docs/src/assets/fe/axi_convergence.png
#      docs/src/assets/fe/axi_contrast.png
#      docs/src/assets/fe/axi_conductivity.png
#      docs/src/assets/fe/axi_results.md      (tables, pasted into the page)
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

const FE = MeanFieldHomogenization.FiniteElements

mand(T) = MeanFieldHomogenization.Core.mandel66_minor(MeanFieldHomogenization.Core._C_array(T))
kelvin_J(M) = (M[1, 1] + 2M[1, 2]) / 3        # 𝕁-eigenvalue of a TI/iso tensor
kelvin_K(M) = M[4, 4]                          # 𝕂-eigenvalue
iso(C) = MeanFieldHomogenization.Core.isotropify(C)

# ── The recycled-concrete aggregate of Adessina et al. (2017) ────────────────
#  A stiff old natural aggregate (the core) inside a shell of adhered old
#  mortar, the whole embedded in fresh cement paste.  Moduli in GPa.
E_agg, ν_agg = 70.0, 0.2
E_paste, ν_paste = 20.0, 0.2
const W_CORE = 0.5                             # old aggregate / whole inclusion

ce(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))
const C_AGG = ce(E_agg, ν_agg)
const C_PASTE = ce(E_paste, ν_paste)

report = IOBuffer()

const SECTIONS = isempty(ARGS) ? nothing : Set(ARGS)
want(name) = SECTIONS === nothing || name in SECTIONS

# ─── Figure 1 — the meridian mesh ────────────────────────────────────────────

"""
The two-dimensional mesh actually solved: the meridian half-plane `ρ ≥ 0`, with
the core, the shell and the surrounding matrix in three colors. A node of this
mesh stands for a whole circle of the three-dimensional body.
"""
function figure_mesh(incl; zoom = nothing, kw...)
    grid = FE._axi_setup(incl).grid
    plt = plot(;
        aspect_ratio = 1, xlabel = "ρ / a", ylabel = "z / a",
        legend = :topright, kw...
    )
    if zoom !== nothing
        plot!(plt; xlims = (-0.05, zoom), ylims = (-zoom, zoom))
    end
    for (setname, col, lab) in (
            (FE.AXI_SET_MATRIX, :grey85, "matrix"),
            (FE.AXI_SET_SHELL, :steelblue, "shell (old mortar)"),
            (FE.AXI_SET_CORE, :firebrick, "core (old aggregate)"),
        )
        xs, ys = Float64[], Float64[]
        for ci in Ferrite.getcellset(grid, setname)
            c = Ferrite.getcoordinates(grid, ci)
            append!(xs, [c[1][1], c[2][1], c[3][1], c[1][1], NaN])
            append!(ys, [c[1][2], c[2][2], c[3][2], c[1][2], NaN])
        end
        plot!(plt, xs, ys; lw = 0.35, lc = col, label = lab)
    end
    return plt
end

# ─── Figure 0 — the morphology, with its symbols on it ───────────────────────
#
#  A definition, not a result: the eccentricity α is a *normalized* offset, and
#  a symbol table alone leaves the reader to reconstruct what it is normalized
#  by.  The sketch says it in one look — α = 1 is where the core touches the
#  outer surface, which is why d = α(a − a_c) and not α·a.

"Circle of radius `r` centered at `(x0, z0)`, as a closed polyline."
function _circle(x0, z0, r; n = 400)
    θ = range(0, 2π; length = n)
    return x0 .+ r .* cos.(θ), z0 .+ r .* sin.(θ)
end

"""
Annotated meridian sketch of the morphology: the whole inclusion of radius `a`,
the core of radius `a_c = a·w^(1/3)` offset by `d = α(a − a_c)` along the
symmetry axis, and the surrounding ball of matrix of radius `R`.
"""
function figure_schematic(α; w = W_CORE, a = 1.0, kw...)
    ac = a * cbrt(w)
    d = α * (a - ac)
    plt = plot(;
        aspect_ratio = 1, legend = false, framestyle = :none,
        xlims = (-1.9a, 1.75a), ylims = (-1.5a, 1.6a), kw...,
    )
    ## The ball of matrix, cropped by the frame — only its curvature is needed.
    plot!(plt, _circle(0, 0, 1.45a)...; lc = :grey75, lw = 1.0, ls = :dot)
    annotate!(plt, 1.06a, -1.16a, text("matrix  ℂ₀", 8, :grey45, :left))
    ## Shell, then core on top of it.
    plot!(plt, _circle(0, 0, a)...; lc = :steelblue, lw = 1.6, fillalpha = 0.16,
        fillcolor = :steelblue, seriestype = :shape)
    plot!(plt, _circle(0, d, ac)...; lc = :firebrick, lw = 1.6, fillalpha = 0.20,
        fillcolor = :firebrick, seriestype = :shape)
    ## The symmetry axis.
    plot!(plt, [0, 0], [-1.36a, 1.40a]; lc = :black, lw = 0.7, ls = :dashdot)
    annotate!(plt, 0.06a, 1.44a, text("z", 9, :black, :left))
    ## a: the outer radius, measured to the right at mid-height of the shell so
    ## that it never runs across the core radius.
    z_a = -0.55a
    x_a = sqrt(max(a^2 - z_a^2, 0.0))
    plot!(plt, [0, x_a], [z_a, z_a]; lc = :black, lw = 1.0)
    plot!(plt, [x_a, x_a], [z_a - 0.05a, z_a + 0.05a]; lc = :black, lw = 1.0)
    annotate!(plt, 0.5x_a, z_a - 0.13a, text("a", 9, :black))
    ## a_c: the core radius, on the up-right diagonal from the core center.
    xe = ac / sqrt(2)
    ze = d + ac / sqrt(2)
    plot!(plt, [0, xe], [d, ze]; lc = :firebrick, lw = 1.0)
    annotate!(plt, xe + 0.06a, ze + 0.02a, text("a_c", 8, :firebrick, :left))
    ## The core center, and d: the offset, bracketed to the left of everything.
    scatter!(plt, [0], [d]; mc = :firebrick, ms = 3, msw = 0)
    if α > 0
        xb = -1.20a
        plot!(plt, [xb, xb], [0, d]; lc = :darkgreen, lw = 1.4)
        for zz in (0, d)
            plot!(plt, [xb - 0.06a, xb + 0.06a], [zz, zz]; lc = :darkgreen, lw = 1.2)
            plot!(plt, [xb, 0], [zz, zz]; lc = :darkgreen, lw = 0.5, ls = :dot)
        end
        annotate!(plt, xb - 0.10a, d / 2, text("d", 9, :darkgreen, :right))
    end
    annotate!(plt, -0.16a, d - 0.42ac, text("core ℂ₁", 8, :firebrick, :right))
    annotate!(plt, 0.30a, -0.90a, text("shell ℂ₂", 8, :steelblue, :left))
    return plt
end

want("schematic") && let
    ## Three eccentricities: concentric, intermediate, and the tangency limit
    ## α = 1 — which is what fixes the normalization of α.
    fig = plot(
        figure_schematic(0.0; title = "α = 0 — concentric"),
        figure_schematic(0.6; title = "α = 0.6"),
        figure_schematic(1.0; title = "α = 1 — tangency, by definition"),
        layout = (1, 3), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1150, 420), titlefontsize = 10,
        plot_title = "a_c = a·w^(1/3),   d = α (a − a_c)" *
            "      (here w = $W_CORE, so a_c/a = $(round(cbrt(W_CORE), digits = 3)))",
        plot_titlefontsize = 11,
    )
    savefig(fig, joinpath(OUT, "axi_schematic.png"))

    println(report, "## Morphology\n")
    ## `@printf` needs a literal format string — no concatenation.
    @printf(report, "`w = %.2f` gives `a_c/a = %.4f`; ", W_CORE, cbrt(W_CORE))
    @printf(
        report,
        "the offset is `d = α(a − a_c)`, so `α = 1` is tangency (`d = %.4f a`).\n\n",
        1 - cbrt(W_CORE)
    )
end

want("mesh") && let
    full = figure_mesh(
        FEExcenteredSphere(
            1.0, (C_AGG, C_PASTE); core_fraction = W_CORE, eccentricity = 0.8,
            nradial = 14, radius_ratio = 4.0
        ); title = "the whole cell, α = 0.8", legend = :topright
    )
    zoom0 = figure_mesh(
        FEExcenteredSphere(
            1.0, (C_AGG, C_PASTE); core_fraction = W_CORE, eccentricity = 0.0,
            nradial = 22, radius_ratio = 4.0
        ); zoom = 1.25, title = "α = 0 (concentric)", legend = false
    )
    zoom8 = figure_mesh(
        FEExcenteredSphere(
            1.0, (C_AGG, C_PASTE); core_fraction = W_CORE, eccentricity = 0.8,
            nradial = 22, radius_ratio = 4.0
        ); zoom = 1.25, title = "α = 0.8", legend = false
    )
    savefig(
        plot(full, zoom0, zoom8; layout = (1, 3), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1300, 460)),
        joinpath(OUT, "axi_mesh.png")
    )
    rep = fe_axi_mesh_report(
        FEExcenteredSphere(
            1.0, (C_AGG, C_PASTE); core_fraction = W_CORE, eccentricity = 0.4,
            nradial = 24, radius_ratio = 4.0
        )
    )
    println(report, "## Mesh (α = 0.4, `nradial = 24`, `radius_ratio = 4`)\n")
    println(report, "| Quantity | Measured | Exact | Δ |")
    println(report, "| :--- | ---: | ---: | ---: |")
    for (nm, got, ex) in (
            ("core volume", rep.volume_core, rep.volume_core_exact),
            ("shell volume", rep.volume_shell, rep.volume_shell_exact),
            ("cell volume", rep.volume_cell, rep.volume_cell_exact),
        )
        @printf(report, "| %s | %.4f | %.4f | %+.2f %% |\n", nm, got, ex, 100(got - ex) / ex)
    end
    @printf(
        report, "\n%d triangles, %d nodes (%d core, %d shell, %d matrix).\n\n",
        rep.ncells, rep.nnodes, rep.ncells_core, rep.ncells_shell, rep.ncells_matrix
    )
end

# ─── Figure 2 — the boundary correction against the cell size ────────────────
#
#  The paper's headline figure: the localization tensor as a function of the
#  ratio (mesh outer radius)/(inclusion radius), with and without the
#  first-order corrected boundary condition.  Here the reference is not a
#  converged computation but the *exact* Hervé-Zaoui solution of the concentric
#  two-layer sphere, which the concentric limit of this morphology must
#  reproduce.

want("convergence") && let
    C₀ = C_PASTE
    C_SHELL = ce(2.0, 0.2)                     # a badly degraded adhered mortar
    sph = LayeredSphere((cbrt(W_CORE), 1.0), (C_AGG, C_SHELL))
    Aex = mand(strain_strain_loc(sph, C₀, C₀))
    Jex, Kex = kelvin_J(Aex), kelvin_K(Aex)

    # The concentric limit, tabulated once at the shipped defaults.
    incl0 = FEExcenteredSphere(
        1.0, (C_AGG, C_SHELL); core_fraction = W_CORE, eccentricity = 0.0,
        nradial = 24, radius_ratio = 4.0
    )
    A0, B0 = fe_axi_localization(incl0, C₀)
    Bex = mand(stress_strain_loc(sph, C₀, C₀))
    println(report, "## Concentric limit against Hervé-Zaoui\n")
    println(report, "| Tensor | Part | Finite elements | Hervé-Zaoui | Δ |")
    println(report, "| :--- | :---: | ---: | ---: | ---: |")
    for (nm, num, ex) in (("A_εε", mand(A0), Aex), ("A_σε", mand(B0), Bex))
        for (pt, f) in (("J", kelvin_J), ("K", kelvin_K))
            @printf(
                report, "| %s | %s | %.6f | %.6f | %+.3f %% |\n",
                nm, pt, f(num), f(ex), 100(f(num) - f(ex)) / f(ex)
            )
        end
    end
    M0 = mand(A0)
    @printf(
        report,
        "\nIsotropy of the assembled tensor: |A11-A33|/A11 = %.1e, |A44-A66|/A44 = %.1e\n\n",
        abs(M0[1, 1] - M0[3, 3]) / M0[1, 1], abs(M0[4, 4] - M0[6, 6]) / M0[4, 4]
    )

    ratios = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0]
    eJu, eKu, eJc, eKc = Float64[], Float64[], Float64[], Float64[]
    for R in ratios
        incl = FEExcenteredSphere(
            1.0, (C_AGG, C_SHELL); core_fraction = W_CORE, eccentricity = 0.0,
            nradial = 24, radius_ratio = R
        )
        r = fe_axi_breakdown(incl, C₀)
        U, C = mand(r.A_uncorrected), mand(r.A)
        push!(eJu, 100abs(kelvin_J(U) - Jex) / Jex)
        push!(eKu, 100abs(kelvin_K(U) - Kex) / Kex)
        push!(eJc, 100abs(kelvin_J(C) - Jex) / Jex)
        push!(eKc, 100abs(kelvin_K(C) - Kex) / Kex)
    end

    plt = plot(;
        xlabel = "R / a  (cell radius / inclusion radius)",
        ylabel = "relative error on 𝔸  (%)",
        yscale = :log10, xscale = :log10, legend = :bottomleft, size = (700, 470),
        xticks = (ratios, string.(ratios)),
        title = "Concentric limit vs Hervé-Zaoui  (α = 0, w = $W_CORE, " *
            "E₁/E_m = $(round(E_agg / E_paste, digits = 1)), E₂/E_m = 0.1, ν = 0.2)",
        titlefontsize = 10,
    )
    plot!(plt, ratios, max.(eJu, 1.0e-4); m = :circle, lc = :grey40, mc = :grey40,
        ls = :dash, label = "𝕁 part, u = E·x")
    plot!(plt, ratios, max.(eKu, 1.0e-4); m = :square, lc = :grey65, mc = :grey65,
        ls = :dash, label = "𝕂 part, u = E·x")
    plot!(plt, ratios, max.(eJc, 1.0e-4); m = :circle, lc = :firebrick, mc = :firebrick,
        label = "𝕁 part, corrected")
    plot!(plt, ratios, max.(eKc, 1.0e-4); m = :square, lc = :steelblue, mc = :steelblue,
        label = "𝕂 part, corrected")
    plot!(
        plt, ratios, eJu[1] .* (ratios[1] ./ ratios) .^ 3;
        lc = :black, ls = :dot, label = "∝ (a/R)³"
    )
    savefig(plt, joinpath(OUT, "axi_convergence.png"))

    println(report, "## Boundary correction against the cell size\n")
    println(report, "Relative error on the localization tensor 𝔸 (%), concentric case,")
    println(report, "`nradial = 24`, against the exact Hervé-Zaoui solution.\n")
    println(report, "| R/a | 𝕁, `u = E·x` | 𝕁, corrected | 𝕂, `u = E·x` | 𝕂, corrected |")
    println(report, "| ---: | ---: | ---: | ---: | ---: |")
    for (i, R) in pairs(ratios)
        @printf(
            report, "| %.1f | %.3f | %.3f | %.3f | %.3f |\n",
            R, eJu[i], eJc[i], eKu[i], eKc[i]
        )
    end
    println(report)
end

# ─── Figure 3 — effective stiffness against the mortar contrast ──────────────
#
#  The engineering figure of the paper: a recycled-concrete aggregate whose
#  adhered mortar softens (E₂ ↓) at fixed old-aggregate fraction, for several
#  eccentricities of the core.  Mori-Tanaka, aggregate volume fraction 0.4.

want("contrast") && let
    C₀ = C_PASTE
    fracs = 0.4
    contrasts = 10 .^ range(-1.0, 0.3, length = 9)         # E_mortar / E_paste
    αs = (0.0, 0.4, 0.8)
    cols = (:black, :steelblue, :firebrick)

    plt = plot(;
        xlabel = "E₂ / E_m   (adhered mortar / fresh paste)",
        ylabel = "E_eff / E_m", xscale = :log10, legend = :topleft, size = (700, 470),
        title = "Recycled aggregate (Mori-Tanaka, f = $fracs, w = $W_CORE, " *
            "E₁/E_m = $(round(E_agg / E_paste, digits = 1)), ν = 0.2)",
        titlefontsize = 10,
    )
    table = Dict{Float64, Vector{Float64}}()
    for (α, col) in zip(αs, cols)
        vals = Float64[]
        for c in contrasts
            C2 = ce(c * E_paste, ν_paste)
            incl = FEExcenteredSphere(
                1.0, (C_AGG, C2); core_fraction = W_CORE, eccentricity = α,
                nradial = 20, radius_ratio = 4.0
            )
            rve = RVE(:m)
            add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀))
            add_phase!(rve, :rca, incl, Dict(:C => C₀); fraction = fracs)
            k, μ = k_mu(iso(homogenize(rve, MoriTanaka(), :C)))
            push!(vals, 9k * μ / (3k + μ) / E_paste)
        end
        table[α] = vals
        plot!(plt, contrasts, vals; m = :circle, lc = col, mc = col, label = "α = $α")
    end
    # The concentric curve must sit on the exact Hervé-Zaoui composite sphere.
    exact = Float64[]
    for c in contrasts
        C2 = ce(c * E_paste, ν_paste)
        sph = LayeredSphere((cbrt(W_CORE), 1.0), (C_AGG, C2))
        rve = RVE(:m)
        add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀))
        add_phase!(rve, :rca, sph, Dict(:C => C₀); fraction = fracs)
        k, μ = k_mu(iso(homogenize(rve, MoriTanaka(), :C)))
        push!(exact, 9k * μ / (3k + μ) / E_paste)
    end
    plot!(
        plt, contrasts, exact; ls = :dash, lc = :grey50, lw = 2,
        label = "α = 0, Hervé-Zaoui"
    )

    # The three curves nearly superpose, which is itself the message; the
    # eccentricity effect only becomes readable relative to the concentric one.
    rel = plot(;
        xlabel = "E₂ / E_m   (adhered mortar / fresh paste)",
        ylabel = "E_eff(α) / E_eff(0) − 1   (%)", xscale = :log10,
        legend = :topright, titlefontsize = 10,
        title = "Effect of the eccentricity (same parameters)",
    )
    for (α, col) in zip(αs[2:end], cols[2:end])
        plot!(
            rel, contrasts, 100 .* (table[α] ./ table[0.0] .- 1);
            m = :circle, lc = col, mc = col, label = "α = $α"
        )
    end
    plot!(
        rel, contrasts, 100 .* (table[0.0] ./ exact .- 1);
        ls = :dash, lc = :grey50, label = "α = 0 vs Hervé-Zaoui (numerical error)"
    )
    hline!(rel, [0.0]; lc = :black, lw = 0.6, label = "")
    savefig(
        plot(plt, rel; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1150, 470)),
        joinpath(OUT, "axi_contrast.png")
    )

    println(report, "## Effective Young modulus of a recycled-aggregate mortar\n")
    println(report, "`E_eff/E_m`, Mori-Tanaka, `f = $fracs`, `w = $W_CORE`,")
    println(report, "`E_agg/E_m = $(E_agg / E_paste)`, all Poisson ratios 0.2.\n")
    println(report, "| E₂/E_m | α = 0 | α = 0.4 | α = 0.8 | α = 0 exact |")
    println(report, "| ---: | ---: | ---: | ---: | ---: |")
    for (i, c) in pairs(contrasts)
        @printf(
            report, "| %.3f | %.4f | %.4f | %.4f | %.4f |\n",
            c, table[0.0][i], table[0.4][i], table[0.8][i], exact[i]
        )
    end
    println(
        report, "\nLargest deviation of the concentric finite-element curve from " *
            @sprintf("the exact one: %.2f %%.\n", 100 * maximum(abs.(table[0.0] .- exact) ./ exact))
    )
end

# ─── Figure 4 — equivalent conductivity against the matrix contrast ──────────
#
#  Transport counterpart, and the shape of the test that `echoes` runs on the
#  same morphology: the equivalent conductivity of the composite particle,
#  transverse and axial, against the conductivity of the reference medium.

want("conductivity") && let
    k1, k2 = TensISO{3}(1.0), TensISO{3}(0.1)      # core, shell
    ks = 10 .^ range(-2.0, 2.0, length = 13)
    plt = plot(;
        xlabel = "k₀ / k₂", ylabel = "k_eq / k₂", xscale = :log10,
        legend = :topleft, size = (700, 470),
        title = "Equivalent conductivity of the particle  " *
            "(w = $W_CORE, k₁/k₂ = 10)",
        titlefontsize = 10,
    )
    rows = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()
    for (α, col) in zip((0.0, 0.4, 0.8), (:black, :steelblue, :firebrick))
        tr, ax = Float64[], Float64[]
        incl = FEExcenteredSphere(
            1.0, (k1, k2); core_fraction = W_CORE, eccentricity = α,
            nradial = 24, radius_ratio = 4.0
        )
        for k in ks
            A, B = fe_axi_localization(incl, TensISO{3}(k * 0.1))
            Am, Bm = TensND.components_canon(A), TensND.components_canon(B)
            push!(tr, Bm[1, 1] / Am[1, 1] / 0.1)
            push!(ax, Bm[3, 3] / Am[3, 3] / 0.1)
        end
        rows[α] = (tr, ax)
        plot!(plt, ks, tr; m = :circle, lc = col, mc = col, label = "transverse, α = $α")
        plot!(
            plt, ks, ax; m = :xcross, lc = col, mc = col, ls = :dash,
            label = "axial, α = $α"
        )
    end
    savefig(plt, joinpath(OUT, "axi_conductivity.png"))

    println(report, "## Equivalent conductivity of the composite particle\n")
    println(report, "`k_eq/k₂` from `⟨q⟩ = k_eq ⟨∇T⟩` over the inclusion,")
    println(report, "`k₁/k₂ = 10`, `w = $W_CORE`.\n")
    println(report, "| k₀/k₂ | α=0 | α=0.4 tr. | α=0.4 ax. | α=0.8 tr. | α=0.8 ax. |")
    println(report, "| ---: | ---: | ---: | ---: | ---: | ---: |")
    for (i, k) in pairs(ks)
        @printf(
            report, "| %.3g | %.4f | %.4f | %.4f | %.4f | %.4f |\n",
            k, rows[0.0][1][i], rows[0.4][1][i], rows[0.4][2][i],
            rows[0.8][1][i], rows[0.8][2][i]
        )
    end
    println(report)
end

# ─── Cost ────────────────────────────────────────────────────────────────────

want("cost") && let
    incl = FEExcenteredSphere(
        1.0, (C_AGG, C_PASTE); core_fraction = W_CORE, eccentricity = 0.4,
        nradial = 24, radius_ratio = 4.0
    )
    fe_axi_mesh_report(incl)                     # build the mesh out of the timing
    t = @elapsed fe_axi_localization(incl, C_PASTE)
    t2 = @elapsed fe_axi_localization(incl, C_PASTE)
    rep = fe_axi_mesh_report(incl)
    println(report, "## Cost\n")
    println(
        report,
        @sprintf("`nradial = 24`, `radius_ratio = 4`, P2: %d triangles; ", rep.ncells) *
            @sprintf("first evaluation %.2f s (three assemblies, eight solves), ", t) *
            @sprintf("cached evaluation %.1e s.", t2)
    )
end

SECTIONS === nothing ?
    write(joinpath(OUT, "axi_results.md"), String(take!(report))) :
    @info "partial run ($(join(ARGS, ", "))) — axi_results.md left untouched"
println("wrote ", OUT)
