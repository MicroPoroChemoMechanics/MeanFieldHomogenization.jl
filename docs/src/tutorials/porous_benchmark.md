# [Porous benchmark: all schemes](@id tut-porous-benchmark)

A solid matrix with spherical pores, porosity sweeping the *entire* range
``\varphi \in [0, 1]``, run through every scheme `MeanFieldHomogenization` implements —
the porous benchmark of the Echoes book [echoes](@cite).

## The benchmark problem

```@example tutporousbench
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

const k_s, μ_s = 72.0, 32.0        # solid moduli
const k_p, μ_p = 1.0e-6, 1.0e-6    # pore moduli (numerical regularization)

const C_s = iso_stiffness(k_s, μ_s)
const C_p = iso_stiffness(k_p, μ_p)
nothing # hide
```

The microstructure the schemes are asked about, at a moderate porosity — drag to
rotate. The second half of the page repeats everything with the pores flattened
to ``\omega = 0.2``, which is the same picture with squashed pores and a very
different answer:

```@setup tutporousbenchviz
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

```@example tutporousbenchviz
plotly_scene(rve_traces(; n = 70, semi_axes = (0.055, 0.055, 0.055),
        color = "#78909c", opacity = 0.95);
    uid = "pb-rve-spheres", height = 450,
    title = "Spherical pores, isotropic distribution")
```

```@example tutporousbenchviz
plotly_scene(rve_traces(; n = 70, semi_axes = (0.075, 0.075, 0.015), seed = 7,
        color = "#78909c", opacity = 0.95);
    uid = "pb-rve-oblate", height = 450,
    title = "Oblate pores, ω = 0.2, random orientations")
```

## Extracting effective moduli

The homogenized stiffness is expected to be isotropic here (spherical
solid and pore geometry), so it can be read back directly with
`k_mu` — as in the [first tutorial](first_estimate.md). Some
schemes can return a result with tiny numerical anisotropy (e.g. from
an iterative solve that has not fully converged); dispatching to
`best_fit_iso` first — the best isotropic projection of the
tensor — keeps the extraction robust without changing the answer on
already-isotropic input:

```@example tutporousbench
extract_kμ(C::TensND.TensISO{4, 3}) = k_mu(C)
extract_kμ(C::TensND.AbstractTens) = k_mu(best_fit_iso(C))
nothing # hide
```

## Every scheme at once

```@example tutporousbench
function build_rve(φ; ω_s = 1.0, ω_p = 1.0, sym_s = nothing, sym_p = nothing)
    r = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(r, :SOLID, Spheroid(ω_s), Dict(:C => C_s); fraction = :rest, symmetrize = sym_s)
    add_phase!(r, :PORE, Spheroid(ω_p), Dict(:C => C_p); fraction = φ, symmetrize = sym_p)
    return r
end

const SCHEMES = [
    (Voigt(), "Voigt", :red, :dash),
    (Reuss(), "Reuss", :blue, :dash),
    (MoriTanaka(), "Mori-Tanaka", :black, :solid),
    (SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true), "Self-Consistent", :red, :solid),
    (AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true), "Asym. SC", :purple, :solid),
    (DifferentialScheme(; nsteps = 100), "Differential", :gold, :solid),
    (Dilute(), "Dilute", :green, :dash),
    (DiluteDual(), "DiluteDual", :green, :dot),
    (Maxwell(), "Maxwell", :blue, :solid),
    (PonteCastanedaWillis(), "PCW", :green, :solid),
]

function sweep!(p_k, p_μ, scheme, label, color, ls, φs; build_kw = (;))
    ks, μs = Float64[], Float64[]
    for φ in φs
        try
            C = homogenize(build_rve(φ; build_kw...), scheme, :C)
            k, μ = extract_kμ(C)
            push!(ks, max(k, 0.0))
            push!(μs, max(μ, 0.0))
        catch
            push!(ks, NaN)
            push!(μs, NaN)
        end
    end
    plot!(p_k, φs, ks; lw = 2, color = color, linestyle = ls, label = label)
    plot!(p_μ, φs, μs; lw = 2, color = color, linestyle = ls)
end

φs = collect(range(0.0, 1.0; length = 51))

p_k = plot(; xlabel = "φ (porosity)", ylabel = "k_hom", xlims = (0, 1), ylims = (0, k_s + 5),
           legend = :topright, title = "Effective bulk modulus")
p_μ = plot(; xlabel = "φ (porosity)", ylabel = "μ_hom", xlims = (0, 1), ylims = (0, μ_s + 5),
           legend = false, title = "Effective shear modulus")

for (scheme, label, color, ls) in SCHEMES
    sweep!(p_k, p_μ, scheme, label, color, ls, φs)
end
plot(p_k, p_μ; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1400, 600), plot_title = "Porous benchmark (spheres)")
```

All schemes start at the solid's moduli and part ways in between:

| Scheme | ``\varphi \to 1`` | Percolation |
| :--- | :--- | :--- |
| `Voigt` / `Reuss` | ``\ne 0`` / ``0`` | bracket the whole family |
| `Mori-Tanaka`, `Maxwell`, `PCW` | ``0`` | none — pores stay isolated inclusions |
| `Self-Consistent`, `Asym. SC` | ``0`` | ``\varphi \approx 0.5``, then collapse |
| `Differential` | ``0`` | gradual |

`select_best = true` (see the [previous tutorial](porous_materials.md)) is what
keeps the SC/ASC curves smooth through the crossover rather than jumping
between branches under Picard noise.

## Non-spherical pores

Real pore populations are rarely spherical. Replacing both phases with
**oblate spheroids** (aspect ratio ``\omega = 0.2``, flatter than a
sphere) and declaring [`IsoSymmetrize`](@ref)`()` on each phase tells
the homogenization kernel to average the localization tensor over a
**uniform spatial distribution of orientations** — so, even though each
individual pore is anisotropic, the macroscopic effective tensor comes
out isotropic:

```@example tutporousbench
const ω_oblate = 0.2

p_k2 = plot(; xlabel = "φ (porosity)", ylabel = "k_hom", xlims = (0, 1), ylims = (0, k_s + 5),
            legend = :topright, title = "Oblate pores (ω=$(ω_oblate)), iso-symmetrized")
p_μ2 = plot(; xlabel = "φ (porosity)", ylabel = "μ_hom", xlims = (0, 1), ylims = (0, μ_s + 5),
            legend = false, title = "Effective shear modulus")

for (scheme, label, color, ls) in SCHEMES
    sweep!(
        p_k2, p_μ2, scheme, label, color, ls, φs;
        build_kw = (; ω_s = ω_oblate, ω_p = ω_oblate, sym_s = IsoSymmetrize(), sym_p = IsoSymmetrize()),
    )
end
plot(p_k2, p_μ2; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1400, 600), plot_title = "Porous benchmark (oblate + iso-symmetrize)")
```

## Numerical summary

```@example tutporousbench
for (scheme, label, _, _) in SCHEMES
    row = String[label]
    for φ in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
        try
            k, μ = extract_kμ(homogenize(build_rve(φ), scheme, :C))
            push!(row, "$(round(k, digits=2))/$(round(μ, digits=2))")
        catch
            push!(row, "NaN/NaN")
        end
    end
    println(rpad(row[1], 16), join(rpad.(row[2:end], 14)))
end
```

Every column reads `k_hom/μ_hom` at a fixed porosity — a compact way
to compare all ten schemes at a glance. The `try`/`catch` guard around
each evaluation matters in practice: a scheme that fails to converge at
one particular porosity (rare, but possible near percolation) reports
`NaN` there instead of aborting the whole sweep.
