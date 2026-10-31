# [Aging basic creep of concrete: a CEM I cement, with and without silica fume](@id app-aging-basic-creep)

!!! info "Before this page"
    [A hydrating blended cement paste, coupled to its chemistry](@ref app-blended-hydration)
    for the four-scale elastic model and the hydration it is fed with, and
    [Aging creep of solidifying cementitious materials](@ref app-aging-creep) for a phase
    that solidifies. The nesting of aging scales is in the
    [manual](@ref man-viscoelasticity).

[Lavergne2020](@citet) turn the elastic model of [Lavergne2018](@citet) into an estimate
of the **basic creep** of concrete: the creep of a sealed specimen, with no drying. Two
things are added to the elastic chain:

- the hydrates **creep**, logarithmically, as nanoindentation suggests;
- the microstructure **changes under load**: hydrates keep forming, grains keep
  dissolving, and a phase formed after the loading carries none of it.

This page follows the same idea with the hydration computed by ChemistryLab, and holds
it against the creep tests of [LeRoy1995](@citet) on two concretes of one cement, loaded
from 0.83 to 28 days.

## The creep of the hydrates

A hydrate creeps with a logarithmic compliance whose Poisson ratio does not change in
time:

```math
J(t,t') = \left[\frac{1}{E} + \frac{1}{E_D}\,\ln\!\left(1+\frac{t-t'}{\tau}\right)\right] H(t-t'),
```

with ``\tau`` half a day. For C-S-H, the ratio ``E_D/E`` follows a quadratic fit in the
Ca/Si ratio, divided by the relative humidity [Lavergne2020; Eq. (11)](@cite): the lower
the humidity, the stiffer the creep. AFm takes the ratio of a C-S-H of Ca/Si 1.7; every
other hydrate is elastic.

```@example basiccreep
using MeanFieldHomogenization
using TensND
using JSON
using Printf
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

const SCRIPTS = joinpath(pkgdir(MeanFieldHomogenization), "scripts")
include(joinpath(SCRIPTS, "common", "paste_micromechanics.jl"))
include(joinpath(SCRIPTS, "common", "aging_creep_micromechanics.jl"))

E, ν = PASTE_E_NU["C-S-H"]
dt = exp10.(range(-2, 3; length = 120))
plot(; xscale = :log10, xlabel = "t - t' [days]", ylabel = "E J(t, t')", legend = :topleft,
    size = (700, 380), title = "Creep of C-S-H, Ca/Si = $(CSH_CA_SI)")
for rh in (1.0, 0.75, 0.5)
    r = csh_log_ratio(CSH_CA_SI, rh)
    plot!(dt, 1 .+ log1p.(dt ./ CREEP_TAU) ./ r; lw = 2, label = @sprintf("RH = %.2f, E_D/E = %.2f", rh, r))
end
current()
```

## Phases that form and dissolve

In aging viscoelasticity a phase keeps one volume fraction over the whole history. What
changes is **when it carries stress**, and two operations of the package say it:

- [`precipitated`](@ref)`(law, t_s)`: a phase formed stress-free at ``t_s``, insensitive to
  any strain applied before;
- [`dissolved`](@ref)`(law, t_d)`: a phase that stops carrying stress at ``t_d``.

The hydration is cut into steps. Each step adds phases to the scales of the elastic
model:

| scale | scheme | phases |
|:--|:--|:--|
| hydrates | self-consistent, spheres | the hydrates formed during one step: one law per step |
| hydrate foam | self-consistent, needles of aspect ratio 7, random orientations | the foam born at step ``i``: the hydrates precipitated at each later step, and the pores left |
| paste | Mori–Tanaka | the foam born first as matrix; a grain consumed at step ``i`` is `dissolved` at ``t_i`` while the foam born at ``t_i`` is `precipitated` in its place |
| concrete | Mori–Tanaka | the paste and elastic aggregates |

Each scale is a [`homogenize_alv`](@ref) on one time grid, and hands its result, a
[`VolterraOperator`](@ref), to the next as a phase property; the paste enters the concrete
as a [`Homogenized`](@ref) cell. Every operator here is isotropic: two ``n\times n``
matrices instead of one ``6n\times 6n``. The elastic phases (the unhydrated grains, the
aggregates) are given by their stiffness tensor itself, whose kernel is
``\mathbb{C}\,H(t - t')``. The model is
`scripts/common/aging_creep_micromechanics.jl`; a phase consumed and replaced, for
instance, is one line of it, the sum of two operators:

```julia
R = dissolved(grain, times[i]) + precipitated(foams[i], times[i])
```

A small check of these two operations: a soft phase in which a stiff one precipitates at
one day, loaded at three days, has the compliance of the stiff phase.

```@example basiccreep
times = [0.5, 1.0, 2.997, 3.0, 30.0]
C = iso_stiffness_E_nu(30.0, 0.2)
soft = VolterraOperator(TensISO{3}(3.0e-6, 2.0e-6), times)
stiffening = soft + precipitated(VolterraOperator(C, times), 1.0)
J = uniaxial_compliance(stiffening, times)
@printf "J(30, 3) = %.5f per GPa, 1/E = %.5f\n" J[end, 4] 1 / 30
```

On the grid, a load "held from ``t_k``" rises over ``[t_{k-1}, t_k]``. The grid therefore
carries an instant just before each loading age (`loading_grid`), so that this rise is a
step.

## Le Roy's concretes

The cement is the Saint-Vigor Portland cement of [LeRoy1995](@citet), whose phase
composition is given by [Lavergne2020](@citet) (Table 4), with the effective
water-to-cement ratios they derive. Two concretes were loaded at several ages:

```@example basiccreep
paper = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "lavergne2020.json"))["quantities"]
for p in ("leroy_ordinary", "leroy_hpc")
    @printf "%-15s w/c = %.3f  a/c = %.2f  cement = %3.0f kg/m3  silica fume = %2.0f %%\n" p paper[p * "_w_c"]["value"] paper[p * "_a_c"]["value"] paper[p * "_cement"]["value"] get(paper, p * "_silica_fume", Dict("value" => 0))["value"]
end
```

The whole chain — hydration, four aging scales, compliance for every loading age — runs in
`scripts/47_aging_basic_creep_concrete.jl`, about eight minutes; this page reads its
results. The measurements are those of the NU database of [Hubler2015](@citet), entries
`c_123_01` to `c_123_10`.

```@example basiccreep
res = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "docs", "src", "assets", "applications", "aging_basic_creep", "results.json"))
tests = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "leroy1995.json"))["tables"]["basic_creep"]
panels = map(res["concretes"]) do conc
    p = plot(; xscale = :log10, xlabel = "t [days]", ylabel = "J(t, t') [1e-6/MPa]",
        ylims = (0, 130), legend = :topleft, title = conc["name"], titlefontsize = 9)
    for (k, cv) in enumerate(conc["curves"])
        test = only(filter(t -> t["nu_file"] in conc["files"] && t["loading_age_days"] == cv["age"], tests))
        rows = test["creep_compliance"]["rows"]
        plot!(p, Float64.(cv["t"]), Float64.(cv["J"]); lw = 2, color = k, label = "t' = $(cv["age"]) d")
        scatter!(p, cv["age"] .+ first.(rows), last.(rows); color = k, ms = 2.5, msw = 0, label = "")
    end
    p
end
plot(panels...; layout = (1, 2), size = (900, 420), left_margin = 3Plots.mm, bottom_margin = 4Plots.mm)
```

Lines: the model; points: the measurements.

## What it shows

- For the ordinary concrete loaded at **3 days or later**, the model follows the shape
  of the creep curves, the elastic part included, and overestimates the compliance by
  6 to 33 %, the most at 3 days.
- Loaded at **0.83 and 1.12 days**, it overestimates the compliance by far (the 0.83-day
  curve of the ordinary concrete is off the scale): the paste is then close to its
  percolation threshold, where the self-consistent foam is most
  sensitive. [Lavergne2020](@citet) observe the same before 3 days.
- For the **high-performance** concrete, the late creep is overestimated. One cause not
  modeled here: at so low a water-to-cement ratio the paste dries itself as it hydrates,
  which lowers its internal humidity and, through the ``1/\mathrm{RH}`` of the creep
  ratio, its creep; the humidity is taken as 1 throughout.

## What is not here

- The C-S-H keeps the Ca/Si of the hydration model's single C-S-H (1.667), even where
  the silica fume lowers it.
- The hydrates formed at a step take the composition of what increased during that step.
  Species that dissolve later (ettringite into monosulfate, portlandite consumed by the
  silica fume) are not taken out of older hydrates; the paper does this last in, first
  out.
- The needles of the foam are averaged over random orientations, where the elastic
  model of [Lavergne2018](@citet) uses 20 orientation bins: the aging self-consistent
  scheme needs an isotropic reference.
- The paper's proposed correction, rescaling the estimate by the elastic compliance
  measured at loading, and its drying-creep model are not applied.

## Reproducing this page

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia scripts/47_aging_basic_creep_concrete.jl
```
