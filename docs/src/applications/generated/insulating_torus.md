```@meta
EditURL = "../../../../scripts/78_insulating_torus.jl"
```

# [Ring-shaped pores in a conducting matrix](@id app-insulating-torus)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/78_insulating_torus.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/78_insulating_torus.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/78_insulating_torus.jl`, runs it end to end.

!!! info "Before this page"
    [Toroidal inclusions](@ref th-toroidal-inclusions), which derives the
    tensor used here, and [Toroidal inclusions](@ref man-toroidal-inclusions),
    which gives the syntax.

A toroidal pore conducts nothing and has no Hill tensor. Its concentration
tensor is a series of toroidal functions, obtained by
[Radi and Sevostianov](@cite radiPRSA2016) and computed in the package by
[`InsulatingTorus`](@ref). This page evaluates it against the ratio of the
two radii, compares it with the values plotted in that article and with
spheroids of the same volume, and gives the conductivity of a matrix that
contains such pores in random orientations.

````@example insulating_torus
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf
using Plots
gr()
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl")) # hide
nothing # hide
````

## The pore and its tensor

The torus is swept by a circle of radius ``r`` whose center describes a
circle of radius ``R > r`` about the axis ``\underline{n}``. Its volume is
``|\Omega| = 2\pi^2 R\,r^2`` and its shape depends on ``\lambda = r/R`` only:
a thin ring for ``\lambda \to 0``, a torus whose hole closes for
``\lambda \to 1``.

````@example insulating_torus
plotly_scene(shape_traces(InsulatingTorus(1.0, 0.5)); uid = "app-torus", height = 400, title = "InsulatingTorus(1.0, 0.5): λ = r/R = 0.5") # hide
````

In a matrix of conductivity ``k_0`` under a remote temperature gradient
``\underline{G}``, the mean gradient over the pore, defined from the
temperature on its surface, is ``\boldsymbol{A}\cdot\underline{G}`` with

```math
\boldsymbol{A} = A_t\,\bigl(\boldsymbol{1} - \underline{n}\otimes\underline{n}\bigr)
  + A_n\,\underline{n}\otimes\underline{n} .
```

``\boldsymbol{A}`` is the gradient concentration tensor of the pore, and its
resistivity contribution tensor per unit volume of the torus is
``\boldsymbol{R} = \boldsymbol{A}/k_0``
([Concentration and contribution tensors](@ref th-concentration)). The two
components are read on the tensor the package returns:

````@example insulating_torus
const K₀ = TensISO{3}(1.0)                     ## conductivity of the matrix, k₀ = 1
const void = TensISO{3}(0.0)                   ## the pore conducts nothing
components(pore) = canonical_components(gradient_gradient_concentration(pore, void, K₀))[[1, 9]]
torus(λ) = InsulatingTorus(1.0, λ)             ## R = 1, r = λ

Aₜ, Aₙ = components(torus(0.5))
````

The whole range of shapes:

````@example insulating_torus
λs = range(0.01, 0.99; length = 99)
A = components.(torus.(λs))
plot(
    λs, [first.(A) last.(A)]; label = ["Aₜ, gradient across the axis" "Aₙ, gradient along the axis"],
    lw = 2, xlabel = "λ = r / R", ylabel = "component of A", ylims = (1.2, 2.2), legend = :right,
    title = "Insulating torus: gradient concentration tensor", titlefontsize = 10, size = (640, 360)
)
````

The thin ring has the limits ``A_t = 3/2`` and ``A_n = 2``
([Toroidal inclusions](@ref th-toroidal-inclusions), section 8), and neither
component moves by more than 10 % over the whole range: per unit volume, the
effect of a toroidal pore hardly depends on its thickness.

## The flux over the surface of the torus

[Radi and Sevostianov](@cite radiPRSA2016) plot, in their Fig. 7(a), the
vector

```math
\underline{\phi} = \oint_{\partial\Omega} T\,\underline{\nu}\,\mathrm{d}S
  = |\Omega|\;\boldsymbol{A}\cdot\underline{G}
```

for a remote flux of density ``q_0``, that is a gradient of norm
``q_0/k_0`` opposite to it, along the axis (``\phi_z``) and across it
(``\phi_y``). With ``|\Omega| = 2\pi^2\lambda^2 R^3``, the quantities they
plot are

```math
-\frac{k_0\,\phi_z}{q_0\,R^3} = 2\pi^2\lambda^2\,A_n ,
\qquad
-\frac{k_0\,\phi_y}{q_0\,R^3} = 2\pi^2\lambda^2\,A_t .
```

````@example insulating_torus
literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
const radi = literature("radiPRSA2016")
const read_on_figure = radi["tables"]["surface_flux"]
const reading = radi["quantities"]["reading_uncertainty"]["value"]
nothing #hide

Main.show_literature("radiPRSA2016") # hide
````

The points are the ordinates read on the published figure, with the
uncertainty of the reading, the thickness of the plotted lines; the curves
are computed here.

````@example insulating_torus
flux(λ) = 2π^2 * λ^2 .* components(torus(λ))           ## (across, along)
φ = flux.(λs)
plt = plot(
    λs, [last.(φ) first.(φ)]; label = ["along the axis, 2π²λ²Aₙ" "across the axis, 2π²λ²Aₜ"], lw = 2,
    xlabel = "λ = r / R", ylabel = "−k₀ ϕ / (q₀ R³)", legend = :topleft,
    title = "Flux over the surface of the torus", titlefontsize = 10, size = (640, 380)
)
scatter!(
    plt, [row["lambda"] for row in read_on_figure], [row["minus_k_phi_z"] for row in read_on_figure];
    yerror = reading, label = "Radi & Sevostianov (2016), Fig. 7(a)", color = 1, markersize = 3
)
scatter!(
    plt, [row["lambda"] for row in read_on_figure], [row["minus_k_phi_y"] for row in read_on_figure];
    yerror = reading, label = "", color = 2, markersize = 3
)
plt
````

The deviations, in the unit of the ordinate:

````@example insulating_torus
for row in read_on_figure
    ϕy, ϕz = flux(row["lambda"])
    @printf "λ = %.2f   along: read %5.2f, computed %5.2f (%+.2f)   across: read %5.2f, computed %5.2f (%+.2f)\n" row["lambda"] row["minus_k_phi_z"] ϕz row["minus_k_phi_z"] - ϕz row["minus_k_phi_y"] ϕy row["minus_k_phi_y"] - ϕy
end
````

The two sets agree to the thickness of the plotted lines, except for the last
point across the axis, read 4 % above the computed value, where the curve
reaches the frame of the figure. The amplitudes of the series and the
integrals over the torus written in the article are recovered to rounding
([Toroidal inclusions](@ref th-toroidal-inclusions), section 7). Figures
7(b), 8 and 9 of the article are not overlaid: their curves vanish with
``\lambda``, whereas the tensor per unit volume of the torus tends to the
limit of the thin ring.

## The torus beside spheroids of the same volume

The article compares the torus with two spheroids of the same axis and the
same volume. The oblate one has the outer radius ``R + r`` of the torus for
equatorial radius, hence the aspect ratio
``\gamma_o = 3\pi\lambda^2/\bigl(2\,(1+\lambda)^3\bigr)``. The prolate one has
the radius ``r`` of the tube for equatorial radius, hence
``\gamma_p = 3\pi/(2\lambda)``. Their tensors are exact, through the Hill
tensor of the [`Ellipsoid`](@ref):

````@example insulating_torus
oblate(λ) = Ellipsoid(1.0, 1.0, 3π * λ^2 / (2 * (1 + λ)^3))
prolate(λ) = Ellipsoid(1.0, 1.0, 3π / (2λ))
sphere = first(components(Ellipsoid(1.0)))            ## 3/2

Ao, Ap = components.(oblate.(λs)), components.(prolate.(λs))
across = plot(
    λs, [first.(A) first.(Ao) first.(Ap)]; label = "", lw = 2, ls = [:solid :dash :dot],
    xlabel = "λ = r / R", ylabel = "Aₜ", ylims = (0.9, 2.2), title = "across the axis", titlefontsize = 9
)
along = plot(
    λs, [last.(A) last.(Ao) last.(Ap)]; label = ["torus" "oblate spheroid" "prolate spheroid"], lw = 2,
    ls = [:solid :dash :dot], xlabel = "λ = r / R", ylabel = "Aₙ", ylims = (0.9, 4.0), legend = :topright,
    title = "along the axis", titlefontsize = 9
)
hline!(across, [sphere]; color = :gray, lw = 1, label = "")
hline!(along, [sphere]; color = :gray, lw = 1, label = "sphere")
plot(across, along; layout = (1, 2), size = (760, 340), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
````

Neither spheroid has the two components of the torus. The oblate one comes
close to it when the hole closes and tends to a disk-shaped crack, whose
component along the axis is unbounded, when the ring is thin. The prolate
one tends to a needle along the axis, with the components 2 across it and 1
along it, where the thin ring has ``3/2`` and 2.

The mean of the three principal components is another matter. It is the only
quantity that enters the conductivity of a matrix with randomly oriented
pores, and that of the needle, ``(2 + 2 + 1)/3``, is that of the thin ring,
``(3/2 + 3/2 + 2)/3``: a ring is a wire, and a wire in random orientations
does not remember that it was bent.

````@example insulating_torus
mean_component(a) = (2 * a[1] + a[2]) / 3
plot(
    λs, [mean_component.(A) mean_component.(Ao) mean_component.(Ap)];
    label = ["torus" "oblate spheroid" "prolate spheroid"], lw = 2, ls = [:solid :dash :dot],
    xlabel = "λ = r / R", ylabel = "(2Aₜ + Aₙ) / 3", ylims = (1.4, 2.4), legend = :topright,
    title = "Mean component, the one seen by randomly oriented pores", titlefontsize = 10, size = (640, 360)
)
hline!([sphere]; color = :gray, lw = 1, label = "sphere")
````

The largest relative deviation of the prolate spheroid from the torus, on
the mean component and over the range of the plot:

````@example insulating_torus
maximum(abs.(mean_component.(Ap) ./ mean_component.(A) .- 1))
````

For randomly oriented toroidal pores, the prolate spheroid of the same volume
and of the same tube radius is thus a close substitute, as the article
concludes; for aligned ones it is not, and the series is needed.

## Conductivity of a matrix with randomly oriented toroidal pores

The pores occupy the volume fraction ``c``. `symmetrize = IsoSymmetrize()`
averages the tensor of the phase over the orientations
([Choosing an inclusion](@ref man-choosing-inclusion)); the effective
conductivity is then isotropic.

````@example insulating_torus
function porous(λ, c)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:K => K₀); fraction = :rest)
    add_phase!(rve, :pores, torus(λ), Dict(:K => void); fraction = c, symmetrize = IsoSymmetrize())
    return rve
end
conductivity(λ, c, scheme) = canonical_components(homogenize(porous(λ, c), scheme, :K))[1, 1]
nothing #hide
````

With ``\bar A = (2A_t + A_n)/3``, four of the estimates are closed forms in
``c``, which the schemes return:

````@example insulating_torus
Ā = mean_component(components(torus(0.5)))
c = 0.2
[
    conductivity(0.5, c, Dilute()) 1 - c * Ā                              ## no interaction, on the conductivity
    conductivity(0.5, c, DiluteDual()) 1 / (1 + c * Ā)                    ## no interaction, on the resistivity
    conductivity(0.5, c, MoriTanaka()) (1 - c) / (1 - c + c * Ā)
    conductivity(0.5, c, DifferentialScheme(; nsteps = 400)) (1 - c)^Ā
]
````

The self-consistent estimate has no such form: it places the pores and the
matrix, the latter as spherical grains, in the effective medium itself, and
vanishes at a finite fraction of pores.

````@example insulating_torus
schemes = [
    "no interaction (resistivity)" => DiluteDual(),
    "Mori-Tanaka" => MoriTanaka(),
    "differential" => DifferentialScheme(; nsteps = 400),
    "self-consistent" => SelfConsistent(),
]
cs = range(0.0, 0.55; length = 23)
plt = plot(;
    xlabel = "volume fraction of pores c", ylabel = "k / k₀", ylims = (0, 1), legend = :bottomleft,
    title = "Randomly oriented toroidal pores, λ = 0.5", titlefontsize = 10, size = (640, 380)
)
for (name, scheme) in schemes
    plot!(plt, cs, [conductivity(0.5, c, scheme) for c in cs]; label = name, lw = 2)
end
plt
````

The shape of the torus matters little: at ``c = 0.3`` the Mori-Tanaka
estimate for three ratios, and for spherical pores,

````@example insulating_torus
for λ in (0.25, 0.5, 0.75)
    @printf "λ = %.2f   mean component %.4f   k/k₀ = %.4f\n" λ mean_component(components(torus(λ))) conductivity(λ, 0.3, MoriTanaka())
end
@printf "sphere     mean component %.4f   k/k₀ = %.4f\n" sphere (1 - 0.3) / (1 - 0.3 + 0.3 * sphere)
````

A toroidal pore lowers the conductivity a little more than a spherical one
of the same volume, by about the same amount whatever its thickness. What
sets it apart is its anisotropy when the pores are aligned: with all the
axes along ``\underline{e}_3`` the conductivity is transversely isotropic,

````@example insulating_torus
aligned = RVE()
add_phase!(aligned, :matrix, Ellipsoid(1.0), Dict(:K => K₀); fraction = :rest)
add_phase!(aligned, :pores, torus(0.5), Dict(:K => void); fraction = 0.3)
diag(canonical_components(homogenize(aligned, MoriTanaka(), :K)))
````

lower along the axis of the rings than across it.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

