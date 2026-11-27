```@meta
EditURL = "../../../../scripts/108_fluid_filled_pores.jl"
```

# [Fluid-filled pores: pressure polarization](@id app-fluid-filled-pores)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/108_fluid_filled_pores.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/108_fluid_filled_pores.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/108_fluid_filled_pores.jl`, runs it end to end.

!!! info "Before this page"
    [Levin's theorem](@ref th-levin), §5, which defines the pressure of a
    fluid sealed in a pore family and derives Skempton's coefficient of a
    sphere, and [Poromechanics](@ref manual-poromechanics), which gives the
    syntax of [`pore_pressure_tensor`](@ref).

A fluid trapped in a pore of a loaded solid takes a pressure that depends on
the shape of the pore and on its orientation with respect to the load: two
pores of the same rock, filled with the same fluid, hold different
pressures. [Shafiro and Kachanov](@cite shafiro1997) call this the pressure
polarization and give it in closed form for a sphere, a crack and a needle.
This page computes the pressure for spheroids of any aspect ratio, against
their formulas, then passes to a finite porosity, where sealed and connected
pores part.

````@example fluid_filled_pores
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf
using Plots
gr()
nothing #hide
````

## The pressure tensor of a pore

A pore family ``r`` filled with a fluid of bulk modulus ``k_f``, sealed in
each pore, is a phase of stiffness ``3k_f\,\mathbb J``. Its pressure under
the remote stress ``\boldsymbol\Sigma`` is
``p_r = -\boldsymbol B^{\mathrm{sk}}_r:\boldsymbol\Sigma``, with
``\boldsymbol B^{\mathrm{sk}}_r = \tfrac13\boldsymbol 1:\mathbb B_r`` read on
the stress concentration tensor of the phase
([Levin's theorem](@ref th-levin), §5). The article writes the same
pressure ``q = \boldsymbol Q:\boldsymbol\Sigma``, eq. (6): ``\boldsymbol Q = -\boldsymbol B^{\mathrm{sk}}``.
An isolated pore is the stress-driven dilute scheme, [`DiluteDual`](@ref),
whose concentration tensors do not depend on the volume fraction.

The material of the article is a granite filled with water, the fluid being
given by its compressibility ``\kappa = 1/k_f``:

````@example fluid_filled_pores
literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
const paper = literature("shafiro1997")
quantity(name) = paper["quantities"][name]["value"]
const E₀, ν₀, κ_w = quantity("E0"), quantity("nu0"), quantity("kappa_water")    ## MPa, 1, 1/MPa
nothing #hide
````

````@example fluid_filled_pores
Main.show_literature("shafiro1997") # hide
````

The pressure tensor of one pore, of shape `shape`, in a solid of Young
modulus `E` and Poisson ratio `ν`, filled with a fluid of compressibility
`κ`; and the compressibility ``C_p`` of the same pore dry, its volume strain
per unit confining pressure, read on its strain concentration tensor
([`gradient_flux_concentration`](@ref)):

````@example fluid_filled_pores
solid(E, ν) = TensISO{3}(E / (1 - 2ν), E / (1 + ν))                ## 3k 𝕁 + 2μ 𝕂
const 𝟏 = TensISO{3}(1.0)
function lone_pore(shape, κ; E = E₀, ν = ν₀)
    rve = RVE()
    add_phase!(rve, :solid, Ellipsoid(1.0), Dict(:C => solid(E, ν)); fraction = :rest)
    add_phase!(rve, :pore, shape, Dict(:C => TensISO{3}(3 / κ, 0.0)); fraction = 1.0e-6)
    return concentration_tensors(rve, DiluteDual())
end
pressure_tensor(shape, κ; kw...) = pore_pressure_tensor(lone_pore(shape, κ; kw...), :pore)
dry_compressibility(shape; kw...) = 𝟏 ⊡ gradient_flux_concentration(lone_pore(shape, Inf; kw...), :pore) ⊡ 𝟏
nothing #hide
````

`κ = Inf` is the dry pore, of stiffness zero. Under a confining pressure
``P``, ``\boldsymbol\Sigma = -P\,\boldsymbol 1``, the pressure is
``p = P\,\boldsymbol 1:\boldsymbol B^{\mathrm{sk}}``: the ratio
``p/P = \boldsymbol 1:\boldsymbol B^{\mathrm{sk}}`` is Skempton's coefficient
of the pore.

````@example fluid_filled_pores
skempton(shape, κ; kw...) = 𝟏 ⊡ pressure_tensor(shape, κ; kw...)
nothing #hide
````

## The sphere: Skempton's coefficient

The volume of a pore filled with fluid changes as the fluid it holds. The
change of volume of the pore is that of the dry pore under the remote
pressure, ``-C_pP``, plus that of the pore under the fluid pressure ``p``
in an unloaded solid, ``(C_p - C_0)\,p``, where ``C_0 = 3(1 - 2\nu_0)/E_0``
is the compressibility of the solid (the second term is the uniform state
``-p\,\boldsymbol 1`` plus the dry pore under the remote stress
``p\,\boldsymbol 1``). The fluid changes its volume by ``-\kappa\,p``, whence
eq. (14) of the article,

```math
\frac{p}{P} = \frac{C_p}{C_p + \kappa - C_0} = \frac1{1 + \delta},
\qquad \delta = \frac{\kappa - C_0}{C_p},
```

valid for any shape of pore, the shape entering through ``C_p`` alone. For
the sphere, ``C_p = 1/k_0 + 3/(4\mu_0)`` (``R = E_0C_p = 9(1 - \nu_0)/2``,
eq. (10)), and the coefficient is that of Eshelby's solution,
``k_f(3k_0 + 4\mu_0)/(k_0(3k_f + 4\mu_0))``
([Levin's theorem](@ref th-levin), §5). The three against the Poisson ratio
of the solid, for water and for fluids ten times more and ten times less
compressible:

````@example fluid_filled_pores
eq14(C_p, κ, E, ν) = C_p / (C_p + κ - 3(1 - 2ν) / E)
function eshelby(κ, E, ν)
    k, μ = E / (3(1 - 2ν)), E / (2(1 + ν))
    return (3k + 4μ) / (k * (3 + 4μ * κ))                        ## k_f(3k + 4μ)/(k(3k_f + 4μ)), k_f = 1/κ
end
sphere = Ellipsoid(1.0)
νs = range(0, 0.45; length = 46)
κs = (0.1κ_w, κ_w, 10κ_w)
B = [skempton(sphere, κ; ν) for ν in νs, κ in κs]
gap_14 = maximum(abs(B[i, j] - eq14(dry_compressibility(sphere; ν), κ, E₀, ν)) for (i, ν) in pairs(νs), (j, κ) in pairs(κs))
gap_eshelby = maximum(abs(B[i, j] - eshelby(κ, E₀, ν)) for (i, ν) in pairs(νs), (j, κ) in pairs(κs))
gap_R = maximum(abs(E₀ * dry_compressibility(sphere; ν) - 9(1 - ν) / 2) for ν in νs)
(eq_14 = gap_14, eshelby = gap_eshelby, R = gap_R)
````

````@example fluid_filled_pores
plot(
    νs, B; lw = 2, label = ["κ = κ_water / 10" "κ = κ_water" "κ = 10 κ_water"], legend = :topleft,
    xlabel = "Poisson ratio ν₀", ylabel = "p / P", title = "Spherical pore in granite (E₀ = 60 GPa)", titlefontsize = 10, size = (600, 340)
)
````

The stiffer the fluid with respect to the pore, the closer the pressure of
the fluid to the confining pressure. At a given ``E_0``, a larger ``\nu_0``
makes the dry pore stiffer (``R = 9(1 - \nu_0)/2`` decreases), and the pore
shields its fluid more: the coefficient decreases.

## Spheroids under a uniaxial compression

A spheroid of axis ``\underline{n}`` (here ``\underline{e}_3``) and aspect
ratio ``\zeta = a_3/a`` is loaded by the compression ``T`` along the unit
vector ``\underline{d}`` at the angle ``\varphi`` from its axis,
``\boldsymbol\Sigma = -T\,\underline{d}\otimes\underline{d}``. Its pressure is
``p/T = \boldsymbol B^{\mathrm{sk}}:(\underline{d}\otimes\underline{d})``.
For the aspect ratios of Fig. 4 of the article, oblate and prolate:

````@example fluid_filled_pores
uniaxial(B_sk, φ) = (d = [sin(φ), 0.0, cos(φ)]; B_sk ⊡ Tens(d * transpose(d)))
ζs = [row["zeta"] for row in paper["tables"]["fig4_aspect_ratios"]]
φs = range(0, π / 2; length = 91)
pressure_curves(ζ) = [uniaxial(pressure_tensor(Spheroid(ζ), κ_w), φ) for φ in φs]
nothing #hide
````

The article gives two limits in closed form. A narrow crack, ``\zeta \ll 1``,
eq. (11): ``R = 4(1 - \nu_0^2)/(\pi\zeta)``, ``\delta = \pi\zeta\,(\kappa E_0 - 3(1 - 2\nu_0))/(4(1 - \nu_0^2))``
and ``\boldsymbol Q = -\underline{n}\otimes\underline{n}/(1 + \delta)``, hence
``p/T = \cos^2\varphi/(1 + \delta)``. A needle, ``\zeta \gg 1``, eq. (13):
``R = 5 - 4\nu_0``, ``\delta = (\kappa E_0 - 3(1 - 2\nu_0))/(5 - 4\nu_0)`` and
``\boldsymbol Q = -\bigl((2 - \nu_0)\,\boldsymbol 1 - (1 + \nu_0)\,\underline{n}\otimes\underline{n}\bigr)/((1 + \delta)(5 - 4\nu_0))``,
hence ``p/T = \bigl(2 - \nu_0 - (1 + \nu_0)\cos^2\varphi\bigr)/((1 + \delta)(5 - 4\nu_0))``.

````@example fluid_filled_pores
crack_δ(ζ) = π * ζ * (κ_w * E₀ - 3(1 - 2ν₀)) / (4(1 - ν₀^2))
crack_limit(ζ, φ) = cos(φ)^2 / (1 + crack_δ(ζ))
const needle_δ = (κ_w * E₀ - 3(1 - 2ν₀)) / (5 - 4ν₀)
needle_limit(φ) = (2 - ν₀ - (1 + ν₀) * cos(φ)^2) / ((1 + needle_δ) * (5 - 4ν₀))
oblate = plot(;
    xlabel = "φ (degrees)", ylabel = "p / T", title = "oblate pores", titlefontsize = 9, xticks = 0:15:90, legend = :topright
)
prolate = plot(;
    xlabel = "φ (degrees)", ylabel = "p / T", title = "prolate pores", titlefontsize = 9, xticks = 0:15:90, legend = :topleft
)
for ζ in ζs
    plot!(ζ < 1 ? oblate : prolate, rad2deg.(φs), pressure_curves(ζ); lw = 2, label = "ζ = $ζ")
end
plot!(oblate, rad2deg.(φs), crack_limit.(minimum(ζs), φs); color = :black, ls = :dash, label = "eq. (11), ζ = $(minimum(ζs))")
plot!(prolate, rad2deg.(φs), needle_limit.(φs); color = :black, ls = :dash, label = "eq. (13), needle")
plot(oblate, prolate; layout = (1, 2), size = (780, 320), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm)
````

A flat pore answers the stress normal to its plane: its pressure is largest
when compressed along its axis and vanishes when compressed along its
plane. A needle answers the stress across its axis, and much less: being
stiffer, it is squeezed less. The pressure tensors of a very flat and of a
very long spheroid against the two limits, by their components along the
axis (``nn``) and across it (``tt``), and the compressibility of the dry pore
as ``R = E_0C_p``:

````@example fluid_filled_pores
const e₃ = Tens([0.0, 0.0, 1.0])
const e₁ = Tens([1.0, 0.0, 0.0])
component(B_sk, a) = a ⋅ B_sk ⋅ a
println("pore            R computed   R closed form    Q_nn computed   Q_nn eq.   Q_tt computed   Q_tt eq.")
for (name, ζ, R, Q_nn, Q_tt) in (
        ("crack, ζ = 1e-4", 1.0e-4, 4(1 - ν₀^2) / (π * 1.0e-4), -1 / (1 + crack_δ(1.0e-4)), 0.0),
        ("needle, ζ = 1e4", 1.0e4, 5 - 4ν₀, -(1 - 2ν₀) / ((1 + needle_δ) * (5 - 4ν₀)), -(2 - ν₀) / ((1 + needle_δ) * (5 - 4ν₀))),
    )
    Q = -pressure_tensor(Spheroid(ζ), κ_w)
    @printf "%-16s %11.4f  %11.4f      %10.6f   %10.6f   %10.6f   %10.6f\n" name E₀ * dry_compressibility(Spheroid(ζ)) R component(Q, e₃) Q_nn component(Q, e₁) Q_tt
end
````

The closed forms keep the leading term in ``\zeta`` (crack) or ``1/\zeta``
(needle); the computed values differ from them by the next order.

## Pressure amplification under a confining pressure

Eq. (14) can exceed one: when the fluid is less compressible than the
solid, ``\kappa < C_0``, ``\delta`` is negative and the fluid carries more
than the confining pressure. Skempton's coefficient against the aspect
ratio, for an incompressible fluid (approached here by
``\kappa E_0 = 10^{-9}``) and three Poisson ratios, and for the granite and
its water:

````@example fluid_filled_pores
ζ_range = 10 .^ range(-2, log10(5); length = 121)
const κ_rigid = 1.0e-9 / E₀
amplification = plot(;
    xlabel = "aspect ratio ζ", ylabel = "p / P", xscale = :log10, legend = :topleft,
    title = "Skempton's coefficient of a spheroidal pore", titlefontsize = 10, size = (600, 360)
)
for ν in (0.0, 0.25, 0.4)
    plot!(amplification, ζ_range, [skempton(Spheroid(ζ), κ_rigid; ν) for ζ in ζ_range]; lw = 2, label = "κ = 0, ν₀ = $ν")
end
plot!(amplification, ζ_range, [skempton(Spheroid(ζ), κ_w) for ζ in ζ_range]; lw = 2, ls = :dash, color = :black, label = "granite and water")
amplification
````

The largest amplification is that of the sphere, for an incompressible
fluid in a solid of Poisson ratio zero. In that case eq. (14) gives 3 for
the sphere (``\delta = -2/3``), ``5/2`` for the needle (``\delta = -3/5``),
and ``1/(1 - 3\pi\zeta/4) \simeq 1 + 2.36\,\zeta`` for a narrow crack: the
values quoted by the article. Computed:

````@example fluid_filled_pores
needle_value = skempton(Spheroid(1.0e4), κ_rigid; ν = 0.0)
crack_slope = (skempton(Spheroid(1.0e-4), κ_rigid; ν = 0.0) - 1) / 1.0e-4
(sphere = skempton(sphere, κ_rigid; ν = 0.0), needle = needle_value, crack_slope = crack_slope, three_π_over_4 = 3π / 4)
````

The same coefficient for the granite and its water, for a sphere and for
the flattest pore of the figure:

````@example fluid_filled_pores
(sphere = skempton(sphere, κ_w), flat = skempton(Spheroid(minimum(ζ_range)), κ_w))
````

The amplification fades as the matrix becomes incompressible: at
``\nu_0 = 1/2``, ``C_0 = 0`` and eq. (14) gives ``p = P`` whatever the shape.
In the granite, the water is more compressible than the rock
(``\kappa > C_0``), and the coefficient stays below one: close to one for
flat pores, so compliant that the water carries nearly all the load, and
small for spheres, stiff enough to shield it.

## A finite porosity: sealed or connected pores

At a finite porosity the pores interact, and [`MoriTanaka`](@ref) replaces
the dilute scheme. When the pores are connected, one pressure reigns in all
of them and the undrained stiffness follows from the drained one by
Gassmann's relation, in its anisotropic form of Brown and Korringa
([`poroelastic_parameters`](@ref) with `k_f`, then
[`undrained_stiffness`](@ref)). When they are sealed, each family keeps its
pressure. A cell of granite with pores of given shapes, either filled with
water or dry:

````@example fluid_filled_pores
const C₀ = solid(E₀, ν₀)
function granite(families; κ = κ_w)
    rve = RVE()
    add_phase!(rve, :solid, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    for (name, shape, f) in families
        add_phase!(rve, name, shape, Dict(:C => TensISO{3}(3 / κ, 0.0)); fraction = f)
    end
    return rve
end
porosity(families) = sum(last, families)
function sealed_and_connected(families)
    C_sealed = homogenize(granite(families), MoriTanaka())
    C_dry = homogenize(granite(families; κ = Inf), MoriTanaka())
    p = poroelastic_parameters(C_dry, C₀, porosity(families); k_f = 1 / κ_w)
    C_connected = undrained_stiffness(C_dry, p.B, p.modulus)
    return bulk_shear(best_fit_iso(C_sealed)), bulk_shear(best_fit_iso(C_connected))
end
nothing #hide
````

With spheres only, at the porosity 0.1, every pore holds the same pressure,
and the two answers coincide:

````@example fluid_filled_pores
spheres = [(:spheres, Ellipsoid(1.0), 0.1)]
sealed_and_connected(spheres)
````

A mixture of the same spheres and of cracks, randomly oriented, of aspect
ratio ``10^{-3}`` and density ``\varepsilon = 0.1``. Each orientation is a
family of its own, with its own pressure: the six axes of the regular
icosahedron, whose averages of the products of up to five components of the
normal are those of the uniform distribution, as on the page
[Squirt flow in a cracked rock](@ref app-squirt-dispersion). A crack of
radius ``a`` occupies ``\tfrac{4\pi}{3}\zeta a^3``:

````@example fluid_filled_pores
const golden = (1 + sqrt(5)) / 2
const icosahedron = normalize.([[0, 1, golden], [0, 1, -golden], [1, golden, 0], [1, -golden, 0], [golden, 0, 1], [-golden, 0, 1]])
axis_angles(n) = (acos(n[3]), atan(n[2], n[1]), 0.0)
ζ_c, ε_c = 1.0e-3, 0.1
cracks = [(Symbol(:cracks, i), Spheroid(ζ_c; euler_angles = axis_angles(n)), 4π / 3 * ε_c * ζ_c / 6) for (i, n) in pairs(icosahedron)]
mixture = vcat(spheres, cracks)
(sealed, connected) = sealed_and_connected(mixture)
println("               k (MPa)     μ (MPa)")
@printf "sealed      %10.1f  %10.1f\n" sealed[1] sealed[2]
@printf "connected   %10.1f  %10.1f\n" connected[1] connected[2]
````

Sealed, the cracks keep their water, which stiffens them; connected, the
water of a compressed crack escapes to the spheres and to the cracks that the
load opens. The pressures of the sealed families under a confining
pressure, by Mori-Tanaka, show the imbalance that drives the flow: the
compliant cracks squeeze their water to about the confining pressure (a
little more, the solid around them carrying more than the remote stress),
the stiff spheres shield theirs.

````@example fluid_filled_pores
ct = concentration_tensors(granite(mixture), MoriTanaka())
(spheres = 𝟏 ⊡ pore_pressure_tensor(ct, :spheres), cracks = 𝟏 ⊡ pore_pressure_tensor(ct, :cracks1))
````

The two stiffnesses are the high- and the low-frequency limits of
[`squirt_flow`](@ref): at ``\omega\tau = \infty`` and ``\omega\tau = 0`` it
gives the sealed and the connected cells, and in between the dispersion of
the page [Squirt flow in a cracked rock](@ref app-squirt-dispersion).

````@example fluid_filled_pores
families = first.(mixture)
dry = granite(mixture; κ = Inf)
limits = [bulk_shear(best_fit_iso(squirt_flow(dry, MoriTanaka(), ωτ; fluid = families, k_f = 1 / κ_w).stiffness)) for ωτ in (Inf, 0.0)]
(sealed = maximum(abs.(limits[1] .- sealed) ./ sealed), connected = maximum(abs.(limits[2] .- connected) ./ connected))
````

## Where to go next

- [Squirt flow in a cracked rock](@ref app-squirt-dispersion): the flow
  between the sealed families, the frequency at which it happens, and what
  it does to the velocities of the waves.
- [Levin's theorem](@ref th-levin), §5: the pressure tensor, Gassmann's
  relation and the exchange between pore families.
- [Seismic signatures](@ref man-seismic): the density and the waves of a
  cell, its fluid included.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

