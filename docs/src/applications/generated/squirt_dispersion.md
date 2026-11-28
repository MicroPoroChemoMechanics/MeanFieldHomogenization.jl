```@meta
EditURL = "../../../../scripts/106_squirt_dispersion.jl"
```

# [Squirt flow in a cracked rock: dispersion and attenuation](@id app-squirt-dispersion)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/106_squirt_dispersion.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/106_squirt_dispersion.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/106_squirt_dispersion.jl`, runs it end to end.

!!! info "Before this page"
    [Levin's theorem](@ref th-levin), §5, which derives the stiffness of a
    solid whose cracks hold a fluid, sealed, connected or exchanging it;
    [Poromechanics](@ref manual-poromechanics) and
    [Seismic signatures](@ref man-seismic), which give the syntax; and the
    [Plane waves](@extref TensND man-plane-waves) of TensND.

A wave that crosses a saturated cracked rock compresses some cracks more
than others. At ultrasonic frequencies the fluid has no time to move: each
crack keeps its own pressure. At seismic frequencies it flows from crack to
crack until the pressure is uniform. The two stiffnesses differ, and the
velocities measured in the laboratory are not those of the field. This page
computes the two limits for the three crack arrangements of
[Guéguen and Sarout](@cite gueguen2011), against their closed forms, then the
velocity and the attenuation at every frequency in between, which the
exchange law of [Jakobsen and Chapman](@cite jakobsen2009) gives.

````@example squirt_dispersion
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf
using Plots
gr()
nothing #hide
````

## The rock and its cracks

The solid is isotropic, of Young modulus ``E_0`` and Poisson ratio
``\nu_0``; the fluid is water, of bulk modulus ``k_f``. The values are those
of all the figures of the article:

````@example squirt_dispersion
literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
const paper = literature("gueguen2011")
quantity(name) = paper["quantities"][name]["value"]
const E₀, ν₀, k_f = quantity("E0"), quantity("nu0"), quantity("Kf")     ## GPa
const ρ_s, ρ_f = quantity("rho_s") / 1000, quantity("rho_f") / 1000    ## g/cm³: velocities in km/s
nothing #hide
````

````@example squirt_dispersion
Main.show_literature("gueguen2011") # hide
````

The cracks are identical oblate spheroids of radius ``a`` and aspect ratio
``\xi`` (half-opening over radius). ``N`` cracks in the volume ``V`` have
the crack density ``\varepsilon = N a^3/V`` (``\rho`` in the article) and
occupy the volume fraction ``\varphi = \tfrac{4\pi}{3}\,\varepsilon\,\xi``.
The article treats them in the non-interacting approximation in compliance:
each crack adds to the compliance of the solid what it would add alone in
the solid under the remote stress. This is the stress-driven dilute scheme,
[`DiluteDual`](@ref).

The pressure of a fluid sealed in cracks of one normal is one number. Cracks
of different normals carry different pressures, and the flow between them is
what disperses the waves. Each orientation is therefore a phase of its own,
with its own pressure: a single phase averaged over the orientations would
hold one pressure only. A set of cracks is given by its normals
``\underline{n}_i``, each family with the density ``\varepsilon/N``; the
Euler angles of [`Spheroid`](@ref) are the inclination of the axis on
``\underline{e}_3`` and its azimuth.

````@example squirt_dispersion
const C₀ = TensISO{3}(E₀ / (1 - 2ν₀), E₀ / (1 + ν₀))              ## 3k₀ 𝕁 + 2μ₀ 𝕂
const k₀, μ₀ = bulk_shear(C₀)
axis_angles(n) = (acos(n[3]), atan(n[2], n[1]), 0.0)
function cracked_rock(normals, ε, ξ)
    rve = RVE()
    add_phase!(rve, :solid, Ellipsoid(1.0), Dict(:C => C₀, :ρ => ρ_s); fraction = :rest)
    families = [Symbol(:cracks, i) for i in eachindex(normals)]
    for (family, n) in zip(families, normals)
        crack = Spheroid(ξ; euler_angles = axis_angles(n))
        add_phase!(rve, family, crack, Dict(:C => 0 * C₀, :ρ => ρ_f); fraction = 4π / 3 * ε * ξ / length(normals))
    end
    return rve, families
end
nothing #hide
````

[`squirt_flow`](@ref) gives the complex stiffness of the saturated rock at
the dimensionless frequency ``\omega\tau``, ``\tau`` being the relaxation
time of the flow between cracks ([Levin's theorem](@ref th-levin), §5). At
``\omega\tau = \infty`` no fluid moves and the cracks are sealed: the
unrelaxed, high-frequency stiffness. At ``\omega\tau = 0`` the pressure is
uniform: the relaxed stiffness of Brown and Korringa, the low-frequency one.
The stiffness of the fluid phases in the cell is ignored: `k_f` replaces it.

````@example squirt_dispersion
saturated(rve, families, ωτ) = squirt_flow(rve, DiluteDual(), ωτ; fluid = families, k_f).stiffness
high(cell) = saturated(cell..., Inf)
low(cell) = saturated(cell..., 0.0)
nothing #hide
````

The two limits are real; their imaginary parts, zero, are dropped before the
velocities are computed by [`plane_waves`](@extref TensND TensND.plane_waves), with
the density of the rock, ``\rho = (1 - \varphi)\rho_s + \varphi\rho_f``,
from [`average_density`](@ref):

````@example squirt_dispersion
real_stiffness(C) = Tens(real.(canonical_components(C)))
velocities(C, ρ, n) = plane_waves(real_stiffness(C), ρ, n).velocity    ## qP, qS1, qS2
relative(A, B) = sqrt(sum(abs2, canonical_components(A - B)) / sum(abs2, canonical_components(B)))
const e₃ = [0.0, 0.0, 1.0]
nothing #hide
````

## Case 1: aligned cracks

All the cracks have the normal ``\underline{e}_3``. Under any load they are
compressed alike and hold the same pressure: no fluid has a reason to move,
and the sealed and the relaxed stiffnesses coincide. One family, at
``\varepsilon = 0.3`` and ``\xi = 10^{-3}``:

````@example squirt_dispersion
aligned = cracked_rock([e₃], 0.3, 1.0e-3)
relative(high(aligned), low(aligned))
````

The dry rock of this case has the anisotropy parameter of Thomsen
``\gamma = (C_{66} - C_{44})/(2C_{44})``, from the compliance of eq. (2) of
the article, ``\gamma = 8(1 - \nu_0)\,\varepsilon/(3(2 - \nu_0))`` for thin
cracks. Its ratio to the crack density, computed on the dry rock
(the cracks of stiffness zero), against the two fractions:

````@example squirt_dispersion
dry_aligned = homogenize(first(aligned), DiluteDual())
γ = thomsen_parameters(best_fit_ti(dry_aligned, e₃), average_density(first(aligned))).γ
(computed = γ / 0.3, from_eq_2 = 8(1 - ν₀) / (3(2 - ν₀)), printed_fraction = 8(1 - ν₀) / (3(1 - 2ν₀)))
````

Eq. (14) of the article prints the fraction ``8(1 - \nu_0)/(3(1 - 2\nu_0))``
and the approximate value ``1.1``. With ``\nu_0 = 0.27`` the fraction printed
gives 4.2, and ``8(1 - \nu_0)/(3(2 - \nu_0))``, which eq. (2) gives, 1.125:
the approximate value and the computation agree with the second form.

## Case 2: randomly oriented cracks

The normals are drawn uniformly on the sphere. A finite set of families
stands for them: the six axes of the regular icosahedron. Its twelve
vertices form a spherical design of degree 5: the averages over them of the
products of up to five components of ``\underline{n}`` are those of the
uniform distribution, and among them the moments of eq. (16) of the article,
``\langle n_1^2\rangle = 1/3``, ``\langle n_1^4\rangle = 1/5``,
``\langle n_1^2n_2^2\rangle = 1/15``.

````@example squirt_dispersion
const golden = (1 + sqrt(5)) / 2
const icosahedron = normalize.([[0, 1, golden], [0, 1, -golden], [1, golden, 0], [1, -golden, 0], [golden, 0, 1], [-golden, 0, 1]])
moment(f) = sum(f, icosahedron) / length(icosahedron)
(n₁² = moment(n -> n[1]^2), n₁⁴ = moment(n -> n[1]^4), n₁²n₂² = moment(n -> n[1]^2 * n[2]^2))
````

A crack family adds to the compliance a tensor of degree four in its normal,
so that six families give an isotropic rock. The distance of the two
stiffnesses to their isotropic projection ([`best_fit_iso`](@extref TensND TensND.best_fit_iso)):

````@example squirt_dispersion
random = cracked_rock(icosahedron, 0.3, 1.0e-3)
C_hf, C_lf = high(random), low(random)
(high = relative(C_hf, best_fit_iso(C_hf)), low = relative(C_lf, best_fit_iso(C_lf)))
````

### The two limits in closed form

The article writes the extra compliance of the sealed cracks with the crack
density tensors ``\boldsymbol\alpha = \varepsilon\langle\underline{n}\otimes\underline{n}\rangle``
and ``\mathbb{\beta} = \varepsilon\langle\underline{n}\otimes\underline{n}\otimes\underline{n}\otimes\underline{n}\rangle``
(eqs. (3), (4), (9)):

```math
\Delta S_{ijkl} = h\Bigl[\tfrac14\bigl(\delta_{ik}\alpha_{jl} + \delta_{il}\alpha_{jk} + \delta_{jk}\alpha_{il} + \delta_{jl}\alpha_{ik}\bigr) + \psi\,\beta_{ijkl}\Bigr],
\qquad h = \frac{32(1 - \nu_0^2)}{3(2 - \nu_0)E_0},
```

```math
\psi = \Bigl(1 - \frac{\nu_0}{2}\Bigr)\frac{\delta_f}{1 + \delta_f} - 1,
\qquad
\delta_f = \frac{\pi\,\xi\,E_0}{4(1 - \nu_0^2)}\Bigl(\frac1{k_f} - \frac1{k_0}\Bigr),
```

the dry cracks having ``\psi = -\nu_0/2``. The relaxed compliance differs
from it by eq. (10),
``S^{hf}_{ijkl} - S^{lf}_{ijkl} = h\,(1 - \nu_0/2)/(1 + \delta_f)\,(\alpha_{ij}\alpha_{kl}/\operatorname{tr}\boldsymbol\alpha - \beta_{ijkl})``.
With the moments of eq. (16), ``\boldsymbol\alpha = (\varepsilon/3)\,\boldsymbol 1``
and ``\mathbb{\beta} = (\varepsilon/15)(\boldsymbol 1\otimes\boldsymbol 1 + 2\,\mathbb I) = \varepsilon(\mathbb J/3 + 2\mathbb K/15)``.
The bracket of the first term is ``(\varepsilon/3)\,\mathbb I``, and with
``\mathbb S_0 = \mathbb J/(3k_0) + \mathbb K/(2\mu_0)``,

```math
\frac1{k^{hf}} = \frac1{k_0} + h\,\varepsilon\,(1 + \psi),
\qquad
\frac1{\mu^{hf}} = \frac1{\mu_0} + 2h\,\varepsilon\Bigl(\frac13 + \frac{2\psi}{15}\Bigr),
```

```math
k^{lf} = k^{hf},
\qquad
\frac1{\mu^{lf}} = \frac1{\mu^{hf}} + \frac{4h\,\varepsilon\,(1 - \nu_0/2)}{15\,(1 + \delta_f)} ,
```

since ``\boldsymbol\alpha\otimes\boldsymbol\alpha/\operatorname{tr}\boldsymbol\alpha - \mathbb\beta = -2\varepsilon\,\mathbb K/15``
has no spherical part. With ``\psi = -\nu_0/2`` the first line is eq. (5),
and ``\mu^{lf}`` is the shear modulus of the dry rock: a pressure uniform in
all the cracks does not resist a shear.

````@example squirt_dispersion
crack_δ(ξ) = π * ξ * E₀ * (1 / k_f - 1 / k₀) / (4(1 - ν₀^2))
function closed_form(ε, ξ)
    h = 32(1 - ν₀^2) / (3(2 - ν₀) * E₀)
    δ_f = crack_δ(ξ)
    ψ = (1 - ν₀ / 2) * δ_f / (1 + δ_f) - 1
    k_hf = inv(1 / k₀ + h * ε * (1 + ψ))
    μ_hf = inv(1 / μ₀ + 2h * ε * (1 / 3 + 2ψ / 15))
    μ_lf = inv(1 / μ_hf + 4h * ε * (1 - ν₀ / 2) / (15(1 + δ_f)))
    return (k_hf = k_hf, μ_hf = μ_hf, k_lf = k_hf, μ_lf = μ_lf)
end
moduli(C) = real.(bulk_shear(best_fit_iso(C)))
k_hf, μ_hf = moduli(C_hf)
k_lf, μ_lf = moduli(C_lf)
μ_dry = moduli(homogenize(first(random), DiluteDual()))[2]
ref = closed_form(0.3, 1.0e-3)
println("ε = 0.3, ξ = 1e-3     k^hf      k^lf      μ^hf      μ^lf      μ dry   (GPa)")
@printf "computed            %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n" k_hf k_lf μ_hf μ_lf μ_dry
@printf "closed form         %8.4f  %8.4f  %8.4f  %8.4f\n" ref.k_hf ref.k_lf ref.μ_hf ref.μ_lf
````

The bulk modulus does not disperse: a hydrostatic stress compresses all the
cracks alike, whatever their normals. The shear modulus does. The closed
forms are those of infinitely thin cracks; the spheroids of the cell differ
from them by terms of the order of ``\xi``.

### Dispersion of the velocities

The article measures the dispersion of a wave by eq. (13),
``\Delta V = (V^{hf} - V^{lf})/V^{solid}``, the velocity of the uncracked
solid in the denominator. The rock being isotropic, any direction will do:

````@example squirt_dispersion
const V_solid = velocities(C₀, ρ_s, e₃)
function dispersion(normals, ε, ξ, n = e₃)
    cell = cracked_rock(normals, ε, ξ)
    ρ = average_density(first(cell))
    return (velocities(high(cell), ρ, n) - velocities(low(cell), ρ, n)) ./ V_solid
end
ΔV = dispersion(icosahedron, 0.3, 1.0e-3)
(ΔV_P = ΔV[1], ΔV_S = ΔV[2], ratio = ΔV[1] / ΔV[2])
````

The article quotes an S-wave dispersion close to ``5\times10^{-2}`` at
``\xi = 10^{-3}`` and ``\varepsilon = 0.3``, and
``\Delta V_P \simeq 0.4\,\Delta V_S``: the two numbers above. The same from the
closed forms, with the velocities ``V_P = \sqrt{(k + 4\mu/3)/\rho_s}`` and
``V_S = \sqrt{\mu/\rho_s}`` of eq. (17):

````@example squirt_dispersion
wave_speeds(k, μ) = (sqrt((k + 4μ / 3) / ρ_s), sqrt(μ / ρ_s))
function closed_dispersion(ε, ξ)
    m = closed_form(ε, ξ)
    return (wave_speeds(m.k_hf, m.μ_hf) .- wave_speeds(m.k_lf, m.μ_lf)) ./ wave_speeds(k₀, μ₀)
end
closed_dispersion(0.3, 1.0e-3)
````

Against the crack density, for the two aspect ratios of the article, and
against the aspect ratio at ``\varepsilon = 0.3``; the closed forms dashed:

````@example squirt_dispersion
εs = range(0.01, quantity("crack_density_max"); length = 25)
ξs = 10 .^ range(log10(quantity("aspect_ratio_min")), log10(quantity("aspect_ratio_max")); length = 25)
panel(; kw...) = plot(; ylabel = "ΔV (%)", legend = :topleft, titlefontsize = 9, kw...)
by_density = map((quantity("aspect_ratio_a"), quantity("aspect_ratio_b"))) do ξ
    p = panel(; xlabel = "crack density ε", title = "random cracks, ξ = $ξ")
    computed = [dispersion(icosahedron, ε, ξ) for ε in εs]
    closed = [closed_dispersion(ε, ξ) for ε in εs]
    plot!(p, εs, 100 .* [getindex.(computed, 2) getindex.(computed, 1)]; lw = 2, color = [1 2], label = ["S" "P"])
    plot!(p, εs, 100 .* [last.(closed) first.(closed)]; lw = 1, ls = :dash, color = [1 2], label = ["closed forms" ""])
end
computed = [dispersion(icosahedron, 0.3, ξ) for ξ in ξs]
closed = [closed_dispersion(0.3, ξ) for ξ in ξs]
by_aspect = panel(; xlabel = "aspect ratio ξ", xscale = :log10, title = "random cracks, ε = 0.3", legend = :bottomleft)
plot!(by_aspect, ξs, 100 .* [getindex.(computed, 2) getindex.(computed, 1)]; lw = 2, color = [1 2], label = ["S" "P"])
plot!(by_aspect, ξs, 100 .* [last.(closed) first.(closed)]; lw = 1, ls = :dash, color = [1 2], label = ["closed forms" ""])
plot(by_density..., by_aspect; layout = (1, 3), size = (900, 300), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm)
````

The dispersion is set by ``\delta_f``, which grows as ``\xi``. When
``\delta_f \ll 1`` the fluid stiffens the sealed cracks almost as an
incompressible one would, and the dispersion reaches a plateau; as ``\xi``
grows, the sealed cracks soften toward the dry ones and the dispersion
falls. The values of ``\delta_f`` at the three aspect ratios of the article:

````@example squirt_dispersion
(crack_δ(1.0e-3), crack_δ(1.0e-2), crack_δ(1.0e-1))
````

The spheroids depart from the thin cracks near ``\xi = 10^{-1}``.

## Between the two limits

The article stops at the two limits. The exchange law of
[Jakobsen and Chapman](@cite jakobsen2009) gives the stiffness at every
frequency: complex, its imaginary part being the energy the flow dissipates.
For a time dependence ``e^{\mathrm{i}\omega t}``, a modulus
``M(\omega) = M' + \mathrm{i}M''`` gives a plane wave of wavenumber
``\omega\sqrt{\rho/M}``, hence the phase velocity and the inverse quality
factor

```math
V = \frac{1}{\operatorname{Re}\sqrt{\rho/M}},
\qquad
Q^{-1} = \frac{M''}{M'} ,
```

with ``M = \mu`` for the S waves and ``M = k + 4\mu/3`` for the P waves of
the isotropic rock. At ``\varepsilon = 0.3`` and ``\xi = 10^{-3}``:

````@example squirt_dispersion
ωτs = 10 .^ range(-4, 2; length = 121)
ρ_random = average_density(first(random))
spectrum = map(ωτs) do ωτ
    k, μ = bulk_shear(best_fit_iso(saturated(random..., ωτ)))
    (S = μ, P = k + 4μ / 3)
end
phase_velocity(M) = 1 / real(sqrt(ρ_random / M))
inverse_Q(M) = imag(M) / real(M)
V_S, V_P = phase_velocity.(getfield.(spectrum, :S)), phase_velocity.(getfield.(spectrum, :P))
Q_S, Q_P = inverse_Q.(getfield.(spectrum, :S)), inverse_Q.(getfield.(spectrum, :P))
nothing #hide
````

````@example squirt_dispersion
velocity_panel = plot(
    ωτs, [V_S ./ V_S[1] V_P ./ V_P[1]]; xscale = :log10, lw = 2, label = ["S" "P"], legend = :topleft,
    xlabel = "ωτ", ylabel = "V / V(ωτ → 0)", title = "phase velocity", titlefontsize = 9
)
attenuation_panel = plot(
    ωτs, 100 .* [Q_S Q_P]; xscale = :log10, lw = 2, label = ["S" "P"], legend = :topleft,
    xlabel = "ωτ", ylabel = "Q⁻¹ (%)", title = "attenuation", titlefontsize = 9
)
plot(velocity_panel, attenuation_panel; layout = (1, 2), size = (760, 300), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm)
````

The velocity climbs from the relaxed to the sealed value over two decades,
and the attenuation peaks in between, at ``\omega\tau`` of a few hundredths
rather than one. The rise of the velocities over the whole range is the
dispersion of the previous section, up to its denominator:

````@example squirt_dispersion
(S = V_S[end] / V_S[1] - 1, P = V_P[end] / V_P[1] - 1)
````

### A single relaxation

Where the peak sits follows from the non-interacting approximation, in
which each family sees the remote stress ``\boldsymbol\Sigma`` alone. The
volume strain of a crack of family ``r`` whose fluid is at the pressure
``p_r`` is that of the dry crack under ``\boldsymbol\Sigma``, plus that of
the crack under the pressure ``p_r`` in an unloaded solid. The latter is the
sum of the uniform stress ``-p_r\boldsymbol 1`` in the solid and in the
crack, and of the dry crack under the remote stress ``p_r\boldsymbol 1``.
With ``C_p`` the compressibility of the dry crack (its volume strain per unit
confining pressure) and ``C_0 = 1/k_0`` that of the solid,

```math
\operatorname{tr}\langle\boldsymbol\varepsilon\rangle_r = \boldsymbol c_r:\boldsymbol\Sigma + (C_p - C_0)\,p_r ,
```

where ``\boldsymbol c_r`` depends on the normal of the family and ``C_p`` on
its shape only. With ``\kappa = 1/k_f``, the exchange law of
[Levin's theorem](@ref th-levin), §5,
``\delta m_r/(\rho_f\varphi_r) = \operatorname{tr}\langle\boldsymbol\varepsilon\rangle_r + \kappa\,p_r``
and ``\mathrm{i}\omega\,\delta m_r = -\rho_f\varphi_r\,\kappa\,(p_r - \bar p)/\tau``, gives

```math
\mathrm{i}\omega\tau\,\bigl(\boldsymbol c_r:\boldsymbol\Sigma + D\,p_r\bigr) = -\kappa\,(p_r - \bar p),
\qquad D = C_p + \kappa - C_0 .
```

The families have one shape and one volume fraction, and the conservation
``\sum_r\delta m_r = 0`` makes ``\bar p`` the mean of the ``p_r``. The mean of
the equation over the families gives ``\bar p = -\bar{\boldsymbol c}:\boldsymbol\Sigma/D``
at every frequency: the common pressure of the relaxed state. Then

```math
p_r = \frac{\bar p + \mathrm{i}\Omega\,p_r^\infty}{1 + \mathrm{i}\Omega},
\qquad p_r^\infty = -\frac{\boldsymbol c_r:\boldsymbol\Sigma}{D},
\qquad \Omega = \frac{\omega\tau\,D}{\kappa},
```

``p_r^\infty`` being the pressure of the sealed crack. The macroscopic strain
is linear in ``\boldsymbol\Sigma`` and in the pressures, hence a single
relaxation:

```math
\mathbb S(\omega) = \mathbb S^{lf} + \bigl(\mathbb S^{hf} - \mathbb S^{lf}\bigr)\,\frac{\mathrm{i}\Omega}{1 + \mathrm{i}\Omega},
\qquad
\Omega = 1 \iff \omega\tau = \frac{\kappa}{C_p + \kappa - C_0} .
```

The shear compliance ``1/\mu`` follows the same law, so that
``Q_S^{-1} = \Omega\,(1/\mu^{lf} - 1/\mu^{hf})/(1/\mu^{lf} + \Omega^2/\mu^{hf})``,
largest at ``\Omega = \sqrt{\mu^{hf}/\mu^{lf}}``, where it equals
``(\mu^{hf} - \mu^{lf})/(2\sqrt{\mu^{hf}\mu^{lf}})``. A thin crack is far more
compliant than the fluid it holds, ``C_p \gg \kappa``: its pressure moves
with little fluid, and the relaxation takes place well below
``\omega\tau = 1``. In terms of the parameter of the previous section,
``\delta_f = (\kappa - C_0)/C_p``, the relaxation is at
``\omega\tau = (\delta_f + C_0/C_p)/(1 + \delta_f)``.

``C_p`` is read on the strain concentration tensor of a dry family,
[`gradient_flux_concentration`](@ref), which gives ``\langle\boldsymbol\varepsilon\rangle_r``
per unit remote stress:

````@example squirt_dispersion
const 𝟏 = TensISO{3}(1.0)
dry_cell = first(cracked_rock([e₃], 0.3, 1.0e-3))
C_p = 𝟏 ⊡ gradient_flux_concentration(concentration_tensors(dry_cell, DiluteDual()), :cracks1) ⊡ 𝟏
κ, C_s = 1 / k_f, 1 / k₀
ωτ_relax = κ / (C_p + κ - C_s)
(C_p = C_p, ωτ_relax = ωτ_relax, thin_cracks = crack_δ(1.0e-3) / (1 + crack_δ(1.0e-3)))
````

The stiffness of `squirt_flow` against the single relaxation, over the whole
range of frequencies, and the peak of ``Q_S^{-1}``:

````@example squirt_dispersion
debye(ωτ) = (Ω = ωτ / ωτ_relax; inv(C_lf) + (inv(C_hf) - inv(C_lf)) * (im * Ω / (1 + im * Ω)))
largest_gap = maximum(relative(inv(saturated(random..., ωτ)), debye(ωτ)) for ωτ in ωτs)
ωτ_peak = ωτ_relax * sqrt(μ_hf / μ_lf)
μ_peak = bulk_shear(best_fit_iso(saturated(random..., ωτ_peak)))[2]
(largest_gap = largest_gap, ωτ_peak = ωτ_peak, Q⁻¹ = inverse_Q(μ_peak), closed_form = (μ_hf - μ_lf) / (2sqrt(μ_hf * μ_lf)))
````

## Case 3: vertical cracks

The normals now lie in the horizontal plane, uniformly in azimuth. Six
families, at the azimuths ``k\pi/6``, give the same averages of the products
of up to four components of ``\underline{n}`` as the uniform distribution,
and the rock is transversely isotropic about ``\underline{e}_3``:

````@example squirt_dispersion
const vertical = [[cos(k * π / 6), sin(k * π / 6), 0.0] for k in 0:5]
upright = cracked_rock(vertical, 0.3, 1.0e-3)
C_up_hf, C_up_lf = high(upright), low(upright)
(high = relative(C_up_hf, best_fit_ti(C_up_hf, e₃)), low = relative(C_up_lf, best_fit_ti(C_up_lf, e₃)))
````

A wave traveling in the plane ``(\underline{e}_1, \underline{e}_3)`` at the
angle ``\theta`` from the vertical has a P wave and two S waves: the SH wave
polarized along ``\underline{e}_2``, normal to that plane, and the SV wave
polarized in it. They are told apart by their polarizations:

````@example squirt_dispersion
function upright_waves(C, ρ, θ)
    w = plane_waves(real_stiffness(C), ρ, [sin(θ), 0.0, cos(θ)])
    sh = abs(w.polarization[2, 2]) > abs(w.polarization[2, 3]) ? 2 : 3    ## the S wave polarized along e₂
    return (P = w.velocity[1], SH = w.velocity[sh], SV = w.velocity[5 - sh])
end
ρ_upright = average_density(first(upright))
println("direction          ΔV_P (%)  ΔV_SH (%)  ΔV_SV (%)")
for (name, θ) in (("vertical", 0.0), ("45°", π / 4), ("horizontal", π / 2))
    hf, lf = upright_waves(C_up_hf, ρ_upright, θ), upright_waves(C_up_lf, ρ_upright, θ)
    @printf "%-17s %8.3f   %8.3f   %8.3f\n" name 100 * (hf.P - lf.P) / V_solid[1] 100 * (hf.SH - lf.SH) / V_solid[2] 100 * (hf.SV - lf.SV) / V_solid[2]
end
````

The vertical P wave loads every vertical crack with the same normal stress,
and its pressure is the same in all of them: no dispersion. The horizontal
SV wave, polarized along ``\underline{e}_3``, applies the shear stress
``\sigma_{13}``, whose traction on a vertical crack lies in its plane: no
pressure at all. The horizontal SH wave applies ``\sigma_{12}``, which
compresses the cracks of one diagonal and opens those of the other: the
largest pressure differences, and the largest dispersion. The velocities of
the P and SH waves against the direction of propagation, sealed and relaxed:

````@example squirt_dispersion
θs = range(0, π / 2; length = 91)
V_P_hf = [upright_waves(C_up_hf, ρ_upright, θ).P for θ in θs]
V_P_lf = [upright_waves(C_up_lf, ρ_upright, θ).P for θ in θs]
V_SH_hf = [upright_waves(C_up_hf, ρ_upright, θ).SH for θ in θs]
V_SH_lf = [upright_waves(C_up_lf, ρ_upright, θ).SH for θ in θs]
plot(
    plot(
        rad2deg.(θs), [V_P_hf V_P_lf]; lw = 2, ls = [:solid :dash], label = ["sealed (hf)" "relaxed (lf)"],
        xlabel = "θ from the vertical (degrees)", ylabel = "V_P (km/s)", title = "P wave", titlefontsize = 9, xticks = 0:15:90
    ),
    plot(
        rad2deg.(θs), [V_SH_hf V_SH_lf]; lw = 2, ls = [:solid :dash], label = ["sealed (hf)" "relaxed (lf)"],
        xlabel = "θ from the vertical (degrees)", ylabel = "V_SH (km/s)", title = "SH wave", titlefontsize = 9, xticks = 0:15:90
    ); layout = (1, 2), size = (760, 300), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm
)
````

Case 3 mixes the two previous ones: anisotropic like case 1, dispersive like
case 2, and the dispersion itself depends on the direction.

## Where to go next

- [Fluid-filled pores: pressure polarization](@ref app-fluid-filled-pores):
  the pressure of each sealed family, which the flow equalizes here, for
  pores of any aspect ratio.
- [Levin's theorem](@ref th-levin), §5: the derivation of the sealed, the
  relaxed and the exchanging limits.
- [Seismic signatures](@ref man-seismic): the density, the waves and the
  compliance of the fractures of a cell.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

