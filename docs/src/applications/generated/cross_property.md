```@meta
EditURL = "../../../../scripts/110_cross_property_connections.jl"
```

# [Elastic and conductive properties of porous and cracked materials](@id app-cross-property)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/110_cross_property_connections.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/110_cross_property_connections.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/110_cross_property_connections.jl`, runs it end to end.

!!! info "Before this page"
    [Cross-property connections](@ref th-cross-property), which says when a
    connection between the two properties is exact and what its approximate
    form drops, [Extension to conductivity](@ref th-conductivity) for the
    resistivity contribution, and [Cracks](@ref man-cracks) and
    [Conductivity](@ref man-conductivity) for the syntax.

Pores and cracks lower both the stiffness and the conductivity of a solid.
This page computes the two changes for one shape at a time, sets their ratio
against the closed forms of Bristow and of Sevostianov and Kachanov, as the
review of [Sevostianov and Kachanov](@cite sevostianov2009) gives them, then
uses the ratio on
two sets of measurements reviewed in the same work: a closed-cell aluminum
foam and plasma-sprayed zirconia coatings.

````@example cross_property
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf
using Plots
gr()
nothing #hide
````

## Two contributions of one inhomogeneity

Without interactions, an inhomogeneity adds its compliance contribution
``\mathbb{H}`` to the compliance ``\mathbb{S}_0`` of the matrix and its
resistivity contribution ``\boldsymbol{R}`` to the resistivity
``\boldsymbol{K}_0^{-1}``. For pores of volume fraction ``f`` and for cracks
of density ``\rho = \frac1V\sum a^3`` (``a`` the radius of a circular crack),

```math
\mathbb{S} = \mathbb{S}_0 + f\,\mathbb{H}
\quad\text{or}\quad \mathbb{S}_0 + \tfrac{4\pi}{3}\rho\,\mathbb{H},
\qquad
\boldsymbol{K}^{-1} = \boldsymbol{K}_0^{-1} + f\,\boldsymbol{R}
\quad\text{or}\quad \boldsymbol{K}_0^{-1} + \tfrac{4\pi}{3}\rho\,\boldsymbol{R} .
```

With the inhomogeneities oriented at random, both contributions are replaced
by their averages over the rotations, ``\overline{\mathbb{H}}`` and
``\overline{\boldsymbol{R}}``, and the material stays isotropic, of Young
modulus ``E`` and conductivity ``k``:

```math
\frac{E_0 - E}{E} = E_0\,\Delta S_{1111},\qquad
\frac{k_0 - k}{k} = k_0\,\Delta r_{11},\qquad
\psi = \frac{(E_0 - E)/E}{(k_0 - k)/k}
     = \frac{E_0\,\overline{H}_{1111}}{k_0\,\overline{R}_{11}} .
```

The weight ``f`` or ``\rho`` cancels in ``\psi``: in this approximation the
connection coefficient depends on the shape and on ``\nu_0`` only, not on
the amount of pores or cracks. [`compliance_contribution`](@ref) and
[`resistivity_contribution`](@ref) give ``\mathbb{H}`` and
``\boldsymbol{R}``; a pore takes zero stiffness and zero conductivity, a dry
crack the properties of the matrix alone; `isotropify` averages over the
rotations.

````@example cross_property
contributions(pore, C₀, K₀) = (compliance_contribution(pore, 0 * C₀, C₀), resistivity_contribution(pore, 0 * K₀, K₀))
contributions(crack::EllipticCrack, C₀, K₀) = (compliance_contribution(crack, C₀), resistivity_contribution(crack, K₀))

function relative_changes(shape, E₀, ν₀, k₀)                        ## per unit weight, random orientations
    H, R = isotropify.(contributions(shape, iso_stiffness_young_poisson(E₀, ν₀), TensISO{3}(k₀)))
    return E₀ * canonical_components(H)[1, 1, 1, 1], k₀ * canonical_components(R)[1, 1]
end
connection(shape, ν₀) = (c = relative_changes(shape, one(ν₀), ν₀, one(ν₀)); c[1] / c[2])
nothing #hide
````

## Cracks at random: Bristow's relations

For circular cracks oriented at random, Bristow's relations, per unit crack
density, are given by [sevostianov2009](@citet), Eqs. (2.1) and (2.2):

```math
\frac{E_0 - E}{E} = \frac{16(1 - \nu_0^2)(10 - 3\nu_0)}{45(2 - \nu_0)}\,\rho,
\qquad
\frac{k_0 - k}{k} = \frac{8}{9}\,\rho ,
```

so that ``\psi = 2(1 - \nu_0^2)(10 - 3\nu_0)/(5(2 - \nu_0))`` (Eq. (2.3)).
The changes computed from the two contributions of `PennyCrack`, multiplied
by ``4\pi/3``:

````@example cross_property
bristow_E(ν) = 16(1 - ν^2) * (10 - 3ν) / (45(2 - ν))
bristow_ψ(ν) = 2(1 - ν^2) * (10 - 3ν) / (5(2 - ν))
penny = PennyCrack(1.0)
println(" ν₀     (E₀-E)/Eρ  Bristow     (k₀-k)/kρ  8/9      ψ computed  Bristow")
for ν in (0.0, 0.2, 0.33, 0.45)
    e, k = (4π / 3) .* relative_changes(penny, 1.0, ν, 1.0)
    @printf "%.2f   %.6f   %.6f    %.6f   %.6f   %.6f    %.6f\n" ν e bristow_E(ν) k 8 / 9 e / k bristow_ψ(ν)
end
````

The same numbers come out of one cell that carries both properties, `:C`
and `:K`, homogenized twice by the scheme without interactions
([`DiluteDual`](@ref)), here at the density ``\rho = 0.1``:

````@example cross_property
function cracked_cell(E₀, ν₀, k₀, ρ)
    C₀, K₀ = iso_stiffness_young_poisson(E₀, ν₀), TensISO{3}(k₀)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C₀, :K => K₀); fraction = :rest)
    add_phase!(rve, :cracks, penny, Dict(:C => C₀, :K => K₀); density = ρ, symmetrize = IsoSymmetrize())
    return rve
end
young(C) = 1 / canonical_components(inv(C))[1, 1, 1, 1]
conductivity(K) = canonical_components(K)[1, 1]
cell = cracked_cell(1.0, 0.33, 1.0, 0.1)
E = young(homogenize(cell, DiluteDual(), :C))
k = conductivity(homogenize(cell, DiluteDual(), :K))
(ΔE = 1 / E - 1, Bristow = 0.1bristow_E(0.33), Δk = 1 / k - 1, closed_form = 0.1 * 8 / 9)
````

## The approximate coefficient

The resistivity contribution of a crack is ``\boldsymbol{R} \propto
\underline{n}\otimes\underline{n}``, its compliance contribution holds besides
a term in ``\underline{n}\otimes\underline{n}\otimes\underline{n}\otimes\underline{n}``.
Dropping that term gives a connection valid for any orientation distribution
of the cracks, between the Young modulus and the conductivity along the same
axis ([sevostianov2009](@citet), Eq. (5.15)):

```math
\frac{E_0 - E_i}{E_i} \approx \frac{4(1 - \nu_0^2)}{2 - \nu_0}\,\frac{k_0 - k_i}{k_i} .
```

For cracks at random the exact coefficient is Bristow's; their ratio is
``(10 - 3\nu_0)/10 = 1 - 0.3\,\nu_0``:

````@example cross_property
approximate_ψ(ν) = 4(1 - ν^2) / (2 - ν)
println(" ν₀     ψ computed   4(1-ν₀²)/(2-ν₀)   ratio      1 - 0.3ν₀")
for ν in (0.0, 0.2, 0.3, 0.33, 0.45)
    ψ = connection(penny, ν)
    @printf "%.2f   %.6f     %.6f          %.6f   %.6f\n" ν ψ approximate_ψ(ν) ψ / approximate_ψ(ν) 1 - 0.3ν
end
````

The approximation overestimates the drop of stiffness by 10 % at
``\nu_0 = 0.33``. The same relation is printed in
[sevostianov2002](@citet) as Eq. (6.5), with the coefficient
``2(1-\nu_0^2)/(2-\nu_0)``; Eq. (6.4a) of that article, from which it is
drawn, gives ``4(1-\nu_0^2)/(2-\nu_0)`` on the Young moduli, as Eq. (5.15) of
the review does, and so does the computation above.

## Spheres and spheroids

For spherical pores both contributions are isotropic and the connection is
exact in the approximation without interactions ([sevostianov2009](@citet),
Eq. (5.12a)):

```math
\psi_{\text{sphere}} = \frac{(1 - \nu_0)(9 + 5\nu_0)}{7 - 5\nu_0} .
```

````@example cross_property
sphere_ψ(ν) = (1 - ν) * (9 + 5ν) / (7 - 5ν)
sphere = Ellipsoid(1.0)
[(ν₀ = ν, computed = connection(sphere, ν), closed_form = sphere_ψ(ν)) for ν in (0.0, 0.2, 0.33, 0.45)]
````

With a symbolic Poisson ratio the same calls return the closed form:

````@example cross_property
import Symbolics
Symbolics.@variables ν
ψ_symbolic = Symbolics.simplify(connection(sphere, ν))
ψ_symbolic, Symbolics.simplify(ψ_symbolic - sphere_ψ(ν))
````

For spheroids of aspect ratio ``\gamma`` (the polar semi-axis over the
equatorial one), oriented at random, the coefficient is computed over four
decades of ``\gamma``, from flat oblate pores to long needles, for two
Poisson ratios. The flat end tends to Bristow's crack value, the
dotted lines:

````@example cross_property
γs = 10.0 .^ range(-2, 2; length = 49)
spheroid(γ) = Ellipsoid(1.0, 1.0, γ)
shape_curves = plot(;
    xscale = :log10, xlabel = "aspect ratio γ", ylabel = "ψ = [(E₀-E)/E] / [(k₀-k)/k]",
    legend = :topright, size = (620, 360), title = "Pores at random: connection coefficient", titlefontsize = 10
)
ψ_spheroids = Dict{Float64, Vector{Float64}}()
for (ν, color) in ((0.2, 1), (0.33, 2))
    ψ_spheroids[ν] = [connection(spheroid(γ), ν) for γ in γs]
    plot!(shape_curves, γs, ψ_spheroids[ν]; lw = 2, color, label = "ν₀ = $ν")
    hline!(shape_curves, [bristow_ψ(ν)]; color, ls = :dot, label = false)
    scatter!(shape_curves, [1.0], [sphere_ψ(ν)]; color, markersize = 4, label = false)
end
shape_curves
````

How much the coefficient moves with the shape, at ``\nu_0 = 0.33``:

````@example cross_property
ψs = ψ_spheroids[0.33]
for (γmin, γmax) in ((0.5, 2.0), (0.2, 5.0), (0.01, 100.0))
    sel = findall(γ -> γmin - 1e-9 ≤ γ ≤ γmax + 1e-9, γs)
    @printf "γ in [%5.2f, %6.1f]:  ψ from %.3f to %.3f\n" γmin γmax extrema(ψs[sel])...
end
````

Between ``\gamma = 0.5`` and ``2`` the coefficient stays within a few
percent of the sphere's: the shape of moderately deformed pores need not be
known to pass from one property to the other. Over the whole range it moves
between the sphere and the crack values, by about 40 %.

## A closed-cell aluminum foam

[sevostianov2009](@citet), section 7.1 (also [kachanov2018](@citet),
section 7.5), report the electric conductivity and the Young modulus of an
AlMg1Si0.6 foam measured on different specimens over a range of porosities
``p``, the volume fraction of pores, with the constants of the bulk alloy:

````@example cross_property
literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
const review = literature("sevostianov2009")
quantity(name) = review["quantities"][name]["value"]
const E₀, ν₀, k₀ = quantity("aluminum_E0"), quantity("aluminum_nu0"), quantity("aluminum_k0")
column(table, name) = [Float64(row[name]) for row in review["tables"][table]]
const p_k, k_measured = column("aluminum_foam_conductivity", "porosity"), column("aluminum_foam_conductivity", "k")
const p_E, E_measured = column("aluminum_foam_young", "porosity"), column("aluminum_foam_young", "E")
(E₀ = E₀, ν₀ = ν₀, k₀ = k₀, conductivity_points = length(p_k), modulus_points = length(p_E))
````

````@example cross_property
Main.show_literature("sevostianov2009") # hide
````

The pores were described as spheroids of mean aspect ratio about 0.7
(section 7.1, Fig. 7.2). The coefficient for spheres and for randomly
oriented spheroids ``\gamma = 0.7``, against the values 1.33 and 1.34 quoted
below Eq. (7.4):

````@example cross_property
ψ_sphere, ψ_07 = connection(sphere, ν₀), connection(spheroid(0.7), ν₀)
(sphere = ψ_sphere, published_sphere = quantity("psi_sphere"), spheroid_07 = ψ_07, published_spheroid = quantity("psi_spheroid_07"))
````

Each measured conductivity gives a predicted modulus,
``E = E_0\big/\bigl(1 + \psi\,(k_0 - k)/k\bigr)``, set against the measured
moduli:

````@example cross_property
predicted(k, ψ) = E₀ / (1 + ψ * (k₀ - k) / k)
foam = scatter(
    p_E, E_measured ./ E₀; marker = :diamond, color = :black, label = "E measured (Table 7.2)",
    xlabel = "porosity p", ylabel = "E / E₀", xlims = (0.4, 0.9), ylims = (0, 0.4), legend = :topright,
    size = (620, 360), title = "Aluminum foam: modulus from the conductivity", titlefontsize = 10
)
scatter!(foam, p_k, predicted.(k_measured, ψ_07) ./ E₀; color = 2, label = "E from k measured (Table 7.1), ψ = $(round(ψ_07; digits = 3))")
foam
````

Where the two sets of specimens overlap, from ``p = 0.7`` to ``0.85``, the
moduli predicted from the conductivities lie among the measured ones,
although at such porosities the pores are far from isolated. The next
section shows that neither property, taken alone, is estimated as well.

## The two properties by the schemes

One cell carries the alloy with both properties and the pores, spheroids of
aspect ratio 0.7 at random, with zero stiffness and zero conductivity. Each
scheme estimates the two properties of the same cell
([Cells and schemes](@ref man-schemes)):

````@example cross_property
function foam_cell(p)
    C₀, K₀ = iso_stiffness_young_poisson(E₀, ν₀), TensISO{3}(k₀)
    rve = RVE()
    add_phase!(rve, :alloy, sphere, Dict(:C => C₀, :K => K₀); fraction = :rest)
    add_phase!(rve, :pores, spheroid(0.7), Dict(:C => 0 * C₀, :K => 0 * K₀); fraction = p, symmetrize = IsoSymmetrize())
    return rve
end
estimates(p, scheme) = (young(homogenize(foam_cell(p), scheme, :C)) / E₀, conductivity(homogenize(foam_cell(p), scheme, :K)) / k₀)
schemes = ("no interaction" => DiluteDual(), "Mori-Tanaka" => MoriTanaka(), "differential" => DifferentialScheme(), "self-consistent" => SelfConsistent())
println("scheme             E/E₀ at p = 0.3   0.45   0.8      k/k₀ at p = 0.3   0.45   0.8")
for (name, scheme) in schemes
    values = [estimates(p, scheme) for p in (0.3, 0.45, 0.8)]
    @printf "%-18s %8.4f %8.4f %8.4f    %8.4f %8.4f %8.4f\n" name first.(values)... last.(values)...
end
````

The self-consistent estimates vanish at a finite porosity, close to
``p = 1/2`` for the modulus (exactly ``1/2`` for spheres) and to ``2/3`` for
the conductivity: they cannot describe a foam of porosity 0.7 to 0.85. The
curves, with the measurements:

````@example cross_property
ps = range(0, 0.9; length = 37)
curves = Dict(name => [estimates(p, scheme) for p in ps] for (name, scheme) in schemes)
panel(i, ylabel, data_p, data) = begin
    fig = plot(; xlabel = "porosity p", ylabel, legend = i == 1 ? :topright : false, xlims = (0, 0.9), ylims = (0, 1))
    for (name, scheme) in schemes
        plot!(fig, ps, getindex.(curves[name], i); lw = 2, label = name)
    end
    scatter!(fig, data_p, data; color = :black, markersize = 3, label = "measured")
end
plot(
    panel(1, "E / E₀", p_E, E_measured ./ E₀), panel(2, "k / k₀", p_k, k_measured ./ k₀);
    layout = (1, 2), size = (760, 340), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm
)
````

No single scheme follows both sets of measurements: the measured moduli lie
between the Mori-Tanaka and the differential curves, the measured
conductivities below all of them. The coefficient ``\psi`` that each scheme
implies, ``[(E_0 - E)/E]/[(k_0 - k)/k]`` on its own two estimates:

````@example cross_property
println("scheme             ψ at p = 0.1     0.3      0.5      0.8")
for (name, scheme) in schemes[1:3]
    ratio(p) = ((E, k) = estimates(p, scheme); (1 / E - 1) / (1 / k - 1))
    @printf "%-18s %10.4f %8.4f %8.4f %8.4f\n" name ratio.((0.1, 0.3, 0.5, 0.8))...
end
````

Without interactions and with Mori and Tanaka's scheme the ratio is that of
the contributions at every porosity: for pores, the Mori-Tanaka estimate is
the one without interactions with ``f`` replaced by ``f/(1-f)``, in both
properties alike. The differential scheme raises it with the porosity, to
2.4 at ``p = 0.8``; the measurements, which follow the constant coefficient
in the figure of the previous section, do not support that rise. The ratio
of the two changes is thus better predicted than either change.

## Plasma-sprayed zirconia coatings

Yttria-stabilized zirconia coatings ([sevostianov2009](@citet), section 7.5;
[kachanov2018](@citet), section 7.1) hold microcracks of two families,
roughly parallel to the substrate plane ``(x_1, x_2)`` and normal to it.
The Young modulus ``E_1`` was measured along the substrate and the thermal
conductivity ``k_3`` across it. Without interactions, a crack density tensor
``\boldsymbol{\alpha} = \frac1V\sum a^3\,\underline{n}\otimes\underline{n}``
governs the resistivity, ``(k_0 - k_i)/k_i = \frac83\alpha_{ii}``, and the
approximate connection ties ``E_1`` to ``k_1``; the ratio
``\alpha_{11}/\alpha_{33}``, read on micrographs, passes from ``k_1`` to
``k_3`` (Eq. (7.16)):

```math
E_1 = E_0\left[1 + \frac{4(1-\nu_0^2)}{2-\nu_0}\,\frac{\alpha_{11}}{\alpha_{33}}\,
  \frac{k_0 - k_3}{k_3}\right]^{-1} .
```

````@example cross_property
const ysz = review["tables"]["ysz_coatings"]
const E₀z, ν₀z, k₀z, anisotropy = quantity("ysz_E0"), quantity("ysz_nu0"), quantity("ysz_k0"), quantity("ysz_alpha11_over_alpha33")
eq716(k₃, c = approximate_ψ(ν₀z)) = E₀z / (1 + c * anisotropy * (k₀z - k₃) / k₃)
nothing #hide
````

The cracks can also be given to the package: horizontal cracks, of normal
``\underline{e}_3`` and density ``\rho_h``, and vertical cracks whose normals
are spread at random in the substrate plane, of density ``\rho_v``. Their
density tensor has ``\alpha_{33} = \rho_h`` and
``\alpha_{11} = \alpha_{22} = \rho_v/2``, so that
``\rho_v = 2.8\,\rho_h`` gives ``\alpha_{11}/\alpha_{33} = 1.4``. The
contributions of the vertical cracks are averaged over the rotations about
``\underline{e}_3`` (`transverse_isotropify`):

````@example cross_property
C₀z, K₀z = iso_stiffness_young_poisson(E₀z, ν₀z), TensISO{3}(k₀z)
horizontal = PennyCrack(1.0)
vertical = PennyCrack(1.0; euler_angles = (π / 2, 0.0, 0.0))       ## normal along e₁
e₃ = [0.0, 0.0, 1.0]
H_h, R_h = contributions(horizontal, C₀z, K₀z)
H_v, R_v = transverse_isotropify.(contributions(vertical, C₀z, K₀z), Ref(e₃))
TensND.vecbasis(vertical.basis, :cov)[:, 3]                       ## the normal of the vertical cracks
````

For one scale ``s``, ``\rho_h = s`` and ``\rho_v = 2.8\,s``, the resistivity
across the coating is
``1/k_3 = 1/k_0 + \frac{4\pi}{3}s\,(R^h_{33} + 2.8\,R^v_{33})``. The
measured ``k_3`` fixes ``s``; the compliance follows, and ``E_1``:

````@example cross_property
function two_families(k₃)
    weight(H_h, H_v) = (4π / 3) * (H_h + 2.8 * H_v)
    s = (1 / k₃ - 1 / k₀z) / canonical_components(weight(R_h, R_v))[3, 3]
    S = inv(C₀z) + s * weight(H_h, H_v)
    return (E₁ = 1 / canonical_components(S)[1, 1, 1, 1], ρ_h = s, ρ_v = 2.8s)
end
println("specimen  k₃     E₁ measured  (7.16) printed  computed  with 2.14   two families  ratio of the changes")
for row in ysz
    fam, k₃ = two_families(row["k3"]), row["k3"]
    @printf "   %s     %.2f   %4.1f ± %.1f    %6.2f        %6.2f    %6.2f      %6.2f        %.3f\n" row["specimen"] k₃ row["E1_measured"] row["E1_uncertainty"] row["E1_predicted"] eq716(k₃) eq716(k₃, 2.14) fam.E₁ (E₀z / fam.E₁ - 1) / (E₀z / eq716(k₃) - 1)
end
````

The printed column is recovered with the coefficient rounded to 2.14, as in
Eq. (7.15), and differs from the unrounded one by a few hundredths of a
GPa. The two families of cracks, with the full compliance contribution of
each crack, predict a stiffer coating: the term in
``\underline{n}\otimes\underline{n}\otimes\underline{n}\otimes\underline{n}``,
dropped by the approximation, lowers the change of compliance by about
11 % here, against ``0.3\,\nu_0 = 9`` % for cracks at random. The measured
moduli lie below both predictions for three specimens out of four; the
approximate relation (7.16), closer to them, owes part of that to the term
it drops. The moduli are four to eleven times lower than that of dense
zirconia, and the cracks that produce such drops have densities of order
one and above, where the approximation without interactions is used well beyond its
range.

## Where to go next

- [Cross-property connections](@ref th-cross-property): when the connection
  is exact and what the approximate one drops.
- [Conductivity](@ref man-conductivity): the conduction counterparts of
  the elastic calls used here.
- [Crack distributions: isotropic or parallel](@ref tut-crack-distributions):
  the schemes on cracked solids, and where they part.
- [Cement paste: chloride diffusivity and elasticity](@ref app-cement-paste-diffusion):
  both properties computed on one hydrating microstructure.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

