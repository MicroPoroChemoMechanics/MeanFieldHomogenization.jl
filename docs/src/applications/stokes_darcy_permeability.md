# [Stokes-to-Darcy permeability of porous media](@id app-stokes-darcy)

!!! info "Before this page"
    [Permeability from Stokes flow](@ref th-brinkman) for the Brinkman
    inhomogeneity problem and the schemes built on it, and
    [Brinkman inclusions](@ref man-brinkman) for their syntax.

The previous page showed that the shape chosen for the pores governs the
percolation of a self-consistent estimate in elasticity and conduction. The
permeability is still more sensitive to the pore structure, and it cannot be
obtained from Darcy–Darcy homogenization when the pores themselves are resolved:
the fluid obeys the Stokes equations, and a pore is not a phase of finite
permeability. Following [bignonnetIJES2022](@citet), this page estimates the
permeability of four families of microstructures from Brinkman inhomogeneities:
beds of impervious grains, mats of fibers, solids crossed by tubes or slits, and
double-porosity media. Each estimate is a scheme of conduction fed with
inclusions whose response depends on their size ``R`` through
``\alpha_0 = R/\sqrt{k_0}``. The comparison shows what each scheme captures and
where it fails. The full study, with the symbolic derivation of every
inclusion, is `scripts/71_brinkman_permeability.jl`, shown at the end of the
page.

```@example stokesdarcy
using MeanFieldHomogenization, TensND, Plots
gr()
iso(x) = TensISO{3}(x)
val(t) = first(TensND.get_data(t))
newton(init) = SelfConsistent(algorithm = NewtonDefault(), init = iso(init), reltol = 1.0e-12)
nothing # hide
```

## 1. Beds of impervious grains

Spheres of radius ``R`` fill a fraction ``1 - \varphi`` of the volume, and the
complement is the pore space. In the resistance form a solid has ``r = \infty``
and the pore phase ``r = 0``. The matrix of the dilute, Mori–Tanaka and
differential schemes is the pore space, and the self-consistent scheme needs a
shape for the pores as well. Following [bignonnetIJES2022](@citet), that shape is
chosen with the same specific surface as the grains: spheres of radius ``R`` or
cylinders of radius ``2R/3``.

```@example stokesdarcy
R = 1.0
function granular(φ, pore)
    rve = RVE()
    add_phase!(rve, :p, pore, Dict(:r => iso(0.0)); fraction = :rest, symmetrize = :iso)
    add_phase!(rve, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => iso(0.0)); fraction = 1 - φ)
    return rve
end
sphere_pore = BrinkmanSphere(R, BrinkmanFluid())
tube_pore = BrinkmanCylinder(2R / 3, BrinkmanFluid())
coated(φ) = (c = RVE(); add_phase!(c, :c, BrinkmanCoatedSphere(R, R / cbrt(1 - φ)), Dict(:r => iso(0.0)); fraction = 1.0); c)

φs = range(0.36, 0.95; length = 24)
k(rve, scheme) = 1 / val(homogenize(rve, scheme, :r))
k_dil = [k(granular(φ, sphere_pore), Dilute(:p)) for φ in φs]
k_mt = [k(granular(φ, sphere_pore), MoriTanaka(:p)) for φ in φs]
k_dif = [k(granular(φ, sphere_pore), DifferentialScheme(:p)) for φ in φs]
k_sc_sph = [k(granular(φ, sphere_pore), newton(1.0)) for φ in φs]
k_sc_cyl = [k(granular(φ, tube_pore), newton(1.0)) for φ in φs]
k_sc_coat = [k(coated(φ), newton(1.0)) for φ in φs]
nothing # hide
```

The cell model of [happelAIChE1958](@citet), with stress-free tangential
conditions on the outer boundary of a coated sphere of porosity ``\varphi``, gives a
closed form that serves as a classical comparison:

```@example stokesdarcy
happel(φ) = (η = cbrt(1 - φ); (1 - η)^3 * (1 + η) * (2 + η + 2η^2) / (3η^3 * (3 + 2η^5)) * R^2)
plot(φs, [k_dil k_mt k_dif k_sc_sph k_sc_cyl k_sc_coat happel.(φs)];
    yscale = :log10, xlabel = "porosity φ", ylabel = "k / R²", legend = :bottomright,
    label = ["dilute" "Mori–Tanaka" "differential" "self-consistent, spherical pores" "self-consistent, cylindrical pores" "self-consistent, coated sphere" "cell model (stress-free)"],
    ls = [:solid :solid :solid :dash :dash :dash :dot], lw = 2, palette = :tab10, size = (760, 500))
```

Three groups appear. The dilute and Mori–Tanaka estimates use the clear fluid as
reference. They reproduce the drag of an isolated grain and nothing of the
others: ``r^{\mathrm{dil}} = 9(1 - \varphi)/(2R^2)`` and ``r^{\mathrm{MT}} = r^{\mathrm{dil}}/\varphi``,
which the [symbolic route of the manual](@ref man-brinkman) returns in closed form.
The differential scheme, which is the iterated dilute approximation of
[wilkinsonPF1985](@citet), lowers the permeability without changing that
reference. The self-consistent estimates use the porous effective medium as
reference, which screens the flow around each grain over ``\sqrt{k}``. With
spherical pores the estimate percolates at ``\varphi_c = 1/3``. With cylindrical
pores it never percolates, and the coated sphere sits close to the cell model,
as both place the porosity as a regular shell around every grain.

The screening by the effective medium is what produces the first correction to
the drag of a sparse random bed, in ``\sqrt{1 - \varphi}``, with the coefficient
``3/\sqrt{2}`` [howellsJFM1974, hinchJFM1977, kimRusselJFM1985](@cite). The
self-consistent scheme with spherical pores recovers it: the scaled departure
below tends to ``3/\sqrt{2}`` as ``1 - \varphi \to 0``, slowly, since it differs
from its limit by a term of order ``\sqrt{1 - \varphi}``.

```@example stokesdarcy
for x in (1.0e-2, 1.0e-3, 1.0e-4)
    φ = 1 - x
    ratio = val(homogenize(granular(φ, sphere_pore), newton(9x / 2), :r)) / (9x / (2R^2))
    println("1 − φ = ", x, ":  (r_sc/r_dil − 1)/√(1 − φ) = ", round((ratio - 1) / sqrt(x); digits = 4),
        "   3/√2 = ", round(3 / sqrt(2); digits = 4))
end
```

## 2. Mats of fibers

Impervious fibers of radius ``R`` with isotropic orientations make the fibrous
counterpart. In a clear fluid the drag on an infinite cylinder vanishes
(Stokes' paradox), so there is no dilute, Mori–Tanaka or differential estimate
from the fluid, and the package refuses the problem:

```@example stokesdarcy
try
    flux_gradient_concentration(BrinkmanCylinder(R, BrinkmanSolid()), iso(0.0), iso(0.0))
catch err
    println(first(sprint(showerror, err), 120), "…")
end
```

The self-consistent estimate remains, with cylindrical pores of the fiber radius
or spherical pores of radius ``3R/2`` (equal specific surfaces):

```@example stokesdarcy
function fibrous(φ, pore)
    rve = RVE()
    add_phase!(rve, :p, pore, Dict(:r => iso(0.0)); fraction = φ, symmetrize = :iso)
    add_phase!(rve, :s, BrinkmanCylinder(R, BrinkmanSolid()), Dict(:r => iso(0.0)); fraction = 1 - φ, symmetrize = :iso)
    return rve
end
φf = range(0.5, 0.97; length = 20)
plot(φf, [[k(fibrous(φ, BrinkmanCylinder(R, BrinkmanFluid())), newton(1.0)) for φ in φf] [k(fibrous(φ, BrinkmanSphere(1.5R, BrinkmanFluid())), newton(1.0)) for φ in φf]];
    yscale = :log10, xlabel = "porosity φ", ylabel = "k / R²", legend = :topleft,
    label = ["self-consistent, cylindrical pores" "self-consistent, spherical pores"], lw = 2, ls = [:dash :dot],
    palette = :tab10, size = (760, 460))
```

## 3. Tubes and slits in an impervious solid

When the flow runs only in tubes of radius ``R`` or slits of thickness ``2R``,
the matrix is impervious, ``\boldsymbol{k}_0 = \boldsymbol{0}``, and the phases are
described in the permeability form. In an impervious matrix a tube carries
Poiseuille's flow, ``\boldsymbol{k}\cdot\boldsymbol{B} = (R^2/8)\,\underline{n}\otimes\underline{n}``, and
a slit ``(R^2/3)\,(\boldsymbol{1} - \underline{n}\otimes\underline{n})``. With isotropic
orientations, the Mori–Tanaka estimates are then ``R^2\varphi/(24(1 - \varphi))``
and ``2R^2\varphi/(9(1 - \varphi))``:

```@example stokesdarcy
P = PermeabilityForm()
function spanning(φ, pore; Rs = R)
    rve = RVE()
    add_phase!(rve, :s, BrinkmanSphere(Rs, BrinkmanSolid(); form = P), Dict(:k => iso(0.0)); fraction = :rest)
    add_phase!(rve, :p, pore, Dict(:k => iso(0.0)); fraction = φ, symmetrize = :iso)
    return rve
end
tube = BrinkmanCylinder(R, BrinkmanFluid(); form = P)
slit = BrinkmanLayer(R, BrinkmanFluid(); form = P)
φ = 0.2
(tubes = val(homogenize(spanning(φ, tube), MoriTanaka(:s), :k)) => R^2 * φ / (24 * (1 - φ)),
 slits = val(homogenize(spanning(φ, slit), MoriTanaka(:s), :k)) => 2R^2 * φ / (9 * (1 - φ)))
```

The differential scheme starts from the solid and adds pores, and the
self-consistent scheme represents the solid by spheres of the same specific
surface as the pores (radius ``3R/2`` for tubes, ``3R`` for slits):

```@example stokesdarcy
φt = range(0.02, 0.5; length = 20)
curves = [val(homogenize(spanning(φ, pore; Rs = Rs), scheme, :k)) for φ in φt,
    (pore, Rs) in ((tube, 1.5R), (slit, 3R)),
    scheme in (MoriTanaka(:s), DifferentialScheme(:s), newton(0.01))]
labels = ["Mori–Tanaka" "differential" "self-consistent"]
plot(φt, [curves[:, 1, j] for j in 1:3]; yscale = :log10, lw = 2, palette = :tab10, label = labels .* ", tubes",
    xlabel = "porosity φ", ylabel = "k / R²", legend = :bottomright, size = (760, 460))
plot!(φt, [curves[:, 2, j] for j in 1:3]; lw = 2, ls = :dash, palette = :tab10, label = labels .* ", slits")
```

## 4. Double porosity

A microporous matrix of permeability ``k_\mu`` holds spherical mesopores of radius
``R`` at volume fraction ``\varphi``. This is a three-dimensional variant, with
spherical mesopores, of the double-porosity media studied by
[bignonnetIJES2022](@citet). The ratio ``\alpha = R/\sqrt{k_\mu}`` compares the
mesopores with the screening length of the matrix. As ``\alpha \to \infty`` the
Brinkman layer around each mesopore becomes thin and Darcy–Darcy homogenization
is recovered, where a mesopore is an inclusion of infinite permeability, for
instance ``k^{\mathrm{MT}}/k_\mu = (1 + 2\varphi)/(1 - \varphi)``:

```@example stokesdarcy
kμ = 1.0
function double_porosity(φ, α; maxwell = false)
    rve = maxwell ? RVE(; distribution_shape = Ellipsoid(1.0)) : RVE()
    add_phase!(rve, :m, BrinkmanSphere(α * sqrt(kμ); form = P), Dict(:k => iso(kμ)); fraction = :rest)
    add_phase!(rve, :p, BrinkmanSphere(α * sqrt(kμ), BrinkmanFluid(); form = P), Dict(:k => iso(0.0)); fraction = φ)
    return rve
end
[(α = α, mori_tanaka = val(homogenize(double_porosity(0.3, α), MoriTanaka(:m), :k)) / kμ) for α in (3.0, 30.0, 3.0e3)]
```

and the Darcy value at ``\varphi = 0.3`` is ``1.6/0.7 \approx 2.2857``. Maxwell's
scheme takes the Hill tensor of conduction for the spherical distribution, which
the permeability form makes legitimate. The distribution carries no size,
whereas the mesopores do. As [bignonnetIJES2022](@citet) found for disks, the
resulting estimate leaves the admissible range at large mesoporosity: it falls
below ``k_\mu/(1 - \varphi)``, the bound obtained by averaging the resistance:

```@example stokesdarcy
φd = range(0.0, 0.7; length = 29)
mt, dif, sc, mx = (zeros(length(φd)) for _ in 1:4)
for (i, φ) in enumerate(φd)
    mt[i] = val(homogenize(double_porosity(φ, 3.0), MoriTanaka(:m), :k))
    dif[i] = val(homogenize(double_porosity(φ, 3.0), DifferentialScheme(:m), :k))
    sc[i] = φ < 0.6 ? val(homogenize(double_porosity(φ, 3.0), newton(kμ), :k)) : NaN
    mx[i] = val(homogenize(double_porosity(φ, 3.0; maxwell = true), Maxwell(), :k))
end
plot(φd, [mt dif sc mx 1 ./ (1 .- φd)] ./ kμ; lw = 2, palette = :tab10, yscale = :log10,
    ls = [:solid :dash :dot :dashdot :solid], color = [1 2 3 4 :gray],
    label = ["Mori–Tanaka" "differential" "self-consistent" "Maxwell" "bound k_μ/(1 − φ)"],
    xlabel = "mesoporosity φ", ylabel = "k / k_μ  (α = 3)", legend = :topleft, size = (760, 460))
```

## 5. Sensitivities

The inclusions are generic in their number type, so ForwardDiff differentiates
an estimate through the self-consistent fixed point. With respect to the grain
radius the answer is known in advance, since at fixed porosity the permeability
scales as ``R^2``. With respect to the porosity it is not:

```@example stokesdarcy
using ForwardDiff
function k_sc(R, φ)
    g = RVE()
    add_phase!(g, :p, BrinkmanSphere(R, BrinkmanFluid()), Dict(:r => iso(0.0)); fraction = :rest)
    add_phase!(g, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => iso(0.0)); fraction = 1 - φ)
    return k(g, newton(1.0))
end
(dlnk_dlnR = ForwardDiff.derivative(r -> log(k_sc(r, 0.6)), 1.0),
 dk_dφ = ForwardDiff.derivative(φ -> k_sc(1.0, φ), 0.6))
```

## 6. Scope

The schemes inherit the assumptions of the Brinkman problem: the same viscosity
in every medium and no stress jump at an interface between a clear fluid and a
porous medium. The shapes available are those with a separable solution
(sphere, cylinder, layer, coated sphere). The percolation threshold of a
self-consistent estimate is set by the shapes chosen for the pores, as on the
[previous page](@ref app-concave-pores), and should not be read as a prediction of
that of a real material.

```@raw html
<details class="source-file"><summary>Show the code of <code>scripts/71_brinkman_permeability.jl</code>: the symbolic derivation of every inclusion, then the full study</summary>
```

```@eval
Main.show_source("scripts/71_brinkman_permeability.jl")
```

```@raw html
</details>
```

## Where to go next

The general concepts end here. The applications that follow are organized by
material, starting with
[Multiscale elasticity of a hydrating cement paste](@ref app-cement-paste). The
transport properties of the paste, its diffusivity in particular, are estimated
by the same schemes in
[Cement paste: chloride diffusivity and elasticity](@ref app-cement-paste-diffusion).

- [Permeability from Stokes flow](@ref th-brinkman): the derivation of the
  inclusions used above.
- [Fractured permeability](@ref fe-permeability): Darcy flow through flowing
  cracks, the Darcy–Darcy counterpart.

## References

```@bibliography
Pages = ["stokes_darcy_permeability.md"]
Canonical = false
```
