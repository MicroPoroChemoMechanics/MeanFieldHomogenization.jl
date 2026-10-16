# [Thermal expansion and Biot tensor](@id tut-levin)

!!! info "Before this page"
    The theory page [Levin's theorem](@ref th-levin), which states and proves
    the two formulas used below, and the manual section
    [Concentration tensors of the phases](@ref man-concentration-tensors), which
    shows how the concentration tensors of a scheme are obtained and read.

A homogenization scheme gives the effective stiffness of a composite, and with
it the concentration tensors of its phases. This page uses the latter for two
loadings that the stiffness alone does not cover: a temperature change, which
makes each phase expand freely by its own amount, and a fluid pressure in the
pores, which acts on the solid as a prestress. Both are answered by Levin's
theorem, first numerically on a two-phase composite and a porous solid, then in
closed form.

## A two-phase composite

The composite is made of a compliant matrix with a large thermal expansion, in
which stiff spherical inclusions with a small one are dispersed. The values are
illustrative, with moduli in GPa and expansion coefficients in 1/K:

| Phase | ``k`` | ``\mu`` | ``\alpha`` |
| :-- | :-- | :-- | :-- |
| matrix `:M` | 5 | 2 | ``60\times 10^{-6}`` |
| inclusions `:I` | 40 | 30 | ``5\times 10^{-6}`` |

```@example tutlevin
using MeanFieldHomogenization
using TensND
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

k_M, μ_M, α_M = 5.0, 2.0, 60.0e-6
k_I, μ_I, α_I = 40.0, 30.0, 5.0e-6
C_M, C_I = iso_stiffness(k_M, μ_M), iso_stiffness(k_I, μ_I)
𝟏 = TensISO{3}(1.0)   # second-order identity

function composite(f; shape = Ellipsoid(1.0))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_M); fraction = :rest)
    add_phase!(rve, :I, shape, Dict(:C => C_I); fraction = f)
    return rve
end
nothing # hide
```

A temperature rise ``\Delta T`` gives each phase the free strain
``\boldsymbol\varepsilon^\ast_r = \alpha_r\,\Delta T\,\boldsymbol 1``, and Levin's
theorem gives the macroscopic one,
``\boldsymbol E^\ast = \sum_r f_r\,\boldsymbol\varepsilon^\ast_r:\mathbb B_r``, from the
stress concentration tensors ``\mathbb B_r`` of the scheme. Taking
``\Delta T = 1``, [`levin_free_strain`](@ref) returns the effective thermal
expansion tensor ``\boldsymbol\alpha^{\mathrm{hom}}`` directly:

```@example tutlevin
ct = concentration_tensors(composite(0.3), MoriTanaka(:M), :C)
α_hom = levin_free_strain(ct, Dict(:M => α_M * 𝟏, :I => α_I * 𝟏))
α_hom[1, 1]
```

For two phases the theorem needs no concentration tensor at all: the free
strain is fixed by the effective compliance alone, and for isotropic phases by
the effective bulk modulus,
``\alpha^{\mathrm{hom}} = \alpha_M + (\alpha_I - \alpha_M)\,(1/k^{\mathrm{hom}} - 1/k_M)/(1/k_I - 1/k_M)``.
The two routes agree:

```@example tutlevin
k_hom, _ = k_mu(ct.effective)
α_exact = α_M + (α_I - α_M) * (1 / k_hom - 1 / k_M) / (1 / k_I - 1 / k_M)
α_hom[1, 1] - α_exact
```

## The estimate depends on the scheme

Each scheme has its own concentration tensors, hence its own estimate of the
thermal expansion. The bounds give two classical rules: under Reuss every phase
carries the macroscopic stress, ``\mathbb B_r = \mathbb I``, and the expansion
coefficients mix by volume; under Voigt every phase undergoes the macroscopic
strain, and they mix weighted by the bulk moduli,
``\alpha^{\mathrm V} = \sum_r f_r\,k_r\,\alpha_r/\sum_r f_r\,k_r``.

```@example tutlevin
function α_of(f, scheme)
    ct = concentration_tensors(composite(f), scheme, :C)
    return levin_free_strain(ct, Dict(:M => α_M * 𝟏, :I => α_I * 𝟏))[1, 1]
end

fs = range(0.0, 0.6; length = 25)
schemes = [
    "Reuss" => Reuss(), "Voigt" => Voigt(), "Mori–Tanaka" => MoriTanaka(:M),
    "self-consistent" => SelfConsistent(), "differential" => DifferentialScheme(:M),
]
plt = plot(;
    xlabel = "volume fraction of inclusions f", ylabel = "α_hom (10⁻⁶/K)",
    title = "k_M = 5, μ_M = 2, α_M = 60e-6; k_I = 40, μ_I = 30, α_I = 5e-6",
    titlefontsize = 9, legend = :topright, framestyle = :box, size = (760, 480),
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)
for (name, scheme) in schemes
    plot!(plt, fs, [1.0e6 * α_of(f, scheme) for f in fs]; label = name, lw = 2)
end
plt
```

Reuss and Voigt still frame the estimates, as they frame the bulk modulus from
which, for two phases, the expansion follows. Of the estimates, Mori–Tanaka
gives the largest expansion, the matrix being the continuous phase that
imposes its own.

## Aligned platelets: an anisotropic expansion

Flat inclusions aligned with the normal ``\underline e_3`` make the composite
transversely isotropic, and its expansion with it. The two-phase formula still
applies, with the effective compliance in place of the bulk modulus:
``\boldsymbol\alpha^{\mathrm{hom}} = \alpha_M\,\boldsymbol 1 + \frac{\alpha_I - \alpha_M}{1/k_I - 1/k_M}\,\big(3\,\boldsymbol 1:\mathbb S^{\mathrm{hom}} - \boldsymbol 1/k_M\big)``.

```@example tutlevin
ct = concentration_tensors(composite(0.3; shape = Ellipsoid(1.0, 1.0, 0.1)), MoriTanaka(:M), :C)
α_hom = levin_free_strain(ct, Dict(:M => α_M * 𝟏, :I => α_I * 𝟏))
α_exact = α_M * 𝟏 + (α_I - α_M) / (1 / k_I - 1 / k_M) * (3 * (𝟏 ⊡ ct.compliance) - 𝟏 / k_M)
(α₁₁ = α_hom[1, 1], α₃₃ = α_hom[3, 3], difference = maximum(abs, get_array(α_hom - α_exact)))
```

In the plane of the platelets the stiff inclusions restrain the matrix, while
across them the composite expands even more than the matrix alone: held back in
the plane, the matrix expands further across it by the Poisson effect.

## A saturated porous solid: the Biot tensor

A fluid at pressure ``p`` filling the pores acts on them as the prestress
``-p\,\boldsymbol 1``, and Levin's theorem in its second form,
``\boldsymbol\Sigma^\ast = \sum_r f_r\,\boldsymbol\sigma^\ast_r:\mathbb A_r``, gives the
macroscopic prestress ``-p\,\boldsymbol B``, where ``\boldsymbol B`` is the Biot
tensor. The solid is the matrix above, with a porosity of 10 % in spherical pores
and a family of penny-shaped cracks of normal ``\underline e_3`` and density 0.1,
filled with the same fluid:

```@example tutlevin
rve = RVE()
add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => C_M); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => zero(C_M)); fraction = 0.1)
add_phase!(rve, :c, PennyCrack(1.0), Dict(:C => C_M); density = 0.1)

ct = concentration_tensors(rve, MoriTanaka(:S), :C)
B = -levin_prestress(ct, Dict(:P => -𝟏, :c => -𝟏))
(B₁₁ = B[1, 1], B₃₃ = B[3, 3])
```

A crack family has no volume: it enters with its density as weight and its
concentration tensor per unit density, and its openings across ``\underline e_3``
make ``B_{33}`` the larger coefficient. With a single solid phase the theorem
reduces to the classical
``\boldsymbol B = \boldsymbol 1:(\mathbb I - \mathbb S_s:\mathbb C^{\mathrm{hom}})`` of
[`biot_tensor`](@ref):

```@example tutlevin
maximum(abs, get_array(B - biot_tensor(ct.effective, C_M)))
```

With several solid phases that shortcut is lost and only the theorem applies;
[Poromechanics](@ref manual-poromechanics) gives an example.

## In closed form

Mori–Tanaka on spheres runs on symbolic moduli, and so do the concentration
tensors and Levin's theorem: the two results above come out as formulas. With
SymPy:

```@example tutlevinsym
using MeanFieldHomogenization
using TensND
using SymPy

@syms k0::positive μ0::positive k1::positive μ1::positive f::positive
@syms α0::positive α1::positive
𝟏 = TensISO{3}(Sym(1))

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(k0, μ0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(k1, μ1)); fraction = f)
ct = concentration_tensors(rve, MoriTanaka(:M), :C)
α_MT = levin_free_strain(ct, Dict(:M => α0 * 𝟏, :I => α1 * 𝟏))[1, 1]
nothing # hide
```

The thermal expansion of the Mori–Tanaka estimate is

```math
\alpha^{\mathrm{MT}} = \alpha_0 + (\alpha_1 - \alpha_0)\,
  \frac{f\,k_1\,(3k_0 + 4\mu_0)}{k_0\,(3k_1 + 4\mu_0) + 4\mu_0\,f\,(k_1 - k_0)} ,
```

which does not depend on the shear modulus of the inclusions, and returns
``\alpha_0`` at ``f = 0`` and ``\alpha_1`` at ``f = 1``:

```@example tutlevinsym
simplify(α_MT - (α0 + (α1 - α0) * f * k1 * (3k0 + 4μ0) / (k0 * (3k1 + 4μ0) + 4μ0 * f * (k1 - k0))))
```

Turning the inclusions into pores filled with a fluid gives the Biot coefficient
of the Mori–Tanaka estimate for spherical pores of porosity ``f``,

```math
b^{\mathrm{MT}} = \frac{f\,(3k_0 + 4\mu_0)}{3f\,k_0 + 4\mu_0} ,
```

which tends to ``1`` for an incompressible solid, the limit of Terzaghi's
effective stress, and to the porosity ``f`` as ``k_0 \to 0``:

```@example tutlevinsym
rve_p = RVE()
add_phase!(rve_p, :S, Ellipsoid(1.0), Dict(:C => iso_stiffness(k0, μ0)); fraction = :rest)
add_phase!(rve_p, :P, Ellipsoid(1.0), Dict(:C => zero(iso_stiffness(k0, μ0))); fraction = f)
ct_p = concentration_tensors(rve_p, MoriTanaka(:S), :C)
b_MT = -levin_prestress(ct_p, Dict(:P => -𝟏))[1, 1]
simplify(b_MT - f * (3k0 + 4μ0) / (3f * k0 + 4μ0))
```

## Where to go next

The theorem and its proof are in [Levin's theorem](@ref th-levin), and the
concentration tensors of each scheme at the end of its theory page, from
[Voigt and Reuss bounds](@ref th-scheme-bounds) to
[The equivalent inclusion method](@ref th-eim). The tutorial
[Symbolic spheres](@ref tut-symbolic-spheres) derives the Mori–Tanaka moduli
used above in closed form, and [Poromechanics](@ref manual-poromechanics) closes
the poroelastic law with the Biot modulus.
