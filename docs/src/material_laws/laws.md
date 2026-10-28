# [The laws shipped](@id ml-laws)

Six laws come with `MaterialLaws`. Each is an instance of
[`ResidualLaw`](@ref) or [`PotentialLaw`](@ref), written in a few lines of
`library.jl`, and each is tested against a closed-form response. This page
gives, for every law, its equations, its parameters, and its behavior along a
few loading paths.

The parameter values below are chosen for this page as plausible orders of
magnitude for the material named in each section; they are illustrations, not
calibrated data. Stresses and moduli are in MPa, times in seconds.

```@example mp_laws
using MeanFieldHomogenization, Plots
nothing # hide
```

## Linear elasticity

```math
\boldsymbol{\sigma} = \mathbb{C}:\boldsymbol{\varepsilon},
\qquad
\psi = \tfrac12\,\boldsymbol{\varepsilon}:\mathbb{C}:\boldsymbol{\varepsilon}.
```

[`LinearElasticity`](@ref) takes any stiffness tensor (TensND, any symmetry,
written in any basis) or, isotropic, Young's modulus ``E`` and Poisson's ratio
``\nu``. It has no internal variable and its tangent is ``\mathbb{C}`` itself.
In uniaxial stress the axial slope is ``E`` and the lateral strain
``-\nu\,\varepsilon_{11}``, for an aluminum alloy here:

```@example mp_laws
alu = LinearElasticity(; E = 70e3, ν = 0.33)
tension = point_response(alu, LoadingPath(range(0, 1; length = 11); ε11 = t -> 2e-3 * t))
tension[:σ11][end] / tension[:ε11][end], -tension[:ε22][end] / tension[:ε11][end]
```

## Power-law elasticity

The nonlinear elasticity of the deformation theory of plasticity: linear in
the volume change, a power of exponent ``m`` in the shear,

```math
\psi = \frac{k}{2}(\operatorname{tr}\boldsymbol\varepsilon)^2
+ \frac{\sigma_0\varepsilon_0}{m+1}\left(\frac{\varepsilon_{eq}}{\varepsilon_0}\right)^{m+1},
\qquad
\sigma_{eq} = \sigma_0\left(\frac{\varepsilon_{eq}}{\varepsilon_0}\right)^{m},
```

with ``\varepsilon_{eq} = \sqrt{\tfrac23\,\boldsymbol{e}:\boldsymbol{e}}`` and
``\sigma_{eq}`` the von Mises stress. ``m = 1`` is linear elasticity of shear
modulus ``\sigma_0/(3\varepsilon_0)``; ``m \to 0`` tends to rigid–perfect
plasticity of yield stress ``\sigma_0``. It is the law of the phases in the
nonlinear composites of [ponteCastaneda1991](@citet).

| parameter | meaning |
|:--|:--|
| `k` | bulk modulus |
| `σ₀`, `ε₀` | the equivalent stress reached at the equivalent strain ``\varepsilon_0`` |
| `m` | the exponent, between 0 (perfectly plastic) and 1 (linear) |

An isochoric extension ``\boldsymbol\varepsilon = e\,(1, -\tfrac12, -\tfrac12)``,
for which ``\varepsilon_{eq} = e``, traces the law of the shear part:

```@example mp_laws
isochoric(e_max) = LoadingPath(range(0, 1; length = 41); ε11 = t -> e_max * t,
    ε22 = t -> -e_max * t / 2, ε33 = t -> -e_max * t / 2, ε23 = 0, ε13 = 0, ε12 = 0)
p = plot(; xlabel = "equivalent strain ε_eq (‰)", ylabel = "equivalent stress σ_eq (MPa)", legend = :topleft)
for m in (0.1, 0.25, 0.5, 1.0)
    law = PowerLawElasticity(; k = 140e3, σ₀ = 60.0, ε₀ = 1e-3, m)
    r = point_response(law, isochoric(4e-3))
    plot!(p, r[:ε11] * 1e3, r[:σ11] - r[:σ22]; label = "m = $m", lw = 2)
end
p
```

## Maxwell viscoelasticity

A spring of stiffness ``\mathbb{C}`` in series with a dashpot of viscosity
``\tau\,\mathbb{C}``, written as two potentials of the viscous strain
``\boldsymbol\varepsilon^v``:

```math
\psi = \tfrac12\,(\boldsymbol\varepsilon - \boldsymbol\varepsilon^v):\mathbb{C}:(\boldsymbol\varepsilon - \boldsymbol\varepsilon^v),
\qquad
\varphi = \frac{\tau}{2}\,\dot{\boldsymbol\varepsilon}^v:\mathbb{C}:\dot{\boldsymbol\varepsilon}^v,
\qquad
\dot{\boldsymbol\sigma} = \mathbb{C}:\dot{\boldsymbol\varepsilon} - \frac{\boldsymbol\sigma}{\tau}.
```

Every mode relaxes with the same time ``\tau``. A polymer binder, held at a
fixed strain and pulled at two strain rates:

```@example mp_laws
binder = MaxwellViscoelasticity(; E = 1e3, ν = 0.35, τ = 30.0)
held = point_response(binder, LoadingPath(range(0, 120; length = 121); ε11 = 1e-3, ε22 = 0, ε33 = 0))
p1 = plot(held[:t], held[:σ11] / held[:σ11][1]; label = "computed", lw = 2,
    xlabel = "time (s)", ylabel = "σ₁₁ / σ₁₁(0)", title = "relaxation at fixed strain")
plot!(p1, held[:t], exp.(-held[:t] / 30); label = "exp(−t/τ)", ls = :dash)
p2 = plot(; xlabel = "axial strain ε₁₁ (‰)", ylabel = "axial stress σ₁₁ (MPa)", title = "constant strain rate", legend = :topleft)
for rate in (1e-5, 1e-4)
    T = 2e-3 / rate
    r = point_response(binder, LoadingPath(range(0, T; length = 81); ε11 = t -> rate * t))
    plot!(p2, r[:ε11] * 1e3, r[:σ11]; label = "ε̇ = $rate /s", lw = 2)
end
plot(p1, p2; layout = (1, 2), size = (900, 340))
```

The slower test leaves more time to relax, so the stress stays lower. In
uniaxial stress it tends to the plateau ``E\,\tau\,\dot\varepsilon`` where the
viscous flow takes the whole strain rate: ``10^3 \times 30 \times 10^{-5} = 0.3``
MPa for the slower one. At a rate fast compared with ``1/\tau`` the response
tends to the elastic one.

## Biot poroelasticity

A law of two coupled fields: the strain and the pore pressure ``p``, conjugate
to the stress and to the change of fluid content ``\phi``
([Coussy (2004)](@cite coussy2004)):

```math
\boldsymbol\sigma = \mathbb{C}:\boldsymbol\varepsilon - b\,p\,\boldsymbol{1},
\qquad
\phi = b\operatorname{tr}\boldsymbol\varepsilon + \frac{p}{M}.
```

| parameter | meaning |
|:--|:--|
| `E`, `ν` | the drained elastic moduli |
| `b` | Biot's coefficient, between the porosity and 1 |
| `M` | Biot's modulus |

A sandstone compressed in an oedometer, drained (the pore pressure held at
zero) and undrained (no fluid exchanged, ``\phi = 0``):

```@example mp_laws
sandstone = BiotPoroelasticity(; E = 12e3, ν = 0.22, b = 0.75, M = 9e3)
oedo = (ε11 = t -> -2e-3 * t, ε22 = 0, ε33 = 0)
drained = point_response(sandstone, LoadingPath(range(0, 1; length = 21); oedo..., p = 0))
undrained = point_response(sandstone, LoadingPath(range(0, 1; length = 21); oedo...))
p1 = plot(-drained[:ε11] * 1e3, -drained[:σ11]; label = "drained", lw = 2,
    xlabel = "axial shortening (‰)", ylabel = "axial compression (MPa)", legend = :topleft)
plot!(p1, -undrained[:ε11] * 1e3, -undrained[:σ11]; label = "undrained", lw = 2)
p2 = plot(-undrained[:ε11] * 1e3, undrained[:p]; label = "undrained", lw = 2, color = 2,
    xlabel = "axial shortening (‰)", ylabel = "pore pressure p (MPa)", legend = :topleft)
plot(p1, p2; layout = (1, 2), size = (900, 340))
```

The trapped fluid takes part of the load as pore pressure, and stiffens the
sample by ``b^2 M``.

## Norton viscoplasticity

Isotropic elasticity with a power-law creep of the plastic strain:

```math
\dot{\boldsymbol\varepsilon}^p = \frac32\left(\frac{\sigma_{eq}}{K}\right)^{n}\frac{\boldsymbol{s}}{\sigma_{eq}},
\qquad
\boldsymbol\sigma = \mathbb{C}:(\boldsymbol\varepsilon - \boldsymbol\varepsilon^p),
```

integrated by the backward Euler scheme. Under a constant uniaxial stress the
creep rate is ``(\sigma/K)^n``: the exponent ``n`` sets how fast it rises with
the stress. A steel creeping at high temperature, ``n = 6``:

```@example mp_laws
hot_steel = NortonViscoplasticity(; E = 160e3, ν = 0.3, K = 3000.0, n = 6.0)
p1 = plot(; xlabel = "time (h)", ylabel = "axial strain ε₁₁ (%)", title = "creep at constant stress", legend = :topleft)
for σ in (150.0, 180.0, 210.0)
    r = point_response(hot_steel, LoadingPath(range(0, 3600 * 10; length = 21); σ11 = σ))
    plot!(p1, r[:t] / 3600, r[:ε11] * 100; label = "σ = $(round(Int, σ)) MPa", lw = 2)
end
held = point_response(hot_steel, LoadingPath(range(0, 3600 * 10; length = 101); ε11 = 2e-3, ε22 = 0, ε33 = 0))
p2 = plot(held[:t] / 3600, held[:σ11]; lw = 2, label = nothing,
    xlabel = "time (h)", ylabel = "axial stress σ₁₁ (MPa)", title = "relaxation at fixed strain")
plot(p1, p2; layout = (1, 2), size = (900, 340))
```

At constant stress the creep is steady, a straight line; at fixed strain the
stress relaxes, fast at first and then ever slower, since the creep rate falls
with the stress as its sixth power.

## Von Mises plasticity

Rate-independent plasticity with the von Mises criterion, isotropic hardening
``R(p)`` and linear kinematic hardening:

```math
f = \sigma_{eq}(\boldsymbol\sigma - \boldsymbol X) - R(p) \le 0,
\qquad
\dot{\boldsymbol\varepsilon}^p = \dot p\,\frac32\,\frac{\operatorname{dev}(\boldsymbol\sigma - \boldsymbol X)}{\sigma_{eq}(\boldsymbol\sigma - \boldsymbol X)},
\qquad
\boldsymbol X = \frac23\,H_{kin}\,\boldsymbol\varepsilon^p,
```

``p`` being the cumulated plastic strain. ``R(p) = \sigma_0 + H\,p`` by default;
any function of ``p`` may be given instead (`hardening = p -> …`), here a
saturating one. The integration over a step is the backward Euler scheme, which
for linear hardening is the radial return [simo1998](@cite).

| parameter | meaning |
|:--|:--|
| `E`, `ν` | the elastic moduli |
| `σ₀` | the initial yield stress |
| `H` | the linear isotropic hardening modulus |
| `Hkin` | the linear kinematic hardening modulus |
| `hardening` | a function ``p \mapsto R(p)`` replacing ``\sigma_0 + H p`` |

A structural steel under one tension–compression cycle, with three hardenings
of comparable strength:

```@example mp_laws
σ₀ = 275.0
steels = (
    "isotropic, linear" => J2Plasticity(; E = 210e3, ν = 0.3, σ₀, H = 5e3),
    "kinematic, linear" => J2Plasticity(; E = 210e3, ν = 0.3, σ₀, Hkin = 5e3),
    "isotropic, saturating" => J2Plasticity(; E = 210e3, ν = 0.3, σ₀, hardening = p -> σ₀ + 150 * (1 - exp(-60p))),
)
cycle(t) = 6e-3 * (t < 0.25 ? 4t : t < 0.75 ? 2 - 4t : 4t - 4)
p = plot(; xlabel = "axial strain ε₁₁ (‰)", ylabel = "axial stress σ₁₁ (MPa)", legend = :outerright, size = (760, 380))
for (name, law) in steels
    r = point_response(law, LoadingPath(range(0, 1; length = 161); ε11 = cycle))
    plot!(p, r[:ε11] * 1e3, r[:σ11]; label = name, lw = 2)
end
p
```

Isotropic hardening expands the elastic domain, so reverse yielding starts late;
kinematic hardening moves it, so reverse yielding starts early (the
Bauschinger effect); the saturating law hardens fast at first and then flattens
to ``\sigma_0 + 150`` MPa.

## Single-crystal viscoplasticity

A crystal deforms by slip on the systems of its lattice: a plane of normal
``\underline{n}_s`` and a direction ``\underline{m}_s`` in it, of Schmid tensor
``\boldsymbol P_s = \operatorname{sym}(\underline m_s \otimes \underline n_s)``. Each
system slips when its resolved shear stress exceeds its critical value, and
hardens with the slip accumulated on all of them ([meric1991](@citet)):

```math
\tau_s = \boldsymbol\sigma : \boldsymbol P_s,
\qquad
\dot\gamma_s = \left\langle \frac{|\tau_s| - r_s}{K} \right\rangle^{n} \operatorname{sign}\tau_s,
\qquad
\boldsymbol\varepsilon^p = \sum_s \gamma_s\,\boldsymbol P_s,
```

```math
\dot v_s = |\dot\gamma_s|,
\qquad
r_s = \tau_0 + Q \sum_j h_{sj}\left(1 - e^{-b\,v_j}\right).
```

| parameter | meaning |
|:--|:--|
| `C₁₁`, `C₁₂`, `C₄₄` | the cubic elastic constants |
| `lattice` | `:fcc` (12 systems ``\{111\}\langle110\rangle``) or `:bcc` (12 systems ``\{110\}\langle111\rangle``), see [`slip_systems`](@ref) |
| `frame` | the orientation: a TensND basis whose vectors are the cube axes |
| `K`, `n` | the viscous stress scale and exponent of the slip rate |
| `τ₀` | the initial critical resolved shear stress |
| `Q`, `b` | the saturation and the rate of the hardening |
| `latent` | the interaction between systems: 1, every system hardens every other alike |

The response depends on the orientation of the crystal with respect to the
load, through the Schmid factors ``(\underline a\cdot\underline n_s)(\underline a\cdot\underline m_s)``
of the loading axis ``\underline a``. A copper-like FCC crystal pulled at a
constant strain rate along three of its directions, each brought onto the
tensile axis ``\underline e_1`` by rotating the crystal:

```@example mp_laws
using TensND, StaticArrays, LinearAlgebra
# The crystal frame that brings the crystal direction d onto the global e₁.
function along(d)
    d = normalize(SVector{3, Float64}(d))
    u = normalize(cross(d, abs(d[3]) < 0.9 ? SVector(0.0, 0, 1) : SVector(0.0, 1, 0)))
    return RotatedBasis(SMatrix{3, 3}(vcat(d', u', cross(d, u)')))
end
crystal(frame) = CrystalViscoplasticity(; C₁₁ = 168e3, C₁₂ = 121e3, C₄₄ = 75e3, frame,
    K = 10.0, n = 10.0, τ₀ = 10.0, Q = 3.0, b = 20.0)
pull = LoadingPath(range(0, 20; length = 81); ε11 = t -> 1e-3 * t)       # 1e-3 /s up to 2 %
p = plot(; xlabel = "axial strain ε₁₁ (%)", ylabel = "axial stress σ₁₁ (MPa)", legend = :bottomright)
for (name, d) in ("[001]" => (0, 0, 1), "[011]" => (0, 1, 1), "[111]" => (1, 1, 1))
    r = point_response(crystal(along(d)), pull)
    plot!(p, r[:ε11] * 100, r[:σ11]; label = "tension along $name", lw = 2)
end
p
```

Along ``[001]`` and ``[011]`` the most stressed systems have the largest Schmid
factor, ``1/\sqrt6 \approx 0.41``, and the crystal yields at the same axial
stress; along ``[111]`` it is ``0.27``, and the crystal yields about 1.5 times
higher. The elastic slopes differ too: a cubic crystal is stiffer along
``[111]`` than along ``[001]``.
