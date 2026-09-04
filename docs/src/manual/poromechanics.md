# [Poromechanics](@id manual-poromechanics)

Once a scheme has produced a **drained** stiffness ``\mathbb{C}^{\rm hom}``, the
poroelastic law of a saturated medium is closed *without any further
homogenization*: for a solid phase with uniform elastic properties
``\mathbb{C}_{\rm s}`` (compliance ``\mathbb{S}_{\rm s} = \mathbb{C}_{\rm s}^{-1}``),

```math
\boldsymbol{B} = \boldsymbol{1} : \left(\mathbb{I} - \mathbb{S}_{\rm s} : \mathbb{C}^{\rm hom}\right),
\qquad
\frac{1}{M} = \boldsymbol{1} : \mathbb{S}_{\rm s} : \left(\boldsymbol{B} - \varphi\,\boldsymbol{1}\right),
```

the **Biot tensor** and **Biot modulus** ([coussy2004](@cite)) — ``\boldsymbol{B}``
here, and nowhere else in this documentation, denotes the Biot tensor rather
than a [crack opening displacement](@ref th-cod-tensors). They enter the constitutive
law as

```math
\dot{\boldsymbol{\Sigma}} = \mathbb{C}^{\rm hom} : \dot{\boldsymbol{E}} - \dot{p}\,\boldsymbol{B},
\qquad
\dot{\varphi} = \boldsymbol{B} : \dot{\boldsymbol{E}} + \frac{\dot{p}}{M} .
```

``\boldsymbol{B}`` is generally **anisotropic** even for an isotropic solid, because
the pore space need not be isotropic.

```@example poro
using MeanFieldHomogenization, TensND

k_s, μ_s, φ = 20.0, 12.0, 0.2
C_s = TensISO{3}(3k_s, 2μ_s)

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_s); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-9, 1.0e-9));
           fraction = φ)

par = poroelastic_parameters(homogenize(rve, MoriTanaka()), C_s, φ)
(b = par.B[1, 1], invM = par.inverse_modulus, M = par.modulus)
```

Spherical pores give ``\boldsymbol{B} = b\,\boldsymbol{1}`` with the familiar
``b = 1 - k^{\rm hom}/k_s``. Aligned cracks do not:

```@example poro
rve_c = RVE()
add_phase!(rve_c, :M, Ellipsoid(1.0), Dict(:C => C_s); fraction = :rest)
add_phase!(rve_c, :CR, PennyCrack(1.0), Dict(:C => C_s); density = 0.08)

B = biot_tensor(homogenize(rve_c, MoriTanaka()), C_s)
(B₁₁ = B[1, 1], B₃₃ = B[3, 3])       # normal to the cracks is e₃
```

!!! warning "The fluid is assumed incompressible"
    ``1/M`` above holds for ``k_f = \infty``. A compressible fluid adds a
    storage term ``\varphi/k_f``. The distinction is not cosmetic: with an
    incompressible fluid and compressible grains the Skempton coefficient
    **exceeds one**, because the pore volume is held fixed while the grains
    themselves compress.

## Effective stresses

Two different measures, easy to confuse:

| | | drives |
|:--|:--|:--|
| [`terzaghi_stress`](@ref) | ``\boldsymbol{\Sigma} + p\,\boldsymbol{1}`` | the **microstructure** — crack opening and closure |
| [`biot_effective_stress`](@ref) | ``\boldsymbol{\Sigma} + p\,\boldsymbol{B}`` | the **macroscopic** law, which it reduces to the drained one |

They coincide only when ``\boldsymbol{B} = \boldsymbol{1}``. That the *Terzaghi*
measure is the one governing the pore space is the argument of
[barthelemyARMA2011](@cite) § 1.1: the loading ``(\boldsymbol{\Sigma}, p)`` splits
into a dry problem under ``\boldsymbol{\Sigma} + p\,\boldsymbol{1}``, plus a
uniform field that carries no strain singularity and so cannot open or close a
flat crack.

## Drained ↔ undrained

```math
\mathbb{C}^{\rm u} = \mathbb{C}^{\rm hom} + M\,\boldsymbol{B} \otimes \boldsymbol{B}
```

via [`undrained_stiffness`](@ref) and [`drained_stiffness`](@ref), with
[`skempton_tensor`](@ref) giving the pore pressure built up by an undrained
stress increment, ``p = -\boldsymbol{B}^{\rm sk} : \boldsymbol{\Sigma}``.

!!! note "Homogeneous solid phase only"
    These relations need a solid phase with *uniform* elastic properties — a
    rock matrix with pores or fractures. A medium built from two distinct solid
    constituents needs the general Levin/eigenstrain route, and ``\mathbb{C}_{\rm s}``
    is then not defined.
