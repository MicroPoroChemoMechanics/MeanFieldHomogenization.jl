# [Nonlinear poroelasticity of a cracked rock](@id mp-theory-closing-cracks)

A rock that contains microcracks is soft under a small confinement and stiffens
as the confinement grows, because its cracks close one after the other. The
same mechanism makes its Biot coefficient decrease with the confinement: a
closed crack no longer stores fluid, and no longer transmits the pore pressure
to the solid. [walsh1965](@citet) explained the nonlinear compressibility of
rocks in this way. [deude2002](@citet) put it in a micromechanical setting of
nonlinear poroelasticity. This page derives the law that MeanFieldHomogenization
builds from the linear poroelasticity of the cracked solid.
[The example](@ref mp-ex-closing-cracks) runs it.

## The linear poroelastic solid at given open cracks

The rock is an elastic solid of stiffness ``\mathbb{C}_s``, containing families
of penny-shaped cracks that an incompressible fluid saturates. Family ``i`` has
the normal ``\underline{n}_i`` and, when all its cracks are open, the density
``d_i``. As long as the set of open cracks does not change, the rock is a linear
poroelastic solid [coussy2004](@cite). Its drained stiffness
``\mathbb{C}^{\mathrm{hom}}`` is that of the solid with the open cracks, from any
scheme of the package. Its Biot tensor and Biot modulus follow from
``\mathbb{C}^{\mathrm{hom}}`` and ``\mathbb{C}_s`` (see
[Poromechanics](@ref manual-poromechanics)):

```math
\boldsymbol{B} = \boldsymbol{1}:\big(\mathbb{I} - \mathbb{S}_s:\mathbb{C}^{\mathrm{hom}}\big),
\qquad
\frac{1}{M} = \boldsymbol{1}:\mathbb{S}_s:(\boldsymbol{B} - \phi\,\boldsymbol{1}),
```

``\mathbb{S}_s`` being the compliance of the solid and ``\phi`` the porosity
that the fluid occupies, here the volume of the cracks, small for thin cracks.
The cracks make ``\mathbb{C}^{\mathrm{hom}}`` softer and ``\boldsymbol{B}`` larger.

## What closes the cracks

The loading of the saturated rock is the macroscopic stress
``\boldsymbol{\Sigma}`` and the pore pressure ``p``. It splits into two problems.
In the first, the pore pressure ``p`` acts everywhere: on the faces of the cracks
and in the pores, and, through a stress ``-p\,\boldsymbol{1}``, on the solid. This
field is uniform; it carries no concentration at the tips of the cracks and
cannot open or close them. The second problem is that of the dry rock under the
**Terzaghi effective stress**

```math
\boldsymbol{\Sigma}' = \boldsymbol{\Sigma} + p\,\boldsymbol{1},
```

and that problem alone opens and closes the cracks. This argument, which the
fractured-rock material of [barthelemyARMA2011](@citet) already rests on, holds
whatever the stiffness of the solid. It is the effective stress of the
closure. The deformation obeys another one, the Biot effective stress
``\boldsymbol{\Sigma} + p\,\boldsymbol{B}``, as the rate equations below show: a
pore pressure reopens the cracks fully, but deforms the rock only through
``\boldsymbol{B}``.

A flat crack of aspect ratio ``\omega`` closes when the compressive normal
traction on its plane reaches a stress proportional to ``\omega`` (and to the
stiffness of the solid). A family is a population of cracks of various aspect
ratios, so that the thinner ones close first. Under the normal effective traction
``\sigma_{n,i} = \underline{n}_i\cdot\boldsymbol{\Sigma}'\cdot\underline{n}_i``, a
fraction ``f_i(\sigma_{n,i})`` of its cracks remains open, and its open density
is ``d_i\,f_i(\sigma_{n,i})``. The fraction depends on the distribution of the
aspect ratios; an exponential distribution of the closure stresses, of mean
``\sigma_c``, gives

```math
f(\sigma_n) = \exp\Big(-\frac{\langle -\sigma_n\rangle}{\sigma_c}\Big),
```

[`ExponentialClosure`](@ref): every crack is open in tension, and a fraction
``1/e`` of them remains open under a compression ``\sigma_c``. A closed crack
transmits both normal and shear tractions, as in the fractured-rock material.

## The rate equations

During a small increment of the loading, the cracks open at that moment behave
as in the linear solid, and a crack that closes stops contributing to the next
increments. The rates are therefore those of the linear solid with the current
open densities:

```math
\dot{\boldsymbol{\Sigma}}' = \mathbb{C}^{\mathrm{hom}}(\boldsymbol{\Sigma}'):\dot{\boldsymbol{E}} + \dot p\,\big(\boldsymbol{1} - \boldsymbol{B}(\boldsymbol{\Sigma}')\big),
\qquad
\dot\varphi = \boldsymbol{B}(\boldsymbol{\Sigma}'):\dot{\boldsymbol{E}} + \frac{\dot p}{M(\boldsymbol{\Sigma}')},
```

``\varphi`` being the change of fluid content. The first equation is
``\dot{\boldsymbol{\Sigma}} = \mathbb{C}^{\mathrm{hom}}:\dot{\boldsymbol{E}} - \dot p\,\boldsymbol{B}``
rewritten for the effective stress. The moduli depend on the state only through
the open densities, hence through the normal effective tractions.

The law is hypoelastic: the moduli are tangent moduli, and in general they are
not the second derivatives of a potential. The crack compliance mixes normal and
shear parts, while the closure depends on the normal traction only. The
response is therefore obtained by integrating the rates along the loading path,
not by a formula of the current state.

## The law in `MaterialPoints`

[`ClosingCrackPoroelasticity`](@ref) builds a [`ResidualLaw`](@ref) with two
gradients, the strain and the pore pressure, and their two fluxes, the stress and
the fluid content. Its internal variables are the effective stress
``\boldsymbol{\Sigma}'`` and the fluid content ``\varphi``. A step integrates the
rates by the backward Euler scheme, with the moduli at the end of the step:

```math
\boldsymbol{\Sigma}' - \boldsymbol{\Sigma}'_n = \mathbb{C}^{\mathrm{hom}}(\boldsymbol{\Sigma}'):\Delta\boldsymbol{E} + \Delta p\,\big(\boldsymbol{1} - \boldsymbol{B}(\boldsymbol{\Sigma}')\big),
\qquad
\varphi - \varphi_n = \boldsymbol{B}(\boldsymbol{\Sigma}'):\Delta\boldsymbol{E} + \frac{\Delta p}{M(\boldsymbol{\Sigma}')}.
```

The stress is ``\boldsymbol{\Sigma} = \boldsymbol{\Sigma}' - p\,\boldsymbol{1}``. The
scheme is first order: halving the steps halves the error. The consistent tangent
follows from the implicit-function theorem and contains the variation of the
moduli with the effective stress during the step; every block of it, between the
strain and the pressure, the stress and the fluid content, is computed.

## Scope

| | |
|:--|:--|
| families | oriented, each seeing one normal traction; a family symmetrized over all orientations is refused, since its cracks see different tractions (several oriented families sample the orientations instead) |
| closure | any open fraction ``f(\sigma_n)``, one per family; closed cracks are welded (no sliding, no friction, no hysteresis) |
| fluid | incompressible, as in the fractured-rock material; the porosity ``\phi`` is held at its given value while the cracks close |
| not covered | the permeability of the cracks; the opening of the cracks beyond their initial aperture under tension |
