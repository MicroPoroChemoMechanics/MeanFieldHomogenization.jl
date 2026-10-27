# [Polycrystals](@id mp-theory-polycrystals)

A metal is an aggregate of grains, each a single crystal that deforms by slip
on its own slip systems, oriented at random or along a texture. Its response
follows from the [single-crystal law](@ref mp-laws) of every grain and from a
rule saying how the grains share the macroscopic strain and stress. This page
states the rule of uniform strain, the Taylor model, for an aggregate of many
grains, and how its orientations are drawn. The
[uniform-field composites](@ref mp-theory-uniform-fields) are the same rules
for a few phases. [The example](@ref mp-ex-polycrystal) runs an aggregate.

## The grains

Grain ``g`` has the volume fraction ``w_g`` and a law of
[`CrystalViscoplasticity`](@ref), whose cube axes are the vectors of its
orientation: cubic elasticity, slip on the twelve systems of the FCC or the BCC
lattice, a critical resolved shear stress ``\tau_0`` and a viscous overstress.
On system ``s`` of the grain, of normal ``\underline{n}_s`` and direction
``\underline{m}_s`` rotated with it, the resolved shear stress is
``\tau_s = \boldsymbol{\sigma}:\operatorname{sym}(\underline{m}_s\otimes\underline{n}_s)``.
As the viscous overstress becomes small (the parameter ``K`` of the law), a grain
slips only on the systems whose resolved stress reaches ``\tau_0``.

## The Taylor model

Every grain undergoes the macroscopic strain:

```math
\boldsymbol{\varepsilon}_g = \boldsymbol{E},
\qquad
\boldsymbol{\Sigma} = \sum_g w_g\,\boldsymbol{\sigma}_g ,
\qquad
\frac{\partial\boldsymbol{\Sigma}}{\partial\boldsymbol{E}} = \sum_g w_g\,\frac{\partial\boldsymbol{\sigma}_g}{\partial\boldsymbol{E}} .
```

It is the [uniform-strain composite](@ref mp-theory-uniform-fields) of the
grains. Since a grain then sees nothing of the others, each is integrated on
its own: [`Polycrystal`](@ref) keeps the state of every grain in a vector and
calls the local integration of each, which scales to hundreds of grains, where
[`UniformStrain`](@ref) solves all the phases in one system of equations.

To follow a plastic strain imposed on it, a grain has to activate enough slip
systems to produce any isochoric strain, five independent ones in general. The
macroscopic flow stress in tension is then ``M\,\tau_0``, ``M`` being the Taylor
factor of the texture. For a random texture of FCC crystals,
[bishopHill1951](@citet) found ``M \approx 3.06``. In elasticity the Taylor model
gives the Voigt average of the rotated stiffnesses; for cubic crystals and a
random texture it is isotropic, of bulk modulus ``(C_{11} + 2C_{12})/3`` and shear
modulus ``(C_{11} - C_{12} + 3C_{44})/5``.

## A random texture

[`random_orientations`](@ref) draws orientations uniformly over the rotations.
With the Euler angles of TensND, ``\theta`` the angle of the third axis from
``\underline{e}_3`` and ``\phi``, ``\psi`` the two azimuths, the uniform measure is
``\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\phi\,\mathrm{d}\psi``: ``\cos\theta``
uniform on ``[-1, 1]``, ``\phi`` and ``\psi`` uniform on ``[0, 2\pi)``. Drawing
``\theta`` itself uniformly would crowd the orientations near the poles.

## What is not covered

- The **Sachs** model, every grain at the macroscopic stress, is the
  [uniform-stress composite](@ref mp-theory-uniform-fields) of the grains, for a
  few of them. For an aggregate of nearly rigid–plastic grains it needs the
  stress of the step to stay below the flow stress of every grain while it is
  sought, which a strain-driven iteration does not ensure; it is not offered
  for large aggregates.
- The **self-consistent** interactions of polycrystal plasticity (the
  elastic interaction of Kröner, its plastic extension by Berveiller and Zaoui,
  the β rule of Cailletaud and Pilvin) are not implemented: they will follow
  their original papers ([kroner1961](@cite), [berveillerZaoui1978](@cite),
  [cailletaudPilvin1994](@cite)).
