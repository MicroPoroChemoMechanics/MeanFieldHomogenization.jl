# [The strength of a porous plastic solid](@id mp-theory-strength)

A porous solid whose solid phase obeys a von Mises criterion has a macroscopic
strength domain: the set of macroscopic stresses it can carry. The tutorial
[From derivatives to a strength criterion](@ref tut-strength-criteria) builds
that domain from the derivatives of the linear effective moduli. This page shows
that the same domain is what the [secant homogenization](@ref mp-theory-secant)
of [`NonlinearHomogenized`](@ref) gives, when the solid is a
rigid–perfectly plastic phase law. The approach is that of
[barthelemyDormieux2003](@citet); [the example](@ref mp-ex-strength) runs it.

## The solid as a phase law

The solid can carry any stress whose deviator has a norm
``\sigma_d = \sqrt{\boldsymbol{\sigma}^d:\boldsymbol{\sigma}^d}`` at most ``\sigma_o``
(no factor under the root, see [Conventions](@ref th-notation-deviatoric)). In
the deformation theory of plasticity, a rigid–perfectly plastic solid is the
power law of [`PowerLawElasticity`](@ref) with the exponent ``m = 0``: its
deviatoric stress has the norm ``\sigma_o`` whatever the deviatoric strain, and
its secant shear modulus is

```math
\mu^{s}(\varepsilon_d) = \frac{\sigma_o}{2\,\varepsilon_d},
\qquad
\varepsilon_d = \sqrt{\boldsymbol{e}:\boldsymbol{e}}.
```

The law of the package is written with the von Mises stress
``\sigma_{\mathrm{eq}} = \sqrt{3/2}\,\sigma_d`` and the equivalent strain
``\varepsilon_{\mathrm{eq}} = \sqrt{2/3}\,\varepsilon_d``, so that `σ₀` is
``\sqrt{3/2}\,\sigma_o``. A positive exponent ``m`` gives a hardening solid whose
response tends to the rigid–plastic one as ``m \to 0``. Plastic flow conserves
the volume: the solid is given a bulk modulus large against its secant shear
modulus.

## The secant estimate is a strength

With the modified secant method, the solid takes the secant modulus at the
quadratic mean of its deviatoric strain in the linear comparison composite,
``\bar\varepsilon_d^2 = \langle\boldsymbol{e}:\boldsymbol{e}\rangle_s``. The
quadratic mean of its deviatoric stress is then exactly the bound:

```math
\langle\boldsymbol{\sigma}^d:\boldsymbol{\sigma}^d\rangle_s
= \big(2\mu^{s}\big)^2\,\langle\boldsymbol{e}:\boldsymbol{e}\rangle_s
= \sigma_o^2 .
```

Multiply the macroscopic strain ``\boldsymbol{E}`` by ``\lambda > 0``. The strains
of the comparison composite scale by ``\lambda``, the secant modulus by
``1/\lambda``, and so does the effective stiffness of a composite of an
incompressible solid and empty pores, which is proportional to the shear modulus
of the solid. The macroscopic stress
``\boldsymbol{\Sigma} = \mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}`` does not change: the
composite flows at a constant stress, which is a point of its strength domain.
Sweeping the direction of ``\boldsymbol{E}`` sweeps the boundary of the domain.

The condition ``\langle\sigma_d^2\rangle_s = \sigma_o^2`` is the one that the
tutorial writes with the derivatives of ``k^{\mathrm{hom}}`` and
``\mu^{\mathrm{hom}}`` with respect to the shear modulus of the solid. For an
isotropic composite it reads

```math
2A\,\Sigma_m^2 + B\,\Sigma_d^2 = (1-\varphi)\,\sigma_o^2,
\qquad
A = \frac{1}{(k^{\mathrm{hom}})^2}\frac{\partial k^{\mathrm{hom}}}{\partial\mu_s}\,\mu_s^2,
\qquad
B = \frac{1}{(\mu^{\mathrm{hom}})^2}\frac{\partial \mu^{\mathrm{hom}}}{\partial\mu_s}\,\mu_s^2,
```

with ``\Sigma_m`` the mean stress, ``\Sigma_d`` the norm of the macroscopic stress
deviator and ``\varphi`` the porosity. For spherical pores and the Mori–Tanaka
scheme it becomes the ellipse
``\tfrac32\,\varphi\,\Sigma_m^2 + (1 + \tfrac23\varphi)\,\Sigma_d^2 = (1-\varphi)^2\sigma_o^2``.
The two routes compute the same domain by different means: the tutorial from two
derivatives of the linear moduli, `NonlinearHomogenized` by solving the secant
fixed point at a given strain. Running the phase law brings what the closed
construction does not: the direction of flow at each point of the domain, the
whole loading path, and any phase law in place of the rigid–plastic one.

## Normality

At a point ``\boldsymbol{\Sigma}`` of the boundary, the macroscopic strain that
produced it is along the outward normal to the domain: the macroscopic flow
obeys the normality rule. The modified secant method coincides with the
variational estimate of [ponteCastaneda1991](@citet) for these phases
[suquet1997](@cite), whose macroscopic potential, in the rigid–plastic limit, is
homogeneous of degree one in the strain; its gradient is a stress of the
boundary, normal to the direction of the strain. The tests of the package check
the normality on the ellipse.

## Numerical notes

| | |
|:--|:--|
| incompressibility | the bulk modulus of the solid must be large against its secant shear modulus, ``\sigma_o/(2\varepsilon_d)``; the error on the domain is of the order of their ratio |
| conditioning | the residual of the fixed point is then accurate to about ``\epsilon\,k/\mu^{s}``, with ``\epsilon`` the machine precision: the tolerance of [`LocalNewton`](@ref) must stay above it (`reltol = 1e-8` for a ratio of ``10^7``) |
| iterative schemes | with a self-consistent scheme, the tolerance of the local Newton iteration must also stay above that of the scheme's own fixed point |
| scope | a von Mises solid. A frictional (Drucker–Prager) solid also needs the second moment of the mean strain, as in [maghous2009](@citet); it is not covered |
