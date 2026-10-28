# [Homogenizing nonlinear phases: secant methods](@id mp-theory-secant)

The laws of this chapter so far describe one material. This page turns to a
composite whose phases are each described by such a law, and asks for the law
of the composite. When the phases are linear, the schemes of the package
answer exactly within their assumptions: the effective stiffness
``\mathbb{C}^{\mathrm{hom}}`` follows from the stiffnesses of the phases, their
fractions and their shapes. When a phase is nonlinear, its stiffness changes
from point to point with the local strain, and no scheme applies as it stands.
The secant methods bring the problem back to a sequence of linear ones. This
page derives them; [the example](@ref mp-ex-composite) runs them.

## The phases

Each phase ``i``, of volume fraction ``f_i``, is a nonlinear elastic material
with no internal variable: its stress depends on the current strain only. It is
isotropic, so that its energy depends on the strain through two invariants
only, the volume change ``\operatorname{tr}\boldsymbol{\varepsilon}`` and the
equivalent strain
``\varepsilon_{\mathrm{eq}} = \sqrt{\tfrac23\,\boldsymbol{e}:\boldsymbol{e}}``,
``\boldsymbol{e}`` being the deviator of ``\boldsymbol{\varepsilon}``. The
power-law material ([`PowerLawElasticity`](@ref)) is the standard example, a
model of the deformation theory of plasticity:

```math
\boldsymbol{\sigma} = k_i\,(\operatorname{tr}\boldsymbol{\varepsilon})\,\boldsymbol{1}
+ \frac{2}{3}\,\frac{\sigma_{\mathrm{eq}}(\varepsilon_{\mathrm{eq}})}{\varepsilon_{\mathrm{eq}}}\,\boldsymbol{e},
\qquad
\sigma_{\mathrm{eq}}(\varepsilon_{\mathrm{eq}}) = \sigma_0\left(\frac{\varepsilon_{\mathrm{eq}}}{\varepsilon_0}\right)^{m}.
```

## Every such law is linear at a frozen strain

The stress above is a linear function of the strain whose coefficients are
evaluated at the strain itself. Writing the isotropic stiffness with the
projectors ``\mathbb{J}`` on volume changes and ``\mathbb{K}`` on deviators,

```math
\boldsymbol{\sigma} = \mathbb{C}^{s}_i(\varepsilon_{\mathrm{eq}}):\boldsymbol{\varepsilon},
\qquad
\mathbb{C}^{s}_i(\varepsilon_{\mathrm{eq}}) = 3k_i\,\mathbb{J} + 2\mu^{s}_i(\varepsilon_{\mathrm{eq}})\,\mathbb{K},
\qquad
\mu^{s}_i(\varepsilon_{\mathrm{eq}}) = \frac{\sigma_{\mathrm{eq}}(\varepsilon_{\mathrm{eq}})}{3\,\varepsilon_{\mathrm{eq}}}.
```

``\mu^{s}_i`` is the **secant shear modulus**: the slope of the straight line
from the origin to the current point of the shear curve, not the slope of the
curve (the tangent modulus). For a law that is linear in the volume change,
``k_i`` is a constant. [`secant_moduli`](@ref) reads both moduli on any law of
the strain alone, from its response to a deviatoric strain and to a volume
change. It needs no formula from the law.

This rewriting is exact, but it does not yet help: in a composite the strain
varies inside each phase, so the secant stiffness varies too, and the phase is
not a homogeneous linear material.

## The linear comparison composite

The approximation of every secant method is to give the whole phase ``i`` a
single secant stiffness, evaluated at one **effective strain**
``\bar\varepsilon_i`` that represents the strain field of the phase:

```math
\mathbb{C}_i = \mathbb{C}^{s}_i(\bar\varepsilon_i).
```

The phases are then homogeneous and linear: they form a *linear comparison
composite*, which any linear scheme homogenizes into
``\mathbb{C}^{\mathrm{hom}}(\bar\varepsilon_1, \ldots, \bar\varepsilon_n)``. The
macroscopic stress at the macroscopic strain ``\boldsymbol{E}`` is

```math
\boldsymbol{\Sigma} = \mathbb{C}^{\mathrm{hom}}(\bar\varepsilon_1, \ldots, \bar\varepsilon_n):\boldsymbol{E}.
```

The effective strains are not known beforehand. They follow from the strain
field of the comparison composite, which depends on them. They are therefore
the solution of a **fixed point**:

```math
\bar\varepsilon_i = \mathcal{M}_i\big(\boldsymbol{E};\ \bar\varepsilon_1, \ldots, \bar\varepsilon_n\big),
\qquad i = 1, \ldots, n,
```

where ``\mathcal{M}_i`` is the measure of the strain of phase ``i`` in the
comparison composite. The choice of that measure is what distinguishes one
secant method from another.

## The classical secant method: the average strain

The simplest measure takes the equivalent strain of the **average** strain of
the phase. A linear scheme gives that average through the strain
concentration tensor ``\mathbb{A}_i`` of the phase,
``\langle\boldsymbol{\varepsilon}\rangle_i = \mathbb{A}_i:\boldsymbol{E}``, so that

```math
\bar\varepsilon_i = \varepsilon_{\mathrm{eq}}\big(\mathbb{A}_i:\boldsymbol{E}\big).
```

This is [`ClassicalSecant`](@ref). The average hides the fluctuations of the
strain inside the phase: a strain that is large in some regions and small in
others can average to a small value. In a matrix that wraps around stiff
particles, for example, the strain concentrates near the particles, and the
classical measure underestimates the strain that sets the secant modulus.

## The modified secant method: the second moment

A measure that sees the fluctuations is the **second moment** of the
equivalent strain over the phase,
``\bar\varepsilon_i^2 = \langle\varepsilon_{\mathrm{eq}}^2\rangle_i``. A linear
scheme does not give second moments directly, but they follow from the energy
of the comparison composite, as the following argument shows.

The energy stored per unit volume in the comparison composite under the
macroscopic strain ``\boldsymbol{E}`` is, by the lemma of Hill, both the
macroscopic energy and the average of the local energies:

```math
W = \frac12\,\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}
= \sum_j f_j\,\Big\langle \frac12\,\boldsymbol{\varepsilon}:\mathbb{C}_j:\boldsymbol{\varepsilon} \Big\rangle_j .
```

The local strain field minimizes the energy among the kinematically admissible
fields. Change the shear modulus ``\mu_i`` of phase ``i`` alone. The energy
changes for two reasons: the explicit change of ``\mathbb{C}_i``, and the change
of the strain field. The second contributes nothing to first order, because
the field is a minimum. Only the explicit dependence remains, and
``\partial\mathbb{C}_i/\partial(2\mu_i) = \mathbb{K}`` acts only on phase ``i``:

```math
\frac{\partial W}{\partial(2\mu_i)}
= f_i\,\Big\langle \frac12\,\boldsymbol{\varepsilon}:\mathbb{K}:\boldsymbol{\varepsilon} \Big\rangle_i
= \frac{f_i}{2}\,\langle\boldsymbol{e}:\boldsymbol{e}\rangle_i
= \frac{3 f_i}{4}\,\langle\varepsilon_{\mathrm{eq}}^2\rangle_i .
```

With ``W`` written as the macroscopic quadratic form, this gives the second
moment from the derivative of the effective stiffness alone:

```math
\bar\varepsilon_i^2 = \langle\varepsilon_{\mathrm{eq}}^2\rangle_i
= \frac{2}{3 f_i}\;\boldsymbol{E}:\frac{\partial\mathbb{C}^{\mathrm{hom}}}{\partial(2\mu_i)}:\boldsymbol{E}.
```

This identity between the energy stored in a phase and the derivative of the
effective moduli is due to [kreher1990](@citet). It is exact for a scheme that
is the exact solution of some microstructure, and it serves as the definition
of the second moment for a scheme used as an estimate. This is
[`ModifiedSecant`](@ref), the default. The package computes the derivative of
``\mathbb{C}^{\mathrm{hom}}`` by automatic differentiation through the scheme:
nothing specific to a scheme is written.

!!! note "The second moment is never below the first"
    The squared equivalent strain is a convex quadratic function of the strain,
    so by Jensen's inequality
    ``\langle\varepsilon_{\mathrm{eq}}^2\rangle_i \ge \varepsilon_{\mathrm{eq}}(\langle\boldsymbol{\varepsilon}\rangle_i)^2``,
    with equality only for a strain uniform in the phase. For the same
    comparison composite the modified effective strain is therefore at least
    the classical one. A phase whose secant modulus decreases with the strain
    (``m < 1``) is then softer under the modified method, and the composite is
    expected to be softer as well, which [the example](@ref mp-ex-composite)
    shows.

The modified method has a further standing. For phases whose energy is a
concave function of ``\varepsilon_{\mathrm{eq}}^2``, as for the power law with
``m \le 1``, the variational procedure of [ponteCastaneda1991](@citet) reaches
the same equations. That procedure expresses the effective energy of the
nonlinear composite through an optimization over linear comparison composites,
and gives a bound when the linear estimate it uses is itself a bound. The review of
[suquet1997](@citet) sets out this connection and the other secant and
variational estimates.

## The composite as a law of `MaterialPoints`

The fixed point has one unknown per nonlinear phase. Written as a residual,

```math
R_i\big(\bar\varepsilon_1, \ldots, \bar\varepsilon_n;\ \boldsymbol{E}\big)
= \bar\varepsilon_i - \mathcal{M}_i\big(\boldsymbol{E};\ \bar\varepsilon_1, \ldots, \bar\varepsilon_n\big) = 0,
```

it has exactly the form of the [local problem of a law](@ref mp-theory): the
effective strains play the role of the internal variables, the macroscopic
strain that of the gradient, and the macroscopic stress that of the flux.
[`NonlinearHomogenized`](@ref) builds that law. Everything the chapter says
about a law then applies to the composite:

- the fixed point is solved by Newton's method, with the Jacobian
  ``\partial R_i/\partial\bar\varepsilon_j`` obtained by differentiating the
  scheme itself;
- the **consistent tangent** of the composite,
  ``\mathrm{d}\boldsymbol{\Sigma}/\mathrm{d}\boldsymbol{E}``, comes from the
  implicit-function theorem. It contains the variation of the effective
  strains with ``\boldsymbol{E}``, which a tangent read on the frozen
  comparison composite (``\mathbb{C}^{\mathrm{hom}}`` alone) would miss;
- the composite can be driven along any path by [`point_response`](@ref), and it
  can be used at the quadrature points of a [finite-element computation](@ref fe-quickstart).

Because the phases have no internal variable, the composite has no history
either. The effective strains are recomputed at every step, the previous ones
serving only as the starting point of the Newton iteration.

## Scope

| | |
|:--|:--|
| phases | isotropic, laws of the strain alone (nonlinear elasticity, deformation theory); a phase without a law keeps its linear stiffness |
| schemes | any linear scheme of the package, with its own assumptions on the microstructure |
| effective strain | one per nonlinear phase: the deviatoric response is secant, the bulk response is read at the same strain size |
| not covered | phases with internal variables (plasticity by increments, viscosity), for which the [incremental approach](@ref mp-theory-incremental-tangent) is needed; for phases without, the [affine formulation](@ref mp-theory-affine) linearizes by the tangent instead of the secant |
