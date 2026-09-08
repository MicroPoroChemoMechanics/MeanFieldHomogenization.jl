# [Manual — reading path](@id man-index)

The Manual answers *how do I do this*. It says what to call, what the arguments
mean, and where each choice bites. The **why** is in
[Theory — reading path](@ref th-index), and the exhaustive
signatures are in the API section, which the pages here link into rather than
repeat.

If you have never used the package, [Installation](@ref man-installation) and
then [Schemes and RVEs](@ref man-schemes) are the two pages that make everything
else readable.

## Inclusions: which page for which morphology

Everything a scheme needs from a phase is one response tensor. What differs
between the pages below is *how* that tensor is obtained — a closed form, a
recurrence, a mesh, a network — and every one of them ends up in the same
`add_phase!` call.

| Your morphology | Page | How the response is obtained |
|:--|:--|:--|
| an ellipsoid, or one of its limits | [Ellipsoidal inclusions](@ref man-ellipsoidal-inclusions) | Eshelby's closed form |
| a fiber, a cylinder | [Cylindrical inclusions](@ref man-cylindrical-inclusions) | the same, in the cylindrical limit |
| a crack, a family of cracks | [Cracks](@ref man-cracks) | the crack-opening tensor and its algebra |
| a coated particle, an interphase | [Layered inclusions](@ref man-layered) | a recurrence over the layers |
| anything else, from your own code | [Custom inclusions](@ref man-custom-inclusions) | you supply it, through one of three gates |
| a shape with no closed form | [Finite-element inclusions](@ref man-fe-inclusions) | one Eshelby cell, solved |
| the same, but cheap and differentiable | [Neural-surrogate inclusions](@ref man-neural-inclusions) | a network trained on it |

[The inclusion gallery](@ref man-inclusion-gallery) draws them, if you would
rather recognize a shape than read a table.

## The rest

- **[Schemes and RVEs](@ref man-schemes)** — building a cell, choosing an
  estimate, and what each scheme assumes. Then
  [particle assemblies](@ref man-assemblies) for a population rather
  than a phase, and [multiscale](@ref man-multiscale) for a cell inside a cell.
- **[Laminates](@ref man-laminates)** — periodic homogenization, exact rather
  than estimated, and a different construction from everything above.
- **Beyond elasticity** — [conduction](@ref man-conductivity),
  [viscoelasticity](@ref man-viscoelasticity) with its
  [rheological models](@ref man-rheological-models) and
  [Laplace–Carson inversion](@ref man-laplace-inversion), and
  [poromechanics](@ref manual-poromechanics).
- **[Sensitivities](@ref man-sensitivities)** — differentiating an effective
  property with respect to a modulus, a fraction or a shape parameter, and which
  inclusion types can and cannot serve that.

Finite elements appear twice in this package and the two are opposite:
[finite-element inclusions](@ref man-fe-inclusions) are a solver called *inside*
homogenization, while [finite-element coupling](@ref fe-coupling) is
homogenization called inside a solver, as a constitutive law at a Gauss point.
