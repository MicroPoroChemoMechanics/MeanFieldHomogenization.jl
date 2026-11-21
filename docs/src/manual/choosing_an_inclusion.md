# [Choosing an inclusion](@id man-choosing-inclusion)

!!! info "Before this page"
    [Concentration and contribution tensors](@ref th-concentration), which
    defines the tensors a scheme reads from a phase, and
    [The inclusion zoo](@ref man-inclusion-gallery), which draws the shapes.

A scheme reads one thing from a phase: the response of a single inclusion
embedded in the reference medium. This page compares the ways the package
obtains that response, so that the route can be chosen from what is known of
the morphology and from what is needed downstream: an anisotropic reference
medium, a derivative, a given accuracy.

## Three levels of entry

The response is supplied at one of three levels, and the package derives
everything above the level supplied
([Custom inclusions](@ref man-custom-inclusions) gives the functions):

| Level | Supplied | Exists when |
| :--- | :--- | :--- |
| A | the Hill tensor ``\mathbb P`` | the field inside the inclusion is uniform: an ellipsoid and its limits ([The Eshelby inclusion problem](@ref th-eshelby-problem)) |
| B | the two concentration tensors, ``\mathbb A_{\varepsilon\varepsilon}`` and ``\mathbb A_{\sigma\varepsilon}`` | always; it is the only level of a heterogeneous inclusion or of a shape that is not an ellipsoid |
| C | the contribution tensors ``\mathbb N`` and ``\mathbb H`` | always; it is what a published table of a shape usually gives |

Levels A and B serve every scheme. Level C does not give the concentration
tensor, so a phase measured by a volume fraction and entered at level C is
refused by `MoriTanaka` and by the self-consistent schemes; a phase measured by
a density (a crack) is not concerned.

## The routes

Each route below ends in the same `add_phase!` call. They differ by how the
response is computed, and therefore by what they allow.

| Route | Types | Level | Response obtained by | Page |
| :--- | :--- | :--- | :--- | :--- |
| ellipsoid and its limits | [`Ellipsoid`](@ref), `Cylinder` | A | closed form, or a quadrature for a general anisotropy | [Ellipsoidal inclusions](@ref man-ellipsoidal-inclusions), [Cylindrical inclusions](@ref man-cylindrical-inclusions) |
| elliptical and ribbon cracks | [`EllipticCrack`](@ref), `RibbonCrack` | density | the crack-opening tensor | [Cracks](@ref man-cracks) |
| layered patterns | `LayeredSphere`, `LayeredCylinder`, `LayeredSpheroid` | B | exact recurrence over the layers, or a series of harmonics | [Layered inclusions](@ref man-layered) |
| toroidal pore (conduction) | [`InsulatingTorus`](@ref) | B | series of toroidal harmonics, exact to rounding | [Toroidal inclusions](@ref man-toroidal-inclusions) |
| prismatic cavity of any section given by a map | [`ConformalHole`](@ref) | B | complex potentials on a conformal map, exact | [Conformal holes](@ref man-conformal-holes) |
| your own formula, table or solver | [`CustomInclusion`](@ref MeanFieldHomogenization.CustomInclusion), or a subtype of [`AbstractCustomInclusion`](@ref) | A, B or C | what you supply | [Custom inclusions](@ref man-custom-inclusions) |
| flat crack of any contour | [`PlanarCrack`](@ref) | density | integral equation of the crack on a grid of Gaussians | [Planar cracks of any contour](@ref man-planar-cracks) |
| any region of space | [`GaussianInclusion`](@ref) | B | integral equation of the heterogeneity on a grid of Gaussians | [Heterogeneities of any shape](@ref man-gaussian-inclusions) |
| a meshed cell | the `FE…` types | density, or B | one inclusion in a finite cell, solved by finite elements | [Finite-element inclusions](@ref man-fe-inclusions) |
| a trained network | `NeuralHillInclusion`, `NeuralConcentrationInclusion` | A or B | evaluation of a network trained on one of the routes above | [Neural-surrogate inclusions](@ref man-neural-inclusions) |

The last four are themselves custom inclusions: they enter the schemes through
the interface of [Custom inclusions](@ref man-custom-inclusions), and a
morphology of your own enters the same way.

## What each route allows

| Route | Reference medium | Derivatives | Accuracy | Cost |
| :--- | :--- | :--- | :--- | :--- |
| ellipsoid and its limits | any anisotropy | shape and properties | exact | none |
| elliptical and ribbon cracks | any anisotropy | shape and properties | exact | none |
| layered patterns | isotropic; transversely isotropic about the axis for the cylinder | shape and properties | exact | none |
| toroidal pore (conduction) | isotropic | shape | exact | none |
| prismatic cavity given by a map | isotropic | shape and properties | exact for the map given | none |
| your own formula, table or solver | what your code accepts | those your code propagates | yours | yours |
| flat crack of any contour | any anisotropy | properties; the contour is refused | controlled by the grid, with an extrapolation between two grids | one solve per reference medium |
| any region of space | isotropic | properties; the shape is refused | a few percent for a convex shape, more for a concave one | seconds in conduction, tens of seconds in elasticity |
| a meshed cell | isotropic | refused | that of the mesh, reported by the inclusion | one factorization per reference medium, kept |
| a trained network | isotropic | shape and properties | that of the fit, stored with the model | none |

Three consequences of this table.

**An isotropic reference medium restricts the iterative schemes.** A
self-consistent or a differential scheme evaluates the inclusion in its current
estimate, which is anisotropic as soon as the inclusions are aligned. A route
limited to an isotropic reference then needs `symmetrize = IsoSymmetrize()` on
the phase; the one-shot schemes (`Dilute`, `MoriTanaka`, `Maxwell`) are not
concerned.

**A derivative with respect to the shape excludes the grid and the mesh.** The
nodes inside a contour or the elements of a mesh do not vary continuously with
a shape parameter, and both routes refuse the request. The closed forms, the
layered patterns and the networks accept it
([Sensitivities](@ref man-sensitivities)).

**An exact route is preferred whenever it exists.** A grid, a mesh or a network
is for a morphology that has no closed form. The accuracy columns of
[Heterogeneities of any shape](@ref man-gaussian-inclusions),
[Finite-element inclusions](@ref man-fe-inclusions) and
[The models shipped today](@ref man-neural-models) measure each of them against
an ellipsoid, where the exact answer is known.

## Which route for which situation

| Situation | Route |
| :--- | :--- |
| the shape is an ellipsoid, a cylinder, an elliptical or a ribbon crack | the closed form |
| a coated particle with concentric, coaxial or confocal layers | a layered pattern |
| a ring-shaped pore, in conduction | `InsulatingTorus` |
| a prismatic pore or channel whose section has corners or lobes | `ConformalHole` |
| the contribution tensors of the shape are published, or a formula is known | `CustomInclusion`, at the level the source gives |
| the response comes from a solver of your own | a subtype of `AbstractCustomInclusion` |
| a flat crack that is not an ellipse, or whose faces touch | `PlanarCrack` |
| a shape given by a function, by voxels or by set operations | `GaussianInclusion` |
| a shape for which a few percent is not enough | a finite-element inclusion |
| the same shape evaluated many times, or differentiated | a neural surrogate |

A morphology used once in a study is best entered with the callbacks of
`CustomInclusion`. One that is reused, and whose parameters are to be
differentiated, deserves a type: the fields of a subtype of
`AbstractCustomInclusion` are reached by `derivative` with no extra code
([Adding a new inclusion](@ref dev-adding-inclusion)).

## Where to go next

- [Custom inclusions](@ref man-custom-inclusions): the interface every route
  above goes through.
- [The custom-inclusion interface: three entry levels, one answer](@ref tut-custom-inclusion-contract):
  one morphology entered at each level.
- [A concave pore through every route](@ref app-pore-routes): one shape with
  no closed form, by a published table, a mesh, a network and a grid, with the
  deviations measured.
