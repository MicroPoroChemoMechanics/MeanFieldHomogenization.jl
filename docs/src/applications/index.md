# [Applications — reading path](@id app-index)

The Applications assemble the pieces of the Manual into complete micromechanical
models of real materials. Each one follows a published model, cited at the top
of its page, and sets the values the paper reports against those the library
computes. Where a tutorial isolates one feature of the API, an application
chains several — a microstructure over two or three scales, a chemistry that
supplies the volume fractions, a time-dependent behavior, a strength criterion —
and says where the model, rather than the library, sets the limit.

If the package is new to you, [Getting started](@ref getting-started) and
[Schemes and RVEs](@ref man-schemes) come first; every page below then opens
with the theory and manual pages it relies on.

The chapter starts with the sections that hold whatever the material (inclusions
of any shape, interactions, cracks, coated inclusions, transport), then turns to
the materials, one section each. Within a section a page comes after the pages
it builds on.

## Inclusions of any shape

One inclusion whose shape is not an ellipsoid, or whose Hill tensor is derived step by step.

| Page | What it models |
| :--- | :--- |
| [Hill tensors by transformation: closed forms derived step by step](@ref app-hill-transformation) | the Hill tensor of an ellipsoid as that of a sphere in a transformed medium, and the closed form it gives for a spheroid in a coaxial transversely isotropic matrix, derived symbolically and held to the numerical back-ends |
| [Concave pores: superspheres and superspheroids](@ref app-concave-pores) | pores with no Eshelby solution, computed by finite elements and set against the only published data for this shape family |
| [A concave pore through every route](@ref app-pore-routes) | one superspherical pore by a published table, a finite-element cell, a neural surrogate and a grid of Gaussians, with the deviations measured |
| [Ring-shaped pores in a conducting matrix](@ref app-insulating-torus) | the toroidal pore in conduction, by a series of toroidal harmonics, against a published study and spheroids of the same volume |
| [Prismatic pores with corners](@ref app-conformal-holes) | pores whose section is given by a conformal map, polygons included, against published compressibilities and shear compliances; perforated plates |
| [The elliptic hole, from an isotropic to an anisotropic plate](@ref app-elliptic-hole) | the stress along the edge of an elliptic hole in an isotropic plate and in an orthotropic plywood sheet, against the closed forms and the tables of the two classical books; the stress ahead of the tip; the same hole as an inclusion and as an insulator |

## Interactions between particles

Particles that see where their neighbors are.

| Page | What it models |
| :--- | :--- |
| [The cluster model on cubic arrays](@ref app-cluster-model) | inclusions that see where their neighbors are: the pairwise interactions within a cluster, on periodic arrays |
| [The equivalent inclusion method, against a published table](@ref app-eim-assembly) | the Galerkin form of the Lippmann–Schwinger equation on a planar assembly of disks, against a published table |
| [Arrays and clusters of spheres by the multipole expansion](@ref app-mem-spheres) | the whole field around each sphere, as a series of harmonics: cubic arrays and a finite cluster in conduction and in elasticity, against published tables |
| [Spheroids and ellipsoids by the multipole expansion](@ref app-mem-spheroids) | oblate spheroids on three proportional lattices with perfect and imperfect interfaces, spheroids bonded by springs, ellipsoids with a Kapitza interface, against the tables of Kushch |

## Cracks

Cracks whose faces touch, and narrow cracks of various contours.

| Page | What it models |
| :--- | :--- |
| [Cracks with contacts between their faces](@ref app-crack-contacts) | a circular crack whose faces touch on islands: one contact, regular patterns, clusters and random sets, by the Gaussian approximation of the crack equation, against a published study |
| [Narrow cracks: rings, rectangles, bands](@ref app-narrow-cracks) | the plane strain model of a narrow crack and its end effects, on rings, rectangles, ellipses and curved bands, against published fits, the annular crack and the closed form of the ellipse |

## Coated inclusions and interphases

A particle in a shell: uniform, graded, approximated by a closure, or aging.

| Page | What it models |
| :--- | :--- |
| [A recycled-concrete aggregate, by axisymmetric Fourier elements](@ref app-recycled-aggregate) | an old aggregate wrapped in a shell of adhered mortar, off center, solved by finite elements and checked against the concentric layered sphere |
| [A graded interphase around a spherical particle](@ref app-graded-interphase) | a zone whose moduli vary with the radius, as thin layers, against an exact series in elasticity and conduction |
| [Coated inclusions: closures on the mean strains, and the exact solution](@ref app-coated-estimates) | the published closures of the coated ellipsoid, each with its assumption, set beside the layered sphere and the confocal spheroid |
| [Coated spheres in an aging viscoelastic matrix](@ref app-aging-coated-spheres) | elastic spheres, coated or not, in a matrix that creeps and ages, by the Mori-Tanaka and generalized self-consistent schemes |

## Transport in pores

The Stokes flow in the pores and the permeability it gives.

| Page | What it models |
| :--- | :--- |
| [Stokes-to-Darcy permeability of porous media](@ref app-stokes-darcy) | the permeability of granular, fibrous, tubular and double-porosity media from the Stokes flow in their pores, through size-dependent Brinkman inclusions |

## Cementitious materials

| Page | What it models |
| :--- | :--- |
| [Multiscale elasticity of a hydrating cement paste](@ref app-cement-paste) | the Young modulus of a Portland cement paste along its hydration, over two scales |
| [A hydrating blended cement paste, coupled to its chemistry](@ref app-blended-hydration) | the volume fractions computed by the chemistry of hydration instead of a Powers-type correlation, then the elasticity of a blended paste |
| [Hydration through the pore solution](@ref app-ionic-hydration) | the same paste, its hydration driven through the ionic species of the pore solution rather than stated solid reactions |
| [Cement paste: chloride diffusivity and elasticity](@ref app-cement-paste-diffusion) | one multiscale model that predicts both the chloride diffusivity and the elastic moduli |
| [Aging creep of solidifying cementitious materials](@ref app-aging-creep) | a phase that solidifies progressively, as C-S-H does, and the aging relaxation kernel that results |
| [Aging basic creep of concrete: a CEM I cement, with and without silica fume](@ref app-aging-basic-creep) | the creep of a hydrating concrete from its composition, over four aging scales, against creep tests at several loading ages |
| [Quasi-brittle strength of cement paste and mortar](@ref app-strength) | the uniaxial compressive strength upscaled from the hydrate foam to the paste and the mortar |
| [The interfacial transition zone in concrete](@ref app-itz-concrete) | the more porous shell of paste around each aggregate, and the stiffness it costs the concrete |
| [The elastic limit of concrete: ITZ failure and ITZ–aggregate separation](@ref app-itz-elastic-limit) | where microcracks start around the aggregates: the stress in the ITZ from the Hadamard jump conditions, and the elastic limit surface |

## Geomaterials

| Page | What it models |
| :--- | :--- |
| [A lamellar porous material: swelling clays and C-S-H](@ref app-lamellar) | stacks of parallel platelets with water in between, over two scales chained on symbols |
| [Friction of a granular medium from its contacts](@ref app-granular-friction) | rigid grains bonded by springs, the macroscopic friction coefficient they give, and how it falls as the medium loosens |
| [The strength of a sandstone: crushing grains and sliding contacts](@ref app-sandstone-strength) | grains that yield and cemented contacts that slide, combined by the modified secant method |

## Bituminous materials

| Page | What it models |
| :--- | :--- |
| [Viscoelastic complex modulus of a bituminous mixture](@ref app-bituminous) | the complex modulus ``E^*(\omega)`` through three nested scales, a 2S2P1D bitumen among elastic mineral phases |

The finite elements met in the first sections are a solver called *inside*
homogenization, for one inclusion; the opposite use, homogenization called
inside a structural solver as the constitutive law at each Gauss point, is the
subject of [Finite-element coupling](@ref fe-coupling).
