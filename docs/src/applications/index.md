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

The chapter starts with general concepts, which hold whatever the material, then
turns to the materials, one section each. Within a section a page comes after
the pages it builds on.

## General concepts

| Page | What it models |
| :--- | :--- |
| [The cluster model on cubic arrays](@ref app-cluster-model) | inclusions that see where their neighbors are: the pairwise interactions within a cluster, on periodic arrays |
| [The equivalent inclusion method, against a published table](@ref app-eim-assembly) | the Galerkin form of the Lippmann–Schwinger equation on a planar assembly of disks, against a published table |
| [A recycled-concrete aggregate, by axisymmetric Fourier elements](@ref app-recycled-aggregate) | an old aggregate wrapped in a shell of adhered mortar, off center, solved by finite elements and checked against the concentric layered sphere |
| [Concave pores: superspheres and superspheroids](@ref app-concave-pores) | pores with no Eshelby solution, computed by finite elements and set against the only published data for this shape family |

## Cementitious materials

| Page | What it models |
| :--- | :--- |
| [Multiscale elasticity of a hydrating cement paste](@ref app-cement-paste) | the Young modulus of a Portland cement paste along its hydration, over two scales |
| [A hydrating blended cement paste, coupled to its chemistry](@ref app-blended-hydration) | the volume fractions computed by the chemistry of hydration instead of a Powers-type correlation, then the elasticity of a blended paste |
| [Hydration through the pore solution](@ref app-ionic-hydration) | the same paste, its hydration driven through the ionic species of the pore solution rather than stated solid reactions |
| [Cement paste: chloride diffusivity and elasticity](@ref app-cement-paste-diffusion) | one multiscale model that predicts both the chloride diffusivity and the elastic moduli |
| [Aging creep of solidifying cementitious materials](@ref app-aging-creep) | a phase that solidifies progressively, as C-S-H does, and the aging relaxation kernel that results |
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

The finite elements met in the first section are a solver called *inside*
homogenization, for one inclusion; the opposite use, homogenization called
inside a structural solver as the constitutive law at each Gauss point, is the
subject of [Finite-element coupling](@ref fe-coupling).
