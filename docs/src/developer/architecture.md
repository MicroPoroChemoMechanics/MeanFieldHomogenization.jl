# [Architecture](@id dev-architecture)

`MeanFieldHomogenization` is organized around a single principle:

> every high-level entry point dispatches via
> `MeanFieldHomogenization.Core._resolve_algo` on
> `(Val(method), inclusion, C₀)`.

The resolved [`AbstractAlgorithm`](@ref) instance is then passed to an
internal `_kernel` method table maintained by each sub-module.
Sub-modules may *extend* (but not redefine) both `_resolve_algo` and
`_kernel`.

## Cells: what `homogenize` accepts

`RVE` is no longer the only container. Everything `homogenize` accepts is an
[`AbstractHomogenizationCell`](@ref MeanFieldHomogenization.Core.AbstractHomogenizationCell),
declared in `src/Core/cells.jl` alongside the rest of the package's generics:

| Cell | Morphology | Solved by |
| :--- | :--- | :--- |
| `Schemes.RVE` | random, through the Eshelby auxiliary problem | every mean-field scheme |
| `Laminates.Laminate` | periodic stack of parallel layers | `Laminated` (exact), `Voigt`, `Reuss` |
| `Assemblies.ParticleAssembly` | individual particles at **known positions** | `ClusterModel`, `EquivalentInclusion`, and the bounds |

A cell carries exactly the information its schemes need, which is why there are
three of them rather than one wide type: adding positions to `RVE` would put a
field on every phase of every one-site model that no one-site model can read.
The N-body schemes are typed on `ParticleAssembly` for the same reason the
mean-field kernels stay typed on `RVE` — a mis-applied scheme then reports
itself instead of silently ignoring data.

Only the **entry points** are typed on the supertype (`homogenize`,
`get_param`/`set_param`, `derivative`/`gradient`/`jacobian`, and the
`_evaluate` fallback). Every scheme kernel and every `_phase_*` helper stays
typed on `RVE`. That split is deliberate: it is what makes the abstraction
regression-free, and what makes a mis-applied scheme report itself.

`src/Core/cells.jl` also carries the **declarative multiscale** seam. A
property value may be a `Homogenized(cell, scheme)`, resolved lazily by
`resolve_property` — called from exactly two places, `phase_property` and
`layer_property` — and memoized per `(cell, key)` for the duration of one
`homogenize` call by a task-local `ScopedValue`. Two invariants to preserve
when touching it:

- **type inspections must go through the raw accessors**
  (`phase_property_raw`, `cell_container_property`); resolving would run a full
  inner homogenization just to look at a type;
- the cache scope must span the whole `_evaluate`, iterative solvers included,
  or a self-consistent loop would re-solve every nested cell once per
  iteration.


## What each sub-module is for

| Module | Responsibility |
| :--- | :--- |
| [`MeanFieldHomogenization.Elliptic`](@ref) | Type-generic Legendre and Carlson elliptic integrals (`ForwardDiff`/`Sym` compatible). |
| [`MeanFieldHomogenization.Core`](@ref) | Abstractions, traits, shared numerics (Green/Newton kernels, Masson residue algorithm, DECUHR seam). |
| [`MeanFieldHomogenization.Elasticity`](@ref) | Hill polarization tensor for ellipsoidal inclusions and infinite cylinders (2D/3D, iso/aniso/TI-coaxial). |
| [`MeanFieldHomogenization.Cracks`](@ref) | Crack-opening-displacement (COD) tensor, compliance contribution, stress/displacement intensity factors. |
| [`MeanFieldHomogenization.Conductivity`](@ref) | Second-order Hill tensor for transport problems (diffusion, conduction, Darcy flow), closed form for any matrix anisotropy. |
| [`MeanFieldHomogenization.LayeredSpheres`](@ref) | `n`-layer composite spheres (Hervé–Zaoui, Christensen–Lo), five interface types, volume-average and pointwise localization. |
| [`MeanFieldHomogenization.LayeredSpheroids`](@ref) | `n`-layer confocal spheroids, conduction, Kapitza / surface-conductive interfaces, series or quadrature evaluation. |
| [`MeanFieldHomogenization.Laminates`](@ref) | Periodic multilayer cell: no matrix, no Eshelby problem — an *exact* solution in elasticity and transport, same imperfect interfaces, per-layer localization. |
| [`MeanFieldHomogenization.Schemes`](@ref) | RVE container and `homogenize`; Voigt, Reuss, Dilute, Mori–Tanaka, Maxwell, PCW, self-consistent, asymmetric SC, differential, cluster model, equivalent inclusion; exact vs. best-fit symmetrization; `ForwardDiff` sensitivities. |
| [`MeanFieldHomogenization.Interactions`](@ref) | Two-inclusion interaction tensor ``\mathbb{T}^{ab}`` and the Green operator of the reference — closed forms for balls and disks, cubature for the anisotropic case. |
| [`MeanFieldHomogenization.Assemblies`](@ref) | `ParticleAssembly`: the cell that carries positions, its generators and boundary treatments — what the two N-body schemes act on. |
| [`MeanFieldHomogenization.Poromechanics`](@ref) | Biot coefficient tensor and skeleton modulus of a porous or cracked microstructure, for saturated and drained responses. |
| [`MeanFieldHomogenization.Constitutive`](@ref) | The Gauss-point contract: a microstructure exposed to a finite-element code as a material law returning stress, tangent and updated state. |
| [`MeanFieldHomogenization.Viscoelasticity`](@ref) | Ageing linear viscoelasticity via Volterra operators — every scheme, cracks and layered spheres included. |
| [`MeanFieldHomogenization.CustomInclusions`](@ref) | The user-defined inclusion contract: `CustomInclusion` (callback-driven) and `check_inclusion_interface`. |
| [`MeanFieldHomogenization.FiniteElements`](@ref) | Inclusions whose response comes out of a finite-element solve of the Eshelby problem, behind a backend contract (`Ferrite` or `Gridap`). |
| [`MeanFieldHomogenization.NeuralInclusions`](@ref) | Inclusions whose response comes out of a trained network, with the sampling and fitting machinery; differentiable in the morphology. |

The table below says the same thing from the other side: not what a
sub-module is for, but what it puts in the namespace.

## What each sub-module exports

| Sub-module         | Exports                                                                                            |
| :----------------- | :------------------------------------------------------------------------------------------------- |
| `Elliptic`         | type-generic elliptic integrals (`ell_K`, `ell_E`, `ell_F`, `ell_RF`, `ell_RD`)                     |
| `Core`             | abstractions, traits, `_resolve_algo`, Newton potentials, Green kernel helpers, Kelvin dipole field, exact ISO/TI rotation averages, moduli extractors |
| `Elasticity`       | `Ellipsoid`, `Cylinder`, auxiliary tensors, `hill_tensor` + 3D/2D kernels                           |
| `Cracks`           | `EllipticCrack`, `RibbonCrack`, `cod_tensor`, `sif`, `dif`, and the crack methods of the contribution generics |
| `Conductivity`     | additional `_kernel` methods for 2nd-order transport tensors                                        |
| `LayeredSpheres`   | `LayeredSphere`, Hervé-Zaoui recurrences, five interface types, localization fields                 |
| `LayeredSpheroids` | `LayeredSpheroid` (confocal, conduction)                                                            |
| `Laminates`        | `Laminate`, the interface algebra, per-layer localization, the ALV twin                              |
| `Interactions`     | Green operator of the reference and the two-inclusion interaction tensor (closed forms, cubature, periodic sums) |
| `Assemblies`       | `ParticleAssembly`, its generators and boundary treatments, the two N-body schemes                  |
| `Schemes`          | `RVE`/`Phase`, `homogenize`, every scheme type, exact symmetrization, ForwardDiff sensitivities     |
| `Poromechanics`    | Biot coefficient tensor and skeleton modulus of a porous or cracked cell                            |
| `Viscoelasticity`  | ageing linear viscoelasticity (Volterra pipeline, ALV variant of every scheme)                      |
| `CustomInclusions` | the user-defined inclusion contract: `CustomInclusion`, `check_inclusion_interface`                 |
| `FiniteElements`   | inclusions solved by finite elements (`FEEllipticCrack`, `FEExcenteredSphere`); the physics lives here, the discretization in a backend extension (`MeanFieldHomogenizationFerriteExt`, `MeanFieldHomogenizationGridapExt`) |
| `NeuralInclusions` | inclusions answered by a trained network, with the sampling/fitting machinery; the training optimizer lives in `MeanFieldHomogenizationLuxExt` |
| `Constitutive`     | the Gauss-point contract of the [finite-element coupling](@ref fe-coupling): `material_response`, the shipped materials, the state types; the Ferrite glue in `MeanFieldHomogenizationFerriteMaterialExt` |

Two files sit at the **top level** rather than in a sub-module, and are loaded
after every geometry sub-module on purpose: `localization.jl` and
`contribution.jl` implement generics declared in `Core` whose methods need
every sub-module's `_kernel` table to be visible. `CustomInclusions`,
`FiniteElements` and `NeuralInclusions` are included after *those*, because
their fallbacks `invoke` the generic methods defined there.

## Extension points

Every generic an inclusion may implement is declared — as a bodyless `function`
— in `Core/abstractions.jl`, so that sub-modules **and user code outside the
package** attach their methods to one canonical function. Together with the
open `_resolve_algo` / `_kernel` tables and the neutral
[`AbstractCustomInclusion`](@ref) branch, that is the whole extension surface.
See [Adding a new inclusion](@ref dev-adding-inclusion), [Adding a new algorithm](@ref dev-adding-algorithm) and
[Adding a homogenization scheme](@ref dev-adding-scheme).
