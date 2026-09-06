# [Roadmap](@id dev-roadmap)

What is shipped, what is open, and — for the two areas where the boundary is
subtle — exactly which pieces of a cited paper are and are not implemented.

## Shipped

- Mean-field schemes: Voigt/Reuss bounds, dilute, Mori–Tanaka, Maxwell,
  Ponte-Castañeda-Willis, self-consistent (Anderson + Newton),
  asymmetric self-consistent, differential.
- **Periodic multilayer** (`Laminate` + the `Laminated` scheme): a matrix-free,
  deterministic cell with an *exact* solution, in elasticity and transport,
  with the four imperfect-interface models and an ageing-viscoelastic twin.
  Saturates Voigt in the plane of the layers and Reuss across them.
- **The homogenization-cell abstraction** (`AbstractHomogenizationCell`) and
  **declarative multiscale chaining** (`Homogenized`, `NestedParameter`): a
  multiscale model as one object, differentiable end to end in a single
  `ForwardDiff` pass, with a call-scoped memoization.
- Representative volume element (RVE) assembly and effective-property
  pipelines mirroring the reference C++ RVE assembly.
- Concentric multi-layer sphere (`LayeredSphere` via
  [`AbstractLayeredInclusion`](@ref)): Hervé-Zaoui bulk / shear /
  conductivity recurrences, five interface types (perfect, spring,
  membrane, Kapitza, surface-conductive), volume-average and pointwise
  localization fields.
- Ageing linear viscoelasticity (ALV): time-domain Volterra pipeline for
  every scheme, structured ISO/TI/ortho fast paths, ALV cracks and the
  ALV layered sphere (bulk **and** shear recurrences).
- Exact rotation-group symmetrization (ISO / TI) of concentration tensors,
  preserving non-major-symmetric content (`TensTI{4,T,8}`), for arbitrary
  multi-axis orientation distributions inside every scheme kernel.
- User-defined inclusions and algorithms: a leveled, documented contract
  ([Adding a new inclusion](@ref dev-adding-inclusion)), the neutral
  [`AbstractCustomInclusion`](@ref) branch, the callback-driven
  [`CustomInclusion`](@ref MeanFieldHomogenization.CustomInclusion), the
  [`check_inclusion_interface`](@ref MeanFieldHomogenization.check_inclusion_interface)
  conformance checker, and `shape_trait`-based inheritance of the crack
  algebra (a user crack needs only `cod_tensor`).
- Real-space Kelvin Green gradient and dipole far field for an isotropic
  matrix ([`green_gradient_iso`](@ref MeanFieldHomogenization.Core.green_gradient_iso),
  [`dipole_displacement_iso`](@ref MeanFieldHomogenization.Core.dipole_displacement_iso)) —
  the boundary correction that makes a finite numerical Eshelby cell behave
  like an infinite medium.
- ForwardDiff sensitivities across all elastic and ALV schemes (fractions,
  moduli, and inclusion geometry).
- NonlinearSolve.jl backend for the self-consistent fixed point
  (`MeanFieldHomogenizationNonlinearSolveExt`): any SciML algorithm
  (`NewtonRaphson`, `TrustRegion`, …) can solve `SelfConsistent` /
  `AsymmetricSelfConsistent`, through an implicit-function-theorem lift
  that keeps `derivative`/`gradient`/`jacobian` exact and free of nested
  `ForwardDiff.Dual`s regardless of algorithm.

## Open

- Extended-COD crack model in conduction: resistive cracks (linear-spring
  analog) **and** conductive cracks (elastic-membrane analog), via a
  tensorial conduction COD.
- Multi-layer extensions: coated cylinders, anisotropic per-layer moduli,
  excentered spheres.
- Laminate extensions: viscoelastic *interface* laws (the interface types
  carry `Number` fields today, so an ageing interface needs a different
  carrier), and `Homogenized` inside an ALV chain (the inner result would have
  to be re-expressible as a `ViscoLaw`).
- `PairwiseDistribution` (Willis 1982) envelope for the PCW scheme.
- Native Anderson acceleration with memory > 1, replacing the current
  `AndersonDefault` (currently Picard with relaxation, memory = 1).
- Optional structured `TensTI{4,T,8}` fast path for the ALV TI schemes.
- **Viscoelasticity in the Laplace-Carson domain — shipped in v0.6.0.** The
  non-ageing half of the viscoelasticity module: four numerical inverse-Laplace
  algorithms generic in the number type (so `ForwardDiff` traverses them, which
  `InverseLaplace.jl` cannot), a catalog of rheological models each exposing
  ``J(t)``, ``R(t)``, ``J^{*}(p)`` and ``R^{*}(p)``, the exact
  generalized-Kelvin ⇄ generalized-Maxwell conversion by root interlacing, and
  [`homogenize_lc`](@ref) tying them together. See
  [the theory](@ref th-laplace-carson), the
  [model manual](@ref man-rheological-models) and the
  [inversion manual](@ref man-laplace-inversion). Still open on that side:
  - **anisotropic tensor pairings.** Only [`IsoRheology`](@ref) exists; a
    transversely isotropic model would need six scalar channels and the
    corresponding `TensTI` assembly.
  - **ageing models in the catalog.** `LogarithmicCreep` is the non-ageing
    skeleton of a law that is normally written with age-dependent `E`, `C` and
    `τ`; expressing that family would need a second, two-argument interface.
- **Finite-element coupling, remaining pieces.** The Gauss-point contract,
  `HomogenizedElastic`, `MicrocrackedMaterial`, the Ferrite glue — including
  the coupled ``(\underline{u}, p)`` element — the poroelastic parameters, the
  fractured permeability, the `FracturedPoroelasticRock` material and the
  [ARMA 2011 well test](@ref fe-arma2011) are shipped (see
  [Finite-element coupling](@ref fe-coupling)). Still open:
  - the **consolidation column** of [barthelemyARMA2011](@cite) § 3.1, models
    M1/M2/M3 — the case where a family actually *closes* during the loading.
    Everything it needs is shipped; it is a driver, not a capability.
  - the paper's **self-consistent permeability**, whose matrix concentration factor
    (order-2 Hill tensor of the effective medium) the simpler estimate in
    `fracture_permeability` omits. It is worth ≈ 55× on the ARMA microstructure
    — the [well test](@ref fe-arma2011-scope) quantifies exactly what it costs.
  - drivers for Gridap, FEniCSx and an Abaqus-shaped UMAT.
- Finite-element inclusions, behind the `FEBackend` contract
  (`MeanFieldHomogenizationFerriteExt`, `MeanFieldHomogenizationGridapExt`), both with the
  first-order corrected boundary condition of
  [adessinaIJES2017](@cite) and an
  isotropic reference medium: the **elliptical crack** in 3-D tetrahedra
  (3 + 3 crack declination) and the **sphere with an off-center core** in
  axisymmetric Fourier elements (the general polarization fixed point).
  Open extensions — anisotropic reference medium (Pan-Chou or Barnett-Willis
  Green gradient); more than one inclusion, or a non-spherical envelope, in the
  axisymmetric cell.
- Neural-surrogate inclusions (`NeuralHillInclusion`,
  `NeuralLocalizationInclusion`), with the sampling, fitting and serialization
  machinery; the optimizer is the weak-dependency extension
  `MeanFieldHomogenizationLuxExt`, evaluation needs nothing extra. Four models ship,
  validated against the analytic ellipsoid. This is also the answer to
  "automatic differentiation through the solve", which the finite-element
  inclusions cannot offer: a surrogate *is* differentiable in the morphology.
  Open extensions — a surrogate trained on `fe_axi_localization` (gate B, the
  heterogeneous case the second type exists for); an anisotropic reference
  medium, which needs a feature set describing it.

## [The elastic layered spheroid — read this before starting](@id dev-elastic-spheroid)

The confocal multi-layer spheroid exists **in conduction only**. The harmonic
solution it rests on ([barthelemyBignonnetIJES2020](@cite)) is specific to the
scalar Laplace equation and does not carry over to the vector elastic problem.
Whoever writes the elastic counterpart will nonetheless reuse the *same*
spheroidal harmonics, and therefore inherits the numerical traps this module
already fell into. They cost real debugging, they are invisible to the
validations one naturally writes first, and they are collected here for that
reason.

### The route

Papkovich–Neuber displacement potentials reduce the Navier equation to
**harmonic** potentials, and harmonic functions separate in spheroidal
coordinates — so the machinery in `LayeredSpheroids/legendre.jl` is directly
reusable. That is exactly the route
[duanRSPA2005](@cite) takes for a spheroidal inhomogeneity **with an
interphase**: three fundamental solutions from Papkovich–Neuber potentials and
spheroidal harmonic expansions. It is the closest published starting point.

Two things to settle before writing code:

- **Fix the Papkovich–Neuber gauge.** The representation
  ``2μ \underline u = -\nabla(φ + \underline r·\underline ψ) + 4(1-ν)\underline ψ``
  is *not* unique — one component of ``\underline ψ`` can generally be dropped.
  Leave the redundancy in and the interface system is singular or, worse,
  merely ill-conditioned, which looks like a convergence problem rather than a
  modeling one.
- **Confocal or similar?** The conduction module stacks *confocal* surfaces,
  which is what makes the transfer clean. Check what the interface geometry in
  the elastic reference actually is before assuming the layer bookkeeping
  transfers; a similar-shape stack is a different problem.

An alternative worth weighing is the multipole route of
[kushch2013](@cite), which handles spheroids in elasticity without the
Papkovich–Neuber gauge question, at the price of its own machinery.

### The four traps, each of which shipped once

1. **Never run `Qₙ` upward.** `Qₙ` is the *minimal* solution of the Legendre
   three-term recurrence: upward, the seed's rounding error picks up the
   dominant `Pₙ` and is amplified by `ρ^{2n}`, `ρ = |x + √(x²−1)|`. Measured
   against the same recurrence at 600 bits, an upward `Q₁₅(5)` was wrong by a
   relative `1.3e13`. Use `legendre_odd`, which already chooses the direction
   from `ρ` — and note the choice runs **both ways**: Miller's downward
   recurrence is the wrong answer when `ρ ≈ 1` (a nearly degenerate spheroid),
   where it needs thousands of steps and upward loses nothing. Forcing Miller
   there returned a `Q` good to only `2e-7`.

2. **Impose regularity, never recover it.** In the matrix every growing
   amplitude above the degree carried by the remote field vanishes
   *identically*. Recomputing that block through the layer transfer leaves the
   linear solve's `O(1e-17)` residue, and `P_{2r-1}(q) ~ q^{2r-1}` amplifies it
   past `1e40` a few hundred radii out. Elasticity makes this **worse**, not
   better: Papkovich–Neuber carries a scalar *and* a vector potential, so there
   are several families of growing modes to zero out rather than one. The same
   applies at the core, where the singular amplitudes must be written down as
   exact zeros — see `_shear_amplitude_seq` in `LayeredSpheres`.

3. **An oblate spheroid carries a complex `q`.** Anything that tests, branches
   on, or divides by a coordinate must cope with `Complex`. Two live examples:
   a guard written `is_hard_numeric(typeof(q))` refuses every oblate particle,
   because it is `abs(q)` that is ordered; and `z / (Inf + 0im)` is
   `NaN + NaN im` where the real division would have given `0`, which turned
   every on-axis evaluation into a silent `NaN`.

4. **The chart is singular on the revolution axis.** At `|p| = 1`, `h_p` is
   infinite and `h_φ` vanishes. Every `0/0` the naive expressions carry there
   turned out to be *removable*, and removable exactly — no asymptotics — once
   the convention `P¹ₙ = -p̄ P′ₙ` is used to cancel the `p̄` of `h_φ` and the
   Legendre equation is used to eliminate `P″ₙ`. Expect the same in elasticity,
   and expect the same trap: the naive form returns a plausible-looking `NaN`
   or, worse, a finite number. **The check that catches a wrong removal is that
   the field on the axis must not depend on the azimuth `φ`**, which is
   undefined there — a wrong coefficient makes it depend on how the point was
   addressed.

### How to validate it

The conduction module was validated against the C++ reference on **effective
properties** and passed — while carrying traps 1 and 2. Integral quantities
sample the field only near the particle and at low truncation; they do not
certify the field that produces them. So:

- check the **pointwise** field, not only the averages;
- check it **far** from the particle, where the growing modes bite;
- check that the answer does **not depend on `Nseries`** — that single test
  would have caught both trap 1 and trap 2 at once;
- reduce to the closed form in every degenerate limit, **including extreme
  aspect ratios** (a 1:60 flat disc is where `ρ → 1` and the recurrence choice
  flips);
- validate special functions against an **outside** reference. Evaluating the
  new algorithm in wider precision only confirms itself; the reference used in
  `test/LayeredSpheroids/test_legendre_stability.jl` is the *original* upward
  recurrence at 600 bits, where the instability does not bite.

## N-body schemes — remaining pieces

The equivalent inclusion method and the cluster model are shipped; this is the
detail of what their references contain and the implementation does not.
See [the theory page](@ref th-interaction) for the shared interaction kernel.
What is left open, and what has since been closed:

- ~~**Anisotropic reference media.**~~ **Done.** The Barnett line integral of
  `echoes_cpp/tests/python/Green/Green.jl` was ported to `Core/green_aniso.jl`,
  so three-dimensional elasticity and conduction now accept any anisotropy —
  which is what makes it possible to chain one N-body estimate into another
  scale, since a cluster estimate on a cubic array is itself anisotropic. Two
  things remain open here:
    - **plane-strain elasticity** with an anisotropic reference, which needs the
      Stroh formalism rather than the Barnett integral, and is refused with a
      message naming the limitation;
    - **cost**. The anisotropic operator is a quadrature differentiated twice
      with forward-mode AD: ~1.5 ms per interaction tensor against ~0.6 µs for
      the isotropic closed form. Deriving the second gradient of the line
      integral analytically (the classical route) would recover most of that,
      and would matter for assemblies of more than a few dozen particles in an
      anisotropic reference.
- **Polarization orders `p ≥ 1`** ([brisard2014](@cite)), which need the
  influence *pseudotensors* of their Appendix C — not tensors, with their own
  change-of-basis machinery, generated by the authors with a computer algebra
  system. Only `order = 0` is implemented; a higher order raises an error.
- **Their Table 2** (polydisperse spheres at `φ = 0.45` in a spherical SVE)
  needs a polydisperse close-packing generator; `random_assembly` is
  monodisperse.
- **The slender-fiber specialization** of Martin et al. (2023): the interaction
  integrals are reachable through the `:quadrature` back-end, but the axial
  polynomial enrichment and the finite-element self-influence coefficients are
  not implemented, and `Cylinder` is not accepted by the pair kernel.
