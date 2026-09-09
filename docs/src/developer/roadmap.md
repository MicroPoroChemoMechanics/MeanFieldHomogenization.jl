# [Roadmap](@id dev-roadmap)

What is left to do, and — for the two areas where the boundary is subtle —
exactly which pieces of a cited paper are and are not implemented. What the
package already does is described in the [manual](@ref man-index) and the
[theory pages](@ref th-index).

## Open

- Extended-COD crack model in conduction: resistive cracks (linear-spring
  analog) **and** conductive cracks (elastic-membrane analog), via a
  tensorial conduction COD.
- Multi-layer extensions: coated cylinders, anisotropic per-layer moduli, and
  an *analytic* excentered sphere (only the finite-element route reaches that
  morphology).
- Laminate extensions: viscoelastic *interface* laws (the interface types
  carry `Number` fields today, so an ageing interface needs a different
  carrier), and `Homogenized` inside an ALV chain (the inner result would have
  to be re-expressible as a `ViscoLaw`).
- `PairwiseDistribution` (Willis 1982) envelope for the PCW scheme.
- Native Anderson acceleration with memory > 1, replacing the current
  `AndersonDefault` (currently Picard with relaxation, memory = 1).
- Optional structured `TensTI{4,T,8}` fast path for the ALV TI schemes.
- **Laplace-Carson viscoelasticity** (see [the theory](@ref th-laplace-carson)
  and the [model manual](@ref man-rheological-models)):
  - **anisotropic tensor pairings.** Only [`IsoRheology`](@ref) exists; a
    transversely isotropic model would need six scalar channels and the
    corresponding `TensTI` assembly.
  - **ageing models in the catalog.** `LogarithmicCreep` is the non-ageing
    skeleton of a law that is normally written with age-dependent `E`, `C` and
    `τ`; expressing that family would need a second, two-argument interface.
- **Finite-element coupling** (see
  [Finite-element coupling](@ref fe-coupling)):
  - the **consolidation column** of [barthelemyARMA2011](@cite) § 3.1, models
    M1/M2/M3 — the case where a family actually *closes* during the loading.
    Nothing new is needed for it: it is a driver, not a capability.
  - the paper's **self-consistent permeability**, whose matrix concentration factor
    (order-2 Hill tensor of the effective medium) the simpler estimate in
    `fracture_permeability` omits. It is worth ≈ 55× on the ARMA microstructure
    — the [well test](@ref fe-arma2011-scope) quantifies exactly what it costs.
  - drivers for Gridap, FEniCSx and an Abaqus-shaped UMAT.
- **Finite-element inclusions**, behind the `FEBackend` contract (see
  [FE inclusions](@ref man-fe-inclusions)):
  - an **anisotropic reference medium**. The corrected boundary data needs
    ``\nabla\mathbb G``, which `green_function_aniso` does not expose — the
    Pan-Chou closed form in the transversely isotropic case, or a differentiated
    Barnett-Willis integral, would.
  - more than one inclusion, or a **non-spherical envelope**, in the
    axisymmetric cell.
  - **transport for the crack**, the 3 + 3 declination being elastic only.
  - **solid** supershape inclusions, which would need the inclusion meshed too
    and would enter gate B with two measured tensors.
- **Neural-surrogate inclusions** (see
  [Neural-surrogate inclusions](@ref man-neural-inclusions)):
  - a `_reference_medium` for `StrainLocTI` / `StressLocTI`, deliberately
    absent because a heterogeneous morphology's reference has to be built from
    contrast features and guessing it would train on corrupted labels.
  - an **anisotropic reference medium**, which needs a feature set describing
    it.

## [The elastic layered spheroid — what is left](@id dev-elastic-spheroid)

The derivation, which is not in the literature, is on
[its theory page](@ref th-spheroid-elasticity); this section is what remains of
the module and the traps that apply to it.

### What remains

- **Pointwise fields.** The solver returns harmonic amplitudes, so `u`, `ε` and
  `σ` at a point are a matter of summing modes — the conduction side already
  offers that through `local_temperature` and its siblings.
- **An arbitrary axis.** The elementary problems are written about `ê₃`;
  `spheroid_strain_concentration` refuses a tilted spheroid rather than
  rotating the result for you.
- **A compliance-side contribution tensor**, the twin of
  `stiffness_contribution`. The stress-side localization is what it needs:
  `strain_strain_loc` reaches the schemes, `stress_strain_loc` does not exist
  for a `LayeredSpheroid`, and the generic `compliance_contribution` is written
  in terms of it.

### What will not happen

**Imperfect interfaces do not fit this formulation**, and it is worth not
attempting them: a uniform spring or membrane law puts an *odd* power of the
metric factor ``w = \sqrt{q^2-p^2}`` into the matching condition, so it stops
being a polynomial identity in ``p``, and the exactness of the projection and
the banding go with it. A perfect interface escapes this only because every
geometric factor there is shared across a confocal surface and cancels.
[The theory page](@ref th-spheroid-imperfect) sets this out, along with why a
thin confocal interphase is not a substitute — such a shell is exactly
``\omega`` times thicker at the equator than at the pole.

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

This is the detail of what the references of the equivalent inclusion method
and of the cluster model contain and the implementation does not. See
[the theory page](@ref th-interaction) for the shared interaction kernel.

- **Plane-strain elasticity with an anisotropic reference**, which needs the
  Stroh formalism rather than the Barnett line integral of `Core/green_aniso.jl`,
  and is refused with a message naming the limitation.
- **The cost of the anisotropic operator.** It is a quadrature differentiated
  twice with forward-mode AD: ~1.5 ms per interaction tensor against ~0.6 µs for
  the isotropic closed form. Deriving the second gradient of the line integral
  analytically (the classical route) would recover most of that, and would
  matter for assemblies of more than a few dozen particles in an anisotropic
  reference.
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
