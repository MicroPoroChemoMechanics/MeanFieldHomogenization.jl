# Changelog

## v0.4.0

The route from the Fourier Green operator to the COD tensor ``𝐁``, written down
and executed symbolically — first for elasticity, then for transport, where it
turns out to be analytic at *full* anisotropy. Writing the transport derivation
down is what exposed a wrong prefactor in the thermal closed forms, which is why
this is a minor rather than a patch release.

### Breaking changes

- **The thermal COD scalar `b` was too small by ``4\pi/(3\eta)`` (elliptic) and
  ``\pi^{2}/4`` (ribbon).** Every thermal crack number changes: `cod_tensor`,
  `compliance_contribution`, `delta_resistivity` and the thermal `dif` on an
  `AbstractTens{2,3}` reference. The corrected closed forms are

  ```
  b_ellipse = 4 / (3 √λ₁ 𝓔_η′)      b_iso = 4/(3 k₀ 𝓔_η)      b_penny = 8/(3π k₀)
  b_ribbon  = π / (2 √det(K₀|₍m,n₎))                          b_ribbon_iso = π/(2 k₀)
  ```

  `b` is normalized exactly like the elastic ``𝐁`` — by the in-plane half-width,
  through ``b = \chi/(b\Lambda)`` with ``\chi^{\mathcal E} = 2/3``,
  ``\chi^{\mathcal R} = \pi/4`` — so the two branches now share one convention,
  which is what the elasticity ↔ conductivity table of the theory page always
  claimed. Three independent routes pin the new values: the ``\xi_n``-integral
  chain (`scripts/16_cod_symbolic_thermal.jl`), the flattening limit
  ``\lim_{\omega\to0}\omega\,\mathbf\Lambda^{-1}``, and the textbook temperature
  jump of an insulating penny crack, ``[\![T]\!](r) = \frac{4\sigma_n a}{\pi
  k_0}\sqrt{1-r^{2}/a^{2}}``, whose surface average gives ``8/(3\pi k_0)``.
  **Unaffected:** the whole elastic branch, and `fracture_permeability` /
  `ConductiveCrack`, whose conduction side does not go through these formulas.
- **`_cod_aniso_ellipse_thermal` no longer uses the ``\mathbf K_0^{-1/2}``
  transform.** The equivalent adjugate form needs one **2×2** eigenvalue problem
  instead of `eigen` on a 3×3 plus `svdvals`, so the anisotropic thermal COD is
  now type-generic: it flows through `ForwardDiff` and evaluates on symbolic
  scalars, where it previously threw. Results agree with the old route wherever
  the old route ran (up to the prefactor above).

### Why 37 green tests did not catch it

Worth recording, because the mechanism is reusable. Nearly every assertion in
`test/Cracks/test_thermal.jl` restated the closed form the code evaluates — a
tautology that no prefactor error can fail. The one independent check, a
Hill-tensor limit, compared its oracle against
`delta_resistivity(crack, R, 1) = (4π/3) R` instead of against `R`: a `b` too
small by `4π/(3η)` is *exactly* canceled by that spurious `4π/3`, so the test
passed on two compensating errors. The file is rewritten to anchor the oracle
where the definition puts it, ``R = \lim_{\omega\to0}\omega\mathbf\Lambda^{-1}``,
and to check the **convergence rate** rather than one tolerance: the truncation
is ``O(\omega)``, so a tenfold smaller ``\omega`` must cut the error tenfold,
which a constant prefactor offset cannot fake. It also adds a
rotate-the-whole-problem invariance test on the adjugate branch. 46 assertions,
up from 37.

### Differentiability and symbolic scalars — one wrong predicate, six sites

`T <: Real` was used throughout as a proxy for "a concrete number I may compare
against a tolerance". It is not one: **`Symbolics.Num <: Real`**, yet `<`, `≈`
and `iszero` on a `Num` return *symbolic* expressions that throw in a boolean
context — while `ForwardDiff.Dual` is also `<: Real` and compares perfectly
well, so no `<: Real` test can separate the two. The predicate is now explicit,
closed and defaults to the safe answer:
`Elliptic.is_hard_numeric`, replacing the wrong test in `_agm_converged`,
`Core._sort_axes_and_basis` (both methods), `Core._classify_shape_2d/3d`,
`Cracks._classify_crack`, `Cracks._is_unit_alignment` and
`Cracks._elliptic_CS`.

- **New weak extension `MeanFieldHomogenizationSymbolicsExt`.** Without it
  `ell_K`/`ell_E` on a `Num` fell through to the AGM recursion and unrolled ~60
  nested `sqrt` into the expression tree — correct, but unusable and
  unprintable. Registered with `@register_symbolic`, they stay an unexpanded
  call on a symbolic argument while any numeric argument still reaches the fast
  `Float64` method, so `Symbolics.build_function` round-trips the result exactly.
  This is the Symbolics counterpart of the existing SymPy extension.
- **`Cracks._ti_aligned` compared `|axis·n̂|` to 1.** On a `TensTI{4, Num}` whose
  axis *is* the crack normal, `Symbolics` leaves `abs(1.0)` unevaluated — even
  `tsimplify` returns `-1 + abs(1.0)` — so the test said "not aligned" and
  `cod_tensor` silently fell through to a **numerical** back-end instead of the
  closed form. It now compares the square, which uses only `*` and does fold.
- **The numerical Green back-ends now say why they cannot run on a boxed
  scalar.** `Core._A_and_Tn` builds an `MArray`, which cannot even be
  *constructed* for a non-`isbits` eltype; the failure used to surface as
  `setindex!() with non-isbitstype eltype …` from StaticArrays, several frames
  deep. It is now an `ArgumentError` naming the type and stating which
  references do have a closed form.

Verified, rather than asserted: `ForwardDiff` through the elastic closed forms
(∂/∂E, ∂/∂ν, ∂/∂η, ribbon, aligned TI) and the thermal ones (∂/∂k₀ against the
exact `-b/k₀`, ∂/∂η, ribbon, `R₃₃`, and a full six-component gradient through the
anisotropic adjugate branch); `Symbolics.Num` through elastic iso, penny, ribbon,
symbolic aspect ratio and aligned TI, and through thermal iso and fully
anisotropic. `Float64` and `Dual` results are bit-unchanged throughout. No
`TensND` change was needed — every frame of the original failure was in this
package, so the `TensND = "0.3.5"` bound stands.

### Bug fixes

- **`cod_tensor` threw in dispatch on a symbolic transversely isotropic
  reference.** `Cracks._ti_aligned` compared the stored symmetry axis with the
  crack normal through `isapprox`, which is undefined on `SymPy.Sym` — the only
  seam of the COD chain without a `T <: Real` guard, while
  `_classify_crack` and `Core._sort_axes_and_basis` both have one. The
  comparison is now structural for non-`Real` element types.
- **`cod_tensor(PennyCrack(one(Sym)), …)` returned `NaN`, silently.**
  `_elliptic_CS` took its removable `η = 1` shortcut only for `T <: Real`, so a
  symbolic penny evaluated `𝒞 = (ℰ − η²𝒦)/k²` as `0/0`. The guard is now
  `iszero(k²)` alone: `false` for a free symbol, which is the general branch and
  the answer we want, `true` for an exact `Sym(1)`. Only non-`Real` *geometry*
  was affected — every `Float64`, `ForwardDiff.Dual` and complex-moduli result
  is bit-for-bit unchanged.

### New — from the Green operator to `𝐁`

- **New theory section** in `theory/cod_tensors.md`: the reduced kernel
  ``\hat{𝐐}^{⋆}_{nn}``, why its ``ξ_n`` integral converges, the three
  reductions (equivariance, degree-1 homogeneity, parity) that make it
  tractable, the resulting ``a_1, a_2, a_3`` for an isotropic and for an
  aligned-TI matrix, and the three master integrals that produce
  ``𝒞_η, 𝒮_η, ℰ_η``. The radical ``σ_γ`` of the published TI closed form is
  **derived** — it is the sum ``γ_1+γ_2`` of the two in-plane Stroh roots — and
  the section closes on what breaks for a general anisotropy.
- **New `scripts/09_cod_symbolic_green.jl`** — the same derivation carried out
  by SymPy end to end, with the shipped closed forms as its oracles. Not added
  to the documentation gallery: SymPy-heavy scripts are re-executed on every
  docs build (repo policy, `docs/literate.jl`).
- **First symbolic tests of the crack chain**, `test/Cracks/test_cod_symbolic.jl`
  (31 tests): `cod_tensor` on `TensISO{4,3,Sym}` and on an aligned
  `TensTI{4,Sym}`, the dispatcher's aligned/non-aligned decision, and the
  ISO-vs-aligned-TI agreement at a free symbolic aspect ratio.

### New — the same derivation for transport (order 2)

- **New section** *From the Green operator to `b`* in `theory/thermal_cracks.md`,
  and **new `scripts/16_cod_symbolic_thermal.jl`**. The order-2 acoustic form
  ``\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}`` is a *scalar*, so
  the numerator of the kernel collapses to a constant and the ``\xi_n`` integral
  closes **at full anisotropy** — no sextic, no residues, no cubature:
  ``\hat{Q}^{\star}_{nn} = \tfrac12\sqrt{(\underline{n}\wedge\underline{\xi}^{\star})
  \cdot\mathrm{adj}\boldsymbol{K}_0\cdot(\underline{n}\wedge\underline{\xi}^{\star})}``.
  The contour integral then reduces to an **effective ellipse**: an arbitrarily
  anisotropic conductor behaves as an isotropic one of conductivity
  ``\sqrt{\lambda_1}`` around a crack of effective aspect ratio
  ``\eta' = \sqrt{\lambda_2/\lambda_1}``, with ``\lambda_{1,2}`` the eigenvalues
  of a **2×2** in-plane restriction of ``\mathrm{adj}\boldsymbol{K}_0``. Being
  2×2, that step is closed form — and symbolically evaluable, where the shipped
  ``\boldsymbol{K}_0^{-1/2}`` route needs `eigen` and `svdvals` on 3×3.
- The section records a **prefactor disagreement** with the thermal closed forms
  it precedes (``4\pi/(3\eta)`` for the ellipse, ``\pi^{2}/4`` for the ribbon),
  supported by three independent checks: the ``\xi_n``-integral chain, the
  flattening limit ``\lim_{\omega\to0}\omega\boldsymbol{\Lambda}^{-1}``, and the
  textbook temperature jump of an insulating penny crack. Nothing is changed in
  the code: resolving it moves every thermal crack result, so it is left visible
  and measured by the script rather than patched in passing.

### Documentation fixes

- The isotropic prefactor was written `8η(1−ν²)/(3E)` in the docstring of
  `_cod_iso_ellipse` and in `theory/cod_tensors.md`, where the code and the
  reference both give `8(1−ν²)/(3E)`. The `η` was parasitic and invisible at the
  penny crack. Prose only — no computed value changes.
- `theory/cod_tensors.md` claimed that `method = :auto` selects `Residue` for
  anisotropic `Float64` input. It never does, and has not since the dispatch
  rework: `:auto` picks `DECUHR` when its extension is loaded and
  `NestedQuadGK` otherwise (`src/Core/dispatch.jl`).
- **Admonition titles rendered their markup literally.** Documenter emits an
  admonition title as *plain text* — no inline markdown, no math — so
  ``` ``\boldsymbol{B}`` ``` and `` `order = 0` `` appeared with their delimiters
  in the rendered header. All 13 affected titles (7 pages, 4 docstrings, 2
  Literate scripts) are now plain prose.
- Stale self-references in the banners of `scripts/10_`…`14_`, left over from a
  renumbering.

## v0.3.1

(v0.3.0 was tagged but never registered, so everything below is folded into the
first published release. No API is removed or renamed; the "breaking" entries
are results that change because they were wrong.)

MeanFieldHomogenization as a **constitutive law inside a finite-element code**,
the poroelastic and hydraulic machinery a fractured-rock model needs, and three
correctness fixes for cracks that are not aligned with the canonical axes.

### Breaking changes

- **`fracture_permeability` now returns a genuinely self-consistent estimate.**
  The broken loop was linear in the crack density and independent of the
  fracture conductivity; the fixed one percolates. On a connected network the
  two differ by orders of magnitude, so **every fractured-permeability number
  changes**, and with it every `FracturedPoroelasticRock` transport output.
- **Numbers change for crack families that are not parallel to one another, or
  not aligned with an anisotropic reference medium.** The COD frame fix below
  is a correctness fix, so any previously published `C_hom` / `S_hom` involving
  tilted cracks was wrong and now changes — self-consistent schemes included,
  since their running estimate is anisotropic by construction. Aligned families
  in an isotropic matrix are unaffected, bit for bit.
- **`SelfConsistent` iterates are now projected onto the major-symmetric
  tensors.** Coaxial configurations move by roundoff; non-coaxial crack
  families, which used to return `NaN`, now converge.
- **`TensND ≥ 0.3.5` is required** (was `0.3`). Versions up to 0.3.3 rotate
  `TensTI` / `TensOrtho` incorrectly in `change_tens` (fixed in 0.3.4), and
  versions up to 0.3.4 drop the basis in `TensISO` products, returning a bare
  `Array` where a `Tens` is expected (fixed in 0.3.5). Both are relied on by the
  fixes above, so the bound states the first version that carries them all.

### Bug fixes

- **`fracture_permeability` returned the dilute estimate, not the
  self-consistent one.** Its fixed-point loop defaulted to `abstol = 1e-10`,
  which for a tight rock (`k ~ 1e-18 m²`) is met by the *first* iterate: the
  loop exited after one step, so the reference medium was never updated. The
  symptom was silent and diagnostic — the result came out exactly linear in the
  crack density and completely independent of the fracture conductivity, hence
  with no percolation threshold at all. `abstol` now defaults to `0` and the
  test is purely relative. Randomly oriented flowing cracks now reproduce the
  closed form `K/k_s = 1/(1 − 32d/9)` to three digits, percolating at
  `d_c = 9/32`, where the broken loop gave `1 + 32d/9`.
- **Cracks tilted with respect to the reference medium gave wrong results.**
  The four anisotropic COD kernels rotated the stiffness into the crack basis
  but passed the crack frame in *global* coordinates. Aligned cracks coincide,
  which is why a suite built on them never caught it. Fixed by
  `Cracks._crack_local_frame`, which returns both in one frame; validated
  against the ECHOES reference across crack inclinations (agreement improved
  from up to 8 % error to ~1.5·10⁻⁷, ECHOES' own quadrature accuracy).
  **Any result involving non-parallel cracks in an anisotropic reference medium
  changes**, self-consistent schemes included, since their running estimate is
  anisotropic by construction.
- **`SelfConsistent` diverged on non-coaxial crack families.** Its body
  assembles `𝔹 : 𝔸⁻¹`, a product of non-commuting tensors, which is not
  major-symmetric when families are not coaxial; the asymmetry grew from
  roundoff to 2·10⁻⁴ over a few iterations and the crack cubature then returned
  a `NaN` integrand. Each iterate is now projected back onto the
  major-symmetric tensors, as the reference implementation does. Requires
  **TensND ≥ 0.3.5**, which fixes two basis bugs of its own.

### New — finite-element coupling (`MeanFieldHomogenization.Constitutive`)

- `AbstractMFHMaterial` / `material_response`, a multi-gradient, multi-flux
  Gauss-point contract with per-point internal state and a consistent tangent.
- `HomogenizedElastic` (linear, any cell and scheme) and
  `MicrocrackedMaterial` (crack families that open and close; piecewise linear,
  so the tangent is exact, with steps split at every closure and reopening).
- `MaterialCache`, keyed on the discrete open/closed set — a thick-cylinder run
  needs **4 scheme solves for 2304 quadrature points**.
- `to_tensors` / `from_tensors` / `plane_strain_response`, the `Tensors.jl`
  boundary, and `check_material_interface`.
- `MeanFieldHomogenizationFerriteMaterialExt`, activated by `import Ferrite`
  alone (no gmsh): `mfh_states`, `mfh_element!`, `mfh_poro_element!` — the
  coupled `(u, p)` element, backward Euler, whose Jacobian *is* the four tangent
  blocks the material declares — plus `annulus_grid` and `cylinder_sector_grid`.
- The **ARMA 2011 well test**: a three-dimensional, anisotropic, coupled
  reservoir simulation driven entirely by the homogenized material
  (`scripts/fe/arma2011_welltest.jl`, static figures in the docs).

### New — poromechanics and fractured permeability

- `MeanFieldHomogenization.Poromechanics`: `biot_tensor`,
  `inverse_biot_modulus`, `poroelastic_parameters`, `undrained_stiffness`,
  `skempton_tensor`, `terzaghi_stress`.
- `ConductiveCrack`, the *flowing* crack (the preferential flow path, as
  opposed to the insulating crack), and `fracture_permeability`, its
  self-consistent effective conductivity.
- `FracturedPoroelasticRock`, the saturated fractured rock of
  [barthelemyARMA2011]: two gradients `(E, p)`, two fluxes `(Σ, φ)`, the four
  Biot tangent blocks, fractures driven by the **Terzaghi** effective stress,
  and a permeability that follows the apertures through the cubic law. The
  coupled *u–p* reservoir simulations of that paper are **not** shipped; the
  material is, and is tested.
- `crack_family_compliances` / `crack_family_residual`: the per-family
  decomposition `S_hom = S_solid + Σ (4π/3) dᵢ 𝕊ᵢ`, which is what drives an
  aperture update.
- `set_amount!`, `set_geometry!`, `set_property!`: mutating `RVE` setters for
  Gauss-point loops (`set_param` remains the immutable, AD-friendly path).

## v0.2.0

Two N-body homogenization models, the machinery they share, and a bridge that
puts every existing scheme on the new cell.

### Breaking changes

- **The documented sign convention of the transport quantities is corrected.**
  The package now states `σ ≡ −q = K·∇T` as the stress analog: on a surface
  of normal `n`, `σ·n` is in *both* theories what the exterior transmits to the
  interior. **No returned value changed** — `flux_gradient_loc`, `flux_flux_loc`
  and `gradient_flux_loc` compute exactly what they always did — but their
  docstrings previously described that quantity as the flux `q` rather than as
  `σ = −q`. Code that followed the old docstrings was interpreting the sign
  backwards; the numbers were always the ones documented now. The same applies
  to the thermal `sif` / `dif`, whose driving vector is `σ∞ ≡ −q∞`.
- **Two names were removed from the export list**: `laminate_interface` and
  `layer_fraction`. Both were exported without ever being defined, so any use
  already raised `UndefVarError`; no working code is affected.

### Added

- **`ParticleAssembly`** — a cell holding individually *located* inclusions,
  with `PeriodicBox` and `MixedBC` far-field treatments, volume fractions
  derived from the geometry (`f_a = |Ω_a|/|Ω|`), particle families, and the
  `cubic_lattice` / `random_assembly` generators.
- **`ClusterModel`** — Molinari & El Mouden (1996): the mean strain of every
  inclusion, resolved pairwise inside a cluster of radius `R_c`. Degenerates
  *exactly* onto Mori-Tanaka at `cluster_radius = 0`.
- **`EquivalentInclusion`** — Brisard, Dormieux & Sab (2014) at order `p = 0`,
  a Galerkin discretization of the weak Lippmann-Schwinger equation, with
  `eim_bound_type` reporting when the estimate is a rigorous bound.
- **`interaction_tensor` / `self_interaction_tensor`** — the two-inclusion
  tensor `𝕋^{ab}` both schemes are built on, with three back-ends
  (`:analytical` closed forms exact at any separation, `:multipole`,
  `:quadrature`) plus `lattice_interaction_tensor` for periodic images.
- **Anisotropic Green operator** — `green_operator_aniso` (Barnett line
  integral for 3D elasticity, closed form for conduction), `green_function_aniso`
  and the `green_operator` dispatcher. This is what makes it possible to chain
  an N-body estimate into a further scale, since a cluster estimate on a cubic
  array is *cubic*, not isotropic.
- **`RVE(asm)`** — forgets the positions and keeps the derived fractions, so
  **every one-site scheme runs directly on an assembly**
  (`homogenize(asm, MoriTanaka(), :C)`). One phase per particle, which is exact
  because every scheme sums over phases linearly. `matrix_geometry` and
  `distribution_shape` are forwarded through `homogenize`.
- **Nano-interfaces** — `surface_stiffness` and `equivalent_particle`
  (Dormieux, Lemarchand & Brisard, 2016): a Gurtin-Murdoch interface condensed
  into an equivalent particle stiffness, after which the classical schemes
  apply unchanged.
- **Parameter lenses for assemblies** — `center_param` and `radius_param`,
  differentiable through both the N-body and the one-site paths.

### Changed

- The Green operator follows [Brisard, Bertin & Legoll (2023)](https://doi.org/10.1016/j.cma.2023.116389),
  Eq. (9): it maps a polarization onto **minus** the induced field, so its
  Fourier symbol is positive semi-definite and its self term is
  `𝕋^{aa} = +ℙ`, coherent with the Hill tensor. Molinari & El Mouden and
  Berveiller et al. use the opposite sign, which the documentation now names
  explicitly wherever a formula is transcribed from them. (This convention
  governs API introduced in this release only.)
- `green_operator_iso` and the other public Green entry points return a `Tens`
  like every other tensor-valued function of the package; the `SArray` kernels
  they wrap are internal.
- The Theory and Manual sections of the documentation are grouped, so that the
  standard theory precedes what is built on it.

## v0.1.0 — 2026-08-07

First public release of MeanFieldHomogenization.jl — a mean-field homogenization toolkit
for elasticity, conductivity and ageing linear viscoelasticity, ported from and
cross-validated against the C++ ECHOES code.

### Core

- **`RVE` container** — representative volume element (`add_matrix!` /
  `add_phase!`); volume fractions and crack densities stored at RVE level;
  solid inclusions (`VolumeFraction`) and flat cracks (`CrackDensity`).
- **Ten homogenization schemes** — Voigt, Reuss, Dilute, DiluteDual,
  MoriTanaka, Maxwell, PonteCastanedaWillis, SelfConsistent,
  AsymmetricSelfConsistent, DifferentialScheme — via a single
  `homogenize(rve, scheme, property)` entry point.
- **Elasticity (`:C`) and conductivity (`:K`)** — order-4 and order-2 tensor
  algebra, crack contributions, and Sevostianov-style interface stiffness
  (spring / Kapitza / membrane) in every pipeline.
- **Self-consistent solvers** — symmetric Hill/Budiansky SC with a built-in
  Newton-Raphson solver (quadratic convergence, backtracking line search) or
  Anderson/Picard, positive-definite guard near percolation; NonlinearSolve.jl
  algorithms available via weak extension.

### Ageing linear viscoelasticity (ALV)

- **`Viscoelasticity` sub-module** — time-domain ALV homogenization after
  Sanahuja (2013) and Barthélémy et al. (2016, 2019): `ViscoLaw` relaxation /
  creep kernels, Sanahuja trapezoidal discretization, block-Volterra inverse,
  discrete ALV Hill kernel.
- **`homogenize_alv(rve, scheme, prop; times)`** for all schemes, order-4 and
  order-2; iso and TI Walpole-basis fast paths; BLAS/LAPACK Volterra fast path.
- **Differential scheme as an adaptive SciML ODE** on the fictitious
  incorporation time `τ ∈ [0,1]` (`Tsit5` default), with functional
  `Path` / `Sequential` / `Proportional` incorporation trajectories.
- **N-layer composite spheres** — full bulk + shear Hervé–Zaoui recurrence in
  elastic and ALV form, with perfect / spring / membrane interfaces.

### Differentiability

- **ForwardDiff throughout** — `derivative`, `gradient`, `jacobian`,
  `sensitivity` of any homogenization result w.r.t. physical, geometric,
  volume-fraction or crack-density parameters (lens API); multi-scale chain
  rule by closure composition.
- RVE-level orientation projection (`symmetrize`: iso / TI Reynolds averaging).

### Added

- **Periodic multilayer homogenization**, `MeanFieldHomogenization.Laminates` — a second
  kind of microstructure, next to the random morphologies the Eshelby-based
  schemes describe:
  - `Laminate`, a periodic unit cell of parallel layers with **no matrix, no
    auxiliary Eshelby problem and no reference medium**, solved *exactly* by
    the new `Laminated` scheme. `Voigt` and `Reuss` apply to it too, and two
    of the bracketings are equalities: a laminate saturates Voigt in the plane
    of the layers and Reuss across them, simultaneously and for arbitrary
    anisotropy.
  - Elasticity **and** transport from one implementation, dispatched on the
    order of the stored property exactly as the mean-field schemes are.
  - The four imperfect-interface models of `LayeredSpheres` reused unchanged —
    a planar interface being the curvature-free case: `SpringInterface` /
    `MembraneInterface` in elasticity, `KapitzaInterface` /
    `SurfaceConductiveInterface` in transport. They act on *complementary*
    halves of the answer (primal out of plane, dual in plane — a planar
    membrane produces no traction jump at all), and enter with the weight
    `1/L`, an interface **density**: hence a genuine size effect, and why a
    laminate stores thicknesses rather than only fractions.
  - **Anisotropic interfaces**, which a plane admits and a sphere does not:
    `AnisotropicSpringInterface` (any symmetric compliance tensor),
    `AnisotropicMembraneInterface` (any 2-D surface stiffness, six
    coefficients) and `AnisotropicSurfaceConductiveInterface`. The spherical
    recurrence needs its jump conditions to share the symmetry of the
    geometry; a plane has a normal and an arbitrary in-plane texture, so it
    does not. Both exact oracles hold unchanged with a full tensor.
    (`KapitzaInterface` needs no counterpart: `[T] = ρ qₙ` relates two
    scalars.)
  - A hand-written symbolic tutorial,
    `docs/src/tutorials/symbolic_laminate.md`, deriving the closed forms from
    the code and displaying the effective Kelvin-Mandel matrix in terms of
    five layer averages — two of which enter *inverted* (harmonic, series,
    out of plane) and two directly (arithmetic, parallel, in plane).
  - Per-layer localization (`layer_strain_localization`, …), the two Hill
    tensors (`laminate_hill`) and the interface displacement jumps
    (`interface_jump`); lenses `ThicknessParameter` and `InterfaceParameter`.
  - Ageing-viscoelastic twin (`laminate_alv`,
    `homogenize_alv(lam, Laminated(), …)`): the *same* kernel with the `3×3`
    cofactor inversion swapped for `volterra_inverse` on the out-of-plane
    restriction. The elastic limit returns the elastic laminate in every
    diagonal time block, and both exact saturations survive the transposition.
  - Symbolically evaluable end to end: `scripts/38_laminate_symbolic.jl`
    derives the Backus (1962) closed forms *from the code*. This is what pins
    the two implementation choices that make it possible — the pseudo-inverse
    is a cofactor inverse of the out-of-plane block, never an SVD-based
    `pinv`, and every intermediate is an `SMatrix`, never an `MMatrix`.
  - Documented theory (`docs/src/theory/laminate.md`), which **corrects** the
    reference note it follows: its appendix introduces a *pair* of
    "in-plane / out-of-plane pseudo-inverses", which is vacuous for the tensor
    actually involved — `⟨ℙ⟩` has an identically zero in-plane block, so its
    in-plane pseudo-inverse does not exist. One ordinary Moore-Penrose
    pseudo-inverse is all that is needed, and the unfinished general case is
    never required.

- **The homogenization-cell abstraction and declarative multiscale chaining**:
  - `AbstractHomogenizationCell` (in `Core/cells.jl`) is now the supertype of
    everything `homogenize` accepts — `RVE` and `Laminate` today. `homogenize`,
    `get_param`/`set_param` and `derivative`/`gradient`/`jacobian` are typed on
    it; every scheme kernel stays on `RVE`, so existing behavior is unchanged.
  - `Homogenized(cell, scheme)` may be stored **as a property value**: a whole
    multiscale model becomes one object, resolved lazily. With no explicit
    `property` it *inherits the key it is stored under*, so one nested cell
    answers `:C` and `:K` alike — a microstructure is described once, not once
    per physics.
  - `NestedParameter` / `nested(...)` addresses a scalar inside a nested cell,
    so `derivative`/`gradient`/`jacobian` cross every scale in a single
    `ForwardDiff` pass, replacing hand-rolled closures and explicit
    `ForwardDiff.Tag` plumbing.
  - Each `(cell, key)` pair is evaluated exactly once per `homogenize` call —
    including across the ~100 iterations of a self-consistent solve — through a
    task-local, call-scoped cache that never leaks a value between two autodiff
    evaluations.
  - Both styles are documented and compared in
    `docs/src/manual/multiscale.md`, with a worked side-by-side in
    `scripts/36_laminate_multiscale.jl`; `applications/strength.md` and
    `applications/cement_paste_diffusion.md` now show the declarative variant
    next to their production (explicit) model, and say when each is preferable.

- **Inclusions whose response is a trained neural network**,
  `MeanFieldHomogenization.NeuralInclusions` — a fourth route into the custom-inclusion
  contract, after the analytic families, the layered patterns and the
  finite-element solves:
  - `NeuralHillInclusion` (gate A, the Hill tensor) and
    `NeuralLocalizationInclusion` (gate B, both localization tensors, for a
    heterogeneous morphology that has none). Gate A keeps the contrast
    dependence and the `ℂ₁ = ℂ₀ ⟹ 𝔸 = 𝕀` limit **exact**, so the fit error is
    confined to a single tensor.
  - What a surrogate buys that the other routes cannot: it is **differentiable
    in the morphology**. `derivative(rve, scheme, geometry(:phase, :field))`
    reaches an aspect ratio, where the finite-element inclusions refuse the
    request outright because their solve would silently return zero. Smooth
    activations only (`tanh`, `softplus`, never a piecewise-linear one), so
    second derivatives exist too.
  - Enforced rather than fitted: the symmetry class and the major symmetry
    (a structured TensND type from the right number of components), homogeneity
    in the reference moduli (`ℙ(λℂ₀) = ℙ(ℂ₀)/λ`), and frame indifference.
    `AffineHill` goes further and removes `ν₀` from the inputs altogether, using
    the exact affine structure `ℙ = d·𝕌ᴬ + 𝕍ᴬ/μ₀` of the isotropic-matrix Hill
    tensor — one input fewer and an order of magnitude more accurate than
    `DimensionlessHill` at equal network size.
  - The learning system: `SampleBox` + Halton low-discrepancy sampling (no RNG
    state, so a dataset is reproducible), `generate_dataset` with a
    two-callback teacher (`geometry`, `response` — the only morphology-dependent
    part), `fit_scaling`, `train_surrogate`, `validate_surrogate`,
    `report_surrogate`.
  - `Provenance` traveling with every model — teacher, sample counts, held-out
    error — and a **domain guard** that refuses (or warns on) an evaluation
    outside the box the model was trained on. Test tolerances are derived from
    the recorded error rather than hard-coded, so a retraining cannot silently
    loosen a threshold.
  - Four models committed under `src/NeuralInclusions/models/` as reviewable
    JSON, validated against the analytic ellipsoid; `scripts/nn/train_models.jl`
    regenerates them and `scripts/84_neural_inclusion_ellipsoid.jl` is the
    published tutorial. Nothing is trained at test or documentation-build time.
- **`MeanFieldHomogenizationLuxExt`** (`Lux` + `Optimisers` + `Zygote`, weak dependencies) —
  the optimizer, and the only part of the surrogate pipeline that needs one. It
  writes its result back into a dependency-free `MLP`, so evaluating a trained
  model needs nothing beyond the package.

- **Inclusions solved by finite elements**, `MeanFieldHomogenization.FiniteElements`, both
  using the finite Eshelby cell with the first-order corrected boundary
  condition of
  [Adessina et al. 2017](https://doi.org/10.1016/j.ijengsci.2017.03.015):
  - `FEEllipticCrack` — flat elliptical crack in 3-D tetrahedra, whose COD
    tensor comes out of the solve. Declaring `shape_trait` and supplying
    `cod_tensor` is enough for ℍ, ℕ, 𝐑, 𝐍_K, the bundled pair and the four
    `delta_*` to be inherited, so it drops into every scheme, `symmetrize`
    included.
  - `FEExcenteredSphere` — sphere with an off-center spherical core, solved by
    **axisymmetric Fourier elements** (modes 0/1/2 in elasticity, 0/1 in
    transport). Three assemblies and eight solves in two dimensions where the
    three-dimensional formulation needs six solves on a mesh two orders of
    magnitude larger. Heterogeneous, so it enters through gate B with both
    localization tensors measured in one pass; it also supplies exact `Voigt`
    and `Reuss` bounds, since the geometry fixes its internal volume fractions.
  - Memoization on the reference medium in the inclusion's local frame:
    under `IsoSymmetrize` / `TISymmetrize` a whole family of orientations
    shares a single solve. `fe_assembly_count` / `fe_reset!` to inspect it.
  - Diagnostics `fe_mesh_report`, `fe_cod_breakdown`, `fe_axi_mesh_report`,
    `fe_axi_breakdown` — the last returning the **uncorrected** tensors beside
    the corrected ones, which is how the correction is shown to work.
- **A two-backend contract for the finite-element solves.** The physics —
  Fourier operators, boundary data, the polarization fixed point, the
  memoization — lives in `src/FiniteElements/`; a backend supplies only the
  discretization, through nine generics (`FEBackend`, `backends.jl`).
  - `FerriteBackend` (`MeanFieldHomogenizationFerriteExt`, needs `Ferrite`, `FerriteGmsh`,
    `Gmsh`) serves both morphologies.
  - `GridapBackend` (`MeanFieldHomogenizationGridapExt`, needs `Gridap`, `GridapGmsh`)
    serves both as well, stating the weak form directly rather than looping
    over elements: `∫( ε(v) ⊙ (σ∘ε(u)) )dΩ` for the crack,
    `∫( Eᵐ(v) ⋅ (D ⋅ Eᵐ(u)) * ρ )dΩ` for the axisymmetric modes.
  - The crack front weld — the merge of the node pairs gmsh's `Crack` plugin
    duplicates in spite of `OpenBoundaryPhysicalGroup` — moved from
    `Ferrite.Grid` surgery to a renumbering of the written `.msh`, so both
    backends read the same welded mesh.
  - `AutoBackend`, the default, resolves at the **first solve** and is then
    pinned: an inclusion can be built and stored in an `RVE` with no
    finite-element package loaded.
  - The two backends build the same discrete system and agree to round-off —
    ≈ 10⁻¹⁴ on every tensor, both physics, both morphologies — which
    `test_gridap_backend.jl` asserts at 10⁻⁸.
- **Isotropic Kelvin dipole field** in `Core`: `green_gradient_iso`,
  `dipole_displacement_iso` — the far field a polarized inclusion radiates,
  and the ingredient of the corrected boundary condition.
- **Compliance formulation of the differential scheme.**
  `DifferentialScheme(; formulation = :compliance)` integrates the dual ODE
  `dS/dτ = Σ φ̇ᵢ ℍᵢ(S)` and inverts the result, instead of
  `dC/dτ = Σ φ̇ᵢ 𝐍ᵢ(C)`. The two are analytically equivalent
  (`ℍ = −𝕊 : 𝐍 : 𝕊`) and agree to solver accuracy; they differ in which
  variable carries the error control, the compliance form being the better
  conditioned one for a medium softening towards percolation. Available in
  elasticity, conduction and ALV (both tensor orders).
- **`differential_path(rve, scheme, property)`** returns `(τ, states)` along
  the incorporation path instead of the `τ = 1` value alone. `scripts/24`
  no longer has to reach into the scheme's internals to plot `C^hom(τ)`.
- **Pair constructors for the trajectories**: `Path(:A => τ -> …, :B => …)`,
  `CustomPath(:A => values, …)`, `Sequential(:A, :B)` — the form the
  documentation already advertised.
- **Differential scheme in order-2 ALV** (viscous conduction / diffusion):
  `differential_alv_order2`, reached through
  `homogenize_alv(rve, DifferentialScheme(), :K; times)`, which previously
  raised a `MethodError`.
- **`LayeredSphere` phases in the ALV differential scheme**, through
  block-matrix (`_at`) variants of the layered-sphere ALV kernels. Mori-Tanaka
  and the self-consistent schemes already supported them.
- Unrecognized `DifferentialScheme` keywords are now forwarded to
  `OrdinaryDiffEq.solve` (`maxiters`, `dtmax`, `callback`, …) instead of being
  silently dropped.
- **New tutorial: [Comparing loading-path trajectories](docs/src/tutorials/differential_loading_paths.md)**,
  a τ-resolved companion to the existing path-dependence tutorial — four
  trajectories to the same target fractions, plotted through
  `differential_path` rather than compared at `τ = 1` only.
- **New tutorial: [Ageing viscoelastic schemes side by side](scripts/62_alv_schemes.jl)**
  (`scripts/62_alv_schemes.jl`, published as `tutorials/generated/alv_schemes.md`).
  Dilute, Mori-Tanaka, Maxwell and PCW on one ageing creep test, after
  [barthelemyIJES2019]: the collapse `MT = Maxwell = PCW` when the distribution
  shape equals the inclusion shape, the aspect-ratio sweep at fixed volume
  fraction, and the volume fraction beyond which the Maxwell/PCW estimate
  leaves the admissible domain of Ponte Castañeda & Willis.
- **`02_hill_elasticity.jl` promoted to a tutorial**
  (`tutorials/generated/hill_tensors.md`): no tutorial called
  [`hill_tensor`](@ref) as its subject, though it is the object every scheme is
  built on. Four geometries against their closed forms, two independent
  algorithms on an anisotropic matrix, the Eshelby tensor against
  [eshelby1957], and the step from `P` to a dilute estimate — checked against
  both the analytical dilute formula and `homogenize(rve, Dilute(), :C)`.
- **`59_alv_sensitivities.jl` promoted to a tutorial**
  (`tutorials/generated/alv_sensitivities.md`): `ForwardDiff` through the
  Volterra assembly, the two patterns that cover every case — the `set_param`
  lens for a parameter carried by the RVE, closure capture for one living
  inside the `ViscoLaw` — a joint gradient combining both, and a
  relaxation-time derivative that has no elastic counterpart. Every value
  validated against a central finite difference.

- **`RVE{T}(:M)` / `RVE{T,S}(:M)` constructors**, strictly equivalent to the
  `RVE(:M; T = …)` keyword form, which is kept.
- **`promote_rve(rve, T)` and `convert(RVE{T}, rve)`** to force an
  element-type floor on an already-built RVE.

- **Bundled localization helpers** — `Core.loc_and_stiffness` /
  `Core.loc_and_stress_average` (plus
  `Cracks.compliance_and_stiffness_contribution` and the `Schemes`-level
  `_phase_*_and_*` wrappers) share the single expensive `hill_tensor` /
  `cod_tensor` / layered-recurrence solve between the two quantities that
  Mori-Tanaka and the self-consistent kernels always request together. They
  used to be computed independently, i.e. twice with identical arguments.
  Results are **bitwise identical**; measured effect: −50 % time and
  allocations on an anisotropic matrix or a crack phase, −18.6 % on a 20-bin
  orientation family (40 → 20 Hill solves).

- **Benchmark suite** (`scripts/bench/bench_suite.jl` + `harness.jl`) with a
  committed baseline, three independent measurement channels (time,
  allocations, work counters), a calibrated noise floor and a bitwise
  checksum gate. See `scripts/bench/README.md` and `scripts/bench/DIAGNOSTIC.md`.

### Changed

- **Documentation reorganized along the Theory / Manual / Tutorials /
  Applications boundary**, with a stable anchor (`@id`) on every page. Two
  pages change section, so their URLs move: `applications/transport.md` →
  `tutorials/transport.md` (it demonstrates the API rather than reproducing a
  published study, unlike the seven pages that remain in Applications) and
  `tutorials/from_echoes.md` → `manual/from_echoes.md` (an API correspondence
  table, not a guided walk-through). The finite Eshelby cell, previously
  derived twice, is now stated once in
  [theory/corrected_cell.md](docs/src/theory/corrected_cell.md) and cited from
  the manual, the tutorial and the application.

- **`method = :auto` no longer selects the residue algorithm for a 3D
  anisotropic elastic reference.** It now picks a cubature — `DECUHR` when its
  extension is loaded, the type-generic `NestedQuadGK` otherwise — and
  `Residue` is reachable on an explicit `method = :residues` only, mirroring
  the `NUMINT` default of ECHOES. The reason is robustness, not taste: the
  residue acoustic polynomial degenerates whenever the reference is
  anisotropic in *type* and isotropic in *value*, and returns `NaN` /
  `DomainError` there. That reference is easy to reach — it is what the
  differential and self-consistent schemes feed back at their first step, and
  `Dilute` / `MoriTanaka` hit it too — so several RVEs that used to fail now
  work: an aligned triaxial ellipsoid or a solid phase combined with an
  aligned crack family in the differential scheme, and any scheme given an
  anisotropically-typed isotropic-valued matrix.

  Consequences to be aware of:

  - anisotropic `:auto` paths are slower (~11 ms with `DECUHR`, ~31 ms with
    `NestedQuadGK`, against ~4 ms for the residues) and their results move.
    Measured against the residue values on the benchmark suite: 7.4e-12 for a
    crack in a triclinic matrix, 1.25e-10 for Mori-Tanaka with an anisotropic
    matrix, and 1.34e-7 for a multi-axis-TI self-consistent estimate — a
    fixed-point scheme amplifies the per-evaluation cubature error (~1e-9 for
    `DECUHR`) up to its own convergence tolerance, so that is the floor on a
    converged SC value now. Pass `method = :residues` explicitly to recover
    both the previous speed and the ~1e-14 accuracy where the reference is
    known to be non-degenerate, or `:nestedquadgk` for ~1e-14 robustly;
  - loading `DECUHR` changes which cubature `:auto` picks, hence the accuracy
    the effective property is computed to. Pass `method` explicitly wherever
    that must not depend on the session;
  - non-`Float64` coefficients (`ForwardDiff.Dual`, `Complex`, symbolic) are
    unaffected: they already routed to `NestedQuadGK`.

- **A `RVE`'s element type is now a promotion floor, not a cast.** The
  `amounts` dict became heterogeneous (`Dict{Symbol,AbstractAmount}`) and
  `add_phase!` stores each amount under `promote_type(T, typeof(value))`
  instead of `convert(T, value)`. Every consequence is a widening: an amount
  narrower than the floor (`Int`, `Float32`, `Rational` under the default
  `T = Float64`) is stored exactly as before, and the only pre-existing calls
  whose stored type changes are those that used to be silently *narrowed*
  (a `BigFloat` fraction in a `Float64` RVE now stays `BigFloat`).

  - complex, `ForwardDiff.Dual` and symbolic amounts are accepted by a plain
    `RVE(:M)`, so `T = ComplexF64` / `T = typeof(f)` / `T = Sym` declarations
    are no longer needed anywhere (they remain valid, and still widen narrower
    amounts);
  - phases may carry amounts of *different* element types — a `Dual` fraction
    next to a `Float64` one, a `Dual` crack density next to a real fraction —
    promoted only where the values meet;
  - `eltype(rve)` now reports the *effective* element type (floor promoted with
    the stored amounts); `eltype(typeof(rve))` reports the declared floor;
  - `matrix_volume_fraction` takes its unit from the accumulator, so a symbolic
    RVE yields `1 - f` rather than `1.0 - f`;
  - `matrix_volume_fraction` is now a **cached field read** (`RVE.f_matrix`),
    refreshed by `add_phase!` with the same loop and the same dict iteration
    order, hence bit-identical. Do not write `rve.amounts` directly; that
    leaves the cache stale.

  **Performance.** Paired against the previous commit on the 67-case
  `scripts/bench` suite: all checksums bit-identical, allocations down on 21
  cases (−19 152 B), median |Δt| 1.1 %, every sub-microsecond scheme at or
  below the previous commit. Heterogeneous amounts cost a type refinement
  (`a isa VolumeFraction` no longer narrows to a concrete
  `VolumeFraction{Float64}`), paid back by the `f_matrix` cache
  (61 ns / 48 B → 2.4 ns / 0 B) and by `scale_by_amount(a, X)`, which puts the
  per-phase product behind a barrier dispatching on the amount's concrete
  type. The crack `delta_*` paths keep the plain `amount_value(a)`: the same
  barrier needs varargs there and measured +1.4 KB per call for no gain.

  Complex moduli never required a declaration in the first place — the moduli
  live in an untyped `Phase.properties` dict and were always promoted at the
  arithmetic. The tutorials that claimed otherwise, and the
  `fraction = ComplexF64(f)` idiom in `scripts/51`, `scripts/61` and the
  bituminous application, have been corrected.

### Fixed

- `set_param(rve, PropertyParameter(...), x)` narrowed the rebuilt property
  dict to `Dict{Symbol,TensND.AbstractTens}` although `Phase.properties` is
  `Dict{Symbol,Any}`. It therefore **threw** on any RVE carrying a non-tensor
  property — every ageing-viscoelastic RVE (`ViscoLaw`) included. Undetected
  because the ALV sensitivity tests only exercised `AmountParameter`.

- **The order-2 (transport) Hill tensor lost the orientation of a rotated
  inclusion.** `_hill_order2_3d_iso` read the Newton-potential components in the
  inclusion's own basis and wrapped them into a *canonical* `Tens`, so
  `hill_tensor(ell, K₀)` returned the tensor of the **unrotated** inclusion —
  exactly, hence silently. Both the ellipsoid and the cylinder kernels were
  affected, in the isotropic-matrix branch only; the anisotropic branch always
  built the shape tensor with the rotation in it, and elasticity was never
  affected. Consequence: any transport homogenization with *oriented*
  non-spherical inclusions was wrong off-axis, and an orientation distribution
  collapsed to a single orientation. The existing `s = ℙ·K₀` consistency test
  could not catch it — both sides dropped the rotation identically — so the new
  guard in `test/Conductivity/test_hill_order2.jl` tests rotation
  **equivariance** and cross-checks the isotropic kernel against the independent
  anisotropic derivation.

- **`Maxwell()` ignored the RVE's distribution shape on the ageing-viscoelastic
  path.** `_homogenize_alv_dispatch(::Maxwell, …)` built its Hill kernel on a
  hard-coded `Spheroid(1.0)`, while the elastic `Schemes.maxwell` reads
  `rve.distribution_shape` and the ALV `PonteCastanedaWillis` — algebraically
  the same formula — read it too. The same scheme on the same RVE therefore
  answered differently depending on which path it was called through, and the
  ALV answer silently ignored a modeling choice the user had made. Unnoticed
  because every test used the spherical default. On a 30 % / 10:1 oblate
  composite the error on `C₁₁₁₁` was 7.7 %; after the fix the ALV and elastic
  results agree to the last bit in the non-ageing limit.

- **A heterogeneous inclusion no longer contributes zero.** The generic
  contribution tensors (`stiffness_contribution` and its three siblings) went
  through `(C₁ − C₀) : 𝔸`, which is meaningless when the inclusion has no
  single `C₁` — and evaluated to zero for `FEExcenteredSphere`. They now switch
  to the exact identities `ℕ = 𝔸_σε − ℂ₀ : 𝔸_εε` and
  `ℍ = (𝔸_εε − 𝕊₀ : 𝔸_σε) : 𝕊₀` when `is_homogeneous_inclusion` is false, which
  makes gate B a complete entry point for a heterogeneous morphology.
- **`AsymmetricSelfConsistent` no longer depends on the bounds.** Nothing in
  the asymmetric algorithm needs one; only the heuristic that chooses once
  between the stiffness and the compliance form evaluated `Voigt`. It now falls
  back to the dilute estimate when the phases expose no layer-wise average, so
  the scheme is available wherever `SelfConsistent` is. New predicate
  `Schemes.has_layer_average` in place of a `try`/`catch`.
- **The isotropy guard of the axisymmetric solver was too tight.** At `rtol =
  1e-8` it refused `SelfConsistent` and `AsymmetricSelfConsistent` on a
  perfectly legitimate isotropic problem: the tensors the solver *returns* are
  isotropic only to the discretization error, a few parts in a million, so an
  iterative scheme can never feed back a reference isotropic to machine
  precision. Relaxed to `1e-4`; genuine anisotropy is orders of magnitude
  larger.
- **The boundary entities of the meshes are now fully declared.** A gmsh
  physical group has a dimension, so a group of curves does not carry its own
  bounding points and a group of surfaces does not carry its edges. A backend
  that reads boundary conditions off entity labels rather than off mesh facets
  therefore left dofs free: three on the outer sphere and six on the axis of
  the axisymmetric cell, eleven on the sphere of the crack cell. Cost on the
  axisymmetric cell: one order of convergence, under the appearance of
  discretization error.
- **Analytic sensitivity through a finite-element geometry is refused instead
  of returning zero.** `_replace_geom_field` copies non-numeric fields by
  reference, so the perturbed inclusion shared the original's `FECache` — whose
  key is the reference medium alone — and served back the unperturbed tensors.
  The derivative came out as exactly zero, with no warning. Use a finite
  difference over freshly constructed inclusions.

- **`method = :nestedquadgk` no longer raises a method ambiguity on an
  isotropic reference.** The `TensISO` disambiguation rules existed for
  `:auto`, `:residues` and `:decuhr` but had never been added for
  `:nestedquadgk`, so the one always-available cubature could not be requested
  explicitly for an isotropic matrix — for an ellipsoid or for a crack.
- **The crack dispatch no longer decides the anisotropic default on its own.**
  `Cracks._ti_crack_dispatch` ended on a hard-coded `Residue()` for a crack in
  a non-aligned TI matrix, so that path kept the residue algorithm as its
  `:auto` regardless of the shared rule — exactly the duplication
  `Core/dispatch.jl` centralizes to avoid. It now defers to
  `_aniso_default_algo`, as the ellipsoid TI refinement already did.
- **Crack densities are now diluted by the solid increments in the
  differential scheme.** Replacing `dφ` of the current medium by solid
  material also destroys the cracks that piece contained, so the volume
  balance extends to `dε_c = dφ_c^ε − ε_c Σ_{j solid} dφ_j`, inverted by the
  same Sherman-Morrison factor: `dφ_c^ε = dε_c + (ε_c/f₀) Σ_{j solid} df_j`.
  The missing term made `ε_c(τ)` an incorporation schedule rather than the
  density actually reached. Unchanged for crack-only RVEs and for
  trajectories where no solid grows while the cracks do.
- **An aligned non-spherical phase no longer crashes the differential
  scheme.** The ODE state was sized from the matrix's symmetry class alone,
  on the assumption that only cracks could leak anisotropy into the running
  estimate. An aligned spheroid does too — its dilute concentration tensor is
  transversely isotropic even between two isotropic materials — so
  `homogenize(rve, DifferentialScheme(), :C)` died on a `DimensionMismatch`
  as soon as a phase was not spherical. The state is now sized by probing the
  phase contributions, and lands in the smallest class that holds the running
  estimate (TI for aligned spheroids and cracks, rather than the full Mandel
  fallback — smaller state, and the Hill backends keep their closed-form
  paths). Isotropic RVEs keep the exact same 2-component state and results.
- **The compliance contribution of a heterogeneous inclusion no longer uses
  its declared (meaningless) phase property.** `_phase_compliance_contribution`
  applied `inv(C₁)` to a placeholder that every stiffness-side kernel ignores,
  so `DiluteDual` and the compliance-side ASC silently depended on what the
  user happened to declare for a `LayeredSphere` / `LayeredSpheroid`. It now
  goes through `ℍ = −𝕊₀ : 𝐍 : 𝕊₀`.
- **The ALV differential scheme no longer returns wrong values for an
  anisotropic running medium.** The ALV Hill kernel exists for an isotropic
  reference only, but the differential scheme evaluates it against its
  *running* medium, which an aligned non-spherical inclusion — or any crack
  without isotropic orientation average — takes out of the iso class;
  `iso_params_from_blocks` read `(α, β)` off it regardless. Such RVEs now
  raise an `ArgumentError` naming the phase and the two ways out. Cracks with
  `symmetrize = :iso` now work, where they used to fail with a cryptic "only
  iso reference is supported".
- `homogenize_alv(rve, DifferentialScheme(), prop; times)` honors `prop`
  instead of always homogenizing `:C`.

- **The n-layer sphere's shear localization `β_k` is validated against ECHOES.**
  `benchmark_nlayers.jl` had dropped the comparison, citing a 1–50 % gap blamed
  on an `echoes.layer_eE` indexing convention. The gap was ours: `β_k` was the
  bare mode-1 amplitude `a_k`, missing `b_k·F_k`, whose omission cancels in the
  degenerate configurations the fallback check used. The term was added later
  by an unrelated fix and the comparison never re-run. Restored alongside
  `α_k`: **30/30 configurations (2 to 8 layers), 4.4e-14**. The header comment
  in `shear_recurrence.jl`, which still described `β_k = a_k`, now matches the
  implementation.

- **`test_complex_moduli.jl` no longer hides scheme failures.** Every scheme
  was wrapped in `try/catch … @test_broken false`, so a scheme that stopped
  working in the complex plane vanished silently from the report. The
  assertions are now explicit and per scheme. Ground truth established by the
  de-masking: all ten schemes work with complex moduli; the single gap is
  `SelfConsistent(algorithm = NewtonDefault())`, whose ForwardDiff Jacobian
  cannot carry a `Dual` over a complex scalar (the default Anderson solver is
  the complex-capable path). This is now pinned by a `@test_throws` and
  documented in the manual.

- **Per-phase helpers now all evaluate a phase in the same reference medium.**
  `_phase_dilute_concentration` (both orders) and
  `_phase_stiffness_contribution` (4th order) evaluated their phase in
  `_project_matrix(P₀, sym)`, whereas `_phase_stiffness_contribution` (2nd
  order) and `_phase_compliance_contribution` (both orders) used the **raw**
  `P₀`. As soon as a phase carried `symmetrize ≠ NoSymmetrize`, the
  concentration tensor `A` and the contribution tensor `N` of one and the same
  phase were therefore computed in two *different* reference media, and the
  Mori-Tanaka denominator `⟨A⟩` mixed the two.

  The discrepancy is invisible on an isotropic matrix — `isotropify` is then a
  no-op to ~1e-16 — which is why every existing `symmetrize` test missed it:
  all of them use isotropic matrices *and* isotropic phase properties. It
  becomes a genuine (non-negligible) difference for an **anisotropic** matrix.
  All published isotropic-matrix results are unchanged to machine precision;
  the `echoes` cross-checks (`benchmark_pichler.jl`, `benchmark_nlayers.jl`,
  `benchmark_hill_derivative.jl`) are unaffected.

  Covered by `test/Schemes/test_loc_bundles.jl`, "every phase helper uses the
  same reference medium", which pins the corrected values on a triclinic
  matrix.

- **`SelfConsistent` with transversely-isotropic phases no longer errors.** The
  exact azimuthal average returns a `TensTI{4,T,8}` — the commutant of SO(2)
  about the axis is 8-dimensional, not 6 — while the analytical TI-coaxial Hill
  builder only had methods for the 5- and 6-parameter forms, so the running
  estimate raised `MethodError: no method matching _hill_3d_ti_coaxial(…,
  ::TensTI{4,Float64,8})`. A stiffness is major-symmetric, hence its average
  has `ℓ₇ = ℓ₈ = 0` exactly and narrows losslessly (`_ti8_to_ti6`).

- **`NewtonDefault` could not solve a fixed point richer than its starting
  guess.** The Newton parameter space was taken from `x0` (often a `TensISO`
  phase property, 2 components) while one application of the scheme can land in
  a larger symmetry class (a `TensTI{4,T,8}`, 8 components), so the residual
  subtracted vectors of different lengths (`DimensionMismatch`).
  `AndersonDefault` was unaffected, since Picard simply propagates whatever the
  step returns. The parametrization is now seeded from `step(x0)`.

### Performance

All figures below are measured against the committed baseline
(`scripts/bench/baseline.json`, 67 cases, noise floor 1.5 %); 64 of the 67
cases stay **bitwise identical**, the other three move by at most 6.9e-16
through pure floating-point reassociation.

- **Crack COD back-ends no longer allocate per quadrature node.**
  `_qnn_pair_components` (the innermost loop of the whole `Cracks` module) is
  now a pure function returning an `SMatrix{3,3,T}` instead of writing into a
  caller buffer through ~10 heap 3×3 `Matrix{T}` temporaries per α node;
  `_A_and_Tn`, `_phi_cache` and `_inv3` likewise return `StaticArrays`.
  `cod_tensor` on an elliptic crack in a triclinic matrix: **−85 % time,
  −99.4 % allocations** (18.5 MB → 111 KB). `StaticArrays` becomes a direct
  dependency (it was already in the manifest transitively).

- **Anisotropic 2D Hill uses one vector-valued quadrature.** It ran 16
  separate `quadgk` calls, each evaluating the full 16-component integrand and
  keeping one component. **−35 % time, −18.5 % allocations.**

- **DECUHR and anisotropic-cylinder Hill use the closed-form acoustic
  inverse.** Both built a symmetric 3×3 `K` with a 4-deep 81-iteration loop
  and then called generic LU `inv` per node; `_sym3_inv_acoustic` already did
  this allocation-free and Dual-safe. **−62 % time, −80.7 % allocations.**

- **Green helpers exploit the major symmetry of `C`.** `Kns` follows from
  `Vs + transpose(Vs)` at zero flops instead of an 81-iteration loop, and `Ks`
  is accumulated on its upper triangle only. **−21.5 %** on `hill_tensor`
  under `ForwardDiff.Dual`.

- **The self-consistent Newton solver stops recomputing residuals it already
  has.** The line search accepts `r_new` and the next iteration recomputed the
  residual at the same point (one full RVE pass — i.e. one `hill_tensor` per
  phase — per iteration); `Tref` was derived from a whole extra evaluation
  whose value was discarded. **−18.4 % allocations.**

- Requires TensND 0.2.6 to benefit from its `TensOrtho` `getindex` and
  `tensor_or_array` fixes (`−57 %` on `⊡` between structured operands).

### Removed

- **15 dead functions**, none reachable from `src/`, `ext/`, `test/` or
  `docs/`: `_Qnn_direct` and `_acoustic_tensor` (`Core/green_kernel.jl`),
  `Core._quadgk` (`Core/quadrature.jl`), `_masson_log`
  (`Core/green_residue.jl`), `_sc_pd_guard` and `_rve_in_compliance_space`
  (`Schemes/self_consistent.jl`), `_amounts_with_promoted_eltype` and
  `_replace_tuple_at` (`Schemes/parameters.jl`), `_mandel66_to_tens`,
  `_block_value_tensor`, `_block_value_mandel`, `_get_block`
  (`Viscoelasticity/trapezoidal.jl`), `_block_value_order2_tens` and
  `_block_value_order2_mat` (`Viscoelasticity/order2_alv.jl`),
  `_sc_alv_mt_body_against` and `_sc_alv_step_echoes`
  (`Viscoelasticity/schemes_alv_sc.jl`).

  `_Qnn_direct` was the starting point of this whole audit: it contracted the
  stiffness with the Green kernel through a six-deep `p,q,r,s,α,β` loop — 729
  iterations where the contraction factorizes into `U = (C·n̂)·ξ` then
  `B = U·K⁻¹·Uᵀ`, roughly 100 flops. The factorized form already existed twice
  in the package (`Cracks/green_residue.jl`, `Core/green_helpers.jl`), so the
  slow copy was deleted rather than optimized, and the header of
  `green_kernel.jl` now points at the two live implementations.

  `Core._quadgk` deserves a special mention: its own header described it as
  the wrapper "all downstream sub-modules should go through", and no call site
  had ever used it.

### Compatibility & validation

- Aligned with TensND 0.3 (snake_case + UPPERCASE-acronym API).
- Cross-validated against C++ ECHOES to ≤ ~1e-3 (moduli) and machine precision
  (elastic limits) across schemes, porous, layered and ALV benchmarks.
- ~3900 tests.

---

## Pre-1.0 development history

The entries below predate the first public release and use the internal
version numbers under which each feature was developed (never published outside
MPCM-Registry). Kept for reference.

### v0.8.1 — Packaging & spelling fix

#### Fixes

- Documentation: unified American English spelling throughout
  (`homogenization` consistently, replacing `homogenisation`)

#### Infrastructure

- Registered in MPCM-Registry; DECUHR.jl resolved via registry
  (no more `[sources]` local path in `Project.toml`)
- GitHub Actions workflows: CI, Documentation, Register, CompatHelper, Format, TagBot
- Multi-version documentation deployment (`docs/deploy_docs.jl`)

---

### v0.8.0 — Differential scheme as a SciML ODE on the fictitious incorporation time

**`DifferentialScheme` is now solved by an adaptive SciML ODE
integrator** (`Tsit5` default) on the fictitious incorporation time
`τ ∈ [0, 1]`, replacing the explicit Euler discretization.  The
underlying multi-phase incorporation-sequence ODE
([Norris 1985](@cite norris1985); user's hand-written DEM note) reads

```math
\frac{\mathrm d \mathbb C^{hom}}{\mathrm d \tau}
  = \sum_\alpha \frac{\mathrm d \varphi_\alpha}{\mathrm d \tau}
                (\mathbb C_\alpha - \mathbb C^{hom}):\mathbb A_\alpha^{dil}(\mathbb C^{hom})
   + \sum_c \frac{\mathrm d \varepsilon_c}{\mathrm d \tau}
            \Delta\mathbb C^{crack}_c(\mathbb C^{hom})
```

with the volume balance `df = (𝟙 − f ⊗ 𝐔) · dφ` inverted by
Sherman-Morrison so the user supplies effective volume fractions
`f_α(τ)` along the chosen `trajectory` (cracks contribute their
density derivative `dε_c/dτ` directly — no Sherman-Morrison
correction needed).  Implemented in elastic, conduction and ALV
pipelines (`src/Schemes/differential.jl`,
`src/Viscoelasticity/schemes_alv_extra.jl::differential_alv`).

`OrdinaryDiffEq` is now a strong dependency.

**New trajectory type `Path(Dict(:phase => τ -> f(τ)))`** for fully
functional incorporation paths.  Derivatives `df_α/dτ` are computed
by `ForwardDiff.derivative` automatically.  The existing
`Proportional()`, `Sequential(order)` and `CustomPath(Dict(:phase => Vector))`
trajectories are preserved for back-compat — internally rewritten
as callables on `τ`.

**`DifferentialScheme` constructor extended** :

```julia
DifferentialScheme(; trajectory = Proportional(),
                     nsteps::Int = 100,
                     abstol::Real = 1e-8,
                     reltol::Real = 1e-6,
                     alg = nothing,    # `nothing` → Tsit5()
                     kwargs...)
```

`nsteps` is now reinterpreted as the **`saveat` density** along `τ`
(the integration step is controlled by `abstol`/`reltol`).  Existing
scripts that pass `nsteps = 50` continue to work.

**ALV cracks in `differential_alv`** : the `# deferred` placeholder
is resolved.  Crack phases now contribute `dε_c/dτ ·
ΔC̃^crack_c(C̃)` to the RHS at every solver step (Sevostianov
interface-stiffness correction propagated automatically when `:Rn`
/ `:Rt` ViscoLaws are attached).

**Demo script** `scripts/46_differential_loading_paths.jl` shows the
genuine path-dependence of DEM : a 3-phase composite reaching the
same target volume fractions `f_α^∞` along four different
trajectories (`Proportional`, two `Sequential` orderings, and a
functional `Path(τ -> τ², 2τ−τ²)`) yields four different effective
moduli at `τ = 1` — the DEM iteration sees a different effective
medium at each infinitesimal phase increment depending on the
incorporation history.

### v0.7.0 — Sevostianov interface stiffness, ECHOES SC body, Newton-Raphson SC

**Sevostianov-style crack interface stiffness** is now supported in
all three pipelines (elasticity, conductivity, ALV).  The COD
compliance becomes `B_eff = B · (𝟙 + b·K·B)^{-1}` with `K` a
3×3 spring-like 2-tensor in elasticity (kept as `:K_interface` on the
crack phase), a scalar Kapitza conductance in conductivity
(`:α_interface`), and a `ViscoLaw`-valued kernel in ALV (a pair
`(:Rn, :Rt)` of normal/tangential relaxation laws).  Validation
scripts `scripts/44_alv_cracks_interface.jl` and
`scripts/45_cracks_iso_interface.jl` cross-check the implementation
against ECHOES C++ via PyCall (≤ 3·10⁻⁴ relative error across MT, SC,
ASC, PCW, Maxwell, Differential at d = 0.30 ALV / d = 0.50 elastic).

**MT and SC schemes now match ECHOES `B·A^{-1}` form for cracks**
(elastic, conduction, ALV).  The textbook symmetric Hill / Budiansky
SC iteration map collapses crack-rich RVEs to the trivial percolated
fixed point and the additive MT for cracks similarly percolates well
below the ECHOES-reported moduli.  Tracing the C++ reference shows
that ECHOES' `compute_strain_Stress` returns `A_α · S_n` for solid
inclusions but the bare `H_c` (no `S_n` factor) for void cracks, so
the trailing `S_n` cancellation only holds for solid-only RVEs.  The
MFH MT and SC bodies now mirror this :

```julia
A_E = (Σ_solids f·sym(A_α(C_n))) · S_n + Σ_cracks ε · sym(H_c(C_n))
B_E = (Σ_solids f·sym(C_α·A_α(C_n))) · S_n     # cracks → 0 (traction-free)
C_eff = B_E · A_E^{-vol}
```

Fix applied symmetrically to MT (`mori_tanaka.jl`), elastic SC
(`self_consistent.jl`), conduction SC (same dispatcher), and ALV SC
(`schemes_alv_sc.jl::_sc_alv_step_echoes_form`).  Brings SC for cracks
from 74 % off ECHOES to 2·10⁻³ at d = 0.50, and ALV SC + interface
stiffness from 1 % off to 1.4·10⁻⁴ at d = 0.30.

**ForwardDiff promoted to a strong dependency.**  The four
`derivative` / `gradient` / `jacobian` / `sensitivity` autodiff
entry points are now available out of the box (no `using ForwardDiff`
needed) — the previous weak extension `MeanFieldHomogenizationForwardDiffExt` is
removed.  This also enables the new built-in Newton-Raphson SC solver
below.

**Built-in Newton-Raphson SC solver** (`SelfConsistent(; algorithm =
NewtonDefault())`).  Replaces the prior weak-extension stub: the
solver now ships with the package and uses `ForwardDiff.jacobian` on
the iso / TI / ortho / aniso canonical components of the running
estimate.  Quadratic convergence (typically 5–10 iterations vs ~100
for `AndersonDefault` Picard) and a backtracking line search make it
robust on high-contrast configurations.  Falls back to a single
Picard step when the line search exhausts.  Available for both
[`SelfConsistent`](@ref) and [`AsymmetricSelfConsistent`](@ref).

**Eigenvalue guard `_sc_pd_guard`** for the SC running estimate :
mirrors ECHOES `homogenization_scheme.h::evaluate` by detecting a
non-positive-definite running estimate and resetting it to a tiny
positive baseline before each step.  Prevents Picard / Newton from
collapsing to the trivial `C = 0` fixed point near the percolation
threshold.

### v0.6.0 — TI ALV fast path, order-2 ALV, BLAS Volterra, ALV cracks roadmap

**TI Walpole-basis fast path** for ALV homogenization : when every
phase 4-tensor and the matrix kernel are TI 4-tensors with the
**common canonical axis n = e₃** (every 6×6 Mandel block matches the
Walpole structure), `homogenize_alv` now routes through new
`*_alv_ti(ℓ_…)` primitives that operate on **6** `n × n` Volterra
matrices `(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)` instead of the generic `(6n × 6n)`
block.  The Walpole 2×2 part `[[ℓ₁, ℓ₃]; [ℓ₄, ℓ₂]]` is packed as a
`(2n)×(2n)` block-Volterra matrix and inverted via
`volterra_inverse(_; block_size = 2)`; the two scalar shears
`(ℓ₅, ℓ₆)` go through the LAPACK scalar fast path below.

ISO inputs are subsumed automatically (iso ⊂ TI), so a TI-matrix +
iso-inclusion combination — common in layered concrete /
fiber-reinforced ALV — gets the fast path "for free".  Storage is 6 ·
n² doubles per phase (vs 36 · n² generic), and the inverse cost drops
from `O((6n)²)` to `O((2n)²) + 2·O(n²) ≈ ×3` cheaper than the generic
6n×6n path.

The TI fast path is integrated into all six schemes (Voigt / Reuss /
Dilute / DiluteDual / Mori-Tanaka / Maxwell), via a new
`_try_ti_tuples` helper analogous to the existing `_try_iso_pairs`
detection.

**Order-2 ALV** : new sub-module covering vector-tensor ageing
viscoelasticity (thermal / electrical conductivity, diffusivity,
permittivity).  Operators are stored as `(3n × 3n)` lower-block-
triangular matrices with 3×3 blocks.  Mirrors the order-4 API:

  * `homogenize_alv_order2(rve, scheme, prop; times)` — public entry
    point, dispatching on Voigt / Reuss / Dilute / DiluteDual /
    Mori-Tanaka / Maxwell schemes.
  * `hill_kernel_order2(ell, K_0_law, times)` — Hill polarization for
    iso ALV matrix + ellipsoidal inclusion (time-space decoupling
    `P̃[block(i,j)] = α₀^{-vol}[i,j] · 𝐈^A`, with `𝐈^A` from the
    existing elastic `tens_IA(ell)`).
  * `iso_order2_params_from_blocks` / `iso_order2_blocks_from_params`
    — per-component parameter extraction (single scalar α for iso
    order-2).
  * `voigt_alv_order2`, `reuss_alv_order2`, `dilute_alv_order2`,
    `dilute_dual_alv_order2`, `mori_tanaka_alv_order2`,
    `maxwell_alv_order2`.

`trapezoidal_matrix(law, times)` now accepts both order-4 (4-tensor /
6×6 Mandel) and order-2 (`TensND.AbstractTens{2,3}` / 3×3 matrix)
sample types, dispatching to the appropriate (B·n)×(B·n) layout.

The order-2 elastic-limit test verifies that ALV reduces to the
existing elastic conductivity `homogenize` to machine precision for
both spherical and spheroidal inclusions.  `scripts/bench_echoes/
bench_order2_alv.jl` provides a template for a Julia–ECHOES
crosscheck on the `fluage_echoes_maxwell_ordre2.py` setup.

**Volterra BLAS / LAPACK fast path** : `volterra_inverse`,
`volterra_left_divide` and `volterra_divide` now dispatch to LAPACK
`trtri` / `trsm` via the `LowerTriangular(...)` wrapper for
`BlasFloat` element types and `n ≥ 64` (the crossover where BLAS
overhead amortizes).  Measured speedups vs the hand-rolled forward
substitution: **×9.7** at `n = 500`, **×14.6** at `n = 1000`.  The
hand-rolled fallback is preserved for small grids and for
non-BlasFloat element types (`BigFloat`, `Sym`, `ForwardDiff.Dual`).

**ALV cracks roadmap** : new file
`src/Viscoelasticity/cracks_alv.jl` documents the planned
`cod_kernel_alv` / `compliance_contribution_alv` API, the time-space
decoupling formulas for pure penny / interface-stiffness cracks in
iso ALV matrices, and the integration points with the existing
`CrackDensity` amount in `Schemes`.  Implementation is scheduled for
v0.6.1.

### v0.5.3 — Non-uniform time grid + multi-layer ALV stability + iso fast path

**Bug fix (membrane interface convention)** : when v0.5.2 introduced
the C++-convention σ-form shear M-matrix, the elastic-limit unit test
for `MembraneInterface` started failing because the membrane jump
expressions and the M-matrix used different angular-component
normalisations.  Resolution: keep the M-matrix in the
**Christensen–Lo / SymPy convention** (matching the elastic
state-space recurrence in `LayeredSpheres`) and derive the analytic
`M^{-1}` from the C++ closed form via the diagonal conjugation

```text
M_C++ = D_row · M_Christ–Lo · D_col,
   D_row = diag(1/2, 1, 1/2, 1),  D_col = diag(1, 1, 2, 2)
⇒ M_Christ–Lo^{-1} = D_col · M_C++^{-1} · D_row
```

This gives the **best of both worlds**: numerical stability of the
closed-form (only `(3κ+4μ)^{-vol}` and `μ^{-vol}` `n×n` Volterra
inverses are needed) under the elastic-compatible mode normalization
where the Christensen–Lo membrane jumps stay correct.  The
`Membrane interface (elastic limit)` test is back to `≤ 1e-10`
tolerance with no convention asterisk.

**Performance — iso-symmetry fast path** : when every phase 4-tensor
and the matrix kernel are iso (`TensISO{4,3}`-valued),
`homogenize_alv` automatically routes through new
`*_alv_iso(αβ_…)` scheme primitives that operate on two scalar
`n × n` Volterra matrices `(α = 3K, β = 2μ)` instead of the generic
`(6n × 6n)` Mandel block matrix.  Theoretical speedups :

  - matrix-matrix product : ~108× cheaper (`216 n³` → `2 n³`)
  - matrix inverse        : ~18× cheaper
  - storage               : 18× smaller

Measured speedup on the script-37 setup (5 phases + Maxwell matrix +
spherical inclusions, MT scheme): **×2.5–7** depending on `n_times`.
The detection happens once per phase via a cheap iso-form pattern
check (`_is_iso_block`); on failure the generic 6n×6n path is
selected automatically — no API change.  New internal helpers
(`iso_schemes_alv.jl`) :
`voigt_alv_iso`, `reuss_alv_iso`, `dilute_alv_iso`,
`dilute_dual_alv_iso`, `mori_tanaka_alv_iso`, `maxwell_alv_iso`,
plus `dilute_concentration_alv_iso`, `dilute_contribution_alv_iso`
used inside `_inclusion_alv_quantities`.

**Bug fix (multi-layer ALV stability)** : on a non-uniform time grid
(e.g. `logspace`) with a matrix relaxation kernel that has multiple
time constants and/or layers with extreme modulus contrast (pores,
step-activated `ViscoLaw`s), the layered-sphere ALV recurrence
diverged from the ECHOES Python reference by 1e-3 to 1e-2.  Two
compounding root causes :

1. **Right vs left Volterra divide.**  The Hervé–Zaoui closed-form
   interface transition `T = M_b^{-1} · M_a` requires the Volterra
   inverse on the **left** of the numerator.  Our `volterra_divide`
   implemented `num · S^{-vol}` (right) ; the two products are equal
   only when `[num, S] = 0`.  Lower-triangular Volterra trapezoidal
   matrices commute pairwise iff they are Toeplitz (uniform time
   grid + same kernel structure) — non-uniform grids broke this
   assumption silently.

2. **Generic 4n×4n inversion of the shear M-matrix is FP-unstable
   for soft phases.**  Even the block forward-substitution
   `volterra_inverse(_; block_size = 4)` collapses when
   `det(M[t,t]) → 0` (pore-like or step-activated layers).
   ECHOES C++ uses a **closed-form analytic 4×4 inverse** whose only
   `n × n` Volterra inverses are `(3κ + 4μ)^{-vol}` and `μ^{-vol}` —
   both regular for any non-vacuum modulus.

**Fix** :

- Added `volterra_left_divide(S, M; block_size = 1|6)` (forward
  substitution on `S · T = M`, rows i = j..n) and switched every
  closed-form transition (perfect / spring / membrane, both bulk and
  shear) to use it.
- Added `_shear_M_inverse_alv(r, M_κ, M_μ, n)` returning the
  closed-form `M(r; κ, μ)^{-1}` in time-major 4n×4n layout, mirroring
  C++ `inclusion_sphere_nlayers.h::set_visco_inv_matrix_dev` and then
  conjugated to the Christensen–Lo convention.  Used by
  `_shear_layer_transfer_alv` (intra-layer transfer) and
  `_shear_amp_blocks_alv` (state → amplitude extraction).
- Reverted the v0.5.2 τ-scaling for the shear M-matrix — the
  closed-form inverse is naturally written in σ-form and FP
  stability now comes from the closed form rather than the rescaling.

**Impact** : `script 37 :layers` now produces smooth, monotonic
creep curves matching the Python reference figure visually
(bounded between elastic limit and matrix Maxwell), in place of
the previous unbounded / oscillating output.  Bench results :

- `bench_layered_alv.jl` (N=2 stiff elastic + Maxwell matrix,
  uniform grid) : 1e-16 (unchanged).
- `bench_layered_alv_step.jl` (N=3 step-activated layers + pore +
  Maxwell matrix, non-uniform grid) : bulk α 1e-15, shear β 1e-14.
- `bench_step_n2.jl` (N=2 step layers, no pore) : 1e-14.
- `bench_layered_alv_nopore.jl` (N=4 elastic, varied moduli) : 1e-15.

### v0.5.1 — Multi-layer sphere shear localization bug fix

**Bug fix** : on a non-uniform time grid (e.g. `logspace`) with a
matrix relaxation kernel that has multiple time constants and/or
layers with extreme modulus contrast (pores, step-activated
`ViscoLaw`s), the layered-sphere ALV recurrence diverged from the
ECHOES Python reference by 1e-3 to 1e-2.  Two compounding root
causes :

1. **Right vs left Volterra divide.**  The Hervé–Zaoui closed-form
   interface transition `T = M_b^{-1} · M_a` requires the Volterra
   inverse on the **left** of the numerator.  Our `volterra_divide`
   implemented `num · S^{-vol}` (right) ; the two products are equal
   only when `[num, S] = 0`.  Lower-triangular Volterra trapezoidal
   matrices commute pairwise iff they are Toeplitz (uniform time
   grid + same kernel structure) — non-uniform grids broke this
   assumption silently.

2. **Generic 4n×4n inversion of the shear M-matrix is FP-unstable
   for soft phases.**  Even the block forward-substitution
   `volterra_inverse(_; block_size = 4)` collapses when
   `det(M[t,t]) → 0` (pore-like or step-activated layers).
   ECHOES C++ uses a **closed-form analytic 4×4 inverse** whose only
   `n × n` Volterra inverses are `(3κ + 4μ)^{-vol}` and `μ^{-vol}` —
   both regular for any non-vacuum modulus.

**Fix** :

- Added `volterra_left_divide(S, M; block_size = 1|6)` (forward
  substitution on `S · T = M`, rows i = j..n) and switched every
  closed-form transition (perfect / spring / membrane, both bulk and
  shear) to use it.
- Added `_shear_M_inverse_alv(r, M_κ, M_μ, n)` returning the
  closed-form `M(r; κ, μ)^{-1}` in time-major 4n×4n layout, mirroring
  C++ `inclusion_sphere_nlayers.h::set_visco_inv_matrix_dev`.  Used
  by `_shear_layer_transfer_alv` (intra-layer transfer) and
  `_shear_amp_blocks_alv` (state → amplitude extraction).
- Reverted the v0.5.2 τ-scaling for the shear M-matrix — the
  closed-form inverse is naturally written in σ-form and FP
  stability now comes from the closed form rather than the rescaling.
- Rewrote `_shear_M_matrix_alv` to use the C++ ECHOES mode
  normalization (mode 1 contributes `U = a · r`, not Christensen–Lo's
  `2a · r`) so it stays consistent with the analytic `M^{-1}`
  formula.  Mode-2 dev contribution factor
  `F_k = (21/5) μ^{-vol} (3κ + μ) (r_b⁵ − r_a⁵)/(r_b³ − r_a³)` was
  unchanged ; it cancels the mode-2 amplitude scaling implicitly.

**Impact** : `script 37 :layers` now produces smooth, monotonic
creep curves matching the Python reference figure visually
(bounded between elastic limit and matrix Maxwell), in place of
the previous unbounded / oscillating output.  Bench results :

- `bench_layered_alv.jl` (N=2 stiff elastic + Maxwell matrix,
  uniform grid) : 1e-16 (unchanged).
- `bench_layered_alv_step.jl` (N=3 step-activated layers + pore +
  Maxwell matrix, non-uniform grid) : bulk α 1e-15, shear β 1e-14.
- `bench_step_n2.jl` (N=2 step layers, no pore) : 1e-14.
- `bench_layered_alv_nopore.jl` (N=4 elastic, varied moduli) : 1e-15.

### v0.5.1 — Multi-layer sphere shear localization bug fix

**Bug fix** : `LayeredSpheres._shear_localization` and the
companion ALV `shear_localization_alv` previously returned only the
mode-1 amplitude `a_k`, which is the correct per-layer dev β only
when `b_k` (mode-2 amplitude) vanishes — true for `N = 1` (single
sphere) and for the degenerate `N = 2` cases tested in
`test_christensen.jl` (shell ≡ matrix or core ≡ shell).  For
genuinely multi-layer composite spheres with distinct core, shell
and matrix moduli, `b_k` is non-zero and contributes to the
volume-averaged deviatoric strain via the mode-2 r³ profile.

The corrected per-layer dev localization is

```text
β_k = a_k + b_k · (21/5) (3κ_k + μ_k)/μ_k · (r_k⁵ − r_{k-1}⁵)/(r_k³ − r_{k-1}³)
```

(modes 3 and 4 contribute zero to the layer-volume-averaged
deviatoric strain by angular orthogonality).  This matches ECHOES
C++ `inclusion_sphere_nlayers.h::get_visco_layer_average_strain_Strain`
to machine precision.

**Impact** : the ALV `:layers` topology of `script 37` now matches
the Python `fluage_echoes_solid.py` reference for N ≥ 2.  The
elastic `stiffness_contribution(LayeredSphere, C₀)` and the ALV
`stiffness_contribution_alv(LayeredSphere, C₀_law, times)` produce
the correct effective dilute / MT moduli.

**Validation** : new cross-check benchmark
`scripts/bench_echoes/bench_layered_alv.{py,jl}` and a regression
test `shear_localization_alv — N=2 cross-check vs ECHOES Python` in
`test/Viscoelasticity/test_layered_alv.jl` pin the ALV per-layer
α(t,t') and β(t,t') Volterra blocks to ECHOES Python at machine
precision (1e-16 on the diagonal, 1e-6 on the off-diagonal blocks).

### v0.5.0 — Ageing linear viscoelasticity (ALV) module

A new `MeanFieldHomogenization.Viscoelasticity` sub-module brings time-domain
viscoelastic homogenization to the package, mirroring the capabilities
of the C++ ECHOES `viscoelasticity/visco_law.h` and
`homogenization_maxwell.h`.  Reference: Sanahuja IJSS 2013 ;
Barthélémy-Giraud-Lavergne-Sanahuja IJSS 2016 ;
Barthélémy-Giraud-Sanahuja-Sevostianov IJES 2019 ; ECHOES manual
chapter 7 and appendix `viscoelastic_hill_kernel.qmd`.

#### Highlights

- **`ViscoLaw`** : abstract relaxation `R(t,t')` or creep `J(t,t')`
  kernel, scalar- or 4-tensor-valued, with built-in convenience
  constructors `maxwell_relaxation`, `kelvin_creep`, `maxwell_iso`,
  `kelvin_iso`, `heaviside_law`.
- **`trapezoidal_matrix`** : Sanahuja-2013 trapezoidal discretization of
  the Stieltjes integral on a time grid `times`, returning a dense
  `Matrix{T}` of size `(B·n) × (B·n)` in lower-block-triangular form
  (`B = 6` for 4-tensor in Mandel convention, `B = 1` for scalar
  kernels).
- **`volterra_inverse`** : block-triangular forward-substitution that
  takes a discrete relaxation matrix to its discrete creep matrix in
  `O(B³ n²)` flops.
- **`hill_kernel`** : discrete ALV Hill polarization tensor for an
  ellipsoidal inclusion in an isotropic ALV matrix, using the
  time-space decoupling formula of the manual appendix : reuses the
  elastic auxiliary tensors `tens_UA`, `tens_VA` and combines them with
  two scalar Volterra inverses (longitudinal and shear moduli).
  Machine-precision agreement with the elastic Hill tensor in the
  Heaviside (elastic) limit.
- **`homogenize_alv(rve, scheme, prop; times)`** : public entry point
  that builds the discrete operators for every phase, computes the
  ALV Hill kernel, and dispatches to the appropriate scheme function.
  Implemented schemes : `Voigt`, `Reuss`, `Dilute`, `DiluteDual`,
  `MoriTanaka`, `Maxwell`.  Each one's output coincides with the
  corresponding elastic homogenization in the Heaviside limit (verified
  to machine precision in the test suite).
- **`Phase.properties` relaxed to `Dict{Symbol, Any}`** : a phase can now
  carry either an elastic `AbstractTens` or a `ViscoLaw` under the
  same key (`:C`).  No regression in the elastic test suite (3421/3421).
- **Scripts** : `scripts/33_visco_law_basics.jl` (kernels + plot),
  `scripts/37_fluage_echoes_solid.jl` (Sanahuja-style ageing creep of a
  solidifying composite, whole-pores topology, mirroring
  `tests/python/creep/fluage_echoes_solid.py` after [@sanahuja2013] and
  chapter 9 §"Ageing creep of solidifying cementitious materials").
- **Tests** : `test/Viscoelasticity/` adds 525 new tests across
  `test_visco_law.jl`, `test_trapezoidal.jl`, `test_volterra_inverse.jl`,
  `test_hill_alv_iso.jl`, `test_schemes_alv.jl`.  Total package test
  count : 3946/3946 PASS.

#### Self-Consistent ALV (added in 0.5.0)

- **`self_consistent_alv(rve, prop; times, abstol, reltol, maxiters,
  damping, verbose, select_best)`** — symmetric SC fixed-point iteration
  on the `(6n × 6n)` block matrix.  Each iteration recomputes the
  per-phase Hill kernels using the running estimate's iso parameters,
  computes the dilute concentration tensors, and forms the next
  iterate.  Convergence on the Frobenius norm.
- Plumbed into the dispatcher via `homogenize_alv(rve, SelfConsistent(),
  :C; times = T)`.
- Tests against the elastic SC limit pass at machine precision.

#### N-layer sphere ALV — full bulk + shear recurrence (added in 0.5.0)

- **`bulk_localization_alv(sphere::LayeredSphere, C0_law, times)`** —
  per-layer bulk localization matrices `α_k(t,t')` of size `n × n`
  (one per layer).  Extends the elastic Hervé-Zaoui bulk recurrence
  ([`LayeredSpheres/bulk_recurrence.jl`]) by replacing every scalar
  modulus with its `n × n` trapezoidal Volterra matrix, building
  `(2n × 2n)` block transfer matrices.
- **`shear_localization_alv(sphere::LayeredSphere, C0_law, times)`** —
  per-layer deviatoric (Y₂-harmonic) localization matrices `β_k(t,t')`
  of size `n × n`.  Builds the Hervé-Zaoui 1993 4×4 fundamental matrix
  in **time-major** layout (`(4n × 4n)` block-lower-triangular with
  4×4 diagonal blocks), inverts it via `volterra_inverse(_;
  block_size = 4)`, propagates two probe states through the layers,
  and selects the linear combination matching unit far-field
  `(a_{N+1}, b_{N+1}) = (I_n, 0)` via a final `(2n × 2n)`
  `block_size = 2` Volterra solve.  Verified to machine precision
  against `LayeredSpheres._shear_localization` in the Heaviside
  (elastic) limit.
- **`bulk_state_seq_alv`** / `_shear_state_seq_alv` — forward
  propagation of the discrete state vectors through every layer.
- **ALV interface transfers** (`_bulk_interface_T_alv`,
  `_shear_interface_T_alv`) cover the same set of imperfect interface
  models as the elastic counterpart : `PerfectInterface`,
  `SpringInterface(kn, kt)` (primal — displacement jump driven by
  `kn`/`kt`) and `MembraneInterface(κs, μs)` (dual — traction jump
  from surface elasticity).  Each interface parameter may be a plain
  scalar (constant in time, the elastic limit) **or** a `ViscoLaw`
  scalar kernel — in the latter case the jump itself is ageing and
  the corresponding `n × n` block is the parameter's trapezoidal
  matrix.  The `(4n × 4n)` shear block-diagonal (4×4 sense) cleanly
  reduces to the elastic 4×4 jump for scalar parameters.
- **Composite-sphere assembly**:
  `strain_strain_loc_alv(sphere, C0_law, times)` builds the volume-
  averaged strain-strain localization tensor
  `A_avg = ⟨α⟩ 𝕁 + ⟨β⟩ 𝕂` (`(6n × 6n)`), and
  `stiffness_contribution_alv(sphere, C0_law, times)` builds the
  size-independent stiffness contribution
  `N = 3 Σ_k f_k (M_κ_k − M_κ_0) ∘ α_k 𝕁
       + 2 Σ_k f_k (M_μ_k − M_μ_0) ∘ β_k 𝕂` (`(6n × 6n)`).
- **`homogenize_alv` extended to `LayeredSphere` phases**: the
  per-inclusion quantities (`A_dut`, `N_dut`) are computed via the
  layered-sphere recurrence instead of the Hill kernel + dilute
  pipeline.  Verified to machine precision against the elastic
  reference for the Dilute scheme and against the elastic MT for the
  `t = t' = 0` block of a Maxwell relaxation kernel.
- **`scripts/37_fluage_echoes_solid.jl`** now exposes a `MODEL`
  constant (`:whole_pores` / `:layers`) selecting the topology;
  the `:layers` branch reproduces the Python `sphere_nlayers(...)`
  setup of `tests/python/creep/fluage_echoes_solid.py` exactly.

#### Deferred to follow-up

- Cracks in ALV (extrapolating from the elastic `cod_tensor` /
  `compliance_contribution` infrastructure).
- Self-Consistent ALV with `LayeredSphere` phases (the current SC
  ALV iteration handles only `Ellipsoid`-geometry inclusions).
- Anisotropic ALV Hill kernel (numerical surface integral with Volterra
  inverse of the 3×3 acoustic tensor at each integration point).

### v0.4.0 — Friendly autodiff sensitivities, RVE-level symmetrize, Hill-symmetric SC

A small but expressive API exposing `ForwardDiff`-based derivatives of any
homogenization result with respect to any scalar input parameter — physical
(stiffness coefficient, conductivity), geometric (radii, semi-axes, crack
opening, distribution-shape envelope) or volume-fraction / crack-density —
*and* for arbitrary scalar fields of inclusion types defined later by the
user. The autodiff path unlocks geometric and user-type sensitivities that
were not previously practical, and the multi-scale chain rule is taken care
of automatically by composing several `homogenize` calls inside a single
closure.

The release also ships an RVE-level orientation-distribution projection
(`symmetrize`), a corrected Hill-symmetric self-consistent scheme that
percolates exactly at φ=0.5 for spherical pores, a `Spheroid` convenience
constructor, Dual-stable SC convergence, and a `select_best` mode that
mirrors the C++ reference's behavior at percolation thresholds.

#### Additions

- **Lens hierarchy** `AbstractParameter` with four concrete kinds
  (`AmountParameter`, `PropertyParameter`, `GeometryParameter`,
  `DistributionShapeParameter`) plus user-friendly helpers (`amount`,
  `property`, `geometry`, `shape_param`).
- **`get_param(rve, p)` / `set_param(rve, p, value)`** — read / immutable
  update of the scalar designated by a lens, with automatic eltype
  promotion to integrate `ForwardDiff.Dual` cleanly.
- **Public autodiff entry points** `derivative`, `gradient`, `jacobian`
  and the closure fallback `sensitivity(f, x₀)`. They become available
  after `using ForwardDiff` (weak extension `MeanFieldHomogenizationForwardDiffExt`).
- **Generic geometry-field reflection** `_replace_geom_field` based on
  `@generated` reconstruction with uniform sibling-field eltype
  promotion. User-defined inclusions whose constructor follows the
  parametric Julia auto-generated pattern (`MyType{T,B}(args...)`) are
  differentiable through their scalar fields without any library change.
- **Symbol selectors for property tensors** mapping named coefficients
  (`:bulk`, `:shear`, `:transverse`, `:axial`, `:ℓ₁`..`:ℓ₆`) to the
  positional indices of `get_data(tensor)` for `TensISO{2}`,
  `TensISO{4,3}`, `TensTI{2}` and `TensTI{4}`.
- **`MeanFieldHomogenizationForwardDiffExt`** weak extension activating the public
  API on `using ForwardDiff`. ForwardDiff is registered in `[weakdeps]`
  alongside NonlinearSolve and SymPy; no new hard dependency.
- **RVE-level orientation symmetrize** via the `symmetrize` keyword on
  `add_matrix!` / `add_phase!`. Three options:
  - `:none` (default): no projection.
  - `:iso`: Reynolds average over `SO(3)` ⇒ isotropic contribution
    (`TensISO`).
  - `:ti` / `TISymmetrize(axis)`: Reynolds average over rotations around
    `axis` ⇒ transversely-isotropic contribution (`TensTI(axis)`).
  Implemented for tensor orders 2 and 4. The TI projection currently
  routes the matrix through an iso projection during the
  localization-tensor computation (workaround for non-coaxial inclusion
  families); see [`src/Schemes/symmetrize.jl`](src/Schemes/symmetrize.jl)
  for the rationale.
- **`Spheroid(ω; euler_angles)`** convenience constructor on top of
  `Ellipsoid`, mirroring the `spheroidal(omega)` helper of the C++
  reference: `ω = c/a` with one polar semi-axis equal to `ω` and two
  equatorial ones equal to `1`. Eshelby/Hill computations are
  scale-invariant so only the aspect ratio matters.
- **`select_best` keyword on the SC fixed-point solver** — when `true`,
  the solver tracks the best iterate seen during Picard iteration
  (smallest residual on the value field) and returns it at the end.
  Useful for high-contrast iterations that oscillate around the fixed
  point near percolation thresholds; matches the C++ reference's
  `select_best=True` mode.

#### Fixes

- **Hill-symmetric self-consistent**: every phase now contributes a
  non-trivial dilute concentration `A_α = inv(I + P(C_α − C_eff))`
  computed in the iterating effective medium, including the matrix
  phase. The previous SC step treated the matrix as having `A = I`
  (Mori-Tanaka-style), which gave the upper SC branch only and
  misplaced the porous-sphere percolation threshold. With the fix,
  porous spheres percolate exactly at φ = 0.5.
- **Dual-stable SC convergence criterion** — the Picard convergence
  test now requires both the value AND every partial of the residual to
  fall below `abstol`. Without that, the value can converge while the
  partials carry residual error of order `‖∂step/∂x‖ × abstol`,
  producing numerically wrong sensitivities through the SC fixed point.
- **TI symmetrize Walpole normalization**: the `_apply_symmetrize` for
  `TISymmetrize` now divides the W₅ and W₆ projection coefficients by
  `‖W_k‖² = 2`, matching the basis-decomposition convention of
  `TensND.TensTI{4}`. Round-trip on a coaxial TI(ez) tensor is now
  exact.

#### Documentation

New manual page `manual/sensitivities.md` (motivation, lens API, closure
fallback, user-inclusion tutorial, multi-scale chain-rule example, and a
section on the `symmetrize` keyword) and auto-API page
`api/sensitivities.md`. Both wired into `docs/make.jl`.

#### Scripts

- `scripts/26_sensitivities.jl` — tour of the API (lenses + gradient +
  jacobian + cross-check vs the Christensen 1990 closed form for
  `∂k_MT/∂f`, agreement to ~1e-16).
- `scripts/27_user_inclusion_sensitivity.jl` — extensibility demo on a
  user-defined inclusion type `MyBlob{T,B}` with two numeric fields
  (`radius`, `eccentricity`).
- `scripts/28_multiscale_strength.jl` — three-scale upscaling of
  cement-paste / mortar elasticity and quasi-brittle compression
  strength following Pichler & Hellmich 2011 (SC + MT + MT). The single
  iso hydrate phase + global-μ autodiff approximation matches the
  effective moduli (k, μ, E) of the reference Python implementation to
  rtol ≈ 1e-3 across the (wc, α) grid.
- `scripts/29_porous_schemes.jl` — porous benchmark across all ten
  schemes (sphere and oblate ω = 0.2 with iso symmetrize). After the
  Hill-symmetric SC fix, spherical-pore SC percolates exactly at φ=0.5.
- `scripts/bench_echoes/benchmark_porous.jl`, `benchmark_pichler.jl`
  — PyCall cross-validation against the C++ reference; ten schemes ×
  two cases (sphere / oblate) for porous, six wc curves × twelve α
  points for Pichler. The moduli match the reference to rtol_mod ≈ 1e-3
  across both benchmarks.

#### Tests

Three new cross-cutting test files:

- `test_parameters.jl` (round-trip, type-promotion, no-mutation
  invariants on every lens kind),
- `test_sensitivities.jl` (FD vs autodiff cross-check on every scheme,
  closed-form Christensen 1990 match to `~1e-12`, closure fallback,
  `MyBlob` user-inclusion demonstration),
- `test_symmetrize.jl` (round-trip iso/TI projections on 2nd- and
  4th-order tensors, TI(ez) coaxial preservation, integration with
  `homogenize`).

Total: 3421 tests pass.

#### Breaking changes

- **SC results differ for systems near percolation** because of the
  Hill-symmetric SC fix. The pre-v0.4 SC step treated the matrix as
  Mori-Tanaka-style (A = I) and therefore selected the upper branch
  unconditionally. The new step is the textbook Hill / Budiansky 1965
  symmetric SC. Users who relied on the old upper-branch behavior for
  porous-sphere systems should switch to `MoriTanaka` (which is also
  not broken by the fix).
- **`homogenize` API** keeps the `homogenize(rve, scheme; property=:C)`
  kwarg form for backward compatibility but the recommended signature
  is now `homogenize(rve, scheme, property::Symbol)` with `property`
  required and positional.

#### Notes

- `Complex{T}` autodiff is not supported (ForwardDiff does not mix Dual
  and Complex cleanly). Symbolic differentiation goes through SymPy on
  the closed-form schemes directly (already supported).
- The `AsymmetricSelfConsistent` scheme follows the symmetric-SC fixed
  point in compliance space when matrix is stiff. The C++ reference's
  ASC uses a different formulation (compliance-form Mori-Tanaka with
  iterating reference) which converges to a different branch for
  porous oblate systems away from percolation; this is documented in
  `scripts/bench_echoes/benchmark_porous.jl`.

### v0.3.0 — RVE container + 10 homogenization schemes

New `MeanFieldHomogenization.Schemes` sub-module: a Representative Volume Element
container plus the ten classical mean-field homogenization schemes ported
from C++ ECHOES, with a few Julia-idiomatic improvements.

#### Additions

- **`RVE`** container with `add_matrix!`, `add_phase!`, helpers
  (`matrix_phase`, `inclusion_phase_names`, `phase_property`,
  `volume_fraction`, `crack_density`, `matrix_volume_fraction`,
  `validate_rve`).  Volume fractions are stored at the RVE level rather
  than on the inclusions — a single inclusion remains usable for
  localization-tensor calculations without any RVE machinery.
- **`AbstractAmount`** hierarchy with `VolumeFraction` (solid inclusions)
  and `CrackDensity` (flat cracks); the matrix amount is implicit
  (`1 − Σ f_inc`) and crack densities are excluded from that sum.
- **`AbstractDistributionShape`** hierarchy with `UniformDistribution`
  (single outer envelope, current behavior); leaves an extension hook
  for a future `PairwiseDistribution` (Willis 1982) without breaking
  the public API.
- **Ten homogenization schemes**: `Voigt`, `Reuss`, `Dilute`,
  `DiluteDual`, `MoriTanaka`, `Maxwell`, `PonteCastanedaWillis`,
  `SelfConsistent`, `AsymmetricSelfConsistent`, `DifferentialScheme`.
- **`homogenize(rve, scheme; property=:C)`** central entry point.  The
  scheme can be a type instance (`MoriTanaka()`,
  `SelfConsistent(algorithm=NewtonRaphson(), abstol=1e-12)`) or a
  `Symbol` shortcut. Canonical Symbol aliases are lowercase
  (`:mt`, `:sc`, `:diff`, …) for consistency with the algorithm-method
  symbols (`:auto`, `:residues`, `:decuhr`); CamelCase and ECHOES
  upper-case codes (`:MT`, `:DIFF`, …) are kept as backwards-compatible
  aliases.
- **Differential trajectories**: `Proportional` (default), `Sequential`
  (phase-by-phase), `CustomPath` (per-phase explicit trajectory) — all
  validated for monotonicity and boundary conditions.
- **SciML weak extension** `MeanFieldHomogenizationNonlinearSolveExt` (active once
  `NonlinearSolve.jl` is loaded into the session) makes every algorithm
  of `NonlinearSolve.jl` available to `SelfConsistent` /
  `AsymmetricSelfConsistent` via the `algorithm` keyword, through a
  `ForwardDiff`-safe implicit-function-theorem lift (no nested `Dual`s,
  regardless of the chosen algorithm). The auto-resolving
  `AutoNonlinear` marker selects a globalized SciML algorithm
  (`TrustRegion`) when the extension is active and falls back to the
  built-in `NewtonDefault` otherwise — it is not the default of
  `SelfConsistent` / `AsymmetricSelfConsistent` (which remains
  `AndersonDefault`, more robust through the porous-percolation
  bifurcation). See the
  [nonlinear solvers tutorial](docs/src/tutorials/12_nonlinear_solvers.md).
- **Conductivity (`property = :K`)** is supported by every scheme
  through 2nd-order tensor algebra (gradient-gradient localization,
  resistivity contributions for cracks).

#### Number-type compatibility

Every new scheme is fully `ForwardDiff.Dual` and `Complex{Float64}`
compatible (frequency-domain viscoelasticity); symbolic `Sym` / `Num`
work on the closed-form schemes (Voigt, Reuss, Dilute, DiluteDual,
Mori-Tanaka, Maxwell, PCW). The asymmetric SC heuristic uses the
Inf-norm rather than the SVD-based 2-norm so it works seamlessly under
`Dual`.

#### Documentation

New theory page `theory/homogenization.md`, manual page
`manual/schemes.md`, API page `api/schemes.md`. Bibliography augmented
with `mori1973`, `christensen1990`, `mclaughlin1977`, `norris1985`,
`ponte1995`, `willis1982`. New scripts `scripts/20_voigt_reuss_bounds.jl`
through `scripts/25_echoes_crosscheck.jl`. The latter cross-validates
Mori-Tanaka against the [Christensen 1990](@cite christensen1990) closed
form (exact match to 6 sig. figs. on bulk and shear at five fractions).

#### Tests

Around 270 new tests covering construction, numerical bounds, closed
forms, Dual sensitivity (every scheme), Complex moduli sweep, Symbol
shortcuts, and CustomPath validation. Total 3312 tests passing.

### v0.2.0 — alignment with TensND 0.2 (breaking)

Follow-up to TensND 0.2's API unification. MeanFieldHomogenization is iso-functional —
all outputs are unchanged — but every mention of a TensND symbol now uses
the new snake_case + UPPERCASE-acronym convention.

#### Breaking changes

- `TensND.TensWalpole` references (type annotations, dispatch rules,
  constructor calls) now use `TensND.TensTI{4}`.  The struct layout is
  identical so numerical behavior is unchanged.
- Accessor renames propagated from TensND: `getbasis` → `get_basis`,
  `tensbasis` → `tens_basis`, `invKM` → `inv_KM`, `getdata` → `get_data`,
  `getarray` → `get_array`, `getvar` → `get_var`, `getdim` → `get_dim`,
  `getorder` → `get_order`.
- Predicate renames: `isISO` → `is_ISO`, `isTI` → `is_TI`,
  `isOrtho` → `is_ORTHO`.
- Tensor factory renames in scripts and docs: `tensId2` → `tens_Id2`,
  `tensJ4` → `tens_J4`, `tensTI` → `tens_TI`, etc.

#### Additions

None — functional surface unchanged.

#### Migration guide

If you have your own code depending on MeanFieldHomogenization dispatch, apply the
same renames as listed in TensND's v0.2 changelog. All MeanFieldHomogenization tests
(2865) pass without behavioral change after migration.
