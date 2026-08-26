# `scripts/` — MeanFieldHomogenization.jl demos & validation

Numbered demonstration / validation scripts, grouped in blocks by theme.
Each is self-contained and, where relevant, states the reference benchmark it
reproduces.

## Which environment a script activates

Every script here activates **`docs/`**:

```julia
import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)         #jl
```

That is deliberate, and it is not the package's own environment. Most of these
scripts use `Plots`, and several use `SymPy`, `Symbolics` or `DECUHR` — none of
which the package environment can load: `Plots` is not a dependency of
`MeanFieldHomogenization` at all, and the other three are *weak* dependencies, so
they are unavailable from `Project.toml` too. Activating `..` therefore only ever
worked by falling back to whatever the machine happened to have in its global
environment: on a fresh clone, `julia scripts/30_average_nlayers.jl` failed on
`using Plots`.

`docs/` is the one environment in this repository guaranteed to load everything a
published script uses — that is precisely what lets Literate execute the notebook
version of the same file. Running standalone and running inside the build are
then the same statement, instead of two that can drift apart.

The four finite-element scripts are the exception: `Ferrite`, `FerriteGmsh` and
`Gmsh` are not in `docs/` either, so `81`, `83`, `88` and `89` activate and
instantiate [`scripts/fe/`](fe/) instead, keeping `gmsh_jll` out of every
documentation build.

The two cross-validation scripts that `pyimport("echoes")` through `PyCall`
(`15`, `60`) need a Python environment with `echoes` installed, and depend
on the machine whichever project is active. They are validation harnesses, not
demos. (`52` used to be a third: it reached for a Python Mittag-Leffler module,
and no longer needs to — the Rabotnov kernel is in the model library, whose
Laplace-Carson transform involves no special function.)

Shared code lives in [`common/`](common/) — currently the Pichler-Hellmich
three-scale model (`common/quasibrittle_strength.jl`), used by both the demo script
`41_multiscale_strength.jl` and the cross-check
`bench_echoes/benchmark_strength.jl`.

## Numbering blocks

| Block | Theme |
|---|---|
| 01–09 | Tensor / Hill / Eshelby toolbox |
| 10–19 | Cracks & COD (16–19 reserved for future conductive / resistive conduction cracks) |
| 20–29 | Elastic homogenization schemes |
| 30–39 | Layered n-layer sphere / spheroid, periodic multilayer |
| 40–49 | Strength & multiscale (Pichler-Hellmich, Lavergne) |
| 50–59 | Viscoelasticity & ALV |
| 60–69 | ALV cracks, interfaces, cross-validations & the Laplace-Carson route |
| 70–79 | Symmetrization showcases |
| 80–89 | Custom (user-defined) inclusions, finite-element and neural-surrogate coupling |
| 90–99 | Interacting particle assemblies: EIM & cluster model |

## Coverage map

`—` = no direct reference benchmark (native demonstration).

### 01–09 Tensor toolbox
| Script | reference / topic | Notes |
|---|---|---|
| `01_auxiliary_tensors.jl` | — | geometric tensors `tens_IA/UA/VA` |
| `02_hill_elasticity.jl` | `eshelby`/`hill` API | **published tutorial** — `hill_tensor` on sphere / prolate / oblate / triaxial, `:residues` vs `:nestedquadgk` on a cubic matrix, the Eshelby tensor against Eshelby (1957), and a dilute estimate checked three ways |
| `03_hill_conductivity.jl` | 2nd-order `hill` | conductivity Hill |
| `04_forwarddiff.jl` | — | AD through Hill tensors |
| `05_symbolic.jl` | — | SymPy genericity |
| `06_cylinder.jl` | cylinder Hill | transverse-plane quadrature |
| `07_hill_ti_coaxial.jl` | `hill(...,TI)` | Barthélémy 2020 TI-coaxial closed form |
| `08_hill_derivatives.jl` | `hill_derivative` | ∂P/∂C by ForwardDiff (ISO, TI), validated vs finite differences |
| `09_cod_symbolic_green.jl` | `barthelemySifAniso` | SymPy, end-to-end derivation of the COD tensor 𝐁 from the Fourier Green operator: the ξ₃ integral of `Q̂*ₙₙ`, the crack-plane integral, ISO and aligned-TI closed forms recovered (σᵞ included), oracles = `cod_tensor` |

### 10–19 Cracks & COD
| Script | reference / topic | Notes |
|---|---|---|
| `10_cod_isotropic.jl` | `crack_compliance` (iso) | COD / H tensor |
| `11_cod_TI.jl` | `crack_compliance` (TI) | Hoenig / Kanaun-Levin |
| `12_cod_aniso_residue.jl` | `crack_compliance(...,RESIDUES)` | general anisotropy |
| `13_cod_ribbon.jl` | ribbon crack | 2D ribbon COD |
| `14_sif_computation.jl` | — | stress/displacement intensity factors |
| `15_cracks_iso_interface.jl` | iso cracks + spring interface | Sevostianov spring interface |
| `16_cod_symbolic_thermal.jl` | `barthelemySifAniso` (transport twin) | SymPy, end-to-end derivation of the **thermal** COD scalar from the order-2 Green operator. The acoustic form is a scalar, so the ξₙ integral closes at **full anisotropy**: `Q̂*ₙₙ = ½√[(n̲∧ξ̲*)·adj(K₀)·(n̲∧ξ̲*)]`, and the contour integral reduces to an effective ellipse (`√λ₁`, `η′`). Cross-checked against the Hill flattening limit and the textbook penny jump |

### 20–29 Elastic schemes
| Script | reference / topic | Notes |
|---|---|---|
| `20_voigt_reuss_bounds.jl` | VOIGT/REUSS | bounds |
| `21_dilute_vs_mori_tanaka.jl` | DIL/MT | dilute vs MT |
| `22_self_consistent_porous.jl` | SC | porous SC percolation |
| `23_differential_trajectories.jl` | DIFF | Norris DEM trajectories |
| `24_differential_loading_paths.jl` | DIFF | path-dependence demo |
| `25_echoes_crosscheck.jl` | Christensen 1990 | cross-check |
| `26_sensitivities.jl` | `homogenize_derivative` | AD sensitivities tour |
| `27_user_inclusion_sensitivity.jl` | — | user-defined inclusion + AD |
| `28_porous_schemes.jl` | porous benchmark | porous scheme comparison |
| `29_symbolic_schemes.jl` | — | SymPy/Symbolics closed forms: Eshelby/Hill, dilute, MT, porous/rigid limits, hand-derived self-consistent |

### 30–39 Layered n-layer sphere / spheroid, periodic multilayer
| Script | reference / topic | Notes |
|---|---|---|
| `30_average_nlayers.jl` | n-layer sphere | volume-average concentration (sphere) |
| `31_local_nlayers.jl` | n-layer sphere | pointwise localization fields (sphere) |
| `32_spheroid_effective_conductivity.jl` | Kushch 2015 setting | **published tutorial** — confocal geometry and API, Kapitza sweep, exact equivalent particle (`𝐤ᵉᑫ = 𝐁_Ω·𝐀_Ω⁻¹`), series-truncation convergence and quadrature vs. BigFloat |
| `35_spheroid_interfaces.jl` | local fields (spheroid) | **published tutorial** — what an interface does to the local fields: temperature map, streamlines (bilateral seeding), GIF over β, interactive 3D |
| `37_spheroid_hc_conductivity.jl` | Kushch 2015, HC interface | **published tutorial** — highly conducting (surface-conductive) interfaces vs. aspect ratio |
| `33_laminate_basics.jl` | Backus 1962 | **published tutorial** — the exact periodic multilayer: closed form, the two bound saturations (in-plane Voigt, out-of-plane Reuss), transport, localization, arbitrary normal |
| `34_laminate_interfaces.jl` | Hervé-Luanco 2014, planar case | **published tutorial** — primal (spring/Kapitza) vs dual (membrane/surface-conductive) interfaces act on complementary halves of the answer; the `1/L` size effect; displacement jumps |
| `36_laminate_multiscale.jl` | — | **published tutorial** — a three-scale model written explicitly *and* declaratively (`Homogenized`), shown to agree exactly; `NestedParameter` sensitivities across every scale; one microstructure, two physics |
| `38_laminate_symbolic.jl` | Backus 1962 | **not published** (SymPy-heavy): derives the closed forms *from the code*, symbolically, and prints the effective matrix in five averages — the check that the pseudo-inverse is a cofactor inverse and not an SVD. Hand-written companion: `docs/src/tutorials/symbolic_laminate.md` |
| `39_laminate_alv.jl` | — | **published tutorial** — the multilayer in ageing viscoelasticity: elastic limit, a creeping binder between elastic reinforcements, the saturations surviving the Volterra transposition |

### 40–49 Strength & multiscale
| Script | reference / topic | Notes |
|---|---|---|
| `40_porous_strength_criterion.jl` | — | porous strength criterion |
| `41_multiscale_strength.jl` | Pichler et al. (CCR 2011) | full 3-scale + strength (ω=1e4). Cross-checked in `bench_echoes/benchmark_strength.jl` (moduli 1 %, fc 2 %) |
| `42_cementpaste_iso.jl` | Pichler et al. (CCR 2011), ISO | elasticity-only ISO variant (**ω=100**, αmax·(1−1e-3)) |
| `43_secant_elastoplasticity.jl` | Suquet (1997) / Ponte Castañeda (1991); Gurson (1977) | **published tutorial** — modified secant method on a porous plastic solid: n-shell composite sphere + SC + `ForwardDiff` second moments; ported from echoes `echoes_tests/elastoplasticity_porous.py` |
| `44_stoichiometric_hydration_micromechanics.jl` | Lavergne et al. (CCR 2018) | **chemistry-driven** four-scale SC/SC/MT/MT paste: volume fractions computed by ChemistryLab (Parrot-Killoh + Waller + molar volumes) instead of a Powers correlation. Activates `docs/`, not the repo root. Backs `applications/hydrating_blended_paste.md` |
| `46_lamellar_porous_swelling.jl` | Dormieux, Lemarchand & Sanahuja, C. R. Mécanique **334** (2006) 304-310, [doi:10.1016/j.crme.2006.03.008](https://doi.org/10.1016/j.crme.2006.03.008) | **SymPy, not in the gallery.** Two-scale clay/CSH model derived symbolically: the particle is a `Laminate` whose interfoliar layer has a normal stiffness alone (singular ⇒ regularized then `tlimit`), the platelets are incompressible (`kₛ → ∞`), the assembly of randomly oriented particles + macropores is closed by the self-consistent scheme with `isotropify` as the exact orientation average. Recovers (5)-(18) of the article, including `μᵃᶜ`, `νᵃᶜ`, `g(φ)` and the `φ = 1/4` percolation threshold. Numeric cross-check against `SelfConsistent()`, figure in `figures/`. Backs `applications/lamellar_clay.md`, which derives the same model without the checking layer |

### 50–59 Viscoelasticity & ALV
| Script | reference / topic | Notes |
|---|---|---|
| `50_visco_law_basics.jl` | `visco_law` | Maxwell/Kelvin kernels |
| `51_frequency_sweep_viscoelastic.jl` | complex moduli | frequency sweep, built on `iso_rheology` + `zener_maxwell` |
| `52_rabotnov_mittag_leffler.jl` | Rabotnov / Mittag-Leffler | Rabotnov closed form; PyCall-free since the `Rabotnov` model landed |
| `53_ageing_creep_solid.jl` | solidifying creep | ALV creep |
| `54_ageing_creep_ellipsoid2.jl` | ellipsoid-2 creep | ALV creep |
| `55_ageing_creep_dirichlet_chains.jl` | Granger creep | ageing creep (Granger–Bažant 1995 law) |
| `56_ageing_creep_order2.jl` | order-2 creep | order-2 ALV |
| `57_ageing_creep_cracks.jl` | crack creep | ALV crack creep |
| `58_alv_kernel_types.jl` | — | structured ALV kernel types |
| `59_alv_sensitivities.jl` | — | **published tutorial** — `ForwardDiff` through the ALV pipeline: `set_param` lens vs closure capture, joint gradient, relaxation-time sensitivity, all validated against central finite differences |

### 60+ ALV cracks, cross-validations, the Laplace-Carson route / symmetrization
| Script | reference / topic | Notes |
|---|---|---|
| `60_alv_cracks_interface.jl` | crack + interface creep | finite interface stiffness |
| `61_freq_vs_time.jl` | Sanahuja (2013) trapezoidal Volterra | **published tutorial** — the **three** routes on one composite: complex moduli, `homogenize_alv`, and `homogenize_lc`. O(Δt²) agreement forward, and the reverse direction now closed by numerical inversion, with the trapezoidal error, the inversion error and the Gaver-Stehfest budget separated column by column. Ported from echoes `creep/comparison_freq_time.py` |
| `62_alv_schemes.jl` | Barthélémy et al. (2019), IJES 144, 103104 | **published tutorial** — Dilute / Mori-Tanaka / Maxwell / PCW on one ageing creep test; the aspect-ratio sweep at fixed fraction; the collapse MT = Maxwell = PCW when the distribution shape equals the inclusion shape, and the PCW admissibility limit when it does not |
| `63_kelvin_maxwell.jl` | echoes `Abderrahim/Kelvin2Maxwell.py` | **published tutorial** — the exact generalized-Kelvin ⇄ generalized-Maxwell conversion: the interlacing that isolates every root before any arithmetic, the round trip staying at `1e-15` out to twenty branches (where the symbolic route fails), and two independent closed forms — the Zener relations and the Burgers `cosh`/`sinh` relaxation — as oracles |
| `64_laplace_inversion.jl` | Abate & Valkó; de Hoog et al.; Trefethen et al. | **published tutorial** — the four inversion algorithms measured on four exact pairs; branch cuts are fine and oscillation is what separates them; the Gaver-Stehfest optimum and why more terms is worse; `ForwardDiff` straight through |
| `65_rheological_models.jl` | Di Benedetto & Olard (2S2P1D); Huet-Sayegh | **published tutorial** — the model catalog in one place: classical chains, the fractional family, the bituminous models with master curves, Cole-Cole and Black diagrams, and the exact 2S2P1D pair in both domains |
| `66_symbolic_viscoelasticity.jl` | SymPy / Symbolics | symbolic model parameters and symbolic Laplace-Carson inversion; deliberately **not** published (SymPy costs ~45 s of build time), the hand-written `tutorials/symbolic_viscoelasticity.md` covers it |
| `70_symmetrization_showcase.jl` | `symmetrize` / `.paramsym` | **exact rotation average vs best-fit projection** on a non-major-symmetric concentration tensor |

### 80–89 Custom inclusions, finite elements & neural surrogates

| Script | reference / topic | Notes |
|---|---|---|
| `80_custom_inclusion_contract.jl` | echoes `user_inclusion` | the three entry gates (Hill / localization / contribution) driven through every scheme — identical to the last digit; plus the density seam and free orientation averaging |
| `81_fe_crack_eshelby.jl` | Adessina et al. (2017), IJES 119, 1-15 | elliptical crack by finite elements (`Ferrite` + `Gmsh`): mesh, first-order corrected boundary condition, `‖B_u‖ ∝ (a/R)³`, convergence and Richardson extrapolation vs the closed-form COD |
| `82_fe_crack_schemes.jl` | — | the finite-element crack as a drop-in `EllipticCrack` in Dilute / MT / SC / Differential, with `IsoSymmetrize` and the memoization count |
| `83_fe_excentered_sphere.jl` | Adessina et al. (2017), IJES 119, 1-15 | the sphere with an off-center core by **axisymmetric Fourier** elements: the concentric limit against Hervé-Zaoui, what the boundary correction buys in `R/a`, the eccentricity sweep, the schemes, and transport |
| `84_neural_inclusion_ellipsoid.jl` | — | **published tutorial** (`neural_inclusion`): a trained network as an inclusion, both phases. §1 how one is trained — Mermaid schematics of the network and of the fitting loop, the recipe (shown, not run) and the committed learning curve; §2 onwards how one is used — what stays exact whatever the fit (zero contrast, homogeneity, symmetry class, frame), accuracy against the closed form, the `AffineHill` factorization that makes ν₀ exact, every scheme, and `ForwardDiff` on the aspect ratio. Loads the committed models: no ML dependency, nothing trained at build time |

| `85_neural_excentered_sphere.jl` | Adessina et al. (2017), IJES 119, 1-15 | **published tutorial** (`neural_excentered_sphere`): a surrogate trained on the *finite-element* localization tensors of `FEExcenteredSphere`. Gate B with the 6-component TI pair, why the features are contrast ratios and not the gate-A homogeneity, accuracy and speed-up against the finite elements, and `ForwardDiff` on the eccentricity — which the finite-element type refuses. Trained by `scripts/nn/train_excentered.jl` (~1500 solves), compared by `scripts/nn/make_excentered_figures.jl`; the page loads the committed models |
| `86_crack_distributions.jl` | echoes `crack` + `symmetrize=[ISO]` | **published tutorial** (`crack_distributions`): the same penny cracks at ε = 0.6 under two orientation rules. MT and the symmetric SC match Echoes 1.0 to ~1e-7 in both cases; `AsymmetricSelfConsistent` is the compliance-form fixed point instead; both percolate but at different densities (9/16 exactly for the compliance form, about 1.158 for the stiffness one, both independent of the matrix Poisson ratio). Also documents the local-versus-canonical component trap of a `TensRotated` result, and reads the five Walpole coefficients with `TensND.ti_params_from_KM` |
| `87_ageing_ages_aspect.jl` | — | **published tutorial** (`ageing_ages_aspect`): ageing creep with `ViscoLaw(J, :creep)` on both phases — three loading ages t' x three inclusion aspect ratios under Mori-Tanaka. Shows that the morphological effect is a near-constant offset independent of t', and that flattening the inclusions *reduces* the effective creep. Eleven ALV runs on a 60-point grid |

Scripts 84 and 85 need nothing beyond the package: it loads the surrogates committed
under `src/NeuralInclusions/models/`. *Training* them is
`scripts/nn/train_models.jl`, which activates its own `scripts/nn/` environment
carrying `Lux`, `Optimisers` and `Zygote` (weak dependencies).

Scripts 81 to 83 need `Ferrite`, `FerriteGmsh` and `Gmsh` (weak dependencies of
`MeanFieldHomogenization`). 81 and 82 take a minute or so — they mesh a ball and factorize
a ~10⁵-dof system per case; 83 is two-dimensional and runs in seconds.

All three also run on the second backend if `Gridap` and `GridapGmsh` are added
to `scripts/fe/`: pass `backend = GridapBackend()` to the constructor. The two
agree to round-off.

The two `scripts/fe/make_*_figures.jl` are maintenance scripts, run by hand,
that regenerate the committed PNGs and result tables of the documentation pages
`manual/fe_inclusions.md` and `applications/recycled_aggregate.md`. Nothing
finite-element runs at documentation-build time.

### 90–99 Interacting particle assemblies (EIM & cluster model)

The two **N-body** schemes: unlike every other scheme of the package they resolve
the interaction between individual inclusions, so they act on a
`ParticleAssembly` (which carries positions) rather than on an `RVE`. Both are
built on the same two-inclusion interaction tensor — Brisard et al. (2014) §3.1
note that their order-zero influence pseudotensors *are* the interaction tensors
of Molinari & El Mouden (1996). The package follows the sign convention of
Brisard, Bertin & Legoll (2023), Eq. (9), for which 𝕋^{aa} = +ℙ; Molinari's is
the opposite, so anything transcribed from his papers is flipped on the way in.

| Script | reference / topic | Notes |
|---|---|---|
| `90_pair_interaction_tensor.jl` | Molinari & El Mouden (1996) App. A; Berveiller et al. (1987) | the shared kernel 𝕋^{ab}: transverse isotropy about the line of centers, the vanishing isotropic part, 𝕋^{aa} = +ℙ (the Brisard sign convention), R⁻³ decay and the ρ² correction, the three back-ends against each other, and the spherical-cutoff lattice sum |
| `91_cluster_cubic_arrays.jl` | Molinari & El Mouden (1996), Figs. 3, 5, 6, 16 | **published tutorial** (`cluster_model`): convergence in cluster radius and the exact Mori-Tanaka degeneracy at R_c = 0, comparison with SC / differential / MT, the bulk modulus that stays exactly Mori-Tanaka, cubic anisotropy, SC vs BCC vs FCC porous arrays, and the EIM ≡ cluster identity |
| `92_multiscale_assemblies.jl` | — | **published tutorial** (`multiscale_assemblies`): an assembly as the inner and as the outer cell of the declarative multiscale seam, the cubic anisotropy a cluster estimate produces (and the bulk modulus that stays exactly Mori-Tanaka), a three-level chain that needs the anisotropic Green operator, a nested sensitivity, and the cost of an anisotropic reference |
| `93_eim_disk_assembly_2d.jl` | Brisard, Dormieux & Sab (2014), Table 1 | **published tutorial** (`eim_assembly`): 160 circular pores at φ = 0.4 in a circular SVE of radius 20a, plane strain, ν₀ = 0.3 — reproduces the p = 0 row (0.310 μ₀) by Monte-Carlo, with the HS bound and the FEM reference for scale |
| `96_nano_spheroids.jl` (published under *Inclusions*, not here: no N-body content) | Dormieux, Lemarchand & Brisard (2016), Eqs. (72)–(78) | **published tutorial** (`nano_spheroids`): the interface stiffness ℂ^int across aspect ratios, its three limiting cases (sphere, platelet, nanofiber) reproduced exactly, and the size effect it produces through an ordinary Mori-Tanaka estimate |

Numbers 94, 95 and 97–99 are free. What is deliberately **not** covered:
Table 2 of Brisard et al. (polydisperse spheres at φ = 0.45) needs a polydisperse
close-packing generator the package does not have, and the slender-fiber
specialization of Martin et al. (2023) needs the axial polynomial enrichment,
which is not implemented — see `docs/src/developer/roadmap.md`.

## Conventions worth knowing

- **Exact vs best-fit symmetrization.** Inside scheme kernels the orientation
  average is EXACT (`transverse_isotropify` → `TensTI{4,T,8}`, non-major-
  symmetric content preserved). `best_fit_ti` (→ `TensTI{4,T,5}`) is the
  echoes `.paramsym(sym=TI)` reporting projection — never used in kernels.
  `70_symmetrization_showcase.jl` demonstrates the difference.
- **Water/air TINY = 1e-3.** The Pichler scripts regularize the exactly-zero
  echoes water/air stiffness with a small positive `TINY`, which selects the
  physical (percolating) Self-Consistent branch. Expect a matching small
  offset from echoes near α→0. The ISO variant (`42_cementpaste_iso.jl`) uses
  the exact echoes convention where it is robust.
- **Needle aspect ratio.** The full CCR2011 model uses ω = 1e4; the companion
  iso variant uses ω = 100 (both faithful to their echoes originals).

## Not yet ported
Biaxial strength envelope (Pichler et al., CCR 2013) and the multi-model
`E(w/c)` comparison — future ports.

## Literate.jl convention (pilot, 2026-07-24)

A script converted to this contract stays runnable exactly as before
(`julia scripts/NN_*.jl`) **and** becomes a source for
[Literate.jl](https://github.com/fredrikekre/Literate.jl), which generates a
Documenter markdown page, a Jupyter notebook, and a cleaned standalone
script from the same file (`julia --project=docs docs/literate.jl`).

**Publication policy** — a script is only *published as a tutorial page*
(added to `PUBLISHED_SCRIPTS` in `docs/literate.jl` and to the `pages`
tree of `docs/make.jl`) if no other tutorial or application already
covers its topic. Scripts that duplicate one (the majority — see
`Assets/plans/MFH_LITERATE_SCRIPTS.md` for the full classification) keep
the plain banner style and are never regenerated into a competing page.

There is no separate "Gallery" section any more: a page generated from a
script and one written by hand are both tutorials, and the reader has no
reason to care which is which. `PUBLISHED_SCRIPTS` maps each script to
its **page name**, so scripts keep their numeric prefixes (a running
order) while pages carry thematic names — inserting a tutorial never
forces a renumbering. That mapping is what Literate's `name` option is
for.

Converting a script to the contract, whether or not it ends up promoted:

- **Title & prose.** Replace the `# ===...===` banner with a Literate
  header: `# # Title`, then prose paragraphs as plain `# ` lines, math as
  ```` # ```math ... ``` ````, section dividers as `# ## §N Title`.
- **`Pkg.activate`.** Suffix both the `import Pkg` and `Pkg.activate(...)`
  lines with `#jl` — kept in the standalone script and the generated
  "cleaned script", stripped from the generated markdown/notebook (which
  run inside the `docs` environment, where `MeanFieldHomogenization` is already
  available via `[sources] path=".."`).
- **Figures.** End the plotting code with the plot object as a bare,
  unmarked final expression (captured inline by `@example`/notebook
  execution). Suffix `figdir`/`mkdir`/`savefig`/`display`/the "Saved:"
  `@printf` with `#jl` — the standalone run still writes the PNG to
  `scripts/figures/`, the doc page shows the figure inline instead.
- **Determinism.** Any script using `Random` needs `Random.seed!(<const>)`
  near the top, so the generated doc page (and notebook) render identical
  numbers on every rebuild.
- **Don't combine `#md #nb` on one line** — Literate's marker matching only
  recognizes a single trailing tag; a line with two strips it from *all*
  three outputs. `gr()` needs no marker at all — leave it plain, exactly as
  the hand-written tutorials already do.

See `docs/literate.jl` for the generator entry point and
`Assets/plans/MFH_LITERATE_SCRIPTS.md` for the gap-filler vs. duplicate
classification of all 41 scripts.

## 88 — MeanFieldHomogenization *inside* a finite-element code

`88_fe_thick_cylinder.jl` is the other direction of the 80-89 block: not an
inclusion whose response comes from a finite-element solve, but a whole
microstructure acting as the constitutive law at every Gauss point of a
structural computation (`src/Constitutive/`). It runs a thick-walled cylinder
twice — a linear composite checked against the Lamé closed form, then a
microcracked solid whose cracks close — and needs only `Ferrite` (no gmsh).

Figures for the documentation come from
`scripts/fe/make_thick_cylinder_figures.jl`, run by hand; the pages are static
so no documentation build re-runs a finite-element solve.
