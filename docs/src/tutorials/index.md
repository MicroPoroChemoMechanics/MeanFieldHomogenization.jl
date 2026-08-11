# [Tutorials](@id tut-index)

Mean-field homogenization replaces a heterogeneous microstructure — a matrix
carrying inclusions, pores, or cracks — by an equivalent homogeneous medium with
the same overall response. `MeanFieldHomogenization` computes that response from phase
properties, geometries, volume fractions, and a **scheme** encoding an
assumption about how the phases interact.

Simplest first within each group. Pages under `generated/` are produced from the
runnable demos in `scripts/` by
[Literate.jl](https://github.com/fredrikekre/Literate.jl).

## Fundamentals

The **porous material** gets two pages: the simplest non-trivial microstructure,
and the one where the choice of scheme matters most.

| Page | What it shows |
| :--- | :--- |
| [A first homogenization](first_estimate.md) | build an RVE; dilute vs Mori–Tanaka |
| [Bounds and classical schemes](bounds_and_schemes.md) | Voigt/Reuss bounds, self-consistent, where each scheme sits |
| [Porous materials and the self-consistent trap](porous_materials.md) | why soft pores break the naive SC iteration |
| [Porous benchmark: all schemes](porous_benchmark.md) | every scheme on the canonical porosity sweep, spheres and oblate pores |
| [The differential scheme and path dependence](differential_paths.md) | incremental homogenization; why mixing order matters |
| [Comparing loading-path trajectories](differential_loading_paths.md) | the same target fractions, four trajectories, watched τ by τ |

## Inclusions, geometries and orientation

| Page | What it shows |
| :--- | :--- |
| [Hill polarization tensors in practice](generated/hill_tensors.md) | `hill_tensor` on four geometries; residues vs cubature on an anisotropic matrix; the Eshelby tensor against its closed form; `P` → a dilute estimate |
| [Cracks and crack density](cracks.md) | volume fraction → crack density; the COD tensor |
| [Crack distributions: isotropic or parallel](generated/crack_distributions.md) | the same density, two orientation rules; the two self-consistent forms and their two percolation thresholds (9/16 exactly for the compliance form, ≈ 1.158 for the stiffness one); the local-versus-global frame of a `TensRotated` result |
| [Layered spheres](generated/layered_sphere.md) | Hervé–Zaoui `n`-layer localization and layer averages |
| [Layered spheroids: geometry and effective conductivity](generated/layered_spheroid_effective.md) | the confocal `n`-layer spheroid, the equivalent particle, harmonic-series accuracy |
| [Imperfect interfaces: what they do to the local fields](generated/layered_spheroid_interfaces.md) | pointwise temperature and flux, streamlines, conductance sweep, 3-D view |
| [Highly conducting interfaces](generated/layered_spheroid_hc.md) | equivalent conductivity of an HC-coated particle vs aspect ratio |
| [Nanocomposites: the equivalent particle](generated/nano_spheroids.md) | Dormieux, Lemarchand & Brisard (2016): a Gurtin-Murdoch interface condensed into a particle stiffness, its three limiting shapes reproduced exactly, and the size effect it produces through an ordinary Mori-Tanaka estimate — no new scheme needed |
| [Symmetrization](generated/symmetrization.md) | exact rotation-group average vs best-fit projection |
| [The custom-inclusion contract](generated/custom_inclusion_contract.md) | plugging an arbitrary morphology into every scheme — the three entry gates, the density seam, free orientation averaging |
| [An inclusion whose response is a neural network](generated/neural_inclusion.md) | both phases — how a surrogate is **trained** (schematics of the network and of the fitting loop, the recorded learning curve) and how a trained one is **used**: what stays exact whatever the fit, accuracy against the closed form, every scheme, and the derivative with respect to the *morphology* |
| [Replacing a finite-element solve by a surrogate](generated/neural_excentered_sphere.md) | the case the machinery exists for: the eccentric-core sphere, whose localization tensors have no closed form. Gate B with the 6-component transversely isotropic pair, the contrast ratios that replace gate A's homogeneity, the accuracy against the finite elements, the speed-up, and a derivative with respect to the **eccentricity** |

Composite inclusions carry no Hill tensor at all: they enter the schemes through
their volume-averaged concentration tensors instead.

## Interacting particle assemblies

The two **N-body** schemes. Every other scheme of the package sees a single
inclusion in a reference medium and averages the interaction; these two resolve
it inclusion by inclusion, so they need a `ParticleAssembly` — a cell that
carries positions — rather than an `RVE`. Both are built on the same
two-inclusion interaction tensor.

| Tutorial | What it shows |
|---|---|
| [The cluster model on cubic arrays](generated/cluster_model.md) | Molinari & El Mouden (1996): convergence in cluster radius, the exact degeneracy onto Mori-Tanaka when the cluster is empty, comparison with the one-site schemes, the bulk modulus that stays exactly Mori-Tanaka whatever the arrangement, and SC vs BCC vs FCC porous arrays |
| [The equivalent inclusion method](generated/eim_assembly.md) | Brisard, Dormieux & Sab (2014), Table 1: 160 circular pores in a circular SVE, plane strain — the `p = 0` bound reproduced by Monte-Carlo, against the Hashin-Shtrikman bound it improves on and the finite-element value it bounds |
| [Chaining scales through an N-body scheme](generated/multiscale_assemblies.md) | the declarative multiscale seam with an assembly on either side, and why chaining two N-body estimates needs the *anisotropic* Green operator: a cluster estimate on a cubic array is cubic, not isotropic. Three scales end to end, plus a sensitivity across them |

Theory: [interaction tensors](@ref th-interaction), [the cluster
model](@ref th-cluster), [the equivalent inclusion method](@ref th-eim); the
API is on the [particle-assembly manual page](@ref man-assemblies).

## Beyond elasticity

| Page | What it shows |
| :--- | :--- |
| [Viscoelastic composites](viscoelasticity.md) | complex moduli in the frequency domain; a first taste of ageing creep |
| [Frequency or time?](generated/freq_vs_time.md) | the complex-modulus and time-domain ALV routes, cross-checked on the same non-ageing composite |
| [Ageing viscoelastic schemes side by side](generated/alv_schemes.md) | Dilute / Mori-Tanaka / Maxwell / PCW on one creep test; the aspect ratio; where the distribution shape decides the answer |
| [Ageing creep: loading age against inclusion shape](generated/ageing_ages_aspect.md) | ageing and morphology on the same output: three loading ages × three aspect ratios, and why the shape effect is an offset independent of the age |
| [Derivatives through the ageing-viscoelastic pipeline](generated/alv_sensitivities.md) | `ForwardDiff` through the Volterra assembly: the `set_param` lens for RVE parameters, closure capture for moduli and relaxation times |

## Differentiation and solvers

| Page | What it shows |
| :--- | :--- |
| [Derivatives and sensitivities](sensitivities.md) | differentiate any result with `ForwardDiff`, no finite differences |
| [From derivatives to a strength criterion](strength_criteria.md) | those derivatives as a macroscopic strength criterion |
| [Nonlinear solvers for the self-consistent fixed point](nonlinear_solvers.md) | `NonlinearSolve.jl` instead of Picard, and sensitivities that agree either way |
| [Nonlinear homogenization by the secant method](generated/secant_elastoplasticity.md) | elastic–perfectly plastic porous solid, closed by second moments |

## Interoperability and tools

| Page | What it shows |
| :--- | :--- |
| [Validating a finite-element crack](fe_crack.md) | what the corrected boundary condition buys, and convergence to the closed-form COD |
| [Transport properties](transport.md) | 2nd-order homogenization: diffusivity of a porous medium, anisotropy from oriented pores |
| [Symbolic spheres](symbolic_spheres.md) | the same tensor algebra on `SymPy` / `Symbolics` expressions: Eshelby/Hill tensors and the closed-form estimates |
