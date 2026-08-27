# [Neural-surrogate inclusions](@id man-neural-inclusions)

A trained network is a fourth way into the
[custom-inclusion contract](@ref man-custom-inclusions), alongside the analytic
families, the layered patterns and the
[finite-element inclusions](@ref man-fe-inclusions).

| Type | Entry gate | For |
|:--|:--|:--|
| [`NeuralHillInclusion`](@ref MeanFieldHomogenization.NeuralHillInclusion) | A — the Hill tensor | a morphology that has a Hill tensor |
| [`NeuralLocalizationInclusion`](@ref MeanFieldHomogenization.NeuralLocalizationInclusion) | B — both localization tensors | an internally heterogeneous morphology, which has none |

**Evaluating** a surrogate needs nothing beyond the package. **Training** one
needs three weak dependencies:

```julia
using MeanFieldHomogenization
import Lux, Optimisers, Zygote        # activates MeanFieldHomogenizationLuxExt
```

## What it buys

For an ellipsoid a surrogate is neither faster nor more accurate than the closed
form. It buys two things the closed form does not need and the expensive routes
cannot give:

- **A derivative with respect to the morphology.** A surrogate is a smooth
  function of its inputs, so `derivative(rve, scheme, geometry(:phase, :field))`
  reaches an aspect ratio. The finite-element inclusions refuse that request
  outright — their solve runs in `Float64` and memoizes on the reference medium,
  so the derivative would come back as a silent zero.
- **Cost, once the teacher is expensive.** One
  [`FEExcenteredSphere`](@ref MeanFieldHomogenization.FEExcenteredSphere) evaluation is
  three assemblies and eight solves, and an iterative scheme changes the
  reference medium at every iteration, defeating the cache. A surrogate trained
  on that solve answers in microseconds.

The ellipsoid is therefore the *validation* case: the one morphology whose labels
are exact, so the pipeline can be held to a closed form before being pointed at
something unknown. `scripts/84_neural_inclusion_ellipsoid.jl` is that check,
[published as a tutorial](@ref tut-index).

## Using a shipped model

Three lines:

```julia
using MeanFieldHomogenization

s = load_surrogate(model_path("spheroid_hill_iso_elastic"))
incl = NeuralHillInclusion((1.0, 1.0, 0.4); elastic = s)     # an oblate spheroid, ω = 0.4

rve = RVE(:m)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => iso_stiffness(30.0, 10.0)))
add_phase!(rve, :i, incl, Dict(:C => iso_stiffness(60.0, 20.0)); fraction = 0.2)
homogenize(rve, MoriTanaka(), :C)
```

Gate A means every scheme works, in elasticity and in transport, and so do
orientation averaging and the sensitivity API. `shipped_models()` lists what is
available.

### The four committed models

Trained by `scripts/nn/train_models.jl` against the **analytic** Hill tensor, so
the labels are exact and the error below is the fit's alone. All were fitted on
a Halton sample with a held-out set drawn from the same sequence; ``\omega`` is
the *distinct over equal* semi-axis ratio, so ``\omega > 1`` is prolate and
``\omega < 1`` oblate.

| Model | Predicts | Features | Domain | Network | Samples | Worst error |
|:--|:--|:--|:--|:--|:--|:--|
| `spheroid_hill_iso_elastic` | ``2\mu_0\mathbb P``, `TensTI{4,·,5}` | `log_aspect`, `nu0` | ``\omega \in [1/20, 20]``, ``\nu_0 \in [0, 0.49]`` | 2→48→48→5 | 6000 / 1500 | `3.1e-3` |
| `spheroid_hill_iso_conduction` | ``k_0\boldsymbol P``, `TensTI{2,·,2}` | `log_aspect` | ``\omega \in [1/20, 20]`` | 1→32→32→2 | 3000 / 800 | `2.1e-4` |
| `triaxial_hill_iso_elastic` | ``2\mu_0\mathbb P``, `TensOrtho` | `log_r2`, `log_r32`, `nu0` | ``a_2/a_1,\, a_3/a_2 \in [1/20, 1/1.05]``, ``\nu_0 \in [0, 0.49]`` | 3→64→64→9 | 12000 / 3000 | `6.7e-3` |
| `spheroid_hill_iso_affine` | ``\mathbb U^{\boldsymbol A}`` and ``\mathbb V^{\boldsymbol A}``, `TensTI{4,·,5}` | `log_aspect` | ``\omega \in [1/20, 20]``, **any** ``\nu_0`` | 1→48→48→10 | 6000 / 1500 | `2.6e-4` |

"Worst error" is `worst_error(s.provenance)`: the largest error over the held-out
set, in the ∞-norm of the component vector relative to its own magnitude. It is
the number a tolerance should be derived from — the test suite does exactly that
rather than hard-coding a literal, so a retraining cannot silently loosen a
threshold. Per-component diagnostics are in
`src/NeuralInclusions/models/training_report.md`.

Note the last row: the affine factorization is **twelve times more accurate**
than the generic one, on a network of the same size, because it does not spend
capacity fitting a dependence that is exactly known. See
[below](@ref man-neural-affine).

## Training your own

Four decisions, then one call. The
[tutorial](@ref tut-index) walks the same ground with a schematic of the network
and of the fitting loop, and shows the recorded learning curve.

**1. Which tensor, and therefore which gate.** ``\mathbb P`` (gate A) whenever
the morphology has one: the contrast dependence and the ``\mathbb C_1 = \mathbb
C_0 \Rightarrow \mathbb A = \mathbb I`` limit then stay exact and the fit error
is confined to a single tensor. The localization pair (gate B) only for a
heterogeneous morphology, which has no ``\mathbb P``.

**2. Which symmetry class**, from the shape: `HillISO` for a sphere (2
components), `HillTI` for a spheroid (5), `HillOrtho` for a triaxial ellipsoid
(9); `HillISO2` and `HillTI2` are their order-2 transport counterparts. The class
is checked against the geometry at construction, because the analytic teacher
returns a *different* tensor class for each and the components would otherwise
mean something else.

**3. Which features, and over what box.** Use logarithms of shape ratios — an
aspect ratio's interesting range spans decades, and only the logarithm makes ``\omega``
and ``1/\omega`` symmetric. Add `:nu0` for a `DimensionlessHill` order-4 surrogate;
leave it out for `AffineHill` and for transport, where the material dependence is
exact. The box becomes the model's validity limits, so choose it as the range you
actually intend to use.

**4. The two teacher callbacks.**

```julia
import Lux, Optimisers, Zygote
const NI = MeanFieldHomogenization.NeuralInclusions

geometry(x) = Ellipsoid(1.0, 1.0, exp(x[1]))     # shape features → morphology
response(g, C₀) = hill_tensor(g, C₀)             # what is to be learned

spec = DimensionlessHill(HillTI())
box = SampleBox([:log_aspect, :nu0], [-log(20), 0.0], [log(20), 0.49])

train, val = generate_dataset(geometry, response, spec, box, 6000; nvalidation = 1500)
s = train_surrogate(spec, box, train, val; teacher_name = "analytic Hill tensor")

report_surrogate(s, val; labels = component_labels(spec))
save_surrogate("my_model.json", s)
```

`response` is the only morphology-dependent line: a finite-element solve, a table
lookup, anything that answers the same question goes there instead. Splitting the
teacher in two is what makes the frame convention safe — the frame in which
components are read is derived from `geometry` through the same code path the
inclusion uses at evaluation time, so the two cannot disagree.

`generate_dataset` refuses a mismatch loudly rather than training on corrupted
labels: it checks `:nu0` against the specification, and it verifies that the
teacher's tensor really lies in the declared symmetry class (the projection
residual must vanish). A wrong axis, a wrong class or a wrong frame is an error
at dataset time, not a quietly bad model.

### The knobs

[`TrainingOptions`](@ref MeanFieldHomogenization.TrainingOptions) — the defaults are sized
for a few thousand samples and a few thousand parameters, which is seconds of
wall time:

| Option | Default | Note |
|:--|:--|:--|
| `hidden` | `[32, 32]` | input and output widths follow from the box and the specification |
| `activation` | `:tanh` | must be smooth; `:softplus` is the alternative, and there is deliberately no ReLU |
| `epochs` / `batchsize` | `4000` / `128` | `batchsize = 0` means full batch |
| `learning_rate` / `decay` / `patience` | `1e-2` / `0.3` / `250` | a plateau decays the step; a plateau at the smallest step stops the run |
| `seed` | `20260730` | the weight initialization, so a retraining is reproducible |

Two things are read off the training set automatically and stored with the model:
the standardization of both ends, and a per-component `:log` transform wherever
the dynamic range calls for it — which is the case of the oblate Walpole
components, several of which grow like ``1/\omega`` as the particle flattens.

## What is exact, and what is fitted

Only what is genuinely unknown is learned. Three properties are enforced by
construction and hold to machine precision *however badly* the network is
trained:

- **Zero contrast.** Gate A supplies ``\mathbb P``, and the package evaluates
  ``\mathbb A_{\varepsilon\varepsilon} =
  [\mathbb I + \mathbb P:(\mathbb C_1-\mathbb C_0)]^{-1}`` exactly, so
  ``\mathbb C_1 = \mathbb C_0 \Rightarrow \mathbb A = \mathbb I``. The eight
  localization tensors also stay exactly consistent with one another — which a
  surrogate predicting ``\mathbb A`` could not guarantee.
  ``\mathbb A_{\varepsilon\varepsilon}`` has **no major symmetry**, so it needs
  the 6-component transversely isotropic form where ``\mathbb P`` needs 5; for
  an oblate spheroid its major-symmetry defect is around 10 %, and forcing it
  onto the 5-component form loses a few percent.
- **Homogeneity.** ``\mathbb P(\lambda\mathbb C_0) = \mathbb P(\mathbb C_0)/\lambda``.
  The network never sees an absolute modulus, only the shape and ``\nu_0``.
- **Symmetry class, major symmetry and frame.** The decoder emits a structured
  TensND type from the right number of components, in the inclusion's own frame;
  the orientation is never an input.

### [Removing the Poisson ratio from the inputs](@id man-neural-affine)

This is the **shape/moduli factorization** of
[Hill polarization tensors](@ref th-hill-tensors), regrouped. That page states

```math
\mathbb P\bigl(\boldsymbol A,\, 3\lambda_0\mathbb I + 2\mu_0\mathbb K\bigr)
= \frac{1}{\lambda_0 + 2\mu_0}\,\mathbb U^{\boldsymbol A}
+ \frac{1}{\mu_0}\bigl(\mathbb V^{\boldsymbol A} - \mathbb U^{\boldsymbol A}\bigr),
```

with the geometric auxiliaries [`tens_UA`](@ref MeanFieldHomogenization.tens_UA)
and [`tens_VA`](@ref MeanFieldHomogenization.tens_VA) depending on the **shape
alone**. Collecting the two terms on ``\mathbb U^{\boldsymbol A}`` and
``\mathbb V^{\boldsymbol A}`` gives

```math
\mathbb P = d\,\mathbb U^{\boldsymbol A} + \frac{1}{\mu_0}\,\mathbb V^{\boldsymbol A},
\qquad d = \frac{1}{\lambda_0 + 2\mu_0} - \frac{1}{\mu_0},
```

which is affine in the two material scalars ``(d, 1/\mu_0)``.
[`AffineHill`](@ref MeanFieldHomogenization.AffineHill) exploits it: the network predicts
those two tensors — twice the components, one fewer input — and the decoder
contracts them with the exact coefficients. The whole material dependence becomes
algebra, and the surrogate is valid at *any* ``\nu_0``, including values no label was
generated at.

Nothing shape-specific is reimplemented to get there. Both tensors live in the
same symmetry class as ``\mathbb P``, so their components are recovered from two teacher
evaluations at two Poisson ratios by solving the 2×2 system componentwise. In
transport the decomposition has a single term and the material dependence is
exact with no material input at all.

Prefer `AffineHill` whenever the reference medium is isotropic: same cost, one
input fewer, an order of magnitude more accurate. Prefer
[`DimensionlessHill`](@ref MeanFieldHomogenization.DimensionlessHill) when you will need to
transfer the recipe to an anisotropic matrix or to a localization pair, where no
affine structure exists.

## The domain guard

A network interpolates. Inside its box it is as good as its recorded error says;
outside it is unbounded and carries no diagnostic. The box therefore travels with
the weights and is checked on every evaluation:

```julia
incl = NeuralHillInclusion((1.0, 1.0, 1e-4); elastic = s, guard = :error)
hill_tensor(incl, C₀)   # ArgumentError: :log_aspect is outside the box
```

`guard` is `:warn` (default), `:error` or `:none`.

## Heterogeneous morphologies

[`NeuralLocalizationInclusion`](@ref MeanFieldHomogenization.NeuralLocalizationInclusion)
takes gate B, the only way in for a morphology with no Hill tensor. Since
[`is_homogeneous_inclusion`](@ref) is `false`, it costs **two** surrogates per
physics — the strain side and the stress side — because
``\mathbb A_{\sigma\varepsilon} = \mathbb C_1:\mathbb A_{\varepsilon\varepsilon}``
presupposes a uniform ``\mathbb C_1`` that a heterogeneous inclusion does not
have. Half a pair is refused at construction: the omission is otherwise silent,
leaving
`Dilute` and `MoriTanaka` right while the self-consistent schemes drift.

Supplying `fractions` and `properties` unlocks the `Voigt` and `Reuss` bounds,
which a heterogeneous inclusion cannot otherwise serve.

No trained model ships for this type yet; it is the seam for surrogates trained
on [`fe_axi_localization`](@ref MeanFieldHomogenization.fe_axi_localization).

## Limitations

- **Isotropic reference medium only.** Both the feature set (``\nu_0`` alone) and the
  exact homogeneity used to decode assume it; an anisotropic ``\mathbb C_0`` is refused
  rather than silently projected. This has a practical consequence: the
  *iterative* schemes — `SelfConsistent`, `AsymmetricSelfConsistent`,
  `DifferentialScheme` — re-evaluate the inclusion in their own running estimate,
  which for a single oriented spheroid is transversely isotropic. Add
  `symmetrize = IsoSymmetrize()` to the phase and the scheme hands the kernel a
  pre-projected isotropic reference at every iteration. One-shot schemes are
  unaffected. This is the same restriction the finite-element inclusions carry,
  for the same reason.
- **One class per surrogate.** `:ti` describes a spheroid, `:ortho` a genuinely
  triaxial ellipsoid, `:iso` a sphere; a mismatch is a constructor error.
- **The shipped models stop short of the crack and needle limits** (``\omega`` within a
  factor 20 of a sphere), where several Walpole components diverge.
- **Accuracy is the fit's, not the closed form's.** A few parts in `10⁴` to
  `10³` on the Hill tensor. Where an exact answer is available, use it — that is
  what the analytic families are for.

## Reproducing the models

```shell
julia scripts/nn/train_models.jl              # all four, plus training_report.md
julia scripts/nn/train_models.jl conduction   # one only
julia scripts/84_neural_inclusion_ellipsoid.jl
```

Nothing is trained at test or documentation-build time: both load the committed
JSON, which is what keeps the suite deterministic and the doc build free of Lux.

## See also

- [Custom inclusions](@ref man-custom-inclusions) — the contract and its three
  entry gates
- [Adding a new inclusion](@ref dev-adding-inclusion) — the leveled developer
  contract
- [Finite-element inclusions](@ref man-fe-inclusions) — the expensive teacher a
  surrogate is meant to replace
- [An inclusion whose response is a neural network](@ref tut-index) — the worked
  pilot: both phases, with the network schematic and the learning curve,
  validated against the closed form
