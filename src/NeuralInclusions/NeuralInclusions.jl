"""
    MeanFieldHomogenization.NeuralInclusions

Inclusions whose response is produced by a **trained neural network** instead of
a closed form or a finite-element solve — a fourth route into the
[`CustomInclusions`](@ref MeanFieldHomogenization.CustomInclusions) contract, alongside the
analytic families, the layered patterns and
[`FiniteElements`](@ref MeanFieldHomogenization.FiniteElements).

| Type | Entry gate | For |
|---|---|---|
| [`NeuralHillInclusion`](@ref) | A — the Hill tensor | a morphology with a Hill tensor: contrast dependence and the `ℂ₁ = ℂ₀ ⟹ 𝔸 = 𝕀` limit stay exact |
| [`NeuralLocalizationInclusion`](@ref) | B — both localization tensors | an internally heterogeneous morphology, which has no Hill tensor |

# Why bother, when the analytic Hill tensor is exact

For the ellipsoid the surrogate is *not* faster than the closed form, and it is
less accurate. What it buys is two things the expensive routes cannot give:

- **differentiability.** A surrogate is a smooth function of its inputs, so
  `derivative(rve, scheme, geometry(:phase, :field))` reaches a *morphology*
  parameter. The finite-element inclusions refuse that request outright
  (`FiniteElements.jl`): their solve runs in `Float64` and memoizes on the
  reference medium, so the derivative would come out as a silent zero.
- **cost, once the teacher is expensive.** One `FEExcenteredSphere` evaluation
  is three assemblies and eight solves, and an iterative scheme changes the
  reference medium at every iteration, defeating the cache. A surrogate trained
  on that solve answers in microseconds.

The ellipsoid is therefore the *validation* case, not the application: it is the
one morphology where the labels are exact, so every part of the pipeline can be
checked against a closed form before being pointed at something unknown.

# The pieces

- `mlp.jl` — a dependency-free, type-generic multilayer perceptron. Evaluation
  needs `LinearAlgebra` and nothing else, which is what lets a committed model
  be loaded anywhere and be traversed by `ForwardDiff`.
- `specs.jl` — the physics: which tensor class is predicted, and how the exact
  invariances (symmetry class, major symmetry, homogeneity in the reference
  moduli, frame indifference) are *enforced* rather than fitted.
- `surrogate.jl` — [`NeuralSurrogate`](@ref): weights, standardization, output
  specification, validity box and [`Provenance`](@ref).
- `dataset.jl` — Halton sampling of a [`SampleBox`](@ref) and labeling by a
  **teacher**, the one seam that changes between morphologies.
- `training.jl` — [`TrainingOptions`](@ref) and the fallback
  [`train_surrogate`](@ref); the optimizer lives in `MeanFieldHomogenizationLuxExt`.
- `io.jl` — [`save_surrogate`](@ref) / [`load_surrogate`](@ref), JSON.

Fitting needs `Lux`, `Zygote` and `Optimisers` (weak dependencies, as for the
finite-element backends); evaluating needs none of them.

See `docs/src/manual/neural_inclusions.md` and
`scripts/84_neural_inclusion_ellipsoid.jl`.
"""
module NeuralInclusions

using TensND

import LinearAlgebra
import Random
import ForwardDiff
import JSON3

using Printf: @printf, @sprintf

import ..Core
# `hill_tensor` and the two identity helpers live at the package's top level,
# in `localization.jl`, which is included before this module. `AnchoredHill`
# needs them to evaluate a closed-form baseline, and importing them keeps the
# convention single-sourced rather than re-deriving the identity here.
import ..hill_tensor
import .._identity_4sym
import .._identity_2
import ..FiniteElements
import ..Elasticity
import ..Schemes

# The network and its serialization
export MLP, NNDense, softplus
export NeuralSurrogate, Provenance, worst_error
export save_surrogate, load_surrogate, model_path, shipped_models

# The physics of the output
export HillISO, HillTI, HillOrtho, HillISO2, HillTI2, StrainLocCubic
export GradLocISO2, GradLocTI2
export StrainLocTI, StressLocTI
export DimensionlessHill, AnchoredHill, AffineHill

# The learning system
export SampleBox, Dataset, generate_dataset, fit_scaling
export TrainingOptions, train_surrogate, assemble_surrogate
export validate_surrogate, report_surrogate, component_labels

# The inclusions
export NeuralShape, NeuralHillInclusion, NeuralLocalizationInclusion

include("mlp.jl")
include("specs.jl")
include("surrogate.jl")
include("dataset.jl")
include("training.jl")
include("io.jl")
include("neural_inclusion.jl")

end # module
