# =============================================================================
#  training.jl — the learning system's package-side half.
#
#  Everything that does not need an optimizer lives here and therefore always
#  works: the options, the box → surrogate assembly, and the reporting.  The
#  optimizer itself is a **weak dependency**, exactly like the finite-element
#  backends: `train_surrogate` below is the informative fallback, and
#  `MeanFieldHomogenizationLuxExt` supplies the real method.
#
#  The split is deliberate and it is what makes a trained surrogate cheap:
#  fitting needs Lux, Zygote and Optimisers, but *evaluating* needs nothing at
#  all, because the extension writes its result back into the dependency-free
#  `MLP` of `mlp.jl`.  A committed model is then a plain JSON file that any
#  installation can read.
# =============================================================================

"""
    TrainingOptions(; kw...)

Everything the optimizer needs, with defaults sized for the ellipsoid pilot
(a few thousand samples, a few thousand parameters, seconds of wall time).

| Option | Default | Meaning |
|---|---|---|
| `hidden` | `[32, 32]` | widths of the hidden layers; input and output widths follow from the feature list and the output specification |
| `activation` | `:tanh` | hidden activation — must be smooth, see `mlp.jl` |
| `epochs` | `4000` | maximum passes over the training set |
| `batchsize` | `128` | mini-batch size; `0` means full batch |
| `learning_rate` | `1.0e-2` | initial Adam step |
| `decay` | `0.3` | multiplicative decay applied to the step at each plateau |
| `patience` | `250` | epochs without validation improvement before decaying, and `3·patience` before stopping |
| `seed` | `20260730` | RNG seed of the weight initialization, so a retraining is reproducible |
| `verbose` | `true` | print the validation curve as it goes |
"""
struct TrainingOptions
    hidden::Vector{Int}
    activation::Symbol
    epochs::Int
    batchsize::Int
    learning_rate::Float64
    decay::Float64
    patience::Int
    seed::Int
    verbose::Bool
end

function TrainingOptions(;
        hidden::AbstractVector{<:Integer} = [32, 32],
        activation::Symbol = :tanh,
        epochs::Integer = 4000,
        batchsize::Integer = 128,
        learning_rate::Real = 1.0e-2,
        decay::Real = 0.3,
        patience::Integer = 250,
        seed::Integer = 20260730,
        verbose::Bool = true,
    )
    all(>(0), hidden) || throw(ArgumentError("hidden widths must be positive"))
    epochs > 0 || throw(ArgumentError("`epochs` must be positive"))
    batchsize ≥ 0 || throw(ArgumentError("`batchsize` must be non-negative"))
    0 < learning_rate || throw(ArgumentError("`learning_rate` must be positive"))
    0 < decay ≤ 1 || throw(ArgumentError("`decay` must lie in (0, 1]"))
    patience > 0 || throw(ArgumentError("`patience` must be positive"))
    # Reject an unknown activation here rather than at evaluation time. The
    # table holds smooth functions only, on purpose: a kink in the activation is
    # a kink in ℙ, hence a wrong sensitivity — a silent error, not a crash.
    activation isa Symbol && haskey(ACTIVATIONS, activation) || throw(
        ArgumentError(
            "unknown hidden activation :$activation; the surrogate format knows " *
                "$(sort(collect(keys(ACTIVATIONS))))"
        )
    )
    return TrainingOptions(
        collect(Int, hidden), activation, Int(epochs), Int(batchsize),
        Float64(learning_rate), Float64(decay), Int(patience), Int(seed), verbose
    )
end

"""
    network_widths(opts, box, spec) -> Vector{Int}

The full `[n_features, hidden…, n_outputs]` architecture implied by the feature
box and the output specification. Both the fallback and the extension go through
this, so the network the extension trains is the network the surrogate expects.
"""
network_widths(opts::TrainingOptions, box::SampleBox, spec::AbstractOutputSpec) =
    [length(box); opts.hidden; noutputs(spec)]

"""
    assemble_surrogate(net, spec, box, scaling, provenance) -> NeuralSurrogate

Bundle a trained network with the box it was trained on.

Centralized on purpose: the surrogate's validity limits *are* the sampling box,
and the feature list *is* the box's names. Letting a caller pass them
separately is how the two drift apart.
"""
function assemble_surrogate(
        net::MLP,
        spec::AbstractOutputSpec,
        box::SampleBox,
        scaling::NamedTuple,
        provenance::Provenance,
    )
    return NeuralSurrogate(;
        net, features = box.names, output = spec,
        scaling...,
        domain_lo = box.lo, domain_hi = box.hi, provenance
    )
end

"""
    train_surrogate(spec, box, train, validation; options = TrainingOptions(),
                    teacher_name = "", notes = "", history = nothing)
        -> NeuralSurrogate

Fit a surrogate of output specification `spec` over the sampling `box`, on the
datasets produced by [`generate_dataset`](@ref), and return it with its
[`Provenance`](@ref) filled in from the held-out set.

**Requires the training extension.** Run

```julia
import Lux, Optimisers, Zygote
```

before calling. Without them this fallback method raises: evaluation of an
already-trained surrogate needs none of the three, so they are weak
dependencies rather than dependencies (`scripts/nn/` carries an environment
that has them).

Pass a `Vector` as `history` to have the learning curve recorded into it, one
`(; epoch, train, validation)` per epoch — which is how the committed training
figure of the documentation is produced without anything being fitted at
build time.
"""
train_surrogate(args...; kwargs...) = error(
    "training a neural surrogate requires the Lux extension: run " *
        "`import Lux, Optimisers, Zygote` first (the environment in `scripts/nn/` " *
        "has them). Evaluating an already-trained surrogate — `load_surrogate`, " *
        "`NeuralHillInclusion`, every scheme — needs none of them."
)

# ─── Reporting ───────────────────────────────────────────────────────────────

"""
    report_surrogate([io], s, data::Dataset; labels = nothing)

Print the per-component validation table of `s` over `data`: the relative error
in the space the network predicts, component by component, plus the worst entry.

Dependency-free, so it works wherever a surrogate can be evaluated. `labels`
overrides the component names, which are otherwise `1:n`.
"""
function report_surrogate(
        io::IO, s::NeuralSurrogate, data::Dataset;
        labels::Union{Nothing, AbstractVector} = nothing
    )
    v = validate_surrogate(s, data)
    nz = length(v.max_rel_error)
    names = labels === nothing ? string.(1:nz) : string.(labels)
    length(names) == nz ||
        throw(DimensionMismatch("`labels` must have $nz entries, got $(length(names))"))
    println(
        io, "surrogate :", spec_name(s.output), "/:", class_name(s.output.class),
        " — ", nsamples(data), " held-out samples"
    )
    println(io, "  component        max rel.err     rms rel.err")
    for i in 1:nz
        @printf(io, "  %-14s   %10.3e      %10.3e\n", names[i], v.max_rel_error[i], v.rms_rel_error[i])
    end
    @printf(io, "  worst relative to the tensor magnitude: %.3e\n", v.max_block_error)
    return v
end

report_surrogate(s::NeuralSurrogate, data::Dataset; kw...) =
    report_surrogate(stdout, s, data; kw...)

"""
    component_labels(class) -> Vector{Symbol}

Human-readable names of a class's independent components, for
[`report_surrogate`](@ref).
"""
component_labels(::HillISO) = [:α, :β]
component_labels(::HillTI) = [:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₅, :ℓ₆]
component_labels(::HillISO2) = [:p]
component_labels(::HillTI2) = [:a, :b]
component_labels(::Union{StrainLocTI, StressLocTI}) = [:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₄, :ℓ₅, :ℓ₆]
component_labels(::StrainLocCubic) = [:α, :β, :γ]
component_labels(::HillOrtho) =
    [:C₁₁, :C₂₂, :C₃₃, :C₁₂, :C₁₃, :C₂₃, :C₄₄, :C₅₅, :C₆₆]

"""
    component_labels(spec::AbstractOutputSpec) -> Vector{Symbol}

Names of the network's outputs. For [`AffineHill`](@ref) the class labels are
repeated once per shape tensor, suffixed `𝕌` and `𝕍`.
"""
component_labels(spec::DimensionlessHill) = component_labels(spec.class)

function component_labels(spec::AffineHill)
    base = component_labels(spec.class)
    nterms(spec) == 1 && return base
    return vcat(
        [Symbol(b, "(𝕌)") for b in base],
        [Symbol(b, "(𝕍)") for b in base],
    )
end
