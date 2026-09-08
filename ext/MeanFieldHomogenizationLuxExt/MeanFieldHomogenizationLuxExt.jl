# =============================================================================
#  MeanFieldHomogenizationLuxExt — the optimizer half of the neural-surrogate pipeline.
#
#  This extension owns exactly one thing: `NeuralInclusions.train_surrogate`.
#  Everything else — the network, the physics of the decode, the sampling, the
#  serialization, the inclusions, the schemes — lives in the package and needs
#  none of `Lux`, `Zygote` or `Optimisers`.
#
#  The seam that makes that possible is the last step of `train_surrogate`:
#  once fitted, the parameters are written back into the plain
#  `NeuralInclusions.MLP` of `src/NeuralInclusions/mlp.jl`.  From that moment the
#  surrogate is autonomous — serializable, loadable anywhere, and traversable by
#  `ForwardDiff`.  Nothing of Lux survives in the trained object.
#
#  For that hand-off to be exact rather than approximate, the `Lux.Chain` built
#  here is *isomorphic* to the target `MLP`: same widths, and — importantly —
#  the very same activation function objects, taken from
#  `NeuralInclusions.activation`.  Using `NNlib.softplus` during training and
#  our own afterwards would leave a small, mystifying discrepancy between the
#  training loss and the validation error reported by the package.
# =============================================================================

module MeanFieldHomogenizationLuxExt

import MeanFieldHomogenization
import Lux
import Optimisers
import Zygote
import Random
import Dates

const NI = MeanFieldHomogenization.NeuralInclusions

# ─── The isomorphic Lux chain ────────────────────────────────────────────────

"""
    _chain(widths, activations) -> Lux.Chain

`Lux.Chain` matching an [`NI.MLP`](@ref) layer for layer. The activations are
looked up in `NI.ACTIVATIONS`, so the trained network and the extracted one
evaluate identically.
"""
function _chain(widths::AbstractVector{Int}, acts::AbstractVector{Symbol})
    layers = ntuple(length(widths) - 1) do k
        Lux.Dense(widths[k] => widths[k + 1], NI.activation(acts[k]))
    end
    return Lux.Chain(layers...)
end

"""
    _to_mlp(ps, widths, activations) -> NI.MLP

Copy Lux parameters into a dependency-free [`NI.MLP`](@ref).

`Lux.Dense` stores `weight` as `n_out × n_in` and `bias` as a length-`n_out`
vector, which is exactly `NNDense`'s layout, so this is a copy and not a
reinterpretation. `vec` guards against a trailing singleton dimension.
"""
function _to_mlp(ps, widths::AbstractVector{Int}, acts::AbstractVector{Symbol})
    layers = ntuple(length(widths) - 1) do k
        p = ps[k]
        W = Matrix{Float64}(p.weight)
        b = Vector{Float64}(vec(p.bias))
        NI.NNDense(W, b, NI.activation(acts[k]))
    end
    return NI.MLP(layers)
end

# ─── Standardized training matrices ──────────────────────────────────────────
#
#  Lux works on `features × batch` matrices, which is already the `Dataset`
#  layout.  What has to happen here is the standardization, using the scaling
#  fitted on the *training* set alone.

function _standardize_inputs(X, scaling)
    return Float32.((X .- scaling.x_shift) ./ scaling.x_scale)
end

function _standardize_targets(Z, scaling)
    Y = similar(Z)
    for i in axes(Z, 1)
        for j in axes(Z, 2)
            Y[i, j] = NI.apply_transform(scaling.y_kind[i], Z[i, j])
        end
    end
    return Float32.((Y .- scaling.y_shift) ./ scaling.y_scale)
end

# Float32 is deliberate for the *fit*: Adam on a few thousand parameters gains
# nothing from double precision, and Lux's default initializers are Float32
# anyway. The extracted `MLP` is widened back to Float64 in `_to_mlp`, and the
# accuracy that matters — the one the tests assert — is measured afterwards by
# `validate_surrogate` on the Float64 network.

_mse(Ŷ, Y) = sum(abs2, Ŷ .- Y) / length(Y)

# Below this step Adam no longer moves a Float32 parameter meaningfully, so a
# further decay would only burn epochs.
const _LR_FLOOR = 1.0e-6

# ─── The fit ─────────────────────────────────────────────────────────────────

function NI.train_surrogate(
        spec::NI.AbstractOutputSpec,
        box::NI.SampleBox,
        train::NI.Dataset,
        validation::NI.Dataset;
        options::NI.TrainingOptions = NI.TrainingOptions(),
        teacher_name::AbstractString = "",
        notes::AbstractString = "",
        history::Union{Nothing, AbstractVector} = nothing,
    )
    NI.nsamples(train) ≥ 2 ||
        throw(ArgumentError("training needs at least 2 samples"))
    NI.nsamples(validation) ≥ 1 || throw(
        ArgumentError(
            "training needs a non-empty held-out set: it drives both the early " *
                "stopping and the error recorded in the surrogate's provenance, " *
                "which the test tolerances are derived from. Pass `nvalidation` to " *
                "`generate_dataset`."
        )
    )
    train.features == box.names || throw(
        ArgumentError(
            "the training set carries features $(Tuple(train.features)) but the box " *
                "declares $(Tuple(box.names))"
        )
    )

    scaling = NI.fit_scaling(train; log_threshold = options.log_threshold)
    Xt = _standardize_inputs(train.X, scaling)
    Yt = _standardize_targets(train.Z, scaling)
    Xv = _standardize_inputs(validation.X, scaling)
    Yv = _standardize_targets(validation.Z, scaling)

    widths = NI.network_widths(options, box, spec)
    nl = length(widths) - 1
    acts = [fill(options.activation, nl - 1); :identity]
    chain = _chain(widths, acts)

    rng = Random.Xoshiro(options.seed)
    ps, st = Lux.setup(rng, chain)

    lr = options.learning_rate
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)

    n = NI.nsamples(train)
    bs = options.batchsize == 0 ? n : min(options.batchsize, n)

    best_ps = deepcopy(ps)
    best_loss = Inf
    best_epoch = 0
    since_best = 0

    for epoch in 1:options.epochs
        perm = Random.randperm(rng, n)
        for s in 1:bs:n
            idx = perm[s:min(s + bs - 1, n)]
            xb, yb = Xt[:, idx], Yt[:, idx]
            gs = first(
                Zygote.gradient(p -> _mse(first(Lux.apply(chain, xb, p, st)), yb), ps)
            )
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end

        vloss = _mse(first(Lux.apply(chain, Xv, ps, st)), Yv)
        if history !== nothing
            tloss = _mse(first(Lux.apply(chain, Xt, ps, st)), Yt)
            push!(history, (; epoch, train = Float64(tloss), validation = Float64(vloss)))
        end
        if vloss < best_loss * (1 - 1.0e-4)
            best_loss, best_epoch, since_best = vloss, epoch, 0
            best_ps = deepcopy(ps)
        else
            since_best += 1
        end

        # A plateau buys a smaller step; a plateau once the step has bottomed out
        # ends the run. Written as a nested branch rather than an `elseif` chain
        # because the two conditions share the counter, and an `elseif` made the
        # stopping criterion reachable only through the decay schedule.
        if since_best ≥ options.patience
            if lr > _LR_FLOOR
                lr *= options.decay
                opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
                since_best = 0
                options.verbose &&
                    @info "plateau — decaying the step" epoch lr best_loss
            else
                options.verbose &&
                    @info "no improvement at the smallest step — stopping" epoch best_epoch best_loss
                break
            end
        end

        if options.verbose && (epoch == 1 || epoch % 500 == 0)
            @info "training" epoch validation_mse = vloss best = best_loss
        end
    end

    net = _to_mlp(best_ps, widths, acts)

    # Assemble twice on purpose: the provenance records the validation error,
    # which can only be measured once the surrogate exists.  The first object is
    # a scaffold, the second is what the caller gets.
    scaffold = NI.assemble_surrogate(net, spec, box, scaling, NI.Provenance())
    v = NI.validate_surrogate(scaffold, validation)
    prov = NI.Provenance(;
        teacher = teacher_name,
        nsamples = NI.nsamples(train),
        nvalidation = NI.nsamples(validation),
        max_block_error = v.max_block_error,
        max_rel_error = v.max_rel_error,
        rms_rel_error = v.rms_rel_error,
        epochs = best_epoch,
        created = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"),
        notes = notes,
    )
    options.verbose && @info "trained" widths best_epoch worst_rel_error = v.worst
    return NI.assemble_surrogate(net, spec, box, scaling, prov)
end

end # module
