# =============================================================================
#  io.jl — reading and writing a trained surrogate.
#
#  The format is JSON, for three reasons and not because it is fast:
#
#  * `JSON3` is already a strong dependency of the package, so loading a model
#    pulls in nothing new — the same argument that keeps the evaluator itself
#    dependency-free;
#  * a committed model is reviewable.  A diff on a retrained network shows the
#    architecture, the box and the validation error changing, which a binary
#    blob would not;
#  * it is version-stable in a way `Serialization` is not: a `.jls` file written
#    by one Julia version is not guaranteed to load in the next, which is
#    disqualifying for an artifact committed to a repository.
#
#  Every field needed to reproduce an evaluation bit-for-bit is written,
#  including the format version. Nothing is inferred at load time.
# =============================================================================

"""
    SURROGATE_FORMAT

Version of the on-disk surrogate format. Bumped when a field changes meaning;
[`load_surrogate`](@ref) refuses a newer major version rather than guessing.
"""
const SURROGATE_FORMAT = v"1.2.0"

"""
    MODEL_DIR

Directory of the surrogates shipped with the package. See
[`model_path`](@ref).
"""
const MODEL_DIR = joinpath(@__DIR__, "models")

"""
    model_path(name) -> String

Absolute path of a shipped model, with or without the `.json` extension:

```julia
load_surrogate(model_path("spheroid_hill_iso_elastic"))
```

Lists what is available when the name is not found, which is more useful than a
bare `SystemError` from the reader.
"""
function model_path(name::AbstractString)
    file = endswith(name, ".json") ? String(name) : name * ".json"
    path = joinpath(MODEL_DIR, file)
    isfile(path) && return path
    available = isdir(MODEL_DIR) ?
        sort([replace(f, ".json" => "") for f in readdir(MODEL_DIR) if endswith(f, ".json")]) :
        String[]
    return throw(
        ArgumentError(
            "no shipped surrogate named \"$name\" in $MODEL_DIR; available: " *
                (isempty(available) ? "(none)" : join(available, ", "))
        )
    )
end

"""
    shipped_models() -> Vector{String}

Names of the surrogates committed with the package, loadable by
[`model_path`](@ref).
"""
shipped_models() =
    isdir(MODEL_DIR) ?
    sort([replace(f, ".json" => "") for f in readdir(MODEL_DIR) if endswith(f, ".json")]) :
    String[]

# ─── Writing ─────────────────────────────────────────────────────────────────

"""
    save_surrogate(path, s::NeuralSurrogate) -> String

Write `s` to `path` as pretty-printed JSON and return the path.

The round trip is exact: weights go out as `Float64` decimals with full
precision, so `load_surrogate(save_surrogate(p, s))` reproduces every
prediction bit-for-bit.
"""
function save_surrogate(path::AbstractString, s::NeuralSurrogate)
    mkpath(dirname(abspath(path)))
    payload = _to_dict(s)
    open(path, "w") do io
        JSON3.pretty(io, payload)
        println(io)
    end
    return String(path)
end

# JSON cannot represent `Inf` or `NaN`. A surrogate that was never validated
# carries `max_block_error = Inf` on purpose — so that a tolerance derived from it
# fails loudly — and that value has to survive the round trip, hence the mapping
# to `null` and back rather than a silent clamp.
_json_number(x::Real) = isfinite(x) ? Float64(x) : nothing

function _read_block_error(p)
    if haskey(p, :max_block_error)
        v = p.max_block_error
        v === nothing && return Inf
        return Float64(v)
    end
    # Format 1.0 had no such field; its headline number was the per-component
    # maximum.
    return isempty(p.max_rel_error) ? Inf : maximum(p.max_rel_error)
end

function _to_dict(s::NeuralSurrogate)
    return Dict(
        "format" => string(SURROGATE_FORMAT),
        "network" => Dict(
            "widths" => layer_widths(s.net),
            "activations" => string.(layer_activations(s.net)),
            "weights" => [collect(Iterators.flatten(eachrow(l.W))) for l in s.net.layers],
            "biases" => [copy(l.b) for l in s.net.layers],
        ),
        "features" => string.(s.features),
        "input" => Dict("shift" => s.x_shift, "scale" => s.x_scale),
        "output" => Dict(
            "spec" => string(spec_name(s.output)),
            # `baseline` appears only for an anchored specification. A reader of
            # an older file finds none, and `output_spec` says so rather than
            # inventing one — the minor bump is for this field alone.
            "baseline" => let b = spec_baseline(s.output)
                b === nothing ? nothing : string(b)
            end,
            "class" => string(class_name(s.output.class)),
            "kind" => string.(s.y_kind),
            "shift" => s.y_shift,
            "scale" => s.y_scale,
        ),
        "domain" => Dict("lo" => s.domain_lo, "hi" => s.domain_hi),
        "provenance" => Dict(
            "teacher" => s.provenance.teacher,
            "nsamples" => s.provenance.nsamples,
            "nvalidation" => s.provenance.nvalidation,
            # JSON has no `Infinity`: an unvalidated model writes `null`.
            "max_block_error" => _json_number(s.provenance.max_block_error),
            "max_rel_error" => s.provenance.max_rel_error,
            "rms_rel_error" => s.provenance.rms_rel_error,
            "epochs" => s.provenance.epochs,
            "created" => s.provenance.created,
            "notes" => s.provenance.notes,
        ),
    )
end

# ─── Reading ─────────────────────────────────────────────────────────────────

"""
    load_surrogate(path) -> NeuralSurrogate

Read a surrogate written by [`save_surrogate`](@ref).

Needs neither Lux nor any other training dependency: a model is trained once and
evaluated everywhere.
"""
function load_surrogate(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such surrogate file: $path"))
    d = JSON3.read(read(path, String))
    _check_format(get(d, :format, nothing), path)

    net = _net_from(d.network)
    spec = output_spec(
        Symbol(d.output.spec), Symbol(d.output.class),
        get(d.output, :baseline, nothing),
    )
    p = d.provenance
    prov = Provenance(;
        teacher = p.teacher, nsamples = p.nsamples, nvalidation = p.nvalidation,
        # Absent from format 1.0: fall back to the per-component maximum, which is
        # what that version recorded as its headline number.
        max_block_error = _read_block_error(p),
        max_rel_error = collect(Float64, p.max_rel_error),
        rms_rel_error = collect(Float64, p.rms_rel_error),
        epochs = p.epochs, created = p.created, notes = p.notes,
    )
    return NeuralSurrogate(;
        net,
        features = Symbol.(d.features),
        output = spec,
        x_shift = collect(Float64, d.input.shift),
        x_scale = collect(Float64, d.input.scale),
        y_kind = Symbol.(d.output.kind),
        y_shift = collect(Float64, d.output.shift),
        y_scale = collect(Float64, d.output.scale),
        domain_lo = collect(Float64, d.domain.lo),
        domain_hi = collect(Float64, d.domain.hi),
        provenance = prov,
    )
end

function _check_format(raw, path)
    raw === nothing && throw(
        ArgumentError("$path declares no `format` field — it is not a surrogate file")
    )
    v = try
        VersionNumber(String(raw))
    catch
        throw(ArgumentError("$path declares an unparsable format \"$raw\""))
    end
    v.major == SURROGATE_FORMAT.major || throw(
        ArgumentError(
            "$path is in surrogate format $v, incompatible with the $(SURROGATE_FORMAT) " *
                "this version reads. Retrain the model, or install the matching " *
                "MeanFieldHomogenization version."
        )
    )
    return v
end

function _net_from(n)
    widths = collect(Int, n.widths)
    acts = Symbol.(n.activations)
    nl = length(widths) - 1
    length(acts) == nl || throw(
        ArgumentError(
            "the stored network declares $(length(widths)) widths (i.e. $nl layers) " *
                "but $(length(acts)) activations"
        )
    )
    length(n.weights) == nl && length(n.biases) == nl || throw(
        ArgumentError(
            "the stored network declares $nl layers but carries " *
                "$(length(n.weights)) weight blocks and $(length(n.biases)) bias blocks"
        )
    )
    layers = ntuple(nl) do k
        nin, nout = widths[k], widths[k + 1]
        w = collect(Float64, n.weights[k])
        length(w) == nin * nout || throw(
            ArgumentError(
                "layer $k should hold $(nin * nout) weights ($nout × $nin) but " *
                    "carries $(length(w))"
            )
        )
        # Written row-major by `_to_dict`, so read back the same way.
        W = permutedims(reshape(w, nin, nout))
        b = collect(Float64, n.biases[k])
        NNDense(W, b, activation(acts[k]))
    end
    return MLP(layers)
end
