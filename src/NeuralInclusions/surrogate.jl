# =============================================================================
#  surrogate.jl — a trained network together with everything needed to trust it.
#
#  A bare `MLP` is not usable as physics: it needs the feature list it was fed,
#  the standardization of both ends, the output specification that turns numbers
#  into a tensor, the box it was trained on, and the error it achieved there.
#  `NeuralSurrogate` carries all of it in one flat, serializable object.
#
#  The two fields that are easiest to underrate are the *domain box* and the
#  *provenance*.  A network is an interpolator: inside its box it is as good as
#  its validation error says, and outside it is an extrapolation with no error
#  bar whatsoever — silently, confidently wrong.  So the box travels with the
#  weights and is checked on every evaluation.  And because the test suite
#  asserts scheme results against the surrogate's own recorded error, that
#  number has to live with the model rather than in a comment.
# =============================================================================

"""
    Provenance

Where a surrogate came from and how well it does, recorded at training time and
serialized with the weights.

| Field | Meaning |
|---|---|
| `teacher` | what generated the labels, e.g. `"hill_tensor(Ellipsoid, TensISO)"` |
| `nsamples` / `nvalidation` | training and held-out set sizes |
| `max_block_error` | worst error over the held-out set, relative to the tensor's own magnitude — the headline number |
| `max_rel_error` / `rms_rel_error` | per-component relative error, a diagnostic |
| `epochs` | how many passes the optimizer actually ran (after early stopping) |
| `created` | timestamp, for telling two retrainings apart |
| `notes` | free text |

`max_block_error` is the number a test tolerance should be derived from — never a
hard-coded literal, so that a retraining cannot silently loosen a threshold. The
per-component vectors are diagnostics: a structurally vanishing component (`𝕍ᴬ`
has no `ℓ₃`) has no meaningful relative error of its own, which is exactly why the
headline number is measured against the block.
"""
struct Provenance
    teacher::String
    nsamples::Int
    nvalidation::Int
    max_block_error::Float64
    max_rel_error::Vector{Float64}
    rms_rel_error::Vector{Float64}
    epochs::Int
    created::String
    notes::String
end

function Provenance(;
        teacher::AbstractString = "",
        nsamples::Integer = 0,
        nvalidation::Integer = 0,
        max_block_error::Real = Inf,
        max_rel_error::AbstractVector{<:Real} = Float64[],
        rms_rel_error::AbstractVector{<:Real} = Float64[],
        epochs::Integer = 0,
        created::AbstractString = "",
        notes::AbstractString = "",
    )
    return Provenance(
        String(teacher), Int(nsamples), Int(nvalidation), Float64(max_block_error),
        collect(Float64, max_rel_error), collect(Float64, rms_rel_error),
        Int(epochs), String(created), String(notes)
    )
end

"""
    worst_error(p::Provenance) -> Float64

Worst error recorded on the held-out set, relative to the tensor's own magnitude
(`max_block_error`), or `Inf` when the surrogate has never been validated — so
that a tolerance derived from it fails loudly rather than passing by accident.
"""
worst_error(p::Provenance) = p.max_block_error

"""
    NeuralSurrogate

A trained network plus its physical contract: which features it consumes, how
both ends are standardized, which tensor it produces, and over what box it is
entitled to be believed.

# Construction

Built by [`train_surrogate`](@ref MeanFieldHomogenization.NeuralInclusions.train_surrogate) or read back by
[`load_surrogate`](@ref).
The direct constructor is keyword-only and validates every length against the
network's own input and output widths, because a surrogate whose feature list
and input layer disagree is a silent mis-evaluation rather than an error.

# Evaluation

    (s::NeuralSurrogate)(x_raw, P₀, frame; guard = :warn)

`x_raw` is the *unstandardized* feature vector — the inclusion builds it, since
only the inclusion knows its own geometry (see `_raw_features`). `frame` is the
symmetry axis or material frame passed straight to [`build`](@ref).

Generic in the element type of `x_raw`: a `ForwardDiff.Dual` feature yields a
`Dual` tensor, which is what makes a *morphology* sensitivity possible.

See also [`Provenance`](@ref), [`validate_surrogate`](@ref),
[`save_surrogate`](@ref).
"""
struct NeuralSurrogate{M <: MLP, O <: AbstractOutputSpec}
    net::M
    features::Vector{Symbol}
    x_shift::Vector{Float64}
    x_scale::Vector{Float64}
    y_kind::Vector{Symbol}
    y_shift::Vector{Float64}
    y_scale::Vector{Float64}
    output::O
    domain_lo::Vector{Float64}
    domain_hi::Vector{Float64}
    provenance::Provenance
end

function NeuralSurrogate(;
        net::MLP,
        features::AbstractVector{Symbol},
        output::AbstractOutputSpec,
        x_shift::AbstractVector{<:Real} = zeros(length(features)),
        x_scale::AbstractVector{<:Real} = ones(length(features)),
        y_kind::AbstractVector{Symbol} = fill(:identity, noutputs(output)),
        y_shift::AbstractVector{<:Real} = zeros(noutputs(output)),
        y_scale::AbstractVector{<:Real} = ones(noutputs(output)),
        domain_lo::AbstractVector{<:Real} = fill(-Inf, length(features)),
        domain_hi::AbstractVector{<:Real} = fill(Inf, length(features)),
        provenance::Provenance = Provenance(),
    )
    nf, no = length(features), noutputs(output)
    n_in(net) == nf || throw(
        DimensionMismatch(
            "the network takes $(n_in(net)) inputs but $nf features are declared " *
                "($(Tuple(features)))"
        )
    )
    n_out(net) == no || throw(
        DimensionMismatch(
            "the network produces $(n_out(net)) outputs but the :$(spec_name(output)) " *
                "specification of class :$(class_name(output.class)) needs $no"
        )
    )
    for (name, v, n) in (
            ("x_shift", x_shift, nf), ("x_scale", x_scale, nf),
            ("domain_lo", domain_lo, nf), ("domain_hi", domain_hi, nf),
            ("y_kind", y_kind, no), ("y_shift", y_shift, no), ("y_scale", y_scale, no),
        )
        length(v) == n ||
            throw(DimensionMismatch("`$name` has length $(length(v)), expected $n"))
    end
    if any(iszero, x_scale)
        throw(
            ArgumentError(
                "`x_scale` has a zero entry — a constant feature cannot be " *
                    "standardized; drop it from the feature list instead"
            )
        )
    end
    if any(iszero, y_scale)
        throw(
            ArgumentError(
                "`y_scale` has a zero entry — a constant output component cannot " *
                    "be standardized"
            )
        )
    end
    foreach(_check_transform_name, y_kind)
    return NeuralSurrogate(
        net, collect(Symbol, features),
        collect(Float64, x_shift), collect(Float64, x_scale),
        collect(Symbol, y_kind), collect(Float64, y_shift), collect(Float64, y_scale),
        output, collect(Float64, domain_lo), collect(Float64, domain_hi), provenance
    )
end

_check_transform_name(kind::Symbol) =
    kind in TRANSFORMS || _bad_transform(kind)

hill_class(s::NeuralSurrogate) = s.output.class
tensor_order(s::NeuralSurrogate) = tensor_order(s.output.class)
ncomponents(s::NeuralSurrogate) = ncomponents(s.output.class)

function Base.show(io::IO, s::NeuralSurrogate)
    e = worst_error(s.provenance)
    err = isfinite(e) ? @sprintf("%.2e", e) : "not validated"
    return print(
        io,
        "NeuralSurrogate(", join(s.features, ", "), " → :",
        spec_name(s.output), "/:", class_name(s.output.class),
        ", order ", tensor_order(s), "; ", s.net, "; worst rel. error ", err, ")"
    )
end

# ─── The domain guard ────────────────────────────────────────────────────────

"""
    check_domain(s, x_raw, guard) -> Nothing

Verify that every feature lies inside the box the surrogate was trained on.

`guard` is `:error` (refuse), `:warn` (warn once per call site, the default) or
`:none` (trust the caller). Extrapolating a network is not a graceful
degradation — the error can be arbitrary and there is no diagnostic in the
result — so the default is deliberately noisy.

The bounds are inclusive up to a relative slack of `1e-9` of the box width, so
that reconstructing a bound by a slightly different arithmetic path — `log(0.1)`
against a stored `log(0.10)` — does not trip the guard on the last bit.
"""
function check_domain(s::NeuralSurrogate, x_raw::AbstractVector, guard::Symbol)
    guard === :none && return nothing
    guard in (:warn, :error, :none) || throw(
        ArgumentError("`guard` must be :warn, :error or :none, got :$guard")
    )
    for (i, name) in enumerate(s.features)
        xi = _value(x_raw[i])
        lo, hi = s.domain_lo[i], s.domain_hi[i]
        # The bounds themselves are legitimate queries, and a caller who writes
        # `log(0.1)` where the box stored `log(0.10)` is one ulp outside. Slacken
        # by a relative whisker of the box width: far too small to hide genuine
        # extrapolation, far larger than any round-off in reconstructing a bound.
        tol = _DOMAIN_SLACK * max(abs(hi - lo), one(hi))
        (lo - tol ≤ xi ≤ hi + tol) && continue
        msg = "feature :$name = $(xi) is outside the box this surrogate was " *
            "trained on ([$(lo), $(hi)]). A network does not extrapolate: the " *
            "result is unbounded and carries no error estimate. Retrain over the " *
            "wider box, or pass `guard = :none` if you accept that."
        guard === :error ? throw(ArgumentError(msg)) : @warn msg
    end
    return nothing
end

# Relative slack on the domain bounds — see `check_domain`.
const _DOMAIN_SLACK = 1.0e-9

# The box is compared on values, so a `ForwardDiff.Dual` feature is checked on
# its primal part rather than erroring on the comparison.
_value(x::Number) = x
_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

# ─── Evaluation ──────────────────────────────────────────────────────────────

function (s::NeuralSurrogate)(
        x_raw::AbstractVector, P₀::TensND.AbstractTens, frame;
        guard::Symbol = :warn
    )
    check_domain(s, x_raw, guard)
    x̂ = (x_raw .- s.x_shift) ./ s.x_scale
    ŷ = s.net(x̂)
    z = map(invert_transform, s.y_kind, ŷ .* s.y_scale .+ s.y_shift)
    # `x_raw` and the feature names go through too: an anchored specification
    # evaluates its baseline from the features, and it is the only place that
    # information is still available.
    return decode(s.output, z, P₀, frame, x_raw, s.features)
end

"""
    predict_components(s, x_raw; guard = :none) -> Vector

The surrogate's raw prediction — the dimensionless or shape components, before
any contraction with the reference moduli. Used by
[`validate_surrogate`](@ref), which compares against labels living in exactly
this space, and by training diagnostics.
"""
function predict_components(
        s::NeuralSurrogate, x_raw::AbstractVector; guard::Symbol = :none
    )
    check_domain(s, x_raw, guard)
    x̂ = (x_raw .- s.x_shift) ./ s.x_scale
    ŷ = s.net(x̂)
    return map(invert_transform, s.y_kind, ŷ .* s.y_scale .+ s.y_shift)
end
