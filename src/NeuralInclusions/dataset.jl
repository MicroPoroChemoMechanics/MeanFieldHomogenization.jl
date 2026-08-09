# =============================================================================
#  dataset.jl — sampling the parameter box and labeling it with a teacher.
#
#  The *teacher* abstraction is the whole point of this file, and the only seam
#  that has to change to move from the analytic pilot to a complex morphology:
#
#      teacher(x_shape::AbstractVector, P₀) -> AbstractTens
#
#  Today the caller closes over `hill_tensor(Ellipsoid(…), P₀)`, whose labels
#  are exact.  Tomorrow it closes over a finite-element solve, or over anything
#  else that can answer the same question.  Nothing else in the module knows
#  the difference.
#
#  Sampling is a Halton low-discrepancy sequence rather than pseudo-random
#  draws: it covers a two- or three-dimensional box far more evenly at small
#  sample counts, and — because it is a deterministic function of the sample
#  index — a dataset is reproducible with no RNG state to carry around.  The
#  held-out set is simply the continuation of the same sequence, so it is
#  interleaved with the training set in parameter space and measures
#  interpolation error, which is what the surrogate is for.
# =============================================================================

# ─── The parameter box ───────────────────────────────────────────────────────

"""
    SampleBox(names, lo, hi; scale = :linear)

Axis-aligned sampling box over the *raw feature* space, one entry per feature.

`scale` is per feature: `:linear` samples uniformly between `lo` and `hi`,
`:log` samples uniformly in the logarithm — the right choice for an aspect
ratio, whose interesting behavior is spread over decades rather than over an
interval.

The box travels into the trained surrogate as its
[`check_domain`](@ref) limits, which is why it is a first-class object and not
just a pair of loops in a script.

# Example

```julia
# a spheroid aspect ratio over two decades, plus the matrix Poisson ratio
SampleBox([:log_aspect, :nu0], [log(1/20), 0.0], [log(20), 0.49])
```

Note that `lo`/`hi` are given in *feature* units: `:log_aspect` is already a
logarithm, so `scale = :linear` on it samples the exponent uniformly, which is
what a log sweep of the aspect ratio means.
"""
struct SampleBox
    names::Vector{Symbol}
    lo::Vector{Float64}
    hi::Vector{Float64}
    scale::Vector{Symbol}
end

function SampleBox(
        names::AbstractVector{Symbol},
        lo::AbstractVector{<:Real},
        hi::AbstractVector{<:Real};
        scale::Union{Symbol, AbstractVector{Symbol}} = :linear,
    )
    n = length(names)
    length(lo) == n && length(hi) == n || throw(
        DimensionMismatch(
            "`lo` and `hi` must have one entry per feature ($n), got " *
                "$(length(lo)) and $(length(hi))"
        )
    )
    sc = scale isa Symbol ? fill(scale, n) : collect(Symbol, scale)
    length(sc) == n ||
        throw(DimensionMismatch("`scale` must have one entry per feature ($n)"))
    for (i, s) in enumerate(sc)
        s in (:linear, :log) ||
            throw(ArgumentError("feature :$(names[i]) has unknown scale :$s"))
        s === :log && lo[i] ≤ 0 && throw(
            ArgumentError(
                "feature :$(names[i]) is sampled on a :log scale but its lower " *
                    "bound is $(lo[i]) ≤ 0"
            )
        )
    end
    all(lo .< hi) || throw(ArgumentError("every `lo` must be strictly below its `hi`"))
    return SampleBox(collect(Symbol, names), collect(Float64, lo), collect(Float64, hi), sc)
end

Base.length(b::SampleBox) = length(b.names)

"""
    feature_index(box, name) -> Int

Position of a named feature in the box, with a readable error when absent.
"""
function feature_index(b::SampleBox, name::Symbol)
    i = findfirst(==(name), b.names)
    i === nothing && throw(
        ArgumentError("this box has no feature :$name (it has $(Tuple(b.names)))")
    )
    return i
end

# ─── Halton low-discrepancy sampling ─────────────────────────────────────────

# The first primes, one per feature dimension.  Halton degrades in high
# dimension; the surrogates here use two or three features, comfortably inside
# the regime where it beats pseudo-random sampling.
const _HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)

"""
    halton(i, base) -> Float64

Radical-inverse of `i` in `base` — the `i`-th point of the one-dimensional
van der Corput sequence, in `(0, 1)`.
"""
function halton(i::Integer, base::Integer)
    f = 1.0
    r = 0.0
    k = i
    while k > 0
        f /= base
        r += f * (k % base)
        k ÷= base
    end
    return r
end

# Discard the first points of the sequence: the low-index Halton points of
# different bases are strongly correlated, which shows up as a visible diagonal
# streak in two dimensions.
const _HALTON_BURN = 32

"""
    sample_box(box, n; offset = 0) -> Matrix{Float64}

`length(box) × n` matrix of raw feature vectors, from the Halton sequence.

`offset` skips that many points, which is how a held-out set is drawn: pass
`offset = n_train` and the two sets are disjoint yet drawn from the same
well-distributed sequence.
"""
function sample_box(box::SampleBox, n::Integer; offset::Integer = 0)
    d = length(box)
    d ≤ length(_HALTON_BASES) || throw(
        ArgumentError(
            "Halton sampling is set up for at most $(length(_HALTON_BASES)) " *
                "features, got $d"
        )
    )
    n ≥ 1 || throw(ArgumentError("`n` must be at least 1"))
    X = Matrix{Float64}(undef, d, n)
    # The bounds are per-dimension, so their logs are invariant in `j`: taking
    # them once turns 2·d·n transcendentals into 2·d. The interpolation is then
    # a fused multiply-add either way, the log branch just working on the
    # pre-taken logarithms.
    lo = Vector{Float64}(undef, d)
    span = Vector{Float64}(undef, d)
    islog = BitVector(undef, d)
    for i in 1:d
        islog[i] = box.scale[i] === :log
        a, b = Float64(box.lo[i]), Float64(box.hi[i])
        lo[i] = islog[i] ? log(a) : a
        span[i] = (islog[i] ? log(b) : b) - lo[i]
    end
    for j in 1:n, i in 1:d
        u = halton(_HALTON_BURN + offset + j, _HALTON_BASES[i])
        x = muladd(u, span[i], lo[i])
        X[i, j] = islog[i] ? exp(x) : x
    end
    return X
end

"""
    grid_box(box, npts) -> Matrix{Float64}

Full tensor grid with `npts` points per feature — `length(box) × npts^d`
columns. Used for error maps and for hitting the corners of the box, which a
low-discrepancy sequence does not.
"""
function grid_box(box::SampleBox, npts::Integer)
    d = length(box)
    npts ≥ 2 || throw(ArgumentError("`npts` must be at least 2"))
    axes_ = map(1:d) do i
        if box.scale[i] === :log
            exp.(range(log(box.lo[i]), log(box.hi[i]); length = npts))
        else
            collect(range(box.lo[i], box.hi[i]; length = npts))
        end
    end
    cols = Iterators.product(axes_...)
    X = Matrix{Float64}(undef, d, npts^d)
    for (j, c) in enumerate(cols)
        X[:, j] .= c
    end
    return X
end

# ─── Labeling ───────────────────────────────────────────────────────────────

"""
    Dataset

Raw features and untransformed targets, one column per sample.

`Z` lives in the space the [`AbstractOutputSpec`](@ref) predicts — dimensionless
components for [`DimensionlessHill`](@ref), shape-tensor components for
[`AffineHill`](@ref) — *not* in the space of the tensor itself. Training and
[`validate_surrogate`](@ref) both work there, so the reference moduli never
enter the loss.
"""
struct Dataset
    X::Matrix{Float64}
    Z::Matrix{Float64}
    features::Vector{Symbol}
end

nsamples(d::Dataset) = size(d.X, 2)
Base.length(d::Dataset) = nsamples(d)

Base.show(io::IO, d::Dataset) = print(
    io, "Dataset(", nsamples(d), " samples, ", length(d.features),
    " features → ", size(d.Z, 1), " outputs)"
)

# The reference media used to expose the affine structure.  Two Poisson ratios
# far enough apart to condition the 2×2 solve, with `μ₀ = 1`: the shear modulus
# cancels out of the decomposition, so its value is arbitrary.
const _AFFINE_NU = (0.1, 0.45)

_iso_ref(nu) = Elasticity.iso_stiffness_E_nu(2 * (1 + nu), nu)   # μ₀ = 1

"""
    generate_dataset(geometry, response, spec, box, n; nvalidation = 0)
        -> (train::Dataset, validation::Dataset)

Sample `box`, label every point, and return the training and held-out sets.

The teacher is split in two callbacks, and the split is what makes the frame
convention safe:

- `geometry(x_shape) -> geom` builds the morphology from the **shape** features
  (the box's features minus `:nu0`). `geom` needs only to answer `semi_axes` and
  `inclusion_basis` — an `Ellipsoid` does, and so does any user type.
- `response(geom, P₀) -> AbstractTens` evaluates the tensor to be learned. For
  the pilot this is `hill_tensor`; for a heterogeneous morphology it is one of
  the localization tensors, and for an expensive one it is a solve.

Because the frame in which the components are read is then obtained from `geom`
through the very same `_class_frame` the inclusion uses at evaluation time, a
mismatch between the two is impossible by construction — rather than being a
silent 90° rotation discovered much later.

```julia
geometry(x) = Ellipsoid(1.0, 1.0, exp(x[1]))
response(g, C₀) = hill_tensor(g, C₀)
train, val = generate_dataset(geometry, response, spec, box, 4000; nvalidation = 1000)
```

What the labels are depends on `spec`:

- [`DimensionlessHill`](@ref) — one `response` call per sample, at the reference
  medium built from the sampled `ν₀` (elasticity) or at unit conductivity
  (transport); the target is `scale · ℙ`.
- [`AffineHill`](@ref) — two `response` calls per sample at two different
  Poisson ratios, and the shape tensors `𝕌ᴬ`, `𝕍ᴬ` are recovered by solving the
  exact 2×2 affine system componentwise. `ν₀` must *not* be in the box: the
  whole point is that it is not a degree of freedom.
"""
function generate_dataset(
        geometry,
        response,
        spec::AbstractOutputSpec,
        box::SampleBox,
        n::Integer;
        nvalidation::Integer = 0,
    )
    _check_box(spec, box)
    Xt = sample_box(box, n)
    Xv = nvalidation > 0 ? sample_box(box, nvalidation; offset = n) :
        Matrix{Float64}(undef, length(box), 0)
    return (
        Dataset(Xt, _label(geometry, response, spec, box, Xt), copy(box.names)),
        Dataset(Xv, _label(geometry, response, spec, box, Xv), copy(box.names)),
    )
end

function _check_box(spec::AbstractOutputSpec, box::SampleBox)
    has_nu = :nu0 in box.names
    if needs_nu(spec) && !has_nu
        throw(
            ArgumentError(
                "a :$(spec_name(spec)) surrogate of class :$(class_name(spec.class)) " *
                    "learns the ν₀ dependence, so :nu0 must be one of the box's " *
                    "features (it has $(Tuple(box.names)))"
            )
        )
    elseif !needs_nu(spec) && has_nu
        throw(
            ArgumentError(
                "a :$(spec_name(spec)) surrogate of class :$(class_name(spec.class)) " *
                    "reproduces the material dependence exactly, so :nu0 must not be " *
                    "a feature — including it would make the network fit a constant " *
                    "and hide the exactness"
            )
        )
    end
    return nothing
end

_shape_rows(box::SampleBox) = findall(!=(:nu0), box.names)

function _label(
        geometry, response, spec::AbstractOutputSpec, box::SampleBox, X::AbstractMatrix
    )
    n = size(X, 2)
    Z = Matrix{Float64}(undef, noutputs(spec), n)
    rows = _shape_rows(box)
    for j in 1:n
        geom = geometry(collect(view(X, rows, j)))
        frame = _class_frame(spec.class, geom)
        Z[:, j] .= _label_one(response, spec, box, geom, frame, view(X, :, j))
    end
    return Z
end

function _label_one(
        response, spec::DimensionlessHill, box::SampleBox, geom, frame, x_full
    )
    P₀ = _reference_medium(spec.class, box, x_full)
    P = response(geom, P₀)
    return collect(Float64, components(spec.class, P, frame)) .*
        dimensionless_scale(spec.class, P₀)
end

function _label_one(response, spec::AffineHill, _box::SampleBox, geom, frame, _x_full)
    class = spec.class
    nc = ncomponents(class)
    if tensor_order(class) == 2
        # A single term: ℙ_K = 𝕍ᴬ/k₀, so one call at unit conductivity is exact.
        K₀ = TensND.TensISO{3}(1.0)
        return collect(Float64, components(class, response(geom, K₀), frame))
    end
    # Two reference media expose the two shape tensors: for each component,
    #     c(ν) = d(ν)·U + m(ν)·W,
    # a 2×2 system whose right-hand sides are two `response` evaluations.
    C_a, C_b = _iso_ref(_AFFINE_NU[1]), _iso_ref(_AFFINE_NU[2])
    c_a = components(class, response(geom, C_a), frame)
    c_b = components(class, response(geom, C_b), frame)
    (d_a, m_a) = material_coeffs(class, C_a)
    (d_b, m_b) = material_coeffs(class, C_b)
    det = d_a * m_b - d_b * m_a
    abs(det) > 1.0e-8 || error(
        "the two reference media used to expose the affine decomposition are " *
            "degenerate (determinant $det). Widen `_AFFINE_NU`."
    )
    z = Vector{Float64}(undef, 2nc)
    for i in 1:nc
        z[i] = (m_b * c_a[i] - m_a * c_b[i]) / det          # 𝕌ᴬ component
        z[i + nc] = (d_a * c_b[i] - d_b * c_a[i]) / det     # 𝕍ᴬ component
    end
    return z
end

# Elasticity: the sampled ν₀ at unit shear modulus — `ℙ` is homogeneous of
# degree −1 in the moduli, so the scale is arbitrary and fixing it is what makes
# the surrogate scale-exact.  Transport: unit conductivity, same argument.
function _reference_medium(
        class::Union{HillISO, HillTI, HillOrtho}, box::SampleBox, x_full
    )
    return _iso_ref(x_full[feature_index(box, :nu0)])
end

_reference_medium(::Union{HillISO2, HillTI2}, ::SampleBox, _x_full) = TensND.TensISO{3}(1.0)

# ─── Scaling fitted on the training set ──────────────────────────────────────

"""
    fit_scaling(train::Dataset; log_threshold = 30.0)
        -> (; x_shift, x_scale, y_kind, y_shift, y_scale)

Standardization of both ends, and the per-component output transform, read off
the **training** set only — the held-out set must not inform them.

A component is fitted on a `:log` scale when it is strictly positive throughout
and its dynamic range `max/min` exceeds `log_threshold`. That is the case of the
oblate Walpole components as the aspect ratio goes to zero, which span decades
and would otherwise monopolize a mean-squared loss.
"""
function fit_scaling(train::Dataset; log_threshold::Real = 30.0)
    X, Z = train.X, train.Z
    nsamples(train) ≥ 2 ||
        throw(ArgumentError("fitting a standardization needs at least 2 samples"))

    x_shift = vec(sum(X; dims = 2)) ./ size(X, 2)
    x_scale = map(i -> _spread(view(X, i, :)), axes(X, 1))

    nz = size(Z, 1)
    y_kind = Vector{Symbol}(undef, nz)
    Zt = similar(Z)
    for i in 1:nz
        row = view(Z, i, :)
        positive = all(>(0), row)
        rng = positive ? maximum(row) / minimum(row) : Inf
        y_kind[i] = (positive && rng > log_threshold) ? :log : :identity
        Zt[i, :] .= apply_transform.(y_kind[i], row)
    end
    y_shift = vec(sum(Zt; dims = 2)) ./ size(Zt, 2)
    y_scale = map(i -> _spread(view(Zt, i, :)), axes(Zt, 1))
    return (; x_shift, x_scale, y_kind, y_shift, y_scale)
end

# Standard deviation, floored: a feature or component that is constant over the
# training set would otherwise divide by zero.  The floor keeps the transform
# invertible and leaves the constant at zero after centering.
function _spread(v)
    m = sum(v) / length(v)
    s = sqrt(sum(x -> (x - m)^2, v) / max(length(v) - 1, 1))
    return s > 1.0e-12 ? s : 1.0
end

# ─── Validation ──────────────────────────────────────────────────────────────

"""
    validate_surrogate(s, data::Dataset)
        -> (; max_rel_error, rms_rel_error, max_block_error, worst)

Error of `s` over `data`, in the space the network predicts, at two granularities.

**`max_block_error` — the headline number.** For each sample, the ∞-norm of the
prediction error over *all* components, relative to the ∞-norm of the target:

    max_j ‖ẑⱼ − zⱼ‖_∞ / ‖zⱼ‖_∞ .

This is the quantity a test tolerance should be derived from, because it is the
one that propagates: the decoded tensor is a linear combination of the whole
component vector, so what matters is the error relative to the tensor's own
magnitude — not to that of its smallest entry. `worst` is an alias for it.

**`max_rel_error` / `rms_rel_error` — the per-component diagnostic.** Component
`i` on sample `j` is scored as

    |ẑᵢⱼ − zᵢⱼ| / max(|zᵢⱼ|, floorᵢ) .

The floor is the component's own RMS over the set, so a component that merely
passes through zero — the Walpole `ℓ₃` of a near-spherical inclusion — does not
report an unbounded error. When a component is **identically** zero over the
whole set the floor falls back to the global RMS instead, which turns a `0/0`
into the honest statement "this component is that fraction of the tensor's
magnitude". A structurally-vanishing component is not a rarity: `𝕍ᴬ` has no `ℓ₃`
at all, since the analytic kernel gives `p₃ = d·u₃` with no `1/μ₀` term.
"""
function validate_surrogate(s::NeuralSurrogate, data::Dataset)
    data.features == s.features || throw(
        ArgumentError(
            "this dataset carries features $(Tuple(data.features)) but the " *
                "surrogate consumes $(Tuple(s.features))"
        )
    )
    n = nsamples(data)
    n > 0 || throw(ArgumentError("cannot validate on an empty dataset"))
    nz = size(data.Z, 1)

    global_rms = sqrt(sum(abs2, data.Z) / max(length(data.Z), 1))
    global_rms > 0 ||
        throw(ArgumentError("every target in this dataset is zero — nothing to validate"))
    # A component whose RMS is negligible against the whole block is structurally
    # zero, not merely small; score it against the block instead of against itself.
    floors = map(1:nz) do i
        r = sqrt(sum(abs2, view(data.Z, i, :)) / n)
        r > 1.0e-12 * global_rms ? r : global_rms
    end

    maxe = zeros(Float64, nz)
    sqe = zeros(Float64, nz)
    block = 0.0
    for j in 1:n
        ẑ = predict_components(s, view(data.X, :, j))
        num = 0.0
        den = 0.0
        for i in 1:nz
            z = data.Z[i, j]
            d = abs(ẑ[i] - z)
            e = d / max(abs(z), floors[i])
            maxe[i] = max(maxe[i], e)
            sqe[i] += e * e
            num = max(num, d)
            den = max(den, abs(z))
        end
        block = max(block, num / max(den, global_rms))
    end
    rms = sqrt.(sqe ./ n)
    return (;
        max_rel_error = maxe, rms_rel_error = rms,
        max_block_error = block, worst = block,
    )
end
