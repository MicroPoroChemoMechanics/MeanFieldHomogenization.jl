# =============================================================================
#  neural_inclusion.jl — the two inclusion types, and how they meet the
#  contract of `docs/src/developer/adding_inclusion.md`.
#
#  `NeuralHillInclusion` enters through **gate A**: it supplies `hill_tensor`,
#  and the package derives the eight localization tensors, the four contribution
#  tensors and every scheme.  Two consequences worth stating, because they are
#  what makes gate A the right choice for a surrogate:
#
#  * the pivot `𝔸_εε = [𝕀 + ℙ:(ℂ₁−ℂ₀)]⁻¹` is evaluated exactly, so the limit
#    `ℂ₁ = ℂ₀ ⟹ 𝔸 = 𝕀` holds to machine precision no matter how badly the
#    network is trained, and the surrogate's error is confined to one tensor;
#  * the eight localization tensors stay exactly consistent with one another,
#    which a surrogate predicting `𝔸` directly could not guarantee.
#
#  `NeuralLocalizationInclusion` enters through **gate B** with both tensors, for
#  a heterogeneous morphology that has no Hill tensor at all — the shape a
#  surrogate trained on `fe_axi_localization` takes.
#
#  Both are ordinary `AbstractCustomInclusion`s: the neutral branch of the
#  hierarchy, so their methods can never become ambiguous with the ellipsoidal,
#  crack or layered families.
#
#  Sensitivity works, and that is the headline.  Every differentiable field is
#  `<:Number` or a tuple of them, so `Schemes._replace_geom_field` rebuilds the
#  struct by reflection and `derivative(rve, scheme, geometry(:phase, :field))`
#  reaches a *morphology* parameter.  Nothing is memoized and nothing is
#  converted to `Float64` on the way in, which is precisely why the
#  finite-element types have to refuse the same request.
# =============================================================================

"""
    NeuralShape

Neutral [`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait) of the neural
inclusions. No kernel dispatches on it — a surrogate supplies its own response —
but the contract requires *some* shape trait.
"""
struct NeuralShape end

"""
    _canonical_axes(axes, basis) -> (axes, basis)

Semi-axes sorted **descending** with the basis columns permuted to match — the
very convention `Ellipsoid` applies, through the very same helper.

Sharing it is not tidiness, it is correctness: a surrogate is trained on the
components the analytic teacher returns, and those are expressed in the sorted
frame. An inclusion that stored `(1, 1, 3)` where the teacher saw `(3, 1, 1)`
would feed the network a reciprocal aspect ratio and read the answer in the
wrong frame.

A 2-tuple or any non-3 dimension is returned untouched: the shipped surrogates
are three-dimensional, and there is no sorted convention to honour elsewhere.
"""
_canonical_axes(axes::NTuple{3}, basis) =
    Core._sort_axes_and_basis(axes, basis, :ellipsoid_3d)

_canonical_axes(axes::NTuple{N}, basis) where {N} = (axes, basis)

# ─── Raw features ────────────────────────────────────────────────────────────
#
#  Only the inclusion knows its own geometry, so building the feature vector is
#  its job; the surrogate merely declares which features it wants, by name.
#  Adding a feature is adding one method here — no change to the surrogate, the
#  dataset or the file format.

"""
    _feature(::Val{name}, incl, P₀) -> Number

One raw (unstandardized) feature of `incl` under reference medium `P₀`.

| Name | Meaning | Range |
|---|---|---|
| `:log_aspect` | `log(distinct axis / equal axes)` of a spheroid | `> 0` prolate, `< 0` oblate, `0` sphere |
| `:log_r2` | `log(a₂/a₁)` of a sorted ellipsoid | `≤ 0` |
| `:log_r32` | `log(a₃/a₂)` of a sorted ellipsoid | `≤ 0` |
| `:nu0` | Poisson ratio of the isotropic reference medium | |

Three conventions here are load-bearing, and each of them is a bug if broken.

**The logarithm** is not cosmetic: an aspect ratio's interesting range spans
decades, and `ω` and `1/ω` are the same amount of anisotropy, which only the
logarithm makes symmetric.

**`:log_aspect` is measured on the *distinct* axis**, not on a fixed slot.
Semi-axes are stored sorted descending (as `Ellipsoid` does), so a prolate
spheroid is `(ω, 1, 1)` and an oblate one `(1, 1, ω)`: `log(a₃/a₁)` would be
negative for both and conflate the two families. Distinct-over-equal is instead
a bijection onto the whole real line, with the sphere at the origin.

**The triaxial pair is `(a₂/a₁, a₃/a₂)`, not `(a₂/a₁, a₃/a₁)`.** With the
sorted convention `a₁ ≥ a₂ ≥ a₃`, the admissible set of the first pair is
exactly the box `log_r2 ≤ 0, log_r32 ≤ 0`, whereas the second pair is confined
to a triangular wedge that no `SampleBox` can express.
"""
_feature(::Val{:log_aspect}, incl, _P₀) = log(_spheroid_ratio(_axes(incl)))
_feature(::Val{:log_r2}, incl, _P₀) = (a = _axes(incl); log(a[2] / a[1]))
_feature(::Val{:log_r32}, incl, _P₀) = (a = _axes(incl); log(a[3] / a[2]))

function _feature(::Val{:nu0}, _incl, P₀::TensND.TensISO{4, 3})
    _, nu = Elasticity.E_nu(P₀)
    return nu
end

_feature(::Val{:nu0}, _incl, P₀::TensND.AbstractTens) = throw(
    ArgumentError(
        "the :nu0 feature needs an isotropic reference medium and got a " *
            "$(nameof(typeof(P₀))). Under `IsoSymmetrize` the scheme pre-projects " *
            "the reference medium, which is one way to satisfy this; otherwise the " *
            "surrogate has to be trained on a feature set that describes an " *
            "anisotropic matrix."
    )
)

# Contrast ratios, on the shear modulus.  A heterogeneous morphology carries its
# constituents inside itself, so `𝔸` depends on `ℂ₀` only through the *contrast*:
# scaling the reference medium and every constituent together leaves `𝔸_εε`
# unchanged and multiplies `𝔸_σε` by the factor. Ratios are therefore the complete
# and minimal material parametrization — and the reason a gate-B surrogate cannot
# reuse the `ℙ(λℂ₀) = ℙ(ℂ₀)/λ` homogeneity of gate A.
for k in 1:4
    @eval function _feature(::Val{$(QuoteNode(Symbol(:log_mu_ratio_, k)))}, incl, P₀)
        return log(_constituent_mu(incl, $k) / _matrix_mu(P₀))
    end
end

_matrix_mu(C₀::TensND.TensISO{4, 3}) = Elasticity.k_mu(C₀)[2]

_matrix_mu(P₀::TensND.AbstractTens) = throw(
    ArgumentError(
        "a contrast feature needs an isotropic reference medium and got a " *
            "$(nameof(typeof(P₀)))"
    )
)

function _constituent_mu(incl, k::Int)
    props = _constituents(incl)
    props === nothing && throw(
        ArgumentError(
            "this inclusion declares no constituents, so it has no contrast " *
                "feature. Pass `properties = (ℂ₁, ℂ₂, …)` at construction."
        )
    )
    k ≤ length(props) || throw(
        ArgumentError(
            "feature :log_mu_ratio_$k asks for constituent $k but the inclusion " *
                "carries $(length(props))"
        )
    )
    return Elasticity.k_mu(props[k])[2]
end

_constituents(_incl) = nothing

# A morphology parameter the surrogate names — `:eccentricity`, `:core_fraction`,
# anything the type chose to expose. Looked up before giving up, so that adding a
# parameter is a constructor argument rather than a new method.
_shape_param(_incl, ::Symbol) = nothing

function _feature(::Val{name}, incl, _P₀) where {name}
    v = _shape_param(incl, name)
    v === nothing && throw(
        ArgumentError(
            "unknown surrogate feature :$name — either name it in the inclusion's " *
                "`shape_params`, or add a `_feature(::Val{:$name}, incl, P₀)` method " *
                "for the geometry that defines it"
        )
    )
    return v
end

"""
    raw_features(incl, s::NeuralSurrogate, P₀) -> Vector

The feature vector `s` expects, read off `incl` and `P₀`. Type-generic: a
`ForwardDiff.Dual` semi-axis yields a `Dual` feature vector, and the whole chain
downstream follows.
"""
function raw_features(incl, s::NeuralSurrogate, P₀::TensND.AbstractTens)
    return [_feature(Val(name), incl, P₀) for name in s.features]
end

# The frame conventions (`_class_frame`, `_axes`, `_spheroid_axis_index`) live in
# `specs.jl`, next to the classes: the labeler and the inclusion must derive the
# frame identically, so there is a single definition serving both.

# ─── Gate A — NeuralHillInclusion ────────────────────────────────────────────

"""
    NeuralHillInclusion(semi_axes; elastic = nothing, transport = nothing,
                        basis = nothing, euler_angles = (), guard = :warn)

Inclusion whose **Hill tensor** is produced by a trained
[`NeuralSurrogate`](@ref) — entry gate A of the inclusion contract, so all eight
localization tensors, all four contribution tensors and every homogenization
scheme follow.

| Option | Meaning |
|---|---|
| `semi_axes` | the geometry, as a tuple; also the differentiable field |
| `elastic` | order-4 surrogate, for `hill_tensor(incl, C₀)` |
| `transport` | order-2 surrogate, for `hill_tensor(incl, K₀)` |
| `basis` / `euler_angles` | the local frame, as everywhere else in the package |
| `guard` | `:warn` (default), `:error` or `:none` — what to do when a feature falls outside the box the surrogate was trained on |

At least one surrogate is required, and each must match its tensor order. A
surrogate is checked against the geometry at construction: a `HillTI` model
needs a spheroid, a `HillISO` one a sphere, so a mismatch is a constructor error
rather than a wrongly shaped tensor much later.

# Example

```julia
s = load_surrogate(NeuralInclusions.model_path("spheroid_hill_iso_elastic"))
incl = NeuralHillInclusion((1.0, 1.0, 0.2); elastic = s)

rve = RVE()
add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => iso_stiffness(20.0, 12.0)); fraction = :rest)
add_phase!(rve, :i, incl, Dict(:C => iso_stiffness(60.0, 30.0)); fraction = 0.2)
homogenize(rve, MoriTanaka(), :C)
```

See also [`NeuralLocalizationInclusion`](@ref), and
`docs/src/manual/neural_inclusions.md`.
"""
struct NeuralHillInclusion{dim, T <: Number, B <: TensND.AbstractBasis, S4, S2} <:
    Core.AbstractCustomInclusion{T}
    semi_axes::NTuple{dim, T}
    basis::B
    elastic::S4
    transport::S2
    guard::Symbol
end

function NeuralHillInclusion(
        semi_axes::Tuple{Vararg{Number}};
        elastic::Union{Nothing, NeuralSurrogate} = nothing,
        transport::Union{Nothing, NeuralSurrogate} = nothing,
        basis::Union{Nothing, TensND.AbstractBasis} = nothing,
        euler_angles::Tuple{Vararg{Real}} = (),
        guard::Symbol = :warn,
    )
    elastic === nothing && transport === nothing && throw(
        ArgumentError(
            "a `NeuralHillInclusion` needs at least one surrogate: `elastic` for " *
                "elasticity (order 4), `transport` for conduction (order 2)"
        )
    )
    _check_order(elastic, 4, :elastic)
    _check_order(transport, 2, :transport)
    guard in (:warn, :error, :none) ||
        throw(ArgumentError("`guard` must be :warn, :error or :none, got :$guard"))
    T = Core._floatlike(promote_type(typeof.(semi_axes)...))
    b0 = basis === nothing ? Core._default_basis(T, euler_angles) : basis
    axes_, b = _canonical_axes(map(T, semi_axes), b0)
    for s in (elastic, transport)
        s === nothing || _check_geometry(s, axes_)
    end
    return NeuralHillInclusion{length(axes_), T, typeof(b), typeof(elastic), typeof(transport)}(
        axes_, b, elastic, transport, guard
    )
end

NeuralHillInclusion(semi_axes::Number...; kw...) =
    NeuralHillInclusion(promote(semi_axes...); kw...)

function _check_order(s::NeuralSurrogate, order::Int, which::Symbol)
    tensor_order(s) == order || throw(
        ArgumentError(
            "the `$which` surrogate predicts an order-$(tensor_order(s)) tensor " *
                "(class :$(class_name(hill_class(s)))) but the `$which` slot serves " *
                "order-$order physics. Elasticity is order 4, conduction order 2."
        )
    )
    return nothing
end

_check_order(::Nothing, _order, _which) = nothing

# A surrogate's class encodes how many components it emits, which is only
# meaningful for the geometry it was trained on.  Refusing the mismatch here
# turns a silently wrong tensor into a constructor error.
function _check_geometry(s::NeuralSurrogate, axes_::NTuple{3})
    a1, a2, a3 = axes_
    class = hill_class(s)
    sphere = _axes_equal(a1, a2) && _axes_equal(a2, a3)
    # Axes are sorted descending, so a spheroid has either its first or its last
    # entry distinct.
    spheroid = !sphere && (_axes_equal(a2, a3) || _axes_equal(a1, a2))
    ok = if class isa Union{HillISO, HillISO2}
        sphere
    elseif class isa Union{HillTI, HillTI2}
        spheroid
    else
        !sphere && !spheroid
    end
    ok || throw(
        ArgumentError(
            "a surrogate of class :$(class_name(class)) does not describe the " *
                "geometry `semi_axes = $(axes_)`: :iso/:iso2 needs a sphere, " *
                ":ti/:ti2 a spheroid (exactly two equal semi-axes), :ortho a " *
                "genuinely triaxial ellipsoid. This is not pedantry — the analytic " *
                "teacher returns a different tensor class for each of the three, so " *
                "the surrogate's components would mean something else. Use the " *
                "surrogate trained for this shape."
        )
    )
    return nothing
end

_check_geometry(_s, _axes) = nothing

# ── Level 0 ──────────────────────────────────────────────────────────────────

Core.dimension(::NeuralHillInclusion{dim}) where {dim} = dim
Core.inclusion_basis(i::NeuralHillInclusion) = i.basis
Core.shape_trait(::NeuralHillInclusion) = NeuralShape
Core.is_homogeneous_inclusion(::NeuralHillInclusion) = true

function Core.shape_tensor(i::NeuralHillInclusion{dim, T}) where {dim, T}
    D = zeros(T, dim, dim)
    @inbounds for k in 1:dim
        D[k, k] = i.semi_axes[k]
    end
    return TensND.Tens(D, i.basis)
end

# ── Level 1, gate A ──────────────────────────────────────────────────────────
#
#  One method per tensor order.  A single method typed on the union
#  `AbstractTens` would be ambiguous with both of the generics, which are
#  themselves declared per order — see the warning in `adding_inclusion.md`.

"""
    hill_tensor(incl::NeuralHillInclusion, P₀; kw...) -> AbstractTens

Hill tensor evaluated by the surrogate: features off the geometry and `P₀`, one
forward pass, then an exact decode into the structured tensor of the
surrogate's class, in the global frame.

Keyword arguments of the analytic entry points (`method`, tolerances) are
accepted and ignored — there is no quadrature to steer — so a scheme that
forwards them works unchanged.
"""
Elasticity.hill_tensor(i::NeuralHillInclusion, C₀::TensND.AbstractTens{4, 3}; kw...) =
    _neural_hill(i, i.elastic, C₀, :elastic)

Elasticity.hill_tensor(i::NeuralHillInclusion, K₀::TensND.AbstractTens{2, 3}; kw...) =
    _neural_hill(i, i.transport, K₀, :transport)

function _neural_hill(incl, s::NeuralSurrogate, P₀, _which)
    x = raw_features(incl, s, P₀)
    return s(x, P₀, _class_frame(hill_class(s), incl); guard = incl.guard)
end

_neural_hill(_incl, ::Nothing, _P₀, which) = throw(
    ArgumentError(
        "this `NeuralHillInclusion` carries no `$which` surrogate, so it cannot " *
            "serve $(which === :elastic ? "elasticity" : "conduction"). Build one " *
            "object per physics, or pass both surrogates at construction."
    )
)

# ─── Gate B — NeuralLocalizationInclusion ────────────────────────────────────

"""
    NeuralLocalizationInclusion(semi_axes; strain = nothing, stress = nothing,
                                gradient = nothing, flux = nothing,
                                fractions = nothing, properties = nothing,
                                basis = nothing, euler_angles = (), guard = :warn)

Inclusion whose **localization tensors** are produced by trained surrogates —
entry gate B, the only way in for an internally heterogeneous morphology, which
has no Hill tensor at all. This is the shape a surrogate trained on
[`fe_axi_localization`](@ref MeanFieldHomogenization.FiniteElements.fe_axi_localization)
takes.

Because [`is_homogeneous_inclusion`](@ref MeanFieldHomogenization.Core.is_homogeneous_inclusion) is `false`, gate B costs **two**
tensors per physics: the strain side *and* the stress side, since
`𝔸_σε = ℂ₁:𝔸_εε` presupposes a single uniform `ℂ₁` that a heterogeneous
inclusion does not have. Given both, the generic contributions switch to the
exact `ℕ = 𝔸_σε − ℂ₀:𝔸_εε` and `ℍ = (𝔸_εε − 𝕊₀:𝔸_σε):𝕊₀`, so gate B is a
complete entry point.

| Option | Meaning |
|---|---|
| `strain` / `stress` | the order-4 pair, `𝔸_εε` and `𝔸_σε` |
| `gradient` / `flux` | the order-2 pair, `𝔸_∇∇` and `𝔸_q∇` |
| `shape_params` | named morphology parameters, e.g. `(; eccentricity = 0.4, core_fraction = 0.5)`. They are the surrogate's features *and* the fields the sensitivity API differentiates, so every value must be a `Number` |
| `fractions` / `properties` | internal volume fractions and constituent properties; supplying them unlocks the `Voigt` and `Reuss` bounds, which a heterogeneous inclusion cannot otherwise serve. `properties` also backs the `:log_mu_ratio_k` contrast features |

A gate-B surrogate's features are the morphology parameters and the **contrast
ratios** — never absolute moduli. The reason is that the constituents live inside
the inclusion, so scaling `ℂ₀` alone changes the contrast: the exact invariance is
under a *simultaneous* scaling of the reference medium and of every constituent,
which leaves `𝔸_εε` unchanged and multiplies `𝔸_σε` by the factor. The
`ℙ(λℂ₀) = ℙ(ℂ₀)/λ` homogeneity that gate A exploits does not transfer.

Supplying only one tensor of a pair is refused at construction: the omission is
silent otherwise — `Dilute` and `MoriTanaka` stay right while `SelfConsistent`
and `AsymmetricSelfConsistent`, the schemes that consume the stress average,
drift by several percent.

!!! note "Not exercised by the shipped models"
    The pilot trains gate-A surrogates, where the contrast dependence is exact.
    This type is the seam for the heterogeneous morphologies, and it is tested
    against an equivalent gate-A inclusion; no trained model ships for it yet.
"""
struct NeuralLocalizationInclusion{
        dim, T <: Number, TS <: Number, B <: TensND.AbstractBasis, S4, S2, NS, F, P,
    } <: Core.AbstractCustomInclusion{T}
    semi_axes::NTuple{dim, T}
    basis::B
    elastic::S4                   # (strain, stress) or nothing
    transport::S2                 # (gradient, flux) or nothing
    ## `TS` is deliberately *not* `T`: the sensitivity API perturbs one field at a
    ## time, so differentiating a morphology parameter hands the constructor
    ## `Dual` shape parameters beside `Float64` semi-axes. Sharing one type
    ## parameter would make that combination unconstructible — and the failure
    ## would be a `MethodError` from inside `_replace_geom_field`, far from here.
    shape_params::NTuple{NS, TS}  # differentiable morphology parameters
    shape_names::NTuple{NS, Symbol}
    fractions::F
    properties::P
    guard::Symbol
end

function NeuralLocalizationInclusion(
        semi_axes::Tuple{Vararg{Number}};
        strain::Union{Nothing, NeuralSurrogate} = nothing,
        stress::Union{Nothing, NeuralSurrogate} = nothing,
        gradient::Union{Nothing, NeuralSurrogate} = nothing,
        flux::Union{Nothing, NeuralSurrogate} = nothing,
        shape_params::NamedTuple = NamedTuple(),
        fractions::Union{Nothing, Tuple{Vararg{Number}}} = nothing,
        properties::Union{Nothing, Tuple} = nothing,
        basis::Union{Nothing, TensND.AbstractBasis} = nothing,
        euler_angles::Tuple{Vararg{Real}} = (),
        guard::Symbol = :warn,
    )
    _check_pair(strain, stress, :strain, :stress, 4)
    _check_pair(gradient, flux, :gradient, :flux, 2)
    strain === nothing && gradient === nothing && throw(
        ArgumentError(
            "a `NeuralLocalizationInclusion` needs at least one pair of " *
                "surrogates: (`strain`, `stress`) for elasticity, (`gradient`, " *
                "`flux`) for conduction"
        )
    )
    guard in (:warn, :error, :none) ||
        throw(ArgumentError("`guard` must be :warn, :error or :none, got :$guard"))
    if (fractions === nothing) != (properties === nothing)
        throw(
            ArgumentError(
                "`fractions` and `properties` go together — a bound averages the " *
                    "constituent properties over the inclusion, so it needs both"
            )
        )
    end
    if fractions !== nothing
        length(fractions) == length(properties) || throw(
            DimensionMismatch(
                "$(length(fractions)) internal fractions but " *
                    "$(length(properties)) constituent properties"
            )
        )
        isapprox(sum(fractions), 1; atol = 1.0e-10) || throw(
            ArgumentError("the internal volume fractions must sum to 1, got $(sum(fractions))")
        )
    end
    all(v -> v isa Number, values(shape_params)) || throw(
        ArgumentError(
            "every `shape_params` value must be a `Number`: they are the fields the " *
                "sensitivity API differentiates, and `Schemes._replace_geom_field` " *
                "rebuilds the struct by reflection over them"
        )
    )
    T = Core._floatlike(promote_type(typeof.(semi_axes)...))
    b0 = basis === nothing ? Core._default_basis(T, euler_angles) : basis
    axes_, b = _canonical_axes(map(T, semi_axes), b0)
    el = strain === nothing ? nothing : (strain, stress)
    tr = gradient === nothing ? nothing : (gradient, flux)
    vals = values(shape_params)
    TS = isempty(vals) ? T : Core._floatlike(promote_type(typeof.(vals)...))
    sp = map(TS, vals)
    sn = keys(shape_params)
    return NeuralLocalizationInclusion{
        length(axes_), T, TS, typeof(b), typeof(el), typeof(tr), length(sp),
        typeof(fractions), typeof(properties),
    }(axes_, b, el, tr, sp, sn, fractions, properties, guard)
end

_constituents(i::NeuralLocalizationInclusion) = i.properties

function _shape_param(i::NeuralLocalizationInclusion, name::Symbol)
    k = findfirst(==(name), i.shape_names)
    return k === nothing ? nothing : i.shape_params[k]
end

NeuralLocalizationInclusion(semi_axes::Number...; kw...) =
    NeuralLocalizationInclusion(promote(semi_axes...); kw...)

function _check_pair(a, b, na::Symbol, nb::Symbol, order::Int)
    if (a === nothing) != (b === nothing)
        missing_, present = a === nothing ? (na, nb) : (nb, na)
        throw(
            ArgumentError(
                "gate B costs two tensors for a heterogeneous inclusion: `$present` " *
                    "was given but `$missing_` was not. The strain side fixes the " *
                    "stress side only through `𝔸_σε = ℂ₁:𝔸_εε`, which needs a single " *
                    "uniform property this inclusion does not have — so omitting it " *
                    "would leave `Dilute` and `MoriTanaka` correct while " *
                    "`SelfConsistent` and `AsymmetricSelfConsistent` drift silently."
            )
        )
    end
    a === nothing && return nothing
    _check_order(a, order, na)
    _check_order(b, order, nb)
    return nothing
end

# ── Level 0 ──────────────────────────────────────────────────────────────────

Core.dimension(::NeuralLocalizationInclusion{dim}) where {dim} = dim
Core.inclusion_basis(i::NeuralLocalizationInclusion) = i.basis
Core.shape_trait(::NeuralLocalizationInclusion) = NeuralShape
Core.is_homogeneous_inclusion(::NeuralLocalizationInclusion) = false

function Core.shape_tensor(i::NeuralLocalizationInclusion{dim, T}) where {dim, T}
    D = zeros(T, dim, dim)
    @inbounds for k in 1:dim
        D[k, k] = i.semi_axes[k]
    end
    return TensND.Tens(D, i.basis)
end

# ── Level 1, gate B ──────────────────────────────────────────────────────────
#
#  As for `LayeredSphere` and `FEExcenteredSphere`, the three-argument scheme
#  signature carries a phase property this inclusion has no use for — its
#  constituents are part of the morphology it was trained on — so the middle
#  argument is accepted and ignored.

for (gen, order, slot, idx, name) in (
        (:(Core.strain_strain_loc), 4, :elastic, 1, :strain),
        (:(Core.stress_strain_loc), 4, :elastic, 2, :stress),
        (:(Core.gradient_gradient_loc), 2, :transport, 1, :gradient),
        (:(Core.flux_gradient_loc), 2, :transport, 2, :flux),
    )
    @eval function $gen(
            i::NeuralLocalizationInclusion,
            ::TensND.AbstractTens{$order, 3},
            P₀::TensND.AbstractTens{$order, 3};
            kw...
        )
        pair = i.$slot
        pair === nothing && throw(
            ArgumentError(
                "this `NeuralLocalizationInclusion` carries no $($(QuoteNode(slot))) " *
                    "surrogates, so it cannot serve `$($(QuoteNode(name)))`"
            )
        )
        s = pair[$idx]
        return s(
            raw_features(i, s, P₀), P₀, _class_frame(hill_class(s), i);
            guard = i.guard
        )
    end
end

# ── Bounds ───────────────────────────────────────────────────────────────────
#
#  A heterogeneous inclusion has no single property for `Voigt`/`Reuss` to read
#  off the RVE, and the internal fractions are not something the RVE carries.
#  When the morphology knows them, the averages are exact and free.

Schemes.has_layer_average(i::NeuralLocalizationInclusion) = i.fractions !== nothing

function _layer_average(i::NeuralLocalizationInclusion, ref::TensND.AbstractTens{O, 3}, f) where {O}
    i.fractions === nothing && throw(
        ArgumentError(
            "this `NeuralLocalizationInclusion` was built without `fractions` and " *
                "`properties`, so the Voigt and Reuss bounds are unavailable for it. " *
                "Every scheme that consumes localization or contribution tensors is " *
                "unaffected, `AsymmetricSelfConsistent` included."
        )
    )
    order = tensor_order(first(i.properties))
    order == O || throw(
        ArgumentError(
            "this inclusion carries order-$order constituents, so it cannot serve " *
                "an order-$O bound"
        )
    )
    return sum(i.fractions[k] * f(i.properties[k]) for k in eachindex(i.fractions))
end

Schemes._layer_voigt(i::NeuralLocalizationInclusion, ref::TensND.AbstractTens) =
    _layer_average(i, ref, identity)
Schemes._layer_reuss(i::NeuralLocalizationInclusion, ref::TensND.AbstractTens) =
    _layer_average(i, ref, inv)
