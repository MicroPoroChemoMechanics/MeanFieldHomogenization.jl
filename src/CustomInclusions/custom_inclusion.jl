# =============================================================================
#  custom_inclusion.jl
#
#  `CustomInclusion` — a ready-made concrete `Core.AbstractCustomInclusion`
#  driven by user callbacks, plus `check_inclusion_interface`, a conformance
#  checker usable on *any* inclusion type.
#
#  This is the value-level counterpart of subtyping: it lets a user plug a
#  morphology into every homogenization scheme without declaring a type and a
#  method table.  It is the direct analog of the `user_inclusion` class of
#  the C++/Python `echoes` codebase, where the user overrides `build_all()`
#  to return the concentration tensors by any means.
#
#  Loaded by the `CustomInclusions` sub-module, itself included after
#  `localization.jl` and `contribution.jl`, because the fallbacks below
#  `invoke` the generic methods defined there.
# =============================================================================

"""
    CustomShape

Neutral [`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait) tag carried by
[`CustomInclusion`](@ref).  No kernel dispatches on it — a custom inclusion
supplies its own response — but the interface requires *some* shape trait.
"""
struct CustomShape end

const _CUSTOM_CALLBACKS = (
    :hill_tensor,
    :strain_strain_loc,
    :stress_strain_loc,
    :gradient_gradient_loc,
    :flux_gradient_loc,
    :stiffness_contribution,
    :compliance_contribution,
    :conductivity_contribution,
    :resistivity_contribution,
)

"""
    CustomInclusion{dim,T,B,NT} <: AbstractCustomInclusion{T}

Inclusion whose mechanical / transport response is supplied by user
callbacks, so that an arbitrary morphology becomes a first-class citizen of
every [`homogenize`](@ref MeanFieldHomogenization.Core.homogenize) scheme without
defining a new type.

# Construction

    CustomInclusion(; dim, semi_axes, basis, euler_angles, homogeneous,
                    density_factor, <callbacks>...)
    CustomInclusion(semi_axes; kw...)          # convenience

| Option | Default | Meaning |
|---|---|---|
| `semi_axes` | `nothing` | half-dimensions of an **equivalent ellipsoidal envelope**, *if* your morphology has one. Purely descriptive: no kernel reads it, and it only serves [`shape_tensor`](@ref MeanFieldHomogenization.Core.shape_tensor). Leave it out for a shape that has no such envelope. |
| `dim` | `3` | spatial dimension, when `semi_axes` is not given |
| `basis` | from `euler_angles` | local frame; for a flat object column 3 is the normal |
| `euler_angles` | `()` | ZYZ angles, ignored when `basis` is given |
| `homogeneous` | `true` | value returned by [`is_homogeneous_inclusion`](@ref MeanFieldHomogenization.Core.is_homogeneous_inclusion) |
| `density_factor` | `nothing` | prefactor of the *amount × contribution* seam; set it (e.g. `4π/3`) to register the phase with a [`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity) amount |

# Callbacks — pick **one** entry gate

Every callback is a keyword argument named exactly after the generic it
implements, and receives the tensor arguments only (the geometry is captured
by the closure).  **All callbacks must accept trailing keyword arguments**:
the schemes forward `method`, tolerances, and — for density-based phases —
`K_interface` / `α_interface`.

| Gate | Callbacks | Signature |
|---|---|---|
| **A — Hill tensor** | `hill_tensor` | `(P₀; kw...) -> Tens` |
| **B — localization** | `strain_strain_loc` (+ `stress_strain_loc` if heterogeneous), `gradient_gradient_loc` (+ `flux_gradient_loc`) | `(P₁, P₀; kw...) -> Tens` |
| **C — contribution, solid** | `stiffness_contribution`, `compliance_contribution`, `conductivity_contribution`, `resistivity_contribution` | `(P₁, P₀; kw...) -> Tens` |
| **C — contribution, flat** | same names, with `density_factor` set | `(P₀; kw...) -> Tens` |

Gate A yields all eight localization and all four contribution tensors for
free; gate B yields the derived localizations and the contributions.
`P₀` is a 4th-order stiffness in elasticity and a 2nd-order conductivity in
transport — a single callback may serve both by dispatching on its argument.

!!! warning "Gate B and heterogeneous inclusions"
    The strain-side localization determines the stress-side one *only* through
    `A_σε = C₁ : A_εε`, which needs a single uniform `C₁`. If you set
    `homogeneous = false`, supply `stress_strain_loc` (and `flux_gradient_loc`
    in transport) as well — otherwise the average stress in the inclusion is
    silently wrong, and with it `SelfConsistent` and
    `AsymmetricSelfConsistent`, the two schemes that consume it. `Dilute` and
    `MoriTanaka` need only `(A, N)` and stay correct, which is precisely what
    makes the omission easy to miss.

    A heterogeneous inclusion also has no single property to feed the `Voigt`
    and `Reuss` bounds — they need its internal volume fractions — so those are
    unavailable unless the type supplies a layer-wise average
    (`Schemes._layer_voigt` / `Schemes._layer_reuss`, plus
    `Schemes.has_layer_average`). Every scheme that consumes localization or
    contribution tensors is unaffected, `AsymmetricSelfConsistent` included.

!!! warning "Gate C and the dilute concentration tensor"
    Contribution tensors alone do not determine `A`.  A **volume-fraction**
    phase entered through gate C therefore works with `Voigt`, `Reuss`,
    `Dilute`, `DiluteDual`, `Maxwell`, `PonteCastanedaWillis` and
    `DifferentialScheme`, but **not** with `MoriTanaka`, `SelfConsistent` or
    `AsymmetricSelfConsistent`, whose kernels also need `A`.  Density-based
    (flat) phases are unaffected: those kernels reconstruct `A` from `ℍ:C₀`.

Orientation averaging (`IsoSymmetrize` / `TISymmetrize`) is applied by the
scheme *after* the callback returns, so the returned tensors must be
expressed in the **global** frame.

# Example — a sphere, through gate A

```julia
sphere = CustomInclusion((1.0, 1.0, 1.0);
    hill_tensor = (C₀; kw...) -> hill_tensor(Ellipsoid(1.0, 1.0, 1.0), C₀; kw...))

rve = RVE()
add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => iso_stiffness(10.0, 6.0)); fraction = :rest)
add_phase!(rve, :i, sphere, Dict(:C => iso_stiffness(1.0, 0.5)); fraction = 0.2)
homogenize(rve, MoriTanaka(), :C)
```

See also [`check_inclusion_interface`](@ref) and the developer page
*Adding a new inclusion* for the full contract.
"""
struct CustomInclusion{dim, T <: Number, B <: TensND.AbstractBasis, NT <: NamedTuple} <:
    Core.AbstractCustomInclusion{T}
    semi_axes::Union{Nothing, NTuple{dim, T}}
    basis::B
    homogeneous::Bool
    density_factor::Union{Nothing, Float64}
    fns::NT
end

function CustomInclusion(
        semi_axes::Union{Nothing, NTuple{<:Any, <:Number}} = nothing;
        dim::Int = semi_axes === nothing ? 3 : length(semi_axes),
        basis::Union{Nothing, TensND.AbstractBasis} = nothing,
        euler_angles::Tuple{Vararg{Real}} = (),
        homogeneous::Bool = true,
        density_factor::Union{Nothing, Real} = nothing,
        callbacks...
    )
    fns = NamedTuple(callbacks)
    unknown = setdiff(keys(fns), _CUSTOM_CALLBACKS)
    isempty(unknown) || throw(
        ArgumentError(
            "unknown CustomInclusion callback(s) $(Tuple(unknown)); " *
                "expected any of $(_CUSTOM_CALLBACKS)"
        )
    )
    isempty(fns) && throw(
        ArgumentError(
            "a CustomInclusion needs at least one callback — see " *
                "`check_inclusion_interface` and the entry-gate table in the docstring"
        )
    )
    semi_axes !== nothing && length(semi_axes) != dim && throw(
        ArgumentError(
            "`dim` = $dim contradicts the $(length(semi_axes)) semi-axes given"
        )
    )
    T = semi_axes === nothing ? Float64 :
        Core._floatlike(promote_type(typeof.(semi_axes)...))
    b = basis === nothing ? Core._default_basis(T, euler_angles) : basis
    axes = semi_axes === nothing ? nothing : map(T, semi_axes)
    return CustomInclusion{dim, T, typeof(b), typeof(fns)}(
        axes, b, homogeneous,
        density_factor === nothing ? nothing : Float64(density_factor), fns
    )
end

CustomInclusion(semi_axes::Number...; kw...) = CustomInclusion(promote(semi_axes...); kw...)

# ─── Level 0 — geometric identity ────────────────────────────────────────────

Core.dimension(::CustomInclusion{dim}) where {dim} = dim
Core.inclusion_basis(c::CustomInclusion) = c.basis
Core.shape_trait(::CustomInclusion) = CustomShape
Core.is_homogeneous_inclusion(c::CustomInclusion) = c.homogeneous

# `shape_tensor` describes an *equivalent ellipsoidal envelope*.  Nothing in the
# package consumes it — a custom inclusion supplies its response directly — so
# it is optional, and a morphology with no such envelope simply does not have
# one.
function Core.shape_tensor(c::CustomInclusion{dim, T}) where {dim, T}
    c.semi_axes === nothing && throw(
        ArgumentError(
            "this CustomInclusion declares no equivalent ellipsoidal envelope, " *
                "so it has no `shape_tensor`. Pass `semi_axes` at construction if " *
                "your morphology has one — nothing in the package requires it."
        )
    )
    D = zeros(T, dim, dim)
    @inbounds for i in 1:dim
        D[i, i] = c.semi_axes[i]
    end
    return TensND.Tens(D, c.basis)
end

# ─── Callback plumbing ───────────────────────────────────────────────────────
#
#  `_has` is a compile-time constant (it reads the NamedTuple *type*), so the
#  branches below fold away and the fallbacks cost nothing.

_has(::CustomInclusion{dim, T, B, NT}, ::Val{k}) where {dim, T, B, NT, k} = hasfield(NT, k)

function _no_callback(c::CustomInclusion, name::Symbol)
    return throw(
        ArgumentError(
            "this CustomInclusion has no `$name` callback (it provides " *
                "$(Tuple(keys(c.fns)))). Supply it at construction, or use a " *
                "different entry gate — see `check_inclusion_interface(incl)`."
        )
    )
end

# ─── Gate A — Hill tensor ────────────────────────────────────────────────────

function Elasticity.hill_tensor(c::CustomInclusion, P₀::TensND.AbstractTens; kw...)
    _has(c, Val(:hill_tensor)) || _no_callback(c, :hill_tensor)
    return c.fns.hill_tensor(P₀; kw...)
end

# ─── Gate B — localization ───────────────────────────────────────────────────

#  Both sides of the pair are forwarded.  For a homogeneous inclusion the
#  stress-side one may be omitted — the generic `A_σε = C₁ : A_εε` fallback is
#  then exact.  For a heterogeneous one it must be supplied, since no single
#  `C₁` represents the inclusion (`is_homogeneous_inclusion`).
for (fname, gen, order) in (
        (:strain_strain_loc, :(Core.strain_strain_loc), 4),
        (:stress_strain_loc, :(Core.stress_strain_loc), 4),
        (:gradient_gradient_loc, :(Core.gradient_gradient_loc), 2),
        (:flux_gradient_loc, :(Core.flux_gradient_loc), 2),
    )
    @eval function $gen(
            c::CustomInclusion,
            P₁::TensND.AbstractTens{$order, 3},
            P₀::TensND.AbstractTens{$order, 3};
            kw...
        )
        _has(c, Val($(QuoteNode(fname)))) && return c.fns.$fname(P₁, P₀; kw...)
        return invoke(
            $gen,
            Tuple{
                Core.AbstractInclusion,
                TensND.AbstractTens{$order, 3},
                TensND.AbstractTens{$order, 3},
            },
            c, P₁, P₀; kw...
        )
    end
end

# ─── Gate C — contribution tensors ───────────────────────────────────────────
#
#  Three-argument (solid) forms fall back to the generic algebra of
#  `contribution.jl`; two-argument (flat) forms have no generic fallback.

#  Note `compliance_contribution` appears at *both* orders: for a flat object
#  the 2nd-order form is the resistivity-like contribution 𝐑 (same convention
#  as `Cracks`), so a single user callback must dispatch on its argument.
for (fname, gen, order) in (
        (:stiffness_contribution, :(Core.stiffness_contribution), 4),
        (:compliance_contribution, :(Core.compliance_contribution), 4),
        (:compliance_contribution, :(Core.compliance_contribution), 2),
        (:conductivity_contribution, :(Core.conductivity_contribution), 2),
        (:resistivity_contribution, :(Core.resistivity_contribution), 2),
    )
    @eval function $gen(
            c::CustomInclusion,
            P₁::TensND.AbstractTens{$order, 3},
            P₀::TensND.AbstractTens{$order, 3};
            kw...
        )
        _has(c, Val($(QuoteNode(fname)))) && return c.fns.$fname(P₁, P₀; kw...)
        return invoke(
            $gen,
            Tuple{
                Core.AbstractInclusion,
                TensND.AbstractTens{$order, 3},
                TensND.AbstractTens{$order, 3},
            },
            c, P₁, P₀; kw...
        )
    end

    @eval function $gen(c::CustomInclusion, P₀::TensND.AbstractTens{$order, 3}; kw...)
        _has(c, Val($(QuoteNode(fname)))) || _no_callback(c, $(QuoteNode(fname)))
        return c.fns.$fname(P₀; kw...)
    end
end

# ─── Level 2 — the amount × contribution seam ────────────────────────────────

function _custom_density_factor(c::CustomInclusion)
    c.density_factor === nothing && throw(
        ArgumentError(
            "this CustomInclusion was built without `density_factor`, so it " *
                "cannot be registered with a `CrackDensity` amount. Either pass " *
                "`density_factor = …` (e.g. `4π/3` for a flat elliptical object) " *
                "or register the phase with `fraction = …`."
        )
    )
    return c.density_factor
end

Core.delta_stiffness(c::CustomInclusion, N, ε) = _custom_density_factor(c) * ε * N
Core.delta_compliance(c::CustomInclusion, H, ε) = _custom_density_factor(c) * ε * H
Core.delta_conductivity(c::CustomInclusion, N, ε) = _custom_density_factor(c) * ε * N
Core.delta_resistivity(c::CustomInclusion, R, ε) = _custom_density_factor(c) * ε * R

# =============================================================================
#  Conformance checker
# =============================================================================

"""
    check_inclusion_interface(incl; physics = :elasticity, amount = :fraction,
                              verbose = true) -> Bool

Report which parts of the inclusion interface `incl` satisfies, and return
`true` when it can be fed to the homogenization schemes for the requested
`physics` (`:elasticity` or `:conduction`) and `amount` (`:fraction` or
`:density`).

Checks, in order:

1. **Level 0** — [`dimension`](@ref MeanFieldHomogenization.Core.dimension),
   [`inclusion_basis`](@ref MeanFieldHomogenization.Core.inclusion_basis),
   [`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait),
   [`shape_tensor`](@ref MeanFieldHomogenization.Core.shape_tensor).
2. **Level 1** — which entry gate is available: the Hill tensor, the
   localization tensor, or the contribution tensors.
3. **Level 2** — for `amount = :density`, the three-argument `delta_*` seams.

Works on any `AbstractInclusion`, not only [`CustomInclusion`](@ref) — use it
on your own type before wiring it into an `RVE`.
"""
function check_inclusion_interface(
        incl::Core.AbstractInclusion;
        physics::Symbol = :elasticity,
        amount::Symbol = :fraction,
        verbose::Bool = true
    )
    physics in (:elasticity, :conduction) ||
        throw(ArgumentError("`physics` must be :elasticity or :conduction, got :$physics"))
    amount in (:fraction, :density) ||
        throw(ArgumentError("`amount` must be :fraction or :density, got :$amount"))

    msgs = String[]
    ok = true

    # ── Level 0 ──────────────────────────────────────────────────────────────
    #  `shape_trait` is the only one a kernel actually reads (it keys the crack
    #  algebra).  `dimension` / `inclusion_basis` are introspection accessors
    #  expected by convention, and `shape_tensor` is optional altogether — it
    #  describes an equivalent ellipsoidal envelope, which not every morphology
    #  has.
    if !applicable(Core.shape_trait, incl)
        ok = false
        push!(msgs, "  ✗ level 0: `shape_trait` has no method for $(typeof(incl))")
    end
    for f in (Core.dimension, Core.inclusion_basis)
        applicable(f, incl) ||
            push!(msgs, "  · level 0: no `$(nameof(f))` (accessor, not required)")
    end

    # ── Level 1 ──────────────────────────────────────────────────────────────
    P₀, P₁ = _probe_tensors(physics)
    gate_a = _provides(incl, :hill_tensor, Elasticity.hill_tensor, incl, P₀) ||
        _provides_kernel(incl, P₀)
    locname = physics === :elasticity ? :strain_strain_loc : :gradient_gradient_loc
    loc = physics === :elasticity ? Core.strain_strain_loc : Core.gradient_gradient_loc
    gate_b = _provides(incl, locname, loc, incl, P₁, P₀)
    contrib = physics === :elasticity ?
        (
            (:stiffness_contribution, Core.stiffness_contribution),
            (:compliance_contribution, Core.compliance_contribution),
        ) :
        (
            (:conductivity_contribution, Core.conductivity_contribution),
            (:resistivity_contribution, Core.resistivity_contribution),
        )
    gate_c = if amount === :density
        all(nf -> _provides(incl, nf[1], nf[2], incl, P₀), contrib)
    else
        all(nf -> _provides(incl, nf[1], nf[2], incl, P₁, P₀), contrib)
    end

    if gate_a
        push!(
            msgs, "  ✓ level 1, gate A: `hill_tensor` — all localization and " *
                "contribution tensors are derived"
        )
    elseif gate_b
        push!(msgs, "  ✓ level 1, gate B: `$locname` — contributions are derived")
        # The strain-side localization fixes the stress-side one only through
        # `A_σε = C₁ : A_εε`, which needs a single uniform property.
        if !Core.is_homogeneous_inclusion(incl)
            sname = physics === :elasticity ? :stress_strain_loc : :flux_gradient_loc
            sfun = physics === :elasticity ? Core.stress_strain_loc :
                Core.flux_gradient_loc
            if !_provides(incl, sname, sfun, incl, P₁, P₀)
                ok = false
                push!(
                    msgs, "  ✗ level 1: this inclusion reports itself heterogeneous " *
                        "but supplies no `$sname`. The generic one assumes a single " *
                        "uniform property, so the average stress in the inclusion — " *
                        "and with it `SelfConsistent` / `AsymmetricSelfConsistent`, " *
                        "the schemes that consume it — would be silently wrong " *
                        "(`Dilute` and `MoriTanaka` would still look fine)."
                )
            end
        end
    elseif gate_c
        push!(
            msgs, "  ✓ level 1, gate C: `$(contrib[1][1])` / " *
                "`$(contrib[2][1])` supplied directly"
        )
        if amount === :fraction
            push!(
                msgs, "    ! gate C alone does not provide the dilute " *
                    "concentration tensor, so `MoriTanaka`, `SelfConsistent` and " *
                    "`AsymmetricSelfConsistent` are unavailable for this " *
                    "volume-fraction phase (Voigt, Reuss, Dilute, DiluteDual, " *
                    "Maxwell, PonteCastanedaWillis and DifferentialScheme work)."
            )
        end
    else
        ok = false
        args = amount === :density ? "(incl, P₀)" : "(incl, P₁, P₀)"
        push!(
            msgs, "  ✗ level 1: no entry gate for :$physics — supply one of " *
                "`hill_tensor(incl, P₀)`, `$locname(incl, P₁, P₀)`, or " *
                "`$(contrib[1][1])$args` + `$(contrib[2][1])$args`"
        )
    end

    # ── Level 2 ──────────────────────────────────────────────────────────────
    if amount === :density
        deltas = physics === :elasticity ?
            (
                (:delta_stiffness, Core.delta_stiffness),
                (:delta_compliance, Core.delta_compliance),
            ) :
            (
                (:delta_conductivity, Core.delta_conductivity),
                (:delta_resistivity, Core.delta_resistivity),
            )
        for (name, f) in deltas
            if !_provides_delta(incl, name, f, P₀)
                ok = false
                push!(
                    msgs, "  ✗ level 2: no `$name(incl, X, ε)` — this is where the " *
                        "geometric prefactor relating the density to the effective " *
                        "correction lives"
                )
            end
        end
    end

    if verbose
        head = ok ? "✓ $(typeof(incl)) satisfies the inclusion contract" :
            "✗ $(typeof(incl)) does NOT satisfy the inclusion contract"
        println(head, " (physics = :$physics, amount = :$amount)")
        foreach(println, msgs)
    end
    return ok
end

# Does `incl` genuinely *provide* the generic `f`, as opposed to merely being
# caught by one of the abstract fallbacks?  `applicable` alone is not enough:
# `strain_strain_loc(::AbstractInclusion, …)` matches every inclusion but only
# re-expresses the answer in terms of `hill_tensor`, and the `hill_tensor`
# entry point declared on `AbstractCustomInclusion` only routes to the
# `_kernel` table.  Methods owned by those two abstract types therefore do not
# count as an implementation.
const _FALLBACK_OWNERS = (Core.AbstractInclusion, Core.AbstractCustomInclusion)

# The declared type of the first argument of the method that would be called.
# `m.sig` is a `UnionAll` whenever the method is parametric, hence the unwrap.
function _dispatch_owner(f, argtypes)
    sig = Base.unwrap_unionall(which(f, argtypes).sig)
    return Base.unwrap_unionall(sig.parameters[2])
end

function _provides(::Core.AbstractInclusion, ::Symbol, f, args...)
    applicable(f, args...) || return false
    return !any(==(_dispatch_owner(f, typeof.(args))), _FALLBACK_OWNERS)
end

# A `CustomInclusion` always *has* the forwarding methods (they raise at run
# time when the callback is missing), so it is interrogated through its
# callback table instead.
_provides(c::CustomInclusion, name::Symbol, _f, _args...) = haskey(c.fns, name)

# A subtype of `AbstractCustomInclusion` may reach `hill_tensor` through the
# open `_kernel` table rather than by overriding `hill_tensor` itself.
_provides_kernel(incl::Core.AbstractInclusion, P₀) =
    applicable(Elasticity._kernel, incl, P₀, Core.Analytical())
_provides_kernel(::CustomInclusion, _P₀) = false

function _provides_delta(incl::Core.AbstractInclusion, ::Symbol, f, probe)
    applicable(f, incl, probe, 1.0) || return false
    owner = _dispatch_owner(f, (typeof(incl), typeof(probe), Float64))
    return !any(==(owner), _FALLBACK_OWNERS)
end

_provides_delta(c::CustomInclusion, ::Symbol, _f, _probe) = c.density_factor !== nothing

function _probe_tensors(physics::Symbol)
    return if physics === :elasticity
        (TensND.TensISO{3}(3.0, 2.0), TensND.TensISO{3}(6.0, 4.0))
    else
        (TensND.TensISO{3}(1.0), TensND.TensISO{3}(2.0))
    end
end
