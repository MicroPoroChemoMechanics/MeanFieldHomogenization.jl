# =============================================================================
#  rheology_interface.jl — the five generics every rheological model answers,
#  and the fallback lattice that makes a model complete from one method.
#
#  A scalar linear viscoelastic model is fully determined by its Laplace-Carson
#  relaxation transform R*(p).  Everything else follows:
#
#      J*(p) = 1 / R*(p)                     exact algebra
#      R(t)  = LC⁻¹[R*](t)                   numerical inversion
#      J(t)  = LC⁻¹[J*](t)                   numerical inversion
#      E*(ω) = R*(iω)                        the complex modulus
#
#  so `carson_relaxation` is the only method a new model *must* provide.  Every
#  model in the catalog overrides whichever of the others it knows in closed
#  form, and the fallbacks cover the rest.
# =============================================================================

"""
    AbstractRheology

Root supertype for **scalar** linear viscoelastic models — a spring, a Prony
series, 2S2P1D, and everything in between.

A concrete model must define [`carson_relaxation`](@ref); it *should* also
define [`glassy_modulus`](@ref) and [`equilibrium_modulus`](@ref), which are
limits of that transform and are almost always elementary.  Anything else it
knows in closed form ([`relaxation`](@ref), [`creep`](@ref),
[`carson_creep`](@ref)) is an override; what it does not know is obtained by
numerical inversion.

Lift a pair of scalar models to a fourth-order isotropic tensor with
[`iso_rheology`](@ref) / [`iso_rheology_E_nu`](@ref); the result is an
[`AbstractTensorRheology`](@ref), which answers the same generics with tensors
and additionally feeds the ageing time-domain pipeline through
`ViscoLaw(model)`.

See the [rheological model catalog](@ref man-rheological-models).
"""
abstract type AbstractRheology end

"""
    AbstractTensorRheology

Root supertype for tensor-valued viscoelastic models — a scalar model in each
independent symmetry channel.  The only concrete subtype at present is
[`IsoRheology`](@ref).
"""
abstract type AbstractTensorRheology end

const AnyRheology = Union{AbstractRheology, AbstractTensorRheology}

# ── The five generics ───────────────────────────────────────────────────────

"""
    carson_relaxation(model, p)

The **Laplace-Carson relaxation transform** ``R^{*}(p) = p\\,\\hat R(p)`` — the
one method a model must implement.

`p` may be real (as [`GaverStehfest`](@ref) needs), complex (`p = iω` gives the
complex modulus), or a `ForwardDiff.Dual`, so implementations must stay generic
in the argument type.

For an [`AbstractTensorRheology`](@ref) the value is a fourth-order tensor.
"""
function carson_relaxation end

"""
    carson_creep(model, p)

The **Laplace-Carson creep transform** ``J^{*}(p)``.

In the Carson domain creep and relaxation are exact reciprocals,
``J^{*}(p)\\,R^{*}(p) = 1``, which is the default implementation.  Models whose
creep transform is the *simpler* of the two ([`PronyCreep`](@ref),
[`burgers`](@ref)) override it and let `carson_relaxation` be the derived one.
"""
carson_creep(model::AbstractRheology, p) = one(p) / carson_relaxation(model, p)

"""
    relaxation(model, t)

The relaxation function ``R(t)``: the stress response to a unit strain step
applied at `t = 0`.

Falls back to numerically inverting [`carson_relaxation`](@ref) with
[`default_inversion`](@ref); models with a closed form override it.  `t = 0` is
answered analytically by [`glassy_modulus`](@ref) — the inversion itself has a
pole there.
"""
function relaxation(model::AnyRheology, t::Real; method = default_inversion(model))
    iszero(t) && return glassy_modulus(model)
    return inverse_carson(p -> carson_relaxation(model, p), t, method)
end

"""
    creep(model, t)

The creep compliance ``J(t)``: the strain response to a unit stress step applied
at `t = 0`.

Falls back to numerically inverting [`carson_creep`](@ref); models with a closed
form override it.
"""
function creep(model::AnyRheology, t::Real; method = default_inversion(model))
    iszero(t) && return inv(glassy_modulus(model))
    return inverse_carson(p -> carson_creep(model, p), t, method)
end

"""
    complex_modulus(model, ω)

The complex modulus ``E^{*}(\\omega) = R^{*}(i\\omega)``, the quantity a
dynamic-mechanical test measures.

Its modulus `abs(E*)` is the *norm* of the complex modulus and its argument
`angle(E*)` the phase angle; [`storage_modulus`](@ref),
[`loss_modulus`](@ref) and [`loss_factor`](@ref) name the usual derived
quantities.
"""
complex_modulus(model::AnyRheology, ω::Real) = carson_relaxation(model, im * ω)

# ── Derived dynamic quantities ──────────────────────────────────────────────

"""
    storage_modulus(model, ω)

``E'(\\omega) = \\Re E^{*}(\\omega)`` — the in-phase, elastically stored part.
"""
storage_modulus(model::AnyRheology, ω::Real) = _realpart(complex_modulus(model, ω))

"""
    loss_modulus(model, ω)

``E''(\\omega) = \\Im E^{*}(\\omega)`` — the out-of-phase, dissipated part.
"""
loss_modulus(model::AbstractRheology, ω::Real) = imag(complex_modulus(model, ω))

"""
    loss_factor(model, ω)

``\\tan\\delta = E''/E'`` — the phase angle's tangent.  Non-negative for any
passive material, which makes it a cheap sanity check on a fitted spectrum.
"""
function loss_factor(model::AbstractRheology, ω::Real)
    E = complex_modulus(model, ω)
    return imag(E) / real(E)
end

# ── Limits ──────────────────────────────────────────────────────────────────

"""
    glassy_modulus(model)

``R(0^{+}) = \\lim_{p\\to\\infty} R^{*}(p)`` — the instantaneous ("glassy")
modulus.

Closed form for every model in the catalog.  Besides being the `t = 0` value
of [`relaxation`](@ref), it is the `f_glassy` argument of
[`inverse_carson_rate`](@ref).
"""
function glassy_modulus end

"""
    equilibrium_modulus(model)

``R(\\infty) = \\lim_{p\\to 0} R^{*}(p)`` — the relaxed ("equilibrium" or
"static") modulus.

Zero exactly when the model is a **fluid**: stress relaxes away completely and
the creep compliance grows without bound.  See [`is_fluid`](@ref).
"""
function equilibrium_modulus end

"""
    is_fluid(model) -> Bool

`true` when [`equilibrium_modulus`](@ref) vanishes, i.e. the model contains a
dashpot in series with everything else and creeps indefinitely.

The distinction is structural rather than cosmetic: a fluid's creep compliance
carries a term linear in `t`, which in the Carson domain is the pole `φ/p` of
[`PronyCreep`](@ref), and it is what decides whether
[`kelvin_to_maxwell`](@ref) and [`maxwell_to_kelvin`](@ref) look for a root in
the outermost interval.
"""
is_fluid(model::AnyRheology) = iszero(equilibrium_modulus(model))

"""
    default_inversion(model) -> AbstractLaplaceInversion

The inversion method [`relaxation`](@ref) and [`creep`](@ref) use when the
model has no closed form and the caller gave none.

Defaults to [`DEFAULT_INVERSION`](@ref); a model overrides it when its
transform has a feature one algorithm handles better.
"""
default_inversion(::AnyRheology) = DEFAULT_INVERSION
