# =============================================================================
#  rheology_iso.jl — lifting scalar models to isotropic fourth-order tensors,
#  the bridge to the ageing time-domain pipeline, and the Laplace-Carson
#  homogenization driver.
#
#  This is where the two routes meet.  One `IsoRheology` object yields both
#
#      p -> carson_relaxation(m, p)   the Laplace-Carson route (homogenize)
#      ViscoLaw(m)                    the ageing time route     (homogenize_alv)
#
#  so a non-ageing material need be described exactly once, and the two
#  pipelines can be compared on identical input.
# =============================================================================

# ── How two scalar channels make an isotropic tensor ────────────────────────

"""
    AbstractIsoPairing

Which pair of scalar quantities an [`IsoRheology`](@ref) is built from.  A
trait parameterizing the type rather than a `Symbol` field, so the choice is
resolved at compile time and stays extensible.

Concrete pairings: [`BulkShear`](@ref), [`YoungPoisson`](@ref).
"""
abstract type AbstractIsoPairing end

"""
    BulkShear

Pairing by **bulk and shear**: the two channels of the isotropic tensor relax
independently, which is the physically natural description and the one every
scheme works in internally.
"""
struct BulkShear <: AbstractIsoPairing end

"""
    YoungPoisson

Pairing by **Young's modulus and Poisson's ratio**, the pair a laboratory
reports.

!!! warning "A viscoelastic Poisson ratio is a Laplace-Carson object"
    The conversion

    ```math
    3k^{*} = \\frac{E^{*}}{1 - 2\\nu^{*}}, \\qquad
    2\\mu^{*} = \\frac{E^{*}}{1 + \\nu^{*}}
    ```

    is a *pointwise* algebraic identity in the Carson domain only.  In the time
    domain the same relation is a Volterra quotient: `ν(t)` does not divide,
    it deconvolves.  (That is exactly what the ECHOES reference does with
    `V3k = VE · inv(I - 2Vν)` on discretized operators —
    `tests/python/creep/modele2S2P1D.py`.)

    So a time-dependent Poisson ratio belongs on this side of the fence.  A
    *constant* `nu` is unambiguous and is handled as a special case, including
    for the closed-form time-domain values.
"""
struct YoungPoisson <: AbstractIsoPairing end

"""
    IsoRheology{P, A, B}(a, b)

An isotropic fourth-order viscoelastic model built from two scalar channels
`a` and `b`, combined according to the pairing `P`.

Build one with [`iso_rheology`](@ref) or [`iso_rheology_E_nu`](@ref) rather
than by calling the constructor.

It answers the same generics as a scalar model — [`carson_relaxation`](@ref),
[`carson_creep`](@ref), [`relaxation`](@ref), [`creep`](@ref),
[`complex_modulus`](@ref) — with `TensISO{4,3}` values, and converts to a
[`ViscoLaw`](@ref) for the ageing pipeline.
"""
struct IsoRheology{P <: AbstractIsoPairing, A, B} <: AbstractTensorRheology
    a::A
    b::B
end

IsoRheology{P}(a::A, b::B) where {P, A, B} = IsoRheology{P, A, B}(a, b)

"""
    iso_rheology(k_model, mu_model) -> IsoRheology{BulkShear}

Isotropic model whose bulk channel follows `k_model` and shear channel
`mu_model`, each any [`AbstractRheology`](@ref).

```math
\\mathbb{C}^{*}(p) = 3k^{*}(p)\\,\\mathbb{J} + 2\\mu^{*}(p)\\,\\mathbb{K}.
```

A `Real` in either slot is promoted to a [`Spring`](@ref), so the common case
of an elastic bulk modulus and a relaxing shear one reads

```julia
iso_rheology(2500.0, Model2S2P1D(1e-7, 1000.0, 2.2, 1.945e-3, 0.22, 0.63, 50.0))
```

# Examples

```jldoctest
julia> using MeanFieldHomogenization, TensND

julia> m = iso_rheology(zener_maxwell(30.0, 10.0, 1.0), zener_maxwell(10.0, 5.0, 0.5));

julia> C = relaxation(m, 0.0);      # the glassy stiffness

julia> all(isapprox.(get_data(C), (3 * 40.0, 2 * 15.0)))
true
```
"""
iso_rheology(k_model, mu_model) =
    IsoRheology{BulkShear}(_as_rheology(k_model), _as_rheology(mu_model))

"""
    iso_rheology_E_nu(E_model, nu) -> IsoRheology{YoungPoisson}

Isotropic model given by a Young's-modulus channel and a Poisson ratio, where
`nu` is either a constant `Real` or itself an [`AbstractRheology`](@ref).

See [`YoungPoisson`](@ref) for why a *time-dependent* `nu` only makes
unambiguous sense in the Laplace-Carson domain.
"""
iso_rheology_E_nu(E_model, nu) = IsoRheology{YoungPoisson}(_as_rheology(E_model), nu)

_as_rheology(m::AbstractRheology) = m
_as_rheology(x::Real) = Spring(x)

# ── The transforms ──────────────────────────────────────────────────────────

carson_relaxation(m::IsoRheology{BulkShear}, p) = TensND.TensISO{3}(
    3 * carson_relaxation(m.a, p),
    2 * carson_relaxation(m.b, p),
)

function carson_relaxation(m::IsoRheology{YoungPoisson}, p)
    E = carson_relaxation(m.a, p)
    ν = _nu_star(m.b, p)
    return TensND.TensISO{3}(E / (one(ν) - 2ν), E / (one(ν) + ν))
end

_nu_star(ν::Real, p) = ν * one(p)
_nu_star(ν::AbstractRheology, p) = carson_relaxation(ν, p)

# `inv` of a `TensISO` is exact and closed-form (reciprocals of the two data
# entries), so the reciprocal identity J* = (R*)^{-1} carries over verbatim.
carson_creep(m::IsoRheology, p) = inv(carson_relaxation(m, p))

# ── Closed-form time-domain values where the channels do not mix ────────────

relaxation(m::IsoRheology{BulkShear}, t::Real; method = default_inversion(m)) =
    TensND.TensISO{3}(
    3 * relaxation(m.a, t; method),
    2 * relaxation(m.b, t; method),
)

# With a constant Poisson ratio the two channels are fixed multiples of `E`, so
# the scalar model's own closed form (if it has one) still applies.  With a
# relaxing `nu` they mix, and the tensor transform must be inverted as a whole
# — which the fallback in `rheology_interface.jl` already does.
function relaxation(
        m::IsoRheology{YoungPoisson, A, <:Real}, t::Real;
        method = default_inversion(m)
    ) where {A}
    E = relaxation(m.a, t; method)
    ν = m.b
    return TensND.TensISO{3}(E / (1 - 2ν), E / (1 + ν))
end

glassy_modulus(m::IsoRheology{BulkShear}) =
    TensND.TensISO{3}(3 * glassy_modulus(m.a), 2 * glassy_modulus(m.b))

equilibrium_modulus(m::IsoRheology{BulkShear}) =
    TensND.TensISO{3}(3 * equilibrium_modulus(m.a), 2 * equilibrium_modulus(m.b))

function glassy_modulus(m::IsoRheology{YoungPoisson})
    E = glassy_modulus(m.a)
    ν = _nu_limit(m.b, glassy_modulus)
    return TensND.TensISO{3}(E / (1 - 2ν), E / (1 + ν))
end

function equilibrium_modulus(m::IsoRheology{YoungPoisson})
    E = equilibrium_modulus(m.a)
    ν = _nu_limit(m.b, equilibrium_modulus)
    return TensND.TensISO{3}(E / (1 - 2ν), E / (1 + ν))
end

_nu_limit(ν::Real, _) = ν
_nu_limit(ν::AbstractRheology, f) = f(ν)

is_fluid(m::IsoRheology) = any(iszero, TensND.get_data(equilibrium_modulus(m)))

# `loss_modulus` and `loss_factor` are scalar notions; on a tensor model they
# are asked channel by channel, which the user does by reaching into the
# scalar sub-models directly.  Only the storage part has an unambiguous
# tensorial meaning, and `rheology_interface.jl` defines it generically.

default_inversion(m::IsoRheology) = default_inversion(m.a)

function Base.show(io::IO, m::IsoRheology{P}) where {P}
    return print(io, "IsoRheology{", nameof(P), "}(", m.a, ", ", m.b, ")")
end

# ── Bridge to the ageing time-domain pipeline ───────────────────────────────

"""
    ViscoLaw(model::AbstractTensorRheology; mode = :relaxation, method = default_inversion(model))
    ViscoLaw(model::AbstractRheology;       mode = :relaxation, method = default_inversion(model))

Package a rheological model as a [`ViscoLaw`](@ref) kernel `(t, t') ↦ X`, so
that the **ageing** machinery — [`trapezoidal_matrix`](@ref),
[`volterra_inverse`](@ref), [`homogenize_alv`](@ref) — can consume a model that
was written in the Laplace-Carson domain.

The kernel produced is of course non-ageing: it depends on `t - t'` only.  That
is the point.  It lets the same material be pushed through both routes and the
answers compared, which is what
[`tutorials/generated/freq_vs_time`](@ref tut-freq-vs-time) does.

`mode = :creep` builds `J(t - t')` instead of `R(t - t')`.

!!! note "Cost"
    If the model has no closed-form time value, every entry of the trapezoidal
    matrix costs one numerical inversion — `N` evaluations of the transform.
    For an `n`-point grid that is `O(n²N)` transform evaluations.  When the
    model is a Prony series (or has been fitted to one with
    [`prony_fit_relaxation`](@ref)) the time value is closed-form and the
    inversion never runs.
"""
function ViscoLaw(
        model::AnyRheology;
        mode::Symbol = :relaxation,
        method::AbstractLaplaceInversion = default_inversion(model)
    )
    mode in VALID_VISCO_MODES ||
        throw(ArgumentError("ViscoLaw: mode must be one of $(VALID_VISCO_MODES); got :$mode"))
    f = mode === :relaxation ? relaxation : creep
    zero_value = _kernel_zero(model, mode)
    eval_fun = (t, t_p) -> t < t_p ? zero_value : f(model, t - t_p; method)
    return ViscoLaw(eval_fun, mode)
end

# Built from the glassy value rather than by calling `zero` on it: for a
# tensor-valued model `zero(::AbstractTens{4,…})` collapses every symmetry
# class onto `TensISO` (see `_accumulate`), and while that happens to be
# harmless for an `IsoRheology`, relying on it would break the moment another
# tensor pairing is added.
_kernel_zero(model::AbstractRheology, mode) = zero(_kernel_scale(model, mode))

function _kernel_zero(model::AbstractTensorRheology, mode)
    scale = _kernel_scale(model, mode)
    return TensND._rebuild(scale, map(zero, TensND.get_data(scale)))
end

_kernel_scale(model, mode) =
    mode === :relaxation ? glassy_modulus(model) : inv(glassy_modulus(model))

# ── The Laplace-Carson homogenization driver ────────────────────────────────

"""
    homogenize_lc(build_cell, scheme, property = :C; p)
    homogenize_lc(build_cell, scheme, property = :C; times, method = DEFAULT_INVERSION, kw...)

Homogenize a **non-ageing** viscoelastic composite through the correspondence
principle, and — in the second form — bring the answer back to the time domain.

`build_cell(p)` is a closure returning the homogenization cell with every phase
property evaluated at the Carson variable `p`.  Any cell works: an
[`RVE`](@ref), a `LayeredSphere`, a `Laminate`, a nested chain of them.  Extra
keyword arguments are forwarded to [`homogenize`](@ref).

# Example

```julia
matrix = iso_rheology(Spring(2500.0), Model2S2P1D(1e-7, 1000.0, 2.2, 1.945e-3, 0.22, 0.63, 50.0))
aggreg = iso_rheology(Spring(30000.0), Spring(22000.0))

function cell(p)
    rve = RVE(:MASTIC)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => carson_relaxation(matrix, p)))
    add_phase!(rve, :AGG, Ellipsoid(1.0), Dict(:C => carson_relaxation(aggreg, p));
               fraction = 0.35)
    return rve
end

E_star = homogenize_lc(cell, MoriTanaka(), :C; p = im * 2π * 10)     # 10 Hz
R_of_t = homogenize_lc(cell, MoriTanaka(), :C; times = exp10.(-4:0.1:2))
```

# Choosing the inversion method

Each time costs one inversion, and each inversion costs `N` full
homogenizations, so the choice is a real one:

  * [`FixedTalbot`](@ref) (the default) — most accurate, 24 *complex*
    homogenizations per time;
  * [`GaverStehfest`](@ref) — 16 **real** homogenizations per time.  Real
    arithmetic is roughly twice as cheap per evaluation, and it is the only way
    to use `SelfConsistent(algorithm = NewtonDefault())` here, whose
    `ForwardDiff` Jacobian cannot carry a `Dual` over a complex scalar.  The
    price is about five significant digits instead of twelve;
  * [`DeHoog`](@ref) — one node set serves a whole block of times, so a
    multi-decade grid costs far fewer homogenizations than any per-point
    method.  This is usually the right answer for a master curve.

# Relation to the time-domain route

For a non-ageing material this and [`homogenize_alv`](@ref) compute the same
thing by disjoint means — no shared code — so agreement between them is a real
check on both.  `ViscoLaw(model)` turns the same model objects into the kernels
`homogenize_alv` needs.
"""
function homogenize_lc(
        build_cell, scheme::HomogenizationScheme, property::Symbol = :C;
        p = nothing,
        times::Union{Nothing, AbstractVector{<:Real}} = nothing,
        method::AbstractLaplaceInversion = DEFAULT_INVERSION,
        kw...
    )
    (p === nothing) == (times === nothing) && throw(
        ArgumentError(
            "homogenize_lc: give exactly one of `p` (a single Carson variable, " *
                "typically `im * ω`) or `times` (a time grid to invert onto)"
        )
    )
    Cstar = q -> MFH_Core.homogenize(build_cell(q), scheme, property; kw...)
    p === nothing || return Cstar(p)
    return inverse_carson(Cstar, times, method)
end
