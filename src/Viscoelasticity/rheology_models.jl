# =============================================================================
#  rheology_models.jl — the catalog of scalar viscoelastic models.
#
#  Each model states its Laplace-Carson relaxation transform `R*(p)`, its two
#  limits, and whichever of `J*(p)`, `R(t)`, `J(t)` it knows in closed form.
#  The fallbacks in `rheology_interface.jl` supply the rest by numerical
#  inversion, so every model answers all five generics whatever it declared.
#
#  Convention throughout: `E00` (or `E_inf`) is the *equilibrium* / static
#  modulus reached as ω → 0, and `E0` the *glassy* / instantaneous one reached
#  as ω → ∞.  This follows Di Benedetto & Olard for 2S2P1D and the ECHOES
#  sources, and is why `glassy_modulus` and `equilibrium_modulus` read the way
#  they do below.
# =============================================================================

# ── Elementary elements ─────────────────────────────────────────────────────

"""
    Spring(E)

A Hookean spring: `R(t) = E`, `J(t) = 1/E`, `R*(p) = E`.

The elastic limit of every other model, and the natural way to give a phase a
constant modulus in one channel while another channel relaxes — the bulk
modulus of a bituminous binder, for instance.

Being its own Laplace-Carson transform is not an accident: it is the property
that makes the Carson convention the natural one in viscoelasticity.
"""
struct Spring{T <: Real} <: AbstractRheology
    E::T
end

carson_relaxation(m::Spring, p) = m.E * one(p)
carson_creep(m::Spring, p) = one(p) / m.E
relaxation(m::Spring, t::Real; method = nothing) = m.E * one(t)
creep(m::Spring, t::Real; method = nothing) = one(t) / m.E
glassy_modulus(m::Spring) = m.E
equilibrium_modulus(m::Spring) = m.E

"""
    Dashpot(η)

A Newtonian dashpot: `J(t) = t/η`, `R*(p) = pη`.

Its relaxation function is `η δ(t)`, a distribution rather than a function, so
[`relaxation`](@ref) throws for this model — as does [`glassy_modulus`](@ref),
which would be infinite.  A dashpot is meant to be assembled with springs
([`MaxwellUnit`](@ref), [`burgers`](@ref)), not loaded on its own.
"""
struct Dashpot{T <: Real} <: AbstractRheology
    η::T
end

carson_relaxation(m::Dashpot, p) = p * m.η
carson_creep(m::Dashpot, p) = one(p) / (p * m.η)
creep(m::Dashpot, t::Real; method = nothing) = t / m.η
equilibrium_modulus(m::Dashpot) = zero(m.η)

_dirac_error(who) = throw(
    ArgumentError(
        "$who: the relaxation function is a Dirac impulse, not a function. " *
            "Assemble the element with a spring (MaxwellUnit, zener_kelvin, burgers) " *
            "before asking for R(t) or the glassy modulus."
    )
)

relaxation(::Dashpot, ::Real; method = nothing) = _dirac_error("Dashpot")
glassy_modulus(::Dashpot) = _dirac_error("Dashpot")

"""
    MaxwellUnit(E, η)

A spring and a dashpot in **series**:

```math
R(t) = E\\,e^{-t/\\tau}, \\qquad J(t) = \\frac{1}{E} + \\frac{t}{\\eta},
\\qquad R^{*}(p) = E\\,\\frac{p\\tau}{1+p\\tau}, \\qquad \\tau = \\eta/E .
```

A **fluid**: the stress relaxes to zero and the strain grows without bound.
Equivalent to `PronyRelaxation(0, [E], [η/E])`, and that is what
[`kelvin_to_maxwell`](@ref) and friends produce; this type exists so the
physical parameters `(E, η)` can be given directly.
"""
struct MaxwellUnit{T <: Real} <: AbstractRheology
    E::T
    η::T
    MaxwellUnit(E::Real, η::Real) = (
        T = promote_type(typeof(E), typeof(η), Float64);
        new{T}(T(E), T(η))
    )
end

relaxation_time(m::MaxwellUnit) = m.η / m.E

carson_relaxation(m::MaxwellUnit, p) = (τ = relaxation_time(m); m.E * p * τ / (1 + p * τ))
carson_creep(m::MaxwellUnit, p) = one(p) / m.E + one(p) / (p * m.η)
relaxation(m::MaxwellUnit, t::Real; method = nothing) = m.E * exp(-t / relaxation_time(m))
creep(m::MaxwellUnit, t::Real; method = nothing) = one(t) / m.E + t / m.η
glassy_modulus(m::MaxwellUnit) = m.E
equilibrium_modulus(m::MaxwellUnit) = zero(m.E)

"""
    KelvinUnit(E, η)

A spring and a dashpot in **parallel**:

```math
J(t) = \\frac{1 - e^{-t/\\tau}}{E}, \\qquad
R^{*}(p) = E\\,(1 + p\\tau), \\qquad \\tau = \\eta/E .
```

The strain is bounded but the *instantaneous* response is rigid, so the
relaxation function again carries a Dirac and [`relaxation`](@ref) throws.  Put
it in series with a spring — [`zener_kelvin`](@ref) — for a usable solid.
"""
struct KelvinUnit{T <: Real} <: AbstractRheology
    E::T
    η::T
    KelvinUnit(E::Real, η::Real) = (
        T = promote_type(typeof(E), typeof(η), Float64);
        new{T}(T(E), T(η))
    )
end

retardation_time(m::KelvinUnit) = m.η / m.E

carson_relaxation(m::KelvinUnit, p) = m.E * (one(p) + p * retardation_time(m))
carson_creep(m::KelvinUnit, p) = one(p) / (m.E * (one(p) + p * retardation_time(m)))
creep(m::KelvinUnit, t::Real; method = nothing) =
    (one(t) - exp(-t / retardation_time(m))) / m.E
equilibrium_modulus(m::KelvinUnit) = m.E

relaxation(::KelvinUnit, ::Real; method = nothing) = _dirac_error("KelvinUnit")
glassy_modulus(::KelvinUnit) = _dirac_error("KelvinUnit")

# ── Standard solid and Burgers: named Prony chains ──────────────────────────

"""
    zener_maxwell(E_inf, E_1, tau_1) -> PronyRelaxation

The standard linear solid in its **Maxwell (relaxation) representation**: an
equilibrium spring `E_inf` in parallel with one Maxwell branch.

```math
R(t) = E_\\infty + E_1 e^{-t/\\tau_1}.
```

[`zener_kelvin`](@ref) is the same material written the other way round, and
[`maxwell_to_kelvin`](@ref) converts between them exactly — with, for one
branch, the classical closed form

```math
J_0 = \\frac{1}{E_\\infty + E_1}, \\qquad
J_1 = \\frac{1}{E_\\infty} - J_0, \\qquad
\\tau^{K}_1 = \\tau_1\\,\\frac{E_\\infty + E_1}{E_\\infty}.
```
"""
zener_maxwell(E_inf::Real, E_1::Real, tau_1::Real) =
    PronyRelaxation(E_inf, [E_1], [tau_1])

"""
    zener_kelvin(E_glassy, E_delayed, tau_1) -> PronyCreep

The standard linear solid in its **Kelvin (creep) representation**: a spring of
modulus `E_glassy` in series with one Kelvin cell of modulus `E_delayed` and
retardation time `tau_1`.

```math
J(t) = \\frac{1}{E_{\\rm glassy}}
     + \\frac{1}{E_{\\rm delayed}}\\bigl(1 - e^{-t/\\tau_1}\\bigr).
```

See [`zener_maxwell`](@ref) and [`kelvin_to_maxwell`](@ref).
"""
zener_kelvin(E_glassy::Real, E_delayed::Real, tau_1::Real) =
    PronyCreep(one(E_glassy) / E_glassy, [one(E_delayed) / E_delayed], [tau_1])

"""
    burgers(k_s, eta_s, k_p, eta_p) -> PronyCreep

The Burgers model: a Maxwell unit `(k_s, eta_s)` in series with a Kelvin cell
`(k_p, eta_p)`.

```math
J(t) = \\frac{1}{k_s} + \\frac{t}{\\eta_s}
     + \\frac{1}{k_p}\\bigl(1 - e^{-t k_p/\\eta_p}\\bigr).
```

A **fluid**, so [`kelvin_to_maxwell`](@ref) returns a *two*-branch
[`PronyRelaxation`](@ref) with no equilibrium spring — one branch more than the
Kelvin form has, which is the degree count of the rational transform.  Its
relaxation function then matches the classical closed-form `cosh`/`sinh`
expression to machine precision, which is how the fluid path of the conversion
is validated (`test/Viscoelasticity/test_prony.jl`).

Parameter names follow the ECHOES reference
`tests/python/creep/fluage_echoes_ijss2013_jsanahuja_relaxBurgers.py`.
"""
burgers(k_s::Real, eta_s::Real, k_p::Real, eta_p::Real) =
    PronyCreep(one(k_s) / k_s, [one(k_p) / k_p], [eta_p / k_p], one(eta_s) / eta_s)

# ── Fractional elements ─────────────────────────────────────────────────────

"""
    ScottBlair(V, α)

The **springpot** (fractional dashpot, Scott-Blair element), interpolating
continuously between a spring (`α = 0`) and a dashpot (`α = 1`):

```math
R^{*}(p) = V p^{\\alpha}, \\qquad
R(t) = \\frac{V\\,t^{-\\alpha}}{\\Gamma(1-\\alpha)}, \\qquad
J(t) = \\frac{t^{\\alpha}}{V\\,\\Gamma(1+\\alpha)} .
```

`0 < α < 1`.  The exact pair ``t^{a} \\leftrightarrow \\Gamma(a+1)\\,p^{-a}`` is
what makes the power-law terms of [`HuetSayegh`](@ref) and
[`Model2S2P1D`](@ref) analytic in both domains.

The transform has a branch cut along the negative real axis, which the Talbot
contours enclose rather than cross — [`FixedTalbot`](@ref) inverts it at
`1e-12`, no special handling needed.
"""
struct ScottBlair{T <: Real} <: AbstractRheology
    V::T
    α::T

    function ScottBlair(V::Real, α::Real)
        0 < α < 1 || throw(ArgumentError("ScottBlair: α must lie in (0,1); got $α"))
        T = promote_type(typeof(V), typeof(α), Float64)
        return new{T}(T(V), T(α))
    end
end

carson_relaxation(m::ScottBlair, p) = m.V * p^m.α
carson_creep(m::ScottBlair, p) = one(p) / (m.V * p^m.α)
relaxation(m::ScottBlair, t::Real; method = nothing) =
    m.V * t^(-m.α) / gamma(1 - m.α)
creep(m::ScottBlair, t::Real; method = nothing) =
    t^m.α / (m.V * gamma(1 + m.α))
equilibrium_modulus(m::ScottBlair) = zero(m.V)
glassy_modulus(::ScottBlair) = _dirac_error("ScottBlair")

"""
    FractionalMaxwell(V_a, α, V_b, β)

Two springpots in **series**:
``R^{*}(p) = \\dfrac{V_a p^{\\alpha}\\,V_b p^{\\beta}}{V_a p^{\\alpha} + V_b p^{\\beta}}``.

By convention `α > β`, so the `α` element is the stiffer one at short times.
A fluid.
"""
struct FractionalMaxwell{T <: Real} <: AbstractRheology
    V_a::T
    α::T
    V_b::T
    β::T

    function FractionalMaxwell(V_a::Real, α::Real, V_b::Real, β::Real)
        (0 ≤ β < α ≤ 1) || throw(
            ArgumentError("FractionalMaxwell: need 0 ≤ β < α ≤ 1; got α = $α, β = $β")
        )
        T = promote_type(typeof(V_a), typeof(α), typeof(V_b), typeof(β), Float64)
        return new{T}(T(V_a), T(α), T(V_b), T(β))
    end
end

function carson_relaxation(m::FractionalMaxwell, p)
    a = m.V_a * p^m.α
    b = m.V_b * p^m.β
    return a * b / (a + b)
end
carson_creep(m::FractionalMaxwell, p) =
    one(p) / (m.V_a * p^m.α) + one(p) / (m.V_b * p^m.β)
equilibrium_modulus(m::FractionalMaxwell) = zero(m.V_a)

"""
    FractionalKelvin(V_a, α, V_b, β)

Two springpots in **parallel**:
``R^{*}(p) = V_a p^{\\alpha} + V_b p^{\\beta}``, with `α > β`.

With `β = 0` the slow element is a spring and the model is a solid of
equilibrium modulus `V_b`.
"""
struct FractionalKelvin{T <: Real} <: AbstractRheology
    V_a::T
    α::T
    V_b::T
    β::T

    function FractionalKelvin(V_a::Real, α::Real, V_b::Real, β::Real)
        (0 ≤ β < α ≤ 1) || throw(
            ArgumentError("FractionalKelvin: need 0 ≤ β < α ≤ 1; got α = $α, β = $β")
        )
        T = promote_type(typeof(V_a), typeof(α), typeof(V_b), typeof(β), Float64)
        return new{T}(T(V_a), T(α), T(V_b), T(β))
    end
end

carson_relaxation(m::FractionalKelvin, p) = m.V_a * p^m.α + m.V_b * p^m.β
equilibrium_modulus(m::FractionalKelvin) = iszero(m.β) ? m.V_b : zero(m.V_b)

"""
    FractionalZener(E_inf, E_0, tau, α)

The fractional standard solid, also known as the **Cole-Cole** model:

```math
R^{*}(p) = E_\\infty + (E_0 - E_\\infty)\\,\\frac{(p\\tau)^{\\alpha}}
                                                 {1 + (p\\tau)^{\\alpha}},
\\qquad
R(t) = E_\\infty + (E_0 - E_\\infty)\\,E_{\\alpha}\\!\\bigl(-(t/\\tau)^{\\alpha}\\bigr),
```

with ``E_\\alpha`` the one-parameter Mittag-Leffler function.  `α = 1` recovers
[`zener_maxwell`](@ref); smaller `α` broadens the relaxation spectrum, which is
what makes the model fit polymers and bitumen where a single exponential cannot.

The time-domain form needs `MittagLeffler.jl` (a weak dependency); without it,
[`relaxation`](@ref) falls back to numerically inverting the closed-form
transform above, which costs a little accuracy and nothing else.
"""
struct FractionalZener{T <: Real} <: AbstractRheology
    E_inf::T
    E_0::T
    tau::T
    α::T

    function FractionalZener(E_inf::Real, E_0::Real, tau::Real, α::Real)
        0 < α ≤ 1 || throw(ArgumentError("FractionalZener: α must lie in (0,1]; got $α"))
        tau > 0 || throw(ArgumentError("FractionalZener: tau must be positive; got $tau"))
        T = promote_type(typeof(E_inf), typeof(E_0), typeof(tau), typeof(α), Float64)
        return new{T}(T(E_inf), T(E_0), T(tau), T(α))
    end
end

function carson_relaxation(m::FractionalZener, p)
    x = (p * m.tau)^m.α
    return m.E_inf + (m.E_0 - m.E_inf) * x / (1 + x)
end
glassy_modulus(m::FractionalZener) = m.E_0
equilibrium_modulus(m::FractionalZener) = m.E_inf

function relaxation(m::FractionalZener, t::Real; method = default_inversion(m))
    iszero(t) && return m.E_0
    ml = _mittag_leffler(m.α, one(m.α), -(t / m.tau)^m.α)
    ml === nothing && return inverse_carson(p -> carson_relaxation(m, p), t, method)
    return m.E_inf + (m.E_0 - m.E_inf) * ml
end

"""
    Rabotnov(mu_0, lambda_0, α, β)

The Rabotnov fractional-exponential kernel, in the form ECHOES uses as its
analytical benchmark for the ageing pipeline
(`tests/python/creep/fluage_echoes_maxwell_papier_rabotnov.py`):

```math
R(t) = \\mu_0\\Bigl(1 + \\lambda_0\\,\\frac{1 - E_{\\alpha+1,1}
        \\bigl(-\\beta t^{\\alpha+1}\\bigr)}{\\beta}\\Bigr).
```

!!! tip "The Carson transform is elementary"
    Since ``\\mathcal{L}\\{E_{a,1}(-\\beta t^{a})\\}(p) = p^{a-1}/(p^{a}+\\beta)``,
    the Carson transform of that kernel collapses to

    ```math
    R^{*}(p) = \\mu_0\\left(1 + \\frac{\\lambda_0}{p^{\\alpha+1} + \\beta}\\right),
    ```

    with **no Mittag-Leffler function anywhere**.  The model is therefore
    complete without the `MittagLeffler` extension: the transform is closed
    form and the inversion supplies the time domain to about `1e-10`.  With the
    extension loaded, `relaxation` uses the closed form instead — the two agree,
    which is one of the cross-checks in `test/Viscoelasticity/test_rheology.jl`.

`mu_0` is the **glassy** modulus (the ``\\lambda_0`` term vanishes as `p → ∞`)
and ``\\mu_0(1 + \\lambda_0/\\beta)`` the equilibrium one.  A passive material
therefore has ``\\lambda_0 < 0``: the benchmark of
[barthelemyIJES2019](@cite) §5 uses `μ₀ = 1.7`, `λ₀ = -0.495`, `α = -0.46`,
`β = 0.98`, and `α ∈ (-1, 0)` is the usual range — it is `α + 1` that must be
positive, not `α`.
"""
struct Rabotnov{T <: Real} <: AbstractRheology
    mu_0::T
    lambda_0::T
    α::T
    β::T

    function Rabotnov(mu_0::Real, lambda_0::Real, α::Real, β::Real)
        β > 0 || throw(ArgumentError("Rabotnov: β must be positive; got $β"))
        α > -1 || throw(
            ArgumentError(
                "Rabotnov: α must exceed -1 (the kernel is built on E_{α+1,1}); got $α"
            )
        )
        T = promote_type(typeof(mu_0), typeof(lambda_0), typeof(α), typeof(β), Float64)
        return new{T}(T(mu_0), T(lambda_0), T(α), T(β))
    end
end

carson_relaxation(m::Rabotnov, p) =
    m.mu_0 * (one(p) + m.lambda_0 / (p^(m.α + 1) + m.β))
glassy_modulus(m::Rabotnov) = m.mu_0
equilibrium_modulus(m::Rabotnov) = m.mu_0 * (1 + m.lambda_0 / m.β)

function relaxation(m::Rabotnov, t::Real; method = default_inversion(m))
    iszero(t) && return m.mu_0
    a = m.α + 1
    ml = _mittag_leffler(a, one(a), -m.β * t^a)
    ml === nothing && return inverse_carson(p -> carson_relaxation(m, p), t, method)
    return m.mu_0 * (1 + m.lambda_0 * (1 - ml) / m.β)
end

# ── Bituminous binders ──────────────────────────────────────────────────────

"""
    HuetSayegh(E00, E0, δ, τ, k, h)

The Huet-Sayegh model: two parabolic (springpot) elements and a spring in
series, the whole in parallel with a spring `E00`.

```math
E^{*}(p) = E_{00} + \\frac{E_0 - E_{00}}
      {1 + \\delta\\,(p\\tau)^{-k} + (p\\tau)^{-h}} .
```

`E00` is the static modulus, `E0` the glassy one, and `0 < k < h < 1`.
A **solid**: unlike [`Model2S2P1D`](@ref) there is no series dashpot, so the
strain stays bounded.
"""
struct HuetSayegh{T <: Real} <: AbstractRheology
    E00::T
    E0::T
    δ::T
    τ::T
    k::T
    h::T

    function HuetSayegh(E00::Real, E0::Real, δ::Real, τ::Real, k::Real, h::Real)
        0 < k < h < 1 || throw(
            ArgumentError("HuetSayegh: need 0 < k < h < 1; got k = $k, h = $h")
        )
        τ > 0 || throw(ArgumentError("HuetSayegh: τ must be positive; got $τ"))
        T = promote_type(
            typeof(E00), typeof(E0), typeof(δ), typeof(τ), typeof(k), typeof(h), Float64
        )
        return new{T}(T(E00), T(E0), T(δ), T(τ), T(k), T(h))
    end
end

carson_relaxation(m::HuetSayegh, p) =
    m.E00 + (m.E0 - m.E00) / (one(p) + m.δ * (p * m.τ)^(-m.k) + (p * m.τ)^(-m.h))
glassy_modulus(m::HuetSayegh) = m.E0
equilibrium_modulus(m::HuetSayegh) = m.E00

"""
    Model2S2P1D(E00, E0, δ, τ_E, k, h, β)

The 2S2P1D model of Di Benedetto and Olard — **2** Springs, **2** Parabolic
elements, **1** Dashpot — the reference rheological model for bituminous
binders and mixtures:

```math
E^{*}(p) = E_{00} + \\frac{E_0 - E_{00}}{\\varphi^{*}(p)},
\\qquad
\\varphi^{*}(p) = 1 + \\delta\\,(p\\tau_E)^{-k} + (p\\tau_E)^{-h}
                 + \\frac{1}{\\beta\\, p\\,\\tau_E} .
```

`E00` is the static modulus, `E0` the glassy one, `0 < k < h < 1`, and the
series dashpot `β` makes the model a **fluid**.

!!! tip "One model, an exact pair in both domains"
    Because ``\\mathcal{LC}\\{t^{a}\\} = \\Gamma(a+1)\\,p^{-a}``, the denominator
    above is the Laplace-Carson transform of an ordinary function of time,

    ```math
    \\varphi(t) = 1 + \\delta\\,\\frac{(t/\\tau_E)^{k}}{\\Gamma(k+1)}
                   + \\frac{(t/\\tau_E)^{h}}{\\Gamma(h+1)}
                   + \\frac{t}{\\beta\\,\\tau_E},
    ```

    term by term — see [`creep_kernel`](@ref).  So 2S2P1D is *simultaneously*
    an exact Laplace-Carson model and an exact Volterra model: the same material
    can be pushed through [`homogenize_lc`](@ref) and through
    [`homogenize_alv`](@ref) with no approximation on the input side, and the
    two must agree.  That is what makes it the headline cross-validation of the
    two routes.

    [`creep_kernel_law`](@ref) returns ``\\varphi`` packaged as a
    [`ViscoLaw`](@ref) for the ageing pipeline.

!!! warning "The field names in older code are swapped"
    `docs/src/applications/bituminous.md` used to define its own struct whose
    fields were named `E0` for the static modulus and `Einf` for the glassy one
    — the reverse of the convention here and in the literature.  The *formula*
    and the *positional order* were already right, so migrating that page is a
    pure rename with no numerical change; the twelve calibrated parameter sets
    carry over argument for argument.
"""
struct Model2S2P1D{T <: Real} <: AbstractRheology
    E00::T
    E0::T
    δ::T
    τ_E::T
    k::T
    h::T
    β::T

    function Model2S2P1D(
            E00::Real, E0::Real, δ::Real, τ_E::Real, k::Real, h::Real, β::Real
        )
        0 < k < h < 1 || throw(
            ArgumentError("Model2S2P1D: need 0 < k < h < 1; got k = $k, h = $h")
        )
        τ_E > 0 || throw(ArgumentError("Model2S2P1D: τ_E must be positive; got $τ_E"))
        β > 0 || throw(ArgumentError("Model2S2P1D: β must be positive; got $β"))
        T = promote_type(
            typeof(E00), typeof(E0), typeof(δ), typeof(τ_E),
            typeof(k), typeof(h), typeof(β), Float64
        )
        return new{T}(T(E00), T(E0), T(δ), T(τ_E), T(k), T(h), T(β))
    end
end

"""
    carson_creep_kernel(m::Model2S2P1D, p)

The dimensionless denominator ``\\varphi^{*}(p)`` of the 2S2P1D transform — the
Laplace-Carson transform of [`creep_kernel`](@ref).
"""
carson_creep_kernel(m::Model2S2P1D, p) =
    one(p) + m.δ * (p * m.τ_E)^(-m.k) + (p * m.τ_E)^(-m.h) + one(p) / (m.β * p * m.τ_E)

"""
    creep_kernel(m::Model2S2P1D, t)

The dimensionless creep kernel ``\\varphi(t)`` in the time domain,

```math
\\varphi(t) = 1 + \\delta\\,\\frac{(t/\\tau_E)^{k}}{\\Gamma(k+1)}
                + \\frac{(t/\\tau_E)^{h}}{\\Gamma(h+1)}
                + \\frac{t}{\\beta\\,\\tau_E},
```

the exact term-by-term counterpart of [`carson_creep_kernel`](@ref).
"""
creep_kernel(m::Model2S2P1D, t::Real) =
    one(t) + m.δ * (t / m.τ_E)^m.k / gamma(m.k + 1) +
    (t / m.τ_E)^m.h / gamma(m.h + 1) + t / (m.β * m.τ_E)

carson_relaxation(m::Model2S2P1D, p) =
    m.E00 + (m.E0 - m.E00) / carson_creep_kernel(m, p)
glassy_modulus(m::Model2S2P1D) = m.E0
equilibrium_modulus(m::Model2S2P1D) = m.E00

"""
    LogarithmicCreep(E, C, τ)

The logarithmic creep law used for concrete at long times:

```math
J(t) = \\frac{1}{E} + \\frac{1}{C}\\,\\ln\\!\\Bigl(1 + \\frac{t}{\\tau}\\Bigr),
\\qquad
J^{*}(p) = \\frac{1}{E} + \\frac{e^{p\\tau}\\,E_1(p\\tau)}{C},
```

with ``E_1`` the exponential integral.  A **fluid** in the sense that the strain
grows without bound, though only logarithmically.

The transform is evaluated through `SpecialFunctions.expintx`, the scaled form
``e^{z}E_1(z)``, so it stays finite for large `pτ` where `exp(pτ)` alone would
overflow.

This is the non-ageing skeleton of the ageing law
`logcompliance` in the ECHOES `ageing_visco_mat.py`; the ageing version, where
`E`, `C` and `τ` depend on the loading age, belongs to the
[time-domain route](@ref man-viscoelasticity) instead.
"""
struct LogarithmicCreep{T <: Real} <: AbstractRheology
    E::T
    C::T
    τ::T

    function LogarithmicCreep(E::Real, C::Real, τ::Real)
        τ > 0 || throw(ArgumentError("LogarithmicCreep: τ must be positive; got $τ"))
        T = promote_type(typeof(E), typeof(C), typeof(τ), Float64)
        return new{T}(T(E), T(C), T(τ))
    end
end

carson_creep(m::LogarithmicCreep, p) =
    one(p) / m.E + expintx(1, p * m.τ) / m.C
carson_relaxation(m::LogarithmicCreep, p) = one(p) / carson_creep(m, p)
creep(m::LogarithmicCreep, t::Real; method = nothing) =
    one(t) / m.E + log1p(t / m.τ) / m.C
glassy_modulus(m::LogarithmicCreep) = m.E
equilibrium_modulus(m::LogarithmicCreep) = zero(m.E)

"""
    creep_kernel_law(m::Model2S2P1D) -> ViscoLaw

The dimensionless 2S2P1D creep kernel ``\\varphi`` packaged as a scalar
`:creep` [`ViscoLaw`](@ref) `(t, t') ↦ φ(t - t')`, ready for the ageing
pipeline.

This is the entry point that lets one 2S2P1D object drive **both** routes.  The
relaxation modulus in the time domain is

```math
R = E_{00}\\,\\mathbf{1} + (E_0 - E_{00})\\,\\varphi^{-1},
```

where ``\\varphi^{-1}`` is the **Volterra** inverse, not the pointwise one:

```julia
m   = Model2S2P1D(1e-7, 1000.0, 2.2, 1.945e-3, 0.22, 0.63, 50.0)
Φ   = trapezoidal_matrix(creep_kernel_law(m), times)     # n × n
R   = m.E00 * I + (m.E0 - m.E00) * volterra_inverse(Φ; block_size = 1)
```

and that discrete `R` must agree with `relaxation(m, t)` obtained by inverting
[`carson_relaxation`](@ref) — the two-route consistency check of
`test/Viscoelasticity/test_rheology.jl`.

Reproduces `VM(...)` of the ECHOES reference
`tests/python/creep/modele2S2P1D.py`.
"""
creep_kernel_law(m::Model2S2P1D) = ViscoLaw(
    (t, t_p) -> t < t_p ? zero(creep_kernel(m, zero(t))) : creep_kernel(m, t - t_p),
    :creep,
)
