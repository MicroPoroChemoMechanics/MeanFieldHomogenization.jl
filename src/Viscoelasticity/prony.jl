# =============================================================================
#  prony.jl — discrete relaxation and retardation spectra, and the exact
#  conversion between them.
#
#  A generalized Maxwell chain and a generalized Kelvin chain describe the same
#  material.  In the Laplace-Carson domain the statement is simply
#  `R*(p) J*(p) = 1`, and both transforms are rational, so passing from one to
#  the other is a partial-fraction decomposition of a rational function.
#
#  The reference implementation of that idea — `Kelvin2Maxwell.py` in ECHOES —
#  does it symbolically: expand `R*(p) Π(1 + pτ_i)` into a polynomial and hand
#  its roots to SymPy.  That is exact on paper and ill-conditioned in practice;
#  it breaks down past a dozen branches, and real relaxation spectra span six
#  to ten decades with dozens of them.
#
#  This file does it differently.  In the retardation-time variable σ = -1/p
#  the zeros of `R*` are *interlaced* with its poles — one strictly between
#  each pair of consecutive input times — because `R*` is a Stieltjes function.
#  Each unknown is therefore isolated by construction, bisection in `log σ`
#  cannot fail whatever the number of branches or their spread, and the
#  residues come out with structurally guaranteed signs.  Differentiating the
#  result is then a single Newton step in the `Dual` type, by the implicit
#  function theorem.
# =============================================================================

# ── Tolerances ──────────────────────────────────────────────────────────────

"""
    PRONY_MERGE_TOL

Relative spacing below which two branches of a Prony spectrum are considered
the same branch and merged on construction.  Also the scale below which
[`maxwell_to_kelvin`](@ref) warns that the converted spectrum, though still
mathematically exact, is ill-conditioned.
"""
const PRONY_MERGE_TOL = 1.0e-8

"""
    PRONY_FLUID_TOL

Relative size below which an equilibrium modulus (or a fluidity) counts as
zero, so the model is treated as a fluid (resp. a solid).  Compared against the
glassy modulus, so it is scale-free.
"""
const PRONY_FLUID_TOL = 1.0e-12

# ── Types ───────────────────────────────────────────────────────────────────

"""
    PronyRelaxation(E_inf, E, tau)

A **generalized Maxwell** chain: an equilibrium spring `E_inf` in parallel with
branches `(E[i], tau[i])`, each a spring and a dashpot in series.

```math
R(t) = E_\\infty + \\sum_i E_i\\,e^{-t/\\tau_i},
\\qquad
R^{*}(p) = E_\\infty + \\sum_i E_i\\,\\frac{p\\tau_i}{1 + p\\tau_i}.
```

`E_inf == 0` means a **fluid**: the stress relaxes away completely.

!!! note "Why the degenerate branches are separate fields"
    The ECHOES Python reference stores the equilibrium term inside the spectrum
    as a branch with `τ = Inf` (and, in the dual type, `τ = 0`).  That forces
    `Inf` into a vector which is then sorted, exponentiated and differentiated
    — and `Inf` is poison for `ForwardDiff`, where `Inf * 0` gives `NaN` in the
    partials.  Keeping `E_inf` as its own scalar removes every special case.

Construction sorts the branches by increasing `tau` and merges any that agree
to within `merge_tol` (relative), summing their moduli — two branches with the
same relaxation time are one branch, not an error.

See [`maxwell_to_kelvin`](@ref) for the conversion, [`prony_fit_relaxation`](@ref)
to obtain one by fitting an arbitrary transform, and [`zener_maxwell`](@ref) for
the one-branch case.
"""
struct PronyRelaxation{T} <: AbstractRheology
    E_inf::T
    E::Vector{T}
    tau::Vector{T}

    function PronyRelaxation{T}(
            E_inf, E::Vector{T}, tau::Vector{T};
            merge_tol::Real = PRONY_MERGE_TOL
        ) where {T}
        length(E) == length(tau) || throw(
            ArgumentError(
                "PronyRelaxation: E and tau must have the same length; " *
                    "got $(length(E)) and $(length(tau))"
            )
        )
        Ec, tc = _sort_and_merge(E, tau, merge_tol, "PronyRelaxation")
        return new{T}(T(E_inf), Ec, tc)
    end
end

"""
    PronyCreep(J_0, J, tau, phi = zero(J_0))

A **generalized Kelvin** chain: an instantaneous spring of compliance `J_0` in
series with branches `(J[i], tau[i])`, each a spring and a dashpot in parallel,
and — when `phi > 0` — a dashpot of fluidity `phi`.

```math
J(t) = J_0 + \\sum_i J_i\\bigl(1 - e^{-t/\\tau_i}\\bigr) + \\varphi\\,t,
\\qquad
J^{*}(p) = J_0 + \\sum_i \\frac{J_i}{1 + p\\tau_i} + \\frac{\\varphi}{p}.
```

!!! note "Fluidity, not viscosity"
    `phi = 1/η` rather than `η`, so a *solid* is `phi = 0` rather than
    `η = Inf`.  Zero is differentiable and sorts and promotes like any other
    number; `Inf` does none of those things.  It also makes the two Prony types
    exactly symmetric:

        phi == 0    ⟺  solid  ⟺  E_inf > 0
        E_inf == 0  ⟺  fluid  ⟺  phi > 0

See [`kelvin_to_maxwell`](@ref), [`prony_fit_creep`](@ref), [`burgers`](@ref)
and [`zener_kelvin`](@ref).
"""
struct PronyCreep{T} <: AbstractRheology
    J_0::T
    J::Vector{T}
    tau::Vector{T}
    phi::T

    function PronyCreep{T}(
            J_0, J::Vector{T}, tau::Vector{T}, phi;
            merge_tol::Real = PRONY_MERGE_TOL
        ) where {T}
        length(J) == length(tau) || throw(
            ArgumentError(
                "PronyCreep: J and tau must have the same length; " *
                    "got $(length(J)) and $(length(tau))"
            )
        )
        _plain_value(J_0) > 0 || throw(
            ArgumentError(
                "PronyCreep: the instantaneous compliance J_0 must be positive; got $J_0"
            )
        )
        Jc, tc = _sort_and_merge(J, tau, merge_tol, "PronyCreep")
        return new{T}(T(J_0), Jc, tc, T(phi))
    end
end

# ── Constructors: validate, sort, merge ─────────────────────────────────────

function PronyRelaxation(
        E_inf::Real, E::AbstractVector{<:Real}, tau::AbstractVector{<:Real};
        merge_tol::Real = PRONY_MERGE_TOL
    )
    T = promote_type(typeof(E_inf), eltype(E), eltype(tau), Float64)
    return PronyRelaxation{T}(T(E_inf), collect(T, E), collect(T, tau); merge_tol)
end

function PronyCreep(
        J_0::Real, J::AbstractVector{<:Real}, tau::AbstractVector{<:Real},
        phi::Real = 0.0; merge_tol::Real = PRONY_MERGE_TOL
    )
    T = promote_type(typeof(J_0), eltype(J), eltype(tau), typeof(phi), Float64)
    return PronyCreep{T}(T(J_0), collect(T, J), collect(T, tau), T(phi); merge_tol)
end

"""
    _sort_and_merge(w, tau, merge_tol, who) -> (w, tau)

Sort a spectrum by increasing time and fold together branches whose times agree
to within `merge_tol` relative, summing their weights.

Merging is the *correct* preprocessing rather than an error: two branches with
the same characteristic time are indistinguishable, and their sum is the same
material.  It also guarantees the strict interlacing the conversion relies on —
[`maxwell_to_kelvin`](@ref) needs `τ_i < τ_{i+1}` with no ties, or the brackets
it bisects in would be empty.
"""
function _sort_and_merge(w::Vector{T}, tau::Vector{T}, merge_tol::Real, who::AbstractString) where {T}
    isempty(tau) && return w, tau
    all(>(0), tau) || throw(
        ArgumentError("$who: every relaxation/retardation time must be strictly positive")
    )
    all(isfinite, tau) || throw(
        ArgumentError(
            "$who: relaxation/retardation times must be finite — the degenerate " *
                "branches live in the `E_inf` / `J_0` and `phi` fields, not in `tau`"
        )
    )
    perm = sortperm(tau; by = _plain_value)
    ws = w[perm]
    ts = tau[perm]

    wout = T[]
    tout = T[]
    for i in eachindex(ts)
        if !isempty(tout) && ts[i] - tout[end] ≤ merge_tol * tout[end]
            wout[end] += ws[i]
        else
            push!(wout, ws[i])
            push!(tout, ts[i])
        end
    end
    if any(<(0), _plain_value.(wout))
        @warn "$who: the spectrum has negative weights; the model is not passive " *
            "and `loss_factor` may come out negative." maxlog = 1
    end
    return wout, tout
end

# ── The interface ───────────────────────────────────────────────────────────

carson_relaxation(m::PronyRelaxation, p) =
    m.E_inf + sum(
    (E * p * τ / (1 + p * τ) for (E, τ) in zip(m.E, m.tau));
    init = zero(p * one(eltype(m.E)))
)

carson_creep(m::PronyCreep, p) =
    m.J_0 + m.phi / p + sum(
    (J / (1 + p * τ) for (J, τ) in zip(m.J, m.tau));
    init = zero(p * one(eltype(m.J)))
)

relaxation(m::PronyRelaxation, t::Real; method = nothing) =
    m.E_inf + sum((E * exp(-t / τ) for (E, τ) in zip(m.E, m.tau)); init = zero(t * one(eltype(m.E))))

creep(m::PronyCreep, t::Real; method = nothing) =
    m.J_0 + m.phi * t +
    sum((J * (1 - exp(-t / τ)) for (J, τ) in zip(m.J, m.tau)); init = zero(t * one(eltype(m.J))))

glassy_modulus(m::PronyRelaxation) = m.E_inf + sum(m.E; init = zero(m.E_inf))
equilibrium_modulus(m::PronyRelaxation) = m.E_inf

glassy_modulus(m::PronyCreep) = one(m.J_0) / m.J_0
equilibrium_modulus(m::PronyCreep) =
    is_fluid(m) ? zero(m.J_0) : one(m.J_0) / (m.J_0 + sum(m.J; init = zero(m.J_0)))

is_fluid(m::PronyRelaxation) = m.E_inf ≤ PRONY_FLUID_TOL * glassy_modulus(m)
is_fluid(m::PronyCreep) = m.phi > PRONY_FLUID_TOL * glassy_modulus(m) / _time_scale(m)

# The fluidity has units of 1/(modulus × time), so testing it against a modulus
# needs a time to compare with; the longest retardation time is the natural one.
_time_scale(m::PronyCreep) = isempty(m.tau) ? one(m.J_0) : last(m.tau)

# The other transform of each type is the derived one: it goes through the
# conversion rather than through a numerical inversion, so it stays exact.
carson_creep(m::PronyRelaxation, p) = carson_creep(maxwell_to_kelvin(m), p)
creep(m::PronyRelaxation, t::Real; method = nothing) = creep(maxwell_to_kelvin(m), t)
carson_relaxation(m::PronyCreep, p) = carson_relaxation(kelvin_to_maxwell(m), p)
relaxation(m::PronyCreep, t::Real; method = nothing) = relaxation(kelvin_to_maxwell(m), t)

Base.length(m::PronyRelaxation) = length(m.E)
Base.length(m::PronyCreep) = length(m.J)

function Base.show(io::IO, m::PronyRelaxation)
    return print(
        io, "PronyRelaxation(E_inf = ", m.E_inf, ", ", length(m.E),
        is_fluid(m) ? " branches, fluid)" : " branches)"
    )
end

function Base.show(io::IO, m::PronyCreep)
    return print(
        io, "PronyCreep(J_0 = ", m.J_0, ", ", length(m.J),
        is_fluid(m) ? " branches, φ = $(m.phi))" : " branches, solid)"
    )
end

# ── The transforms in retardation-time space ────────────────────────────────
#
#  σ = -1/p turns both transforms into strictly monotone functions with simple
#  poles at the input times.  Everything below rests on that.

"""
    _phi_maxwell(m::PronyRelaxation, σ)

``\\Phi(\\sigma) = R^{*}(-1/\\sigma) = E_\\infty - \\sum_j E_j\\tau_j/(\\sigma-\\tau_j)``.

Its zeros are the retardation times of the equivalent Kelvin chain.  Strictly
increasing between consecutive poles, with `Φ(0⁺) = E_glassy > 0` and
`Φ(∞) = E_∞`.
"""
_phi_maxwell(m::PronyRelaxation, σ) =
    m.E_inf - sum((E * τ / (σ - τ) for (E, τ) in zip(m.E, m.tau)); init = zero(σ * one(eltype(m.E))))

"""
    _dphi_maxwell(m::PronyRelaxation, σ)

``\\Phi'(\\sigma) = \\sum_j E_j\\tau_j/(\\sigma-\\tau_j)^2``, strictly positive
for a positive spectrum — which is why the converted compliances
`J_j = 1/(σ_j Φ'(σ_j))` come out positive by construction.
"""
_dphi_maxwell(m::PronyRelaxation, σ) =
    sum((E * τ / (σ - τ)^2 for (E, τ) in zip(m.E, m.tau)); init = zero(σ * one(eltype(m.E))))

"""
    _psi_kelvin(m::PronyCreep, σ)

``\\Psi(\\sigma) = J^{*}(-1/\\sigma)
   = J_0 + \\sum_j J_j\\sigma/(\\sigma-\\tau_j) - \\varphi\\sigma``.

Its zeros are the relaxation times of the equivalent Maxwell chain.  Strictly
*decreasing* between consecutive poles, with `Ψ(0⁺) = J_0 > 0`.
"""
_psi_kelvin(m::PronyCreep, σ) =
    m.J_0 - m.phi * σ +
    sum((J * σ / (σ - τ) for (J, τ) in zip(m.J, m.tau)); init = zero(σ * one(eltype(m.J))))

"""
    _dpsi_kelvin(m::PronyCreep, σ)

``\\Psi'(\\sigma) = -\\sum_j J_j\\tau_j/(\\sigma-\\tau_j)^2 - \\varphi``,
strictly negative — hence `E_j = -1/(σ_j Ψ'(σ_j)) > 0`.
"""
_dpsi_kelvin(m::PronyCreep, σ) =
    -m.phi - sum((J * τ / (σ - τ)^2 for (J, τ) in zip(m.J, m.tau)); init = zero(σ * one(eltype(m.J))))

# ── Bracketed root finding ──────────────────────────────────────────────────

"""
    _bracketed_root(f, lo, hi; rtol = 1e-14, maxiter = 200) -> Float64

Bisection in `log σ` on a bracket known to contain exactly one simple root.

Logarithmic rather than linear because relaxation spectra span six to ten
decades: the geometric midpoint `√(lo·hi)` halves the *relative* interval, so
the iteration count depends on the requested relative accuracy and not at all
on where in the spectrum the root sits.

The caller guarantees the sign change; a violation is a bug in the bracketing
tables, not a user error, so it throws.
"""
function _bracketed_root(f, lo::Real, hi::Real; rtol::Real = 1.0e-14, maxiter::Int = 200)
    a, b = float(lo), float(hi)
    fa, fb = f(a), f(b)
    (isfinite(fa) && isfinite(fb)) || throw(
        ArgumentError("_bracketed_root: the transform is not finite at a bracket endpoint")
    )
    signbit(fa) == signbit(fb) && throw(
        ArgumentError(
            "_bracketed_root: no sign change on [$a, $b] (f = $fa, $fb) — " *
                "the interlacing bracket is wrong"
        )
    )
    for _ in 1:maxiter
        (b - a) ≤ rtol * a && break
        c = sqrt(a * b)                       # geometric midpoint = log midpoint
        (c ≤ a || c ≥ b) && break             # exhausted the representable range
        fc = f(c)
        iszero(fc) && return c
        if signbit(fc) == signbit(fa)
            a, fa = c, fc
        else
            b, fb = c, fc
        end
    end
    return sqrt(a * b)
end

"""
    _expand_bracket(f, lo, growth, maxsteps) -> hi

Find an upper end for a root known to lie somewhere in `(lo, ∞)` by repeatedly
multiplying by `growth` until the sign flips.  Used for the outermost interval,
which is unbounded whenever the model has a degenerate branch at the slow end.
"""
function _expand_bracket(f, lo::Real, growth::Real = 4.0, maxsteps::Int = 200)
    flo = f(lo)
    hi = float(lo)
    for _ in 1:maxsteps
        hi *= growth
        signbit(f(hi)) == signbit(flo) || return hi
    end
    throw(
        ArgumentError(
            "_expand_bracket: no sign change found out to σ = $hi; " *
                "the spectrum may not be a valid Stieltjes function"
        )
    )
end

"""
    _shrink_bracket(f, hi, shrink, maxsteps) -> lo

The mirror image of [`_expand_bracket`](@ref) for a root in `(0, hi)`.
"""
function _shrink_bracket(f, hi::Real, shrink::Real = 4.0, maxsteps::Int = 200)
    fhi = f(hi)
    lo = float(hi)
    for _ in 1:maxsteps
        lo /= shrink
        signbit(f(lo)) == signbit(fhi) || return lo
    end
    throw(
        ArgumentError(
            "_shrink_bracket: no sign change found down to σ = $lo; " *
                "the spectrum may not be a valid Stieltjes function"
        )
    )
end

"""
    _root_with_ad(f, df, lo, hi) -> σ

Locate the root of `f` in `[lo, hi]` and hand back a value that carries the
correct `ForwardDiff` partials.

The bisection runs on values only — an iteration count is discrete, and
differentiating through it is meaningless.  A **single Newton step taken in the
full `Dual` type** then supplies the derivatives: since `f(σ_val) ≈ 0` at the
value level, `σ - f(σ)/f'(σ)` leaves the value alone and sets the partials to

```math
\\frac{\\partial\\sigma}{\\partial\\theta}
  = -\\frac{\\partial_\\theta f}{\\partial_\\sigma f},
```

which is exactly the implicit function theorem for a simple root.  No nested
duals, and no differentiation of the solver.
"""
function _root_with_ad(f, df, lo::Real, hi::Real; rtol::Real = 1.0e-14)
    σ = _bracketed_root(σ -> _plain_value(f(σ)), _plain_value(lo), _plain_value(hi); rtol)
    return σ - f(σ) / df(σ)
end

# The poles must never be evaluated *at*; step inside by a relative sliver.
const _POLE_OFFSET = 1.0e-12

_just_above(τ) = τ * (1 + _POLE_OFFSET)
_just_below(τ) = τ * (1 - _POLE_OFFSET)

# ── Maxwell → Kelvin ────────────────────────────────────────────────────────

"""
    maxwell_to_kelvin(m::PronyRelaxation; rtol = 1e-14) -> PronyCreep

Convert a generalized Maxwell chain into the **exactly equivalent** generalized
Kelvin chain, so that `J*(p) R*(p) = 1` identically.

# How

In `σ = -1/p`, the transform becomes
``\\Phi(\\sigma) = E_\\infty - \\sum_j E_j\\tau_j/(\\sigma-\\tau_j)``, which is
strictly increasing between consecutive poles and runs from `-∞` to `+∞` across
every gap.  The retardation times are its zeros, and they **interlace** the
relaxation times:

```
     0 <  τ₁  <  σ₁  <  τ₂  <  σ₂  < … <  τ_m  < [σ_m]
```

with the last root present exactly when `E_∞ > 0`.  There is deliberately no
root in `(0, τ₁)`: `Φ(0⁺) = E_glassy > 0` and `Φ` only increases from there —
the bracketing is *not* symmetric with [`kelvin_to_maxwell`](@ref), which does
have a root below its first pole.

Each root is thus isolated before any arithmetic happens, and bisection in
`log σ` finds it whatever the number of branches or their spread.  The residues
follow from `R*'(-1/σ) = σ²Φ'(σ)`:

```math
J_0 = \\frac{1}{E_\\infty + \\sum_i E_i},
\\qquad
J_j = \\frac{1}{\\sigma_j\\,\\Phi'(\\sigma_j)},
\\qquad
\\tau^{K}_j = \\sigma_j ,
```

and `Φ' > 0` makes every `J_j` positive by construction rather than by luck.

# Fluids

When `E_∞ = 0` the outermost root recedes to infinity: that *is* the series
dashpot.  It is picked up exactly, as
``\\varphi = 1/\\sum_i E_i\\tau_i = 1/R^{*\\prime}(0)``, and the result has one
branch fewer than the input.

# Differentiability

`ForwardDiff` traverses the conversion: the roots are lifted by the implicit
function theorem (see [`_root_with_ad`](@ref)), so gradients of `J_j` and
`τ^K_j` with respect to `(E_∞, E, τ)` are available.

# Examples

```jldoctest
julia> using MeanFieldHomogenization

julia> m = PronyRelaxation(8.0, [3.0, 17.0], [12.0, 23.0]);

julia> k = maxwell_to_kelvin(m);

julia> isapprox(carson_creep(k, 0.37) * carson_relaxation(m, 0.37), 1.0; rtol = 1e-12)
true
```

See also [`kelvin_to_maxwell`](@ref), [`prony_fit_creep`](@ref).
"""
function maxwell_to_kelvin(m::PronyRelaxation{T}; rtol::Real = 1.0e-14) where {T}
    n = length(m.E)
    n == 0 && return PronyCreep(one(T) / m.E_inf, T[], T[], zero(T))

    Φ = σ -> _phi_maxwell(m, σ)
    dΦ = σ -> _dphi_maxwell(m, σ)
    fluid = is_fluid(m)

    _warn_ill_conditioned(m.tau, "maxwell_to_kelvin")

    σs = T[]
    for j in 1:(n - 1)                       # one root strictly inside every gap
        push!(
            σs,
            _root_with_ad(Φ, dΦ, _just_above(m.tau[j]), _just_below(m.tau[j + 1]); rtol)
        )
    end
    if !fluid                                 # the outermost root is finite
        lo = _just_above(m.tau[n])
        hi = _expand_bracket(σ -> _plain_value(Φ(σ)), _plain_value(lo))
        push!(σs, _root_with_ad(Φ, dΦ, lo, hi; rtol))
    end

    J_0 = one(T) / glassy_modulus(m)
    Js = [one(T) / (σ * dΦ(σ)) for σ in σs]
    # η = R*'(0) = Σ E_i τ_i exactly; for a solid the fluidity is zero.
    φ = fluid ? one(T) / sum(E * τ for (E, τ) in zip(m.E, m.tau)) : zero(T)

    _assert_positive(Js, "maxwell_to_kelvin", "compliance")
    return PronyCreep(J_0, Js, σs, φ)
end

# ── Kelvin → Maxwell ────────────────────────────────────────────────────────

"""
    kelvin_to_maxwell(k::PronyCreep; rtol = 1e-14) -> PronyRelaxation

Convert a generalized Kelvin chain into the **exactly equivalent** generalized
Maxwell chain, so that `R*(p) J*(p) = 1` identically.  The inverse of
[`maxwell_to_kelvin`](@ref), and the operation `Kelvin2Maxwell.py` performs
symbolically in ECHOES.

# How

In `σ = -1/p`, ``\\Psi(\\sigma) = J_0 + \\sum_j J_j\\sigma/(\\sigma-\\tau_j) -
\\varphi\\sigma`` is strictly *decreasing* between consecutive poles.  Its zeros
are the relaxation times, and here — unlike the other direction — there **is**
one below the first pole, because `Ψ(0⁺) = J_0 > 0` while `Ψ(τ₁⁻) = -∞`:

```
     0 <  σ₁  <  τ₁  <  σ₂  <  τ₂  < … <  τ_n  < [σ_{n+1}]
```

the last root existing exactly when `φ > 0`.  So a **solid** Kelvin chain of
`n` branches gives `n` Maxwell branches with `E_∞ > 0`, and a **fluid** one
gives `n+1` branches with `E_∞ = 0` — which is the degree count of the rational
transform, as it should be.

The residues, from `J*'(-1/σ) = σ²Ψ'(σ)`:

```math
E_{\\rm glassy} = \\frac{1}{J_0},
\\qquad
E_j = -\\frac{1}{\\sigma_j\\,\\Psi'(\\sigma_j)},
\\qquad
E_\\infty = E_{\\rm glassy} - \\sum_j E_j ,
```

with `Ψ' < 0` making every `E_j` positive by construction.

For a solid the result satisfies a free consistency identity — both sides are
`R*(0)`:

```math
\\frac{1}{J_0} - \\sum_j E_j \\;=\\; \\frac{1}{J_0 + \\sum_i J_i},
```

which is checked internally and would catch any sign slip in the residues.

# Examples

The Burgers model is a fluid Kelvin chain with one branch, so it must convert
to a two-branch Maxwell chain with no equilibrium spring:

```jldoctest
julia> using MeanFieldHomogenization

julia> b = burgers(1.0, 3.0, 2.0, 6.0);

julia> r = kelvin_to_maxwell(b);

julia> length(r), isapprox(equilibrium_modulus(r), 0.0; atol = 1e-12)
(2, true)
```

See also [`maxwell_to_kelvin`](@ref), [`prony_fit_relaxation`](@ref).
"""
function kelvin_to_maxwell(k::PronyCreep{T}; rtol::Real = 1.0e-14) where {T}
    n = length(k.J)
    fluid = is_fluid(k)
    E_glassy = one(T) / k.J_0

    if n == 0
        return fluid ?
            PronyRelaxation(zero(T), [E_glassy], [k.J_0 / k.phi]) :
            PronyRelaxation(E_glassy, T[], T[])
    end

    Ψ = σ -> _psi_kelvin(k, σ)
    dΨ = σ -> _dpsi_kelvin(k, σ)

    _warn_ill_conditioned(k.tau, "kelvin_to_maxwell")

    σs = T[]
    hi1 = _just_below(k.tau[1])               # the root below the first pole
    lo1 = _shrink_bracket(σ -> _plain_value(Ψ(σ)), _plain_value(hi1))
    push!(σs, _root_with_ad(Ψ, dΨ, lo1, hi1; rtol))
    for j in 1:(n - 1)
        push!(
            σs,
            _root_with_ad(Ψ, dΨ, _just_above(k.tau[j]), _just_below(k.tau[j + 1]); rtol)
        )
    end
    if fluid                                  # the extra branch of a fluid
        lo = _just_above(k.tau[n])
        hi = _expand_bracket(σ -> _plain_value(Ψ(σ)), _plain_value(lo))
        push!(σs, _root_with_ad(Ψ, dΨ, lo, hi; rtol))
    end

    Es = [-one(T) / (σ * dΨ(σ)) for σ in σs]
    _assert_positive(Es, "kelvin_to_maxwell", "modulus")

    E_inf = fluid ? zero(T) : E_glassy - sum(Es)
    if !fluid
        expected = one(T) / (k.J_0 + sum(k.J))
        isapprox(_plain_value(E_inf), _plain_value(expected); rtol = 1.0e-8) || @warn """
        kelvin_to_maxwell: the equilibrium modulus from the residues \
        ($(_plain_value(E_inf))) disagrees with 1/J*(0) ($(_plain_value(expected))). \
        The input spectrum is probably not a valid Stieltjes function.""" maxlog = 1
    end
    return PronyRelaxation(E_inf, Es, σs)
end

# ── Shared diagnostics ──────────────────────────────────────────────────────

function _warn_ill_conditioned(tau, who::AbstractString)
    for j in 1:(length(tau) - 1)
        lo, hi = _plain_value(tau[j]), _plain_value(tau[j + 1])
        if hi - lo ≤ 1.0e3 * PRONY_MERGE_TOL * lo
            @warn """
            $who: the times $lo and $hi are nearly equal. The conversion stays \
            exact, but the resulting spectrum is ill-conditioned — its \
            coefficients will be large and nearly cancelling. Merge the two \
            branches (a larger `merge_tol`) if they are meant to be one.""" maxlog = 1
            return nothing
        end
    end
    return nothing
end

function _assert_positive(v, who::AbstractString, what::AbstractString)
    for x in v
        _plain_value(x) > 0 || throw(
            ErrorException(
                "$who: obtained a non-positive $what ($(_plain_value(x))). " *
                    "The sign is structural for a valid spectrum, so this means a " *
                    "root was located in the wrong bracket — please report it."
            )
        )
    end
    return nothing
end

# ── Collocation: fitting a Prony series to an arbitrary transform ───────────
#
#  This is the general bridge from "some Laplace-Carson transform" to "a model
#  the whole library understands".  Fit once, and the result has a closed-form
#  time function, an exact conversion to the dual chain, a `ViscoLaw` for the
#  ageing pipeline, and derivatives — none of which an opaque `p -> F(p)` has.
#  It is the Schapery collocation method, and the ECHOES counterpart is
#  `collocationR` / `collocationF` in `tests/python/creep/modele2S2P1D.py`.

"""
    prony_fit_relaxation(Rstar, taus; points = nothing, E_inf = nothing,
                         nonneg = true, lambda = 0.0) -> PronyRelaxation

Fit a generalized Maxwell chain with the prescribed relaxation times `taus` to
an arbitrary Laplace-Carson relaxation transform `Rstar`, by least squares on a
set of collocation points.

The model is linear in its unknowns,

```math
R^{*}(p_u) - E_\\infty \\;\\approx\\; \\sum_i X_i\\,\\frac{p_u\\tau_i}{1+p_u\\tau_i},
```

so the fit is one linear solve — no iteration, no starting guess.

  * `points` — the Carson variables to collocate at.  Defaults to
    `2 * length(taus)` values log-spaced over `[1/(10 τ_max), 10/τ_min]`, which
    brackets the spectrum by a decade on each side.
  * `E_inf` — the equilibrium modulus.  Defaults to `Rstar` evaluated at
    `1/(100 τ_max)`.

    !!! note
        The ECHOES reference uses `rstar(1e-100)` for this.  That overflows
        `p^{-k}` for any fractional model — 2S2P1D included — so the default
        here is tied to the spectrum's own slowest time instead.

  * `nonneg` — constrain `X_i ≥ 0` (the default).  A non-negative spectrum is
    what makes the fitted function completely monotone, hence passive; an
    unconstrained fit routinely produces small negative moduli that give a
    negative [`loss_factor`](@ref) somewhere.
  * `lambda` — Tikhonov weight, useful when the times are packed more densely
    than the data can resolve.

!!! warning "Not differentiable"
    With `nonneg = true` the solve is an active-set method: the returned
    coefficients are a piecewise-smooth function of the input with a
    combinatorial switch in the middle.  Treat a fit as a **calibration step**
    that produces a model, and differentiate the model — not the fit.
    An unconstrained fit (`nonneg = false`) is a plain linear solve and *is*
    differentiable.

See [`prony_fit_creep`](@ref) for the dual, and [`maxwell_to_kelvin`](@ref) for
the exact conversion once a chain is in hand.
"""
function prony_fit_relaxation(
        Rstar, taus::AbstractVector{<:Real};
        points::Union{Nothing, AbstractVector} = nothing,
        E_inf::Union{Nothing, Real} = nothing,
        nonneg::Bool = true,
        lambda::Real = 0.0
    )
    τ = sort(collect(float.(taus)))
    ps = points === nothing ? _default_collocation(τ) : collect(points)
    E∞ = E_inf === nothing ? real(Rstar(1 / (100 * last(τ)))) : E_inf
    A = [real(p * t / (1 + p * t)) for p in ps, t in τ]
    b = [real(Rstar(p)) - E∞ for p in ps]
    X = _lsq(A, b, nonneg, lambda)
    return PronyRelaxation(E∞, X, τ)
end

"""
    prony_fit_creep(Jstar, taus; points = nothing, J_0 = nothing, phi = 0.0,
                    nonneg = true, lambda = 0.0) -> PronyCreep

The creep-side twin of [`prony_fit_relaxation`](@ref): fit a generalized Kelvin
chain to a Laplace-Carson creep transform `Jstar`,

```math
J^{*}(p_u) - J_0 - \\frac{\\varphi}{p_u}
    \\;\\approx\\; \\sum_i \\frac{X_i}{1 + p_u\\tau_i}.
```

`J_0` defaults to `Jstar` at `100/τ_min`, i.e. well above the fastest
retardation time.

`phi` is **not** fitted: a `φ/p` pole is a qualitatively different object from
the rest of the sum, and reading it off noisy samples is unreliable.  Pass it
explicitly when the material is a fluid — it is `1/η` of the series dashpot,
and for a transform known in closed form it is `lim_{p→0} p J*(p)`.
"""
function prony_fit_creep(
        Jstar, taus::AbstractVector{<:Real};
        points::Union{Nothing, AbstractVector} = nothing,
        J_0::Union{Nothing, Real} = nothing,
        phi::Real = 0.0,
        nonneg::Bool = true,
        lambda::Real = 0.0
    )
    τ = sort(collect(float.(taus)))
    ps = points === nothing ? _default_collocation(τ) : collect(points)
    J0 = J_0 === nothing ? real(Jstar(100 / first(τ))) : J_0
    A = [real(1 / (1 + p * t)) for p in ps, t in τ]
    b = [real(Jstar(p)) - J0 - real(phi / p) for p in ps]
    X = _lsq(A, b, nonneg, lambda)
    return PronyCreep(J0, X, τ, phi)
end

"""
    _default_collocation(taus) -> Vector{Float64}

`2 length(taus)` Carson variables log-spaced over `[1/(10 τ_max), 10/τ_min]`.

Real and positive, so the transform is only ever sampled where every model in
the catalog is defined — including the fractional ones, whose `p^{-k}` needs
a branch choice off the positive real axis.
"""
function _default_collocation(taus::AbstractVector{<:Real})
    lo = 1 / (10 * last(taus))
    hi = 10 / first(taus)
    n = 2 * length(taus)
    return exp10.(range(log10(lo), log10(hi); length = n))
end

"""
    _lsq(A, b, nonneg, lambda) -> x

Least-squares solve of `A x ≈ b`, optionally Tikhonov-regularized by `lambda`
and optionally constrained to `x ≥ 0`.
"""
function _lsq(A::AbstractMatrix, b::AbstractVector, nonneg::Bool, lambda::Real)
    if iszero(lambda)
        return nonneg ? _nnls(A, b) : A \ b
    end
    n = size(A, 2)
    Aa = vcat(A, sqrt(lambda) * Matrix{eltype(A)}(LinearAlgebra.I, n, n))
    ba = vcat(b, zeros(eltype(b), n))
    return nonneg ? _nnls(Aa, ba) : Aa \ ba
end

"""
    _nnls(A, b; maxiter, tol) -> x

Lawson-Hanson non-negative least squares: minimize `‖Ax - b‖₂` subject to
`x ≥ 0`.

Implemented here rather than pulled in as a dependency — it is forty lines, and
the alternative in the ECHOES reference is an `nlopt` call.  Being an
active-set method it is not differentiable; see the warning on
[`prony_fit_relaxation`](@ref).
"""
function _nnls(
        A::AbstractMatrix{T}, b::AbstractVector{T};
        maxiter::Int = 3 * size(A, 2), tol::Real = 1.0e-12
    ) where {T <: Real}
    n = size(A, 2)
    x = zeros(T, n)
    passive = falses(n)
    w = A' * (b - A * x)
    scale = maximum(abs, w; init = one(T))

    for _ in 1:maxiter
        cand = findall(j -> !passive[j] && w[j] > tol * scale, 1:n)
        isempty(cand) && break
        j = cand[argmax(w[cand])]
        passive[j] = true

        for _ in 1:(3n)                       # inner feasibility loop
            idx = findall(passive)
            s = zeros(T, n)
            s[idx] = A[:, idx] \ b
            all(>(0), s[idx]) && (x = s; break)
            α = minimum(
                x[k] / (x[k] - s[k]) for k in idx if s[k] ≤ 0 && x[k] != s[k];
                init = one(T)
            )
            x = x + α * (s - x)
            for k in idx
                abs(x[k]) ≤ tol * scale && (passive[k] = false; x[k] = zero(T))
            end
        end
        w = A' * (b - A * x)
    end
    return x
end
