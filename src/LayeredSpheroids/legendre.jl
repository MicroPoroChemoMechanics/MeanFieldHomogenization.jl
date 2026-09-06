# =============================================================================
#  legendre.jl — associated Legendre functions of order m = 0, 1, first
#  (P) and second (Q) kind, by upward three-term recurrence.
#
#  Faithful transliteration of the seed values and recurrences used in
#  Barthélémy & Bignonnet (IJES 2020, eq:Legformc/d, eq:Leg2formc/d).
#
#  Two arguments are used throughout the spheroid solution:
#    - the "p" argument, real, |p| ≤ 1 (the angular / polar coordinate);
#    - the "q" argument, |q| > 1 for prolate (real) or purely imaginary
#      `q = iτ` for oblate (`τ` real) — carried generically as `Q<:Number`
#      (`T` for prolate, `Complex{T}` for oblate). No branch is taken on
#      the caller's side: `sqrt`, `atanh` dispatch to the right method
#      once `x` has the right static type (`Complex` for oblate).
#
#  Only ODD degrees 1, 3, …, 2𝒩−1 ever enter the axial/transverse
#  spheroid problem (by symmetry, see eq:Taxi/eq:Ttrans); the table
#  builders below return exactly those `𝒩` values (and derivatives),
#  never materializing the unused even-degree entries.
#
#  Stability: P upward, Q DOWNWARD
#  ------------------------------
#  The three-term recurrence has two solutions, `Pₙ` growing with `n` and
#  `Qₙ` decaying — `|Qₙ/Pₙ| ~ ρ^{-(2n+1)}` with `ρ = |x + √(x²−1)| > 1`.
#  Running it **upward** follows the dominant solution, so it is stable for
#  `P` and catastrophically unstable for `Q`: the rounding error of the seed
#  contains a `P` component that is amplified by `ρ^{2n}`.  Measured against
#  `BigFloat`, an upward `Q₁₅(5)` was wrong by a relative `1.3e13`, and
#  `Q₁₅(50)` by `7.9e26` — and the error grows with the truncation order, so
#  raising `Nseries` to gain accuracy lost it instead.
#
#  `Q` is therefore obtained by **Miller's algorithm**: the recurrence is run
#  DOWNWARD from a degree well above the one requested, seeded arbitrarily,
#  which suppresses the unwanted `P` component by `ρ^{-2k}` over `k` steps;
#  the arbitrary overall factor is then fixed on a low degree whose closed
#  form is exact.  Derivatives come from the exact identity
#
#      (x² − 1) dQₙᵐ/dx = n x Qₙᵐ − (n + m) Qₙ₋₁ᵐ,
#
#  verified here against `BigFloat` to `1e-100`, rather than from a second
#  unstable recurrence.
# =============================================================================

@inline _arccoth(x) = atanh(one(x) / x)

@inline function _legendre_next(n, m, x, Pnm, Pnm1)
    return ((2n + 1) * x * Pnm - (n + m) * Pnm1) / (n - m + 1)
end

@inline function _legendre_next_der(n, m, x, Pnm, dPnm, dPnm1)
    return ((2n + 1) * (Pnm + x * dPnm) - (n + m) * dPnm1) / (n - m + 1)
end

"""
    _legendre_grow!(tab, dtab, Nmax, m, x)

Grow the value/derivative tables `tab`, `dtab` (1-indexed, `tab[k+1]` =
degree-`k` value) up to degree `Nmax` (inclusive) by the upward
recurrence [`_legendre_next`](@ref) / [`_legendre_next_der`](@ref),
order `m`. `tab` and `dtab` must already hold their required seed
degrees (2 seeds for the standard case, 3 for [`_Q1_table`](@ref)'s
special low-degree closed forms).
"""
function _legendre_grow!(tab::Vector{Tx}, dtab::Vector{Tx}, Nmax::Int, m::Int, x) where {Tx}
    n = length(tab) - 1
    while n < Nmax
        Pn = tab[n + 1]
        Pnm1 = tab[n]
        Pnext = _legendre_next(n, m, x, Pn, Pnm1)
        push!(tab, Pnext)
        if length(dtab) == length(tab) - 1
            dPn = dtab[n + 1]
            dPnm1 = dtab[n]
            push!(dtab, _legendre_next_der(n, m, x, Pn, dPn, dPnm1))
        end
        n += 1
    end
    return tab, dtab
end

"""
    _q_recurrence_plan(x, Nmax, Tx) -> (:upward | :downward, k)

Which direction to run the `Q` recurrence, and how many extra degrees Miller
needs when it is the downward one.

A single quantity decides, and it decides both ways:

    ρ = |x + √(x²−1)| > 1,    |Qₙ/Pₙ| ~ ρ^{-(2n+1)}.

Going **up**, the seed's rounding error picks up the dominant `P` solution and
is amplified by `ρ^{2n}`; over the `Nmax` degrees needed that costs
`2 Nmax log ρ` nats of precision.  Going **down**, Miller suppresses that same
`P` component by `ρ^{-2k}` over `k` steps, so it needs
`k ≈ log(1/ε) / (2 log ρ)`.

The two are reciprocal in `log ρ`, so exactly one of them is always cheap:

- `ρ` near 1 — a nearly degenerate spheroid, `Q` barely decaying — upward
  loses almost nothing, while Miller would need thousands of steps;
- `ρ` well above 1 — upward is hopeless, and Miller converges in a handful.

Getting this backwards is not a small error: capping Miller's step count and
using it anyway at `ρ = 1.017` (a 1:60 flat disc) silently returned a
`Q` accurate to only `2e-7`.
"""
function _q_recurrence_plan(x, Nmax::Int, ::Type{Tx}) where {Tx}
    # Miller's downward recurrence exists for one reason: to keep floating-point
    # cancellation from swamping the minimal solution. An exact element type has
    # no cancellation to control, and the upward recurrence — the plain
    # three-term identity — is then exact. It is also the only branch a symbolic
    # type can take at all: `eps`, `isfinite` and `ceil(Int, ·)` are all
    # meaningless on a `Sym`.
    #
    # The predicate is on `real(Tx)`, and neither `Tx` nor `float(Tx)` will do:
    #
    #   * an oblate spheroid carries `q = iτ`, so `Tx` is `Complex{Float64}`,
    #     which is NOT hard numeric — testing `Tx` itself sends every oblate
    #     case down the upward branch and hands back the `7.9e26` errors this
    #     recurrence plan exists to prevent;
    #   * `float(Sym) === Float64` (Base derives it from `typeof(float(zero(T)))`,
    #     and SymPy answers with a `Float64`), so `float(Tx)` reports a symbolic
    #     type as numeric and lets `ceil(Int, ·)` throw further down.
    #
    # `real` separates the two cleanly: `real(Complex{Float64}) === Float64`
    # while `real(Sym) === Sym`.
    is_hard_numeric(real(Tx)) || return (:upward, 0)
    ε = eps(real(float(Tx)))
    budget = log(1 / ε)                       # nats of precision available
    ρ = abs(x + sqrt(x^2 - one(x)))
    (isfinite(ρ) && ρ > 1) || return (:upward, 0)
    lρ = log(ρ)

    # Upward amplifies the seed's rounding error by `ρ^{2 Nmax}`; keep it when
    # that costs at most a factor 100, i.e. two digits.
    loss = 2 * Nmax * lρ
    loss ≤ log(100) && return (:upward, 0)

    k = ceil(Int, budget / (2 * lρ)) + 5
    k_max = 10 * Nmax + 100
    k ≤ k_max && return (:downward, k)

    # Unreachable for any `ρ` that makes upward unsafe, and provably so:
    # `k > k_max` forces `lρ < budget / (2 k_max)`, hence
    # `loss = 2 Nmax lρ < Nmax·budget / (10 Nmax + 100) < budget/10` — under
    # four nats, a factor of about 40.  So whenever Miller would be too
    # expensive, upward is accurate.
    return (:upward, 0)
end

"""
    _Q_values_miller(x, Nmax, m, n_norm, q_norm) -> Vector

`Qₙᵐ(x)` for `n = 0, …, Nmax` by Miller's downward recurrence, normalized so
that degree `n_norm` equals the exact closed form `q_norm`.

The recurrence is [`_legendre_next`](@ref) solved for the lower neighbor,

    Qₙ₋₁ = ((2n+1) x Qₙ − (n−m+1) Qₙ₊₁) / (n+m),

started at `Nmax + _miller_extra` with `(Qₙ₊₁, Qₙ) = (0, 1)`.  Going down,
`Q` grows like `ρⁿ`, so the running values are rescaled whenever they get
large; a global factor is irrelevant, the final normalization removes it.
"""
function _Q_values_miller(x::Tx, Nmax::Int, m::Int, n_norm::Int, q_norm::Tx, k::Int) where {Tx}
    N_start = Nmax + k
    tab = zeros(Tx, N_start + 2)
    tab[N_start + 2] = zero(Tx)
    tab[N_start + 1] = one(Tx)
    big_thresh = convert(real(float(Tx)), 1.0e100)
    for n in N_start:-1:1
        tab[n] = ((2n + 1) * x * tab[n + 1] - (n - m + 1) * tab[n + 2]) / (n + m)
        if abs(tab[n]) > big_thresh
            # Rescale everything computed so far; the tail above is discarded.
            inv_s = one(Tx) / tab[n]
            @inbounds for j in n:(N_start + 2)
                tab[j] *= inv_s
            end
        end
    end
    scale = q_norm / tab[n_norm + 1]
    return [tab[j] * scale for j in 1:(Nmax + 1)]
end

"""
    _Q_derivatives(tab, x, m, dtab_low) -> Vector

Derivatives of `Qₙᵐ` from the exact identity
`(x² − 1) dQₙᵐ/dx = n x Qₙᵐ − (n + m) Qₙ₋₁ᵐ`, with the low degrees for which
that identity is unusable taken from the closed forms in `dtab_low`.

For `m = 1` the identity needs `Q₀¹`, which the tables here carry as a
placeholder zero (the `m = 1` recurrence is singular at `n = 0` and that entry
is never used), so degrees `0` and `1` come from `dtab_low`; for `m = 0` only
degree `0` does.
"""
function _Q_derivatives(tab::Vector{Tx}, x::Tx, m::Int, dtab_low::Vector{Tx}) where {Tx}
    Nmax = length(tab) - 1
    x2m1 = x^2 - one(Tx)
    n_low = length(dtab_low) - 1
    dtab = Vector{Tx}(undef, Nmax + 1)
    @inbounds for n in 0:Nmax
        dtab[n + 1] = n ≤ n_low ? dtab_low[n + 1] :
            (n * x * tab[n + 1] - (n + m) * tab[n]) / x2m1
    end
    return dtab
end

"""
    _P0_table(x, Nmax) -> (tab, dtab)

`Pₙ(x)`, `n = 0, …, Nmax`, plain Legendre polynomials (order `m = 0`).
Valid for any argument (the `p` branch, `|p| ≤ 1`, or the `q` branch).
"""
function _P0_table(x::Tx, Nmax::Int) where {Tx}
    tab = Tx[one(Tx), x]
    dtab = Tx[zero(Tx), one(Tx)]
    return _legendre_grow!(tab, dtab, Nmax, 0, x)
end

"""
    _Q0_table(x, Nmax) -> (tab, dtab)

`Qₙ(x)`, `n = 0, …, Nmax`, Legendre functions of the second kind
(order `m = 0`). Valid for the `q` branch (`|x| > 1`, real or the
oblate `iτ` substitute).
"""
function _Q0_table(x::Tx, Nmax::Int) where {Tx}
    ax = _arccoth(x)
    x2m1 = x^2 - one(Tx)
    Nmax ≤ 1 && return (
        Tx[ax, x * ax - one(Tx)][1:(Nmax + 1)],
        Tx[-one(Tx) / x2m1, ax - x / x2m1][1:(Nmax + 1)],
    )
    dir, k = _q_recurrence_plan(x, Nmax, Tx)
    if dir === :upward
        tab = Tx[ax, x * ax - one(Tx)]
        dtab = Tx[-one(Tx) / x2m1, ax - x / x2m1]
        return _legendre_grow!(tab, dtab, Nmax, 0, x)
    end
    tab = _Q_values_miller(x, Nmax, 0, 0, ax, k)      # normalized on Q₀ = arccoth x
    dtab = _Q_derivatives(tab, x, 0, Tx[-one(Tx) / x2m1])
    return tab, dtab
end

"""
    _P1p_table(x, Nmax) -> (tab, dtab)

`Pₙ¹(x)`, `n = 0, …, Nmax`, associated Legendre of the first kind,
order `m = 1`, on the `p` branch (`|p| ≤ 1`), seeded with
`P₁¹(p) = -√(1-p²)`.
"""
function _P1p_table(x::Tx, Nmax::Int) where {Tx}
    xb = -sqrt(one(Tx) - x^2)
    tab = Tx[zero(Tx), xb]
    dtab = Tx[zero(Tx), -x / xb]
    return _legendre_grow!(tab, dtab, Nmax, 1, x)
end

"""
    _P1_table(x, Nmax) -> (tab, dtab)

`Pₙ¹(x)`, `n = 0, …, Nmax`, associated Legendre of the first kind,
order `m = 1`, on the `q` branch (`|x| > 1`), seeded with
`P₁¹(q) = √(q²-1)`.
"""
function _P1_table(x::Tx, Nmax::Int) where {Tx}
    xb = sqrt(x^2 - one(Tx))
    tab = Tx[zero(Tx), xb]
    dtab = Tx[zero(Tx), x / xb]
    return _legendre_grow!(tab, dtab, Nmax, 1, x)
end

"""
    _Q1_table(x, Nmax) -> (tab, dtab)

`Qₙ¹(x)`, `n = 0, …, Nmax`, associated Legendre of the second kind,
order `m = 1`, on the `q` branch. The recurrence for `m = 1` is
singular at `n = 0`, so the degrees `0, 1, 2` are seeded from closed
forms and the upward recurrence resumes from `n = 2`.
"""
function _Q1_table(x::Tx, Nmax::Int) where {Tx}
    ax = _arccoth(x)
    xb = sqrt(x^2 - one(Tx))
    x2 = x^2
    x2m1 = x2 - one(Tx)
    tab_low = Tx[
        zero(Tx),
        xb * ax - x / xb,
        x * xb * (3 * ax - (3 * x2 - 2) / (x * x2m1)),
    ]
    dtab_low = Tx[
        zero(Tx),
        x / xb * (ax + (2 - x2) / (x * x2m1)),
        (2 * x2 - 1) / xb * (3 * ax - x * (6 * x2 - 7) / ((2 * x2 - 1) * x2m1)),
    ]
    Nmax ≤ 2 && return (tab_low[1:(Nmax + 1)], dtab_low[1:(Nmax + 1)])
    dir, k = _q_recurrence_plan(x, Nmax, Tx)
    if dir === :upward
        return _legendre_grow!(copy(tab_low), copy(dtab_low), Nmax, 1, x)
    end
    # Normalized on Q₁¹, the lowest degree with an exact closed form here (the
    # `m = 1` recurrence is singular at `n = 0`, so `Q₀¹` stays a placeholder).
    tab = _Q_values_miller(x, Nmax, 1, 1, tab_low[2], k)
    tab[1] = zero(Tx)
    dtab = _Q_derivatives(tab, x, 1, dtab_low[1:2])
    return tab, dtab
end

"""
    legendre_odd(kind::Symbol, x, Nseries::Int) -> (vals, derivs)

Values and derivatives of the requested Legendre kind at the `Nseries`
ODD degrees `1, 3, …, 2·Nseries − 1`, as length-`Nseries` `Vector`s
(index `r` ↔ degree `2r − 1`).

`kind ∈ (:P0, :Q0, :P1, :P1p, :Q1)`:
- `:P0`  — `Pₙ(x)`   (m=0, any branch)
- `:Q0`  — `Qₙ(x)`   (m=0, q branch, |x|>1)
- `:P1`  — `Pₙ¹(x)`  (m=1, q branch, |x|>1)
- `:P1p` — `Pₙ¹(x)`  (m=1, p branch, |x|≤1)
- `:Q1`  — `Qₙ¹(x)`  (m=1, q branch, |x|>1)
"""
function legendre_odd(kind::Symbol, x, Nseries::Int)
    Nmax = 2 * Nseries - 1
    tab, dtab = if kind === :P0
        _P0_table(x, Nmax)
    elseif kind === :Q0
        _Q0_table(x, Nmax)
    elseif kind === :P1
        _P1_table(x, Nmax)
    elseif kind === :P1p
        _P1p_table(x, Nmax)
    elseif kind === :Q1
        _Q1_table(x, Nmax)
    else
        throw(ArgumentError("legendre_odd: unknown kind $kind"))
    end
    idx = 2:2:(2 * Nseries)
    return tab[idx], dtab[idx]
end
