# =============================================================================
#  laplace_inversion.jl — numerical inversion of the Laplace and
#  Laplace-Carson transforms.
#
#  Three of the four algorithms here evaluate the transform at a finite set of
#  nodes and form a weighted sum,
#
#      f(t) ≈ Σ_k w_k(t, N) · F(p_k(t, N)),
#
#  so for them the inversion operator is *linear* in the transform values.
#  De Hoog is the exception: its quotient-difference acceleration divides
#  samples by one another, so it is inverted component by component.
#
#  Consequences that shape the whole file:
#
#    * the transform may return a scalar, a `TensND` tensor of any symmetry
#      class or a `6×6` Mandel matrix;
#    * `ForwardDiff` flows through unimpeded, provided the accumulator is
#      never seeded with `zero(...)` and no working type is hard-coded.
#      Signatures therefore take `t::Real`, never `t::AbstractFloat`
#      (`Dual <: Real` holds, `Dual <: AbstractFloat` does not);
#    * the Gaver-Stehfest weights depend on `N` alone, so they are computed
#      once in exact rational arithmetic and converted to whatever working
#      type the caller brings.
#
#  This file depends on nothing from MeanFieldHomogenization except `TensND`
#  and `ForwardDiff`.
# =============================================================================

# ── Method types ────────────────────────────────────────────────────────────

"""
    AbstractLaplaceInversion

Root supertype for numerical inverse-Laplace algorithms: [`GaverStehfest`](@ref),
[`FixedTalbot`](@ref), [`TalbotTrefethen`](@ref) and [`DeHoog`](@ref).

All are consumed through [`inverse_laplace`](@ref) and [`inverse_carson`](@ref).
"""
abstract type AbstractLaplaceInversion end

"""
    GaverStehfest(N::Int = 16)

Gaver-Stehfest inversion,

```math
f(t) \\;\\approx\\; \\frac{\\ln 2}{t} \\sum_{k=1}^{N} V_k \\,
      \\hat f\\!\\left(\\frac{k \\ln 2}{t}\\right),
```

with the Salzer weights ``V_k``.

**All nodes are real and positive.** That is why this is the default: a
transform defined only for real `p` works, an entire homogenization scheme run
through [`homogenize_lc`](@ref) stays in real arithmetic, and `ForwardDiff`
sees plain `Dual` numbers rather than `Complex{Dual}`.

`N` must be even with `4 ≤ N ≤ 40`.

!!! warning "More terms is not more accurate"
    The weights alternate in sign with magnitudes up to ``10^{N/2}`` times the
    answer, so the method needs roughly `2.3 M` digits of working precision to
    return `M` correct ones.  In `Float64` that puts a hard ceiling on `N`:

    | `N` | relative error on ``1/(p+2)`` at `t = 1`, in `Float64` |
    |:---|:---|
    | 8  | 8e-3 |
    | 12 | 4e-4 |
    | **16** (default) | **1e-5** |
    | 18 | 2e-6 — the optimum |
    | 22 | 2e-3 |
    | 26 | 4e-1 |
    | 30 | 1e+3 — and it keeps growing |

    Past `N ≈ 18` the round-off *grows* — the single most surprising property
    of the method, and the reason `N` is validated rather than left free.

    Working in `BigFloat` removes the ceiling entirely: the weights are cached
    as exact `Rational{BigInt}` and converted to the working type on demand, so
    under `setprecision(256)` the error keeps falling with `N` all the way to
    `1e-14` at `N = 34` — where `Float64` is at `1e5`.  See
    [the inversion tutorial](@ref tut-laplace-inversion) §2 for the plot.

At its best on completely monotone kernels — creep and relaxation functions —
and poor on oscillatory ones: on ``\\sin(\\omega t)/\\omega`` at `N = 16` the
relative error is O(1), where [`FixedTalbot`](@ref) reaches `1e-12`.

Measured on ``R(t) = E_\\infty + E_1 e^{-t/\\tau}`` with `τ = 1`, `N = 16`:

| `t/τ` | relative error, `E_∞ > 0` | relative error, `E_∞ = 0` |
|:---|:---|:---|
| 0.01 | 4e-8 | 6e-8 |
| 1    | 3e-9 | 5e-7 |
| 10   | 3e-5 | 0.4 |
| 40   | 3e-6 | 5e11 |

The second column is not a defect of this method — see the note on the tail
under [`inverse_laplace`](@ref).  The first is the regime viscoelastic solids
actually live in.
"""
struct GaverStehfest <: AbstractLaplaceInversion
    N::Int

    function GaverStehfest(N::Int = 16)
        iseven(N) || throw(ArgumentError("GaverStehfest: N must be even; got $N"))
        4 ≤ N ≤ 40 ||
            throw(ArgumentError("GaverStehfest: N must satisfy 4 ≤ N ≤ 40; got $N"))
        return new(N)
    end
end

"""
    FixedTalbot(N::Int = 24)

Fixed-Talbot inversion (Abate & Valkó): the Bromwich contour is deformed into
the cotangent curve

```math
p(\\theta) = \\frac{r\\theta}{t}\\,(\\cot\\theta + i), \\qquad
\\theta \\in (-\\pi, \\pi), \\qquad r = \\frac{2N}{5},
```

on which ``e^{pt}`` decays fast enough for the midpoint trapezoidal rule to
converge geometrically.  Near machine precision with `N = 24` evaluations on
meromorphic transforms — Prony series, Zener, Burgers.

**This is [`DEFAULT_INVERSION`](@ref)**, and on every kernel tested it is also
the most accurate of the four — around `1e-12`–`1e-13` relative, for 24
evaluations of the transform:

| transform | `FixedTalbot(24)` | `GaverStehfest(16)` |
|:---|:---|:---|
| `1/(p+a)` (exponential) | 3e-12 | 1e-5 |
| `1/p³` (polynomial) | 1e-14 | 3e-7 |
| `1/√p` (branch cut) | 4e-12 | 4e-7 |
| `1/(p²+ω²)` (oscillatory) | 1e-12 | O(1) |
| 2S2P1D creep kernel | 5e-13 | 3e-8 |

!!! note "Branch cuts are fine — that is what the contour is for"
    The cotangent curve is a Hankel-type contour: it *wraps around* the
    negative real axis rather than crossing it, which is precisely how Talbot
    quadratures are designed to handle singularities there.  Fractional models
    ([`ScottBlair`](@ref), [`HuetSayegh`](@ref), [`Model2S2P1D`](@ref),
    [`Rabotnov`](@ref)) invert at full accuracy — verified against their exact
    ``t \\leftrightarrow p`` pairs in `test/Viscoelasticity/`.

    What it cannot do is reach a singularity in the *right* half-plane; use
    [`TalbotTrefethen`](@ref)`(; shift = …)` for that.

Requires the transform to accept complex arguments.  When it must stay real —
so that a whole homogenization scheme runs in real arithmetic — use
[`GaverStehfest`](@ref) instead.
"""
struct FixedTalbot <: AbstractLaplaceInversion
    N::Int

    function FixedTalbot(N::Int = 24)
        N ≥ 4 || throw(ArgumentError("FixedTalbot: N must be ≥ 4; got $N"))
        return new(N)
    end
end

"""
    TalbotTrefethen(N::Int = 24; shift = 0.0)

Talbot inversion on the Trefethen-Weideman-Schmelzer contour,

```math
p(\\theta) = \\frac{N}{t}\\left(c_1\\,\\theta\\cot(c_2\\theta) - c_3
            + i\\,c_4\\,\\theta\\right),
```

with ``c_1 = 0.5017``, ``c_2 = 0.6407``, ``c_3 = 0.6122``, ``c_4 = 0.2645``.
Converges as ``O(3^{-N})``, and like [`FixedTalbot`](@ref) it handles branch
cuts on ``(-\\infty, 0]`` at full accuracy.  This is the variant used by the
ECHOES Python reference (`tests/python/creep/laplace_inversion.py`), so results
can be cross-checked against it directly.

It differs from [`FixedTalbot`](@ref) in one respect that matters: its contour
is tuned for decaying kernels and degrades badly on oscillatory ones
(`6e-2` on ``\\sin(3t)/3`` at `t = 2`, against `9e-13` for `FixedTalbot`).
Prefer `FixedTalbot` unless cross-checking, or unless `shift` is needed.

`shift` moves the contour to the right when the transform has poles on the
positive real axis.
"""
struct TalbotTrefethen <: AbstractLaplaceInversion
    N::Int
    shift::Float64

    function TalbotTrefethen(N::Int = 24; shift::Real = 0.0)
        N ≥ 4 || throw(ArgumentError("TalbotTrefethen: N must be ≥ 4; got $N"))
        return new(N, Float64(shift))
    end
end

"""
    DeHoog(; N = 16, T = nothing, tol = 1e-9)

De Hoog-Knight-Stokes inversion: the Bromwich integral is discretized into a
Fourier series on the vertical line ``\\Re p = a``,

```math
p_k = a + \\frac{ik\\pi}{T}, \\qquad k = 0,\\dots,2N,
      \\qquad aT = -\\tfrac12\\log(\\texttt{tol}),
```

whose slow convergence is accelerated by the quotient-difference algorithm and
summed as a continued fraction.  Ten to thirteen digits in practice.

Because the line stays in the right half-plane it never crosses a branch cut on
``(-\\infty, 0]`` — unlike the Talbot family — so this is the general-purpose
choice for fractional models when more accuracy than [`GaverStehfest`](@ref) is
wanted.

`T` is the scaling *period*.  The nodes depend on `T` but **not on `t`**, so one
node set — one pass over `F` — can serve several times at once.  What limits
the sharing is that the relative accuracy depends on the ratio `t/T` alone:

| `t/T` | relative error (`N = 16`, `tol = 1e-9`) |
|:---|:---|
| 0.5 (i.e. `T = 2t`) | ≈ 1e-9 |
| 0.15 | ≈ 2e-10 |
| 0.05 | ≈ 1e-6 |
| 0.005 | ≈ 1e-2 |
| 0.0005 | ≈ 0.4 — meaningless |

and raising `N` barely helps below `t/T ≈ 0.05`.  A single node set therefore
covers a window of times spanning a factor of about three, not several decades.

Two modes follow:

  * `T = nothing` (the default) — on a grid, the times are sorted and split
    into blocks spanning at most a factor of three, each block getting its own
    `T = 2 t_max` and its own single pass over `F`.  Accuracy is uniform, and a
    200-point grid over seven decades costs roughly 15 × (2N+1) ≈ 500
    evaluations of `F` instead of 200 × (2N+1) ≈ 6600.  That is the reason to
    reach for `DeHoog` when each evaluation of `F` is a homogenization scheme.
  * `T` given explicitly — one node set for everything, `2N + 1` evaluations
    total.  Only do this when the grid really is narrow; a warning is emitted
    for any time falling below `t/T = 0.15`.

!!! note "Not a weighted sum"
    Unlike the other three, de Hoog is *not* linear in the transform values:
    the quotient-difference tables divide consecutive samples by one another.
    Tensor- and matrix-valued transforms are therefore inverted component by
    component, and a component that vanishes identically is short-circuited to
    zero rather than run through a `0/0`.
"""
struct DeHoog <: AbstractLaplaceInversion
    N::Int
    T::Union{Nothing, Float64}
    tol::Float64

    function DeHoog(; N::Int = 16, T::Union{Nothing, Real} = nothing, tol::Real = 1.0e-9)
        N ≥ 2 || throw(ArgumentError("DeHoog: N must be ≥ 2; got $N"))
        0 < tol < 1 || throw(ArgumentError("DeHoog: tol must lie in (0,1); got $tol"))
        (T === nothing || T > 0) ||
            throw(ArgumentError("DeHoog: T must be positive; got $T"))
        return new(N, T === nothing ? nothing : Float64(T), Float64(tol))
    end
end

"""
    DEFAULT_INVERSION

The inversion method used when none is given: `FixedTalbot(24)`.

Chosen on measured accuracy: it is the only one of the four that stays near
`1e-12` on *every* kernel tested — exponential, polynomial, branch-cut,
oscillatory and the fractional 2S2P1D pair alike (see [`FixedTalbot`](@ref) for
the table).

Reasons to override it:

  * [`GaverStehfest`](@ref) when the transform must be evaluated at **real**
    `p` only — that keeps a whole [`homogenize_lc`](@ref) sweep in real
    arithmetic, costs a third fewer evaluations per point, and is the one way
    to use `SelfConsistent(algorithm = NewtonDefault())` in the Laplace-Carson
    route;
  * [`DeHoog`](@ref) when inverting on a grid whose every point costs a
    homogenization, since one node set serves a whole block of times;
  * [`TalbotTrefethen`](@ref) to cross-check against the ECHOES reference, or
    when the transform has poles in the right half-plane.
"""
const DEFAULT_INVERSION = FixedTalbot(24)

# ── Value algebra: real part, and scalar decomposition ──────────────────────
#
# `TensND` provides no `real(::AbstractTens)`.  Going through `_rebuild`
# (rather than `get_array`) is what keeps a `TensTI` a `TensTI` with its axis
# intact — dropping to a generic `Tens` would be the same class of bug as the
# `zero` trap documented on `_accumulate` below.

const _StructuredTens = Union{TensND.TensISO, TensND.TensTI, TensND.TensOrtho}

_realpart(x::Real) = x
_realpart(x::Number) = real(x)
_realpart(M::AbstractMatrix) = real.(M)
_realpart(A::_StructuredTens) = TensND._rebuild(A, real.(TensND.get_data(A)))
_realpart(A::TensND.AbstractTens) =
    TensND.Tens(real.(TensND.get_array(A)), TensND.get_basis(A), TensND.get_var(A))

"""
    _decompose(x) -> (components::Vector, rebuild)

Split a transform value into a vector of scalar components together with a
closure putting them back into the same shape and symmetry class.

Used only by [`DeHoog`](@ref), whose quotient-difference acceleration divides
samples by one another and is therefore *not* linear in the transform values.
The other three methods never need this: they are weighted sums, so they run on
the value type directly.
"""
_decompose(x::Number) = ([x], c -> @inbounds c[1])

function _decompose(A::_StructuredTens)
    d = TensND.get_data(A)
    n = length(d)
    return collect(d), c -> TensND._rebuild(A, ntuple(i -> @inbounds(c[i]), n))
end

function _decompose(A::TensND.AbstractTens)
    arr = TensND.get_array(A)
    basis = TensND.get_basis(A)
    var = TensND.get_var(A)
    sz = size(arr)
    return vec(collect(arr)), c -> TensND.Tens(reshape(collect(c), sz), basis, var)
end

function _decompose(M::AbstractMatrix)
    sz = size(M)
    return vec(collect(M)), c -> reshape(collect(c), sz)
end

# ── Weighted accumulation ───────────────────────────────────────────────────

"""
    _accumulate(values, weights)

Form `Σ_k weights[k] * values[k]`, seeding the accumulator with the **first
term** rather than with `zero(...)`.

That single choice carries two unrelated correctness requirements at once:

  * `ForwardDiff` — a `zero(T)` seed derived from the *node* type would pin the
    accumulator to a plain float and silently drop the partials of the
    transform values;
  * `TensND` — `Base.zero(::AbstractTens{4,dim,T})` returns a `TensISO`
    **whatever the input class is** (the `@eval` loop over `one`/`zero` in
    `TensND/src/tens_isotropic.jl`).  Seeding with `zero(A)` would therefore
    collapse a `TensTI` transform onto the isotropic class and lose its axis.
    Never calling `zero` sidesteps the trap entirely.
"""
@inline function _accumulate(values, weights)
    s = @inbounds weights[1] * values[1]
    @inbounds for k in 2:length(values)
        s = s + weights[k] * values[k]
    end
    return s
end

# ── Working types ───────────────────────────────────────────────────────────
#
# `typeof(float(t))` rather than a hard-coded `Float64`: this is what lets a
# `ForwardDiff.Dual` or a `BigFloat` time survive all the way to the nodes.

@inline _worktype(t::Real) = typeof(float(t))

"""
    _scalar_float(T)

The plain floating-point type underlying `T`, peeling `ForwardDiff.Dual`
wrappers.  The quadrature weights are built in this type: they never carry
partials (they depend on `N` and `t` only), and promotion does the rest.
"""
_scalar_float(::Type{T}) where {T <: AbstractFloat} = T
_scalar_float(::Type{T}) where {T <: Real} = float(T)
_scalar_float(::Type{D}) where {D <: ForwardDiff.Dual} =
    _scalar_float(ForwardDiff.valtype(D))

"""
    _plain_value(x)

Strip every `ForwardDiff.Dual` layer, returning the underlying float.

Used only to choose [`DeHoog`](@ref)'s scaling period `T`.  `T` is a free
algorithmic parameter — it selects the node set, not the mathematics — so
freezing it at the value level is legitimate and keeps the `Dual` confined to
the one place where `t` enters analytically, `exp(iπt/T)`.  Letting `T` carry
partials would instead push them into `S(Tscale)` and fail.
"""
_plain_value(x::Real) = x
_plain_value(d::ForwardDiff.Dual) = _plain_value(ForwardDiff.value(d))

# ── Gaver-Stehfest weights: exact rationals, cached per N ───────────────────
#
# The Salzer weights depend on N and on nothing else — not on `t`, not on the
# transform, not on any model parameter.  No `Dual` can ever reach them, so
# computing them in `Rational{BigInt}` (Base, no dependency) costs nothing in
# genericity and buys exactness: `Float64` accumulation of the alternating sum
# below is wrong by O(1) already at N = 12.

const _GS_LOCK = ReentrantLock()
const _GS_EXACT = Dict{Int, Vector{Rational{BigInt}}}()
const _GS_CACHE = Dict{Tuple{DataType, Int}, Vector}()

"""
    _gs_exact_weights(N) -> Vector{Rational{BigInt}}

Salzer weights of the Gaver-Stehfest scheme, exactly,

```math
V_i = (-1)^{i+M} \\sum_{k=\\lfloor (i+1)/2 \\rfloor}^{\\min(i,M)}
      \\frac{k^{M}\\,(2k)!}{(M-k)!\\,k!\\,(k-1)!\\,(i-k)!\\,(2k-i)!},
      \\qquad M = N/2 .
```
"""
function _gs_exact_weights(N::Int)
    M = N ÷ 2
    fact = Vector{BigInt}(undef, 2N + 1)      # fact[j+1] = j!
    fact[1] = big(1)
    for j in 1:(2N)
        fact[j + 1] = fact[j] * j
    end
    f(j) = fact[j + 1]

    V = Vector{Rational{BigInt}}(undef, N)
    for i in 1:N
        acc = zero(Rational{BigInt})
        for k in fld(i + 1, 2):min(i, M)
            num = big(k)^M * f(2k)
            den = f(M - k) * f(k) * f(k - 1) * f(i - k) * f(2k - i)
            acc += num // den
        end
        V[i] = iseven(i + M) ? acc : -acc
    end
    return V
end

function _gs_weights(::Type{T}, N::Int) where {T <: Real}
    return lock(_GS_LOCK) do
        exact = get!(() -> _gs_exact_weights(N), _GS_EXACT, N)
        # BigFloat conversions depend on the ambient precision, so they are
        # never cached; fixed-width types are.
        T === BigFloat && return BigFloat.(exact)
        return get!(() -> T.(exact), _GS_CACHE, (T, N))::Vector{T}
    end
end

# ── Node/weight construction, per method ────────────────────────────────────

function _nodes_weights(m::GaverStehfest, t::Real)
    T = _worktype(t)
    S = _scalar_float(T)
    a = log(S(2)) / t
    V = _gs_weights(S, m.N)
    nodes = [a * k for k in 1:(m.N)]
    weights = [a * V[k] for k in 1:(m.N)]
    return nodes, weights
end

function _nodes_weights(m::FixedTalbot, t::Real)
    T = _worktype(t)
    S = _scalar_float(T)
    N = m.N
    r = 2 * S(N) / (5 * t)

    nodes = Vector{Complex{T}}(undef, N)
    weights = Vector{Complex{T}}(undef, N)

    # k = 0: the contour crosses the real axis; the trapezoidal weight is halved.
    nodes[1] = Complex(r, zero(r))
    weights[1] = Complex(exp(r * t) * r / (2 * N), zero(r))

    for k in 1:(N - 1)
        θ = S(k) * S(π) / N
        cotθ = cot(θ)
        s = r * θ * Complex(cotθ, one(cotθ))
        σ = θ + (θ * cotθ - one(θ)) * cotθ     # Abate-Valkó weight factor
        nodes[k + 1] = s
        weights[k + 1] = (r / N) * exp(s * t) * Complex(one(σ), σ)
    end
    return nodes, weights
end

function _nodes_weights(m::TalbotTrefethen, t::Real)
    T = _worktype(t)
    S = _scalar_float(T)
    N = m.N
    c1, c2, c3, c4 = S(0.5017), S(0.6407), S(0.6122), S(0.2645)
    h = 2 * S(π) / N
    shift = S(m.shift)

    nodes = Vector{Complex{T}}(undef, N)
    weights = Vector{Complex{T}}(undef, N)
    for k in 0:(N - 1)
        θ = -S(π) + (k + S(1) / 2) * h
        tanc2θ = tan(c2 * θ)
        sinc2θ = sin(c2 * θ)
        z = shift + (S(N) / t) * (c1 * θ / tanc2θ - c3 + Complex(zero(θ), c4 * θ))
        dz = (S(N) / t) * (-c1 * c2 * θ / sinc2θ^2 + c1 / tanc2θ + Complex(zero(θ), c4))
        nodes[k + 1] = z
        # w = h/(2πi) · e^{zt} · dz ; the 1/i is written as multiplication by -i.
        weights[k + 1] = (h / (2 * S(π))) * Complex(zero(θ), -one(θ)) * exp(z * t) * dz
    end
    return nodes, weights
end

_dehoog_offset(m::DeHoog, ::Type{S}, Tscale) where {S <: Real} =
    -log(S(m.tol)) / (2 * S(Tscale))

function _dehoog_nodes(m::DeHoog, ::Type{S}, Tscale) where {S <: Real}
    a = _dehoog_offset(m, S, Tscale)
    return [Complex(a, S(k) * S(π) / S(Tscale)) for k in 0:(2 * m.N)]
end

# ── Public entry points ─────────────────────────────────────────────────────

"""
    inverse_laplace(F, t, method = DEFAULT_INVERSION)
    inverse_laplace(F, times::AbstractVector, method = DEFAULT_INVERSION)

Numerically invert the Laplace transform `F`, returning `f(t)` with

```math
\\hat f(p) = \\int_0^\\infty f(t)\\,e^{-pt}\\,\\mathrm{d}t .
```

`F` is any callable `p -> F(p)` whose value supports `+` and multiplication by
a scalar.  Scalars, `TensND` tensors of any symmetry class and `6×6` Mandel
matrices all work, and the symmetry class of the result is the class `F`
returns — a `TensTI` transform gives back a `TensTI` with its axis preserved.

`method` selects the algorithm; see [`AbstractLaplaceInversion`](@ref).
[`FixedTalbot`](@ref), [`TalbotTrefethen`](@ref) and [`DeHoog`](@ref) call `F`
with `Complex` arguments; [`GaverStehfest`](@ref) only with real positive ones.

The vector form returns a `Vector` of results.  With `DeHoog(; T = ...)` it
shares one node set across the whole grid, which matters when each evaluation
of `F` is a homogenization scheme.

Throws `DomainError` for `t ≤ 0`: every method places its nodes at `O(1/t)`.

!!! note "Name clash with Symbolics.jl"
    `Symbolics` exports a function also called `inverse_laplace` — a
    five-argument symbolic transform, `inverse_laplace(expr, F, s, f, t)`, used
    for solving ODEs. The two are different functions, so
    `using MeanFieldHomogenization, Symbolics` makes the bare name ambiguous
    and Julia refuses it. Qualify whichever you mean:

    ```julia
    MeanFieldHomogenization.inverse_laplace(F, t)     # this one
    Symbolics.inverse_laplace(expr, F, s, f, t)       # theirs
    ```

    [`inverse_carson`](@ref) — the one to use for anything in the rheology
    catalog — is not affected.

!!! warning "Accuracy is absolute, not relative, in the tail"
    Every algorithm here controls the error against the *scale of `f`*, not
    against `f(t)` at the point asked for.  Once `f` has decayed many orders of
    magnitude below its initial value the relative error is unbounded: on
    ``3e^{-t}`` at `t = 40`, where the function is `1e-17`, `FixedTalbot(24)`
    is off by a relative `1e4` — an absolute `1e-13`, which is exactly what it
    promises.

    This is intrinsic to numerical Laplace inversion, not a defect of one
    method, and it is why viscoelastic *solids* — whose relaxation function
    settles on a plateau ``E_\\infty > 0`` rather than decaying to zero — are the
    comfortable case.  For a fluid, read the tail as an absolute quantity.

# Examples

```jldoctest
julia> using MeanFieldHomogenization

julia> isapprox(inverse_laplace(p -> 1 / (p + 2), 1.0), exp(-2.0); rtol = 1e-6)
true
```

See also [`inverse_carson`](@ref), [`inverse_carson_rate`](@ref).
"""
function inverse_laplace(F, t::Real, method::AbstractLaplaceInversion = DEFAULT_INVERSION)
    _check_time(t)
    return _invert(F, t, method)
end

function inverse_laplace(
        F, times::AbstractVector{<:Real},
        method::AbstractLaplaceInversion = DEFAULT_INVERSION
    )
    foreach(_check_time, times)
    return _invert_grid(F, times, method)
end

@inline function _check_time(t::Real)
    t > 0 || throw(
        DomainError(
            t,
            "inverse_laplace: the time must be strictly positive " *
                "(the quadrature nodes scale as 1/t)"
        )
    )
    return nothing
end

"""
    inverse_carson(Fstar, t, method = DEFAULT_INVERSION)
    inverse_carson(Fstar, times::AbstractVector, method = DEFAULT_INVERSION)

Invert the **Laplace-Carson** transform

```math
f^{*}(p) \\;=\\; p\\int_0^\\infty f(t)\\,e^{-pt}\\,\\mathrm{d}t
       \\;=\\; p\\,\\hat f(p),
```

i.e. `inverse_laplace(p -> Fstar(p) / p, t, method)`.

Laplace-Carson is the transform of choice in viscoelasticity because it maps a
constant to itself: an elastic modulus is its own transform, so the
correspondence principle reads `C*(p)` in place of `C` with no extra factor.
Every model in the [rheology catalog](@ref man-rheological-models) exposes its
transform in this convention through [`carson_relaxation`](@ref) and
[`carson_creep`](@ref).

# Examples

```jldoctest
julia> using MeanFieldHomogenization

julia> R = zener_maxwell(2.0, 3.0, 1.0);      # E_∞ = 2, E₁ = 3, τ₁ = 1

julia> r = inverse_carson(p -> carson_relaxation(R, p), 0.5);

julia> isapprox(r, relaxation(R, 0.5); rtol = 1e-6)
true
```
"""
inverse_carson(Fstar, t::Real, method::AbstractLaplaceInversion = DEFAULT_INVERSION) =
    inverse_laplace(p -> Fstar(p) / p, t, method)

inverse_carson(
    Fstar, times::AbstractVector{<:Real},
    method::AbstractLaplaceInversion = DEFAULT_INVERSION
) = inverse_laplace(p -> Fstar(p) / p, times, method)

"""
    inverse_carson_rate(Fstar, t, method = DEFAULT_INVERSION; f_glassy)

Return the **time derivative** `ḟ(t)` of the function whose Laplace-Carson
transform is `Fstar`, without differentiating the inversion.

Because `L{ḟ}(p) = p f̂(p) - f(0⁺) = f*(p) - f(0⁺)`, the rate is itself an
ordinary inverse Laplace transform:

```math
\\dot f(t) = \\mathcal{L}^{-1}\\bigl[f^{*}(p) - f(0^{+})\\bigr](t).
```

`f_glassy` is `f(0⁺) = lim_{p→∞} f*(p)`, available in closed form for every
model in the catalog via [`glassy_modulus`](@ref).

!!! note "Relation to `ForwardDiff.derivative(t -> ..., t)`"
    Differentiating the inversion with respect to `t` also works — every method
    here accepts a `Dual` time — and on the kernels tested the two routes are of
    comparable accuracy.  The identity is worth having anyway because it asks
    nothing of the transform beyond what it already provides: no `Dual` is
    pushed into the nodes, so it applies when `F` is not differentiable, and it
    is the cheaper route when the rate is wanted alongside the value.
"""
inverse_carson_rate(
    Fstar, t::Real,
    method::AbstractLaplaceInversion = DEFAULT_INVERSION;
    f_glassy
) = inverse_laplace(p -> Fstar(p) - f_glassy, t, method)

# ── Implementations ─────────────────────────────────────────────────────────

function _invert(F, t::Real, method::AbstractLaplaceInversion)
    nodes, weights = _nodes_weights(method, t)
    values = map(F, nodes)
    return _realpart(_accumulate(values, weights))
end

function _invert(F, t::Real, method::DeHoog)
    Tscale = method.T === nothing ? 2 * _plain_value(t) : method.T
    S = _scalar_float(_worktype(t))
    nodes = _dehoog_nodes(method, S, Tscale)
    values = map(F, nodes)
    return _dehoog_from_values(values, t, method, Tscale)
end

_invert_grid(F, times::AbstractVector{<:Real}, method::AbstractLaplaceInversion) =
    [_invert(F, t, method) for t in times]

"""
    _DEHOOG_MIN_RATIO

Smallest `t/T` at which [`DeHoog`](@ref) still returns close to full accuracy.
Measured on exponential and power-law kernels: the relative error is ≈ 2e-10 at
`t/T = 0.15` and ≈ 1e-6 already at `t/T = 0.05`, essentially independent of `N`.

A shared node set therefore spans a factor `0.5 / 0.15 ≈ 3.3` in time, which is
what [`_dehoog_blocks`](@ref) uses.
"""
const _DEHOOG_MIN_RATIO = 0.15

"""
    _dehoog_blocks(order, times, ratio) -> Vector{UnitRange}

Split the *sorted* indices `order` into consecutive runs whose times span at
most `ratio`, so each run can share one de Hoog node set.
"""
function _dehoog_blocks(order, times, ratio::Real)
    blocks = UnitRange{Int}[]
    isempty(order) && return blocks
    start = 1
    for i in 2:length(order)
        if times[order[i]] > ratio * times[order[start]]
            push!(blocks, start:(i - 1))
            start = i
        end
    end
    push!(blocks, start:length(order))
    return blocks
end

# One node set — one pass over `F` — per scale block.  With an explicit `T` the
# user asked for a single block covering everything, so we honor that and warn
# about any time the node set cannot resolve.
function _invert_grid(F, times::AbstractVector{<:Real}, method::DeHoog)
    S = _scalar_float(_worktype(float(first(times))))
    out = Vector{Any}(undef, length(times))


    if method.T !== nothing
        _warn_dehoog_ratio(times, method.T)
        nodes = _dehoog_nodes(method, S, method.T)
        values = map(F, nodes)
        for i in eachindex(times)
            out[i] = _dehoog_from_values(values, times[i], method, method.T)
        end
        return identity.(out)
    end

    order = sortperm(times)
    ratio = 1 / (2 * _DEHOOG_MIN_RATIO)          # t_max / t_min within a block
    for blk in _dehoog_blocks(order, times, ratio)
        Tscale = 2 * _plain_value(times[order[last(blk)]])
        nodes = _dehoog_nodes(method, S, Tscale)
        values = map(F, nodes)
        for j in blk
            i = order[j]
            out[i] = _dehoog_from_values(values, times[i], method, Tscale)
        end
    end
    return identity.(out)
end

function _warn_dehoog_ratio(times, Tscale)
    tmin = minimum(times)
    if tmin < _DEHOOG_MIN_RATIO * Tscale
        @warn """
        DeHoog: some times fall below t/T = $(_DEHOOG_MIN_RATIO), where the \
        continued-fraction acceleration loses most of its digits. \
        Smallest t/T requested = $(tmin / Tscale). \
        Pass `T = nothing` to let the grid be split into scale blocks, or \
        restrict `T` to the narrow window you actually need.""" maxlog = 1
    end
    return nothing
end

"""
    _dehoog_from_values(values, t, method, Tscale)

De Hoog-Knight-Stokes evaluation from transform samples already taken on the
`2N+1` nodes.

The Fourier series ``\\sum_k a_k z^k`` — with ``a_0 = \\tfrac12 F(p_0)``,
``a_k = F(p_k)`` and ``z = e^{i\\pi t/T}`` — is turned into a continued
fraction by the quotient-difference algorithm, summed by the standard
three-term recurrence, and closed with the remainder estimate `R2M` that
removes the leading truncation error.

The QD tables divide samples by one another, so this is **not** linear in the
transform values: tensor- and matrix-valued transforms are inverted component
by component through [`_decompose`](@ref).
"""
function _dehoog_from_values(values, t::Real, method::DeHoog, Tscale)
    comps, rebuild = _decompose(first(values))
    ncomp = length(comps)
    if ncomp == 1 && first(values) isa Number
        return _realpart(_dehoog_scalar([v for v in values], t, method, Tscale))
    end
    out = map(1:ncomp) do i
        _realpart(_dehoog_scalar([_decompose(v)[1][i] for v in values], t, method, Tscale))
    end
    return rebuild(out)
end

function _dehoog_scalar(a::AbstractVector, t::Real, method::DeHoog, Tscale)
    # The quotient-difference tables are built from ratios of consecutive
    # samples, so a component that is identically zero — the off-diagonal
    # entries of a structurally diagonal Mandel matrix, say — would produce
    # `0/0`.  Its inverse transform is zero, and saying so here keeps the
    # tensor and matrix paths clean.
    all(iszero, a) && return zero(_worktype(t)) * zero(eltype(a))

    M = method.N
    n = 2M
    T = _worktype(t)
    S = _scalar_float(T)
    # The working type must come from the *samples* and from `t`, never be
    # hard-coded: a `ForwardDiff.Dual` reaches this routine either through the
    # transform values (differentiation wrt a model parameter) or through `t`
    # itself, and pinning `Complex{Float64}` here would drop its partials.
    CS = promote_type(Complex{S}, eltype(a), Complex{T})

    α = _dehoog_offset(method, S, Tscale)
    z = exp(Complex(zero(S), S(π) * t / S(Tscale)))

    A0 = CS(a[1]) / 2                       # halve the k = 0 sample
    coeff = Vector{CS}(undef, n + 1)
    coeff[1] = A0
    @inbounds for k in 2:(n + 1)
        coeff[k] = CS(a[k])
    end

    # Quotient-difference tables. `e[k+1, r+1]` holds e_{k,r}, likewise for q.
    e = zeros(CS, n + 1, M + 1)
    q = zeros(CS, n + 1, M + 1)
    @inbounds for k in 0:(n - 1)
        q[k + 1, 2] = coeff[k + 2] / coeff[k + 1]
    end
    @inbounds for r in 1:M
        for k in 0:(n - 2r)
            e[k + 1, r + 1] = q[k + 2, r + 1] - q[k + 1, r + 1] + e[k + 2, r]
        end
        if r < M
            for k in 0:(n - 2r - 1)
                q[k + 1, r + 2] = q[k + 2, r + 1] * e[k + 2, r + 1] / e[k + 1, r + 1]
            end
        end
    end

    # Continued-fraction coefficients d_0 = a_0, d_{2r-1} = -q_{0,r},
    # d_{2r} = -e_{0,r}.
    d = Vector{CS}(undef, n + 1)
    d[1] = coeff[1]
    @inbounds for r in 1:M
        d[2r] = -q[1, r + 1]
        2r + 1 ≤ n + 1 && (d[2r + 1] = -e[1, r + 1])
    end

    # Three-term recurrence  A_j = A_{j-1} + d_j z A_{j-2}, same for B.
    Am2, Am1 = zero(CS), d[1]
    Bm2, Bm1 = one(CS), one(CS)
    @inbounds for j in 1:(n - 1)
        dz = d[j + 1] * z
        Am2, Am1 = Am1, Am1 + dz * Am2
        Bm2, Bm1 = Bm1, Bm1 + dz * Bm2
    end

    # Remainder estimate closing the fraction (de Hoog et al., eq. 23).
    h2M = (one(CS) + (d[n] - d[n + 1]) * z) / 2
    R2M = -h2M * (one(CS) - sqrt(one(CS) + d[n + 1] * z / h2M^2))
    Anum = Am1 + R2M * Am2
    Aden = Bm1 + R2M * Bm2

    return exp(α * t) / S(Tscale) * (Anum / Aden)
end
