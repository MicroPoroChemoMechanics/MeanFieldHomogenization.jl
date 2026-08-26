# [Numerical Laplace inversion](@id man-laplace-inversion)

Going *to* the Laplace-Carson domain is a quadrature. Coming back is the
ill-posed direction, and there is no algorithm that is best everywhere. This
page says which one to reach for and what it will cost you; the
[tutorial](@ref tut-laplace-inversion) shows the measurements behind every
claim here.

```@example inv
using MeanFieldHomogenization

inverse_laplace(p -> 1 / (p + 2), 1.0)      # ↔ exp(-2t)
```

## 1. The two entry points

```julia
inverse_laplace(F, t, method = DEFAULT_INVERSION)          # f̂(p) = ∫ f e^{-pt} dt
inverse_carson(Fstar, t, method = DEFAULT_INVERSION)       # f*(p) = p f̂(p)
```

Both accept a vector of times as well, returning a vector. `F` may return a
scalar, a `TensND` tensor of any symmetry class, or a `6×6` Mandel matrix — the
symmetry class of the result is the class the transform returns, axis included.

```@example inv
using TensND

Cstar(p) = TensISO{3}(3 * (2 + 3p / (p + 1)), 2 * (1 + 4p / (p + 2)))
inverse_carson(Cstar, 0.7)
```

`inverse_carson` is the one to use for anything from the
[model catalog](@ref man-rheological-models), because that is the convention
[`carson_relaxation`](@ref) and [`carson_creep`](@ref) return.

## 2. Choosing a method

| | nodes | evaluations of `F` | reach for it when |
|:---|:---|:---|:---|
| [`FixedTalbot`](@ref)`(24)` | complex | 24 per time | **the default.** Most accurate on every kernel tested — around `1e-12` |
| [`GaverStehfest`](@ref)`(16)` | **real, positive** | 16 per time | the transform must stay real (see below). About five significant digits |
| [`DeHoog`](@ref)`()` | complex | 33, shared across a block of times | a multi-decade grid where each point costs a homogenization |
| [`TalbotTrefethen`](@ref)`(24)` | complex | 24 per time | cross-checking against the ECHOES reference; poles in the right half-plane (`shift`) |

Three things people expect that turn out to be false, and one that is true:

**Branch cuts are not a problem.** The Talbot contours are Hankel-shaped: they
wrap *around* the negative real axis rather than crossing it, which is exactly
how they were designed to handle singularities there. Every fractional model —
[`ScottBlair`](@ref), [`HuetSayegh`](@ref), [`Model2S2P1D`](@ref) — inverts at
full accuracy under [`FixedTalbot`](@ref).

**Oscillation is what actually separates them.** A rotating phase is invisible
to a method whose nodes are all real. On ``\sin(3t)/3``, `FixedTalbot` gives
`1e-12` and `GaverStehfest` gives O(1).

**More terms is not more accurate.** The Gaver-Stehfest weights alternate in
sign with magnitudes up to ``10^{N/2}`` times the answer, so past `N ≈ 18` in
`Float64` the round-off *grows*:

| `N` | 8 | 12 | **16** | 18 | 22 | 26 | 30 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| relative error | 8e-3 | 4e-4 | **1e-5** | 2e-6 | 2e-3 | 4e-1 | 1e+3 |

The weights themselves are exact — cached as `Rational{BigInt}` and converted
to the working type on demand — so working in `BigFloat` removes the ceiling
entirely and the error keeps falling with `N`.

**The error is absolute, not relative.** Every method controls its error
against the *scale* of `f`, never against `f(t)` at the point asked for. Once a
relaxation function has decayed many decades below its initial value, the
relative error is unbounded while the absolute one still meets the method's
promise. A viscoelastic **solid**, whose relaxation function settles on a
plateau, is the comfortable case; a fluid's tail must be read as an absolute
quantity.

### Why [`GaverStehfest`](@ref) exists despite being the least accurate

Its nodes are real and positive. That is not a curiosity — it means an entire
[`homogenize_lc`](@ref) sweep runs in **real arithmetic**, which

* costs roughly half as much per scheme evaluation as complex arithmetic, on
  top of needing a third fewer evaluations;
* is the only way to use `SelfConsistent(algorithm = NewtonDefault())` on this
  route, since its `ForwardDiff` Jacobian cannot carry a `Dual` over a complex
  scalar;
* lets `ForwardDiff` see plain `Dual` numbers rather than `Complex{Dual}`.

Five digits is ample for an engineering answer. Use it when each evaluation of
the transform is expensive, and [`FixedTalbot`](@ref) when it is not.

### Why [`DeHoog`](@ref) exists

Its nodes depend on the scaling period `T`, not on `t`, so one node set — one
pass over `F` — can serve several times at once. What limits the sharing is
that the accuracy depends on the ratio `t/T` alone:

| `t/T` | 0.5 | 0.15 | 0.05 | 0.005 |
|:---|:---|:---|:---|:---|
| relative error | 1e-9 | 2e-10 | 1e-6 | 1e-2 |

and raising `N` barely helps below `t/T ≈ 0.05`. A single node set therefore
covers a window spanning a factor of about three, not several decades. The
default (`T = nothing`) handles this for you: on a grid it sorts the times,
splits them into blocks spanning at most a factor of three, and gives each block
its own node set. A 200-point grid over seven decades then costs roughly
`15 × 33 ≈ 500` evaluations of `F` instead of `200 × 33 = 6600` — with uniform
accuracy.

Passing `T` explicitly gives you one node set for everything, `2N+1`
evaluations total, and a warning for any time below `t/T = 0.15`.

## 3. Autodiff

All four methods are generic in the number type and take `t::Real` rather than
`t::AbstractFloat`, so `ForwardDiff` traverses them:

```@example inv
using ForwardDiff

Rstar(τ, p) = 2.0 + 3.0 * p * τ / (1 + p * τ)
g = ForwardDiff.derivative(τ -> inverse_carson(p -> Rstar(τ, p), 0.8), 1.5)
exact = 3.0 * 0.8 * exp(-0.8 / 1.5) / 1.5^2
(g, exact)
```

Differentiation with respect to a model parameter, with respect to `t`, nested
duals, and `Complex{Dual}` on the contour methods all work; the gradients carry
the same accuracy as the values, because the inversion is a linear combination
of transform values and the partials ride along the same sum.

!!! note "Why not `InverseLaplace.jl`"
    The only registered numerical-ILT package is not usable here: its entry
    points are annotated `t::AbstractFloat` and `ForwardDiff.Dual` is `<: Real`
    but **not** `<: AbstractFloat`, so a `Dual` is rejected before any
    arithmetic happens. Its `Weeks` method is hard-coded to `Float64` and pulls
    FFTW, and its Stehfest coefficients are `BigFloat`. The algorithms here are
    written to the same references and are differentiable.

### Time derivatives without autodiff

For a rate specifically there is an identity that needs no `Dual` at all. Since
``\mathcal{L}\{\dot f\}(p) = f^{*}(p) - f(0^{+})``,

```@example inv
d = inverse_carson_rate(p -> Rstar(1.5, p), 0.8; f_glassy = 5.0)
(d, -3.0 / 1.5 * exp(-0.8 / 1.5))
```

Both routes are of comparable accuracy on the kernels tested. The identity is
worth having because it asks nothing of the transform beyond what it already
provides — it applies when `F` is not differentiable, and it is the cheaper
route when the rate is wanted alongside the value.
[`glassy_modulus`](@ref) supplies `f_glassy` for every model in the catalog.

## 4. Inversion by fitting

There is a route that is not a quadrature: fit a Prony series to the transform
once, and read the time function off the fit. See
[`prony_fit_relaxation`](@ref) and
[the model manual](@ref man-rheological-models) §3. The result is a *model* —
closed-form in time, exactly convertible to its dual chain, and usable by the
ageing pipeline — rather than a value at one point.

## See also

* [the tutorial](@ref tut-laplace-inversion) — every claim on this page,
  measured and plotted;
* [the theory](@ref th-laplace-carson);
* [`homogenize_lc`](@ref) — where the choice of method actually costs money.
