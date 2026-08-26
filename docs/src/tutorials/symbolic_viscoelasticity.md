# [Symbolic viscoelasticity: closed forms, derived](@id tut-symbolic-viscoelasticity)

The [rheology catalog](@ref man-rheological-models) is generic in its scalar
type, exactly as the tensor algebra is. Two consequences, independent of one
another:

* every model accepts **symbolic parameters** — SymPy `Sym` or Symbolics `Num` —
  and returns its Laplace-Carson transform as an exact expression;
* with `SymPy` loaded, [`inverse_carson`](@ref) accepts a **symbolic time** and
  hands the inversion to SymPy, returning `R(t)` or `J(t)` in closed form
  instead of a number.

Together they let the closed forms be *derived* rather than transcribed — which
is how the Burgers relaxation function, this package's sharpest numerical
oracle, is checked below.

Terser script:
[`scripts/66_symbolic_viscoelasticity.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/66_symbolic_viscoelasticity.jl).

## 1. Symbolic parameters

```@example tutsymvisco
using MeanFieldHomogenization
using TensND
using Symbolics

Symbolics.@variables E∞ E₁ τ₁ p t

m = zener_maxwell(E∞, E₁, τ₁)
Symbolics.simplify(carson_relaxation(m, p))
```

The closed-form time function comes out symbolic too:

```@example tutsymvisco
relaxation(m, t)
```

and the reciprocity identity — the definition of the creep transform — holds
as an identity rather than to a tolerance:

```@example tutsymvisco
Symbolics.simplify(carson_creep(m, p) * carson_relaxation(m, p) - 1)
```

!!! note "Why this needed a change"
    The constructors validate their arguments — `0 < k < h < 1`, `τ > 0`, and
    so on. A symbolic `τ` makes `0 < τ` an *expression*, not a `Bool`, and
    using it in a boolean context throws. `Symbolics.Num <: Real` is famously
    **not** the predicate that separates the two: `ForwardDiff.Dual` is also
    `<: Real` and *does* compare to a `Bool`.

    Every validation is therefore guarded by
    `MeanFieldHomogenization.Elliptic.is_hard_numeric`, so a symbolic model is
    simply not validated. That is the same guard the elliptic integrals have
    used since the confusion first bit.

The fractional and bituminous models, whose constructors carry the tightest
validations, build symbolically as well, and the lift to a fourth-order tensor
keeps everything symbolic:

```@example tutsymvisco
Symbolics.@variables kk
carson_relaxation(iso_rheology(Spring(kk), m), p)
```

### The one thing a symbolic spectrum cannot do

[`maxwell_to_kelvin`](@ref) locates its roots by **bisection** between
consecutive poles. That needs a real ordering of the relaxation times and a
real sign test, neither of which a symbolic `τ` provides — so it refuses, with
the reason, rather than failing later inside the bracketing:

```@example tutsymvisco
try
    maxwell_to_kelvin(m)
catch e
    println(first(split(sprint(showerror, e), " The transform")))
end
```

The transform itself is unaffected: `carson_creep` falls back on the exact
`J* = 1/R*`, which *is* symbolic. Only the change of representation — and the
closed-form time function that depends on it — needs numbers.

## 2. Symbolic Laplace-Carson inversion

With `SymPy` loaded, asking for the value at a **symbolic** time means
something different from asking at a numeric one: not *the value there* but
*the function*.

```@example tutsymvisco
using SymPy

SymPy.@syms ts::positive τ::positive Einf::positive E1::positive

z = zener_maxwell(Einf, E1, τ)
inverse_carson(p -> carson_relaxation(z, p), ts)
```

It agrees with the closed form the model already carries — which is a check on
both:

```@example tutsymvisco
SymPy.simplify(inverse_carson(p -> carson_relaxation(z, p), ts) - relaxation(z, ts))
```

The creep function of the same standard solid, recovered from `R*(p)` alone:

```@example tutsymvisco
SymPy.simplify(inverse_carson(p -> carson_creep(z, p), ts))
```

### Burgers: the oracle, derived

The relaxation function of the Burgers model is the `cosh`/`sinh` expression
that had to be produced in a computer-algebra system to serve as the numerical
oracle in `test/Viscoelasticity/test_prony.jl`. Here it is derived from the
transform in one line:

```@example tutsymvisco
SymPy.@syms ks::positive ηs::positive kp::positive ηp::positive

b = burgers(ks, ηs, kp, ηp)
inverse_carson(p -> carson_relaxation(b, p), ts)
```

and the creep function, which carries the term linear in `t` that makes Burgers
a fluid:

```@example tutsymvisco
SymPy.simplify(inverse_carson(p -> carson_creep(b, p), ts) - creep(b, ts))
```

!!! note "Name clash with Symbolics.jl"
    `Symbolics` exports its own `inverse_laplace` — a five-argument transform
    for solving ODEs — so once both packages are `using`-ed the bare name is
    ambiguous and must be qualified,
    `MeanFieldHomogenization.inverse_laplace(F, t)`.
    [`inverse_carson`](@ref), the one to use for anything in the catalog, is
    not affected.

## 3. How far the symbolic route reaches

This is worth stating precisely, because the boundary is not where one would
guess.

**A rational exponent inverts exactly.**

```@example tutsymvisco
SymPy.@syms V::positive
sb = ScottBlair(V, Sym(2) // 5)
(inverse_carson(q -> carson_creep(sb, q), ts), creep(sb, ts))
```

Those two are the same number written two ways — `5/(2Γ(2/5)) = 1/Γ(7/5)`.

**The Cole-Cole model at `α = 1/2` has a genuine closed form**, since
``E_{1/2}(-x) = e^{x^2}\operatorname{erfc}(x)``, and SymPy finds it:

```@example tutsymvisco
SymPy.@syms E0::positive
inverse_carson(q -> carson_relaxation(FractionalZener(Einf, E0, τ, Sym(1) // 2), q), ts)
```

**A symbolic exponent is where it stops — and it stops badly.** SymPy returns
`nan` rather than an unevaluated transform, which would propagate silently
through every later substitution, so the extension warns:

```@example tutsymvisco
SymPy.@syms α::positive
inverse_carson(q -> carson_creep(ScottBlair(V, α), q), ts)
```

**A sum of several fractional powers** — [`HuetSayegh`](@ref),
[`Model2S2P1D`](@ref) — is left unevaluated, which is at least honest, and is
also warned about.

None of this is a gap in the package: those inverses are Mittag-Leffler
functions or worse. It is precisely why [`inverse_carson`](@ref) with a
*numeric* time and one of the four quadratures exists —
[`FixedTalbot`](@ref)`(24)` inverts every one of these transforms to about
`1e-12`, as [the inversion tutorial](@ref tut-laplace-inversion) measures.

## 4. The two routes, on the same model

The symbolic and numerical inversions are entirely disjoint implementations, so
their agreement is a real cross-check:

```@example tutsymvisco
R_t = inverse_carson(p -> carson_relaxation(z, p), ts)
zn = zener_maxwell(2.0, 3.0, 1.5)

[
    (
        tv,
        Float64(R_t.subs(Dict(Einf => 2, E1 => 3, τ => Sym(3) // 2, ts => Sym(tv)))),
        inverse_carson(p -> carson_relaxation(zn, p), tv),
    ) for tv in (0.2, 1.0, 5.0)
]
```

## See also

* [the model catalog](@ref man-rheological-models);
* [choosing a numerical inversion](@ref man-laplace-inversion);
* [symbolic spheres](@ref tut-symbolic-spheres), for the elastic counterpart.
