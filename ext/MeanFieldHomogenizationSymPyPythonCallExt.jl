module MeanFieldHomogenizationSymPyPythonCallExt

using MeanFieldHomogenization
using SymPyPythonCall

# ──────────────────────────────────────────────────────────────────────────────
#  Symbolic closed forms via SymPy's `elliptic_{k,e,f}`
#
#  SymPy conventions (identical to `MeanFieldHomogenization.Elliptic` ≡ `Elliptic.jl`):
#  the parameter is `m = k²` (not the modulus `k`).
#
#  Without this extension, the generic AGM path would unfold ~60 nested
#  `sqrt` expressions on a `Sym` input and overwhelm SymPy's pretty-printer.
# ──────────────────────────────────────────────────────────────────────────────

MeanFieldHomogenization.Elliptic.ell_K(m::Sym) = sympy.elliptic_k(m)
MeanFieldHomogenization.Elliptic.ell_E(m::Sym) = sympy.elliptic_e(m)

MeanFieldHomogenization.Elliptic.ell_F(φ::Sym, m::Sym) = sympy.elliptic_f(φ, m)
MeanFieldHomogenization.Elliptic.ell_E(φ::Sym, m::Sym) = sympy.elliptic_e(φ, m)

# Mixed-type cases — promote the non-Sym argument
MeanFieldHomogenization.Elliptic.ell_F(φ::Sym, m::Number) = sympy.elliptic_f(φ, Sym(m))
MeanFieldHomogenization.Elliptic.ell_F(φ::Number, m::Sym) = sympy.elliptic_f(Sym(φ), m)
MeanFieldHomogenization.Elliptic.ell_E(φ::Sym, m::Number) = sympy.elliptic_e(φ, Sym(m))
MeanFieldHomogenization.Elliptic.ell_E(φ::Number, m::Sym) = sympy.elliptic_e(Sym(φ), m)

# ──────────────────────────────────────────────────────────────────────────────
#  Carlson's integrals as SymPy functions
#
#  SymPy has no R_F nor R_D. Without them a `Sym` argument ran the duplication
#  of the numerical method, 50 steps unrolled into the expression. They are
#  defined here as SymPy functions, `elliprf` and `elliprd` (the names of mpmath
#  and SciPy): unevaluated calls, which SymPy differentiates by the same rules
#  as the Symbolics extension and evaluates numerically by mpmath's. Python
#  objects are not precompiled, so they are built when the extension loads.
# ──────────────────────────────────────────────────────────────────────────────

const _CARLSON = """
import sympy, mpmath

class elliprf(sympy.Function):
    nargs = 3
    def fdiff(self, argindex=1):
        x, y, z = self.args
        return -elliprd(*[(y, z, x), (z, x, y), (x, y, z)][argindex - 1]) / 6
    def _eval_evalf(self, prec):
        if all(a.is_number for a in self.args):
            with mpmath.workprec(prec):
                return sympy.Expr._from_mpmath(mpmath.elliprf(*[a._to_mpmath(prec) for a in self.args]), prec)

class elliprd(sympy.Function):
    nargs = 3
    def fdiff(self, argindex=1):
        x, y, z = self.args
        dx = (elliprd(x, y, z) - elliprd(y, z, x)) / (2 * (z - x))
        dy = (elliprd(x, y, z) - elliprd(z, x, y)) / (2 * (z - y))
        return [dx, dy, (-sympy.Rational(3, 2) * elliprd(x, y, z) - x * dx - y * dy) / z][argindex - 1]
    def _eval_evalf(self, prec):
        if all(a.is_number for a in self.args):
            with mpmath.workprec(prec):
                return sympy.Expr._from_mpmath(mpmath.elliprd(*[a._to_mpmath(prec) for a in self.args]), prec)
"""
const _ELLIPRF = Ref{SymPyPythonCall.PythonCall.Py}()
const _ELLIPRD = Ref{SymPyPythonCall.PythonCall.Py}()

function __init__()
    ns = SymPyPythonCall.PythonCall.pydict()
    SymPyPythonCall.PythonCall.pyexec(_CARLSON, ns)
    _ELLIPRF[] = ns["elliprf"]
    _ELLIPRD[] = ns["elliprd"]
    return nothing
end

# Any argument symbolic: the call, the others made symbolic.
_carlson(f, x, y, z) = Sym(f(Sym(x).o, Sym(y).o, Sym(z).o))
for (name, f) in ((:ell_RF, :_ELLIPRF), (:ell_RD, :_ELLIPRD)),
        sig in (
            (:Sym, :Number, :Number), (:Number, :Sym, :Number), (:Number, :Number, :Sym),
            (:Sym, :Sym, :Number), (:Sym, :Number, :Sym), (:Number, :Sym, :Sym), (:Sym, :Sym, :Sym),
        )
    @eval MeanFieldHomogenization.Elliptic.$name(x::$(sig[1]), y::$(sig[2]), z::$(sig[3])) = _carlson($f[], x, y, z)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Symbolic Laplace and Laplace-Carson inversion
#
#  `inverse_laplace(F, t)` with a *numeric* `t` runs one of the four
#  quadratures.  With a `Sym` time it means something else entirely: not "give
#  me the value there" but "give me the function".  SymPy can often do that in
#  closed form, so the same two names dispatch to it.
#
#  This is the direction that matters for viscoelasticity.  Every model in the
#  catalog states its Laplace-Carson transform; when the parameters are
#  symbolic too, `inverse_carson(p -> carson_relaxation(m, p), t)` returns
#  `R(t)` as an expression — the closed form the model would otherwise have had
#  to hard-code.
# ──────────────────────────────────────────────────────────────────────────────

const _V = MeanFieldHomogenization.Viscoelasticity

"""
    inverse_laplace(F, t::Sym; p = Sym("p"), simplify = true)

Symbolic inverse Laplace transform: return `f(t)` as an expression rather than
a number.

`F` is called once, on the symbol `p`, and the resulting expression is handed to
SymPy's `inverse_laplace_transform`. The `Heaviside(t)` factor SymPy attaches is
removed — causality is already carried by the `t ≥ t'` branch of every kernel in
this package, and keeping it would make every downstream `simplify` unwieldy.

Returns the unevaluated `InverseLaplaceTransform` object when SymPy cannot do
the integral, which is SymPy's own way of saying so.

# Example

```julia
using SymPyPythonCall
@syms t::positive τ::positive E::positive
inverse_laplace(p -> E / (p + 1 / τ), t)     # E*exp(-t/τ)
```
"""
function MeanFieldHomogenization.inverse_laplace(
        F, t::Sym; p::Sym = Sym("p"), simplify::Bool = true
    )
    return _ilt(F(p), p, t, simplify)
end

"""
    inverse_carson(Fstar, t::Sym; p = Sym("p"), simplify = true)

Symbolic inverse **Laplace-Carson** transform, i.e.
`inverse_laplace(p -> Fstar(p) / p, t)`.

This is the one to use on anything from the
[rheology catalog](@ref man-rheological-models), since `carson_relaxation` and
`carson_creep` return transforms in that convention. With symbolic model
parameters it recovers the closed-form time function:

```julia
using SymPyPythonCall, MeanFieldHomogenization
@syms t::positive τ::positive E∞::positive E₁::positive
m = zener_maxwell(E∞, E₁, τ)
inverse_carson(p -> carson_relaxation(m, p), t)     # E∞ + E₁*exp(-t/τ)
```

!!! note "Where it stops"
    SymPy inverts rational transforms and many algebraic ones. It does **not**
    generally invert the fractional models — `(pτ)^{-k}` with a symbolic `k`
    has no elementary inverse — and returns the unevaluated transform there.
    That is not a gap this package can close: those inverses are
    Mittag-Leffler functions or worse, which is exactly why
    [`inverse_carson`](@ref) with a numeric time and one of the four
    quadratures exists.
"""
function MeanFieldHomogenization.inverse_carson(
        Fstar, t::Sym; p::Sym = Sym("p"), simplify::Bool = true
    )
    return _ilt(Fstar(p) / p, p, t, simplify)
end

function _ilt(expr::Sym, p::Sym, t::Sym, do_simplify::Bool)
    out = sympy.inverse_laplace_transform(expr, p, t)
    out = _strip_heaviside(out, t)
    out = do_simplify ? sympy.simplify(out) : out
    _warn_if_not_inverted(out, expr)
    return out
end

_ilt(expr, p::Sym, t::Sym, do_simplify::Bool) = _ilt(Sym(expr), p, t, do_simplify)

# SymPy writes the answer as `f(t)·Heaviside(t)`; substituting `Heaviside → 1`
# is the documented way to drop it, and is safe because every caller here has
# already restricted itself to `t > 0`.
_strip_heaviside(e::Sym, t::Sym) = e.subs(sympy.Heaviside(t), 1)
_strip_heaviside(e, ::Sym) = e

"""
    _warn_if_not_inverted(out, expr)

Say so when SymPy did not actually invert the transform.

There are two ways it declines, and only one of them is honest:

  * it returns an unevaluated `InverseLaplaceTransform(...)` — visible in the
    result, and harmless;
  * it returns **`nan`** — which is not, because `nan` propagates silently
    through every subsequent substitution and plot. A springpot with a
    *symbolic* exponent does exactly this.

Neither is a defect of this package: those inverses are Mittag-Leffler
functions or worse. But a silent `nan` deserves a word, so the caller knows to
fall back on [`inverse_carson`](@ref) with a numeric time and one of the four
quadratures, which invert precisely these transforms to `1e-12`.
"""
function _warn_if_not_inverted(out, expr)
    txt = string(out)
    if occursin("nan", lowercase(txt))
        @warn """
        Symbolic inversion returned `nan`: SymPy could not invert this \
        transform and said so in the worst possible way. Transform: $(expr). \
        Use `inverse_carson(F, t::Real, FixedTalbot(24))` instead — the \
        numerical quadratures handle fractional transforms at ~1e-12."""
    elseif occursin("InverseLaplaceTransform", txt)
        @warn """
        SymPy left the transform uninverted (a sum of several fractional \
        powers has no elementary inverse). The result is the unevaluated \
        `InverseLaplaceTransform`. Use a numeric time and one of the four \
        quadratures instead."""
    end
    return nothing
end

end
