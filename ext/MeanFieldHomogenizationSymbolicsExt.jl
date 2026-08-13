module MeanFieldHomogenizationSymbolicsExt

using MeanFieldHomogenization
using Symbolics

# ──────────────────────────────────────────────────────────────────────────────
#  Complete elliptic integrals as *opaque* symbolic functions on `Symbolics.Num`
#
#  The Symbolics counterpart of `MeanFieldHomogenizationSymPyExt`, and it exists
#  for the same reason — but the failure mode it avoids is worse.
#
#  Without this extension `ell_K(m::Num)` falls through to the generic AGM
#  recursion, which is *correct* but unrolls 60 iterations of
#  `a=(a+b)/2; b=sqrt(a*b)` into the expression tree: the answer is a monster
#  with ~60 nested `sqrt`, unusable downstream and effectively unprintable.
#  SymPy escapes this through `sympy.elliptic_{k,e}`; Symbolics has no elliptic
#  integrals of its own, so we register ours.
#
#  `@register_symbolic` makes `ell_K`/`ell_E` stop at an unexpanded call node on
#  a symbolic argument, while any *numeric* argument still reaches the fast
#  Float64 method. So a symbolic result stays compact and
#  `Symbolics.substitute` + `Symbolics.value` evaluates it exactly.
#
#  **Differentiation is deliberately not wired here.** Registering the analytic
#  derivatives (NIST DLMF 19.4.1–19.4.2,
#  `dK/dm = (E - (1-m)K)/(2m(1-m))`, `dE/dm = (E - K)/(2m)`) through
#  `Symbolics.derivative(::typeof(f), ::NTuple{1,Any}, ::Val{1})` was tried and
#  did **not** fire: `expand_derivatives` still returns an unexpanded
#  `Differential(x)(ell_E(…))`. Rather than ship a rule that silently does
#  nothing, the supported and *verified* AD route is `ForwardDiff`, which
#  differentiates the AGM arithmetic directly and needs no registration at all
#  (see `test/Cracks/test_cod_symbolic.jl`). Use Symbolics to *derive* and
#  `build_function` to evaluate; use ForwardDiff to differentiate.
# ──────────────────────────────────────────────────────────────────────────────

const _Ell = MeanFieldHomogenization.Elliptic

Symbolics.@register_symbolic _Ell.ell_K(m)
Symbolics.@register_symbolic _Ell.ell_E(m)

end
