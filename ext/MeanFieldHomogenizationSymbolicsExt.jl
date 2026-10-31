module MeanFieldHomogenizationSymbolicsExt

using MeanFieldHomogenization
using Symbolics

# ──────────────────────────────────────────────────────────────────────────────
#  Elliptic integrals as *opaque* symbolic functions on `Symbolics.Num`
#
#  The Symbolics counterpart of `MeanFieldHomogenizationSymPyPythonCallExt`, and it exists
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
#  Float64 method. The incomplete integrals `ell_F(φ, m)` and `ell_E(φ, m)`, on
#  which the Newton potential of a triaxial ellipsoid rests, and Carlson's
#  `ell_RF` and `ell_RD` are registered for the same reason: the duplication
#  needs a stopping test that a `Num` cannot answer. So a symbolic result stays compact and
#  `Symbolics.substitute(…; fold = Val(true))` + `Symbolics.value` evaluates it
#  exactly (without `fold`, the call node is kept with numeric arguments).
#
#  Each carries its derivatives, so that `expand_derivatives` differentiates a
#  symbolic Hill tensor or crack opening through them, and the result stays a
#  closed form in the same integrals (parameter m = k², Δ = √(1 − m sin²φ)):
#
#      dK/dm = (E − (1 − m)K) / (2m(1 − m)),     dE/dm = (E − K) / (2m),
#      ∂F/∂φ = 1/Δ,   ∂F/∂m = E(φ,m)/(2m(1 − m)) − F/(2m) − sin 2φ / (4(1 − m)Δ),
#      ∂E/∂φ = Δ,     ∂E/∂m = (E(φ,m) − F(φ,m)) / (2m)          (NIST DLMF 19.4),
#      ∂R_F/∂x = −R_D(y, z, x)/6, and by symmetry in the other arguments
#      (DLMF 19.18.1);
#      ∂R_D/∂x = (R_D(x, y, z) − R_D(y, z, x)) / (2(z − x)), likewise in y, from
#      the partial fractions of the integrand; ∂R_D/∂z from the homogeneity of
#      R_D of degree −3/2, x∂ₓ + y∂ᵧ + z∂_z = −(3/2)R_D.
#
#  Each is tested against ForwardDiff, which differentiates the AGM and Carlson
#  arithmetic of the numerical methods independently.
# ──────────────────────────────────────────────────────────────────────────────

const _Ell = MeanFieldHomogenization.Elliptic

Symbolics.@register_symbolic _Ell.ell_K(m)
Symbolics.@register_symbolic _Ell.ell_E(m)
Symbolics.@register_symbolic _Ell.ell_F(φ, m)
# A second overload of `ell_E`: its shape rule, which takes any number of
# arguments, came with the first, so only the two-argument type rule is added.
Symbolics.@register_symbolic _Ell.ell_E(φ, m) false
Symbolics.SymbolicUtils.promote_symtype(::typeof(_Ell.ell_E), ::Type, ::Type) = Real
# Carlson's integrals, exported too, for the same reason.
Symbolics.@register_symbolic _Ell.ell_RF(x, y, z)
Symbolics.@register_symbolic _Ell.ell_RD(x, y, z)

Symbolics.@register_derivative _Ell.ell_K(m) 1 (_Ell.ell_E(m) - (1 - m) * _Ell.ell_K(m)) / (2m * (1 - m))
Symbolics.@register_derivative _Ell.ell_E(m) 1 (_Ell.ell_E(m) - _Ell.ell_K(m)) / (2m)
Symbolics.@register_derivative _Ell.ell_F(φ, m) 1 1 / sqrt(1 - m * sin(φ)^2)
Symbolics.@register_derivative _Ell.ell_F(φ, m) 2 _Ell.ell_E(φ, m) / (2m * (1 - m)) - _Ell.ell_F(φ, m) / (2m) -
    sin(2φ) / (4 * (1 - m) * sqrt(1 - m * sin(φ)^2))
Symbolics.@register_derivative _Ell.ell_E(φ, m) 1 sqrt(1 - m * sin(φ)^2)
Symbolics.@register_derivative _Ell.ell_E(φ, m) 2 (_Ell.ell_E(φ, m) - _Ell.ell_F(φ, m)) / (2m)
Symbolics.@register_derivative _Ell.ell_RF(x, y, z) 1 -_Ell.ell_RD(y, z, x) / 6
Symbolics.@register_derivative _Ell.ell_RF(x, y, z) 2 -_Ell.ell_RD(z, x, y) / 6
Symbolics.@register_derivative _Ell.ell_RF(x, y, z) 3 -_Ell.ell_RD(x, y, z) / 6
Symbolics.@register_derivative _Ell.ell_RD(x, y, z) 1 (_Ell.ell_RD(x, y, z) - _Ell.ell_RD(y, z, x)) / (2 * (z - x))
Symbolics.@register_derivative _Ell.ell_RD(x, y, z) 2 (_Ell.ell_RD(x, y, z) - _Ell.ell_RD(z, x, y)) / (2 * (z - y))
Symbolics.@register_derivative _Ell.ell_RD(x, y, z) 3 begin
    ∂x = (_Ell.ell_RD(x, y, z) - _Ell.ell_RD(y, z, x)) / (2 * (z - x))
    ∂y = (_Ell.ell_RD(x, y, z) - _Ell.ell_RD(z, x, y)) / (2 * (z - y))
    (-3 * _Ell.ell_RD(x, y, z) / 2 - x * ∂x - y * ∂y) / z
end

# ──────────────────────────────────────────────────────────────────────────────
#  Dividing an aging viscoelastic operator by a symbolic scalar
#
#  `VolterraOperator{…, T} <: AbstractMatrix{T}`, and Symbolics defines
#  `/(::AbstractArray{<:Real}, ::Num)`. Against the package's own
#  `/(::VolterraOperator, ::Number)` neither signature wins — one is more
#  specific on the left, the other on the right — so `R / num` was an ambiguity.
#
#  The package's method is the one that means something: it divides the stored
#  blocks and hands back an operator of the same class. Symbolics' generic array
#  method would return a bare matrix and lose the class. The method below says
#  so. It lives here rather than in `src/` because Symbolics is a weak
#  dependency: without it loaded there is no `Num` and no ambiguity to resolve.
# ──────────────────────────────────────────────────────────────────────────────

const _V = MeanFieldHomogenization.Viscoelasticity

Base.:/(A::_V.VolterraOperator, c::Symbolics.Num) = invoke(/, Tuple{_V.VolterraOperator, Number}, A, c)

end
