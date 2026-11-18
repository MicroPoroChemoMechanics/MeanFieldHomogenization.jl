# =============================================================================
#  legendre.jl — the Legendre functions of the spheroid solution, read from
#  TensND (`legendre_P`, `radial_P`, `radial_Q` and their derivatives).
#
#  Two arguments are used throughout the spheroid solution:
#    - the "p" argument, real, |p| ≤ 1 (the angular coordinate): Ferrers'
#      P_n^m(p) = (−√(1−p²))^m dᵐP_n/dpᵐ, P_1^1(p) = −√(1−p²);
#    - the "q" argument, q > 1 for a prolate spheroid or the imaginary q = iτ
#      for an oblate one, carried as `Complex`: Hobson's P_n^m(q), Q_n^m(q),
#      with the principal √(q²−1) (√((iτ)²−1) = i√(τ²+1)).
#
#  TensND's oblate functions are the real 𝒫_n^m(τ) = i⁻ⁿ P_n^m(iτ) and
#  𝒬_n^m(τ) = iⁿ⁺¹ Q_n^m(iτ); with d/dq = −i d/dτ the complex values and
#  derivatives follow:
#
#      P_n^m(iτ) = iⁿ 𝒫_n^m ,          dP_n^m/dq = i^{n−1} 𝒫_n^m′ ,
#      Q_n^m(iτ) = i^{−(n+1)} 𝒬_n^m ,  dQ_n^m/dq = i^{−(n+2)} 𝒬_n^m′ .
#
#  The order 0 of the first kind is a polynomial: Legendre's P_n on both
#  branches, real or complex.
# =============================================================================

const _PROLATE_UNIT = TensND.ProlateSpheroidalChart(1.0)
const _OBLATE_UNIT = TensND.OblateSpheroidalChart(1.0)

# iᵏ, exactly.
_ipow(k::Integer) = (1 + 0im, 0 + 1im, -1 + 0im, 0 - 1im)[mod(k, 4) + 1]

# Hobson's P_n^m and Q_n^m on the q branch, values and derivatives, n = 0 … N:
# the radial functions of TensND. A q that is not complex is prolate, whatever
# its number type (a dual, a symbol); a complex q is oblate, q = iτ, when its
# real part vanishes (τ = 0, the focal disk, included), and prolate otherwise,
# the spheroid code carrying a prolate q as `complex(q)` as well.
_q_first(x, m::Int, N::Int) = (TensND.radial_P(_PROLATE_UNIT, N, m, x), TensND.radial_dP(_PROLATE_UNIT, N, m, x))
_q_second(x, m::Int, N::Int) = (TensND.radial_Q(_PROLATE_UNIT, N, m, x), TensND.radial_dQ(_PROLATE_UNIT, N, m, x))
_q_first(x::Complex, m::Int, N::Int) = iszero(real(x)) ? _oblate_first(imag(x), m, N) : _complexified(_q_first(real(x), m, N))
_q_second(x::Complex, m::Int, N::Int) = iszero(real(x)) ? _oblate_second(imag(x), m, N) : _complexified(_q_second(real(x), m, N))
_complexified((v, d)) = (complex.(v), complex.(d))
function _oblate_first(τ, m::Int, N::Int)
    P, dP = TensND.radial_P(_OBLATE_UNIT, N, m, τ), TensND.radial_dP(_OBLATE_UNIT, N, m, τ)
    return [_ipow(n) * P[n + 1] for n in 0:N], [_ipow(n - 1) * dP[n + 1] for n in 0:N]
end
function _oblate_second(τ, m::Int, N::Int)
    Q, dQ = TensND.radial_Q(_OBLATE_UNIT, N, m, τ), TensND.radial_dQ(_OBLATE_UNIT, N, m, τ)
    return [_ipow(-(n + 1)) * Q[n + 1] for n in 0:N], [_ipow(-(n + 2)) * dQ[n + 1] for n in 0:N]
end

"""
    legendre_order_p(x, m, nmax) -> (P, dP)

Ferrers' functions ``P_n^m(x)`` on the cut (``|x| \\le 1``) and their
derivatives, at the order `m` and the degrees ``n = 0, \\dots,`` `nmax` (index
``n + 1``; zero for ``n < m``), from TensND's `legendre_P` and `legendre_dP`.
"""
legendre_order_p(x, m::Int, nmax::Int) = (TensND.legendre_P(nmax, m, x), TensND.legendre_dP(nmax, m, x))

"""
    legendre_order_q(x, m, nmax) -> (P, dP, Q, dQ)

Hobson's functions ``P_n^m(x)`` and ``Q_n^m(x)`` of the first and second kinds
and their derivatives, off the cut (``x > 1`` for a prolate spheroid,
``x = \\mathrm{i}\\tau`` for an oblate one), at the order `m` and the degrees
``n = 0, \\dots,`` `nmax` (index ``n + 1``; ``P_n^m = 0`` for ``n < m``), from the
radial functions of TensND.
"""
legendre_order_q(x, m::Int, nmax::Int) = (_q_first(x, m, nmax)..., _q_second(x, m, nmax)...)

"""
    legendre_table(kind::Symbol, x, Nmax::Int) -> (vals, derivs)

Values and derivatives of the requested Legendre kind at **every** degree
``0, 1, \\dots, N_{\\max}``, as length-`Nmax + 1` `Vector`s (index ``n + 1`` ``\\leftrightarrow`` degree ``n``).

`kind ∈ (:P0, :Q0, :P1, :P1p, :Q1, :P2, :P2p, :Q2)`:
- `:P0`  — ``P_n(x)``   (``m = 0``, any branch)
- `:Q0`  — ``Q_n(x)``   (``m = 0``, ``q`` branch)
- `:P1`  — ``P_n^1(x)``  (``m = 1``, ``q`` branch)
- `:P1p` — ``P_n^1(x)``  (``m = 1``, ``p`` branch, ``|x| \\le 1``)
- `:Q1`  — ``Q_n^1(x)``  (``m = 1``, ``q`` branch)
- `:P2`  — ``P_n^2(x)``  (``m = 2``, ``q`` branch)
- `:P2p` — ``P_n^2(x)``  (``m = 2``, ``p`` branch, ``|x| \\le 1``)
- `:Q2`  — ``Q_n^2(x)``  (``m = 2``, ``q`` branch)

The ``q`` branch is ``x > 1`` (prolate) or ``x = \\mathrm{i}\\tau`` (oblate).
"""
function legendre_table(kind::Symbol, x, Nmax::Int)
    kind === :P0 && return legendre_order_p(x, 0, Nmax)
    kind === :Q0 && return _q_second(x, 0, Nmax)
    kind === :P1 && return _q_first(x, 1, Nmax)
    kind === :P1p && return legendre_order_p(x, 1, Nmax)
    kind === :Q1 && return _q_second(x, 1, Nmax)
    kind === :P2 && return _q_first(x, 2, Nmax)
    kind === :P2p && return legendre_order_p(x, 2, Nmax)
    kind === :Q2 && return _q_second(x, 2, Nmax)
    throw(ArgumentError("legendre_table: unknown kind $kind"))
end

"""
    legendre_odd(kind::Symbol, x, Nseries::Int) -> (vals, derivs)

Values and derivatives of the requested Legendre kind (see
[`legendre_table`](@ref)) at the `Nseries` odd degrees
``1, 3, \\dots, 2\\mathcal{N} - 1`` (``\\mathcal{N}`` = `Nseries`), as length-`Nseries`
`Vector`s (index ``r`` ``\\leftrightarrow`` degree ``2r - 1``): the degrees of the
conduction problem.
"""
legendre_odd(kind::Symbol, x, Nseries::Int) = legendre_degrees(kind, x, 1:2:(2 * Nseries - 1))

"""
    legendre_degrees(kind::Symbol, x, degrees) -> (vals, derivs)

Values and derivatives of the requested Legendre kind (see
[`legendre_table`](@ref)) at an arbitrary set of `degrees`, in the order
given. Conduction needs the odd degrees alone ([`legendre_odd`](@ref));
elasticity does not, the case-I Papkovich–Neuber potentials splitting by
parity — ``\\varphi_0`` on the even degrees, ``\\varphi_3`` on the odd ones.
"""
function legendre_degrees(kind::Symbol, x, degrees)
    isempty(degrees) && throw(ArgumentError("legendre_degrees: empty degree set"))
    minimum(degrees) ≥ 0 || throw(ArgumentError("legendre_degrees: degrees must be non-negative"))
    tab, dtab = legendre_table(kind, x, maximum(degrees))
    idx = [n + 1 for n in degrees]
    return tab[idx], dtab[idx]
end

# (2k − 1)!! in the element type, (−1)!! = 1.
_double_factorial(::Type{T}, k::Int) where {T} = prod((one(T) * j for j in k:-2:1); init = one(T))
