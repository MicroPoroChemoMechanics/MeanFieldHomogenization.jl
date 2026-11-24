# =============================================================================
#  radial_systems.jl — linear systems of the first order in the radius.
#
#  The problems of a body of spherical or cylindrical symmetry under a remote
#  uniform loading reduce, loading mode by loading mode, to
#
#      r dY/dr = M Y,
#
#  Y being the state that an interface transmits (radial functions of the
#  displacement and of r times the traction, or of the temperature and of r
#  times the flux) and M a matrix of the moduli at the radius r. Where the
#  moduli are uniform, M is constant and
#
#      Y(r_out) = (r_out/r_in)^M Y(r_in),      ρ^M = exp(M ln ρ).
#
#  The eigenvalues of M are the exponents of the radial modes r^s. They come in
#  pairs symmetric about tr(M)/N (−1/2 for a sphere, 0 for a cylinder): with
#  X = M − (tr M/N) 𝟙, of eigenvalues ±t, the solutions regular at the center
#  are those of the eigenvalues of X of positive real part.
#
#  This file gives the power and a basis of the regular solutions, for numbers
#  (real, complex, dual) of any size met, and in closed form at size 2, where
#  X² = t² 𝟙, for symbols as well. The matrices M themselves are with the
#  inclusions (`LayeredSpheres/radial_layers.jl`, `LayeredCylinders/…`).
# =============================================================================

_trace_free(M::SMatrix{N, N}) where {N} = M - (tr(M) / N) * one(M)
_exponent(X::SMatrix{2, 2}) = sqrt(X[1, 1]^2 + X[1, 2] * X[2, 1])

"""
    radial_power(M::SMatrix, ρ) -> SMatrix

``\\rho^{\\boldsymbol{M}} = \\exp(\\boldsymbol{M}\\ln\\rho)``, the matrix that carries the
state of ``r\\,\\mathrm{d}Y/\\mathrm{d}r = \\boldsymbol{M}\\,Y`` from a radius to the
radius ``\\rho`` times larger, ``\\boldsymbol{M}`` being constant in between. At size
2 it is the closed form
``\\rho^{\\mathrm{tr}\\boldsymbol{M}/2}\\,[\\cosh(t\\ln\\rho)\\,\\boldsymbol{1} + \\sinh(t\\ln\\rho)\\,\\boldsymbol{X}/t]``
with ``\\boldsymbol{X} = \\boldsymbol{M} - \\tfrac12\\mathrm{tr}\\boldsymbol{M}\\,\\boldsymbol{1}``
and ``\\boldsymbol{X}^2 = t^2\\,\\boldsymbol{1}``, which symbols go through.
"""
function radial_power(M::SMatrix{2, 2}, ρ)
    X = _trace_free(M)
    t = _exponent(X)
    x = t * log(ρ)
    return ρ^(tr(M) / 2) * (cosh(x) * one(M) + (sinh(x) / t) * X)
end
# Of any size, by the exponential: exponents real or complex, distinct or not.
radial_power(M::SMatrix, ρ) = exp(M * log(ρ))

"""
    regular_states(M::SMatrix{N,N}, Val(n)) -> SMatrix{N,n}

A basis of the states of the solutions of
``r\\,\\mathrm{d}Y/\\mathrm{d}r = \\boldsymbol{M}\\,Y`` that are regular at
``r = 0``, ``\\boldsymbol{M}`` being constant near the center: `n` columns of the
projector on the eigenvalues of ``\\boldsymbol{M}`` above the mean of its
eigenvalues, the best conditioned ones.
"""
function regular_states(M::SMatrix{2, 2}, ::Val{1})
    X = _trace_free(M)
    P = X + _exponent(X) * one(M)
    return _columns(P, _largest_columns(Val(is_hard_numeric(real(eltype(P)))), P))
end
function regular_states(M::SMatrix{4, 4}, ::Val{2})
    P = (one(M) + _matrix_sign(_trace_free(M))) / 2
    return _columns(P, _independent_columns(P))
end

# The sign of a matrix without imaginary eigenvalue, by Newton's iteration
# X ← (X + X⁻¹)/2, which converges quadratically. It is stopped on the values;
# two more steps then bring the derivatives carried by dual numbers, up to the
# third order, to the same accuracy.
function _matrix_sign(X::SMatrix)
    tol = 16 * eps(typeof(_value_norm(X)))
    for _ in 1:100
        Y = (X + inv(X)) / 2
        settled = _value_norm(Y - X) ≤ tol * _value_norm(Y)
        X = Y
        settled && break
    end
    X = (X + inv(X)) / 2
    return (X + inv(X)) / 2
end
# Norms and comparisons are taken on the values: those of the real and of the
# imaginary part of a complex number, which may each be a dual number.
_value(x) = TensND.primal(x)
_value(z::Complex) = complex(TensND.primal(real(z)), TensND.primal(imag(z)))
_value_norm(X) = sqrt(sum(abs2 ∘ _value, X))

# The column of P of largest norm, or the two whose Gram determinant is the
# largest, so that the basis kept is well conditioned. Compared on values; a
# symbolic projector of size 2 has proportional columns, the first one is kept.
_columns(P::SMatrix{N, N}, idx::NTuple{1, Int}) where {N} = SMatrix{N, 1}(P[:, idx[1]])
_columns(P::SMatrix{N, N}, idx::NTuple{2, Int}) where {N} = hcat(P[:, idx[1]], P[:, idx[2]])
_largest_columns(::Val{false}, P) = (1,)
_largest_columns(::Val{true}, P::SMatrix{N, N}) where {N} = (argmax(j -> _value_norm(P[:, j]), 1:N),)
function _independent_columns(P::SMatrix{4, 4})
    V = map(_value, P)
    gram(i, j) = real(V[:, i] ⋅ V[:, i]) * real(V[:, j] ⋅ V[:, j]) - abs2(V[:, i] ⋅ V[:, j])
    pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    return pairs[argmax(map(ij -> gram(ij...), pairs))]
end
