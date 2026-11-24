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

# ── Moduli transversely isotropic about the radial direction ─────────────────

"""
    RadialField{order, C}

A tensor field of TensND, `ChartTens`, of the chart `C`, whose pattern is
transversely isotropic: about the radial vector of the chart when
[`is_radial`](@ref) holds, the case the layered sphere and the layered cylinder
solve.
"""
const RadialField{order, C} = TensND.ChartTens{order, 3, <:Any, <:TensND.TensTI{order}, C}

"Whether the axis of the pattern of `c` is the radial direction of its chart."
is_radial(c::RadialField) = TensND.isparallel(TensND.axis(c.pattern), TensND.radial_axis(c.chart))

"""
    radial_hill_moduli(c::RadialField{4}) -> (; n, l, k, m, p)

Hill's moduli about the radial direction ``r``, with ``t, t'`` two orthogonal
directions across it: ``n = C_{rrrr}``, ``l = C_{rrtt}``,
``k = (C_{tttt} + C_{ttt't'})/2``, ``m = (C_{tttt} - C_{ttt't'})/2``,
``p = C_{rtrt}``, from the Walpole coefficients of the pattern.
"""
function radial_hill_moduli(c::RadialField{4})
    ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆ = TensND.get_data(c.pattern)
    return (n = ℓ₁, l = ℓ₃ / sqrt(convert(TensND.sqrt2_eltype(typeof(ℓ₃)), 2)), k = ℓ₂ / 2, m = ℓ₅ / 2, p = ℓ₆ / 2)
end

"The coefficients `(tangential, radial)` of a field of order 2 transversely isotropic about the radius."
radial_coefficients(c::RadialField{2}) = TensND.get_data(c.pattern)

# ── The power of a system and its regular solutions ──────────────────────────

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

# ── Moduli that vary with the radius ─────────────────────────────────────────

"""
    GradedLayer(profile; reltol = 1.0e-10, abstol = 1.0e-12, start = 1.0e-8)

A layer of a [`LayeredSphere`](@ref) or of a [`LayeredCylinder`](@ref) whose
modulus varies with the radius: `profile(r)` is the modulus at the radius `r`,
an isotropic tensor or a field transversely isotropic about the radial
direction (`TensND.radial_ti`). A function given as the modulus of a layer is
wrapped in a `GradedLayer` with the default keywords.

The radial problems of such a layer are the systems
``r\\,\\mathrm{d}Y/\\mathrm{d}r = \\boldsymbol{M}(r)\\,Y`` of a uniform one, with
``\\boldsymbol{M}`` taken at the moduli of each radius, integrated numerically
across the layer to the tolerances `reltol` and `abstol`. When it is the core,
`profile(0)` must exist: the integration starts at the radius `start` times
that of the core, from the solutions regular at the center of a medium of
modulus `profile(0)`.
"""
struct GradedLayer{F}
    profile::F
    reltol::Float64
    abstol::Float64
    start::Float64
end
GradedLayer(profile; reltol::Real = 1.0e-10, abstol::Real = 1.0e-12, start::Real = 1.0e-8) =
    GradedLayer(profile, Float64(reltol), Float64(abstol), Float64(start))

"""
    radial_transfer(system, layer::GradedLayer, r_in, r_out, Y) -> SMatrix

The states `Y` (one column per solution) of
``r\\,\\mathrm{d}Y/\\mathrm{d}r = \\boldsymbol{M}(r)\\,Y`` at `r_in`, carried to
`r_out`, with `system(r)` the matrix ``\\boldsymbol{M}(r)``. The equation is
integrated in ``s = \\ln(r/r_{\\mathrm{in}})/\\ln(r_{\\mathrm{out}}/r_{\\mathrm{in}})``
from 0 to 1, so that the radii may be dual numbers. The step is controlled on
values: the numbers compare (real, complex or dual), they are not symbols.
"""
function radial_transfer(system::F, layer::GradedLayer, r_in, r_out, Y::SMatrix) where {F}
    L = log(r_out / r_in)
    T = promote_type(eltype(Y), typeof(L), eltype(system(r_in)))
    rhs(Z, _, s) = L * (system(r_in * exp(s * L)) * Z)
    problem = ODEProblem(rhs, map(T, Y), (0.0, 1.0))
    return solve(problem, Tsit5(); reltol = layer.reltol, abstol = layer.abstol, save_everystep = false).u[end]
end

"""
    regular_transfer(system, layer::GradedLayer, r, Val(n)) -> SMatrix{2n,n}

A basis of the states at the radius `r` of the solutions of
``r\\,\\mathrm{d}Y/\\mathrm{d}r = \\boldsymbol{M}(r)\\,Y`` that are regular at the
center, for moduli that vary from the center to `r`.

Carrying `n` solutions from a small radius makes them parallel, the one of
largest exponent taking over. What is carried instead is their subspace, by its
impedance ``\\boldsymbol{Z}``, which maps the first half of a regular state (the
displacement) to the second one (``r`` times the traction):

``r\\,\\mathrm{d}\\boldsymbol{Z}/\\mathrm{d}r = \\boldsymbol{M}_{21} + \\boldsymbol{M}_{22}\\boldsymbol{Z}
 - \\boldsymbol{Z}\\boldsymbol{M}_{11} - \\boldsymbol{Z}\\boldsymbol{M}_{12}\\boldsymbol{Z}``,

a Riccati equation of which the impedance of the regular solutions is the
stable solution in the direction of increasing radii. It starts at the radius
`layer.start * r`, from the regular solutions of the medium found at the center.
"""
function regular_transfer(system::F, layer::GradedLayer, r, ::Val{n}) where {F, n}
    top, bottom = SVector{n}(1:n), SVector{n}((n + 1):(2n))
    Y₀ = regular_states(system(zero(r)), Val(n))
    Z₀ = Y₀[bottom, :] * inv(Y₀[top, :])
    L = -log(oftype(float(real(one(r))), layer.start))                 # ln(r / r_start)
    function rhs(Z, _, s)
        M = system(r * layer.start * exp(s * L))
        return L * (M[bottom, top] + M[bottom, bottom] * Z - Z * M[top, top] - Z * (M[top, bottom] * Z))
    end
    T = promote_type(eltype(Z₀), eltype(system(r)))
    Z = solve(ODEProblem(rhs, map(T, Z₀), (0.0, 1.0)), Tsit5(); reltol = layer.reltol, abstol = layer.abstol, save_everystep = false).u[end]
    return vcat(one(Z), Z)
end
