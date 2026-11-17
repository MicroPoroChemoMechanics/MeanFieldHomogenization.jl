# =============================================================================
#  harmonic_transfer.jl — elastic response of a layered sphere to the harmonic
#  of any degree.
#
#  The uniform remote strain of Eshelby's problem excites the degrees 0 (the
#  dilatation, `bulk_recurrence.jl`) and 2 (the deviatoric strain,
#  `shear_recurrence.jl`). A scheme that resolves the field around each sphere
#  needs every degree n, and at each one the three solutions of the Lamé
#  equation of TensND (`regular_lame_solution`, `irregular_lame_solution`), regular u⁽ʲ⁾_nm and decaying
#  U⁽ʲ⁾_nm.
#
#  On a sphere of radius r the displacement and the traction of each solution
#  are combinations of the vector surface harmonics of the same degree and
#  order,
#
#       S⁽¹⁾ = r grad χ_n^m ,   S⁽²⁾ = r curl(χ_n^m e_r) ,   S⁽³⁾ = χ_n^m e_r ,
#
#  the tangential gradient, the tangential rotation and the normal. With d = n
#  for the regular solutions and d = −(n+1) for the decaying ones, and up to
#  the factorial of the normalization (kushch2013, Eqs. (4.34) and (4.44)),
#
#    displacement             traction / 2μ
#    u⁽¹⁾: r^{d−1} (S⁽¹⁾ + d S⁽³⁾)         (d−1) u⁽¹⁾ / r
#    u⁽²⁾: −r^d S⁽²⁾/(d+1)                 (d−1) u⁽²⁾ / (2r)
#    u⁽³⁾: r^{d+1} (β_d S⁽¹⁾ + γ_d S⁽³⁾)   r^d (β′_d S⁽¹⁾ + γ′_d S⁽³⁾)
#
#    β_d = (d+5−4ν)/((d+1)(2d+3)) ,  γ_d = (d−2+4ν)/(2d+3) ,
#    β′_d = (d+1) β_d − 2(1−ν)/(d+1) ,  γ′_d = (d+1) γ_d − 2ν .
#
#  The state of a harmonic at a radius is the vector of its six components
#  (u₁, u₂, u₃, τ₁, τ₂, τ₃) on (S⁽¹⁾, S⁽²⁾, S⁽³⁾); `_lame_state` gives it for the
#  three solutions of one kind. It is carried from the center outward: through
#  a layer by the six solutions of that layer, across an interface by the jump
#  of its model, as in the two recurrences of the uniform strain. The sphere is
#  isotropic, so the order m plays no part.
#
#  Interface jumps, the outer side from the inner one:
#    spring     u₁ += sₜ τ₁ ,  u₂ += sₜ τ₂ ,  u₃ += sₙ τ₃ ;
#    membrane   τ₁ += [−2κˢ u₃ + (n(n+1)(κˢ+μˢ) − 2μˢ) u₁] / r² ,
#               τ₂ += (n(n+1) − 2) μˢ u₂ / r² ,
#               τ₃ += [4κˢ u₃ − 2n(n+1) κˢ u₁] / r² ,
#  the surface divergence of the membrane stress on the harmonic of degree n
#  (kushch2013, Eq. (4.72), without residual surface tension); at n = 2 they
#  are the jumps of `interface_transfer.jl`.
#
#  The degree 0 holds one regular solution, the dilatation u⁽³⁾_00, and one
#  decaying one, the center of dilatation U⁽¹⁾_00, both radial: its state is
#  (u₃, τ₃) and its interface jumps are those of the bulk recurrence.
# =============================================================================

# Poisson ratio from s = μ/(k+μ), finite for an incompressible layer (s = 0).
# A void (k = μ = 0) carries no traction whatever its ratio: it takes zero.
@inline function _lame_poisson(κ, μ)
    is_hard_numeric(typeof(μ)) && iszero(μ) && iszero(κ) && return zero(μ / one(κ))
    s = μ / (κ + μ)
    return (3 - 5 * s) / (6 - 4 * s)
end

"""
    _lame_state(d, r, μ, ν) -> Matrix(6×3)

Components of the displacement (rows 1 to 3) and of the traction (rows 4 to 6)
on ``(\\underline{S}^{(1)}, \\underline{S}^{(2)}, \\underline{S}^{(3)})``, at the
radius `r`, of the three solutions of the Lamé equation of one kind and one
degree: the regular ones of degree ``n`` for `d = n ≥ 1`, the decaying ones for
`d = -(n+1)`.
"""
function _lame_state(d::Integer, r, μ, ν)
    T = promote_type(typeof(r), typeof(μ), typeof(ν))
    o = zero(T)
    β = (d + 5 - 4 * ν) / ((d + 1) * (2d + 3))
    γ = (d - 2 + 4 * ν) / (2d + 3)
    β′ = (d + 1) * β - 2 * (1 - ν) / (d + 1)
    γ′ = (d + 1) * γ - 2 * ν
    p = T(r)^(d - 1)
    q = 2 * μ * p / r
    r² = r * r
    return T[
        p o p * r² * β
        o -p * r / (d + 1) o
        p * d o p * r² * γ
        q * (d - 1) o q * r² * β′
        o -q * r * (d - 1) / (2 * (d + 1)) o
        q * d * (d - 1) o q * r² * γ′
    ]
end

# The radial state (u₃, τ₃) of the degree 0: the dilatation u⁽³⁾_00 = γ₀ x and
# the center of dilatation U⁽¹⁾_00 = −x/r³.
function _lame_state0(r, μ, ν)
    T = promote_type(typeof(r), typeof(μ), typeof(ν))
    γ₀ = (4 * ν - 2) / 3
    return T[γ₀ * r -1 / r^2; 2 * μ * (γ₀ - 2 * ν) 4 * μ / r^3]
end

"""
    _lame_interface_T(n, intf, r, T) -> Matrix(6×6)

Jump of the state ``(u_1, u_2, u_3, \\tau_1, \\tau_2, \\tau_3)`` of the harmonic
of degree `n ≥ 1` across the interface `intf` at the radius `r`, the outer side
from the inner one.
"""
function _lame_interface_T end

_lame_interface_T(::Integer, ::PerfectInterface, r, ::Type{T}) where {T} = _eye(T, 6)

function _lame_interface_T(::Integer, intf::SpringInterface, r, ::Type{T}) where {T}
    J = _eye(T, 6)
    sn, st = spring_compliances(intf)
    J[1, 4] = st
    J[2, 5] = st
    J[3, 6] = sn
    return J
end

function _lame_interface_T(n::Integer, intf::MembraneInterface, r, ::Type{T}) where {T}
    J = _eye(T, 6)
    κs, μs = T(intf.κs), T(intf.μs)
    r² = r * r
    ℓ = n * (n + 1)
    J[4, 1] = (ℓ * (κs + μs) - 2 * μs) / r²
    J[4, 3] = -2 * κs / r²
    J[5, 2] = (ℓ - 2) * μs / r²
    J[6, 1] = -2 * ℓ * κs / r²
    J[6, 3] = 4 * κs / r²
    return J
end

"""
    harmonic_response(sphere::LayeredSphere, C₀, n) -> Matrix(3×3)

Elastic response of a layered sphere to the harmonic of degree `n` in a matrix
of isotropic stiffness `C₀`: the matrix ``\\mathbf{Z}_n`` such that, the sphere
receiving the regular solutions with the coefficients ``a^{(j)}_{nm}`` and
sending the decaying ones with the coefficients ``b^{(i)}_{nm}``,

```math
\\underline{u} = \\sum_{j} a^{(j)}_{nm}\\,\\underline{u}^{(j)}_{nm}
  + \\sum_{i} b^{(i)}_{nm}\\,\\underline{U}^{(i)}_{nm} ,
\\qquad (n-m)!\\,(n+m)!\\;b^{(i)}_{nm} = \\sum_j (\\mathbf{Z}_n)_{ij}\\,a^{(j)}_{nm}
```

in the matrix, ``\\underline{u}^{(j)}_{nm}`` and ``\\underline{U}^{(i)}_{nm}`` being
the solutions of the Lamé equation of [`lame_regular`](@ref MeanFieldHomogenization.regular_lame_solution) and
[`lame_irregular`](@ref MeanFieldHomogenization.irregular_lame_solution). The sphere is isotropic, so the response does not
depend on the order ``m``. At the degree 0 the only regular solution is the
dilatation and the only decaying one the center of dilatation: the response
reduces to its entry `[1, 3]`.

The layers and the interface models (perfect, spring, membrane) enter through
transfer matrices, as for the uniform strain, whose two parts are the degrees
0 and 2. The linear solves pivot on the values, so the moduli are numbers,
`ForwardDiff` ones included, not symbols.
"""
function harmonic_response(sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}, n::Integer) where {T, N}
    n ≥ 0 || throw(ArgumentError("harmonic_response: the degree must be non-negative, got $n"))
    κμ = _bulk_layer_moduli(sphere)
    κ₀, μ₀ = TensND.k_mu(C₀)
    TP = _bulk_promote(sphere, κμ, κ₀, μ₀)
    radii = sphere.radii
    ν(k) = TP(_lame_poisson(κμ[k]...))
    μ(k) = TP(κμ[k][2])
    ν₀ = TP(_lame_poisson(κ₀, μ₀))
    R = TP(radii[N])
    Z = zeros(TP, 3, 3)
    if n == 0
        S = _lame_state0(TP(radii[1]), μ(1), ν(1))[:, 1:1]
        for k in 1:N
            r = TP(radii[k])
            S = _bulk_interface_T(layer_interface(sphere, k), TP(κμ[k][1]), μ(k), r) * S
            k < N && (S = _lame_state0(TP(radii[k + 1]), μ(k + 1), ν(k + 1)) * (_lame_state0(r, μ(k + 1), ν(k + 1)) \ S))
        end
        M = _lame_state0(R, TP(μ₀), ν₀)
        Z[1, 3] = (hcat(M[:, 2:2], -S) \ (-M[:, 1]))[1]
        return Z
    end
    both(r, k) = hcat(_lame_state(n, r, μ(k), ν(k)), _lame_state(-(n + 1), r, μ(k), ν(k)))
    S = _lame_state(n, TP(radii[1]), μ(1), ν(1))
    for k in 1:N
        r = TP(radii[k])
        S = _lame_interface_T(n, layer_interface(sphere, k), r, TP) * S
        k < N && (S = both(TP(radii[k + 1]), k + 1) * (both(r, k + 1) \ S))
    end
    X = hcat(_lame_state(-(n + 1), R, TP(μ₀), ν₀), -S) \ (-_lame_state(n, R, TP(μ₀), ν₀))
    Z .= @view X[1:3, :]
    return Z
end
