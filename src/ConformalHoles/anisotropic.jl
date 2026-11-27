# =============================================================================
#  anisotropic.jl — the elliptic hole in an anisotropic matrix of the plane.
#
#  The complex representation of plane elasticity extends to an anisotropic
#  medium (Lekhnitskii 1981, sections 26 and 30): with the compliances of the
#  plane in the engineering form,
#
#       ε₁₁ = a₁₁σ₁₁ + a₁₂σ₂₂ + a₁₆σ₁₂ ,  ε₂₂ = a₁₂σ₁₁ + a₂₂σ₂₂ + a₂₆σ₁₂ ,
#       2ε₁₂ = a₁₆σ₁₁ + a₂₆σ₂₂ + a₆₆σ₁₂ ,
#
#  the stress derives from two functions Φ₁(z₁), Φ₂(z₂) of the variables
#  zₖ = x₁ + μₖ x₂, where μ₁, μ₂ are the roots of positive imaginary part of
#
#       a₁₁ μ⁴ − 2a₁₆ μ³ + (2a₁₂ + a₆₆) μ² − 2a₂₆ μ + a₂₂ = 0 .
#
#  The variable zₖ is an affine image of the plane. An ellipse is sent on an
#  ellipse, and the outside of each of them is the image of the outside of the
#  unit disk by a map with two terms, the three maps taking the same value on
#  the circle: the boundary condition is then written on the circle for the
#  three functions at once. That is special to the ellipse, and it is why a
#  `ConformalHole` with more than one term stays refused in such a matrix.
#
#  The two functions are found from a 2×2 system of determinant μ₁ − μ₂, which
#  vanishes for an isotropic matrix (μ₁ = μ₂ = i). Every quantity computed here
#  is symmetric in (μ₁, μ₂): it is written with their sum s and their product
#  p, the divisions by μ₁ − μ₂ being carried out by hand, so that the isotropic
#  matrix is an ordinary case. The pair (s, p) is itself a smooth function of
#  the compliances, the two roots kept being separated from their conjugates.
#
#  The derivation is in `docs/src/theory/complex_potentials.md`.
# =============================================================================

# z₁ / z₂ without the comparisons of the division of `Base`, which symbols do
# not take.
_cdiv(z₁, z₂) = (z₁ * conj(z₂)) / abs2(z₂)

# The compliances (a₁₁, a₂₂, a₁₂, a₆₆, a₁₆, a₂₆) read on the Kelvin-Mandel
# matrix `M` of the compliance of the plane.
function _engineering_compliances(M::AbstractMatrix)
    r = sqrt(2 * one(eltype(M)))
    return (M[1, 1], M[2, 2], M[1, 2], 2 * M[3, 3], r * M[1, 3], r * M[2, 3])
end

_canonical_compliances(C₀::TensND.AbstractTens{4, 2}) =
    _engineering_compliances(TensND.mandel(inv(C₀), TensND.CanonicalBasis{2, eltype(C₀)}()))

# ── The characteristic pair ──────────────────────────────────────────────────

# In the axes of an orthotropic matrix a₁₆ = a₂₆ = 0 and the equation is
# biquadratic: p = −√(a₂₂/a₁₁), and s² = −(2a₁₂ + a₆₆)/a₁₁ + 2p, which is
# negative for a positive definite compliance.
function _material_pair(a₁₁, a₂₂, a₁₂, a₆₆)
    ρ = sqrt(a₂₂ / a₁₁)
    return complex(zero(ρ), sqrt((2 * a₁₂ + a₆₆) / a₁₁ + 2 * ρ)), complex(-ρ)
end

# The pair in axes from which the material ones are turned by the angle of
# cosine c and sine n: x₁′ + μ′x₂′ is proportional to x₁ + μ x₂ with
# μ = (μ′c + n)/(c − μ′n), whose sum and product are rational in (s′, p′).
function _turned_pair(s′, p′, c, n)
    d = c^2 - s′ * (c * n) + p′ * n^2
    return _cdiv(s′ * (c^2 - n^2) + 2 * (c * n) * (1 - p′), d), _cdiv(p′ * c^2 + s′ * (c * n) + n^2, d)
end

"""
    _characteristic_pair(C₀::AbstractTens{4,2}) -> (s, p)

Sum and product of the two roots of positive imaginary part of the
characteristic equation of the stiffness `C₀` of the plane, in the canonical
axes.

An isotropic matrix has the double root ``\\mathrm{i}``; an orthotropic or a
square one has the pair of its own axes in closed form, turned to the
canonical ones: these three accept symbols. Any other tensor has its equation
solved numerically, and must be major-symmetric.
"""
_characteristic_pair(C₀::TensND.TensISO{4, 2}) = (complex(zero(eltype(C₀)), 2 * one(eltype(C₀))), complex(-one(eltype(C₀))))
function _characteristic_pair(C₀::Union{TensND.TensOrtho{4, 2, <:Any, 4}, TensND.TensCubic{2}})
    M = TensND.mandel_material(inv(C₀))
    e = TensND.vecbasis(TensND.frame(C₀))
    return _turned_pair(_material_pair(M[1, 1], M[2, 2], M[1, 2], 2 * M[3, 3])..., e[1, 1], e[2, 1])
end
_characteristic_pair(C₀::TensND.AbstractTens{4, 2}) = _characteristic_pair(Val(is_hard_numeric(eltype(C₀))), C₀)
_characteristic_pair(::Val{false}, C₀) = throw(
    ArgumentError(
        "the characteristic equation of a symbolic $(nameof(typeof(C₀))) of the plane is not solved in closed form; " *
            "give the matrix as a TensISO, a TensCubic or a TensOrtho"
    )
)
function _characteristic_pair(::Val{true}, C₀)
    # On the values: the norm of a null difference has no derivative.
    M = TensND.primal.(TensND.mandel(C₀, TensND.CanonicalBasis{2, eltype(C₀)}()))
    asym = maximum(abs, M - transpose(M)) / maximum(abs, M)
    asym ≤ 1.0e-10 || throw(
        ArgumentError(
            "the elliptic hole in an anisotropic matrix of the plane takes a major-symmetric stiffness; " *
                "got a relative asymmetry $asym"
        )
    )
    return _characteristic_pair(_canonical_compliances(C₀))
end

# The quartic divided by a₁₁ is (μ² − sμ + p)(μ² − s̄μ + p̄): its coefficients
# give Re s and three equations on (Im s, Re p, Im p). The roots of the
# companion matrix, computed on the values, start a Newton iteration on these,
# which brings the pair to the precision of its type and carries the
# derivatives. The roots are sorted by a comparison: numbers only.
function _characteristic_pair(a::NTuple{6, T}) where {T}
    a₁₁, a₂₂, a₁₂, a₆₆, a₁₆, a₂₆ = a
    c₃, c₂, c₁, c₀ = -2 * a₁₆ / a₁₁, (2 * a₁₂ + a₆₆) / a₁₁, -2 * a₂₆ / a₁₁, a₂₂ / a₁₁
    f₃, f₂, f₁, f₀ = Float64.(TensND.primal.((c₃, c₂, c₁, c₀)))
    roots = filter(μ -> imag(μ) > 0, eigvals([0.0 0.0 0.0 -f₀; 1.0 0.0 0.0 -f₁; 0.0 1.0 0.0 -f₂; 0.0 0.0 1.0 -f₃]))
    length(roots) == 2 || throw(
        DomainError(
            (f₃, f₂, f₁, f₀),
            "the characteristic equation of the matrix has real roots: its stiffness is not positive definite"
        )
    )
    sᵣ, c = -c₃ / 2, (c₂, c₁, c₀)
    x = SVector{3, T}(imag(roots[1] + roots[2]), real(roots[1] * roots[2]), imag(roots[1] * roots[2]))
    # The start has half the digits at worst (the companion matrix of a double
    # root) and each step doubles them: the values reach the precision of
    # `Float64` in two steps, and each further step makes one more order of
    # derivatives exact.
    for _ in 1:5
        x -= _pair_correction(x, sᵣ, c)
    end
    return complex(sᵣ, x[1]), complex(x[2], x[3])
end

# Newton correction of x = (Im s, Re p, Im p) on the three equations.
function _pair_correction(x::SVector{3, T}, sᵣ, (c₂, c₁, c₀)) where {T}
    sᵢ, pᵣ, pᵢ = x
    F = SVector{3, T}(sᵣ^2 + sᵢ^2 + 2 * pᵣ - c₂, 2 * (sᵣ * pᵣ + sᵢ * pᵢ) + c₁, pᵣ^2 + pᵢ^2 - c₀)
    J = SMatrix{3, 3, T}(2 * sᵢ, 2 * pᵢ, zero(T), 2 * one(T), 2 * sᵣ, 2 * pᵣ, zero(T), 2 * sᵢ, 2 * pᵢ)
    return J \ F
end

# ── The solution on the contour ──────────────────────────────────────────────

# The contour is x₁ = Re(ξ σ̄), x₂ = Re(η σ̄), σ = e^{iθ}, for the map ζ + m/ζ.
_contour_vectors(m::Complex) = (complex(1 + real(m), imag(m)), complex(imag(m), 1 - real(m)))

# The free surface of the hole under the remote stress (Σ₁₁, Σ₂₂, Σ₁₂):
# Φₖ = Bₖ/ζₖ with B₁ + B₂ = c₁ and μ₁B₁ + μ₂B₂ = c₂.
_boundary_constants(Σ, ξ, η) = ((Σ[3] * η - Σ[2] * ξ) / 2, (Σ[3] * ξ - Σ[1] * η) / 2)

# The three unit loads of the Kelvin-Mandel notation, as (Σ₁₁, Σ₂₂, Σ₁₂).
_unit_loads(::Type{T}) where {T} =
    (o = one(T); z = zero(T); ((o, z, z), (z, o, z), (z, z, 1 / sqrt(2 * o))))

"""
    _anisotropic_ellipse_concentration(m, C₀) -> SMatrix{3, 3}

Kelvin-Mandel matrix, in the canonical axes, of the strain concentration of
the elliptic hole of map ``\\zeta + m/\\zeta`` in the matrix of stiffness `C₀`.

The displacement added by the hole is, on the contour,
``u_1 = 2\\operatorname{Re}(U\\bar\\sigma)`` and
``u_2 = 2\\operatorname{Re}(V\\bar\\sigma)``, with

```math
U = c_2\\,(a_{11}s - a_{16}) + c_1\\,(a_{12} - a_{11}p),
\\qquad
V = c_2\\,\\Bigl(a_{12} - \\frac{a_{22}}{p}\\Bigr) + c_1\\,\\Bigl(\\frac{a_{22}\\,s}{p} - a_{26}\\Bigr) ,
```

and the mean strain it adds to that of the matrix follows from the integral
of ``\\underline{u}\\otimes\\underline{n}`` on the contour.
"""
function _anisotropic_ellipse_concentration(m::Complex, C₀::TensND.AbstractTens{4, 2})
    s, p = _characteristic_pair(C₀)
    a₁₁, a₂₂, a₁₂, _, a₁₆, a₂₆ = _canonical_compliances(C₀)
    ξ, η = _contour_vectors(m)
    area = 1 - abs2(m)
    T = promote_type(typeof(real(s)), typeof(a₁₁), typeof(area))
    p⁻¹ = _cdiv(one(p), p)
    q = s * p⁻¹                                   # s/p
    columns = map(_unit_loads(T)) do Σ
        c₁, c₂ = _boundary_constants(Σ, ξ, η)
        U = c₂ * (a₁₁ * s - a₁₆) + c₁ * (a₁₂ - a₁₁ * p)
        V = c₂ * (a₁₂ - a₂₂ * p⁻¹) + c₁ * (a₂₂ * q - a₂₆)
        ε₁₁, ε₂₂ = 2 * imag(conj(U) * η) / area, -2 * imag(conj(V) * ξ) / area
        ε₁₂ = (imag(conj(V) * η) - imag(conj(U) * ξ)) / area
        (ε₁₁, ε₂₂, sqrt(2 * one(T)) * ε₁₂)
    end
    ΔH = SMatrix{3, 3}(columns[1]..., columns[2]..., columns[3]...)
    return I + ΔH * SMatrix{3, 3}(TensND.mandel(C₀, TensND.CanonicalBasis{2, eltype(C₀)}()))
end

"""
    _anisotropic_ellipse_stress(m, s, p, Σ, θ) -> (σ₁₁, σ₂₂, σ₁₂), (x₁′, x₂′)

Stress at the point ``\\sigma = \\mathrm{e}^{\\mathrm{i}\\theta}`` of the contour
of the free elliptic hole of map ``\\zeta + m/\\zeta``, under the remote stress
`Σ = (Σ₁₁, Σ₂₂, Σ₁₂)`, the matrix having the characteristic pair `(s, p)`,
and the tangent to the contour there.

With ``(x_1', x_2')`` the derivative of the point of the contour with respect
to ``\\theta`` and ``Q = x_1'^2 + s\\,x_1'x_2' + p\\,x_2'^2``,

```math
\\sigma_{22} = \\Sigma_{22} + 2\\operatorname{Re}\\frac{\\mathrm{i}\\bar\\sigma N_0}{Q},\\quad
\\sigma_{12} = \\Sigma_{12} - 2\\operatorname{Re}\\frac{\\mathrm{i}\\bar\\sigma N_1}{Q},\\quad
\\sigma_{11} = \\Sigma_{11} + 2\\operatorname{Re}\\frac{\\mathrm{i}\\bar\\sigma N_2}{Q} ,
```

where ``N_0 = c_2x_2' - c_1(x_1' + s\\,x_2')``, ``N_1 = -c_2x_1' - c_1p\\,x_2'``
and ``N_2 = -c_2(s\\,x_1' + p\\,x_2') + c_1p\\,x_1'``.
"""
function _anisotropic_ellipse_stress(m::Complex, s, p, Σ, θ)
    ξ, η = _contour_vectors(m)
    c₁, c₂ = _boundary_constants(Σ, ξ, η)
    sθ, cθ = sincos(θ)
    σ̄ = complex(cθ, -sθ)
    x₁′, x₂′ = imag(ξ * σ̄), imag(η * σ̄)
    w = _cdiv(complex(zero(cθ), one(cθ)) * σ̄, x₁′^2 + s * (x₁′ * x₂′) + p * x₂′^2)
    N₀ = c₂ * x₂′ - c₁ * (x₁′ + s * x₂′)
    N₁ = -c₂ * x₁′ - c₁ * p * x₂′
    N₂ = -c₂ * (s * x₁′ + p * x₂′) + c₁ * p * x₁′
    return (Σ[1] + 2 * real(w * N₂), Σ[2] + 2 * real(w * N₀), Σ[3] - 2 * real(w * N₁)), (x₁′, x₂′)
end
