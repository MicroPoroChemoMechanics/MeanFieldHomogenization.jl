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

# The characteristic pair (s, p) and the engineering compliances of the matrix
# are TensND's (`characteristic_pair`, `engineering_compliances`), the static
# limit of the slowness problem of its interface waves.
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
    s, p = TensND.characteristic_pair(C₀)
    a₁₁, a₂₂, a₁₂, _, a₁₆, a₂₆ = TensND.engineering_compliances(C₀)
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

# ── Conduction ───────────────────────────────────────────────────────────────

"""
    _anisotropic_ellipse_gradient(m, K₀) -> SMatrix{2, 2}

Gradient concentration, in the canonical axes, of the insulating elliptic hole
of map ``\\zeta + m/\\zeta`` in a matrix of symmetric conductivity `K₀`.

The temperature is the real part of a function of ``z_\\mu = x_1 + \\mu\\,x_2``,
with ``\\mu`` the root of positive imaginary part of
``K_{22}\\mu^2 + 2K_{12}\\mu + K_{11} = 0``, in closed form. A remote gradient
``\\underline{G}`` is ``\\operatorname{Re}(\\gamma\\,z_\\mu)``; the flux
function ``\\operatorname{Re}(c\\,f)``, ``c = K_{12} + K_{22}\\mu`` purely
imaginary, is constant on the contour for the perturbation
``-(\\gamma\\beta - \\overline{\\gamma\\alpha})/\\zeta_\\mu``, and the temperature on
the contour is ``\\operatorname{Re}(2\\,\\overline{\\gamma\\alpha}\\,\\bar\\sigma)``,
whose integral with the normal gives the mean gradient.
"""
function _anisotropic_ellipse_gradient(m::Complex, K₀::TensND.AbstractTens{2, 2})
    K = TensND.canonical_components(K₀)
    K₁₁, K₂₂, K₁₂ = K[1, 1], K[2, 2], (K[1, 2] + K[2, 1]) / 2
    μ = complex(-K₁₂ / K₂₂, sqrt(K₁₁ * K₂₂ - K₁₂^2) / K₂₂)
    ξ, η = _contour_vectors(m)
    α = (conj(ξ) + μ * conj(η)) / 2
    area = 1 - abs2(m)
    columns = map(((1, 0), (0, 1))) do (G₁, G₂)
        w = 2 * complex(G₁, (G₁ * real(μ) - G₂) / imag(μ)) * α      # 2 γ α
        (imag(w * η) / area, -imag(w * ξ) / area)
    end
    return SMatrix{2, 2}(columns[1]..., columns[2]...)
end
