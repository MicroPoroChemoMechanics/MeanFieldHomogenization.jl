# ── The hole ─────────────────────────────────────────────────────────────────

"""
    ConformalShape

Shape tag of the holes given by a conformal map.
"""
struct ConformalShape end

"""
    ConformalHole(coefficients; euler_angles = (), basis = nothing)
    ConformalHole(degree => coefficient, ...; euler_angles = (), basis = nothing)

An infinite cylindrical cavity of an isotropic matrix, whose cross-section is
bounded by the image of the unit circle by the map

```math
z = R\\,\\Bigl(\\zeta + \\sum_{k=1}^{N} m_k\\,\\zeta^{-k}\\Bigr),\\qquad |\\zeta| = 1 ,
```

with `coefficients = (m₁, …, m_N)`, real or complex, or the nonzero ones given
as pairs. As for a `Cylinder`, the axis is the first vector of `basis` (built
from the ZYZ `euler_angles` when no basis is given) and the complex plane
``z = x_2 + \\mathrm{i}\\,x_3`` is that of the two others. The size ``R`` does
not enter: the tensors are per unit volume of the cavity.

`m₁ = m` alone is the ellipse of semi-axes ``R(1 \\pm m)``; `n => m` alone is a
hypotrochoid, with ``n + 1`` rounded corners; [`polygon_map`](@ref) gives the
terms of a regular polygon. The coefficients must satisfy
``\\sum k\\,|m_k| \\le 1``, which makes the map one-to-one; the equality gives
cusps.

The hole is a cavity in elasticity and an insulator in conduction. Its strain
concentration tensor, the mean strain of the cavity per unit remote strain,
gathers three exact solutions: the plane-strain problem, whose complex
potentials are finite sums for this map; the antiplane shear, a problem of
Laplace solved in closed form; and the axial extension, which the cavity does
not disturb. In conduction the gradient concentration tensor is, in the axes of
the hole,

```math
\\boldsymbol{A} = \\underline{e}_1\\otimes\\underline{e}_1 + \\frac{2}{1 - \\sum k\\,|m_k|^2}
\\begin{pmatrix} 1 - \\operatorname{Re} m_1 & -\\operatorname{Im} m_1\\\\ -\\operatorname{Im} m_1 & 1 + \\operatorname{Re} m_1\\end{pmatrix}_{(\\underline{e}_2,\\,\\underline{e}_3)} .
```

The stress, or flux, concentration vanishes. The property of the phase is not
read and the reference medium must be isotropic. The coefficients are numbers,
duals or symbols; with symbols the condition on the coefficients is not
checked.

# Examples
```jldoctest
julia> using TensND

julia> square = ConformalHole(polygon_map(4, 3)...);

julia> A = gradient_gradient_concentration(square, TensISO{3}(0.0), TensISO{3}(1.0));

julia> round.(canonical_components(A); digits = 4)
3×3 Matrix{Float64}:
 1.0  0.0    0.0
 0.0  2.188  0.0
 0.0  0.0    2.188
```

See also [`ConformalShape`](@ref).
"""
struct ConformalHole{T <: Number, B <: TensND.AbstractBasis} <: Core.AbstractCustomInclusion{T}
    coefficients::Vector{Complex{T}}
    basis::B
    # the map is checked here, whatever the constructor it comes from
    ConformalHole{T, B}(coefficients, basis) where {T <: Number, B <: TensND.AbstractBasis} =
        (_check_map(coefficients); new{T, B}(coefficients, basis))
end

function ConformalHole(
        coefficients::Union{Tuple{Vararg{Number}}, AbstractVector{<:Number}};
        euler_angles::Tuple{Vararg{Real}} = (), basis::Union{Nothing, TensND.AbstractBasis} = nothing,
    )
    T = TensND.float_like(mapreduce(c -> typeof(real(c)), promote_type, coefficients; init = Int))
    # the circle is the map with m₁ = 0: one coefficient at least is kept
    m = Complex{T}[k ≤ length(coefficients) ? coefficients[k] : zero(T) for k in 1:max(length(coefficients), 1)]
    b = _hole_basis(basis, euler_angles, T)
    return ConformalHole{T, typeof(b)}(m, b)
end
function ConformalHole(first::Pair{<:Integer, <:Number}, others::Pair{<:Integer, <:Number}...; kw...)
    terms = (first, others...)
    all(t -> t.first ≥ 1, terms) || throw(
        ArgumentError("the degrees of the terms of a ConformalHole are 1 at least; got $(map(t -> t.first, terms))")
    )
    T = mapreduce(t -> typeof(t.second), promote_type, terms)
    m = zeros(T, maximum(t -> t.first, terms))
    for t in terms
        m[t.first] += t.second
    end
    return ConformalHole(m; kw...)
end

_hole_basis(basis::TensND.AbstractBasis, _, _) = basis
_hole_basis(::Nothing, euler_angles, ::Type{T}) where {T} =
    TensND.euler_basis(euler_angles; T = TensND.canonical_eltype(T))

Base.show(io::IO, h::ConformalHole{T}) where {T} =
    print(io, "ConformalHole{", T, "} (", length(h.coefficients), " coefficient(s) of the map)")

Core.dimension(::ConformalHole) = 3
Core.inclusion_basis(h::ConformalHole) = h.basis
Core.shape_trait(::ConformalHole) = ConformalShape()
# No single property describes a cavity: the contributions follow from the two
# concentration tensors.
Core.is_homogeneous_inclusion(::ConformalHole) = false

# The three solutions hold in an isotropic medium.
_require_isotropic(P₀::TensND.TensISO{order, 3}) where {order} = P₀
_require_isotropic(P₀::TensND.AbstractTens) = throw(
    ArgumentError(
        "a ConformalHole needs an isotropic reference medium (a TensISO), got a " *
            "$(nameof(typeof(P₀))). Register the phase with `symmetrize = IsoSymmetrize()`, " *
            "or give an isotropic reference."
    )
)

"""
    _potential_coefficients(m) -> Matrix

Coefficients ``a_k`` of the potential
``\\varphi(\\zeta) = \\Gamma\\,\\zeta + \\sum_k a_k\\,\\zeta^{-k}`` of the hole of
coefficients `m`, for the three remote stresses
``\\underline{e}_1\\otimes\\underline{e}_1``,
``\\underline{e}_2\\otimes\\underline{e}_2`` and
``(\\underline{e}_1\\otimes\\underline{e}_2 + \\underline{e}_2\\otimes\\underline{e}_1)/\\sqrt2``,
one column each, together with the pairs ``(\\Gamma, \\Gamma')`` of these
loadings.

With ``g_l`` the coefficients of ``1/\\omega'``, the free surface of the hole
gives, for ``1 \\le j \\le N``,

```math
\\overline{a_j} - \\sum_{l} W_{jl}\\,a_l = -\\Gamma'\\,\\delta_{j1} - \\Gamma\\sum_{k \\ge j}\\overline{m_k}\\,g_{k-j},
\\qquad W_{jl} = l\\sum_{k \\ge j+l+1}\\overline{m_k}\\,g_{k-j-l-1} ,
```

a system on the real and imaginary parts of the ``a_k``.
"""
function _potential_coefficients(m::Vector{Complex{T}}) where {T}
    N = length(m)
    g = _inverse_derivative_series(m, N - 1)
    W = [l * sum(conj(m[k]) * g[k - j - l] for k in (j + l + 1):N; init = zero(Complex{T})) for j in 1:N, l in 1:N]
    c = [sum(conj(m[k]) * g[k - j + 1] for k in j:N) for j in 1:N]
    half, quarter, r = one(T) / 2, one(T) / 4, 1 / sqrt(2 * one(T))
    loads = ((quarter, Complex(-half, zero(T))), (quarter, Complex(half, zero(T))), (zero(T), Complex(zero(T), r)))
    rhs = [-(j == 1 ? Γ′ : zero(Complex{T})) - Γ * c[j] for j in 1:N, (Γ, Γ′) in loads]
    U, V = real.(W), imag.(W)
    # ā - W a = r, on a = x + i y: (1 - U) x + V y = Re r and -V x - (1 + U) y = Im r
    identity = [T(j == l) for j in 1:N, l in 1:N]
    xy = [identity - U V; -V -identity - U] \ [real.(rhs); imag.(rhs)]
    return complex.(xy[1:N, :], xy[(N + 1):(2N), :]), loads
end

"""
    _hole_compliance(m) -> SMatrix{3, 3}

Compliance contribution of the hole of coefficients `m` per unit area, in the
Kelvin-Mandel notation of its own axes, divided by ``(\\kappa + 1)/\\mu``,
where ``\\mu`` is the shear modulus of the matrix and ``\\kappa`` its Kolosov
constant.

On the free surface the displacement is ``2\\mu\\,u = (\\kappa + 1)\\,\\varphi``,
and the mean strain of the hole follows from two contour integrals: with
``A = 1 - \\sum k\\,|m_k|^2``,

```math
\\operatorname{tr}\\boldsymbol\\varepsilon = \\frac{\\kappa+1}{\\mu\\,A}\\operatorname{Re}\\Bigl(\\Gamma - \\sum_k k\\,\\overline{m_k}\\,a_k\\Bigr),
\\qquad
\\varepsilon_{11} - \\varepsilon_{22} + 2\\mathrm{i}\\,\\varepsilon_{12} = \\frac{\\kappa+1}{\\mu\\,A}\\,\\bigl(a_1 - \\Gamma\\,m_1\\bigr) .
```
"""
function _hole_compliance(m::Vector{Complex{T}}) where {T}
    a, loads = _potential_coefficients(m)
    area = _hole_area(m)
    columns = map(1:3) do q
        Γ = loads[q][1]
        trace = real(Γ - sum(k * conj(m[k]) * a[k, q] for k in eachindex(m))) / area
        deviator = (a[1, q] - Γ * m[1]) / area
        ((trace + real(deviator)) / 2, (trace - real(deviator)) / 2, imag(deviator) / sqrt(2 * one(T)))
    end
    return SMatrix{3, 3}(columns[1]..., columns[2]..., columns[3]...)
end

# Gradient concentration of the insulating cross-section, in the plane of the
# hole: the temperature is Re(ḡ ζ + g/ζ) for a remote gradient g, and the
# integral of T n over the contour is 2π (g - m₁ ḡ).
function _plane_gradient_concentration(m::Vector{<:Complex})
    scale = 2 / _hole_area(m)
    off = 0 - imag(m[1])                 # not `-imag`: no negative zero for a real m₁
    return scale * SMatrix{2, 2}(1 - real(m[1]), off, off, 1 + real(m[1]))
end

"""
    Core.gradient_gradient_concentration(hole::ConformalHole, P₁, P₀)

Mean strain (stiffness `P₀` of order 4) or mean temperature gradient
(conductivity `P₀` of order 2) of the cavity per unit remote one, the boundary
values being those of the matrix. `P₁` is not read; `P₀` must be isotropic.

The compliance contribution ``\\mathbb H`` of the cavity is assembled in the
Kelvin-Mandel notation of its axes, the first being that of the cylinder, and
the result is ``\\mathbb H : \\mathbb C_0``. With ``E``, ``\\nu`` and ``\\mu``
the moduli of the matrix:

- in the plane ``(2, 3)``, the plane-strain compliance of the hole,
  ``4(1-\\nu)/\\mu`` times the matrix of `_hole_compliance`;
- an axial stress ``\\sigma_{11} - \\nu\\,(\\sigma_{22} + \\sigma_{33})``, which
  brings the remote axial stress of the plane-strain state to its value, adds
  the uniform strain it produces in the matrix;
- an antiplane shear ``\\sigma_{1\\beta}`` gives
  ``\\varepsilon_{1\\alpha} = A_{\\alpha\\beta}\\,\\sigma_{1\\beta}/(2\\mu)``,
  with ``\\boldsymbol A`` the gradient concentration of the insulating section.
"""
function Core.gradient_gradient_concentration(
        h::ConformalHole, ::TensND.AbstractTens{4, 3}, C₀::TensND.AbstractTens{4, 3}; kw...
    )
    α, β = TensND.get_data(_require_isotropic(C₀))        # 3K and 2μ
    ν, E, μ = (α - β) / (2α + β), 3α * β / (2α + β), β / 2
    P = (8 * (α + 2β) / (β * (2α + β))) * _hole_compliance(h.coefficients)
    S = _plane_gradient_concentration(h.coefficients) / (2μ)
    c, o = ν^2 / E, zero(E)
    H = SMatrix{6, 6}(
        1 / E, -ν / E, -ν / E, o, o, o,
        -ν / E, P[1, 1] + c, P[2, 1] + c, P[3, 1], o, o,
        -ν / E, P[1, 2] + c, P[2, 2] + c, P[3, 2], o, o,
        o, P[1, 3], P[2, 3], P[3, 3], o, o,
        o, o, o, o, S[2, 2], S[1, 2],
        o, o, o, o, S[2, 1], S[1, 1],
    )
    J = SMatrix{6, 6}(i ≤ 3 && j ≤ 3 ? 1 // 3 : 0 // 1 for i in 1:6, j in 1:6)
    K = SMatrix{6, 6}((i == j) - J[i, j] for i in 1:6, j in 1:6)
    return TensND.from_mandel(H * (α * J + β * K), h.basis)
end

function Core.gradient_gradient_concentration(
        h::ConformalHole, ::TensND.AbstractTens{2, 3}, K₀::TensND.AbstractTens{2, 3}; kw...
    )
    _require_isotropic(K₀)
    S = _plane_gradient_concentration(h.coefficients)
    # symmetric, in the Kelvin-Mandel notation: (11, 22, 33, √2·23, √2·13, √2·12)
    o = zero(eltype(S))
    return TensND.from_mandel(SVector{6}(one(o), S[1, 1], S[2, 2], sqrt(2 * one(o)) * S[2, 1], o, o), h.basis)
end

"""
    Core.flux_gradient_concentration(hole::ConformalHole, P₁, P₀)

Zero: the cavity carries neither stress nor flux.
"""
function Core.flux_gradient_concentration(
        h::ConformalHole{T}, ::TensND.AbstractTens{4, 3}, C₀::TensND.AbstractTens{4, 3}; kw...
    ) where {T}
    _require_isotropic(C₀)
    z = zero(promote_type(T, eltype(C₀)))
    return TensND.TensISO{3}(z, z)
end
function Core.flux_gradient_concentration(
        h::ConformalHole{T}, ::TensND.AbstractTens{2, 3}, K₀::TensND.AbstractTens{2, 3}; kw...
    ) where {T}
    _require_isotropic(K₀)
    return TensND.TensISO{3}(zero(promote_type(T, eltype(K₀))))
end
