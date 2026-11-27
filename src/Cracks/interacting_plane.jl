# =============================================================================
#  interacting_plane.jl — stress intensity factors of interacting cracks of a
#  two-dimensional isotropic body, by the method of average tractions.
#
#  By superposition, the cracks of a body under the remote stress Σ are those
#  of a body at rest whose faces carry the tractions of the uncracked field,
#  each crack loaded by Σ · n and by the stress the other cracks send onto its
#  line. Kachanov (1987) keeps of the load each crack sends only the part due
#  to the mean of its own traction: a crack under a uniform traction has a field
#  in closed form (Westergaard's functions), so that the mean tractions solve a
#  linear system of size 2N, and the traction along each crack, mean of the
#  others included, gives its stress intensity factors. In an isotropic plane
#  the stresses of a loaded crack do not depend on the elastic constants.
# =============================================================================

# The stress, in the frame (m, n) of a crack of half-length l centered at the
# origin, that its faces under the uniform opening traction p and sliding
# traction τ create at the point (x, y) of that frame (Westergaard):
#   mode I,  Z = p (z/√(z² − l²) − 1):  σ_xx = Re Z − y Im Z′, σ_yy = Re Z + y Im Z′, σ_xy = −y Re Z′;
#   mode II, Z = τ (z/√(z² − l²) − 1):  σ_xx = 2 Im Z + y Re Z′, σ_yy = −y Re Z′, σ_xy = Re Z − y Im Z′;
# Z′ = −l² (z² − l²)^(−3/2). The root is √(z − l) √(z + l), cut along the
# crack and equal to z at infinity.
function _loaded_crack_stress(x, y, l, p, τ)
    z = complex(x, y)
    r = sqrt(z - l) * sqrt(z + l)
    Z = z / r - 1
    Z′ = -l^2 / r^3
    σxx = p * (real(Z) - y * imag(Z′)) + τ * (2 * imag(Z) + y * real(Z′))
    σyy = p * (real(Z) + y * imag(Z′)) - τ * y * real(Z′)
    σxy = -p * y * real(Z′) + τ * (real(Z) - y * imag(Z′))
    return SMatrix{2, 2}(σxx, σxy, σxy, σyy)
end

# One crack placed in the plane: its center, half-length, direction m and
# normal n, in the axes of the plane.
struct _PlacedSegment{T}
    c::SVector{2, T}
    l::T
    m::SVector{2, T}
    n::SVector{2, T}
end
function _placed(center, crack::RibbonCrack)
    n, m = _plane_normal(crack), _plane_direction(crack)
    T = promote_type(eltype(center), typeof(crack.b), eltype(n))
    return _PlacedSegment{T}(SVector{2, T}(center[1], center[2]), T(crack.b), SVector{2, T}(m), SVector{2, T}(n))
end

# The point at abscissa ξ along crack k, and the tractions (opening, sliding) on
# its line of the stress the faces of crack j create under (p, τ).
_point(k::_PlacedSegment, ξ) = k.c + ξ * k.m
function _traction_from(j::_PlacedSegment, k::_PlacedSegment, ξ, p, τ)
    d = _point(k, ξ) - j.c
    R = hcat(j.m, j.n)                              # frame of crack j
    σ = R * _loaded_crack_stress(transpose(j.m) * d, transpose(j.n) * d, j.l, p, τ) * transpose(R)
    t = σ * k.n
    return SVector(transpose(k.n) * t, transpose(k.m) * t)
end

# The mean over crack k of that traction, per unit (p, τ) of crack j: a 2×2 block.
# The quadratures run on fixed intervals whose bounds have the type of the
# primal number: a dual number carries its derivatives through the integrand.
_bounds_type(x) = typeof(float(TensND.primal(x)))
function _transmission(j::_PlacedSegment, k::_PlacedSegment; rtol)
    R = _bounds_type(k.l)
    col(p, τ) = first(quadgk(u -> _traction_from(j, k, u * k.l, p, τ), -one(R), one(R); rtol)) / 2
    return hcat(col(one(k.l), zero(k.l)), col(zero(k.l), one(k.l)))
end

"""
    sif(cracks, C₀, Σ; rtol = 1e-10) -> Vector of (plus, minus, traction)

Stress intensity factors of interacting cracks of a two-dimensional isotropic
body under the remote stress `Σ` (a tensor of the plane), by the method of
average tractions of [kachanov1987](@citet). `cracks` lists `center => crack`
pairs, each crack a [`RibbonCrack`](@ref) (a segment of half-length `b`) placed
at `center`, a point of the plane; `C₀`, isotropic, only says that the problem is
that of the plane (the stresses do not depend on the elastic constants).

Each crack ``k`` carries the traction ``\\boldsymbol\\Sigma\\cdot\\underline n_k`` and
the tractions the faces of the other cracks send onto its line; every crack
``j`` sending the field of its *mean* traction ``\\langle\\underline t_j\\rangle``,
the means solve

```math
\\langle\\underline t_k\\rangle = \\boldsymbol\\Sigma\\cdot\\underline n_k
  + \\sum_{j \\neq k} \\boldsymbol\\Lambda_{kj}\\cdot\\langle\\underline t_j\\rangle ,
```

``\\boldsymbol\\Lambda_{kj}`` the mean over crack ``k`` of the traction created by
crack ``j`` under a unit uniform load. The traction ``\\underline t_k(\\xi)`` along
crack ``k`` then gives, at its tips ``\\xi = \\pm l``,

```math
K_{\\mathrm I}^{\\pm} + \\mathrm i\\,K_{\\mathrm{II}}^{\\pm}
  = \\frac{1}{\\sqrt{\\pi l}}\\int_{-l}^{l}\\big(t_n(\\xi) + \\mathrm i\\,t_m(\\xi)\\big)
    \\sqrt{\\frac{l \\pm \\xi}{l \\mp \\xi}}\\,\\mathrm d\\xi ,
```

``t_n``, ``t_m`` the opening and sliding components (eq. 14 of the article).
Each entry of the result holds `plus` and `minus`, the pairs ``(K_I, K_{II})`` at
the tips ``\\pm b`` along the direction of the crack, and `traction`, the pair of
mean tractions. Numbers and dual numbers; one crack gives the
[`sif`](@ref) of an isolated one.
"""
function sif(cracks::AbstractVector{<:Pair}, C₀::_Plane4, Σ::_Plane2; rtol::Real = 1.0e-10)
    _check_isotropic_plane(C₀)
    segs = [_placed(first(c), last(c)) for c in cracks]
    S = SMatrix{2, 2}(TensND.canonical_components(Σ))
    N = length(segs)
    remote = [SVector(transpose(s.n) * S * s.n, transpose(s.m) * S * s.n) for s in segs]
    T = mapreduce(s -> eltype(s.c), promote_type, segs; init = eltype(S))
    # Λ by 2×2 blocks, Λ_kj sent by crack j onto crack k; none onto itself.
    blocks = [j == k ? zero(SMatrix{2, 2, T}) : _transmission(segs[j], segs[k]; rtol) for k in 1:N, j in 1:N]
    Λ = reduce(vcat, [reduce(hcat, map(Matrix, blocks[k, :])) for k in 1:N])   # dynamic: N varies
    t = (I - Λ) \ reduce(vcat, remote)
    means = [SVector(t[2k - 1], t[2k]) for k in 1:N]
    return [_tip_factors(segs, means, remote, k; rtol) for k in 1:N]
end

# The factors at the two tips of crack k from its traction along its line.
function _tip_factors(segs, means, remote, k; rtol)
    s = segs[k]
    t(ξ) = remote[k] + sum((_traction_from(segs[j], s, ξ, means[j]...) for j in eachindex(segs) if j != k); init = zero(remote[k]))
    # ξ = ∓l cos θ makes the weight √((l ± ξ)/(l ∓ ξ)) dξ = l (1 ∓ cos θ) dθ regular.
    R = _bounds_type(s.l)
    K(sgn) = first(quadgk(θ -> t(-sgn * s.l * cos(θ)) * s.l * (1 - cos(θ)), zero(R), R(π); rtol)) / sqrt(R(π) * s.l)
    return (plus = K(1), minus = K(-1), traction = means[k])
end

_check_isotropic_plane(::TensND.TensISO) = nothing
_check_isotropic_plane(C₀) = throw(
    ArgumentError(
        "the method of average tractions is implemented for an isotropic plane; got a " *
            "$(nameof(typeof(C₀))). An anisotropic plane needs the fields of a loaded crack " *
            "of Mauge and Kachanov (1994)"
    )
)
