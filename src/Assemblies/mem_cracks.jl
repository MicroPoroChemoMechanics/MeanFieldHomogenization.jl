# =============================================================================
#  mem_cracks.jl — penny-shaped cracks in the multipole expansion method
#  (kushch2013, sections 6.2 and 6.9.2).
#
#  A penny-shaped crack of radius a and normal e₃ is the surface τ = 0 of the
#  oblate confocal coordinates of focal distance a: in the convention of
#  `mem_spheroids.jl`, c = −i·a and ξ₀ = i·0, and the two lips are the halves
#  η > 0 (x₃ = 0⁺) and η < 0 (x₃ = 0⁻) of that surface,
#
#       x₁ + i x₂ = a √(1 − η²) e^{iφ} ,    x₃ = a τ η = 0 .
#
#  The decaying harmonics F_n^m keep finite values there, so the crack is the
#  exact surface ξ = ξ₀ of the spheroidal expansion, not the limit of a thin
#  spheroid.
#
#  RESPONSE. An insulating crack makes the normal flux vanish on both lips,
#  ∂_τ(a f + b F) = 0 at τ = 0 order by order, hence
#
#       b_n^m = −P_n^m′(ξ₀)/Q_n^m′(ξ₀) a_n^m ,
#
#  the response of a cavity taken on the coordinate surface itself
#  (kushch2013, Eq. (6.50)). At ξ₀ = i·0, P_n^m′ vanishes for n − m even and
#  P_n^m for n − m odd: the crack answers only the harmonics odd in x₃, those
#  whose normal derivative does not vanish on its plane.
#
#  NEAR READING. The values of a regular field on the crack say nothing of its
#  odd part, which vanishes there; its normal derivative carries it. On the
#  plane of the crack ∂τ/∂x₃ = 1/(aη) and ∂η/∂x₃ = 0, so that
#
#       a η ∂₃(Σ a_k^l f_k^l) = i Σ a_k^l P̄_k^l′(ξ₀) χ_k^l(η, φ) ,
#
#  in which only the odd k − |l| remain; the χ_k^l of one parity being
#  orthogonal on the upper lip η ∈ (0, 1) as on the whole range, with half the
#  norm,
#
#       a_k^l = 2(2k+1)/(4π i P_k^{|l|}′(ξ₀)) ∫₀¹∫ a η ∂₃u conj(χ_k^l) dη dφ ,
#
#  for k − |l| odd; the even coefficients, which the crack does not answer,
#  are left at zero.
# =============================================================================

# A particle as the penny-shaped crack of normal e₃ whose responses are asked
# for, by the shape the type of the crack carries.
_mem_spheroid(geom::Cracks.EllipticCrack, asm, name::Symbol, prop::Symbol) = _mem_penny(geom, _crack_shape(geom), name)
_crack_shape(::Cracks.EllipticCrack{T, S}) where {T, S} = S()
function _mem_penny(geom, ::Cracks.Penny, name::Symbol)
    _mem_axis_e3(TensND.canonical_components(Cracks.crack_normal(geom)), name)
    return geom
end
_mem_penny(geom, ::Cracks.EllipticShape, name::Symbol) = throw(
    ArgumentError(
        "MultipoleExpansion: particle :$(name) is an elliptical crack of semi-axes " *
            "$(geom.a) and $(geom.b); the cracks of the expansion are penny-shaped"
    )
)

# The data of an insulating crack: response, scales of its unknowns, reading
# on its upper lip.
function _mem_spheroid_data(crack::Cracks.EllipticCrack, K₀, nmax::Int)
    T = float(typeof(crack.a))
    c = complex(zero(T), -crack.a)
    ξ₀ = complex(zero(T))
    C, R = typeof(c), T
    nb = _mem_block(nmax)
    tables = [LayeredSpheroids.legendre_order_q(ξ₀, m, nmax) for m in 0:nmax]
    Z = zeros(C, nb, nb)
    scales = zeros(R, nb)
    for m in 0:nmax
        P, dP, _, dQ = tables[m + 1]
        for n in max(m, 1):nmax, μ in (m == 0 ? (0,) : (m, -m))
            i = _mem_index(n, μ)
            # P_n^m(ξ₀) vanishes for n − m odd, P_n^m′(ξ₀) for n − m even: the
            # other one, chosen by the parity, which the derivatives also follow
            scales[i] = LayeredSpheroids._factorial_ratio(R, n, m) * abs(isodd(n - m) ? dP[n + 1] : P[n + 1])
            Z[i, i] = -dP[n + 1] / dQ[n + 1]
        end
    end
    Π, lip = _mem_lip_projection(c, [table[2] for table in tables], nmax)
    return _MemSpheroid{C, R, _MemNormalDerivative}(
        c, ξ₀, scales, Z, Π, lip, _mem_spheroid_maps(c, nb)..., _MemNormalDerivative()
    )
end

# Quadrature of the upper lip η ∈ (0, 1), φ ∈ [0, 2π) of a crack of radius a:
# the points, and their (η, φ) coordinates and weights.
function _mem_lip_points(a::R, nmax::Int) where {R}
    nη = 2 * nmax + 24
    nφ = 2 * nη
    xg, wg = LayeredSpheroids.QuadGK.gauss(nη)
    lip = zeros(R, 3, nη * nφ)
    coords = Vector{NTuple{3, R}}(undef, nη * nφ)          # (η, φ, weight)
    for j in 1:nη, i in 1:nφ
        η, w = (1 + xg[j]) / 2, wg[j] / 2                   # Gauss–Legendre on (0, 1)
        φ = 2 * R(π) * (i - 1) / nφ
        col = (j - 1) * nφ + i
        lip[:, col] .= (a * sqrt(1 - η^2) * cos(φ), a * sqrt(1 - η^2) * sin(φ), 0)
        coords[col] = (η, φ, w * (2 * R(π) / nφ))
    end
    return lip, coords
end

# The matrix Π that turns the normal derivatives on the upper lip of a regular
# field into its odd coefficients a_k^l.
function _mem_lip_projection(c::C, dP, nmax::Int) where {C}
    R = real(C)
    a = abs(c)
    lip, coords = _mem_lip_points(a, nmax)
    Π = zeros(C, _mem_block(nmax), size(lip, 2))
    for (col, (η, φ, w)) in enumerate(coords)
        Pη = [LayeredSpheroids.legendre_order_p(η, m, nmax)[1] for m in 0:nmax]
        for k in 1:nmax, l in (-k):k
            μ = abs(l)
            iseven(k - μ) && continue
            χ = _mem_surface_harmonic(Pη[μ + 1][k + 1], l, φ)
            Π[_mem_index(k, l), col] = 2 * (2k + 1) / (4 * R(π) * im * dP[μ + 1][k + 1]) * w * a * η * conj(χ)
        end
    end
    return Π, lip
end

# ─── Elasticity ──────────────────────────────────────────────────────────────
#
#  A crack sends the solutions V⁽ⁱ⁾_nm of a spheroid (`mem_spheroid_elasticity.jl`)
#  built on the decaying harmonics of focal distance c = −i·a, and its lips are
#  free of traction. The two lips are the same points of space; they are told
#  apart by their oblate coordinates (φ, p, τ = 0), p ≷ 0, on which the
#  solutions are smooth, and the strains follow from the derivatives along the
#  coordinates by the inverse Jacobian of the chart, regular on the lips off
#  the axis and the edge. The response is the least-squares solution of
#  σ·n = 0 at the Gauss points in p of one meridian, both lips together; a
#  close neighbor is read through the traction σ·e₃ of its field on the upper
#  lip, the only part of it the crack answers.

function _mem_elastic_spheroid(crack::Cracks.EllipticCrack, name::Symbol, κ₀, μ₀, ν₀, nmax::Int)
    a = crack.a
    a isa Float64 || throw(
        ArgumentError(
            "MultipoleExpansion: the radius of the crack :$(name) is a plain floating-point " *
                "number in elasticity, got $(typeof(a))"
        )
    )
    shape = _MemSpheroidShape(complex(0.0, -a))
    on = _MemOnChart(TensND.OblateSpheroidalChart(a))
    p, w = LayeredSpheroids.QuadGK.gauss(2 * nmax + 24)
    rows = [
        (
            out = _mem_values_tractions(ξ -> _mem_emitted(shape, ξ, on, nmax), ξ, on, n, κ₀, μ₀),
            remote = _mem_values_tractions(ξ -> _mem_spherical_lame(_mem_cartesian(on, ξ), nmax), ξ, on, n, κ₀, μ₀),
            inner = nothing,
            n = n,
        ) for (ξ, n) in ((SVector(0.0, pk, 0.0), SVector(0.0, 0.0, sign(pk))) for pk in p)
    ]
    Z = _mem_orderwise_response(rows, w, a / TensND.primal(μ₀), (false, false), nmax)
    K = nmax + _MEM_ELASTIC_EXTRA
    lip, coords = _mem_lip_points(a, nmax)
    return _mem_elastic_shaped(
        shape, Z, _mem_spheroid_far_map(shape.c, 2a, ν₀, nmax, K), a, K, ν₀, lip, last.(coords), nmax;
        reading = _MemNormalTraction(promote(κ₀, μ₀)...)
    )
end
