# =============================================================================
#  mem_shaped_elasticity.jl — particles other than spheres in the elastic
#  multipole expansion: what spheroids (`mem_spheroid_elasticity.jl`) and
#  ellipsoids (`mem_ellipsoid_elasticity.jl`) share.
#
#  A particle enters the solver of the spheres (`mem_elasticity.jl`) through
#  its response, from the regular spherical solutions u⁽ʲ⁾_nm it receives about
#  its center to the decaying spherical solutions U⁽ⁱ⁾_nm it sends. Its own
#  decaying solutions, adapted to its shape (the `shape` methods
#  `_mem_emitted`, `_mem_focal`), give that response and its field close by:
#
#  RESPONSE. The emitted solutions outside, regular spherical solutions for
#  the Poisson ratio of the particle inside; on the surface, of unit normal n,
#  the traction is continuous and the displacement jumps by the compliance of
#  the interface, u₀ − u₁ = S t, S = sₙ n⊗n + sₜ (𝟏 − n⊗n), zero for a perfect
#  bond; on a cavity the traction is zero. The conditions are imposed by least
#  squares at quadrature points of the surface, the traction brought to a
#  displacement by R/μ₀.
#
#  NEAR. Closer than _MEM_NEAR times the sum of their focal distances, the
#  field a particle sends is evaluated at points of the surface of the
#  receiver and projected there, by least squares, on the regular spherical
#  solutions; farther, its spherical form and the translation of the spheres
#  apply.
# =============================================================================

# The degrees sent beyond nmax in the spherical form of a particle: a pair
# goes through that form beyond the threshold, where its terms of degree n′
# decrease at least as 5^{−n′}.
const _MEM_ELASTIC_EXTRA = 12

# Everything the elastic expansion needs of one particle of shape `shape`, in
# the scaled unknowns of `_mem_vector_system` (radius R, its largest semi-axis):
#   Ẑ        received â (degree ≤ nmax) → sent spherical b̂ (degree ≤ K);
#   W        received â → coefficients of the emitted solutions;
#   Π        displacements at the points `surface` (3 per point) → received â.
struct _MemElasticShaped{C, T, S}
    shape::S
    R::Float64
    K::Int
    ν₀::T
    Ẑ::Matrix{C}
    W::Matrix{C}
    Π::Matrix{C}
    surface::Matrix{Float64}
end

_mem_sent_degree(d::_MemElasticShaped, nmax::Int) = d.K
_mem_focal(d::_MemElasticShaped) = _mem_focal(d.shape)
_mem_convert(d::_MemElasticShaped, ::Type{C}) where {C} =
    _MemElasticShaped{C, typeof(d.ν₀), typeof(d.shape)}(d.shape, d.R, d.K, d.ν₀, d.Ẑ, d.W, d.Π, d.surface)

# The data from the natural response Z (received a → emitted coefficients) and
# the spherical form μ of the emitted solutions (emitted → sent b, degree ≤ K).
function _mem_elastic_shaped(shape, Z, μ, R, K::Int, ν₀, surface, weights, nmax::Int)
    W = Z ./ transpose(_mem_received_scales(R, nmax))
    Ẑ = _mem_sent_scales(R, K) .* (μ * W)
    Π = _mem_near_projection(surface, weights, R, ν₀, nmax)
    C = promote_type(eltype(Ẑ), eltype(W), eltype(Π))
    return _MemElasticShaped{C, typeof(ν₀), typeof(shape)}(shape, R, K, ν₀, Ẑ, W, Π, surface)
end

# ─── Spherical solutions ─────────────────────────────────────────────────────

# The regular spherical solutions u⁽ʲ⁾_nm at x, n ≤ nmax, as u = P + ν Q:
# columns by `_mem_eindex`, rows the three Cartesian components.
function _mem_spherical_lame(x, nmax::Int)
    y = regular_harmonics(x, nmax + 1)
    C = eltype(y.values)
    P, Q = zeros(C, 3, _mem_eblock(nmax)), zeros(C, 3, _mem_eblock(nmax))
    for n in 0:nmax, m in (-n):n, j in _mem_received(n)
        col = _mem_eindex(j, n, m)
        P[:, col] .= Harmonics.lame_regular(j, n, m, x, 0, y)
        j == 3 && (Q[:, col] .= -4 * Harmonics.vector_regular_harmonic(3, n, m, y))
    end
    return P, Q
end

# The decaying spherical solutions U⁽ⁱ⁾_nm at x, n ≤ K, as U = P + ν Q.
function _mem_spherical_lame_decaying(x, K::Int)
    Y = irregular_harmonics(x, K + 1)
    C = eltype(Y.values)
    P, Q = zeros(C, 3, _mem_eblock(K)), zeros(C, 3, _mem_eblock(K))
    for n in 0:K, m in (-n):n, i in _mem_sent(n)
        col = _mem_eindex(i, n, m)
        P[:, col] .= Harmonics.lame_irregular(i, n, m, x, 0, Y)
        i == 3 && (Q[:, col] .= -4 * Harmonics.vector_irregular_harmonic(3, n, m, Y))
    end
    return P, Q
end

# â = D a: the received coefficients scaled to a displacement at radius R, and
# b̂ = D′ b the sent ones (`mem_elasticity.jl`, SCALING).
_mem_received_scales(R, nmax::Int) =
    [(n == 0 ? R : R^(n - 2 + j) / harmonic_norm(n, m)) for n in 0:nmax for m in (-n):n for j in _mem_received(n)]
_mem_sent_scales(R, K::Int) =
    [(n == 0 ? 1 / R^2 : harmonic_norm(n, m) / R^(n + 3 - i)) for n in 0:K for m in (-n):n for i in _mem_sent(n)]

# ─── Response ────────────────────────────────────────────────────────────────

# Values and tractions on the normal n of a family u = P + ν Q at a point of a
# phase (κ, μ): rows u₁, u₂, u₃, t₁, t₂, t₃, columns as `fields`; the gradients
# come from forward-mode differentiation of the two parts.
function _mem_values_tractions(fields::F, x::SVector{3, Float64}, n, κ, μ) where {F}
    P, Q = fields(x)
    nc = size(P, 2)
    J = ForwardDiff.jacobian(y -> _mem_packed_parts(fields(y)...), x)
    ν = LayeredSpheres._lame_poisson(κ, μ)
    λ = κ - 2μ / 3
    out = zeros(Complex{promote_type(typeof(κ), typeof(μ))}, 6, nc)
    # rows of the Jacobian for component a of column j: part offset + (j−1)·3 + a
    block(off, j) = view(J, (off + 3 * (j - 1) + 1):(off + 3 * j), :)
    L = 3 * nc
    for j in 1:nc
        G = complex.(block(0, j), block(L, j)) .+ ν .* complex.(block(2L, j), block(3L, j))
        ε = (G .+ transpose(G)) ./ 2
        out[1:3, j] .= view(P, :, j) .+ ν .* view(Q, :, j)
        out[4:6, j] .= λ * tr(ε) .* n .+ 2μ .* (ε * n)
    end
    return out
end

# The real and imaginary parts of the two parts, in one real vector.
_mem_packed_parts(P, Q) = vcat(vec(real.(P)), vec(imag.(P)), vec(real.(Q)), vec(imag.(Q)))

# The interface conditions at the points of the surface, by `_mem_values_tractions`:
# for each point x with normal n and weight w, the families sent outside
# (`emitted`), received (`remote`) and, for an inclusion of moduli (κ₁, μ₁),
# regular inside. Returns the rows for each point.
function _mem_interface_rows(emitted::F, points, normals, κ₀, μ₀, κ₁, μ₁, nmax::Int) where {F}
    cavity = _mem_is_cavity(κ₁, μ₁)
    return [
        (
            out = _mem_values_tractions(emitted, x, n, κ₀, μ₀),
            remote = _mem_values_tractions(y -> _mem_spherical_lame(y, nmax), x, n, κ₀, μ₀),
            inner = cavity ? nothing : _mem_values_tractions(y -> _mem_spherical_lame(y, nmax), x, n, κ₁, μ₁),
            n = n,
        ) for (x, n) in zip(points, normals)
    ]
end

_mem_is_cavity(κ₁, μ₁) = iszero(TensND.primal(μ₁)) && iszero(TensND.primal(κ₁))

# The least-squares system for the emitted columns `sent` and the received
# columns `received`: unknowns the emitted coefficients and, for an inclusion,
# those of the inner field; one block of rows per point, weighted by √w and
# the traction by `tscale`; (sₙ, sₜ) the compliances of the interface.
function _mem_interface_system(rows, sent, received, w, tscale, compliances)
    blocks = [
        _mem_interface_block(r.out[:, sent], r.remote[:, received], r.inner, received, r.n, sqrt(wk), tscale, compliances)
            for (r, wk) in zip(rows, w)
    ]
    return reduce(vcat, first.(blocks)), reduce(vcat, last.(blocks))
end

# The rows of one point: on a cavity the traction alone, zero; on an inclusion
# the jump of displacement and the continuity of traction.
_mem_interface_block(out, remote, ::Nothing, received, n, k, tscale, compliances) =
    (k * tscale) .* (out[4:6, :], -remote[4:6, :])
function _mem_interface_block(out, remote, inner_all::AbstractMatrix, received, n, k, tscale, (sₙ, sₜ))
    inner = inner_all[:, received]
    S = sₙ * n * transpose(n) + sₜ * (I - n * transpose(n))
    jump = inner[1:3, :] .+ S * inner[4:6, :]
    lhs = vcat(hcat(out[1:3, :], -jump), tscale .* hcat(out[4:6, :], -inner[4:6, :]))
    rhs = vcat(-remote[1:3, :], -tscale .* remote[4:6, :])
    return (k .* lhs, k .* rhs)
end

# The compliances (sₙ, sₜ) of an interface in elasticity.
_mem_compliances(::LayeredSpheres.PerfectInterface) = (false, false)
_mem_compliances(s::LayeredSpheres.SpringInterface) = (s.sn, s.st)
_mem_compliances(intf) = throw(
    ArgumentError(
        "MultipoleExpansion: in elasticity a particle is bonded perfectly or by springs, not " *
            "by a $(nameof(typeof(intf)))"
    )
)

# A \ b with the columns of A brought to unit norm first (on the primal values),
# so that harmonics of different degrees weigh alike in the least squares.
function _mem_column_scaled_solve(A, b)
    s = [inv(norm(TensND.primal.(view(A, :, j)))) for j in axes(A, 2)]
    return s .* ((A .* transpose(s)) \ b)
end

# ─── Near pairs ──────────────────────────────────────────────────────────────

# The projection of the displacements at points of the surface (3 rows per
# point, weights w) on the scaled regular spherical solutions about the center.
function _mem_near_projection(surface, weights, R, ν₀, nmax::Int)
    A = reduce(
        vcat, [
            (u = _mem_spherical_lame(SVector{3}(view(surface, :, k)), nmax); sqrt(wk) .* (u[1] .+ ν₀ .* u[2]))
                for (k, wk) in enumerate(weights)
        ]
    )
    # the pseudo-inverse of A, times the weights of the rows, by its QR factors
    s = [inv(norm(TensND.primal.(view(A, :, j)))) for j in axes(A, 2)]
    F = qr(A .* transpose(s))
    return (s .* _mem_received_scales(R, nmax)) .* (F.R \ (Matrix(F.Q)' .* transpose(repeat(sqrt.(weights); inner = 3))))
end

# The received coefficients of q due to p at −r from it, read on the surface of q.
function _mem_elastic_near(dq::_MemElasticShaped, dp::_MemElasticShaped, r, nmax::Int)
    U = reduce(
        vcat, [
            (V = _mem_emitted(dp.shape, r .+ view(dq.surface, :, k), nmax); V[1] .+ dp.ν₀ .* V[2])
                for k in axes(dq.surface, 2)
        ]
    )
    return dq.Π * (U * dp.W)
end

function _mem_couple_elastic!(M, rows, cols, b::MixedBC, dq::_MemElasticShaped, dp::_MemElasticShaped, r, self::Bool, ν₀, norms, nmax::Int)
    self && return M
    if norm(TensND.primal.(r)) < _MEM_NEAR * (_mem_focal(dq) + _mem_focal(dp))
        @views M[rows, cols] .-= _mem_elastic_near(dq, dp, r, nmax)
        return M
    end
    return _mem_couple_far!(M, rows, cols, b, (irregular_harmonics(r, nmax + dp.K), r), dq, dp, ν₀, norms, nmax)
end

# On a periodic cell the images closer than the threshold are read on the
# surface of q and left out of the lattice sums.
function _mem_couple_elastic!(M, rows, cols, b::PeriodicBox, dq::_MemElasticShaped, dp::_MemElasticShaped, r, self::Bool, ν₀, norms, nmax::Int)
    L = box_periods(b)
    rn = _MEM_NEAR * (_mem_focal(dq) + _mem_focal(dp))
    near = _mem_near_images(r, L, rn, self)
    _mem_couple_far!(M, rows, cols, b, _mem_far_sums(r, L, near, rn, nmax + dp.K, Val(true)), dq, dp, ν₀, norms, nmax)
    for n in near
        @views M[rows, cols] .-= _mem_elastic_near(dq, dp, r .+ (n[1] * L[1], n[2] * L[2], n[3] * L[3]), nmax)
    end
    return M
end
