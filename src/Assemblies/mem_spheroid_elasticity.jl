# =============================================================================
#  mem_spheroid_elasticity.jl — the multipole expansion method for spheroids
#  in elasticity (kushch2013, chapter 7).
#
#  SOLUTIONS. With F̃_t^s the renormalized spheroidal harmonics of
#  `_mem_spheroidal_table`, which tend to the spherical Y_t^s far from the
#  spheroid, the decaying solutions of the Lamé equation about a spheroid are
#  built as those of the sphere, with the vector harmonics F̃⁽²⁾, F̃⁽³⁾ of the
#  table (kushch2013, Eq. (7.2)):
#
#       V⁽¹⁾_nm = grad F̃_n^m ,   V⁽²⁾_nm = F̃⁽²⁾_nm ,
#       V⁽³⁾_nm = 4(1−ν₀) F̃⁽³⁾_nm − grad(x·F̃⁽³⁾_nm) ,
#
#  a potential field, a harmonic divergence-free one and a Papkovich-Neuber
#  field. For the sphere grad Y_n^m = Y⁽¹⁾_nm; for a spheroid grad F̃_n^m is not a
#  finite combination of the F̃⁽¹⁾, and taking it as the first family puts the
#  field of a spheroid under a uniform remote strain (Eshelby's) in the
#  solutions of degree 2 and below, as the correction −c ξ₀² grad f_t^s of
#  kushch2013, Eq. (7.16), does for the regular ones. The derivatives of the
#  table come from forward-mode differentiation.
#
#  RESPONSE. A spheroid of revolution axis e₃, perfectly bonded or a cavity,
#  receives the regular spherical solutions u⁽ʲ⁾_nm about its center and sends
#  the V⁽ⁱ⁾_nm; inside, the field is a series of regular spherical solutions
#  for its own Poisson ratio. Continuity of displacement and traction on its
#  surface (traction zero on a cavity) is imposed by least squares at points of
#  one meridian, order by order: the spheroid and its phases being invariant
#  about e₃, the order m is kept, and a field of order m is known everywhere
#  from its values on one meridian.
#
#  SPHERICAL FORM. Outside the sphere that holds its focal segment or disk, the
#  field a spheroid sends is a series of the decaying spherical solutions,
#  whose coefficients are fitted by least squares on a sphere of twice its
#  largest semi-axis, order by order, up to the degree nmax + 2 _MEM_FAR_TERMS.
#  Far pairs and the images of a periodic cell go through this form and the
#  translation of the spheres; the pairs closer than _MEM_NEAR times the sum of
#  their focal distances are read on the surface of the receiver, where the
#  field sent is projected on the regular spherical solutions by least squares.
# =============================================================================

# Everything the elastic expansion needs of one spheroid, in the scaled
# unknowns of `_mem_vector_system` (radius R, the largest semi-axis):
#   Ẑ        received â (degree ≤ nmax) → sent spherical b̂ (degree ≤ K);
#   W        received â → coefficients of the V⁽ⁱ⁾_nm it sends (degree ≤ nmax);
#   Π        displacements at the points `surface` (3 per point) → received â.
struct _MemElasticSpheroid{C, T}
    c::ComplexF64
    R::Float64
    K::Int
    ν₀::T
    Ẑ::Matrix{C}
    W::Matrix{C}
    Π::Matrix{C}
    surface::Matrix{Float64}
end

_mem_sent_degree(d::_MemElasticSpheroid, nmax::Int) = d.K
_mem_focal(d::_MemElasticSpheroid) = abs(d.c)

# ─── The solutions about a spheroid ──────────────────────────────────────────

# The decaying solutions V⁽ⁱ⁾_nm at x, n ≤ nmax, as V = P + ν Q: columns by
# `_mem_eindex`, rows the three Cartesian components.
function _mem_spheroidal_lame(x, c, nmax::Int)
    deg = nmax + 1
    table(y) = _mem_spheroidal_table(y, c, deg)
    F̃ = table(x)
    np = length(F̃.values)
    J = ForwardDiff.jacobian(y -> (v = table(y).values; vcat(real.(v), imag.(v))), x)
    ∂F̃ = ntuple(k -> Harmonics.HarmonicTable(deg, complex.(view(J, 1:np, k), view(J, (np + 1):(2np), k))), 3)
    xs = SVector{3}(x)
    C = eltype(F̃.values)
    P, Q = zeros(C, 3, _mem_eblock(nmax)), zeros(C, 3, _mem_eblock(nmax))
    for n in 0:nmax, m in (-n):n, i in _mem_sent(n)
        col = _mem_eindex(i, n, m)
        if i == 1
            P[:, col] .= (∂F̃[1][n, m], ∂F̃[2][n, m], ∂F̃[3][n, m])
        elseif i == 2
            P[:, col] .= Harmonics.vector_irregular_harmonic(2, n, m, F̃)
        else
            ψ = Harmonics.vector_irregular_harmonic(3, n, m, F̃)
            # grad(x·ψ)_k = ψ_k + x·∂_kψ
            g = ψ + SVector(ntuple(k -> transpose(xs) * Harmonics.vector_irregular_harmonic(3, n, m, ∂F̃[k]), 3))
            P[:, col] .= 4 * ψ - g
            Q[:, col] .= -4 * ψ
        end
    end
    return P, Q
end

# The regular spherical solutions u⁽ʲ⁾_nm at x, n ≤ nmax, as u = P + ν Q.
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

# ─── The response of a spheroid ──────────────────────────────────────────────

# Points of the meridian φ = 0 of the spheroid of semi-axes (ρ₀, ρ₀, z₀), with
# their unit normals and the Gauss weights in η.
function _mem_meridian(ρ₀, z₀, npts::Int)
    η, w = LayeredSpheroids.QuadGK.gauss(npts)
    points = [SVector(ρ₀ * sqrt(1 - e^2), 0.0, z₀ * e) for e in η]
    normals = [normalize(SVector(p[1] / ρ₀^2, 0.0, p[3] / z₀^2)) for p in points]
    return points, normals, w
end

# The columns of the order m among those of a family, sent or received.
_mem_order_columns(nmax::Int, m::Int, families) =
    [_mem_eindex(i, n, m) for n in max(abs(m), 0):nmax for i in families(n) if n > 0 || m == 0]

# Response in natural units, received a (spherical) → sent coefficients of the
# V⁽ⁱ⁾_nm, of a spheroid of focal distance c, semi-axes (ρ₀, ρ₀, z₀), in the
# matrix (κ₀, μ₀), of moduli (κ₁, μ₁), zero for a cavity.
function _mem_spheroid_elastic_response(c, ρ₀, z₀, κ₀, μ₀, κ₁, μ₁, nmax::Int)
    points, normals, w = _mem_meridian(ρ₀, z₀, 2 * nmax + 24)
    R = max(ρ₀, z₀)
    cavity = iszero(TensND.primal(μ₁)) && iszero(TensND.primal(κ₁))
    rows = [
        (
            out = _mem_values_tractions(y -> _mem_spheroidal_lame(y, c, nmax), x, n, κ₀, μ₀),
            remote = _mem_values_tractions(y -> _mem_spherical_lame(y, nmax), x, n, κ₀, μ₀),
            inner = cavity ? nothing : _mem_values_tractions(y -> _mem_spherical_lame(y, nmax), x, n, κ₁, μ₁),
        ) for (x, n) in zip(points, normals)
    ]
    C = promote_type(eltype(rows[1].out), eltype(rows[1].remote), cavity ? Bool : eltype(rows[1].inner))
    Z = zeros(C, _mem_eblock(nmax), _mem_eblock(nmax))
    # tractions brought to a displacement, R/μ₀; weights of the quadrature
    scale = [ones(3); fill(R / TensND.primal(μ₀), 3)]
    for m in (-nmax):nmax
        sent = _mem_order_columns(nmax, m, _mem_sent)
        received = _mem_order_columns(nmax, m, _mem_received)
        A, b = _mem_order_system(rows, sent, received, w, scale, cavity)
        Z[sent, received] .= _mem_column_scaled_solve(A, b)[1:length(sent), :]
    end
    return Z
end

# The least-squares system of one order: unknowns the sent coefficients and,
# for an inclusion, those of the inner field; one block of rows per point.
function _mem_order_system(rows, sent, received, w, scale, cavity::Bool)
    pick(r) = cavity ? (4:6) : (1:6)
    blocks = map(zip(rows, w)) do (r, wk)
        k = sqrt(wk) .* scale[pick(r)]
        lhs = cavity ? r.out[pick(r), sent] : hcat(r.out[:, sent], -r.inner[:, received])
        (k .* lhs, -(k .* r.remote[pick(r), received]))
    end
    return reduce(vcat, first.(blocks)), reduce(vcat, last.(blocks))
end

# A \ b with the columns of A brought to unit norm first (on the primal values),
# so that harmonics of different degrees weigh alike in the least squares.
function _mem_column_scaled_solve(A, b)
    s = [inv(norm(TensND.primal.(view(A, :, j)))) for j in axes(A, 2)]
    return s .* ((A .* transpose(s)) \ b)
end

# ─── The spherical form and the near projection ─────────────────────────────

# The coefficients of the decaying spherical solutions (degree ≤ K) of the
# field of the V⁽ⁱ⁾_nm (degree ≤ nmax), fitted on the sphere of radius r₀.
function _mem_spheroid_far_map(c, r₀, ν₀, nmax::Int, K::Int)
    θ, w = LayeredSpheroids.QuadGK.gauss(K + 24)
    points = [SVector(r₀ * sqrt(1 - p^2), 0.0, r₀ * p) for p in θ]
    V = [_mem_spheroidal_lame(x, c, nmax) for x in points]
    U = [_mem_spherical_lame_decaying(x, K) for x in points]
    μ = zeros(Complex{promote_type(Float64, typeof(ν₀))}, _mem_eblock(K), _mem_eblock(nmax))
    for m in (-nmax):nmax
        sent = _mem_order_columns(nmax, m, _mem_sent)
        spherical = _mem_order_columns(K, m, _mem_sent)
        A = reduce(vcat, [sqrt(wk) .* (u[1][:, spherical] .+ ν₀ .* u[2][:, spherical]) for (u, wk) in zip(U, w)])
        b = reduce(vcat, [sqrt(wk) .* (v[1][:, sent] .+ ν₀ .* v[2][:, sent]) for (v, wk) in zip(V, w)])
        μ[spherical, sent] .= _mem_column_scaled_solve(A, b)
    end
    return μ
end

# The projection of the displacements at points of the surface (3 rows per
# point) on the scaled regular spherical solutions about the center.
function _mem_spheroid_near_projection(c, ξ₀, R, ν₀, nmax::Int)
    _, surface, weights = _mem_surface_projection(complex(c), complex(ξ₀), nmax)
    A = reduce(
        vcat, [
            (u = _mem_spherical_lame(SVector{3}(view(surface, :, k)), nmax); sqrt(wk) .* (u[1] .+ ν₀ .* u[2]))
                for (k, wk) in enumerate(weights)
        ]
    )
    # the pseudo-inverse of A, times the weights of the rows, by its QR factors
    s = [inv(norm(TensND.primal.(view(A, :, j)))) for j in axes(A, 2)]
    F = qr(A .* transpose(s))
    Π = (s .* _mem_received_scales(R, nmax)) .* (F.R \ (Matrix(F.Q)' .* transpose(repeat(sqrt.(weights); inner = 3))))
    return Π, surface
end

# â = D a: the received coefficients scaled to a displacement at radius R, and
# b̂ = D′ b the sent ones (`mem_elasticity.jl`, SCALING).
_mem_received_scales(R, nmax::Int) =
    [(n == 0 ? R : R^(n - 2 + j) / harmonic_norm(n, m)) for n in 0:nmax for m in (-n):n for j in _mem_received(n)]
_mem_sent_scales(R, K::Int) =
    [(n == 0 ? 1 / R^2 : harmonic_norm(n, m) / R^(n + 3 - i)) for n in 0:K for m in (-n):n for i in _mem_sent(n)]

# ─── The spheroids of an assembly ────────────────────────────────────────────

# The data of a one-layer spheroid, perfectly bonded, of isotropic moduli, in
# the matrix (κ₀, μ₀); its geometry in floating point, which the harmonics of
# its shape are built in.
function _mem_elastic_spheroid(s::LayeredSpheroids.LayeredSpheroid, name::Symbol, κ₀, μ₀, ν₀, nmax::Int)
    (length(s.q) == 1 && only(s.interfaces) isa LayeredSpheres.PerfectInterface) || throw(
        ArgumentError(
            "MultipoleExpansion: in elasticity the spheroid :$(name) is a single phase, perfectly " *
                "bonded"
        )
    )
    c, ξ₀ = complex(s.c), complex(only(s.q))
    real(typeof(c)) === Float64 && real(typeof(ξ₀)) === Float64 || throw(
        ArgumentError(
            "MultipoleExpansion: the semi-axes of the spheroid :$(name) are plain floating-point " *
                "numbers in elasticity, got $(real(typeof(c)))"
        )
    )
    κ₁, μ₁ = TensND.k_mu(only(s.moduli))
    ρ₀, z₀ = real(c * sqrt(ξ₀^2 - 1)), real(c * ξ₀)
    R = max(ρ₀, z₀)
    K = nmax + 2 * _MEM_FAR_TERMS
    Z = _mem_spheroid_elastic_response(c, ρ₀, z₀, κ₀, μ₀, κ₁, μ₁, nmax)
    W = Z ./ transpose(_mem_received_scales(R, nmax))
    Ẑ = _mem_sent_scales(R, K) .* (_mem_spheroid_far_map(c, 2R, ν₀, nmax, K) * W)
    Π, surface = _mem_spheroid_near_projection(c, ξ₀, R, ν₀, nmax)
    C = promote_type(eltype(Ẑ), eltype(W), eltype(Π))
    return _MemElasticSpheroid{C, typeof(ν₀)}(c, R, K, ν₀, Ẑ, W, Π, surface)
end

_mem_convert(d::_MemElasticSpheroid, ::Type{C}) where {C} =
    _MemElasticSpheroid{C, typeof(d.ν₀)}(d.c, d.R, d.K, d.ν₀, d.Ẑ, d.W, d.Π, d.surface)

# The received coefficients of q due to p at −r from it, read on the surface of q.
function _mem_elastic_near(dq::_MemElasticSpheroid, dp::_MemElasticSpheroid, r, nmax::Int)
    U = reduce(
        vcat, [
            (V = _mem_spheroidal_lame(r .+ view(dq.surface, :, k), dp.c, nmax); V[1] .+ dp.ν₀ .* V[2])
                for k in axes(dq.surface, 2)
        ]
    )
    return dq.Π * (U * dp.W)
end

function _mem_couple_elastic!(M, rows, cols, b::MixedBC, dq::_MemElasticSpheroid, dp::_MemElasticSpheroid, r, self::Bool, ν₀, norms, nmax::Int)
    self && return M
    if norm(TensND.primal.(r)) < _MEM_NEAR * (_mem_focal(dq) + _mem_focal(dp))
        @views M[rows, cols] .-= _mem_elastic_near(dq, dp, r, nmax)
        return M
    end
    return _mem_couple_far!(M, rows, cols, b, (irregular_harmonics(r, nmax + dp.K), r), dq, dp, ν₀, norms, nmax)
end

# On a periodic cell the images closer than the threshold are read on the
# surface of q and left out of the lattice sums.
function _mem_couple_elastic!(M, rows, cols, b::PeriodicBox, dq::_MemElasticSpheroid, dp::_MemElasticSpheroid, r, self::Bool, ν₀, norms, nmax::Int)
    L = box_periods(b)
    rn = _MEM_NEAR * (_mem_focal(dq) + _mem_focal(dp))
    near = _mem_near_images(r, L, rn, self)
    _mem_couple_far!(M, rows, cols, b, _mem_far_sums(r, L, near, rn, nmax + dp.K, Val(true)), dq, dp, ν₀, norms, nmax)
    for n in near
        @views M[rows, cols] .-= _mem_elastic_near(dq, dp, r .+ (n[1] * L[1], n[2] * L[2], n[3] * L[3]), nmax)
    end
    return M
end
