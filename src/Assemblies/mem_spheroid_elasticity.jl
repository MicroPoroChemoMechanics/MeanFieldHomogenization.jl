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
#  RESPONSE. A spheroid of revolution axis e₃, bonded perfectly or by springs,
#  or a cavity, receives the regular spherical solutions u⁽ʲ⁾_nm about its
#  center and sends the V⁽ⁱ⁾_nm; the interface conditions of
#  `mem_shaped_elasticity.jl` are imposed at points of one meridian, order by
#  order: the spheroid, its phases and its interface being invariant about e₃,
#  the order m is kept, and a field of order m is known everywhere from its
#  values on one meridian.
#
#  SPHERICAL FORM. Outside the sphere that holds its focal segment or disk, the
#  field a spheroid sends is a series of the decaying spherical solutions,
#  whose coefficients are fitted by least squares on a sphere of twice its
#  largest semi-axis, order by order, up to the degree nmax + _MEM_ELASTIC_EXTRA;
#  the near pairs are those of `mem_shaped_elasticity.jl`.
# =============================================================================

# The shape of a spheroid: its focal distance, complex for an oblate one.
struct _MemSpheroidShape
    c::ComplexF64
end

_mem_focal(s::_MemSpheroidShape) = abs(s.c)

# ─── The solutions about a spheroid ──────────────────────────────────────────

# The decaying solutions V⁽ⁱ⁾_nm at x, n ≤ nmax, as V = P + ν Q: columns by
# `_mem_eindex`, rows the three Cartesian components.
function _mem_emitted(s::_MemSpheroidShape, x, nmax::Int)
    table(y) = _mem_spheroidal_table(y, s.c, nmax + 1)
    J = ForwardDiff.jacobian(y -> (v = table(y).values; vcat(real.(v), imag.(v))), x)
    return _mem_emitted(table(x), J, x, nmax)
end
# The same at the point of coordinates ξ = (φ, p, τ) on the oblate chart of
# the spheroid (a lip of a crack): the table from the coordinates, its
# gradient by ∂/∂ξ · (∂x/∂ξ)⁻¹.
function _mem_emitted(s::_MemSpheroidShape, ξ, on::_MemOnChart, nmax::Int)
    table(y) = _mem_spheroidal_table(on.chart, y, nmax + 1)
    Jξ = ForwardDiff.jacobian(y -> (v = table(y).values; vcat(real.(v), imag.(v))), ξ)
    return _mem_emitted(table(ξ), Jξ / _mem_chart_jacobian(on, ξ), _mem_cartesian(on, ξ), nmax)
end
# From the table F̃ at x and the Jacobian J of its real and imaginary parts.
function _mem_emitted(F̃::TensND.SolidHarmonicTable, J, x, nmax::Int)
    deg = nmax + 1
    np = length(F̃.values)
    ∂F̃ = ntuple(k -> TensND.SolidHarmonicTable(deg, complex.(view(J, 1:np, k), view(J, (np + 1):(2np), k))), 3)
    xs = SVector{3}(x)
    C = promote_type(eltype(F̃.values), eltype(J))
    P, Q = zeros(C, 3, _mem_eblock(nmax)), zeros(C, 3, _mem_eblock(nmax))
    for n in 0:nmax, m in (-n):n, i in _mem_sent(n)
        col = _mem_eindex(i, n, m)
        if i == 1
            P[:, col] .= (∂F̃[1][n, m], ∂F̃[2][n, m], ∂F̃[3][n, m])
        elseif i == 2
            P[:, col] .= irregular_vector_harmonic(2, n, m, F̃)
        else
            ψ = irregular_vector_harmonic(3, n, m, F̃)
            # grad(x·ψ)_k = ψ_k + x·∂_kψ
            g = ψ + SVector(ntuple(k -> transpose(xs) * irregular_vector_harmonic(3, n, m, ∂F̃[k]), 3))
            P[:, col] .= 4 * ψ - g
            Q[:, col] .= -4 * ψ
        end
    end
    return P, Q
end

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
    [_mem_eindex(i, n, m) for n in abs(m):nmax for i in families(n) if n > 0 || m == 0]

# Response in natural units, received a (spherical) → sent coefficients of the
# V⁽ⁱ⁾_nm, of a spheroid of focal distance c, semi-axes (ρ₀, ρ₀, z₀), in the
# matrix (κ₀, μ₀), of moduli (κ₁, μ₁), zero for a cavity, with the
# compliances of its interface.
function _mem_spheroid_elastic_response(c, ρ₀, z₀, κ₀, μ₀, κ₁, μ₁, compliances, nmax::Int)
    points, normals, w = _mem_meridian(ρ₀, z₀, 2 * nmax + 24)
    rows = _mem_interface_rows(y -> _mem_emitted(_MemSpheroidShape(c), y, nmax), points, normals, κ₀, μ₀, κ₁, μ₁, nmax)
    return _mem_orderwise_response(rows, w, max(ρ₀, z₀) / TensND.primal(μ₀), compliances, nmax)
end

# The response from the rows of the interface conditions at the points of one
# meridian (weights w), order by order.
function _mem_orderwise_response(rows, w, tscale, compliances, nmax::Int)
    C = promote_type(eltype(rows[1].out), eltype(rows[1].remote), _mem_inner_eltype(rows[1].inner), map(typeof, compliances)...)
    Z = zeros(C, _mem_eblock(nmax), _mem_eblock(nmax))
    for m in (-nmax):nmax
        sent = _mem_order_columns(nmax, m, _mem_sent)
        received = _mem_order_columns(nmax, m, _mem_received)
        A, b = _mem_interface_system(rows, sent, received, w, tscale, compliances)
        Z[sent, received] .= _mem_column_scaled_solve(A, b)[1:length(sent), :]
    end
    return Z
end
_mem_inner_eltype(::Nothing) = Bool          # a cavity or a crack: no inner field
_mem_inner_eltype(inner) = eltype(inner)

# ─── The spherical form ──────────────────────────────────────────────────────

# The coefficients of the decaying spherical solutions (degree ≤ K) of the
# field of the V⁽ⁱ⁾_nm (degree ≤ nmax), fitted on the sphere of radius r₀.
function _mem_spheroid_far_map(c, r₀, ν₀, nmax::Int, K::Int)
    θ, w = LayeredSpheroids.QuadGK.gauss(K + 24)
    points = [SVector(r₀ * sqrt(1 - p^2), 0.0, r₀ * p) for p in θ]
    V = [_mem_emitted(_MemSpheroidShape(c), x, nmax) for x in points]
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

# ─── The spheroids of an assembly ────────────────────────────────────────────

# The data of a one-layer spheroid of isotropic moduli in the matrix (κ₀, μ₀);
# its geometry in floating point, which the harmonics of its shape are built in.
function _mem_elastic_spheroid(s::LayeredSpheroids.LayeredSpheroid, name::Symbol, κ₀, μ₀, ν₀, nmax::Int)
    length(s.q) == 1 || throw(
        ArgumentError("MultipoleExpansion: in elasticity the spheroid :$(name) is of a single phase")
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
    K = nmax + _MEM_ELASTIC_EXTRA
    Z = _mem_spheroid_elastic_response(c, ρ₀, z₀, κ₀, μ₀, κ₁, μ₁, _mem_compliances(only(s.interfaces)), nmax)
    _, surface, weights = _mem_surface_projection(c, ξ₀, nmax)
    return _mem_elastic_shaped(_MemSpheroidShape(c), Z, _mem_spheroid_far_map(c, 2R, ν₀, nmax, K), R, K, ν₀, surface, weights, nmax)
end
