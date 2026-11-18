# =============================================================================
#  mem_ellipsoid_elasticity.jl — the multipole expansion method for ellipsoids
#  in elasticity (kushch2019a, kushch2019b).
#
#  SOLUTIONS. Outside an ellipsoid of semi-axes a₁ > a₂ > a₃, the displacement
#  is written with the four potentials of Papkovich and Neuber, each a series
#  of the decaying ellipsoidal harmonics 𝔽_k of `ellipsoidal.jl`, degree 0
#  included (kushch2019a, Eqs. (4), (6)):
#
#       u = 4(1−ν₀) B − grad(x·B + B₀) ,   B = Σ_j B_j i_j ,
#
#  so that each 𝔽_k carries the solutions B_j = 𝔽_k,
#  u = (3−4ν₀) 𝔽_k i_j − x_j grad 𝔽_k, and B₀ = 𝔽_k, u = −grad 𝔽_k. Three
#  potentials suffice (kushch2019a, Section 2.3); B₀ is kept at the degree 2,
#  where it puts the field of an ellipsoid under a uniform strain in the
#  degrees up to 2, and nowhere else, where it would make the four potentials
#  nearly redundant. The gradients of 𝔽 come from forward-mode
#  differentiation of their evaluation.
#
#  SPHERICAL FORM. With 𝔽_k = Σ μ_kl Y_k^l outside the sphere of radius 2a₁
#  (`_mem_sphere_coefficients`), each solution is a Papkovich-Neuber field of
#  i_j Y_k^l or of Y_k^l. In the complex basis, i₁ = e₁ + e₂, i₂ = i(e₂ − e₁),
#  i₃ = e₃, and e₁Y_k^{m−1}, e₂Y_k^{m+1}, e₃Y_k^m are combinations of the vector
#  harmonics Y⁽¹⁾_{k−1,m}, Y⁽²⁾_{k,m}, Y⁽³⁾_{k+1,m} of `vector_harmonics.jl`, by
#  inverting their definitions. The Papkovich-Neuber field of each is a
#  decaying spherical solution: with x·Y⁽¹⁾_nm = −(n+1) Y_n^m and x·Y⁽²⁾_nm = 0,
#
#       Y⁽¹⁾_{k−1,m} → (4(1−ν₀) + k) U⁽¹⁾_{k−1,m} ,  Y⁽²⁾_{k,m} → 4(1−ν₀) U⁽²⁾_{k,m} ,
#       Y⁽³⁾_{k+1,m} → U⁽³⁾_{k+1,m} ,
#
#  and grad Y_k^l = U⁽¹⁾_{k,l}. The response and the near pairs are those of
#  `mem_shaped_elasticity.jl`, the conditions imposed at the quadrature points
#  of the surface of `_mem_ellipsoid_projection`.
# =============================================================================

# The shape of an ellipsoid: its geometry, its Lamé functions of the degrees
# 0 to nmax (the class j of the degree n at index n² + j) and the quadrature
# of the functions of the second kind.
struct _MemEllipsoidShape
    g::_MemEllipsoidGeometry
    lame::Vector{TensND.LameFunction}
    nodes::Tuple{Vector{Float64}, Vector{Float64}}
end

_mem_focal(s::_MemEllipsoidShape) = s.g.h2

# The solutions at x, four per 𝔽_k, as u = P + ν Q: columns 4(k − 1) + j, j = 1,
# 2, 3 for B_j and j = 4 for B₀.
function _mem_emitted(s::_MemEllipsoidShape, x, nmax::Int)
    potentials(y) = _mem_ellipsoidal_irregular!(zeros(eltype(y), length(s.lame)), y, s.g, s.lame, s.nodes)
    F = potentials(x)
    ∇F = ForwardDiff.jacobian(potentials, x)
    T = promote_type(eltype(F), eltype(∇F))
    P, Q = zeros(T, 3, 4 * length(F)), zeros(T, 3, 4 * length(F))
    for k in eachindex(F)
        for j in 1:3
            P[:, 4(k - 1) + j] .= -x[j] .* view(∇F, k, :)
            P[j, 4(k - 1) + j] += 3 * F[k]
            Q[j, 4(k - 1) + j] = -4 * F[k]
        end
        P[:, 4k] .= -view(∇F, k, :)
    end
    return P, Q
end

# The Lamé functions of the degrees 0 to nmax.
_mem_lame_through(g, nmax::Int) = reduce(vcat, [lame_functions(g.chart, n) for n in 0:nmax])

# ─── The spherical form ──────────────────────────────────────────────────────

# The sent spherical coefficients (degree ≤ K) of the solutions of the 𝔽_k:
# the scalar coefficients μ of the 𝔽_k on the Y_k^l, degree ≤ K − 1, carried
# through the decomposition of i_j Y_k^l and the Papkovich-Neuber fields.
function _mem_ellipsoid_elastic_far_map(s::_MemEllipsoidShape, ν₀, K::Int)
    nf = length(s.lame)
    μ = _mem_sphere_coefficients((f, x) -> _mem_ellipsoidal_irregular!(f, x, s.g, s.lame, s.nodes), nf, 2 * s.g.a[1], K - 1)
    C = Complex{promote_type(Float64, typeof(ν₀))}
    out = zeros(C, _mem_eblock(K), 4 * nf)
    # i_j in the complex basis: the coefficients of e₁, e₂, e₃
    cartesian = ((1, 1, 0), (-im, im, 0), (0, 0, 1))
    for k in 0:(K - 1), l in (-k):k
        row = μ[k^2 + k + l + 1, :]
        # B₀ = Y_k^l: u = −grad Y_k^l = −U⁽¹⁾_{k,l}
        out[_mem_eindex(1, k, l), 4:4:end] .-= row
        for (c, member_shift) in ((1, 1), (2, -1), (3, 0))
            # e_c Y_k^l is the member c of the vector order m = l + shift
            m = l + member_shift
            for (f, β) in _mem_vector_members(k, m, c)
                n, i, factor = _mem_pn_family(f, k, ν₀)
                for j in 1:3
                    w = cartesian[j][c]
                    iszero(w) && continue
                    out[_mem_eindex(i, n, m), j:4:end] .+= (w * β * factor) .* row
                end
            end
        end
    end
    return out
end

# The family f (1: Y⁽¹⁾_{k−1,m}, 2: Y⁽²⁾_{k,m}, 3: Y⁽³⁾_{k+1,m}) as the sent
# spherical solution (degree, family) and the factor of its Papkovich-Neuber field.
_mem_pn_family(f::Int, k::Int, ν₀) =
    f == 1 ? (k - 1, 1, 4 * (1 - ν₀) + k) : f == 2 ? (k, 2, 4 * (1 - ν₀)) : (k + 1, 3, one(ν₀))

# The member c of the vector order m at the scalar degree k (1: e₁Y_k^{m−1},
# 2: e₂Y_k^{m+1}, 3: e₃Y_k^m) as pairs (family, coefficient), by inverting the
# definitions of the families that exist on the members that exist.
function _mem_vector_members(k::Int, m::Int, c::Int)
    families = [f for f in 1:3 if (f == 1 ? k ≥ 1 && abs(m) ≤ k - 1 : f == 2 ? k ≥ 1 && abs(m) ≤ k : abs(m) ≤ k + 1)]
    members = [d for d in 1:3 if abs(m + (d == 1 ? -1 : d == 2 ? 1 : 0)) ≤ k]
    B = [_mem_vector_definition(f, d, k, m) for f in families, d in members]
    row = inv(B)[findfirst(==(c), members), :]
    return zip(families, row)
end

# The coefficient of the member d in the family f (`vector_harmonics.jl`).
function _mem_vector_definition(f::Int, d::Int, k::Int, m::Int)
    f == 1 && return ComplexF64((1, -1, -1)[d])
    f == 2 && return im / k * (k + m, k - m, -m)[d]
    return ((k + m) * (k + m + 1), -(k - m) * (k - m + 1), (k - m + 1) * (k + m + 1))[d] / ((k + 1) * (2k + 1))
end

# ─── The ellipsoids of an assembly ───────────────────────────────────────────

# The data of an ellipsoid of semi-axes `semi` (decreasing, in the frame of its
# axes), of isotropic moduli (κ₁, μ₁) and bonded by `interface`, in the matrix
# (κ₀, μ₀).
function _mem_elastic_ellipsoid(semi, κ₀, μ₀, ν₀, κ₁, μ₁, interface, nmax::Int)
    g = _mem_ellipsoid_geometry(semi...)
    shape = _MemEllipsoidShape(g, _mem_lame_through(g, nmax), LayeredSpheroids.QuadGK.gauss(nmax + 24, 0.0, 1.0))
    higher = shape.lame[2:end]
    _, surface, _, w, _ = _mem_ellipsoid_projection(g, higher, ones(length(higher)), nmax)
    points = [SVector{3}(view(surface, :, k)) for k in axes(surface, 2)]
    normals = [normalize(x ./ SVector{3}(g.a) .^ 2) for x in points]
    rows = _mem_interface_rows(y -> _mem_emitted(shape, y, nmax), points, normals, κ₀, μ₀, κ₁, μ₁, nmax)
    nout = 4 * length(shape.lame)
    # B₀ at the degree 2 alone: there it puts the field of the ellipsoid under a
    # uniform strain (Eshelby's) in the degrees up to 2; at the other degrees
    # it makes the four potentials nearly redundant: kept at all of them, the
    # stiffness differs by 1e-6 between plain and dual arithmetic, 1e-13 here
    sent = [c for c in 1:nout if c % 4 != 0 || isqrt(c ÷ 4 - 1) == 2]
    A, b = _mem_interface_system(rows, sent, 1:_mem_eblock(nmax), w, g.a[1] / TensND.primal(μ₀), _mem_compliances(interface))
    sol = _mem_column_scaled_solve(A, b)
    Z = zeros(eltype(sol), nout, size(sol, 2))
    Z[sent, :] .= sol[1:length(sent), :]
    K = nmax + _MEM_ELASTIC_EXTRA
    return _mem_elastic_shaped(shape, Z, _mem_ellipsoid_elastic_far_map(shape, ν₀, K), g.a[1], K, ν₀, surface, w, nmax)
end

function _mem_elastic_particles(::Val{:ellipsoid}, asm, C₀, prop::Symbol, names, geometries, nmax::Int)
    κ₀, μ₀ = TensND.k_mu(C₀)
    ν₀ = LayeredSpheres._lame_poisson(κ₀, μ₀)
    axes = [_mem_ellipsoid_axes(g, nm) for (g, nm) in zip(geometries, names)]
    order = last(first(axes))
    all(a -> last(a) == order, axes) || throw(
        ArgumentError(
            "MultipoleExpansion: the ellipsoids of an assembly share their orientation, the " *
                "largest semi-axis along the same axis for all"
        )
    )
    raw = map(zip(names, geometries, axes)) do (nm, geom, (semi, _))
        P = particle_property(asm, nm, prop)
        P isa TensND.TensISO || throw(
            ArgumentError(
                "MultipoleExpansion: in elasticity the ellipsoid :$(nm) has isotropic moduli (TensISO)"
            )
        )
        _mem_elastic_ellipsoid(semi, κ₀, μ₀, ν₀, TensND.k_mu(P)..., _mem_interface(geom), nmax)
    end
    C = mapreduce(d -> eltype(d.Ẑ), promote_type, raw)
    return [_mem_convert(d, C) for d in raw], order
end
