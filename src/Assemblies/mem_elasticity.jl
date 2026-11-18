# =============================================================================
#  mem_elasticity.jl — the multipole expansion method in elasticity.
#
#  The structure is that of conduction (`mem.jl`), with three solutions of the
#  Lamé equation per harmonic instead of one harmonic function. This file
#  writes it for spheres; a particle of another shape enters the same system
#  through its response, from the received solutions about its center to the
#  decaying spherical solutions it sends (`mem_spheroid_elasticity.jl`). In the matrix,
#  of shear modulus μ₀ and Poisson ratio ν₀,
#
#       u(x) = E·x + Σ_p Σ_{n′,m′} Σ_i b⁽ⁱ⁾⁽ᵖ⁾_n′m′ U⁽ⁱ⁾_n′m′(x − X_p) ,
#
#  U⁽ⁱ⁾_n′m′ being the decaying solutions of the Lamé equation of TensND (`irregular_lame_solution`). The
#  degree 0 holds the center of dilatation U⁽¹⁾_00 alone; the degree 1 holds no
#  net force and no net torque on a sphere in equilibrium, which the response
#  of the sphere gives by itself (it sends nothing back to a rigid motion).
#
#  About the center of sphere q the series of the others are regular fields,
#  expanded by `lame_solution_translation`:
#
#       U⁽ⁱ⁾_n′m′(x − X_p) = Σ_{j,n,m} (−1)^{n+m} η⁽ⁱ⁾⁽ʲ⁾_n′nm′m(X_q − X_p) u⁽ʲ⁾_nm(x − X_q) ,
#
#  so that sphere q receives on u⁽ʲ⁾_nm the coefficient
#
#       a⁽ʲ⁾⁽q⁾_nm = e⁽ʲ⁾_nm + Σ_{p≠q} Σ_{i,n′,m′} (−1)^{n+m} η⁽ⁱ⁾⁽ʲ⁾_n′nm′m(X_q − X_p) b⁽ⁱ⁾⁽ᵖ⁾_n′m′ ,
#
#  e being the coefficients of E·x (kushch2013, Eq. (4.52)):
#       e⁽³⁾_00 = tr E / (4ν₀ − 2) ,
#       e⁽¹⁾_20 = (2E₃₃ − E₁₁ − E₂₂)/3 ,  e⁽¹⁾_21 = E₁₃ − i E₂₃ ,  e⁽¹⁾_22 = E₁₁ − E₂₂ − 2i E₁₂ ,
#       e⁽¹⁾_2,−m = (−1)^m conj(e⁽¹⁾_2m) ,
#  and answers by its response, (n−m)!(n+m)! b_nm = 𝐙_n a_nm (`harmonic_response`).
#
#  SCALING.  Every unknown is brought to a displacement at the surface of its
#  sphere, divided by N_nm = √((n−m)!(n+m)!):
#
#       â⁽ʲ⁾_nm = a⁽ʲ⁾_nm R^{n−2+j} / N_nm      (u⁽ʲ⁾_nm grows as r^{n−2+j}) ,
#       b̂⁽ⁱ⁾_n′m′ = b⁽ⁱ⁾_n′m′ N_n′m′ / R^{n′+3−i}      (U⁽ⁱ⁾_n′m′ decays as r^{−(n′+3−i)}) ,
#
#  hence b̂ = 𝐙̂_n′ â with (𝐙̂_n′)_ij = (𝐙_n′)_ij / R^{2n′+1+j−i}, dimensionless, and
#
#       â⁽ʲ⁾⁽q⁾_nm − Σ_{p≠q} Σ_{i,n′,m′} W · b̂⁽ⁱ⁾⁽ᵖ⁾_n′m′ = e⁽ʲ⁾_nm R_q^{n−2+j} / N_nm ,
#       W = (−1)^{n+m} η⁽ⁱ⁾⁽ʲ⁾_n′nm′m(X_q − X_p) R_q^{n−2+j} R_p^{n′+3−i} / (N_nm N_n′m′) .
#
#  EFFECTIVE STIFFNESS.  With τ = σ − ℂ₀:ε the polarization, zero in the matrix,
#  the average of τ over a volume V holding the spheres reduces to integrals
#  over their surfaces in which only the dipoles of the series remain, the
#  center of dilatation b⁽¹⁾_00 and the five b⁽³⁾_2m (kushch2013, Eq. (5.75)):
#
#       tr ⟨τ⟩              =  24π μ₀ (1−ν₀)/(1−2ν₀) Σ_q b⁽¹⁾_00 / V ,
#       2⟨τ₃₃⟩ − ⟨τ₁₁⟩ − ⟨τ₂₂⟩ = −32π μ₀ (1−ν₀) Σ_q b⁽³⁾_20 / V ,
#       ⟨τ₁₃⟩ − i ⟨τ₂₃⟩      = −16π μ₀ (1−ν₀) Σ_q b⁽³⁾_21 / V ,
#       ⟨τ₁₁⟩ − ⟨τ₂₂⟩ − 2i ⟨τ₁₂⟩ = −64π μ₀ (1−ν₀) Σ_q b⁽³⁾_22 / V .
#
#  Solved for the six unit strains, they give 𝚺 such that ⟨τ⟩ = 𝚺 : E, closed
#  as in conduction.
#
#  PERIODIC ARRAYS.  Each decaying solution is an operator with constant
#  coefficients applied to (1/r, r) (U⁽³⁾ through 𝒟_n^m r = −r² Y_n^m/(2n−1));
#  its sum over the images is the same operator applied to (S, ρ), Δρ = 2S
#  (`lattice_solid_biharmonics`), plus the uniform terms of `_mem_vector_background!`.
#  The images of a sphere act on it, and ℂ = ℂ₀ + 𝚺 without closure.
# =============================================================================

# Position of the solution (j, n, m) among the unknowns of one sphere: the
# dilatation first, then three solutions per harmonic of degree n ≥ 1.
@inline _mem_eindex(j::Integer, n::Integer, m::Integer) = n == 0 ? 1 : 1 + 3 * (_mem_index(n, m) - 1) + j
@inline _mem_eblock(nmax::Integer) = 1 + 3 * _mem_block(nmax)

# The solutions that exist at a degree: sent families, received families.
@inline _mem_sent(n::Integer) = n == 0 ? (1:1) : (1:3)
@inline _mem_received(n::Integer) = n == 0 ? (3:3) : (1:3)

"""
    ElasticMultipoleSolution

Solution of the multipole expansion of an assembly in elasticity: for each
particle, in the order of `names`, the radius of its scaling (`radii`), its
response (`responses[q]`), the matrix that turns the scaled coefficients it
receives into the scaled coefficients of the decaying spherical solutions it
sends, and the scaled coefficients it receives under the six unit strains of
the Kelvin-Mandel basis (`received`, one column per strain), all in the frame
whose axes are those of the assembly in the order `axes`.
"""
struct ElasticMultipoleSolution{T, C}
    names::Vector{Symbol}
    radii::Vector{T}
    responses::Vector{Matrix{C}}
    received::Matrix{C}
    nmax::Int
    axes::NTuple{3, Int}
end

# A sphere in the elastic expansion: the radius R of the scaling and the scaled
# response Ẑ, from the received solutions (j, n, m) to the sent ones (i, n, m),
# both up to the degree nmax.
struct _MemElasticSphere{T}
    R::T
    Ẑ::Matrix{T}
end

# The response of a sphere from the 3 × 3 matrices 𝐙_n of `harmonic_response`,
# made dimensionless, 𝐙̂_n = 𝐙_n / R^{2n+1+j−i}; the degree and the order are kept.
function _mem_elastic_sphere(Z::Vector, R::T, nmax::Int) where {T}
    Ẑ = zeros(T, _mem_eblock(nmax), _mem_eblock(nmax))
    for n in 0:nmax, m in (-n):n, i in _mem_sent(n), j in _mem_received(n)
        Ẑ[_mem_eindex(i, n, m), _mem_eindex(j, n, m)] = Z[n + 1][i, j] / R^(2n + 1 + j - i)
    end
    return _MemElasticSphere(R, Ẑ)
end

# The highest degree a particle sends: nmax for a sphere, whose response keeps
# the degree.
_mem_sent_degree(d::_MemElasticSphere, nmax::Int) = nmax

function _mem_solve(asm::ParticleAssembly, C₀::TensND.AbstractTens{4, 3}, prop::Symbol, nmax::Int)
    _mem_isotropic(C₀)
    # the dipoles of elasticity, from which the stiffness follows, are of degree 2
    nmax ≥ 2 || throw(
        ArgumentError(
            "MultipoleExpansion: elasticity needs `nmax` at least 2, the degree of the " *
                "dipoles that give the stiffness; got $nmax"
        )
    )
    κ₀, μ₀ = TensND.k_mu(C₀)
    ν₀ = LayeredSpheres._lame_poisson(κ₀, μ₀)
    2 * ν₀ < 1 || throw(
        ArgumentError(
            "MultipoleExpansion: the matrix is incompressible (Poisson ratio 1/2), for which " *
                "the dilatation of the remote strain has no finite amplitude"
        )
    )
    names = particle_names(asm)
    geometries = [particle_geometry(asm, nm) for nm in names]
    # the particles, and the order of the axes of the assembly in their frame
    particles, order = _mem_elastic_particles(Val(_mem_one_kind(geometries)), asm, C₀, prop, names, geometries, nmax)
    centers = [particle_center(asm, nm) for nm in names]
    # a center given as a tuple may mix plain numbers and one carrying a derivative
    TX = mapreduce(c -> mapreduce(typeof, promote_type, c), promote_type, centers)
    X = [SVector{3, TX}(c[collect(order)]) for c in centers]
    â = _mem_vector_system(particles, X, ν₀, _mem_permuted(asm.boundary, order), nmax)
    C = eltype(â)
    return ElasticMultipoleSolution(
        copy(names), [real(C)(d.R) for d in particles], [Matrix{C}(d.Ẑ) for d in particles], â, nmax, order
    )
end

function _mem_elastic_particles(::Val{:sphere}, asm, C₀, prop::Symbol, names, geometries, nmax::Int)
    spheres = Any[_mem_sphere(g, asm, nm, prop) for (g, nm) in zip(geometries, names)]
    # one element type for all, any radius or response possibly carrying a derivative
    R = _common_eltype([LayeredSpheres.outer_radius(sphere) for sphere in spheres])
    raw = [[harmonic_response(sphere, C₀, n) for n in 0:nmax] for sphere in spheres]
    T = float(promote_type(eltype(R), mapreduce(Z -> eltype(first(Z)), promote_type, raw)))
    return [_mem_elastic_sphere(raw[q], T(R[q]), nmax) for q in eachindex(spheres)], (1, 2, 3)
end

function _mem_elastic_particles(::Val{:spheroid}, asm, C₀, prop::Symbol, names, geometries, nmax::Int)
    κ₀, μ₀ = TensND.k_mu(C₀)
    ν₀ = LayeredSpheres._lame_poisson(κ₀, μ₀)
    raw = [_mem_elastic_spheroid(_mem_spheroid(g, asm, nm, prop), nm, κ₀, μ₀, ν₀, nmax) for (g, nm) in zip(geometries, names)]
    C = mapreduce(d -> eltype(d.Ẑ), promote_type, raw)
    return [_mem_convert(d, C) for d in raw], (1, 2, 3)
end

# Function barrier, as in conduction. Returns the scaled coefficients received
# by every particle, one column per unit strain of the Kelvin-Mandel basis.
function _mem_vector_system(particles::Vector, X::Vector, ν₀, boundary, nmax::Int)
    T = float(promote_type(eltype(eltype(X)), typeof(ν₀), mapreduce(d -> real(eltype(d.Ẑ)), promote_type, particles)))
    C = Complex{T}
    N, n = length(particles), _mem_eblock(nmax)
    _check_system_size(N * n, C, N, nmax)
    M = zeros(C, N * n, N * n)
    rhs = zeros(C, N * n, 6)
    K = maximum(d -> _mem_sent_degree(d, nmax), particles)
    norms = [solid_harmonic_norm(k, m) for k in 0:K for m in (-k):k]
    for q in 1:N
        rows = ((q - 1) * n + 1):(q * n)
        for i in rows
            M[i, i] = one(T)
        end
        _mem_remote_strain!(rhs, rows[1] - 1, T(particles[q].R), T(ν₀))
        for p in 1:N
            cols = ((p - 1) * n + 1):(p * n)
            _mem_couple_elastic!(M, rows, cols, boundary, particles[q], particles[p], X[q] - X[p], p == q, ν₀, norms, nmax)
        end
    end
    return M \ rhs
end

# Position of N_nm in the table of the norms, degree 0 included.
@inline _mem_norm_index(n::Integer, m::Integer) = n * n + n + m + 1

# Right-hand side of the six unit strains of the Kelvin-Mandel basis
# (11, 22, 33, √2·23, √2·13, √2·12) on the rows of one sphere.
function _mem_remote_strain!(rhs::Matrix{Complex{T}}, oq::Int, R::T, ν₀::T) where {T}
    γ₀ = (4 * ν₀ - 2) / 3
    h = sqrt(T(2)) / 2                                  # the shear of a unit Mandel strain
    N₂ = (solid_harmonic_norm(2, 0), solid_harmonic_norm(2, 1), solid_harmonic_norm(2, 2))
    row(m) = oq + _mem_eindex(1, 2, m)
    # the dilatation: tr E = 1 for the three normal strains
    for m in 1:3
        rhs[oq + 1, m] = R / (3 * γ₀)
    end
    # e⁽¹⁾_20 = (2E₃₃ − E₁₁ − E₂₂)/3
    e₂₀ = (-one(T) / 3, -one(T) / 3, 2 * one(T) / 3)
    for m in 1:3
        rhs[row(0), m] = e₂₀[m] * R / N₂[1]
    end
    # e⁽¹⁾_22 = E₁₁ − E₂₂ − 2i E₁₂ and its conjugate at m = −2
    for (m, e) in ((1, complex(one(T))), (2, complex(-one(T))), (6, -2im * h))
        rhs[row(2), m] = e * R / N₂[3]
        rhs[row(-2), m] = conj(e) * R / N₂[3]
    end
    # e⁽¹⁾_21 = E₁₃ − i E₂₃ and e⁽¹⁾_2,−1 = −conj(e⁽¹⁾_21)
    for (m, e) in ((5, complex(h)), (4, -im * h))
        rhs[row(1), m] = e * R / N₂[2]
        rhs[row(-1), m] = -conj(e) * R / N₂[2]
    end
    return rhs
end

# The field particle p sends to particle q: the translation of its decaying
# spherical solutions to the regular ones about q, times its response, which
# turns what p receives (the unknowns) into what it sends; and the uniform
# background of a periodic array.
function _mem_couple_elastic!(M, rows, cols, boundary, dq, dp, r, self::Bool, ν₀, norms, nmax::Int)
    W = _mem_vector_translation(boundary, r, self, nmax + _mem_sent_degree(dp, nmax))
    return _mem_couple_far!(M, rows, cols, boundary, W, dq, dp, ν₀, norms, nmax)
end

# The same from the tables W of the translation (`nothing` for none).
function _mem_couple_far!(M, rows, cols, boundary, W, dq, dp, ν₀, norms, nmax::Int)
    if W !== nothing
        T = _mem_vector_translation_matrix(W..., dq.R, dp.R, ν₀, norms, nmax, _mem_sent_degree(dp, nmax))
        @views M[rows, cols] .-= T * dp.Ẑ
    end
    _mem_vector_background!(M, rows, cols, boundary, dq.R, dp.R, dp.Ẑ, ν₀)
    return M
end

# The scaled coefficient a⁽ʲ⁾_nm received about q per scaled coefficient b⁽ⁱ⁾_n′m′
# sent by p, n ≤ nmax and n′ ≤ K (`lame_solution_translation`).
function _mem_vector_translation_matrix(Y, r, Rq, Rp, ν₀, norms, nmax::Int, K::Int)
    T = promote_type(eltype(Y.values), typeof(ν₀), typeof(Rq), typeof(Rp))
    W = zeros(T, _mem_eblock(nmax), _mem_eblock(K))
    for n in 0:nmax, m in (-n):n, j in _mem_received(n)
        row = _mem_eindex(j, n, m)
        wq = Rq^(n - 2 + j) / norms[_mem_norm_index(n, m)]
        isodd(n + m) && (wq = -wq)
        for n′ in 0:K, m′ in (-n′):n′, i in _mem_sent(n′)
            a, b = lame_solution_translation(i, j, n′, n, m′, m, Y, r)
            W[row, _mem_eindex(i, n′, m′)] += wq * (a + 4 * (1 - ν₀) * b) * Rp^(n′ + 3 - i) / norms[_mem_norm_index(n′, m′)]
        end
    end
    return W
end

# Tables of the translation from sphere p to sphere q: the harmonics at the
# separation and the separation itself for a finite assembly; the lattice sums of
# S and ρ for a periodic one, where the images of sphere q act on it as well.
_mem_vector_translation(::MixedBC, r, self::Bool, nmax::Int) =
    self ? nothing : (irregular_solid_harmonics(r, nmax), r)
_mem_vector_translation(b::PeriodicBox, r, self::Bool, nmax::Int) =
    lattice_solid_biharmonics(r, box_periods(b), nmax)

# Uniform terms of the background of a periodic array. About any point, S and ρ
# differ from functions to which the free translation applies by
# c |y|² and d |y|⁴, c = 2π/(3V), d = π/(15V) (ΔS = 4π/V off the sources,
# Δρ = 2S). The operators that build the decaying solutions from (1/r, r) give
# on these polynomials (𝒴_n^m(y) = (−1)^n N_nm² y_n^m/(2n−1)!!):
#   U⁽¹⁾_00 → 2c y              = (2c/γ₀) u⁽³⁾_00 ,   γ₀ = (4ν₀ − 2)/3 ,
#   U⁽¹⁾_1m → 2c grad 𝒴_1^m     = −2c N_1m² u⁽¹⁾_1m   (a translation),
#   U⁽²⁾_1m →                     4c N_1m² u⁽²⁾_1m   (a rotation),
#   U⁽³⁾_2m → [8d/3 − 4c(1−ν₀)/3] N_2m² u⁽¹⁾_2m = (8π/45V)(5ν₀ − 4) N_2m² u⁽¹⁾_2m ,
# and nothing at higher degrees. U⁽³⁾_1m, a point force, would add a quadratic
# field: a sphere in equilibrium never sends it (its response is zero there).
_mem_vector_background!(M, rows, cols, ::MixedBC, Rq, Rp, Ẑp, ν₀) = M
function _mem_vector_background!(M, rows, cols, b::PeriodicBox, Rq, Rp, Ẑp, ν₀)
    V = prod(box_periods(b))
    c = 2π / (3 * V)
    γ₀ = (4 * ν₀ - 2) / 3
    # (sent i, n′; received j, n; factor without N²): the order m is kept
    _mem_background_term!(M, rows, cols, Rq, Rp, Ẑp, 1, 0, 3, 0, 0, 2 * c / γ₀)
    for m in -1:1
        N² = solid_harmonic_norm(1, m)^2
        _mem_background_term!(M, rows, cols, Rq, Rp, Ẑp, 1, 1, 1, 1, m, -2 * c * N²)
        _mem_background_term!(M, rows, cols, Rq, Rp, Ẑp, 2, 1, 2, 1, m, 4 * c * N²)
    end
    for m in -2:2
        N² = solid_harmonic_norm(2, m)^2
        _mem_background_term!(M, rows, cols, Rq, Rp, Ẑp, 3, 2, 1, 2, m, 8π / (45 * V) * (5 * ν₀ - 4) * N²)
    end
    return M
end

# The received coefficient a⁽ʲ⁾_nm of q gains β b⁽ⁱ⁾_n′m of p, in the scaled
# unknowns of `_mem_vector_system`.
function _mem_background_term!(M, rows, cols, Rq, Rp, Ẑp, i, n′, j, n, m, β)
    N = solid_harmonic_norm(n, m)
    w = β * Rq^(n - 2 + j) * Rp^(n′ + 3 - i) / (N * N)
    @views M[rows[_mem_eindex(j, n, m)], cols] .-= w .* Ẑp[_mem_eindex(i, n′, m), :]
    return M
end

"""
    _mem_contribution(asm, C₀, sol::ElasticMultipoleSolution) -> AbstractTens

Sum ``\\boldsymbol{\\Sigma}`` of the contributions of the spheres per unit volume
of the assembly, interactions included: the tensor such that the average of the
polarization ``\\boldsymbol{\\sigma} - \\mathbb{C}_0 : \\boldsymbol{\\varepsilon}`` is
``\\boldsymbol{\\Sigma} : \\boldsymbol{E}`` under the remote strain
``\\boldsymbol{E}``. It follows from the dipoles of the series, the centers of
dilatation and the five solutions of the third family at the degree 2.
"""
function _mem_contribution(
        asm::ParticleAssembly, C₀::TensND.AbstractTens{4, 3}, sol::ElasticMultipoleSolution{T}
    ) where {T}
    κ₀, μ₀ = TensND.k_mu(C₀)
    ν₀ = LayeredSpheres._lame_poisson(κ₀, μ₀)
    n = _mem_eblock(sol.nmax)
    N₂ = (solid_harmonic_norm(2, 0), solid_harmonic_norm(2, 1), solid_harmonic_norm(2, 2))
    Σ = zeros(T, 6, 6)
    h = sqrt(T(2))
    for q in eachindex(sol.names)
        R² = sol.radii[q]^2
        # the sent coefficients, b⁽¹⁾_00 = b̂ R² and b⁽³⁾_2m = b̂ R² / N_2m
        b̂ = sol.responses[q] * view(sol.received, ((q - 1) * n + 1):(q * n), :)
        b₀ = R² .* view(b̂, _mem_eindex(1, 0, 0), :)
        b₂(m) = (R² / N₂[m + 1]) .* view(b̂, _mem_eindex(3, 2, m), :)
        tr_τ = 24π * μ₀ * (1 - ν₀) / (1 - 2 * ν₀) .* real.(b₀)
        dev = -32π * μ₀ * (1 - ν₀) .* real.(b₂(0))               # 2τ₃₃ − τ₁₁ − τ₂₂
        q₁ = -16π * μ₀ * (1 - ν₀) .* b₂(1)                      # τ₁₃ − i τ₂₃
        q₂ = -64π * μ₀ * (1 - ν₀) .* b₂(2)                      # τ₁₁ − τ₂₂ − 2i τ₁₂
        τ₃₃ = (tr_τ .+ dev) ./ 3
        Σ[1, :] .+= (tr_τ .- τ₃₃ .+ real.(q₂)) ./ 2
        Σ[2, :] .+= (tr_τ .- τ₃₃ .- real.(q₂)) ./ 2
        Σ[3, :] .+= τ₃₃
        Σ[4, :] .+= -h .* imag.(q₁)
        Σ[5, :] .+= h .* real.(q₁)
        Σ[6, :] .+= -h .* imag.(q₂) ./ 2
    end
    Σ ./= assembly_volume(asm)
    # back to the frame of the assembly; major-symmetric by reciprocity, the
    # rounding of the solve taken off
    p = _mem_mandel_permutation(sol.axes)
    Σa = zeros(T, 6, 6)
    Σa[p, p] .= (Σ .+ transpose(Σ)) ./ 2
    return TensND.from_mandel(Σa)
end

# The Kelvin-Mandel components (11, 22, 33, 23, 13, 12) of the frame of axes
# `order` among those of the assembly.
function _mem_mandel_permutation(order)
    pair(i, j) = 9 - order[i] - order[j]          # the shear component of (oᵢ, oⱼ): 4, 5 or 6
    return [order[1], order[2], order[3], pair(2, 3), pair(1, 3), pair(1, 2)]
end
