# =============================================================================
#  mem_elasticity.jl — the multipole expansion method in elasticity.
#
#  The structure is that of conduction (`mem.jl`), with three solutions of the
#  Lamé equation per harmonic instead of one harmonic function. In the matrix,
#  of shear modulus μ₀ and Poisson ratio ν₀,
#
#       u(x) = E·x + Σ_p Σ_{n′,m′} Σ_i b⁽ⁱ⁾⁽ᵖ⁾_n′m′ U⁽ⁱ⁾_n′m′(x − X_p) ,
#
#  U⁽ⁱ⁾_n′m′ being the decaying solutions of `Harmonics/vector_harmonics.jl`. The
#  degree 0 holds the center of dilatation U⁽¹⁾_00 alone; the degree 1 holds no
#  net force and no net torque on a sphere in equilibrium, which the response
#  of the sphere gives by itself (it sends nothing back to a rigid motion).
#
#  About the center of sphere q the series of the others are regular fields,
#  expanded by `lame_translation`:
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
sphere, in the order of `names`, its radius (`radii`), its responses
``(\\mathbf{Z}_n)_{ij}/R^{2n+1+j-i}``, made dimensionless (`responses[q][n + 1]`), and the scaled
coefficients it receives under the six unit strains of the Kelvin-Mandel basis
(`received`, one column per strain).
"""
struct ElasticMultipoleSolution{T, C}
    names::Vector{Symbol}
    radii::Vector{T}
    responses::Vector{Vector{Matrix{T}}}
    received::Matrix{C}
    nmax::Int
end

function _mem_solve(asm::ParticleAssembly, C₀::TensND.AbstractTens{4, 3}, prop::Symbol, nmax::Int)
    _mem_isotropic(C₀)
    _mem_elastic_boundary(asm.boundary)
    κ₀, μ₀ = TensND.k_mu(C₀)
    ν₀ = LayeredSpheres._lame_poisson(κ₀, μ₀)
    2 * ν₀ < 1 || throw(
        ArgumentError(
            "MultipoleExpansion: the matrix is incompressible (Poisson ratio 1/2), for which " *
                "the dilatation of the remote strain has no finite amplitude"
        )
    )
    names = particle_names(asm)
    spheres = Any[_mem_sphere(particle_geometry(asm, nm), asm, nm, prop) for nm in names]
    R = _common_eltype([LayeredSpheres.outer_radius(sphere) for sphere in spheres])
    raw = [[harmonic_response(sphere, C₀, n) for n in 0:nmax] for sphere in spheres]
    centers = [particle_center(asm, nm) for nm in names]
    TX = mapreduce(eltype, promote_type, centers)
    T = float(promote_type(eltype(R), TX, typeof(ν₀), mapreduce(Z -> eltype(first(Z)), promote_type, raw)))
    Rᵀ = convert(Vector{T}, R)
    # 𝐙̂_n = 𝐙_n / R^{2n+1+j−i}
    Ẑ = [
        [T[raw[q][n + 1][i, j] / Rᵀ[q]^(2n + 1 + j - i) for i in 1:3, j in 1:3] for n in 0:nmax]
            for q in eachindex(spheres)
    ]
    X = [SVector{3, TX}(c) for c in centers]
    â = _mem_vector_system(Rᵀ, Ẑ, X, T(ν₀), asm.boundary, nmax)
    return ElasticMultipoleSolution(copy(names), Rᵀ, Ẑ, â, nmax)
end

# The sums of the Lamé solutions over the images of a periodic assembly hold
# more than the lattice harmonics of conduction: they are not written yet.
_mem_elastic_boundary(::MixedBC) = nothing
_mem_elastic_boundary(::PeriodicBox) = throw(
    ArgumentError(
        "MultipoleExpansion: elasticity is available on a finite assembly (MixedBC); the " *
            "periodic one (PeriodicBox) is available in conduction only"
    )
)

# Function barrier, as in conduction. Returns the scaled coefficients received
# by every sphere, one column per unit strain of the Kelvin-Mandel basis.
function _mem_vector_system(
        R::Vector{T}, Ẑ::Vector{Vector{Matrix{T}}}, X::Vector, ν₀::T, boundary, nmax::Int
    ) where {T}
    N, n = length(R), _mem_eblock(nmax)
    _check_system_size(N * n, Complex{T}, N, nmax)
    M = zeros(Complex{T}, N * n, N * n)
    rhs = zeros(Complex{T}, N * n, 6)
    norms = [harmonic_norm(n, m) for n in 0:nmax for m in (-n):n]
    for q in 1:N
        oq = (q - 1) * n
        for i in 1:n
            M[oq + i, oq + i] = one(T)
        end
        _mem_remote_strain!(rhs, oq, R[q], ν₀)
        for p in 1:N
            p == q && continue
            r = X[q] - X[p]
            Y = _mem_translation(boundary, r, false, 2 * nmax)
            _mem_couple_vector!(M, oq, (p - 1) * n, Y, r, R[q], R[p], Ẑ[p], ν₀, norms, nmax)
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
    N₂ = (harmonic_norm(2, 0), harmonic_norm(2, 1), harmonic_norm(2, 2))
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

# Block of the field sphere p sends to sphere q: for each solution received
# (j, n, m) and each one sent (i, n′, m′), the translation coefficient times the
# response of p, which turns what p receives (the unknowns) into what it sends.
function _mem_couple_vector!(M::Matrix, oq::Int, op::Int, Y, r, Rq, Rp, Ẑp, ν₀, norms, nmax::Int)
    for n in 0:nmax, m in (-n):n, j in _mem_received(n)
        row = oq + _mem_eindex(j, n, m)
        wq = Rq^(n - 2 + j) / norms[_mem_norm_index(n, m)]
        isodd(n + m) && (wq = -wq)
        for n′ in 0:nmax, m′ in (-n′):n′, i in _mem_sent(n′)
            a, b = lame_translation(i, j, n′, n, m′, m, Y, r)
            w = wq * (a + 4 * (1 - ν₀) * b) * Rp^(n′ + 3 - i) / norms[_mem_norm_index(n′, m′)]
            for j′ in _mem_received(n′)
                M[row, op + _mem_eindex(j′, n′, m′)] -= w * Ẑp[n′ + 1][i, j′]
            end
        end
    end
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
    N₂ = (harmonic_norm(2, 0), harmonic_norm(2, 1), harmonic_norm(2, 2))
    Σ = zeros(T, 6, 6)
    h = sqrt(T(2))
    for q in eachindex(sol.names)
        oq = (q - 1) * n
        R² = sol.radii[q]^2
        Ẑ₀, Ẑ₂ = sol.responses[q][1], sol.responses[q][3]
        # b⁽¹⁾_00 = b̂ R² and b⁽³⁾_2m = b̂ R² / N_2m, with b̂ = 𝐙̂ â
        b₀ = Ẑ₀[1, 3] * R² .* view(sol.received, oq + 1, :)
        b₂(m) = (R² / N₂[m + 1]) .* sum(
            Ẑ₂[3, j] .* view(sol.received, oq + _mem_eindex(j, 2, m), :) for j in 1:3
        )
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
    # major-symmetric by reciprocity; the rounding of the solve is taken off
    return TensND.from_mandel((Σ .+ transpose(Σ)) ./ 2)
end
