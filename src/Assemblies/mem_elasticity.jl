# =============================================================================
#  mem_elasticity.jl — the multipole expansion method in elasticity.
#
#  The structure is that of conduction (`mem.jl`), with three solutions of the
#  Lamé equation per harmonic instead of one harmonic function. In the matrix,
#  of shear modulus μ₀ and Poisson ratio ν₀,
#
#       u(x) = E·x + Σ_p Σ_{t,s} Σ_i A⁽ⁱ⁾⁽ᵖ⁾_ts U⁽ⁱ⁾_ts(x − X_p) ,
#
#  U⁽ⁱ⁾_ts being the decaying solutions of `Harmonics/vector_harmonics.jl`. The
#  degree 0 holds the center of dilatation U⁽¹⁾_00 alone; the degree 1 holds no
#  net force and no net torque on a sphere in equilibrium, which the response
#  of the sphere gives by itself (it sends nothing back to a rigid motion).
#
#  About the center of sphere q the series of the others are regular fields,
#  expanded by `lame_translation`:
#
#       U⁽ⁱ⁾_ts(x − X_p) = Σ_{j,k,l} (−1)^{k+l} η⁽ⁱ⁾⁽ʲ⁾_tksl(X_q − X_p) u⁽ʲ⁾_kl(x − X_q) ,
#
#  so that sphere q receives on u⁽ʲ⁾_kl the amplitude
#
#       ĉ⁽ʲ⁾⁽q⁾_kl = c⁽ʲ⁾⁽q⁾_kl + Σ_{p≠q} Σ_{i,t,s} (−1)^{k+l} η⁽ⁱ⁾⁽ʲ⁾_tksl A⁽ⁱ⁾⁽ᵖ⁾_ts ,
#
#  c being the coefficients of E·x (kushch2013, Eq. (4.52)):
#       c⁽³⁾_00 = tr E / (3γ₀) ,  γ₀ = (4ν₀ − 2)/3 ,
#       c⁽¹⁾_20 = (2E₃₃ − E₁₁ − E₂₂)/3 ,  c⁽¹⁾_21 = E₁₃ − i E₂₃ ,  c⁽¹⁾_22 = E₁₁ − E₂₂ − 2i E₁₂ ,
#       c⁽¹⁾_2,−s = (−1)^s conj(c⁽¹⁾_2s) ,
#  and answers by its response, (t−s)!(t+s)! A_ts = 𝐁_t ĉ_ts (`harmonic_response`).
#
#  SCALING.  Every unknown is brought to a displacement at the surface of its
#  sphere, divided by N_ts = √((t−s)!(t+s)!):
#
#       e⁽ʲ⁾_kl = ĉ⁽ʲ⁾_kl R^{k−2+j} / N_kl      (u⁽ʲ⁾_kl grows as r^{k−2+j}) ,
#       a⁽ⁱ⁾_ts = A⁽ⁱ⁾_ts N_ts / R^{t+3−i}      (U⁽ⁱ⁾_ts decays as r^{−(t+3−i)}) ,
#
#  hence a = 𝐁̂_t e with (𝐁̂_t)_ij = (𝐁_t)_ij / R^{2t+1+j−i}, dimensionless, and
#
#       e⁽ʲ⁾⁽q⁾_kl − Σ_{p≠q} Σ_{i,t,s} W · a⁽ⁱ⁾⁽ᵖ⁾_ts = c⁽ʲ⁾_kl R_q^{k−2+j} / N_kl ,
#       W = (−1)^{k+l} η⁽ⁱ⁾⁽ʲ⁾_tksl(X_q − X_p) R_q^{k−2+j} R_p^{t+3−i} / (N_kl N_ts) .
#
#  EFFECTIVE STIFFNESS.  With τ = σ − ℂ₀:ε the polarization, zero in the matrix,
#  the average of τ over a volume V holding the spheres reduces to integrals
#  over their surfaces in which only the dipoles of the series remain, the
#  center of dilatation A⁽¹⁾_00 and the five A⁽³⁾_2s (kushch2013, Eq. (5.75)):
#
#       tr ⟨τ⟩              =  24π μ₀ (1−ν₀)/(1−2ν₀) Σ_q A⁽¹⁾_00 / V ,
#       2⟨τ₃₃⟩ − ⟨τ₁₁⟩ − ⟨τ₂₂⟩ = −32π μ₀ (1−ν₀) Σ_q A⁽³⁾_20 / V ,
#       ⟨τ₁₃⟩ − i ⟨τ₂₃⟩      = −16π μ₀ (1−ν₀) Σ_q A⁽³⁾_21 / V ,
#       ⟨τ₁₁⟩ − ⟨τ₂₂⟩ − 2i ⟨τ₁₂⟩ = −64π μ₀ (1−ν₀) Σ_q A⁽³⁾_22 / V .
#
#  Solved for the six unit strains, they give 𝚺 such that ⟨τ⟩ = 𝚺 : E, closed
#  as in conduction.
# =============================================================================

# Position of the solution (j, t, s) among the unknowns of one sphere: the
# dilatation first, then three solutions per harmonic of degree t ≥ 1.
@inline _mem_eindex(j::Integer, t::Integer, s::Integer) = t == 0 ? 1 : 1 + 3 * (_mem_index(t, s) - 1) + j
@inline _mem_eblock(tmax::Integer) = 1 + 3 * _mem_block(tmax)

# The solutions that exist at a degree: sent families, received families.
@inline _mem_sent(t::Integer) = t == 0 ? (1:1) : (1:3)
@inline _mem_received(k::Integer) = k == 0 ? (3:3) : (1:3)

"""
    ElasticMultipoleSolution

Solution of the multipole expansion of an assembly in elasticity: for each
sphere, in the order of `names`, its radius (`radii`), its responses
``(\\mathbf{B}_t)_{ij}/R^{2t+1+j-i}``, made dimensionless (`responses[q][t + 1]`), and the scaled
amplitudes it receives under the six unit strains of the Kelvin-Mandel basis
(`received`, one column per strain).
"""
struct ElasticMultipoleSolution{T, C}
    names::Vector{Symbol}
    radii::Vector{T}
    responses::Vector{Vector{Matrix{T}}}
    received::Matrix{C}
    tmax::Int
end

function _mem_solve(asm::ParticleAssembly, C₀::TensND.AbstractTens{4, 3}, prop::Symbol, tmax::Int)
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
    R = _common_eltype([LayeredSpheres.outer_radius(s) for s in spheres])
    raw = [[harmonic_response(s, C₀, t) for t in 0:tmax] for s in spheres]
    centers = [particle_center(asm, nm) for nm in names]
    TX = mapreduce(eltype, promote_type, centers)
    T = float(promote_type(eltype(R), TX, typeof(ν₀), mapreduce(B -> eltype(first(B)), promote_type, raw)))
    Rᵀ = convert(Vector{T}, R)
    # 𝐁̂_t = 𝐁_t / R^{2t+1+j−i}
    B̂ = [
        [T[raw[q][t + 1][i, j] / Rᵀ[q]^(2t + 1 + j - i) for i in 1:3, j in 1:3] for t in 0:tmax]
            for q in eachindex(spheres)
    ]
    X = [SVector{3, TX}(c) for c in centers]
    e = _mem_vector_system(Rᵀ, B̂, X, T(ν₀), asm.boundary, tmax)
    return ElasticMultipoleSolution(copy(names), Rᵀ, B̂, e, tmax)
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

# Function barrier, as in conduction. Returns the scaled amplitudes received
# by every sphere, one column per unit strain of the Kelvin-Mandel basis.
function _mem_vector_system(
        R::Vector{T}, B̂::Vector{Vector{Matrix{T}}}, X::Vector, ν₀::T, boundary, tmax::Int
    ) where {T}
    N, n = length(R), _mem_eblock(tmax)
    _check_system_size(N * n, Complex{T}, N, tmax)
    M = zeros(Complex{T}, N * n, N * n)
    rhs = zeros(Complex{T}, N * n, 6)
    norms = [harmonic_norm(t, s) for t in 0:tmax for s in (-t):t]
    for q in 1:N
        oq = (q - 1) * n
        for i in 1:n
            M[oq + i, oq + i] = one(T)
        end
        _mem_remote_strain!(rhs, oq, R[q], ν₀)
        for p in 1:N
            p == q && continue
            r = X[q] - X[p]
            Y = _mem_translation(boundary, r, false, 2 * tmax)
            _mem_couple_vector!(M, oq, (p - 1) * n, Y, r, R[q], R[p], B̂[p], ν₀, norms, tmax)
        end
    end
    return M \ rhs
end

# Position of N_ts in the table of the norms, degree 0 included.
@inline _mem_norm_index(t::Integer, s::Integer) = t * t + t + s + 1

# Right-hand side of the six unit strains of the Kelvin-Mandel basis
# (11, 22, 33, √2·23, √2·13, √2·12) on the rows of one sphere.
function _mem_remote_strain!(rhs::Matrix{Complex{T}}, oq::Int, R::T, ν₀::T) where {T}
    γ₀ = (4 * ν₀ - 2) / 3
    h = sqrt(T(2)) / 2                                  # the shear of a unit Mandel strain
    N₂ = (harmonic_norm(2, 0), harmonic_norm(2, 1), harmonic_norm(2, 2))
    row(s) = oq + _mem_eindex(1, 2, s)
    # the dilatation: tr E = 1 for the three normal strains
    for m in 1:3
        rhs[oq + 1, m] = R / (3 * γ₀)
    end
    # c⁽¹⁾_20 = (2E₃₃ − E₁₁ − E₂₂)/3
    c₂₀ = (-one(T) / 3, -one(T) / 3, 2 * one(T) / 3)
    for m in 1:3
        rhs[row(0), m] = c₂₀[m] * R / N₂[1]
    end
    # c⁽¹⁾_22 = E₁₁ − E₂₂ − 2i E₁₂ and its conjugate at s = −2
    for (m, c) in ((1, complex(one(T))), (2, complex(-one(T))), (6, -2im * h))
        rhs[row(2), m] = c * R / N₂[3]
        rhs[row(-2), m] = conj(c) * R / N₂[3]
    end
    # c⁽¹⁾_21 = E₁₃ − i E₂₃ and c⁽¹⁾_2,−1 = −conj(c⁽¹⁾_21)
    for (m, c) in ((5, complex(h)), (4, -im * h))
        rhs[row(1), m] = c * R / N₂[2]
        rhs[row(-1), m] = -conj(c) * R / N₂[2]
    end
    return rhs
end

# Block of the field sphere p sends to sphere q: for each solution received
# (j, k, l) and each one sent (i, t, s), the translation coefficient times the
# response of p, which turns what p receives (the unknowns) into what it sends.
function _mem_couple_vector!(M::Matrix, oq::Int, op::Int, Y, r, Rq, Rp, B̂p, ν₀, norms, tmax::Int)
    for k in 0:tmax, l in (-k):k, j in _mem_received(k)
        row = oq + _mem_eindex(j, k, l)
        wq = Rq^(k - 2 + j) / norms[_mem_norm_index(k, l)]
        isodd(k + l) && (wq = -wq)
        for t in 0:tmax, s in (-t):t, i in _mem_sent(t)
            a, b = lame_translation(i, j, t, k, s, l, Y, r)
            w = wq * (a + 4 * (1 - ν₀) * b) * Rp^(t + 3 - i) / norms[_mem_norm_index(t, s)]
            for j′ in _mem_received(t)
                M[row, op + _mem_eindex(j′, t, s)] -= w * B̂p[t + 1][i, j′]
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
    n = _mem_eblock(sol.tmax)
    N₂ = (harmonic_norm(2, 0), harmonic_norm(2, 1), harmonic_norm(2, 2))
    Σ = zeros(T, 6, 6)
    h = sqrt(T(2))
    for q in eachindex(sol.names)
        oq = (q - 1) * n
        R² = sol.radii[q]^2
        B̂₀, B̂₂ = sol.responses[q][1], sol.responses[q][3]
        # A⁽¹⁾_00 = a R² and A⁽³⁾_2s = a R² / N_2s, with a = 𝐁̂ e
        A₀ = B̂₀[1, 3] * R² .* view(sol.received, oq + 1, :)
        A₂(s) = (R² / N₂[s + 1]) .* sum(
            B̂₂[3, j] .* view(sol.received, oq + _mem_eindex(j, 2, s), :) for j in 1:3
        )
        tr_τ = 24π * μ₀ * (1 - ν₀) / (1 - 2 * ν₀) .* real.(A₀)
        dev = -32π * μ₀ * (1 - ν₀) .* real.(A₂(0))               # 2τ₃₃ − τ₁₁ − τ₂₂
        q₁ = -16π * μ₀ * (1 - ν₀) .* A₂(1)                      # τ₁₃ − i τ₂₃
        q₂ = -64π * μ₀ * (1 - ν₀) .* A₂(2)                      # τ₁₁ − τ₂₂ − 2i τ₁₂
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
