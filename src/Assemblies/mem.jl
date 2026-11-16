# =============================================================================
#  mem.jl — the multipole expansion method of Kushch (kushch2013).
#
#  CONDUCTION.  In the matrix, of conductivity k₀, the temperature is the remote
#  field plus one series of decaying solid harmonics per particle,
#
#       T(x) = G·x + Σ_p Σ_{k≥1} Σ_{|l|≤k} A_kl^(p) Y_k^l(x − X_p) .
#
#  Seen from the center of particle q, the series of every other particle is a
#  regular field: the translation formula of `Harmonics/solid_harmonics.jl`
#  writes it on the regular harmonics y_t^s(x − X_q),
#
#       Y_k^l(x − X_p) = Σ_{t,s} (−1)^{t+s} Y_{k+t}^{l−s}(X_q − X_p) y_t^s(x − X_q) .
#
#  Particle q thus receives, on the harmonic (t, s), the amplitude
#
#       c_ts^(q) + Σ_{p≠q} Σ_{k,l} (−1)^{t+s} Y_{k+t}^{l−s}(X_q − X_p) A_kl^(p) ,
#
#  c_ts being the coefficients of G·x (degree 1 only, since a constant sends
#  nothing back), and answers by its own response ρ_t = `harmonic_response`:
#  with the radial functions rᵗ/(t+s)! and (t−s)!/r^{t+1} of the two families,
#
#       (t−s)! (t+s)! A_ts^(q) = ρ_t^(q) [ c_ts^(q) + Σ_{p≠q} … ] .
#
#  That is the linear system. Its unknowns are scaled as in kushch2013,
#  Eq. (2.52), x_ts^(q) = N_ts A_ts^(q) / R_q^{t+1} with N_ts = √((t−s)!(t+s)!)
#  and R_q the radius, and its rows by R_q^t / N_ts, which leaves
#
#       x_ts^(q) − ρ̂_t^(q) Σ_{p≠q} Σ_{k,l} W_{ts,kl}^{qp} x_kl^(p) = ρ̂_t^(q) R_q^t c_ts^(q) / N_ts ,
#
#       ρ̂_t = ρ_t / R^{2t+1} ,
#       W_{ts,kl}^{qp} = (−1)^{t+s} R_q^t R_p^{k+1} Y_{k+t}^{l−s}(X_q − X_p) / (N_ts N_kl) ,
#
#  with entries of order one at every degree, and a particle without contrast
#  (ρ̂ = 0) simply silent.
#
#  EFFECTIVE PROPERTY.  Far from particle q its series reduces to the dipole
#  d^(q)·x/r³, d^(q) = (A_11 − A_1,−1, i (A_11 + A_1,−1), A_10), and the average
#  flux of a volume V holding the particles is −k₀ G + (4π k₀ / V) Σ_q d^(q)
#  (kushch2013, Eq. (2.12)). Solving for the three unit gradients gives the
#  tensor 𝐃 = Σ_q 𝐃^(q), d^(q) = 𝐃^(q)·G, hence the contribution per unit volume
#
#       𝚺 = −(4π k₀ / V) 𝐃 ,
#
#  which a finite assembly closes by Maxwell's argument, `_maxwell_closure`,
#  with the Hill tensor of its domain.
# =============================================================================

# Position of the harmonic (t, s), t ≥ 1, among the unknowns of one particle.
@inline _mem_index(t::Integer, s::Integer) = t * t + t + s
@inline _mem_block(tmax::Integer) = tmax * (tmax + 2)

function MFH_Core.validate_cell(asm::ParticleAssembly, scheme::MultipoleExpansion)
    validate_assembly(asm)
    _assembly_dimension(asm) == 3 || throw(
        ArgumentError(
            "MultipoleExpansion: the expansion in solid harmonics is three-dimensional; " *
                "got a $(_assembly_dimension(asm))-D assembly"
        )
    )
    tmax = _mem_tmax(scheme)
    tmax ≥ 1 || throw(ArgumentError("MultipoleExpansion: `tmax` must be at least 1, got $tmax"))
    return asm
end

_mem_tmax(scheme::MultipoleExpansion) = Int(get(scheme.options, :tmax, 3))

"""
    _evaluate(asm::ParticleAssembly, scheme::MultipoleExpansion, ::Val{p}; kw...) -> AbstractTens

Homogenization of property `:p` over a particle assembly by the multipole
expansion method [kushch2013](@cite). The derivation is in the header of the
file and in the theory page [The multipole expansion method](@ref th-mem).
"""
function Schemes._evaluate(
        asm::ParticleAssembly, scheme::MultipoleExpansion, ::Val{p}; kw...
    ) where {p}
    P₀ = matrix_property(asm, p)
    Σ = _mem_contribution(asm, P₀, p, _mem_tmax(scheme))
    return _mem_closure(asm.boundary, P₀, Σ; kw...)
end

_mem_closure(b::MixedBC, P₀, Σ; kw...) =
    Schemes._maxwell_closure(P₀, Σ, Elasticity.hill_tensor(b.shape, P₀; kw...))

# A periodic assembly needs no closure: the average gradient is the remote one.
_mem_closure(::PeriodicBox, P₀, Σ; kw...) = P₀ + Σ

# ─── The spheres of the assembly ─────────────────────────────────────────────

# A particle as the layered sphere whose harmonic response is asked for: a
# ball becomes its one-layer sphere, a layered sphere is taken as it is.
_mem_sphere(geom::LayeredSpheres.LayeredSphere, asm, name::Symbol, prop::Symbol) = geom

function _mem_sphere(geom::Elasticity.Ellipsoid{3}, asm, name::Symbol, prop::Symbol)
    a = geom.semi_axes
    a[1] == a[2] == a[3] || _mem_not_a_sphere(name, geom)
    P = particle_property(asm, name, prop)
    P isa TensND.TensISO || throw(
        ArgumentError(
            "MultipoleExpansion: particle :$(name) has a $(nameof(typeof(P))) property " *
                ":$(prop); the spheres of the expansion are isotropic (TensISO)"
        )
    )
    return LayeredSpheres.LayeredSphere((a[1],), (P,))
end

_mem_not_a_sphere(name::Symbol, geom) = throw(
    ArgumentError(
        "MultipoleExpansion: particle :$(name) is a $(nameof(typeof(geom))) that is not a " *
            "sphere; the scheme takes balls (an Ellipsoid of equal semi-axes) and LayeredSphere"
    )
)

function _mem_matrix_scalar(P₀)
    P₀ isa TensND.TensISO || throw(
        ArgumentError(
            "MultipoleExpansion: the matrix property is a $(nameof(typeof(P₀))); the " *
                "expansion in solid harmonics is written for an isotropic matrix (TensISO)"
        )
    )
    return LayeredSpheres._iso_scalar(P₀)
end

# ─── Conduction ──────────────────────────────────────────────────────────────

"""
    _mem_contribution(asm, K₀, prop, tmax) -> AbstractTens

Sum ``\\boldsymbol{\\Sigma}`` of the contributions of the particles per unit
volume of the assembly, interactions included: the tensor such that the average
flux is ``-(\\boldsymbol{K}_0 + \\boldsymbol{\\Sigma})\\cdot\\underline{G}`` under the
remote gradient ``\\underline{G}``.
"""
function _mem_contribution(asm::ParticleAssembly, K₀::TensND.AbstractTens{2, 3}, prop::Symbol, tmax::Int)
    k₀ = _mem_matrix_scalar(K₀)
    names = particle_names(asm)
    spheres = [_mem_sphere(particle_geometry(asm, nm), asm, nm, prop) for nm in names]
    # Particles differ in their number types (one radius or one center carrying
    # a derivative, the others not): each array is brought to its common type.
    R = _common_eltype([LayeredSpheres.outer_radius(s) for s in spheres])
    ρ̂ = _common_eltype(
        [harmonic_response(spheres[q], K₀, t) / R[q]^(2t + 1) for q in eachindex(spheres), t in 1:tmax]
    )
    centers = [particle_center(asm, nm) for nm in names]
    TX = mapreduce(eltype, promote_type, centers)
    X = [SVector{3, TX}(c) for c in centers]
    D = _mem_scalar_dipoles(R, ρ̂, X, asm.boundary, tmax)
    return TensND.Tens((-4π * k₀ / assembly_volume(asm)) .* (D .+ transpose(D)) ./ 2)
end

# An array of numbers of different types, in the type that holds them all.
_common_eltype(v::AbstractArray) = convert(AbstractArray{mapreduce(typeof, promote_type, v)}, v)

# Function barrier: radii, responses and centers arrive with their concrete
# element types, whatever the particles were.
function _mem_scalar_dipoles(R::Vector, ρ̂::Matrix, X::Vector, boundary, tmax::Int)
    T = float(promote_type(eltype(R), eltype(ρ̂), eltype(eltype(X))))
    N, n = length(R), _mem_block(tmax)
    _check_system_size(N * n, Complex{T}, N, tmax)
    M = zeros(Complex{T}, N * n, N * n)
    B = zeros(Complex{T}, N * n, 3)
    norms = [harmonic_norm(t, s) for t in 1:tmax for s in (-t):t]
    for q in 1:N
        oq = (q - 1) * n
        for t in 1:tmax, s in (-t):t
            M[oq + _mem_index(t, s), oq + _mem_index(t, s)] = one(T)
        end
        _mem_remote_gradient!(B, oq, ρ̂[q, 1] * R[q])
        for p in 1:N
            Y = _mem_translation(boundary, X[q] - X[p], p == q, 2 * tmax)
            Y === nothing && continue
            _mem_couple!(M, oq, (p - 1) * n, Y, R[q], R[p], view(ρ̂, q, :), norms, tmax)
            _mem_background!(M, oq, (p - 1) * n, boundary, ρ̂[q, 1] * R[q] * R[p]^2)
        end
    end
    x = M \ B
    D = zeros(T, 3, 3)
    for q in 1:N
        oq = (q - 1) * n
        s = R[q]^2
        for i in 1:3
            a₀ = x[oq + _mem_index(1, 0), i] * s
            a₊ = x[oq + _mem_index(1, 1), i] * s / sqrt(T(2))
            a₋ = x[oq + _mem_index(1, -1), i] * s / sqrt(T(2))
            D[1, i] += real(a₊ - a₋)
            D[2, i] += real(im * (a₊ + a₋))
            D[3, i] += real(a₀)
        end
    end
    return D
end

# Right-hand side of the three unit gradients on the degree-1 rows of one
# particle: G·x = G₃ y₁⁰ + (G₁ − iG₂) y₁¹ − (G₁ + iG₂) y₁⁻¹, scaled by
# ρ̂₁ R / N₁ₛ with N₁₀ = 1 and N₁,±₁ = √2.
function _mem_remote_gradient!(B::Matrix{Complex{T}}, oq::Int, ρR) where {T}
    h = ρR / sqrt(T(2))
    B[oq + _mem_index(1, 1), 1] = h
    B[oq + _mem_index(1, -1), 1] = -h
    B[oq + _mem_index(1, 1), 2] = -im * h
    B[oq + _mem_index(1, -1), 2] = -im * h
    B[oq + _mem_index(1, 0), 3] = ρR
    return B
end

# Block of the field particle p sends to particle q, from the table `Y` of the
# decaying harmonics of the vector joining p to q.
function _mem_couple!(M::Matrix, oq::Int, op::Int, Y, Rq, Rp, ρ̂q, norms, tmax::Int)
    for t in 1:tmax, s in (-t):t
        i = _mem_index(t, s)
        row = ρ̂q[t] * Rq^t / norms[i]
        isodd(t + s) && (row = -row)
        for k in 1:tmax, l in (-k):k
            j = _mem_index(k, l)
            M[oq + i, op + j] -= row * Rp^(k + 1) * Y[k + t, l - s] / norms[j]
        end
    end
    return M
end

# Decaying harmonics of the vector from the sender to the receiver; a particle
# of a finite assembly sends nothing to itself.
_mem_translation(::MixedBC, r, self::Bool, tmax::Int) = self ? nothing : irregular_harmonics(r, tmax)

# In a periodic assembly each particle also receives the harmonics of the
# images, its own included, summed over the lattice.
_mem_translation(b::PeriodicBox, r, self::Bool, tmax::Int) =
    lattice_harmonics(r, box_periods(b), tmax)

# Uniform field of the background that makes a periodic dipole periodic
# (`Harmonics/lattice_sums.jl`): the regular part of S has the Laplacian 4π/V,
# so a dipole of the lattice sends to every particle, on top of the degree-2
# harmonics, the field −(4π/3V) × (its moment), the same in the three
# directions. In the scaled unknowns it is one term on the three degree-1
# entries of the block, `w` being ρ̂₁ R_q R_p².
_mem_background!(M, oq::Int, op::Int, ::MixedBC, w) = M
function _mem_background!(M, oq::Int, op::Int, b::PeriodicBox, w)
    g = 4π / (3 * prod(box_periods(b)))
    for s in -1:1
        M[oq + _mem_index(1, s), op + _mem_index(1, s)] += w * g
    end
    return M
end

# The system is dense: refuse a size the machine cannot hold before allocating
# it, naming what to reduce.
function _check_system_size(n::Int, ::Type{C}, N::Int, tmax::Int; available = Sys.free_memory()) where {C}
    bytes = 2 * n^2 * sizeof(C)     # the matrix and its factorization
    bytes ≤ available / 2 || throw(
        ArgumentError(
            "MultipoleExpansion: $N particles at tmax = $tmax give a dense system of " *
                "$n unknowns ($(round(bytes / 2^30; digits = 1)) GiB), more than half the " *
                "free memory; lower `tmax` or the number of particles"
        )
    )
    return nothing
end
