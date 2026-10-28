# =============================================================================
#  crack_stiffness.jl — the stiffness of a solid as a function of the densities
#  of its crack families, shared by the laws whose state moves the densities.
#
#  Generic route: rebuild the RVE at the densities and homogenize it, for any
#  scheme. Affine route: with Mori–Tanaka, and the matrix as the only phase
#  carrying volume, the compliance is exactly affine in the densities,
#
#      𝕊(d) = 𝕊⁰ + Σᵢ dᵢ ℍᵢ ,
#
#  (see `Schemes.crack_family_compliances`), so that the stiffness at any
#  densities is one 6 × 6 inversion. 𝕊⁰ and the ℍᵢ are taken once, at
#  construction, from the generic route itself: ℍᵢ = ∂𝕊/∂dᵢ, a constant since
#  𝕊 is affine, and 𝕊⁰ = 𝕊(d) − Σᵢ dᵢ ℍᵢ. Nothing is written by hand.
# =============================================================================

abstract type AbstractCrackedStiffness end

# Any scheme: the RVE is rebuilt at the densities and homogenized.
struct GenericCrackedStiffness{R, S, N} <: AbstractCrackedStiffness
    rve::R
    scheme::S
    families::NTuple{N, Symbol}
end

# Mori–Tanaka on a matrix and cracks: 𝕊(d) = 𝕊⁰ + Σᵢ dᵢ ℍᵢ, in Mandel components.
struct AffineCrackCompliance{N, T} <: AbstractCrackedStiffness
    S⁰::SMatrix{6, 6, T, 36}
    H::NTuple{N, SMatrix{6, 6, T, 36}}
end

function _generic_stiffness(m::GenericCrackedStiffness{<:Any, <:Any, N}, d) where {N}
    rve = foldl(1:N; init = m.rve) do rve, i
        Schemes.set_param(rve, Schemes.AmountParameter(m.families[i]), d[i])
    end
    return _mandel(TensND.change_tens_canon(Schemes.homogenize(rve, m.scheme, :C)))
end

"""
    cracked_stiffness(rve, scheme, families) -> AbstractCrackedStiffness

The stiffness of `rve` as a function of the densities of its crack `families`:
the affine compliance of Mori–Tanaka when the matrix is the only phase carrying
volume, the generic homogenization otherwise.
"""
function cracked_stiffness(rve, scheme, families::NTuple{N, Symbol}) where {N}
    generic = GenericCrackedStiffness(rve, scheme, families)
    return _affine_or_generic(scheme, rve, generic)
end

_affine_or_generic(scheme, rve, generic) = generic
function _affine_or_generic(scheme::Schemes.MoriTanaka, rve, generic::GenericCrackedStiffness{<:Any, <:Any, N}) where {N}
    # The affine identity holds only when no other phase carries volume.
    isempty(Schemes._volume_phase_names(rve, Schemes.matrix_name(scheme, rve))) || return generic
    compliance(d) = inv(_generic_stiffness(generic, d))
    S = compliance(SVector{N}(map(φ -> Schemes.get_param(rve, Schemes.AmountParameter(φ)), generic.families)))
    # The densities take the element type of the compliance, which carries any
    # dual number of the phases' moduli: the tag of the derivative below then
    # orders after it.
    d₀ = SVector{N, eltype(S)}(map(φ -> Schemes.get_param(rve, Schemes.AmountParameter(φ)), generic.families))
    ∂S = ForwardDiff.jacobian(d -> vec(compliance(d)), d₀)
    H = ntuple(i -> SMatrix{6, 6}(reshape(∂S[:, i], 6, 6)), Val(N))
    S⁰ = S - sum(d₀[i] * H[i] for i in 1:N)
    return AffineCrackCompliance(SMatrix{6, 6}(S⁰), H)
end

# The stiffness at the densities d (an SVector, in the order of the families).
_stiffness_at(m::GenericCrackedStiffness, d) = _generic_stiffness(m, d)
_stiffness_at(m::AffineCrackCompliance{N}, d) where {N} = inv(m.S⁰ + sum(d[i] * m.H[i] for i in 1:N))

# The forces driving the families at the strain ε, Yᵢ = −∂W/∂dᵢ with
# W = ½ ε:ℂ(d):ε. Generic: by automatic differentiation of W. Affine: since
# ∂ℂ/∂dᵢ = −ℂ ℍᵢ ℂ, exactly Yᵢ = ½ σ:ℍᵢ:σ with σ = ℂ:ε.
_driving_forces_at(m::GenericCrackedStiffness, ε, d) = -ForwardDiff.gradient(x -> _quadratic(ε, _generic_stiffness(m, x)), d)
function _driving_forces_at(m::AffineCrackCompliance{N}, ε, d) where {N}
    σ = _stiffness_at(m, d) * ε
    return SVector(ntuple(i -> _quadratic(σ, m.H[i]), Val(N)))
end
