# =============================================================================
#  concentrations.jl — concentration tensors of the N-body schemes.
#
#  The cluster model solves directly for the strain concentration tensor of
#  every family; the equivalent inclusion method solves for the polarization of
#  every particle, from which its mean strain follows as (ℂ_a − ℂ₀)⁻¹ : τ_a, the
#  stationarity condition of the Galerkin problem.  In both, the matrix takes
#  the rest of the averages.
# =============================================================================

function Schemes._concentration_tensors(asm::ParticleAssembly, scheme::ClusterModel, ::Val{p}; kw...) where {p}
    P₀ = matrix_property(asm, p)
    R_c = get(scheme.options, :cluster_radius, nothing)
    A, reps = _cluster_concentrations(asm, P₀, p, R_c, _kernel_options(scheme.options); kw...)
    P_hom = _effective_from_concentrations(asm, P₀, p, A, reps)
    # Families carry integer labels; the result is keyed by `:family_<label>`.
    families = [particle_family(asm, r) for r in reps]
    labels = [_family_key(lab) for lab in families]
    weights = Dict(_family_key(lab) => _family_fraction(asm, lab) for lab in families)
    strain = Dict{Symbol, TensND.AbstractTens}(lab => A[k] for (k, lab) in enumerate(labels))
    stress = Dict{Symbol, TensND.AbstractTens}(
        lab => particle_property(asm, reps[k], p) ⊙ A[k] for (k, lab) in enumerate(labels)
    )
    properties = Dict{Symbol, TensND.AbstractTens}(
        lab => particle_property(asm, reps[k], p) for (k, lab) in enumerate(labels)
    )
    return _with_assembly_matrix(asm, P₀, P_hom, labels, strain, stress, weights, properties)
end

function Schemes._concentration_tensors(
        asm::ParticleAssembly, scheme::EquivalentInclusion, v::Val{p}; kw...
    ) where {p}
    P_hom = Schemes._evaluate(asm, scheme, v; kw...)
    P₀ = matrix_property(asm, p)
    τ, names = _eim_polarizations(asm, P₀, p, scheme; kw...)
    weights = Dict(n => particle_volume_fraction(asm, n) for n in names)
    properties = Dict{Symbol, TensND.AbstractTens}(n => particle_property(asm, n, p) for n in names)
    strain = Dict{Symbol, TensND.AbstractTens}(
        n => inv(properties[n] - P₀) ⊙ τ[k] for (k, n) in enumerate(names)
    )
    stress = Dict{Symbol, TensND.AbstractTens}(n => properties[n] ⊙ strain[n] for n in names)
    return _with_assembly_matrix(asm, P₀, P_hom, names, strain, stress, weights, properties)
end

_family_key(label::Integer) = Symbol(:family_, label)

function _with_assembly_matrix(asm, P₀, P_hom, names, strain, stress, weights, properties)
    m = asm.matrix_name
    f_m = one(eltype(P_hom)) - sum(values(weights))
    Id = _identity_like(P₀)
    strain[m] = Schemes._per_weight(Id - sum((weights[n] * strain[n] for n in names); init = zero(P₀)), f_m)
    stress[m] = Schemes._per_weight(P_hom - sum((weights[n] * stress[n] for n in names); init = zero(P_hom)), f_m)
    weights_all = merge(weights, Dict(m => f_m))
    properties[m] = P₀
    return Schemes._concentration_result(P_hom, vcat(m, names), strain, stress, weights_all, properties, m)
end
