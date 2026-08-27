# =============================================================================
#  eim.jl — the variational equivalent inclusion method of Brisard, Dormieux
#  & Sab (2014), at order p = 0 (polarization constant over each inclusion).
#
#  The method is a Galerkin discretization of the weak form of the modified
#  Lippmann-Schwinger equation.  Their Eq. (37), specialized to p = 0 and
#  divided by |Ω_a|, reads
#
#     [(ℂ_a-ℂ₀)⁻¹ + ℙ_a - f_a ℙ_Ω] : τ_a
#       + Σ_{b≠a} [𝕋^{ab} - f_b ℙ_Ω] : τ_b  =  E ,
#
#     ℂ^app : E = ℂ₀ : E + Σ_a f_a τ_a .
#
#  SIGN.  The package shares Brisard's convention (see the block in
#  `Interactions/api.jl`), so this is his Eq. (37) transcribed verbatim: his
#  self term |Ω_a|⁻¹S⁰⁰_a is +ℙ_a, which is `𝕋^{aa}`, and every block above
#  carries a plus.  Nothing is flipped on the way in, and the test that EIM
#  and the cluster model agree on a periodic assembly is what keeps the two
#  transcriptions consistent with each other.
#
#  ℙ_Ω, the Hill tensor of the SVE DOMAIN, is the mixed-boundary-condition
#  term: it closes the far field without periodization and without the
#  conditionally convergent lattice sum that a periodic formulation needs.
#  Under `PeriodicBox` there is no such domain, the ℙ_Ω terms are replaced by
#  the far-field operator ℙ₀ of the inclusion shape, and the formulation
#  becomes Molinari's — which is why the two schemes then coincide.
#
#  BOUNDS.  The discrete problem minimizes (resp. maximizes) a
#  Hashin-Shtrikman functional over a finite-dimensional space when the matrix
#  is stiffer (resp. softer) than every inhomogeneity, so the estimate is then
#  a rigorous upper (resp. lower) bound on the apparent stiffness — §3.2 of the
#  paper.  `eim_bound_type` reports which case holds.
# =============================================================================

"""
    _evaluate(asm::ParticleAssembly, scheme::EquivalentInclusion, ::Val{p}; kw...) -> AbstractTens

Equivalent-inclusion homogenization of property `:p` over a particle assembly
([brisard2014](@cite)).

Only `order = 0` (piecewise-constant polarization) is implemented; higher
orders need the influence *pseudotensors* of the paper's Appendix C, which are
not tensors and require their own change-of-basis machinery.
"""
function Schemes._evaluate(
        asm::ParticleAssembly, scheme::EquivalentInclusion, ::Val{p}; kw...
    ) where {p}
    order = get(scheme.options, :order, 0)
    order == 0 || throw(
        ArgumentError(
            "EquivalentInclusion: only `order = 0` (piecewise-constant " *
                "polarization) is implemented; got $(order). Higher orders require " *
                "the influence pseudotensors of Brisard et al. (2014), App. C."
        )
    )
    P₀ = matrix_property(asm, p)
    τ, names = _eim_polarizations(asm, P₀, p, scheme; kw...)
    return _eim_effective(asm, P₀, p, τ, names)
end

"""
    _eim_polarizations(asm, P₀, prop, scheme; kw...) -> (Vector, Vector{Symbol})

Solve the EIM system for the uniform polarization of every particle.

The system is solved against a *unit* macroscopic loading, so each unknown
comes out as the operator mapping `E` onto the polarization of one particle
rather than as a single polarization — which is what the effective property
needs, and what makes the local fields available for any loading at no extra
cost.
"""
function _eim_polarizations(asm, P₀, prop::Symbol, scheme; kw...)
    names = particle_names(asm)
    n = length(names)
    pass = _kernel_options(scheme.options)
    R_c = get(scheme.options, :cluster_radius, nothing)
    L, cutoff = _lattice_parameters(asm, R_c)
    P_far = _far_field_operator(asm, P₀, names; kw...)

    f = [particle_volume_fraction(asm, nm) for nm in names]
    # See the note in `cluster_model.jl`: the blocks are heterogeneous in
    # TensND symmetry class, so the container stays untyped.
    blocks = [Vector{Any}(undef, n) for _ in 1:n]
    for i in 1:n
        gi = particle_geometry(asm, names[i])
        ci = particle_center(asm, names[i])
        ΔP = particle_property(asm, names[i], prop) - P₀
        Pi = Elasticity.hill_tensor(gi, P₀; kw...)
        for k in 1:n
            gk = particle_geometry(asm, names[k])
            if i == k
                # Self block: the inverse contrast, the Hill tensor of the
                # inclusion (which IS 𝕋^{aa}), and the boundary term.  A
                # lattice adds the interaction with the particle's own
                # periodic images.
                𝕋self = Interactions.lattice_interaction_tensor(
                    gi, gk, zeros(eltype(ci), length(ci)), P₀, L, cutoff; pass...
                )
                blocks[i][k] = inv(ΔP) + Pi - f[k] * P_far + 𝕋self
            else
                r = [particle_center(asm, names[k])[q] - ci[q] for q in eachindex(ci)]
                # `lattice_interaction_tensor` already contains the primary
                # pair (the null translation is dropped only when r = 0), so
                # the direct term must NOT be added again here.
                𝕋 = Interactions.lattice_interaction_tensor(gi, gk, r, P₀, L, cutoff; pass...)
                blocks[i][k] = 𝕋 - f[k] * P_far
            end
        end
    end
    rhs = [_identity_like(P₀) for _ in 1:n]
    return _solve_tensor_system(blocks, rhs, P₀), names
end

"""
    _far_field_operator(asm, P₀, names; kw...) -> AbstractTens

The tensor closing the far field of the assembly: the Hill tensor
``\\mathbb{P}_\\Omega`` of the SVE domain under [`MixedBC`](@ref) — Brisard's
mixed boundary conditions — and the Hill tensor of the inclusion shape under a
[`PeriodicBox`](@ref), which is Molinari's far-field operator ``\\mathbb E^0``.

That single substitution is what makes the two N-body schemes of this package
coincide on a periodic assembly.
"""
_far_field_operator(asm::ParticleAssembly, P₀, names; kw...) =
    _far_field_operator(asm.boundary, asm, P₀, names; kw...)

_far_field_operator(b::MixedBC, asm, P₀, names; kw...) =
    Elasticity.hill_tensor(b.shape, P₀; kw...)

_far_field_operator(::PeriodicBox, asm, P₀, names; kw...) =
    Elasticity.hill_tensor(particle_geometry(asm, names[1]), P₀; kw...)

"""
    _eim_effective(asm, P₀, prop, τ, names) -> AbstractTens

Apparent property from the polarizations,
``\\mathbb{C}^{app} : E = \\mathbb{C}_0 : E + \\sum_a f_a \\boldsymbol{\\tau}_a``.

Because the solve is carried out against a unit macroscopic loading, each
`τ_a` is already the *operator* mapping `E` onto the polarization, and the sum
is assembled directly.
"""
function _eim_effective(asm, P₀, prop::Symbol, τ, names)
    out = P₀
    for k in eachindex(names)
        out = out + particle_volume_fraction(asm, names[k]) * τ[k]
    end
    return out
end

# ─── Public queries ──────────────────────────────────────────────────────────

"""
    eim_bound_type(asm, prop = :C) -> Symbol

Whether the [`EquivalentInclusion`](@ref) estimate of property `prop` is a
rigorous bound on the apparent stiffness of the assembly, and which one:

- `:upper` — the matrix is stiffer than every inhomogeneity;
- `:lower` — the matrix is softer than every inhomogeneity;
- `:none`  — the contrasts have mixed signs, and the estimate is only an
  estimate.

This is the extremum property of [brisard2014](@cite),
§3.2, inherited from the Hashin-Shtrikman variational principle. It also
implies that the estimate improves monotonically with the polarization order.

Definiteness is tested on the Kelvin-Mandel matrix of the contrast
``\\mathbb{C}_a - \\mathbb{C}_0``, which is the exact statement of
``\\mathbb{C}_a \\le \\mathbb{C}_0`` in the sense of quadratic forms.
"""
function eim_bound_type(asm::ParticleAssembly, prop::Symbol = :C)
    P₀ = matrix_property(asm, prop)
    all_neg = true
    all_pos = true
    for nm in particle_names(asm)
        M = _to_mandel(particle_property(asm, nm, prop) - P₀, P₀)
        λ = eigvals(Symmetric(Float64.(M)))
        all(λ .≤ 1.0e-12) || (all_neg = false)
        all(λ .≥ -1.0e-12) || (all_pos = false)
    end
    all_neg && return :upper
    all_pos && return :lower
    return :none
end

"""
    eim_polarizations(asm, prop = :C; kw...) -> (Vector, Vector{Symbol})

Per-particle polarization operators of the [`EquivalentInclusion`](@ref)
solution, together with the particle names, without going through
`homogenize`.

Local fields follow directly: the strain operator of particle `a` is
``(\\mathbb{C}_a - \\mathbb{C}_0)^{-1} : \\boldsymbol{\\tau}_a`` and its stress operator is
``\\mathbb{C}_0 : \\boldsymbol{\\varepsilon}_a + \\boldsymbol{\\tau}_a``.
"""
function eim_polarizations(
        asm::ParticleAssembly, prop::Symbol = :C;
        scheme::EquivalentInclusion = EquivalentInclusion(), kw...
    )
    validate_assembly(asm)
    P₀ = matrix_property(asm, prop)
    return _eim_polarizations(asm, P₀, prop, scheme; kw...)
end

"""
    cluster_localizations(asm, prop = :C; cluster_radius = nothing, kw...)
        -> (Vector, Vector{Symbol})

Per-family localization tensors of the [`ClusterModel`](@ref) solution,
together with the representative particle of each family.

`A[k]` maps the macroscopic strain onto the mean strain of family `k`, so the
mean stress follows as ``\\mathbb{C}_k : \\mathbb{A}_k : E``.
"""
function cluster_localizations(
        asm::ParticleAssembly, prop::Symbol = :C;
        cluster_radius = nothing, kw...
    )
    validate_assembly(asm)
    P₀ = matrix_property(asm, prop)
    return _cluster_localizations(asm, P₀, prop, cluster_radius, NamedTuple(); kw...)
end
