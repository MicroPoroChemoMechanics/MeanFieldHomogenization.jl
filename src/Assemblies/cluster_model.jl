# =============================================================================
#  cluster_model.jl — the cluster model of Molinari & El Mouden (1996).
#
#  Starting from the Zeller-Dederichs integral equation with the matrix as
#  reference medium and a uniform strain per inclusion, the mean strain of
#  inclusion I obeys
#
#     ε^I = E - Σ_J 𝕋^{IJ} : δℂ^J : ε^J + 𝔼⁰ : Σ_K f_K δℂ^K : ε^K ,     (23)
#
#  the last term carrying the far field of the periodic array.  Two identities
#  tie this to the rest of the package:
#
#     𝕋^{II} = +ℙ        and        𝔼⁰ = +ℙ₀ ,
#
#  i.e. the self term of the interaction family IS the Hill tensor
#  (`self_interaction_tensor`) and the far-field operator is the Hill tensor of
#  the inclusion shape.  Splitting 𝕋^{II} out of the sum and writing ε^K = 𝔸^K:E
#  turns (23) into the block system Σ_K 𝕄_{IK}:𝔸^K = 𝕀 with the single form
#
#     𝕄_{IK} = δ_{IK} 𝕀 + [𝕋̄_{IK} + (δ_{IK} - f_K) ℙ₀] : δℂ_K ,
#
#  where 𝕋̄_{IK} sums the pair interactions between the reference inclusion of
#  family I and every inclusion of family K inside the cluster, excluding the
#  reference inclusion itself.
#
#  SIGN.  Molinari & El Mouden's own Γ^{IJ} is the OPPOSITE of the package's
#  𝕋^{IJ} — theirs satisfies Γ^{II} = -ℙ.  Eq. (23) above is therefore their
#  equation with the sign of the interaction sum already flipped, which is also
#  what turns their two block cases into the one expression above.  See the
#  convention block of `Interactions/api.jl`.
#
#  The Mori-Tanaka limit is immediate and is the acceptance gate of this file:
#  with an empty cluster every 𝕋̄ vanishes, 𝕄_{II} = 𝕀 + (1-f)ℙ₀:δℂ and
#  𝕄_{IK} = -f_K ℙ₀:δℂ_K, which is exactly the Mori-Tanaka system (Appendix C
#  of the paper).
# =============================================================================

"""
    _evaluate(asm::ParticleAssembly, scheme::ClusterModel, ::Val{p}; kw...) -> AbstractTens

Cluster-model homogenization of property `:p` over a particle assembly
([Molinari & El Mouden 1996](@cite molinari1996)).

Elasticity (4th-order property) and conduction (2nd-order property) go through
the same code: the only difference is the order of the tensors being solved
for, which the Kelvin-Mandel flattening in `block_solve.jl` handles.
"""
function Schemes._evaluate(
        asm::ParticleAssembly, scheme::ClusterModel, ::Val{p}; kw...
    ) where {p}
    P₀ = matrix_property(asm, p)
    R_c = get(scheme.options, :cluster_radius, nothing)
    pass = _kernel_options(scheme.options)
    A, names = _cluster_localizations(asm, P₀, p, R_c, pass; kw...)
    return _effective_from_localizations(asm, P₀, p, A, names)
end

"""
    _cluster_localizations(asm, P₀, prop, R_c, pass; kw...) -> (Vector, Vector{Symbol})

Solve the cluster system for the strain (or gradient) localization tensor of
every family, returning them alongside the representative particle of each
family.
"""
function _cluster_localizations(asm, P₀, prop::Symbol, R_c, pass; kw...)
    labels = family_labels(asm)
    reps = [_family_representative(asm, lab) for lab in labels]
    n = length(labels)
    L, cutoff = _lattice_parameters(asm, R_c)
    Id = _identity_like(P₀)

    # Volume fraction and property contrast, per family.
    f = [_family_fraction(asm, lab) for lab in labels]
    δP = [particle_property(asm, reps[k], prop) - P₀ for k in 1:n]
    # The far-field operator 𝔼⁰ is the Hill tensor of the inclusion shape in
    # the matrix; with a single shape per family it is read off the
    # representative particle.
    P_hill = [
        Elasticity.hill_tensor(particle_geometry(asm, reps[k]), P₀; kw...) for k in 1:n
    ]

    # `Any` element type on purpose: the blocks come out with whatever TensND
    # symmetry class the algebra produces (`TensISO` for an isotropic pair,
    # `TensCanonical` once an interaction breaks the symmetry), and pinning the
    # container to the first of them would force a conversion that does not
    # exist.  The cost is irrelevant — `_solve_tensor_system` flattens
    # everything onto Kelvin-Mandel immediately, and the O(N²) interaction
    # assembly above dominates by orders of magnitude.
    blocks = [Vector{Any}(undef, n) for _ in 1:n]
    rhs = [Id for _ in 1:n]
    for i in 1:n
        for k in 1:n
            𝕋̄ = _family_interaction(asm, reps[i], labels[k], P₀, L, cutoff, pass)
            # 𝕄_{IK} = δ_{IK} 𝕀 + [𝕋̄_{IK} + (δ_{IK} - f_K) ℙ₀] : δℂ_K — one
            # expression for both cases, which is what the Brisard sign
            # convention buys over Molinari's (see the file header).
            M = if i == k
                Id + (𝕋̄ + (1 - f[k]) * P_hill[i]) ⊙ δP[k]
            else
                (𝕋̄ - f[k] * P_hill[i]) ⊙ δP[k]
            end
            blocks[i][k] = M
        end
    end
    return _solve_tensor_system(blocks, rhs, P₀), reps
end

"""
    _family_interaction(asm, rep, label, P₀, L, cutoff, pass) -> AbstractTens

``\\bar{\\mathbb{T}}_{IK}``: the sum of the pairwise interaction tensors between the
reference particle `rep` of family `I` and every particle of family `label`
lying inside the cluster — including the periodic images, and excluding `rep`
itself.
"""
function _family_interaction(asm, rep::Symbol, label::Int, P₀, L, cutoff, pass)
    acc = _zero_like(P₀)
    ga = particle_geometry(asm, rep)
    ca = particle_center(asm, rep)
    for nm in particle_names(asm)
        particle_family(asm, nm) == label || continue
        gb = particle_geometry(asm, nm)
        r = [particle_center(asm, nm)[i] - ca[i] for i in eachindex(ca)]
        acc = acc + Interactions.lattice_interaction_tensor(ga, gb, r, P₀, L, cutoff; pass...)
    end
    return acc
end

# ─── Effective property from the localization tensors ────────────────────────

"""
    _effective_from_localizations(asm, P₀, prop, A, reps) -> AbstractTens

Assemble the effective property from the per-family localization tensors,

```math
\\mathbb{C}^{hom} = f_m\\, \\mathbb{C}_m : \\mathbb{A}_m
  + \\sum_I f_I\\, \\mathbb{C}_I : \\mathbb{A}_I ,
```

with the matrix localization following from the strain average rule,
``f_m \\mathbb{A}_m = \\mathbb{I} - \\sum_I f_I \\mathbb{A}_I``.
"""
function _effective_from_localizations(asm, P₀, prop::Symbol, A, reps)
    Id = _identity_like(P₀)
    f = [_family_fraction(asm, particle_family(asm, reps[k])) for k in eachindex(reps)]
    acc = Id
    for k in eachindex(reps)
        acc = acc - f[k] * A[k]
    end
    # acc is now f_m 𝔸_m; contracting it with the matrix property gives the
    # matrix share of the average directly, with no division by f_m.
    out = P₀ ⊙ acc
    for k in eachindex(reps)
        out = out + f[k] * (particle_property(asm, reps[k], prop) ⊙ A[k])
    end
    return out
end

# ─── Small helpers shared with the EIM kernel ────────────────────────────────

"""
    _family_representative(asm, label) -> Symbol

First particle carrying the given family label. Particles of one family are
constrained to share their unknown, so any of them describes the family.
"""
function _family_representative(asm::ParticleAssembly, label::Int)
    for nm in particle_names(asm)
        particle_family(asm, nm) == label && return nm
    end
    return throw(ArgumentError("ParticleAssembly: no particle in family $(label)"))
end

"""
    _family_fraction(asm, label) -> Real

Total volume fraction of the particles sharing a family label.
"""
_family_fraction(asm::ParticleAssembly, label::Int) = sum(
    particle_volume_fraction(asm, nm)
        for nm in particle_names(asm) if particle_family(asm, nm) == label;
    init = 0.0
)

"""
    _lattice_parameters(asm, override) -> (period, cutoff)

Period and cluster cutoff used for the interaction sums. A `PeriodicBox` gives
both; under [`MixedBC`](@ref) there is no lattice, so the period is reported as
infinite and the cutoff as zero, which makes every image sum collapse to the
single primary pair.
"""
function _lattice_parameters(asm::ParticleAssembly, override)
    b = asm.boundary
    if b isa PeriodicBox
        return (b.period, override === nothing ? b.cutoff : override)
    end
    # No periodicity: keep only the primary images.  A period larger than any
    # cutoff achieves that without a special case in the summation loop.
    c = override === nothing ? _mixed_bc_reach(asm) : override
    return (10 * c + one(c), c)
end

# Under mixed boundary conditions every particle interacts with every other
# one, and only with them — a reach spanning the SVE does exactly that.
_mixed_bc_reach(asm::ParticleAssembly) = 4 * _bounding_radius(asm.boundary.shape)

# Options forwarded to `interaction_tensor` (back-end selection, multipole
# order, quadrature nodes), separated from the options meant for the scheme.
const _SCHEME_ONLY_OPTIONS = (:cluster_radius, :order)

_kernel_options(opts::NamedTuple) = Base.structdiff(opts, NamedTuple{_SCHEME_ONLY_OPTIONS})

# Order-agnostic algebra: `⊙` is the contraction matching the tensor order —
# `⊡` for 4th-order properties, `⋅` for 2nd-order ones.  Writing the kernels
# against it is what lets one implementation serve elasticity and conduction,
# exactly as the one-site schemes do with their `_dispatch` pairs.
⊙(A::TensND.AbstractTens{4}, B::TensND.AbstractTens{4}) = A ⊡ B
⊙(A::TensND.AbstractTens{2}, B::TensND.AbstractTens{2}) = A ⋅ B

# The element type is taken from the reference property, not defaulted:
# `tens_Id4()` builds a symbolic identity, which would poison an otherwise
# `Float64` (or `Dual`) system on first contact.
_identity_like(P₀::TensND.AbstractTens{4, 3}) = TensND.tens_Id4(Val(3), Val(eltype(P₀)))
_identity_like(P₀::TensND.AbstractTens{4, 2}) = TensND.tens_Id4(Val(2), Val(eltype(P₀)))
_identity_like(P₀::TensND.AbstractTens{2, 3}) = TensND.tens_Id2(Val(3), Val(eltype(P₀)))
_identity_like(P₀::TensND.AbstractTens{2, 2}) = TensND.tens_Id2(Val(2), Val(eltype(P₀)))

_zero_like(P₀::TensND.AbstractTens{4, 3}) = TensND.Tens(zeros(eltype(P₀), 3, 3, 3, 3))
_zero_like(P₀::TensND.AbstractTens{4, 2}) = TensND.Tens(zeros(eltype(P₀), 2, 2, 2, 2))
_zero_like(P₀::TensND.AbstractTens{2, 3}) = TensND.Tens(zeros(eltype(P₀), 3, 3))
_zero_like(P₀::TensND.AbstractTens{2, 2}) = TensND.Tens(zeros(eltype(P₀), 2, 2))
