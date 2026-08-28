# =============================================================================
#  schemes_alv.jl — time-domain viscoelastic homogenization schemes.
#
#  All schemes work on the discrete `(6n × 6n)` block matrices produced
#  by `trapezoidal_matrix`.  The Volterra product is the regular matrix
#  product (`*`); the Volterra inverse is `volterra_inverse`.
#
#  Reference: sanahuja2013 ; barthelemyIJSS2016 in docs/src/references.bib
#  IJES 2019 §3 ; ECHOES manual ch07_viscoelasticity (`viscoelasticity_time.qmd`).
# =============================================================================

# Promoted scalar type of every input an iterative ALV scheme's running
# estimate must carry (phase kernels, volume fractions / densities,
# geometry-derived Mandel tensors, crack interface matrices) — so that
# `ForwardDiff.Dual` parameters propagate through the fixed point instead
# of hitting a `Float64` container.
function _alv_promoted_eltype(C_list, fractions, U_list, crack_data)
    Tp = isempty(fractions) ? Float64 : eltype(fractions)
    for C in C_list
        Tp = promote_type(Tp, eltype(C))
    end
    for U in U_list
        Tp = promote_type(Tp, eltype(U))
    end
    for (_, ε, _, Rn, Rt) in crack_data
        Tp = promote_type(Tp, typeof(ε))
        Rn === nothing || (Tp = promote_type(Tp, eltype(Rn)))
        Rt === nothing || (Tp = promote_type(Tp, eltype(Rt)))
    end
    return Tp
end

# A discrete identity 6n×6n matrix in Mandel form: block-diagonal with
# 6×6 identity blocks.
function _identity_alv(n::Int, T::Type)
    M = zeros(T, 6 * n, 6 * n)
    @inbounds for i in 1:n
        for k in 1:6
            M[6 * (i - 1) + k, 6 * (i - 1) + k] = one(T)
        end
    end
    return M
end

# ── Dilute concentration & contribution tensors ─────────────────────────────

"""
    dilute_concentration_alv(C_E, C_0, P) -> Matrix

Dilute strain concentration kernel `Ã^dil = (𝟙 + P̃ ∘ ΔC̃)^{-vol}`
([barthelemyIJES2019](@cite) eq. 16).  All inputs are `(6n × 6n)` block
matrices ; the result is also `(6n × 6n)` and lower-block-triangular.
"""
function dilute_concentration_alv(
        C_E::AbstractMatrix, C_0::AbstractMatrix,
        P::AbstractMatrix
    )
    sz = size(C_E, 1)
    @assert size(C_E) == size(C_0) == size(P) == (sz, sz)
    @assert sz % 6 == 0
    n = sz ÷ 6
    T = promote_type(eltype(C_E), eltype(C_0), eltype(P))
    Id = _identity_alv(n, T)
    ΔC = C_E - C_0
    return volterra_inverse(Id + P * ΔC; block_size = 6)
end

"""
    dilute_contribution_alv(C_E, C_0, P) -> Matrix

Dilute strain contribution kernel `Ñ = ΔC̃ ∘ Ã^dil`
([barthelemyIJES2019](@cite) eq. 17).  This is the size-independent stiffness
contribution of a single inclusion.
"""
function dilute_contribution_alv(
        C_E::AbstractMatrix, C_0::AbstractMatrix,
        P::AbstractMatrix
    )
    A_dil = dilute_concentration_alv(C_E, C_0, P)
    ΔC = C_E - C_0
    return ΔC * A_dil
end

# ── Voigt / Reuss bounds ────────────────────────────────────────────────────

"""
    voigt_alv(C_phases::AbstractVector, fractions::AbstractVector) -> Matrix

Voigt (uniform-strain) bound: `C_eff = Σ_r f_r · C̃^r`.  Each `C_phases[r]`
is a `(6n × 6n)` block matrix and `fractions[r]` is the volume fraction
of phase `r`.
"""
function voigt_alv(
        C_phases::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    length(C_phases) == length(fractions) ||
        throw(ArgumentError("voigt_alv: C_phases and fractions length mismatch"))
    isempty(C_phases) && throw(ArgumentError("voigt_alv: at least one phase required"))
    T = promote_type(eltype(C_phases[1]), eltype(fractions))
    C = zeros(T, size(C_phases[1])...)
    @inbounds for r in eachindex(C_phases)
        @. C += fractions[r] * C_phases[r]
    end
    return C
end

"""
    reuss_alv(C_phases::AbstractVector, fractions::AbstractVector) -> Matrix

Reuss (uniform-stress) bound: invert each `C̃^r`, do the volume average
of compliances, then invert the result.

  J̃_eff = Σ_r f_r · J̃^r ,  C̃_eff = (J̃_eff)^{-vol}
"""
function reuss_alv(
        C_phases::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    length(C_phases) == length(fractions) ||
        throw(ArgumentError("reuss_alv: C_phases and fractions length mismatch"))
    isempty(C_phases) && throw(ArgumentError("reuss_alv: at least one phase required"))
    J_phases = [volterra_inverse(C; block_size = 6) for C in C_phases]
    J_eff = voigt_alv(J_phases, fractions)
    return volterra_inverse(J_eff; block_size = 6)
end

# ── Dilute scheme ───────────────────────────────────────────────────────────

"""
    dilute_alv(C_0, contribs, fractions) -> Matrix

Dilute scheme: `C̃_eff = C̃^0 + Σ_r f_r · Ñ^{r,dil}` where the dilute
contributions `Ñ^{r,dil}` of each inclusion phase are pre-computed via
`dilute_contribution_alv` (each `(6n × 6n)`), `f_r` is the volume
fraction of phase `r`, and the matrix `C̃^0` is the reference.

Note that for the dilute scheme the matrix volume fraction `f_0`
does *not* appear: the inclusions are treated as if they were
embedded in an infinite matrix.
"""
function dilute_alv(
        C_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    length(contribs) == length(fractions) ||
        throw(ArgumentError("dilute_alv: contribs and fractions length mismatch"))
    T = promote_type(
        eltype(C_0),
        (isempty(contribs) ? Float64 : eltype(contribs[1])),
        eltype(fractions)
    )
    C = T.(C_0)
    @inbounds for r in eachindex(contribs)
        @. C += fractions[r] * contribs[r]
    end
    return C
end

# ── Mori-Tanaka scheme ──────────────────────────────────────────────────────

"""
    mori_tanaka_alv(C_0, A_duts, contribs, fractions, f_matrix) -> Matrix

Mori-Tanaka scheme: the average matrix-strain is taken as the reference
"infinite-matrix" strain, hence each inclusion experiences a perturbed
strain proportional to the inverse of the volume-weighted concentration
average.

  C̃_eff = C̃^0 + (Σ_r f_r Ñ^{r,dil}) ∘ (f_0 · 𝟙 + Σ_s f_s Ã^{s,dil})^{-vol}

`A_duts[r]` is the dilute concentration kernel of phase `r`,
`contribs[r] = ΔC^r ∘ A_duts[r]`, `fractions[r]` its volume fraction
in the RVE, and `f_matrix` the matrix volume fraction `f_0`.
"""
function mori_tanaka_alv(
        C_0::AbstractMatrix,
        A_duts::AbstractVector{<:AbstractMatrix},
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real},
        f_matrix::Real
    )
    length(A_duts) == length(contribs) == length(fractions) ||
        throw(ArgumentError("mori_tanaka_alv: phase counts mismatch"))
    sz = size(C_0, 1)
    @assert sz % 6 == 0
    n = sz ÷ 6
    T = promote_type(
        eltype(C_0),
        (isempty(A_duts) ? Float64 : eltype(A_duts[1])),
        eltype(fractions), typeof(f_matrix)
    )
    Id = _identity_alv(n, T)
    # Numerator: Σ_r f_r Ñ^{r,dil}
    num = zeros(T, sz, sz)
    # Denominator: f_0 · 𝟙 + Σ_s f_s Ã^{s,dil}
    den = T(f_matrix) .* Id
    @inbounds for r in eachindex(A_duts)
        @. num += fractions[r] * contribs[r]
        @. den += fractions[r] * A_duts[r]
    end
    factor = num * volterra_inverse(den; block_size = 6)
    return T.(C_0) + factor
end

# ── Maxwell scheme ──────────────────────────────────────────────────────────

"""
    maxwell_alv(C_0, contribs, fractions; H_0) -> Matrix

Maxwell scheme: the reference is replaced by a "host medium" with a
prescribed distribution shape, whose Hill kernel is `H_0 = P̃_d`.
The Volterra-discrete formula is

  C̃_eff = C̃^0 + Σ̃ ∘ (𝟙 - P̃_d ∘ Σ̃)^{-vol}

where `Σ̃ = Σ_r f_r Ñ^{r,dil}` and the dilute contributions are the
ones computed by [`dilute_contribution_alv`](@ref).

When the distribution shape coincides with the inclusion shape and
the matrix is the reference, Maxwell reduces to Mori-Tanaka.

Pass `H_0 = hill_kernel(distribution_shape, C_0_law, times)` for the
default uniform-sphere distribution.
"""
function maxwell_alv(
        C_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real};
        H_0::AbstractMatrix
    )
    length(contribs) == length(fractions) ||
        throw(ArgumentError("maxwell_alv: phase counts mismatch"))
    sz = size(C_0, 1)
    @assert sz % 6 == 0
    n = sz ÷ 6
    T = promote_type(
        eltype(C_0),
        (isempty(contribs) ? Float64 : eltype(contribs[1])),
        eltype(fractions), eltype(H_0)
    )
    Id = _identity_alv(n, T)
    Σ = zeros(T, sz, sz)
    @inbounds for r in eachindex(contribs)
        @. Σ += fractions[r] * contribs[r]
    end
    factor = Σ * volterra_inverse(Id - H_0 * Σ; block_size = 6)
    return T.(C_0) + factor
end

# ── DiluteDual: compliance-side dilute scheme ───────────────────────────────

"""
    dilute_dual_alv(C_0, contribs, fractions) -> Matrix

DiluteDual scheme: the dilute contribution is computed in compliance
space.  Equivalent in form to [`dilute_alv`](@ref) but with the matrix
acting on the compliance kernel.  Useful when the inclusions are weak
(e.g. cracks) — the linearization around the matrix compliance
converges better.

  J̃_eff = J̃^0 + Σ_r f_r · H̃^{r,dil}
  C̃_eff = (J̃_eff)^{-vol}

with `H̃^{r,dil} = (J^r - J^0) ∘ B̃^{r,dil}` and
`B̃^{r,dil} = (𝟙 + (𝟙 - Ã^{r,dil}))^{-vol}` … see Sevostianov 2008
for the dual formulation in the elastic case.

Implementation routes through `volterra_inverse` of `C_0` to obtain
`J^0` and back.
"""
function dilute_dual_alv(
        C_0::AbstractMatrix,
        contribs_compliance::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    length(contribs_compliance) == length(fractions) ||
        throw(ArgumentError("dilute_dual_alv: phase counts mismatch"))
    J_0 = volterra_inverse(C_0; block_size = 6)
    J = dilute_alv(J_0, contribs_compliance, fractions)
    return volterra_inverse(J; block_size = 6)
end
