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

Dilute strain concentration kernel
``\\widetilde{\\mathbb{A}}^{\\mathrm{dil}} = (H\\,\\mathbb{I} + \\widetilde{\\mathbb{P}}\\circ\\Delta\\widetilde{\\mathbb{C}})^{-\\circ}``,
``\\Delta\\widetilde{\\mathbb{C}} = \\widetilde{\\mathbb{C}}^{\\mathcal{E}} - \\widetilde{\\mathbb{C}}_0``
[barthelemyIJES2019; eq. 16](@cite).  All inputs are ``6n\\times 6n`` block
matrices ; the result is also ``6n\\times 6n`` and lower-block-triangular.
"""
function dilute_concentration_alv(
        C_E::AbstractMatrix, C_0::AbstractMatrix,
        P::AbstractMatrix
    )
    _alg_check_dense("dilute_concentration_alv", _DenseALV{6}(), C_E, C_0, P)
    return _alg_dilute_concentration(_DenseALV{6}(), C_E, C_0, P)
end

"""
    dilute_contribution_alv(C_E, C_0, P) -> Matrix

Dilute strain contribution kernel ``\\widetilde{\\mathbb{N}} = \\Delta\\widetilde{\\mathbb{C}}\\circ\\widetilde{\\mathbb{A}}^{\\mathrm{dil}}``
[barthelemyIJES2019; eq. 17](@cite).  This is the size-independent stiffness
contribution of a single inclusion.
"""
function dilute_contribution_alv(
        C_E::AbstractMatrix, C_0::AbstractMatrix,
        P::AbstractMatrix
    )
    _alg_check_dense("dilute_contribution_alv", _DenseALV{6}(), C_E, C_0, P)
    return _alg_dilute_contribution(_DenseALV{6}(), C_E, C_0, P)
end

# ── Voigt / Reuss bounds ────────────────────────────────────────────────────

"""
    voigt_alv(C_phases::AbstractVector, fractions::AbstractVector) -> Matrix

Voigt (uniform-strain) bound: ``\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\sum_i f_i\\,\\widetilde{\\mathbb{C}}_i``.  Each `C_phases[i]`
is a ``6n\\times 6n`` block matrix and `fractions[i]` is the volume fraction ``f_i``
of phase ``i``.
"""
function voigt_alv(
        C_phases::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    _alg_check("voigt_alv", C_phases, fractions)
    isempty(C_phases) && throw(ArgumentError("voigt_alv: at least one phase required"))
    return _alg_voigt(_DenseALV{6}(), C_phases, fractions)
end

"""
    reuss_alv(C_phases::AbstractVector, fractions::AbstractVector) -> Matrix

Reuss (uniform-stress) bound: invert each ``\\widetilde{\\mathbb{C}}_i``, do the volume average
of compliances, then invert the result.

```math
\\widetilde{\\mathbb{L}}^{\\mathrm{hom}} = \\sum_i f_i\\,\\widetilde{\\mathbb{L}}_i,
\\qquad
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\bigl(\\widetilde{\\mathbb{L}}^{\\mathrm{hom}}\\bigr)^{-\\circ}
```
"""
function reuss_alv(
        C_phases::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    _alg_check("reuss_alv", C_phases, fractions)
    isempty(C_phases) && throw(ArgumentError("reuss_alv: at least one phase required"))
    return _alg_reuss(_DenseALV{6}(), C_phases, fractions)
end

# ── Dilute scheme ───────────────────────────────────────────────────────────

"""
    dilute_alv(C_0, contribs, fractions) -> Matrix

Dilute scheme: ``\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0 + \\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i`` where the dilute
contributions ``\\widetilde{\\mathbb{N}}_i`` of each inclusion phase are pre-computed via
`dilute_contribution_alv` (each ``6n\\times 6n``), ``f_i`` is the volume
fraction of phase ``i``, and the matrix ``\\widetilde{\\mathbb{C}}_0`` is the reference.

Note that for the dilute scheme the matrix volume fraction ``f_0``
does *not* appear: the inclusions are treated as if they were
embedded in an infinite matrix.
"""
function dilute_alv(
        C_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    _alg_check("dilute_alv", contribs, fractions)
    return _alg_dilute(_DenseALV{6}(), C_0, contribs, fractions)
end

# ── Mori-Tanaka scheme ──────────────────────────────────────────────────────

"""
    mori_tanaka_alv(C_0, A_duts, contribs, fractions, f_matrix) -> Matrix

Mori-Tanaka scheme: the average matrix-strain is taken as the reference
"infinite-matrix" strain, hence each inclusion experiences a perturbed
strain proportional to the inverse of the volume-weighted concentration
average.

```math
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0
  + \\Bigl(\\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i\\Bigr)\\circ
    \\Bigl(f_0\\,H\\,\\mathbb{I} + \\sum_j f_j\\,\\widetilde{\\mathbb{A}}_j^{\\mathrm{dil}}\\Bigr)^{-\\circ}
```

`A_duts[i]` is the dilute concentration kernel ``\\widetilde{\\mathbb{A}}_i^{\\mathrm{dil}}`` of phase ``i``,
`contribs[i]` ``= \\widetilde{\\mathbb{N}}_i = (\\widetilde{\\mathbb{C}}_i - \\widetilde{\\mathbb{C}}_0)\\circ\\widetilde{\\mathbb{A}}_i^{\\mathrm{dil}}``,
`fractions[i]` its volume fraction ``f_i`` in the RVE, and `f_matrix` the matrix
volume fraction ``f_0``.
"""
function mori_tanaka_alv(
        C_0::AbstractMatrix,
        A_duts::AbstractVector{<:AbstractMatrix},
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real},
        f_matrix::Real
    )
    _alg_check("mori_tanaka_alv", A_duts, contribs, fractions)
    return _alg_mori_tanaka(_DenseALV{6}(), C_0, A_duts, contribs, fractions, f_matrix)
end

# ── Maxwell scheme ──────────────────────────────────────────────────────────

"""
    maxwell_alv(C_0, contribs, fractions; H_0) -> Matrix

Maxwell scheme: the reference is replaced by a "host medium" with a
prescribed distribution shape, whose Hill kernel is `H_0` ``= \\widetilde{\\mathbb{P}}_{\\Omega}``.
The Volterra-discrete formula is

```math
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0
  + \\widetilde{\\mathbb{N}}\\circ\\bigl(H\\,\\mathbb{I} - \\widetilde{\\mathbb{P}}_{\\Omega}\\circ\\widetilde{\\mathbb{N}}\\bigr)^{-\\circ}
```

where ``\\widetilde{\\mathbb{N}} = \\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i`` and the dilute contributions are the
ones computed by [`dilute_contribution_alv`](@ref).

When every inclusion phase has the shape of the distribution and the matrix is
the reference, Maxwell reduces to Mori-Tanaka.

Pass `H_0 = hill_kernel(distribution_shape, C_0_law, times)`, the distribution
shape being the one declared on the RVE; there is no default.
"""
function maxwell_alv(
        C_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real};
        H_0::AbstractMatrix
    )
    _alg_check("maxwell_alv", contribs, fractions)
    return _alg_maxwell(_DenseALV{6}(), C_0, contribs, fractions, H_0)
end

# ── DiluteDual: compliance-side dilute scheme ───────────────────────────────

"""
    dilute_dual_alv(C_0, contribs, fractions) -> Matrix

DiluteDual scheme: the dilute contribution is computed in compliance
space.  Equivalent in form to [`dilute_alv`](@ref) but with the matrix
acting on the compliance kernel.  Useful when the inclusions are weak
(e.g. cracks) — the linearization around the matrix compliance
converges better.

```math
\\widetilde{\\mathbb{L}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{L}}_0 + \\sum_i f_i\\,\\widetilde{\\mathbb{H}}_i,
\\qquad
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\bigl(\\widetilde{\\mathbb{L}}^{\\mathrm{hom}}\\bigr)^{-\\circ}
```

with the compliance contribution kernels
``\\widetilde{\\mathbb{H}}_i = -\\,\\widetilde{\\mathbb{L}}_0\\circ\\widetilde{\\mathbb{N}}_i\\circ\\widetilde{\\mathbb{L}}_0``,
the Volterra counterpart of the elastic identity
``\\mathbb{H}_i = -\\,\\mathbb{S}_0:\\mathbb{N}_i:\\mathbb{S}_0`` [barthelemyIJES2019](@cite);
`contribs_compliance` holds them.

Implementation routes through `volterra_inverse` of `C_0` to obtain
``\\widetilde{\\mathbb{L}}_0`` and back.
"""
function dilute_dual_alv(
        C_0::AbstractMatrix,
        contribs_compliance::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real}
    )
    _alg_check("dilute_dual_alv", contribs_compliance, fractions)
    return _alg_dilute_dual(_DenseALV{6}(), C_0, contribs_compliance, fractions)
end
