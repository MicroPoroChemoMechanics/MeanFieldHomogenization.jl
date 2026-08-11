# =============================================================================
#  pair_multipole.jl — interaction tensor of two general ellipsoids by a
#  truncated multipole expansion (Brisard, Dormieux & Sab 2014, §4.2).
#
#  For a region Ω of volume V, centroid at the origin and normalized second
#  moment
#
#       M²_pq = (1/V) ∫_Ω y_p y_q dV ,
#
#  the average of a smooth field over Ω is
#
#       ⟨f⟩_Ω = f + ½ M²_pq ∂_p∂_q f + O(4th moments) .
#
#  Applying it to the source region and to the receiver region gives
#
#       𝕋^{ab} = V_b [ 𝔾⁰(r) + ½ (M²_a + M²_b)_pq ∂_p∂_q 𝔾⁰(r) + … ] ,   (∗)
#
#  which is the expansion of Brisard et al. in the equivalent "moment" form.
#  For an ellipsoid of semi-axes (a₁, …, a_d) the second moment is diagonal in
#  the principal frame, M² = diag(a₁², …, a_d²)/(d+2).
#
#  Balls are the special case M² = (a²/(d+2)) 𝟏, for which (∗) reduces to the
#  Laplacian form of `pair_ball_iso.jl` — and there the series TERMINATES
#  because the elastic Green function is biharmonic.  For a general ellipsoid
#  it does not: (∗) is genuinely asymptotic in (size / separation), exactly as
#  Brisard et al. state, and `order = 2` is the truncation implemented here.
#
#  The contraction M²_pq ∂_p∂_q is not evaluated by building a rank-6 array.
#  M² is symmetric positive definite, so its spectral decomposition
#  M² = Σ_k λ_k v_k ⊗ v_k turns the contraction into d second *directional*
#  derivatives,
#
#       M²_pq ∂_p∂_q 𝔾⁰(r) = Σ_k λ_k d²/dt² 𝔾⁰(r + t v_k) |_{t=0} ,
#
#  each of which is one nested-dual evaluation of the whole kernel.  This is
#  exact (no finite differences), type-generic, and costs d evaluations
#  instead of d² Hessians.
# =============================================================================

"""
    _pair_multipole(incl_a, incl_b, r, P₀; order=2, kw...) -> AbstractTens

Interaction tensor between two general ellipsoids by the truncated multipole
expansion of [Brisard et al. 2014](@cite brisard2014), §4.2.

`order = 0` keeps the leading (point-dipole) term `V_b 𝔾⁰(r)`; `order = 2`
adds the second-moment correction, which is the first non-vanishing one
because the first moments vanish about the centroids. Accuracy degrades as
the inclusions approach each other — the expansion parameter is the ratio of
inclusion size to center distance.
"""
function _pair_multipole(
        incl_a, incl_b, r::AbstractVector, P₀::TensND.AbstractTens;
        order::Int = 2, kw...
    )
    order in (0, 2) || throw(
        ArgumentError(
            "interaction_tensor: the multipole expansion is implemented at " *
                "`order = 0` (point dipole) and `order = 2` (second moment); got $(order)."
        )
    )
    V_b = _inclusion_volume(incl_b)
    G = MFH_Core._green_operator(P₀, r; kw...)
    order == 0 && return TensND.Tens(V_b * G)
    M² = _second_moment(incl_a) + _second_moment(incl_b)
    return TensND.Tens(V_b * (G + _moment_contraction(P₀, r, M²; kw...) / 2))
end

# Σ_k λ_k d²/dt² 𝔾⁰(r + t v_k) — spectral form of M²_pq ∂_p∂_q 𝔾⁰.
function _moment_contraction(P₀, r::AbstractVector, M²::AbstractMatrix; kw...)
    λ, V = eigen(Symmetric(_as_float_matrix(M²)))
    d = length(r)
    acc = _second_directional(P₀, r, @view(V[:, 1]); kw...) * λ[1]
    for k in 2:d
        acc = acc + _second_directional(P₀, r, @view(V[:, k]); kw...) * λ[k]
    end
    return acc
end

# Exact second directional derivative through one nested-dual evaluation.
function _second_directional(P₀, r::AbstractVector, v; kw...)
    f = t -> MFH_Core._green_operator(P₀, r .+ t .* v; kw...)
    df = t -> ForwardDiff.derivative(f, t)
    return ForwardDiff.derivative(df, zero(eltype(r)))
end

# `eigen` needs a concrete floating-point matrix; the eigenvectors of the
# second-moment tensor are a *geometric* datum (the inclusion axes) and carry
# no sensitivity of their own, so taking their values is legitimate here —
# the differentiable dependence stays in `λ` and in the kernel evaluation.
_as_float_matrix(M::AbstractMatrix{T}) where {T} = Float64.(ForwardDiff.value.(M))
_as_float_matrix(M::AbstractMatrix{Float64}) = M

"""
    _second_moment(incl) -> Matrix

Normalized second moment `M²_pq = (1/V) ∫_Ω y_p y_q dV` of an inclusion about
its centroid, expressed in the global frame. For an ellipsoid of semi-axes
`(a₁, …, a_d)` it is `Q diag(a₁², …, a_d²) Qᵀ / (d + 2)`, with `Q` the matrix
whose columns are the principal axes.
"""
function _second_moment(ell::Ellipsoid{dim}) where {dim}
    Q = MFH_Core._basis_matrix(ell.basis)
    a = ell.semi_axes
    D = [i == j ? a[i]^2 / (dim + 2) : zero(a[1]) for i in 1:dim, j in 1:dim]
    return Q * D * Q'
end

"""
    _inclusion_volume(incl) -> Real

Volume (3D) or area (2D) of an inclusion — the factor that turns a
polarization density into a polarization moment.
"""
_inclusion_volume(ell::Ellipsoid{3}) =
    4 * π * ell.semi_axes[1] * ell.semi_axes[2] * ell.semi_axes[3] / 3
_inclusion_volume(ell::Ellipsoid{2}) = π * ell.semi_axes[1] * ell.semi_axes[2]

_inclusion_volume(incl::MFH_Core.AbstractInclusion) = throw(
    ArgumentError(
        "interaction_tensor: no volume is defined for a $(nameof(typeof(incl))). " *
            "The interaction kernels need a bounded region of known measure."
    )
)
