# =============================================================================
#  pair_multipole.jl — interaction tensor of two general ellipsoids by a
#  truncated multipole expansion (brisard2014, §4.2).
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
#  With D(u) = d²/dt² 𝔾⁰(r + t u)|_{t=0} = u_p u_q ∂_p∂_q 𝔾⁰ the second
#  derivative along u, and M² = Σ_k λ_k v_k ⊗ v_k its spectral decomposition
#  (M² is symmetric positive semi-definite),
#
#       M²_pq ∂_p∂_q 𝔾⁰(r) = Σ_k λ_k D(v_k) ,
#
#  d second directional derivatives, each one nested-dual evaluation of the
#  whole kernel: exact (no finite differences), type-generic, and d
#  evaluations instead of d² Hessians.
#
#  When M² carries derivatives (with respect to the semi-axes or the
#  orientation), `eigen` cannot follow them, and at a double eigenvalue the
#  eigenvectors have none to give. The frame is then taken on the value of M²
#  alone, and M² is written in it, N = Vᵀ M² V: N is diagonal in value, and its
#  entries carry all the derivatives. The diagonal enters as λ_k above; an
#  off-diagonal entry, zero in value, enters by the polarization identity
#  D(v_k + v_l) − D(v_k) − D(v_l) = 2 v_k·∇∇𝔾⁰·v_l, and only when it does carry
#  a derivative — a moment turned with an angle — so that a derivative with
#  respect to a semi-axis still costs d evaluations.
# =============================================================================

"""
    _pair_multipole(incl_a, incl_b, r, P₀; order=2, kw...) -> AbstractTens

Interaction tensor between two general ellipsoids by the truncated multipole
expansion of [brisard2014](@citet), §4.2.

`order = 0` keeps the leading (point-dipole) term ``V_b\\,\\mathbb{G}^0(\\underline{r})``; `order = 2`
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

# M²_pq ∂_p∂_q 𝔾⁰ = Σ_k λ_k D(v_k), spectral form (header of this file).
function _moment_contraction(P₀, r::AbstractVector, M²::AbstractMatrix{<:AbstractFloat}; kw...)
    λ, V = eigen(Symmetric(M²))
    D = [_second_directional(P₀, r, @view(V[:, k]); kw...) for k in eachindex(λ)]
    return _spectral_sum(D, λ)
end

# M² carrying derivatives: the frame on its value, every derivative in
# N = Vᵀ M² V (header of this file).
function _moment_contraction(P₀, r::AbstractVector, M²::AbstractMatrix; kw...)
    λ₀, V = eigen(Symmetric(MFH_Core._primal.(M²)))
    N = V' * M² * V
    # What is left of an entry of N once its value is taken off: its
    # derivatives, at every level of nesting. The diagonal keeps λ₀ for its
    # value, so that the value of a differentiated evaluation is the
    # evaluation itself; the off-diagonal is zero in value.
    δ(x) = x - MFH_Core._primal(x)
    λ = [λ₀[k] + δ(N[k, k]) for k in eachindex(λ₀)]
    D = [_second_directional(P₀, r, @view(V[:, k]); kw...) for k in eachindex(λ₀)]
    acc = _spectral_sum(D, λ)
    for k in eachindex(λ₀), l in (k + 1):lastindex(λ₀)
        δN = δ(N[k, l])
        iszero(δN) && continue
        Dkl = _second_directional(P₀, r, @view(V[:, k]) .+ @view(V[:, l]); kw...)
        acc = acc + (Dkl - D[k] - D[l]) * δN
    end
    return acc
end

# Σ_k D_k λ_k, in the order of the eigenvalues.
function _spectral_sum(D, λ)
    acc = D[1] * λ[1]
    for k in 2:length(λ)
        acc = acc + D[k] * λ[k]
    end
    return acc
end

# Exact second directional derivative through one nested-dual evaluation.
function _second_directional(P₀, r::AbstractVector, v; kw...)
    f = t -> MFH_Core._green_operator(P₀, r .+ t .* v; kw...)
    df = t -> ForwardDiff.derivative(f, t)
    return ForwardDiff.derivative(df, zero(eltype(r)))
end

"""
    _second_moment(incl) -> Matrix

Normalized second moment ``M^2_{pq} = \\frac{1}{V}\\int_\\Omega y_p\\,y_q\\,\\mathrm{d}V`` of an inclusion about
its centroid, expressed in the global frame. For an ellipsoid of semi-axes
``(a_1, \\dots, a_d)`` it is ``\\mathbf{Q}\\,\\mathrm{diag}(a_1^2, \\dots, a_d^2)\\,\\mathbf{Q}^{T}/(d + 2)``,
with ``\\mathbf{Q}`` the matrix whose columns are the principal axes.
"""
function _second_moment(ell::Ellipsoid{dim}) where {dim}
    Q = MFH_Core._frame_matrix(ell.basis)
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
