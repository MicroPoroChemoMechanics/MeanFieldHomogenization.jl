# =============================================================================
#  conductivity.jl — mean gradients of the layers of a confocal ellipsoid.
#
#  With φₖ the volume fraction of the layer k in the inclusion, ω_J = Σ_{K≤J} φ_K,
#  ℙ_M(Ω) the Hill tensor of the ellipsoid Ω in the isotropic medium of
#  conductivity k_M and ḡᴷ the mean gradient of the layer K:
#
#    ḡᴶ⁺¹ = (1/ω_J) Σ_{K≤J} φ_K [𝟏 − (k_{J+1} − k_K) 𝕋_{J+1}]·ḡᴷ ,      J < N,
#    𝕋_{J+1} = ℙ_{J+1}(Ω_{J+1}) − (ω_{J+1}/φ_{J+1}) (ℙ_{J+1}(Ω_{J+1}) − ℙ_{J+1}(Ω_J)),
#    Σ_K φ_K [𝟏 + ℙ₀(Ω_N)·(k_K 𝟏 − 𝐊₀)]·ḡᴷ = 𝐠⁰ .
#
#  The first relations give every ḡᴷ from that of the core, ḡᴷ = 𝐫_K·ḡ¹; the
#  last one then gives ḡ¹. They are exact for confocal boundaries: the exterior
#  potentials of confocal homogeneous ellipsoids are in the ratio of their
#  volumes (theory page "Layered ellipsoid"). All the tensors share the axes
#  of the inclusion, and are handled by their matrices in the canonical basis.
# =============================================================================

_matrix(t::TensND.AbstractTens{2, 3}) = SMatrix{3, 3}(TensND.canonical_components(t))
_scalar(K::TensND.TensISO{2, 3}) = first(TensND.get_data(K))

# Hill tensor of the boundary of the layers 1 to k in the isotropic medium of
# conductivity 1; that of the medium of conductivity κ is this one over κ.
# The type of an `Ellipsoid` depends on the values of its semi-axes (sphere,
# spheroid, triaxial); that of its Hill tensor does not, and is asserted here
# so that what follows is inferred.
_unit_hill(e::LayeredEllipsoid{T}, k::Int) where {T} =
    _matrix(Elasticity.hill_tensor(_boundary(e, k), TensND.TensISO{3}(one(T))))::SMatrix{3, 3, T, 9}

"""
    _mean_gradients(e::LayeredEllipsoid, K₀) -> (gradients, fractions)

The mean gradient of each layer per unit remote gradient, as matrices in the
canonical basis, and the volume fractions of the layers in the inclusion.
"""
function _mean_gradients(e::LayeredEllipsoid{T, N}, K₀::TensND.TensISO{2, 3}) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    k = map(_scalar, e.moduli)
    k₀ = _scalar(K₀)
    ω = _cumulated_fractions(e)
    φ = ntuple(J -> J == 1 ? ω[1] : ω[J] - ω[J - 1], Val(N))
    P = ntuple(J -> _unit_hill(e, J), Val(N))
    S = promote_type(T, map(typeof, k)..., typeof(k₀))
    𝟏 = SMatrix{3, 3, S}(I)
    # A scalar is broadcast on the matrices: `scalar * SMatrix` is a symbolic
    # array, not a static matrix, when the scalar is a symbol.
    # 𝐫_K, from the core outward; the sum Σ_{K≤J} φ_K [𝟏 − (k_{J+1} − k_K) 𝕋_{J+1}]·𝐫_K is rebuilt
    # for each J, the tensor 𝕋 changing with it.
    r = Vector{typeof(𝟏)}(undef, N)
    r[1] = 𝟏
    for J in 1:(N - 1)
        𝕋 = (P[J + 1] .- (ω[J + 1] / φ[J + 1]) .* (P[J + 1] .- P[J])) ./ k[J + 1]
        r[J + 1] = sum(φ[K] .* ((𝟏 .- (k[J + 1] - k[K]) .* 𝕋) * r[K]) for K in 1:J) ./ ω[J]
    end
    core = inv(sum(φ[K] .* ((𝟏 .+ (k[K] / k₀ - 1) .* P[N]) * r[K]) for K in 1:N))
    return ntuple(K -> r[K] * core, Val(N)), φ
end

_whole_gradient(g, φ) = sum(map((f, x) -> f .* x, φ, g))
_whole_flux(e::LayeredEllipsoid, g, φ) = sum(map((f, K, x) -> (f * _scalar(K)) .* x, φ, e.moduli, g))
# The tensors are symmetric, the ellipsoids sharing their axes: the symmetric
# storage, which also makes the type of the result independent of the values.
_tensor(M::SMatrix{3, 3}) = TensND.Tens(Tensors.symmetric(Tensors.Tensor{2, 3}(Tuple(M))))

"""
    gradient_gradient_concentration(e::LayeredEllipsoid, K₁, K₀::TensISO{2,3}) -> AbstractTens{2,3}
    gradient_gradient_concentration(e::LayeredEllipsoid, K₀::TensISO{2,3}; layer) -> AbstractTens{2,3}

The mean gradient of the whole inclusion, or of its layer `layer`, per unit
remote gradient in the isotropic medium `K₀`. `K₁` is not read: the layers
carry their conductivities. The tensor has the axes of the ellipsoid.
"""
function gradient_gradient_concentration(
        e::LayeredEllipsoid, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; kw...
    )
    return _tensor(_whole_gradient(_mean_gradients(e, K₀)...))
end
function gradient_gradient_concentration(e::LayeredEllipsoid, K₀::TensND.TensISO{2, 3}; layer::Int, kw...)
    1 ≤ layer ≤ layer_count(e) ||
        throw(ArgumentError("LayeredEllipsoid: the layer $layer does not exist; the inclusion has $(layer_count(e)) layers"))
    return _tensor(first(_mean_gradients(e, K₀))[layer])
end

"""
    flux_gradient_concentration(e::LayeredEllipsoid, K₁, K₀::TensISO{2,3}) -> AbstractTens{2,3}

The mean of ``\\boldsymbol{K}\\cdot\\underline{\\operatorname{grad}}\\,T`` over the
inclusion per unit remote gradient.
"""
function flux_gradient_concentration(
        e::LayeredEllipsoid, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; kw...
    )
    g, φ = _mean_gradients(e, K₀)
    return _tensor(_whole_flux(e, g, φ))
end

"""
    property_contribution(e::LayeredEllipsoid, K₁, K₀::TensISO{2,3}) -> AbstractTens{2,3}

The conductivity contribution tensor of the inclusion, the mean flux minus
``\\boldsymbol{K}_0`` times the mean gradient, per unit remote gradient.
"""
function property_contribution(
        e::LayeredEllipsoid, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; kw...
    )
    g, φ = _mean_gradients(e, K₀)
    return _tensor(_whole_flux(e, g, φ) .- _scalar(K₀) .* _whole_gradient(g, φ))
end

# One solution for the two tensors a scheme asks together.
function MFH_Core.concentration_and_contribution(
        e::LayeredEllipsoid, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; kw...
    )
    g, φ = _mean_gradients(e, K₀)
    A = _whole_gradient(g, φ)
    return (_tensor(A), _tensor(_whole_flux(e, g, φ) .- _scalar(K₀) .* A))
end
function MFH_Core.concentration_and_flux_average(
        e::LayeredEllipsoid, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; kw...
    )
    g, φ = _mean_gradients(e, K₀)
    return (_tensor(_whole_gradient(g, φ)), _tensor(_whole_flux(e, g, φ)))
end

# The layers being isotropic, the relations between them hold whatever the
# medium; the solution has been checked against an independent one in isotropic
# media only, hence the restriction.
_isotropic_medium_only(K₀) = throw(
    ArgumentError(
        "LayeredEllipsoid: the surrounding medium must be isotropic (`TensISO{3}(k)`), got a $(nameof(typeof(K₀))). " *
            "A scheme whose reference becomes anisotropic (self-consistent with aligned ellipsoids) does not apply"
    )
)
for f in (:gradient_gradient_concentration, :flux_gradient_concentration, :property_contribution)
    @eval $f(::LayeredEllipsoid, ::TensND.AbstractTens{2, 3}, K₀::TensND.AbstractTens{2, 3}; kw...) =
        _isotropic_medium_only(K₀)
end

# ── Layer averages of the conductivities (Voigt and Reuss bounds) ────────────

_average(e::LayeredEllipsoid{T, N}, f::F) where {T, N, F} =
    TensND.TensISO{3}(sum(ntuple(k -> layer_volume_fraction(e, k) * f(_scalar(e.moduli[k])), Val(N))))
layer_property_average(e::LayeredEllipsoid, ::Union{Val{2}, TensND.AbstractTens{2, 3}}) = _average(e, identity)
layer_inverse_property_average(e::LayeredEllipsoid, ::Union{Val{2}, TensND.AbstractTens{2, 3}}) = _average(e, inv)
