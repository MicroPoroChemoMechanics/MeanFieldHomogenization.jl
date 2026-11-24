# =============================================================================
#  radial_layers.jl — layers transversely isotropic about the radial direction.
#
#  A layer may be a field of TensND, `ChartTens`, transversely isotropic about
#  the unit radial vector e_r of the spherical chart at every point
#  (`radial_ti(SphericalChart(), …)`): a stiffness of five moduli or a
#  conductivity of two. Such a layer keeps the spherical symmetry of the
#  composite sphere, so that the remote strain still splits into a spherical
#  and a deviatoric part. Each part, and the conduction problem, is a system of
#  the first order in the radius, r dY/dr = M Y, on the state Y that an
#  interface transmits (displacement and traction, temperature and flux), with
#  M a function of the moduli at that radius: Hill's moduli (n, l, k, m, p)
#  about e_r, or the two conductivities. The exponents of the radial modes
#  (which are no longer integers, and may be complex in the deviatoric part)
#  are the eigenvalues of M; the layers are crossed by the power of M, without
#  them. The systems are derived symbolically in the tutorial "Layered sphere:
#  layers transversely isotropic about the radius", and stated in its theory
#  page.
#
#  The states (displacement and traction, or temperature and flux) are carried
#  from the core to the matrix through the interface matrices of
#  `interface_transfer.jl`, as for isotropic layers. What a scheme needs is then
#  read on the states at the boundaries of the layers, by the divergence
#  theorem: the mean strain of a region is the moment of the displacement on its
#  boundary, its mean stress that of the traction. No amplitude is extracted, so
#  that nothing here depends on the law of a layer.
#
#  A `LayeredSphere` one of whose moduli is such a field holds such fields only
#  (its constructor converts the isotropic ones, which are particular cases):
#  the methods of this file are selected by that type.
# =============================================================================

# A layer transversely isotropic about the radius: a field of the spherical
# chart whose pattern is transversely isotropic about the radial vector of the
# local frame (`TensND.radial_ti(SphericalChart(), …)`).
const RadialField{order} = MFH_Core.RadialField{order, TensND.SphericalChart}

_radial_field(c::RadialField) = MFH_Core.is_radial(c) ? c : _not_radial_field(c)
_radial_field(c::TensND.TensISO) = TensND.radial_ti(TensND.SphericalChart(), c)
_radial_field(c) = _not_radial_field(c)
_not_radial_field(c) = throw(
    ArgumentError(
        "a layer of a sphere with radial layers is isotropic or transversely isotropic about the " *
            "radius (`radial_ti(SphericalChart(), …)`); got $(c). A tensor of fixed axes, or a field " *
            "of another axis or of another chart, does not have the symmetry of the sphere"
    )
)

# Hill's moduli about e_r, n = C_rrrr, l = C_rrθθ, k = (C_θθθθ + C_θθφφ)/2,
# m = (C_θθθθ − C_θθφφ)/2, p = C_rθrθ, and the conductivities across the radius
# and along it.
_hill_moduli(c::RadialField{4}) = MFH_Core.radial_hill_moduli(c)
_conductivities(c::RadialField{2}) = MFH_Core.radial_coefficients(c)

# The mean over all directions of the tensor of the layer: an isotropic tensor,
# which is its volume average over a spherical shell.
_shell_average(c::RadialField{4}) = TensND.best_fit_iso(c.pattern)
_shell_average(c::RadialField{2}) = ((kₜ, kᵣ) = _conductivities(c); TensISO{3}((2 * kₜ + kᵣ) / 3))

# A layer of a sphere solved through its states: a uniform radial field, or
# moduli that vary with the radius.
const _RadialLayer = Union{RadialField, MFH_Core.GradedLayer, MFH_Core.PowerLawLayer}
const RadialLayers = Tuple{_RadialLayer, Vararg{_RadialLayer}}

# The modulus of a layer that varies with the radius, at a radius, as a radial field.
_modulus_at(g::MFH_Core.GradedLayer, r) = _radial_field(g.profile(r))
_modulus_at(c::MFH_Core.PowerLawLayer, r) = _radial_field(c(r))
const RadialLayeredSphere{T, N} = LayeredSphere{T, N, <:RadialLayers}

# ── The radial problems as first-order systems ───────────────────────────────
#
# With the state Y made of the radial functions of the displacement and of r
# times those of the traction (of the temperature and of r times the outward
# flux), each radial problem reads
#
#     r dY/dr = M Y,
#
# M depending on the moduli at the radius r alone: no derivative of a modulus
# enters, the state being what an interface transmits. The matrices below are
# derived symbolically in the tutorial `scripts/101`. In a layer of uniform
# moduli M is constant, and the state is carried from a radius to another by
# the power (r_out/r_in)^M, the solutions regular at the center being a
# subspace of the states: both are in `Core/radial_systems.jl`, shared with the
# layered cylinder.

# Spherical part, Y = (u_r, r σ_rr).
_bulk_system(h) = SMatrix{2, 2}(-2 * h.l / h.n, 4 * h.k - 4 * h.l^2 / h.n, 1 / h.n, 2 * h.l / h.n - 1)

# Deviatoric part, Y = (U, W, r σ_rr, r σ_rθ) for u = U P₂ e_r + W P₂′ e_θ.
function _shear_system(h)
    (; n, l, k, m, p) = h
    c = k - l^2 / n
    z = zero(c)
    return SMatrix{4, 4}(
        -2 * l / n, -one(c), 4 * c, -2 * c,
        6 * l / n, one(c), -12 * c, 6 * c + 4 * m,
        1 / n, z, 2 * l / n - 1, -l / n,
        z, 1 / p, 6 * one(c), -2 * one(c),
    )
end

# Conduction, Y = (T̂, r q̂) with q̂ = −k_r ∂T̂/∂r the outward flux.
function _cond_system(c::RadialField{2})
    kₜ, kᵣ = _conductivities(c)
    return SMatrix{2, 2}(zero(kₜ), -2 * kₜ, -1 / kᵣ, -one(kₜ))
end

_system(c::RadialField{4}, ::Val{1}) = _bulk_system(_hill_moduli(c))
_system(c::RadialField{4}, ::Val{2}) = _shear_system(_hill_moduli(c))
_system(c::RadialField{2}, ::Val{1}) = _cond_system(c)

# The state the interface matrices act on is (displacement, traction) at the
# radius r; Y carries r times the traction in its second half.
# The argument gives the number of rows of the state only.
_scaling(::SMatrix{2}, r) = SMatrix{2, 2}(one(r), zero(r), zero(r), r)
_scaling(::SMatrix{4}, r) = SMatrix{4, 4}(Diagonal(SVector(one(r), one(r), r, r)))

# ── The states at the boundaries of the layers ───────────────────────────────

# The states (displacement, traction) of the solutions regular at the center,
# at the radius of the core, and the crossing of a layer. A uniform layer is
# crossed by the power of its system; a graded one by the integration of its
# system, started for a core near the center on the regular solutions of the
# medium found there.
function _core_states(c::RadialField, probes::Val, r, ::Type{TP}) where {TP}
    M = map(TP, _system(c, probes))
    return inv(_scaling(M, r)) * MFH_Core.regular_states(M, probes)
end
function _core_states(g::MFH_Core.GradedLayer, probes::Val, r, ::Type{TP}) where {TP}
    system(ρ) = map(TP, _system(_modulus_at(g, ρ), probes))
    Y = MFH_Core.regular_transfer(system, g, r, probes)
    return inv(_scaling(Y, r)) * Y
end

function _cross(c::RadialField, probes::Val, r_in, r_out, S, ::Type{TP}) where {TP}
    M = map(TP, _system(c, probes))
    return inv(_scaling(M, r_out)) * (MFH_Core.radial_power(M, r_out / r_in) * (_scaling(M, r_in) * S))
end
function _core_states(c::MFH_Core.PowerLawLayer, probes::Val, r, ::Type)
    throw(
        ArgumentError(
            "a PowerLawLayer cannot be the core of a LayeredSphere: its modulus, in the power " *
                "$(c.exponent) of the radius, vanishes or is infinite at the center"
        )
    )
end
# A layer in a power of the radius: the power of the system of its reference
# modulus, shifted (`Core.power_law_transfer`).
function _cross(c::MFH_Core.PowerLawLayer, probes::Val, r_in, r_out, S, ::Type{TP}) where {TP}
    M = map(TP, _system(_radial_field(c.reference), probes))
    P = MFH_Core.power_law_transfer(M, c.exponent, r_in / c.at, r_out / c.at)
    return inv(_scaling(M, r_out)) * (P * (_scaling(M, r_in) * S))
end
function _cross(g::MFH_Core.GradedLayer, probes::Val, r_in, r_out, S, ::Type{TP}) where {TP}
    system(ρ) = map(TP, _system(_modulus_at(g, ρ), probes))
    return inv(_scaling(S, r_out)) * MFH_Core.radial_transfer(system, g, r_in, r_out, _scaling(S, r_in) * S)
end

# The states of `nprobes` solutions regular at the center, carried to the
# matrix: for each interface k, the state on the inside (r_k⁻) and on the
# outside (r_k⁺), one column per solution.
function _radial_states(sphere::RadialLayeredSphere{T, N}, interface, probes::Val, ::Type{TP}) where {T, N, TP}
    radii = map(TP, sphere.radii)
    S = _core_states(layer_modulus(sphere, 1), probes, radii[1], TP)
    inside = Vector{typeof(S)}(undef, N)
    outside = Vector{typeof(S)}(undef, N)
    for k in 1:N
        inside[k] = S
        S = typeof(S)(interface(layer_interface(sphere, k), radii[k]) * S)
        outside[k] = S
        k < N && (S = typeof(S)(_cross(layer_modulus(sphere, k + 1), probes, radii[k], radii[k + 1], S, TP)))
    end
    return inside, outside
end

# The number type of a layer: that of its modulus, read for a graded layer at
# its outer radius.
_layer_eltype(c::RadialField, r) = eltype(c)
_layer_eltype(g::MFH_Core.GradedLayer, r) = eltype(_modulus_at(g, r))
_layer_eltype(c::MFH_Core.PowerLawLayer, r) = promote_type(eltype(_modulus_at(c, r)), typeof(c.exponent), typeof(c.at))

_radial_eltype(sphere::RadialLayeredSphere{T}, x₀...) where {T} = TensND.float_like(
    promote_type(
        T, map(typeof, x₀)..., map(_layer_eltype, sphere.moduli, sphere.radii)...,
        interfaces_eltype(sphere.interfaces)
    )
)

# The states per unit remote strain, spherical then deviatoric.
function _radial_elastic_states(sphere::RadialLayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}) where {T, N}
    κ₀, μ₀ = TensND.k_mu(C₀)
    TP = _radial_eltype(sphere, κ₀, μ₀)
    R = TP(sphere.radii[N])
    u = one(TP)
    bulk_in, bulk_out = _radial_states(sphere, (itf, r) -> _bulk_interface_T(itf, u, u, r), Val(1), TP)
    A∞, _ = _bulk_extract_AB(R, TP(κ₀), TP(μ₀), bulk_out[N][1], bulk_out[N][2])
    dev_in, dev_out = _radial_states(sphere, (itf, r) -> _shear_interface_T(itf, u, u, r), Val(2), TP)
    # The combination of the two regular solutions whose remote part is the unit
    # uniform strain: amplitudes (a, b) = (1, 0) in the matrix.
    amplitudes = _shear_M_inverse(R, TP(κ₀), TP(μ₀)) * dev_out[N]
    λ = inv(SMatrix{2, 2}(amplitudes[1, 1], amplitudes[2, 1], amplitudes[1, 2], amplitudes[2, 2])) * SVector(u, zero(u))
    return (
        bulk_in = map(S -> S[:, 1] / A∞, bulk_in), bulk_out = map(S -> S[:, 1] / A∞, bulk_out),
        dev_in = map(S -> S * λ, dev_in), dev_out = map(S -> S * λ, dev_out),
    )
end

# The states per unit remote gradient.
function _radial_conduction_states(sphere::RadialLayeredSphere{T, N}, K₀::TensND.TensISO{2, 3}) where {T, N}
    k₀ = _iso_scalar(K₀)
    TP = _radial_eltype(sphere, k₀)
    inside, outside = _radial_states(sphere, (itf, r) -> _cond_interface_T(itf, one(TP), r), Val(1), TP)
    A∞, _ = _cond_extract_AB(TP(sphere.radii[N]), TP(k₀), outside[N][1], outside[N][2])
    return (inside = map(S -> S[:, 1] / A∞, inside), outside = map(S -> S[:, 1] / A∞, outside))
end

# ── Moments on a sphere of radius r ──────────────────────────────────────────
#
# Divided by the cube of the radius of a ball, the moment of the displacement on
# its boundary is its mean strain and that of the traction its mean stress, per
# unit remote strain, on 𝕁 (spherical state) and on 𝕂 (deviatoric state).
_strain_moment(r, s::SVector{2}) = r^2 * s[1]
_stress_moment(r, s::SVector{2}) = r^3 * s[2]
_strain_moment(r, s::SVector{4}) = r^2 * (s[1] + 3 * s[2]) / 5
_stress_moment(r, s::SVector{4}) = r^3 * (s[3] + 3 * s[4]) / 5

# The difference of a moment between the two boundaries of the layer k, the
# material of the layer alone, divided by the cube that measures its volume.
function _layer_mean(moment, radii, inside, outside, k)
    V = radii[k]^3 - (k == 1 ? zero(radii[k]) : radii[k - 1]^3)
    inner = k == 1 ? zero(moment(radii[k], inside[k])) : moment(radii[k - 1], outside[k - 1])
    return (moment(radii[k], inside[k]) - inner) / V
end

# The state on the boundary of the whole sphere: outside its last interface
# (what that interface carries is counted) or inside it.
_boundary_state(inside, outside, external::Bool) = external ? last(outside) : last(inside)

# ── The moduli of a sphere with radial layers ────────────────────────────────

# One radial layer makes every layer a radial field: an isotropic layer is the
# particular case whose five moduli are those of an isotropic tensor.
_as_layers(moduli::Tuple) = any(c -> c isa Union{TensND.ChartTens, Function, _RadialLayer}, moduli) ? map(_radial_layer, moduli) : moduli
_as_layers(moduli) = moduli
_radial_layer(c) = _radial_field(c)
_radial_layer(g::Union{MFH_Core.GradedLayer, MFH_Core.PowerLawLayer}) = g
_radial_layer(profile::Function) = MFH_Core.GradedLayer(profile)

# ── What the schemes ask of the sphere ───────────────────────────────────────
#
# Whole sphere: the moments on its boundary. `C₁` (or `K₁`), the property
# declared for the phase, is ignored, as for isotropic layers.

function _radial_whole(sphere::RadialLayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}, external::Bool) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    st = _radial_elastic_states(sphere, C₀)
    R = sphere.radii[N]
    bulk = _boundary_state(st.bulk_in, st.bulk_out, external)
    dev = _boundary_state(st.dev_in, st.dev_out, external)
    A = TensISO{3}(_strain_moment(R, bulk) / R^3, _strain_moment(R, dev) / R^3)
    B = TensISO{3}(_stress_moment(R, bulk) / R^3, _stress_moment(R, dev) / R^3)
    return A, B
end

# N = ⟨σ⟩ − C₀ : ⟨ε⟩ over the sphere, on 𝕁 and 𝕂.
_contribution(A::TensISO, B::TensISO, P₀::TensISO) = TensISO{3}(map((b, p, a) -> b - p * a, TensND.get_data(B), TensND.get_data(P₀), TensND.get_data(A))...)

gradient_gradient_concentration(
    sphere::RadialLayeredSphere, ::TensND.AbstractTens{4, 3}, C₀::TensND.TensISO{4, 3}; external::Bool = true, kw...
) = first(_radial_whole(sphere, C₀, external))

flux_gradient_concentration(
    sphere::RadialLayeredSphere, ::TensND.AbstractTens{4, 3}, C₀::TensND.TensISO{4, 3}; external::Bool = true, kw...
) = last(_radial_whole(sphere, C₀, external))

function property_contribution(
        sphere::RadialLayeredSphere, ::TensND.AbstractTens{4, 3}, C₀::TensND.TensISO{4, 3}; external::Bool = true, kw...
    )
    A, B = _radial_whole(sphere, C₀, external)
    return _contribution(A, B, C₀)
end

function MFH_Core.concentration_and_contribution(
        sphere::RadialLayeredSphere, ::TensND.AbstractTens{4, 3}, C₀::TensND.TensISO{4, 3}; external::Bool = true, kw...
    )
    A, B = _radial_whole(sphere, C₀, external)
    return (A, _contribution(A, B, C₀))
end

MFH_Core.concentration_and_flux_average(
    sphere::RadialLayeredSphere, ::TensND.AbstractTens{4, 3}, C₀::TensND.TensISO{4, 3}; external::Bool = true, kw...
) = _radial_whole(sphere, C₀, external)

# One layer: the moments on its two boundaries. By default the material of the
# layer alone; `external` counts what its outer interface carries, `internal`
# what its inner one does.
function _radial_layer(moment, radii, inside, outside, k, external::Bool, internal::Bool)
    V = radii[k]^3 - (k == 1 ? zero(radii[k]) : radii[k - 1]^3)
    outer = moment(radii[k], external ? outside[k] : inside[k])
    k == 1 && return outer / V
    return (outer - moment(radii[k - 1], internal ? inside[k - 1] : outside[k - 1])) / V
end

function gradient_gradient_concentration(
        sphere::RadialLayeredSphere{T, N}, C₀::TensND.TensISO{4, 3};
        layer::Int, external::Bool = false, internal::Bool = false, kw...
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    st = _radial_elastic_states(sphere, C₀)
    return TensISO{3}(
        _radial_layer(_strain_moment, sphere.radii, st.bulk_in, st.bulk_out, layer, external, internal),
        _radial_layer(_strain_moment, sphere.radii, st.dev_in, st.dev_out, layer, external, internal),
    )
end

"""
    flux_gradient_concentration(sphere, C₀; layer) -> TensISO{4,3}

Mean stress of the material of one layer per unit remote strain, for a sphere
with layers transversely isotropic about the radius: the stiffness of such a
layer turns with the point, so that this mean is not the product of a stiffness
by the mean strain of the layer. It is read on the tractions at the two
boundaries of the layer.
"""
function flux_gradient_concentration(
        sphere::RadialLayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}; layer::Int, kw...
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    st = _radial_elastic_states(sphere, C₀)
    return TensISO{3}(
        _radial_layer(_stress_moment, sphere.radii, st.bulk_in, st.bulk_out, layer, false, false),
        _radial_layer(_stress_moment, sphere.radii, st.dev_in, st.dev_out, layer, false, false),
    )
end

# ── Conduction ───────────────────────────────────────────────────────────────

# ⟨K·grad T⟩ = −⟨q⟩: the moment of the inward normal flux.
_gradient_moment(r, s::SVector{2}) = r^2 * s[1]
_flux_moment(r, s::SVector{2}) = -r^3 * s[2]

function _radial_whole(sphere::RadialLayeredSphere{T, N}, K₀::TensND.TensISO{2, 3}, external::Bool) where {T, N}
    st = _radial_conduction_states(sphere, K₀)
    R = sphere.radii[N]
    s = _boundary_state(st.inside, st.outside, external)
    return TensISO{3}(_gradient_moment(R, s) / R^3), TensISO{3}(_flux_moment(R, s) / R^3)
end

gradient_gradient_concentration(
    sphere::RadialLayeredSphere, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; external::Bool = true, kw...
) = first(_radial_whole(sphere, K₀, external))

flux_gradient_concentration(
    sphere::RadialLayeredSphere, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; external::Bool = true, kw...
) = last(_radial_whole(sphere, K₀, external))

function MFH_Core.property_contribution(sphere::RadialLayeredSphere, K₀::TensND.TensISO{2, 3}; external::Bool = true, kw...)
    A, B = _radial_whole(sphere, K₀, external)
    return _contribution(A, B, K₀)
end

function MFH_Core.concentration_and_contribution(
        sphere::RadialLayeredSphere, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; external::Bool = true, kw...
    )
    A, B = _radial_whole(sphere, K₀, external)
    return (A, _contribution(A, B, K₀))
end

MFH_Core.concentration_and_flux_average(
    sphere::RadialLayeredSphere, ::TensND.AbstractTens{2, 3}, K₀::TensND.TensISO{2, 3}; external::Bool = true, kw...
) = _radial_whole(sphere, K₀, external)

function gradient_gradient_concentration(
        sphere::RadialLayeredSphere{T, N}, K₀::TensND.TensISO{2, 3};
        layer::Int, external::Bool = false, internal::Bool = false, kw...
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    st = _radial_conduction_states(sphere, K₀)
    return TensISO{3}(_radial_layer(_gradient_moment, sphere.radii, st.inside, st.outside, layer, external, internal))
end

# ── Bounds: the volume averages of the moduli ────────────────────────────────

# The mean of `modulus` of a layer over its volume, between the radii a and b:
# that over the directions for a uniform layer, weighted by r² along the radius
# for a graded one.
_volume_average(c::RadialField, modulus, a, b) = _shell_average(modulus(c))
_volume_average(c::MFH_Core.PowerLawLayer, modulus, a, b) = _volume_average(MFH_Core.GradedLayer(c), modulus, a, b)
function _volume_average(g::MFH_Core.GradedLayer, modulus, a, b)
    data(r) = collect(TensND.get_data(_shell_average(modulus(_modulus_at(g, r))))) .* r^2
    integral = MFH_Core.radial_integral(data, a, b; rtol = g.reltol)
    return TensISO{3}((3 .* integral ./ (b^3 - a^3))...)
end

_radial_average(sphere::RadialLayeredSphere{T, N}, modulus) where {T, N} = sum(
    layer_volume_fraction(sphere, k) *
        _volume_average(layer_modulus(sphere, k), modulus, k == 1 ? zero(T) : sphere.radii[k - 1], sphere.radii[k])
        for k in 1:N
)
layer_property_average(sphere::RadialLayeredSphere, ::Val{4}) = _radial_average(sphere, identity)
layer_property_average(sphere::RadialLayeredSphere, ::Val{2}) = _radial_average(sphere, identity)
layer_inverse_property_average(sphere::RadialLayeredSphere, ::Val{4}) = _radial_average(sphere, _inverse)
layer_inverse_property_average(sphere::RadialLayeredSphere, ::Val{2}) = _radial_average(sphere, _inverse)
_inverse(c::RadialField) = inv(c)

# ── What is not available for radial layers ──────────────────────────────────

_no_radial(what) = throw(
    ArgumentError(
        "$what is not available for a sphere with layers transversely isotropic about the radius: " *
            "the mean strain and stress of the sphere and of its layers are " *
            "(`gradient_gradient_concentration`, `flux_gradient_concentration`, and the schemes)"
    )
)
LayeredSphereFields(::RadialLayeredSphere, ::TensND.TensISO{4, 3}) = _no_radial("the pointwise field")
LayeredSphereTransportFields(::RadialLayeredSphere, ::TensND.TensISO{2, 3}) = _no_radial("the pointwise field")
harmonic_response(::RadialLayeredSphere, ::TensND.TensISO{4, 3}, ::Integer) = _no_radial("the response to a harmonic of any degree")
harmonic_response(::RadialLayeredSphere, ::TensND.TensISO{2, 3}, ::Integer) = _no_radial("the response to a harmonic of any degree")
_bulk_layer_moduli(::RadialLayeredSphere) = _no_radial("this quantity, computed from the amplitudes of isotropic layers,")
_cond_layer_moduli(::RadialLayeredSphere) = _no_radial("this quantity, computed from the amplitudes of isotropic layers,")
_layer_iso_pairs(::RadialLayeredSphere) = _no_radial("this quantity, computed from the amplitudes of isotropic layers,")
