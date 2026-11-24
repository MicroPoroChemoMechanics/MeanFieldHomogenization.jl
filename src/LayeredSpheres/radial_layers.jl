# =============================================================================
#  radial_layers.jl — layers transversely isotropic about the radial direction.
#
#  A layer may be a field of TensND, `ChartTens`, transversely isotropic about
#  the unit radial vector e_r of the spherical chart at every point
#  (`radial_ti(SphericalChart(), …)`): a stiffness of five moduli or a
#  conductivity of two. Such a layer keeps the spherical symmetry of the
#  composite sphere, so that the remote strain still splits into a spherical
#  and a deviatoric part and each part is still carried by a few radial modes,
#  now powers of r with exponents that are no longer integers:
#
#    spherical part    u = r^s e_r,                 n s(s + 1) = 2(2k − l);
#    deviatoric part   u = r^s (a P₂ e_r + b P₂′ e_θ),
#                      n p t⁴ − B t² + C = 0 with t = s + 1/2;
#    conduction        T = r^s cos θ,               k_r s(s + 1) = 2 k_t,
#
#  (n, l, k, m, p) being Hill's moduli about e_r. They are derived symbolically
#  in the tutorial "Layered sphere: layers transversely isotropic about the
#  radius", and the theory page states them.
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
const RadialField{order} = TensND.ChartTens{order, 3, <:Any, <:TensND.TensTI{order}, TensND.SphericalChart}

_radial_field(c::RadialField) = TensND.isparallel(TensND.axis(c.pattern), TensND.radial_axis(c.chart)) ? c : _not_radial_field(c)
_radial_field(c::TensND.TensISO) = TensND.radial_ti(TensND.SphericalChart(), c)
_radial_field(c) = _not_radial_field(c)
_not_radial_field(c) = throw(
    ArgumentError(
        "a layer of a sphere with radial layers is isotropic or transversely isotropic about the " *
            "radius (`radial_ti(SphericalChart(), …)`); got $(c). A tensor of fixed axes, or a field " *
            "of another axis or of another chart, does not have the symmetry of the sphere"
    )
)

# Hill's moduli about e_r: n = C_rrrr, l = C_rrθθ, k = (C_θθθθ + C_θθφφ)/2,
# m = (C_θθθθ − C_θθφφ)/2, p = C_rθrθ, from the Walpole coefficients of the pattern.
function _hill_moduli(c::RadialField{4})
    ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆ = TensND.get_data(c.pattern)
    return (n = ℓ₁, l = ℓ₃ / sqrt(convert(TensND.sqrt2_eltype(typeof(ℓ₃)), 2)), k = ℓ₂ / 2, m = ℓ₅ / 2, p = ℓ₆ / 2)
end
# The conductivities across the radius and along it.
_conductivities(c::RadialField{2}) = TensND.get_data(c.pattern)

# The mean over all directions of the tensor of the layer: an isotropic tensor,
# which is its volume average over a spherical shell.
_shell_average(c::RadialField{4}) = TensND.best_fit_iso(c.pattern)
_shell_average(c::RadialField{2}) = ((kₜ, kᵣ) = _conductivities(c); TensISO{3}((2 * kₜ + kᵣ) / 3))

const RadialLayers = Tuple{RadialField, Vararg{RadialField}}
const RadialLayeredSphere{T, N} = LayeredSphere{T, N, <:RadialLayers}

# ── Exponents ────────────────────────────────────────────────────────────────
#
# s = −1/2 ± t. In the spherical problem and in conduction, t² is positive for
# every positive definite modulus: 1/4 + 2(2k − l)/n ≥ (1/2 − 2√(k/n))², since
# n k > l². In the deviatoric problem the two values of t² are the roots of a
# quadratic, which are complex conjugates for some positive definite moduli:
# the radial modes then oscillate, and the computation is made on complex
# numbers, `CT`, the solution being real.

_bulk_exponent(h) = sqrt(1 // 4 + 2 * (2 * h.k - h.l) / h.n)
_cond_exponent(c::RadialField{2}) = ((kₜ, kᵣ) = _conductivities(c); sqrt(1 // 4 + 2 * kₜ / kᵣ))

# n p τ² − B τ + C = 0, of roots τ = t²: its coefficients B and C.
function _shear_quadratic(h)
    (; n, l, k, m, p) = h
    B = 6 * k * n + 4 * k * p - 6 * l^2 - 14 * l * p + 4 * m * n + 5 * n * p / 2
    C = 16 * k * m + 3 * k * n / 2 + 9 * k * p - 3 * l^2 / 2 - 8 * l * m + 9 * l * p / 2 + m * n + 24 * m * p + 9 * n * p / 16
    return B, C
end

function _shear_exponents(h, ::Type{CT}) where {CT}
    B, C = _shear_quadratic(h)
    Δ = sqrt(CT(B^2 - 4 * h.n * h.p * C))
    return sqrt((B - Δ) / (2 * h.n * h.p)), sqrt((B + Δ) / (2 * h.n * h.p))
end

# Whether the exponents of a layer are complex. A symbol is not compared: its
# exponents are kept as the expressions they are.
_oscillates(c::RadialField{4}) = _oscillates(Val(is_hard_numeric(eltype(c))), _hill_moduli(c))
_oscillates(::Val{false}, h) = false
function _oscillates(::Val{true}, h)
    B, C = _shear_quadratic(h)
    return B^2 < 4 * h.n * h.p * C
end
_mode_type(sphere, ::Type{TP}) where {TP} = any(_oscillates, sphere.moduli) ? Complex{TP} : TP
_real_part(x::AbstractArray{<:Complex}) = real(x)
_real_part(x::AbstractArray) = x
# r^s. With a complex exponent, by the exponential: the power of Julia converts
# an exponent whose value is an integer to an `Int`, which a dual number is not.
_power(r, s) = r^s
_power(r::Complex, s) = exp(s * log(r))

# ── Fundamental matrices: the state of each mode at the radius r ─────────────

# Spherical part, state (u_r, σ_rr), modes r^(−1/2 + t) and r^(−1/2 − t).
function _bulk_modes(h, r)
    t = oftype(r, _bulk_exponent(h))
    column(s) = (rs = _power(r, s); (rs, (h.n * s + 2 * h.l) * rs / r))
    return SMatrix{2, 2}(column(-1 // 2 + t)..., column(-1 // 2 - t)...)
end

# Deviatoric part, state (U, W, σ_rr, σ_rθ) of u = U P₂ e_r + W P₂′ e_θ. The
# shape (a, b) of a mode is in the kernel of the two equilibrium equations; of
# the two expressions of it, the one of larger norm is kept, so that neither
# vanishes at the moduli where the other one does.
function _shear_shape(h, s)
    (; n, l, k, m, p) = h
    q = s * (s + 1)
    from_θ = (6 * k + 4 * m + 2 * p - p * q, 2 * k + (l + p) * s + 2 * p)
    from_r = (6 * l * (s + 1) + 6 * p * (s - 1) - 12 * k, n * q - 4 * k + 2 * l - 6 * p)
    return _larger(Val(is_hard_numeric(typeof(q))), from_θ, from_r)
end
_larger(::Val{false}, u, v) = u
_larger(::Val{true}, u, v) = sum(abs2, u) ≥ sum(abs2, v) ? u : v

function _shear_modes(h, r::CT) where {CT}
    t₁, t₂ = _shear_exponents(h, CT)
    function column(s)
        a, b = _shear_shape(h, s)
        rs = _power(r, s)
        return (a * rs, b * rs, ((h.n * s + 2 * h.l) * a - 6 * h.l * b) * rs / r, h.p * (a + (s - 1) * b) * rs / r)
    end
    # The two modes that are regular at the center first.
    return SMatrix{4, 4}(column(-1 // 2 + t₁)..., column(-1 // 2 + t₂)..., column(-1 // 2 - t₁)..., column(-1 // 2 - t₂)...)
end

# Conduction, state (T̂, q̂) with q̂ = −k_r ∂T̂/∂r the outward flux.
function _cond_modes(c::RadialField{2}, r)
    t = oftype(r, _cond_exponent(c))
    kᵣ = last(_conductivities(c))
    column(s) = (rs = _power(r, s); (rs, -kᵣ * s * rs / r))
    return SMatrix{2, 2}(column(-1 // 2 + t)..., column(-1 // 2 - t)...)
end

# ── The states at the boundaries of the layers ───────────────────────────────

# The states of `nprobes` solutions regular at the center, carried to the
# matrix: for each interface k, the state on the inside (r_k⁻) and on the
# outside (r_k⁺), one column per solution.
function _radial_states(sphere::RadialLayeredSphere{T, N}, modes, interface, ::Val{nprobes}, ::Type{TP}) where {T, N, nprobes, TP}
    radii = sphere.radii
    S = modes(layer_modulus(sphere, 1), TP(radii[1]))[:, SVector{nprobes}(1:nprobes)]
    inside = Vector{typeof(S)}(undef, N)
    outside = Vector{typeof(S)}(undef, N)
    for k in 1:N
        inside[k] = S
        S = typeof(S)(interface(layer_interface(sphere, k), TP(radii[k])) * S)
        outside[k] = S
        k < N && (S = modes(layer_modulus(sphere, k + 1), TP(radii[k + 1])) * (inv(modes(layer_modulus(sphere, k + 1), TP(radii[k]))) * S))
    end
    return inside, outside
end

_radial_eltype(sphere::RadialLayeredSphere{T}, x₀...) where {T} = TensND.float_like(
    promote_type(T, map(typeof, x₀)..., map(eltype, sphere.moduli)..., interfaces_eltype(sphere.interfaces))
)

# The states per unit remote strain, spherical then deviatoric.
function _radial_elastic_states(sphere::RadialLayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}) where {T, N}
    κ₀, μ₀ = TensND.k_mu(C₀)
    TP = _radial_eltype(sphere, κ₀, μ₀)
    R = TP(sphere.radii[N])
    u = one(TP)
    bulk_in, bulk_out = _radial_states(
        sphere, (c, r) -> _bulk_modes(_hill_moduli(c), r), (itf, r) -> _bulk_interface_T(itf, u, u, r), Val(1), TP
    )
    A∞, _ = _bulk_extract_AB(R, TP(κ₀), TP(μ₀), bulk_out[N][1], bulk_out[N][2])
    dev_in, dev_out = _radial_states(
        sphere, (c, r) -> _shear_modes(_hill_moduli(c), r), (itf, r) -> _shear_interface_T(itf, u, u, real(r)), Val(2),
        _mode_type(sphere, TP)
    )
    # The combination of the two regular solutions whose remote part is the unit
    # uniform strain: amplitudes (a, b) = (1, 0) in the matrix.
    amplitudes = _shear_M_inverse(R, TP(κ₀), TP(μ₀)) * dev_out[N]
    λ = inv(SMatrix{2, 2}(amplitudes[1, 1], amplitudes[2, 1], amplitudes[1, 2], amplitudes[2, 2])) * SVector(u, zero(u))
    return (
        bulk_in = map(S -> S[:, 1] / A∞, bulk_in), bulk_out = map(S -> S[:, 1] / A∞, bulk_out),
        dev_in = map(S -> _real_part(S * λ), dev_in), dev_out = map(S -> _real_part(S * λ), dev_out),
    )
end

# The states per unit remote gradient.
function _radial_conduction_states(sphere::RadialLayeredSphere{T, N}, K₀::TensND.TensISO{2, 3}) where {T, N}
    k₀ = _iso_scalar(K₀)
    TP = _radial_eltype(sphere, k₀)
    inside, outside = _radial_states(sphere, _cond_modes, (itf, r) -> _cond_interface_T(itf, one(TP), r), Val(1), TP)
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
_as_layers(moduli::Tuple) = any(c -> c isa TensND.ChartTens, moduli) ? map(_radial_field, moduli) : moduli
_as_layers(moduli) = moduli

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

_radial_average(sphere::RadialLayeredSphere{T, N}, modulus) where {T, N} =
    sum(layer_volume_fraction(sphere, k) * _shell_average(modulus(layer_modulus(sphere, k))) for k in 1:N)
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
