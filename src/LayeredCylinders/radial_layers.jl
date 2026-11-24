# =============================================================================
#  radial_layers.jl — layers transversely isotropic about the radial direction
#  of the cylinder.
#
#  A layer may be a field of TensND, `ChartTens`, transversely isotropic about
#  the unit vector e_r of the cylindrical chart at every point
#  (`radial_ti(CylindricalChart(), …)`): a wound or a deposited material, whose
#  plane of isotropy (e_θ, e_z) turns around the axis. Its Hill moduli about e_r
#  are n = C_rrrr, l = C_rrθθ = C_rrzz, k = (C_θθθθ + C_θθzz)/2,
#  m = (C_θθθθ − C_θθzz)/2 = C_θzθz and p = C_rθrθ = C_rzrz.
#
#  The four radial problems of the cylinder (`kernel.jl`) keep their states and
#  their interface conditions. In such a layer each one is a system of the first
#  order r dY/dr = M Y, on the displacement functions and r times the traction
#  functions, crossed by the power of M and started on the axis from the
#  solutions regular there (`Core/radial_systems.jl`). The systems are derived
#  symbolically in the tutorial `scripts/102`.
#
#  The chart is that of the cylinder: its axis is the axis of the cylinder,
#  whatever the direction of the latter in space.
#
#  Such a layer has no amplitudes. The second entry kept by a sequence for each
#  layer (`amps`) holds, for the axisymmetric problem, the integral of u_r across
#  the layer, which the mean axial stress needs, and zeros otherwise.
# =============================================================================

const RadialField{order} = MFH_Core.RadialField{order, TensND.CylindricalChart}

"Hill's moduli of a layer about the radial direction of the cylinder."
struct RadialModuli{T}
    n::T
    l::T
    k::T
    m::T
    p::T
end
RadialModuli(h::NamedTuple) = RadialModuli(promote(h.n, h.l, h.k, h.m, h.p)...)

"The coefficient of a potential problem along the radius and across it."
struct RadialPair{T}
    radial::T
    tangential::T
end

_radial_layer(c::RadialField) = MFH_Core.is_radial(c) ? c : throw(
        ArgumentError(
            "a layer of a LayeredCylinder given as a field of the cylindrical chart is transversely " *
            "isotropic about its radial direction (`radial_ti(CylindricalChart(), …)`); got $(c)"
        )
    )

_hill_moduli(c::RadialField{4}, axis) = RadialModuli(MFH_Core.radial_hill_moduli(_radial_layer(c)))
_hill_moduli(c::RadialField{2}, axis) = _wrong_order(c.pattern, "a stiffness")
function _conductivities(c::RadialField{2}, axis)
    kₜ, kᵣ = MFH_Core.radial_coefficients(_radial_layer(c))
    return (; t = RadialPair(kᵣ, kₜ), a = kₜ)
end
_conductivities(c::RadialField{4}, axis) = _wrong_order(c.pattern, "a conductivity")

const _RadialConductivities = NamedTuple{(:t, :a), <:Tuple{RadialPair, Any}}

_entries_eltype(::RadialModuli{T}) where {T} = T
_entries_eltype(::RadialPair{T}) where {T} = T
_entries_eltype(K::_RadialConductivities) = promote_type(_entries_eltype(K.t), typeof(K.a))
_in_type(::Type{S}, L::RadialModuli) where {S} = RadialModuli(S(L.n), S(L.l), S(L.k), S(L.m), S(L.p))
_in_type(::Type{S}, c::RadialPair) where {S} = RadialPair(S(c.radial), S(c.tangential))

function _check_stiffness(::Val{true}, L::RadialModuli, where_)
    isfinite(L.n) && isfinite(L.l) && isfinite(L.k) && L.n > 0 && L.m > 0 && L.p > 0 && L.n * L.k > L.l^2 || throw(
        ArgumentError(
            "$where_ of the LayeredCylinder has Hill moduli about the radius (n, l, k, m, p) = " *
                "$((L.n, L.l, L.k, L.m, L.p)): they must be finite, with n, m and p positive and n k > l²"
        )
    )
    return nothing
end
function _check_conductivity(::Val{true}, K::_RadialConductivities, where_, positive)
    isfinite(K.t.radial) && isfinite(K.t.tangential) && K.t.radial > 0 && K.t.tangential > 0 || throw(
        ArgumentError(
            "$where_ of the LayeredCylinder has the radial and tangential conductivities " *
                "$((K.t.radial, K.t.tangential)): they must be finite and positive"
        )
    )
    return nothing
end
_is_impermeable(::RadialPair) = false
_antiplane_modulus(L::RadialModuli) = RadialPair(L.p, L.m)

# ── The systems r dY/dr = M Y ────────────────────────────────────────────────

# Potential problem, Y = (φ, r f) with f = c_r φ′.
_pot_system(c::RadialPair) = SMatrix{2, 2}(zero(c.radial), c.tangential, 1 / c.radial, zero(c.radial))

# Axisymmetric problem with the axial strain ε, on Y = (u_r, r σ_rr, ε r, J):
# ε r, which forces the first two, obeys r d(ε r)/dr = ε r, and J = (1/r) ∫ u_r dr
# across the layer obeys r dJ/dr = u_r − J, so that the system is autonomous.
function _axi_system(L::RadialModuli)
    (; n, l, k, m) = L
    z, o = zero(n), one(n)
    return SMatrix{4, 4}(
        -l / n, k + m - l^2 / n, z, o,
        1 / n, l / n, z, z,
        -l / n, k - m - l^2 / n, o, z,
        z, z, z, -o,
    )
end

# Transverse shear, Y = (U, V, r σ_rr, r σ_rθ).
function _shear_system(L::RadialModuli)
    (; n, l, k, m, p) = L
    c = k + m - l^2 / n
    z, o = zero(n), one(n)
    return SMatrix{4, 4}(
        -l / n, 2 * o, c, 2 * c,
        -2 * l / n, o, 2 * c, 4 * c,
        1 / n, z, l / n, 2 * l / n,
        z, 1 / p, -2 * o, -o,
    )
end

_no_amplitudes(x, ::Val{N}) where {N} = ntuple(_ -> zero(x), Val(N))

# ── A layer: its start on the axis, and its crossing ─────────────────────────

function _pot_core(r, c::RadialPair, A₁)
    Y = MFH_Core.regular_states(_pot_system(c), Val(1))
    s = A₁ * r / Y[1]                       # φ = A₁ r at the radius of the core, as for an amplitude A₁
    return _no_amplitudes(A₁, Val(2)), (Y[1] * s, Y[2] * s / r)
end
function _pot_cross(r_in, r_out, c::RadialPair, (φ, f))
    Y = MFH_Core.radial_power(_pot_system(c), r_out / r_in) * SVector(φ, r_in * f)
    return _no_amplitudes(φ, Val(2)), (Y[1], Y[2] / r_out)
end

# The regular solutions of the axisymmetric system span two directions: the
# combination whose displacement is A₁ r and whose forcing is ε r at the radius
# of the core.
function _axi_core(r, L::RadialModuli, A₁, ε)
    R = MFH_Core.regular_states(_axi_system(L), Val(2))
    c = SMatrix{2, 2}(R[1, 1], R[3, 1], R[1, 2], R[3, 2]) \ SVector(A₁ * r, ε * r)
    Y = R * c
    return (r * Y[4], zero(A₁)), (Y[1], Y[2] / r)
end
function _axi_cross(r_in, r_out, L::RadialModuli, (u, σ), ε)
    Y = MFH_Core.radial_power(_axi_system(L), r_out / r_in) * SVector(u, r_in * σ, ε * r_in, zero(u))
    return (r_out * Y[4], zero(u)), (Y[1], Y[2] / r_out)
end

function _shear_core(r, L::RadialModuli, a₁, b₁)
    Y = MFH_Core.regular_states(_shear_system(L), Val(2)) * SVector(a₁, b₁)
    return _no_amplitudes(a₁, Val(4)), (Y[1], Y[2], Y[3] / r, Y[4] / r)
end
function _shear_cross(r_in, r_out, L::RadialModuli, (U, V, σrr, σrθ))
    Y = MFH_Core.radial_power(_shear_system(L), r_out / r_in) * SVector(U, V, r_in * σrr, r_in * σrθ)
    return _no_amplitudes(U, Val(4)), (Y[1], Y[2], Y[3] / r_out, Y[4] / r_out)
end

# ── Means over a layer ───────────────────────────────────────────────────────
#
# The in-plane mean stress of an annulus is the moment of the traction on its two
# boundaries (the stress being in equilibrium), as its mean strain is that of
# the displacement.
@inline _stress_moment(::Axisymmetric, s) = s[2]
@inline _stress_moment(::TransverseShear, s) = (s[3] - s[4]) / 2
@inline _stress_moment(::Antiplane, s) = 2 * s[2]
@inline _stress_moment(::TransverseConduction, s) = s[2]

function _annulus_flux(pb, seq, radii, k::Int)
    r = radii[k]
    g = r^2 * _stress_moment(pb, seq.inside[k])
    k == 1 && return g / r^2
    rᵢ = radii[k - 1]
    return (g - rᵢ^2 * _stress_moment(pb, seq.outside[k - 1])) / (r^2 - rᵢ^2)
end

# σ_zz = (k + m) ε + l u′ + (k − m) u/r: its mean over the annulus, from the
# displacement at its two radii and from the integral of u across it.
function _layer_axial_stress(L::RadialModuli, sol, seq, k::Int, ε)
    radii = sol.radii
    r = radii[k]
    I = seq.amps[k][1]
    inner = k == 1 ? zero(I) : radii[k - 1] * seq.outside[k - 1][1]
    area = k == 1 ? r^2 : r^2 - radii[k - 1]^2
    return (L.k + L.m) * ε + 2 * (L.l * (r * seq.inside[k][1] - inner - I) + (L.k - L.m) * I) / area
end

_layer_stress(L::RadialModuli, sol, k::Int) = _radial_layer_stress(L, sol, k)
function _radial_layer_stress(L, sol, k::Int)
    𝟙 = one(sol.L₀.k)
    return (;
        zz = _layer_axial_stress(L, sol, sol.axial, k, 𝟙),
        ze = _layer_axial_stress(L, sol, sol.dilatation, k, zero(𝟙)),
        mz = _annulus_flux(Axisymmetric(), sol.axial, sol.radii, k),
        me = _annulus_flux(Axisymmetric(), sol.dilatation, sol.radii, k),
        T = _annulus_flux(TransverseShear(), sol.shear, sol.radii, k),
        L = _annulus_flux(Antiplane(), sol.antiplane, sol.radii, k),
    )
end

_layer_flux(K::_RadialConductivities, sol, k::Int) =
    (; t = _annulus_flux(TransverseConduction(), sol.transverse, sol.radii, k), a = K.a * one(sol.K₀.t))

# The modulus of a layer and its conductivities, read with its outer radius `r`,
# which a uniform layer does not need.
_layer_hill(C, axis, r) = _hill_moduli(C, axis)
_layer_conductivities(K, axis, r) = _conductivities(K, axis)
_axial_conductivity(K, radii, k::Int) = K.a

# ── The mean of the moduli over a layer (bounds) ─────────────────────────────
#
# The mean over the angle of a tensor that turns about the axis is its
# projection on the tensors transversely isotropic about that axis.
function _about_axis(pattern::TensND.TensTI{order}, axis) where {order}
    frame = TensND.frame_from_axis(axis)
    e₁ = Tuple(TensND.vecbasis(frame)[:, 1])
    T = promote_type(eltype(pattern), eltype(e₁))
    turned = TensND.TensTI{order, T, length(TensND.get_data(pattern))}(map(T, TensND.get_data(pattern)), map(T, e₁))
    return first(TensND.project(turned, Val(:TI), axis))
end
_mean_tensor(c::RadialField, axis) = _about_axis(_radial_layer(c).pattern, axis)
_mean_inverse_tensor(c::RadialField, axis) = _about_axis(inv(_radial_layer(c).pattern), axis)
# Between two radii: a uniform layer does not depend on them.
_mean_tensor(P, axis, a, b) = _mean_tensor(P, axis)
_mean_inverse_tensor(P, axis, a, b) = _mean_inverse_tensor(P, axis)

# ── What is not available for radial layers ──────────────────────────────────

_has_radial_layer(cyl::LayeredCylinder) = any(c -> c isa Union{TensND.ChartTens, _Varying}, cyl.moduli)
_no_radial(what) = throw(
    ArgumentError(
        "$what is not available for a cylinder with layers transversely isotropic about its radial " *
            "direction: the mean strain and stress of the cylinder and of its layers are " *
            "(`gradient_gradient_concentration`, `flux_gradient_concentration`, and the schemes)"
    )
)

# ── Layers whose moduli vary with the radius ─────────────────────────────────
#
# A `GradedLayer` (or a function of the radius, or a `PowerLawLayer`) gives at
# each radius an isotropic modulus or one transversely isotropic about the
# radial direction. Its problems are the systems above with the moduli of each
# radius, integrated across the layer (`Core.radial_transfer`). On the axis, a
# problem with one regular solution starts at a small radius from the uniform
# medium found there; the transverse shear, which has two, carries their
# subspace by its impedance (`Core.regular_transfer`).

"Hill's moduli about the radial direction as a function `at` of the radius, of entries of type `S`."
struct GradedModuli{F, S}
    at::F
    layer::MFH_Core.GradedLayer
    GradedModuli{S}(at::F, layer) where {F, S} = new{F, S}(at, layer)
end
"The two coefficients of a potential problem as a function `at` of the radius."
struct GradedPair{F, S}
    at::F
    layer::MFH_Core.GradedLayer
    GradedPair{S}(at::F, layer) where {F, S} = new{F, S}(at, layer)
end

const _Varying = Union{Function, MFH_Core.GradedLayer, MFH_Core.PowerLawLayer}
# A function of the radius given as the modulus of a layer is a graded layer.
_stored_layer(c) = c
_stored_layer(profile::Function) = MFH_Core.GradedLayer(profile)
_graded(g::MFH_Core.GradedLayer) = g
_graded(profile::Union{Function, MFH_Core.PowerLawLayer}) = MFH_Core.GradedLayer(profile)

# The modulus found at a radius, read as one transversely isotropic about e_r.
_radial_moduli(C::TensND.TensISO{4, 3}) = _radial_moduli(TensND.radial_ti(TensND.CylindricalChart(), C))
_radial_moduli(c::RadialField{4}) = RadialModuli(MFH_Core.radial_hill_moduli(_radial_layer(c)))
_radial_pair(K::TensND.TensISO{2, 3}) = (k = first(TensND.get_data(K)); RadialPair(k, k))
_radial_pair(c::RadialField{2}) = ((kₜ, kᵣ) = MFH_Core.radial_coefficients(_radial_layer(c)); RadialPair(kᵣ, kₜ))
_not_graded(P) = throw(
    ArgumentError(
        "the modulus of a graded layer of a LayeredCylinder is, at each radius, isotropic or transversely " *
            "isotropic about the radial direction (`radial_ti(CylindricalChart(), …)`); got a $(nameof(typeof(P)))"
    )
)
_radial_moduli(P) = _not_graded(P)
_radial_pair(P) = _not_graded(P)

# `r` is the outer radius of the layer, where the number type of its modulus is read.
function _layer_hill(profile::_Varying, axis, r)
    g = _graded(profile)
    at(ρ) = _radial_moduli(g.profile(ρ))
    return GradedModuli{_entries_eltype(at(r))}(at, g)
end
function _layer_conductivities(profile::_Varying, axis, r)
    g = _graded(profile)
    at(ρ) = _radial_pair(g.profile(ρ))
    return (; t = GradedPair{_entries_eltype(at(r))}(at, g), a = nothing)
end

const _GradedConductivities = NamedTuple{(:t, :a), <:Tuple{GradedPair, Any}}
_entries_eltype(::GradedModuli{F, S}) where {F, S} = S
_entries_eltype(::GradedPair{F, S}) where {F, S} = S
_entries_eltype(K::_GradedConductivities) = _entries_eltype(K.t)
_in_type(::Type{S}, L::GradedModuli) where {S} = (at = L.at; GradedModuli{S}(r -> _in_type(S, at(r)), L.layer))
_in_type(::Type{S}, c::GradedPair) where {S} = (at = c.at; GradedPair{S}(r -> _in_type(S, at(r)), c.layer))
_in_type(::Type, ::Nothing) = nothing
_check_stiffness(::Val{true}, ::GradedModuli, where_) = nothing
_check_conductivity(::Val{true}, ::_GradedConductivities, where_, positive) = nothing
_is_impermeable(::GradedPair) = false
_antiplane_modulus(L::GradedModuli{F, S}) where {F, S} = (at = L.at; GradedPair{S}(r -> _antiplane_modulus(at(r)), L.layer))

# The axisymmetric system of a graded layer, on Y = (u_r, r σ_rr, ε r, I) with
# I = ∫ r σ_zz dr across the layer: r dI/dr = r² σ_zz, linear in the three others.
function _graded_axi_system(L::RadialModuli, r)
    (; n, l, k, m) = L
    z, o = zero(n), one(n)
    return SMatrix{4, 4}(
        -l / n, k + m - l^2 / n, z, r * (k - m - l^2 / n),
        1 / n, l / n, z, r * l / n,
        -l / n, k - m - l^2 / n, o, r * (k + m - l^2 / n),
        z, z, z, z,
    )
end

_column(x...) = SMatrix{length(x), 1}(x...)

function _pot_core(r, c::GradedPair, A₁)
    Y = MFH_Core.regular_transfer(ρ -> _pot_system(c.at(ρ)), c.layer, r, Val(1))
    s = A₁ * r / Y[1]
    return _no_amplitudes(A₁, Val(2)), (Y[1] * s, Y[2] * s / r)
end
function _pot_cross(r_in, r_out, c::GradedPair, (φ, f))
    Y = MFH_Core.radial_transfer(ρ -> _pot_system(c.at(ρ)), c.layer, r_in, r_out, _column(φ, r_in * f))
    return _no_amplitudes(φ, Val(2)), (Y[1], Y[2] / r_out)
end

# One regular solution and the one forced by ε: both are started at a small
# radius, in the uniform medium of the axis, where the integral I is negligible.
function _axi_core(r, L::GradedModuli, A₁, ε)
    r₀ = L.layer.start * r
    _, state = _axi_core(r₀, L.at(zero(r)), A₁, ε)
    return _axi_cross(r₀, r, L, state, ε)
end
function _axi_cross(r_in, r_out, L::GradedModuli, (u, σ), ε)
    Y = MFH_Core.radial_transfer(ρ -> _graded_axi_system(L.at(ρ), ρ), L.layer, r_in, r_out, _column(u, r_in * σ, ε * r_in, zero(u)))
    return (Y[4], zero(u)), (Y[1], Y[2] / r_out)
end

function _shear_core(r, L::GradedModuli, a₁, b₁)
    Y = MFH_Core.regular_transfer(ρ -> _shear_system(L.at(ρ)), L.layer, r, Val(2)) * SVector(a₁, b₁)
    return _no_amplitudes(a₁, Val(4)), (Y[1], Y[2], Y[3] / r, Y[4] / r)
end
function _shear_cross(r_in, r_out, L::GradedModuli, (U, V, σrr, σrθ))
    Y = MFH_Core.radial_transfer(ρ -> _shear_system(L.at(ρ)), L.layer, r_in, r_out, _column(U, V, r_in * σrr, r_in * σrθ))
    return _no_amplitudes(U, Val(4)), (Y[1], Y[2], Y[3] / r_out, Y[4] / r_out)
end

# The mean axial stress of a graded layer: the integral carried, over the area.
function _layer_axial_stress(::GradedModuli, sol, seq, k::Int, ε)
    r = sol.radii[k]
    return 2 * seq.amps[k][1] / (k == 1 ? r^2 : r^2 - sol.radii[k - 1]^2)
end
_layer_stress(L::GradedModuli, sol, k::Int) = _radial_layer_stress(L, sol, k)

# The mean over the section of a quantity `q(r)` of a graded layer, between a and b.
function _area_mean(q, layer::MFH_Core.GradedLayer, a, b)
    return 2 .* MFH_Core.radial_integral(r -> r .* q(r), a, b; rtol = layer.reltol) ./ (b^2 - a^2)
end
_inner_radius(radii, k::Int) = k == 1 ? zero(radii[k]) : radii[k - 1]

_axial_conductivity(K::_GradedConductivities, radii, k::Int) =
    _area_mean(r -> K.t.at(r).tangential, K.t.layer, _inner_radius(radii, k), radii[k])
_layer_flux(K::_GradedConductivities, sol, k::Int) = (;
    t = _annulus_flux(TransverseConduction(), sol.transverse, sol.radii, k),
    a = _axial_conductivity(K, sol.radii, k) * one(sol.K₀.t),
)

function _mean_tensor(profile::_Varying, axis, a, b, modulus = identity)
    g = _graded(profile)
    sample = _mean_tensor(modulus(g.profile(b)), axis)
    data = _area_mean(r -> collect(TensND.get_data(_mean_tensor(modulus(g.profile(r)), axis))), g, a, b)
    return _rebuild_ti(sample, data, axis)
end
_mean_inverse_tensor(profile::_Varying, axis, a, b) = _mean_tensor(profile, axis, a, b, inv)
