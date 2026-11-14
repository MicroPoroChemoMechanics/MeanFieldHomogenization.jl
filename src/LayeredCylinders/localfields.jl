# =============================================================================
#  localfields.jl — pointwise fields of a `LayeredCylinder`: strain, stress and
#  displacement in elasticity, temperature, gradient and flux in conduction, in
#  any layer and in the matrix.
#
#  The entry points and their conventions are those of the layered sphere:
#  results in the canonical basis, regions `1..N` for the layers and `N + 1`
#  for the matrix, and a `side` keyword for a point lying on an interface,
#  where the fields are discontinuous.
#
#  In the cylindrical frame of the point the strain concentration depends on
#  the radius only (`kernel.jl`); it is rotated to the canonical frame. It has
#  the minor symmetries and no major one, and no symmetry about the axis: it
#  comes back as a general `Tens`.
# =============================================================================

"""
    get_layer(cyl::LayeredCylinder, r; side = :outer) -> Int

Index of the region containing the distance `r` to the axis: ``1, \\dots, N``
for the layers, ``N+1`` for the matrix. On an interface ``r = r_k``,
`side = :outer` (default) gives the region outside it and `side = :inner` the
one inside, as for a [`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere). A symbolic radius is not
comparable: name the region with the `layer` keyword of the pointwise entry
points instead.
"""
function get_layer(cyl::LayeredCylinder{T, N}, r; side::Symbol = :outer) where {T, N}
    side === :outer || side === :inner ||
        throw(ArgumentError("get_layer: `side` must be :outer or :inner, got $(side)"))
    is_hard_numeric(typeof(r)) || throw(
        ArgumentError(
            "get_layer: a symbolic radius cannot be located by comparison " *
                "(got $(typeof(r))); name the region with `layer = k` instead " *
                "(1..$(N) for a layer, $(N + 1) for the matrix)"
        )
    )
    return _locate_region(cyl.radii, r, side)
end

# ── Cylindrical frame of a point ─────────────────────────────────────────────

# A unit vector normal to the axis, chosen once per axis: the canonical vector
# least aligned with it, made orthogonal. It only fixes the origin of θ.
function _transverse_direction(n::NTuple{3})
    is_hard_numeric(eltype(n)) || throw(
        ArgumentError(
            "the cylindrical frame of a LayeredCylinder of symbolic axis $n has no " *
                "origin of θ that can be chosen by comparison; give the point as a vector " *
                "off the axis"
        )
    )
    i = argmin(map(abs, n))
    e = ntuple(j -> oftype(n[1], j == i), 3)
    t = e .- n[i] .* n
    return t ./ sqrt(sum(abs2, t))
end

# The canonical components of a point or of a gradient: a first-order tensor is
# read in the canonical basis, whatever the basis it is stored in.
_canonical_vector(x::TensND.AbstractTens{1}) = Tuple(TensND.canonical_components(x))
_canonical_vector(x) = TensND._extract_vec(x)

_cross(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

"""
    _cylindrical_frame(cyl, x) -> (r, z, Q)
    _cylindrical_frame(cyl, r, θ, z) -> (r, z, Q)

Distance to the axis, abscissa along it, and the matrix whose columns are
``(\\underline e_r, \\underline e_\\theta, \\underline e_z)`` in canonical
components, for a point given as a vector or by its cylindrical coordinates
(``\\theta`` measured from a fixed direction normal to the axis,
``\\underline e_1`` when the axis is ``\\underline e_3``). On the axis
``\\underline e_r`` is that fixed direction.
"""
function _cylindrical_frame(cyl::LayeredCylinder, x)
    v = _canonical_vector(x)
    n = cyl.axis
    z = sum(v[i] * n[i] for i in 1:3)
    ρ = ntuple(i -> v[i] - z * n[i], 3)
    r = sqrt(sum(ρ[i]^2 for i in 1:3))
    _on_axis(r, z) && return zero(r), z, _frame_matrix(oftype.(ρ, _transverse_direction(n)), n)
    return r, z, _frame_matrix(ρ ./ r, n)
end

# A point whose distance to the axis is within rounding of zero: the radial
# direction read on it would be noise. The fields are regular on the axis, so
# reading them at r = 0 changes them by O(r²), below rounding.
_on_axis(r, z) = is_hard_numeric(typeof(r)) && r ≤ 8 * eps(typeof(r)) * abs(z)

function _cylindrical_frame(cyl::LayeredCylinder, r, θ, z)
    n = cyl.axis
    t₁ = _transverse_direction(n)
    t₂ = _cross(n, t₁)
    s, c = sincos(θ)
    return r, z, _frame_matrix(c .* t₁ .+ s .* t₂, n)
end

function _frame_matrix(eᵣ, n)
    eθ = _cross(n, eᵣ)
    return [eᵣ[1] eθ[1] n[1]; eᵣ[2] eθ[2] n[2]; eᵣ[3] eθ[3] n[3]]
end

# ── Cached solution: elasticity ──────────────────────────────────────────────

"""
    LayeredCylinderFields(cyl, C₀)

Pointwise elastic solution of a [`LayeredCylinder`](@ref) in the reference
`C₀` (isotropic, or transversely isotropic about the axis), under a unit
remote strain: the amplitudes of the radial problems in every layer and in
the matrix. Build it once for many points; every `local_*` entry point also
takes `(cyl, C₀)` and then solves the problem at each call.
"""
struct LayeredCylinderFields{C <: LayeredCylinder, S, M, T0}
    cylinder::C
    sol::S
    matrix::M
    C₀::T0
end

function LayeredCylinderFields(cyl::LayeredCylinder, C₀::_Order4)
    sol = _elastic_solution(cyl, C₀)
    return LayeredCylinderFields(cyl, sol, _matrix_amplitudes(sol), _stiffness_tensor(sol.L₀, cyl.axis))
end

"""
    region_stiffness(f::LayeredCylinderFields, k) -> TensTI{4}

Stiffness of region `k`, the reference for ``k = N+1``.
"""
function region_stiffness(f::LayeredCylinderFields, k::Int)
    N = layer_count(f.cylinder)
    return _stiffness_tensor(k ≤ N ? f.sol.Ls[k] : f.sol.L₀, f.cylinder.axis)
end

_resolve(f, r, side::Symbol, layer) =
    _resolve_region(f.cylinder, layer_count(f.cylinder), r, side, layer)

# 𝔸(x) in the canonical basis: the Mandel matrix of the cylindrical frame,
# rotated.
function _local_A(f::LayeredCylinderFields, r, Q, k::Int)
    M = _local_mandel(_local_strain_entries(f.sol, f.matrix, k, r))
    R = TensND.mandel_rotation(Q)
    # a 6 × 6 Mandel matrix of a minor-symmetric fourth-order tensor, said by type
    return TensND.from_mandel(Tensors.SymmetricTensor{4, 3}, R * M * transpose(R))
end

function _local_A_scaled(f::LayeredCylinderFields, r, Q, k::Int, pre::Bool, post::Bool)
    A = _local_A(f, r, Q, k)
    pre && (A = region_stiffness(f, k) ⊡ A)
    post && (A = A ⊡ inv(f.C₀))
    return A
end

# Displacement: the fields of the four problems, plus the rigid rotation that
# makes the remote displacement E·x (the antiplane problem is solved with
# u_z = 2 E_rz r).
function _local_u(f::LayeredCylinderFields, r, z, Q, k::Int, ε∞)
    sol = f.sol
    N = length(sol.radii)
    L = k ≤ N ? sol.Ls[k] : sol.L₀
    amp(seq, name) = k ≤ N ? seq.amps[k] : getfield(f.matrix, name)
    E = transpose(Q) * TensND.canonical_components(ε∞) * Q      # cylindrical components
    e = (E[1, 1] + E[2, 2]) / 2
    D = (E[1, 1] - E[2, 2]) / 2
    Aₑ, Bₑ = amp(sol.dilatation, :dilatation)
    A_z, B_z = amp(sol.axial, :axial)
    U, V = _shear_displacements(r, L.k, L.m, amp(sol.shear, :shear)...)
    A, B = amp(sol.antiplane, :antiplane)
    φ = r * A + _over(B, r, 1)
    uᵣ = (r * Aₑ + _over(Bₑ, r, 1)) * e + (r * A_z + _over(B_z, r, 1)) * E[3, 3] + U * D + z * E[1, 3]
    uθ = -V * E[1, 2] + z * E[2, 3]
    u_z = z * E[3, 3] + (2 * φ - r) * E[1, 3]
    return TensND.Vec{3}(ntuple(i -> Q[i, 1] * uᵣ + Q[i, 2] * uθ + Q[i, 3] * u_z, 3))
end

# ── Cached solution: conduction ──────────────────────────────────────────────

"""
    LayeredCylinderTransportFields(cyl, K₀)

Pointwise conduction solution of a [`LayeredCylinder`](@ref) in the reference
`K₀`, under a unit remote gradient: conduction twin of
[`LayeredCylinderFields`](@ref).
"""
struct LayeredCylinderTransportFields{C <: LayeredCylinder, S, M, T0}
    cylinder::C
    sol::S
    matrix::M
    K₀::T0
end

function LayeredCylinderTransportFields(cyl::LayeredCylinder, K₀::_Order2)
    sol = _conduction_solution(cyl, K₀)
    N = layer_count(cyl)
    mat = _pot_amplitudes(sol.radii[N], sol.K₀.t, sol.transverse.outside[N]...)
    return LayeredCylinderTransportFields(cyl, sol, mat, _ti_tensor(sol.K₀, cyl.axis))
end

function _region_conductivity(f::LayeredCylinderTransportFields, k::Int)
    N = layer_count(f.cylinder)
    return _ti_tensor(k ≤ N ? f.sol.Ks[k] : f.sol.K₀, f.cylinder.axis)
end

_transport_amplitudes(f::LayeredCylinderTransportFields, k::Int) =
    k ≤ layer_count(f.cylinder) ? f.sol.transverse.amps[k] : f.matrix

# 𝑨(x) = φ′ e_r ⊗ e_r + (φ/r) e_θ ⊗ e_θ + e_z ⊗ e_z, in canonical components.
function _local_K(f::LayeredCylinderTransportFields, r, Q, k::Int)
    dφ, φ_r = _pot_gradients(r, _transport_amplitudes(f, k)...)
    A = Q * Diagonal([dφ, φ_r, one(dφ)]) * transpose(Q)
    return MFH_Core._tens_in(A, MFH_Core.inclusion_basis(f.cylinder))
end

function _local_K_scaled(f::LayeredCylinderTransportFields, r, Q, k::Int, pre::Bool, post::Bool)
    A = _local_K(f, r, Q, k)
    pre && (A = -_region_conductivity(f, k) ⋅ A)
    post && (A = A ⋅ inv(f.K₀))
    return A
end

function _local_T(f::LayeredCylinderTransportFields, r, z, Q, k::Int, ∇T∞)
    A, B = _transport_amplitudes(f, k)
    G = transpose(Q) * collect(_canonical_vector(∇T∞))
    return (r * A + _over(B, r, 1)) * G[1] + z * G[3]
end

# ── Entry points ─────────────────────────────────────────────────────────────
#
#  A point is a vector `x`, or cylindrical coordinates `(r, θ, z)`.

const _CylFields = Union{LayeredCylinderFields, LayeredCylinderTransportFields}

_concentration(f::LayeredCylinderFields, r, Q, k, pre, post) = _local_A_scaled(f, r, Q, k, pre, post)
_concentration(f::LayeredCylinderTransportFields, r, Q, k, pre, post) = _local_K_scaled(f, r, Q, k, pre, post)

for (name, pre, post) in (
        (:local_gradient_gradient_concentration, false, false),
        (:local_flux_gradient_concentration, true, false),
        (:local_gradient_flux_concentration, false, true),
        (:local_flux_flux_concentration, true, true),
    )
    @eval function $name(f::_CylFields, point...; side::Symbol = :outer, layer = nothing)
        r, _, Q = _cylindrical_frame(f.cylinder, point...)
        return _concentration(f, r, Q, _resolve(f, r, side, layer), $pre, $post)
    end
end

_apply(A::TensND.AbstractTens{4}, ε∞) = A ⊡ ε∞
_apply(A::TensND.AbstractTens{2}, ∇T∞) = TensND.Vec{3}(
    Tuple(TensND.canonical_components(A) * collect(_canonical_vector(∇T∞)))
)

# The remote loading is the last positional argument, the point what precedes.
function local_gradient(f::_CylFields, args...; side::Symbol = :outer, layer = nothing)
    r, _, Q = _cylindrical_frame(f.cylinder, Base.front(args)...)
    return _apply(_concentration(f, r, Q, _resolve(f, r, side, layer), false, false), last(args))
end

function local_flux(f::_CylFields, args...; side::Symbol = :outer, layer = nothing)
    r, _, Q = _cylindrical_frame(f.cylinder, Base.front(args)...)
    return _apply(_concentration(f, r, Q, _resolve(f, r, side, layer), true, false), last(args))
end

function local_displacement(
        f::LayeredCylinderFields, args...; side::Symbol = :outer, layer = nothing
    )
    r, z, Q = _cylindrical_frame(f.cylinder, Base.front(args)...)
    return _local_u(f, r, z, Q, _resolve(f, r, side, layer), last(args))
end

function local_temperature(
        f::LayeredCylinderTransportFields, args...; side::Symbol = :outer, layer = nothing
    )
    r, z, Q = _cylindrical_frame(f.cylinder, Base.front(args)...)
    return _local_T(f, r, z, Q, _resolve(f, r, side, layer), last(args))
end

# One-off evaluations from `(cyl, C₀)` or `(cyl, K₀)`: the problem is solved at
# each call.
for name in (
        :local_gradient_gradient_concentration, :local_flux_gradient_concentration,
        :local_gradient_flux_concentration, :local_flux_flux_concentration,
        :local_gradient, :local_flux,
    )
    @eval $name(cyl::LayeredCylinder, C₀::_Order4, args...; kw...) =
        $name(LayeredCylinderFields(cyl, C₀), args...; kw...)
    @eval $name(cyl::LayeredCylinder, K₀::_Order2, args...; kw...) =
        $name(LayeredCylinderTransportFields(cyl, K₀), args...; kw...)
end

local_displacement(cyl::LayeredCylinder, C₀::_Order4, args...; kw...) =
    local_displacement(LayeredCylinderFields(cyl, C₀), args...; kw...)
local_temperature(cyl::LayeredCylinder, K₀::_Order2, args...; kw...) =
    local_temperature(LayeredCylinderTransportFields(cyl, K₀), args...; kw...)
