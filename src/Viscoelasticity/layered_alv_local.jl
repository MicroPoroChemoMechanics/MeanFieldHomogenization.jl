# =============================================================================
#  layered_alv_local.jl — pointwise operators of the n-layer sphere in aging
#  linear viscoelasticity, orders 4 (elasticity) and 2 (conduction).
#
#  The Volterra counterpart of `LayeredSpheres/localfields.jl`. The field at a
#  point x = r n is built from radial functions, each a sum of modes r^p whose
#  coefficients are n × n Volterra matrices acting on the remote history:
#
#   * Y₀ (spherical part) and conduction: `f = A r + B / r²`;
#   * Y₂ (deviatoric part): the displacement amplitudes `U` and `W` of the
#     state vector, read on the two displacement rows of the fundamental matrix
#     `_shear_M_matrix_alv` — modes r, r³, r⁻⁴, r⁻² — applied to the amplitudes
#     (a, b, c, d) of the region. With u = g (εᵈ·n) + h (n·εᵈ·n) n they are
#     `U = 2 (g + h)`, `W = g`, so `g = W` and `h = U/2 − W`.
#
#  The moduli sit on the left of the amplitudes, the normalization by the
#  remote amplitude on their right, as in the recurrences. The pointwise
#  operator is transversely isotropic about n: its six Walpole coefficients are
#  the elastic combinations of f/r, f′, g/r, g′, h/r, h′, each an n × n matrix,
#  and the 6n × 6n block matrix is Σᵢ ℓᵢ ⊗ Wᵢ(n), Wᵢ(n) the Mandel matrix of the
#  i-th Walpole tensor in the canonical frame.
#
#  The singular modes have zero amplitudes in the core: they are written as
#  zeros, and their weights r^p are not evaluated at r = 0.
# =============================================================================

"""
    LayeredSphereALVFields(sphere, ref, times)

Pointwise solution of a [`LayeredSphere`](@ref) against the isotropic aging
reference `ref` (a `ViscoLaw`, or the Volterra matrix of a running estimate) on
the instants `times`: the mode amplitudes of the radial functions in every
layer and in the matrix, as ``n\\times n`` Volterra matrices per unit remote
history. The order of `ref` selects elasticity or conduction.

Build it once for many points and pass it to
[`local_gradient_gradient_concentration_alv`](@ref),
[`local_flux_gradient_concentration_alv`](@ref),
[`local_displacement_alv`](@ref) and [`local_temperature_alv`](@ref): the forms
taking `(sphere, ref, times)` run the recurrences again at every call.
"""
struct LayeredSphereALVFields{O, S <: LayeredSphere, P, R}
    order::Val{O}
    sphere::S
    profiles::P         # per region, the mode coefficients of its radial functions
    relaxations::R      # block matrices of the layers, then of the reference
end

function LayeredSphereALVFields(
        sphere::LayeredSphere, ref::_ALVReference, times::AbstractVector{<:Real}
    )
    o = _alv_order(ref, times)
    N = layer_count(sphere)
    relaxations = [
        _alv_blocks(k ≤ N ? layer_modulus(sphere, k) : ref, times, _block_rows(o)) for k in 1:(N + 1)
    ]
    return LayeredSphereALVFields(o, sphere, _radial_profiles_alv(o, sphere, ref, times), relaxations)
end

# The exponents of the modes of f (Y₀ and conduction) and of U, W (Y₂).
const _RADIAL_POWERS = (1, -2)
const _SHEAR_POWERS = (1, 3, -4, -2)

# The profiles of every region with one element type: a dual number in one
# layer reaches the coefficients of the layers outside it only.
function _common_profiles(profiles::AbstractVector{<:NamedTuple})
    E = promote_type((eltype(M) for p in profiles for coefs in p for M in coefs)...)
    return [map(coefs -> map(Matrix{E}, coefs), p) for p in profiles]
end

function _radial_profiles_alv(
        ::Val{4}, sphere::LayeredSphere{T, N}, ref::_ALVReference, times::AbstractVector{<:Real}
    ) where {T, N}
    n = length(times)
    radii = sphere.radii
    layers, M_κ_0, M_μ_0 = _bulk_layer_moduli_alv(sphere, ref, times)
    bulk, A_∞, B_∞ = bulk_amplitude_seq_alv(sphere, ref, times)
    A_∞⁻¹ = volterra_inverse(A_∞; block_size = 1)
    inside, outside = _shear_unit_states_alv(sphere, layers, M_κ_0, M_μ_0, times)
    profiles = map(1:(N + 1)) do k
        M_κ, M_μ = k ≤ N ? layers[k] : (M_κ_0, M_μ_0)
        A, B = k ≤ N ? bulk[k] : (A_∞, B_∞)
        amplitudes = _shear_amplitudes_alv(
            radii[min(k, N)], M_κ, M_μ, n, k ≤ N ? inside[k] : outside
        )
        # The amplitudes the solution fixes are written, not read back from a
        # solve whose round-off a weight r⁻⁵ or r² amplifies: the singular modes
        # are absent from the core (regularity at the origin), and the regular
        # ones of the matrix are the remote strain, (a, b) = (𝟙, 0).
        a, b = k ≤ N ? amplitudes[1:2] : (one(amplitudes[1]), zero(amplitudes[2]))
        c, d = k == 1 ? zero.(amplitudes[3:4]) : amplitudes[3:4]
        U₂, W₂, U₄ = _shear_mode_operators_alv(M_κ, M_μ)
        (;
            f = (k ≤ N ? A * A_∞⁻¹ : one(A_∞), B * A_∞⁻¹),
            U = (2 .* a, U₂ * b, 3 .* c, U₄ * d),
            W = (a, W₂ * b, -c, d),
        )
    end
    return _common_profiles(profiles)
end

_radial_profiles_alv(
    ::Val{2}, sphere::LayeredSphere, ref::_ALVReference, times::AbstractVector{<:Real}
) = [(; f = ab) for ab in _cond_unit_solution_alv(sphere, ref, times).amps]

# The weight r^q of a mode. At the origin a singular one is not evaluated: its
# coefficient is zero there (`_region_at` admits the origin in the core only).
_radial_weight(r, q::Int) = q < 0 && LayeredSpheres._at_origin(r) ? zero(r) : r^q

# (X/r, X′) of X(r) = Σⱼ cⱼ r^pⱼ, as mode sums: no division of X by r.
function _mode_sum(coefs::NTuple{M, AbstractMatrix}, powers::NTuple{M, Int}, r) where {M}
    w = map(p -> _radial_weight(r, p - 1), powers)
    return sum(map((wⱼ, c) -> wⱼ .* c, w, coefs)), sum(map((p, wⱼ, c) -> (p * wⱼ) .* c, powers, w, coefs))
end

# Radius (a float one), radial direction and region of a point.
function _region_at(f::LayeredSphereALVFields, point::Tuple, side::Symbol, layer)
    r, n = LayeredSpheres._radial_frame(point...)
    k = LayeredSpheres._resolve_layer(f.sphere, r; side, layer)
    LayeredSpheres._at_origin(r) && k != 1 && throw(
        ArgumentError("the origin belongs to the core (layer 1); layer = $k was asked at r = 0")
    )
    return r / one(r), n, k
end

# The Mandel matrices, in the canonical frame, of the six Walpole tensors of
# axis n.
_walpole_mandel(n::NTuple{3, E}) where {E} =
    map(W -> TensND.mandel(W, TensND.CanonicalBasis{3, E}()), TensND.walpole_basis(n))

# 𝔸(x): the elastic combinations of `LayeredSpheres._local_A`, with
# 𝔸 = 𝔸_bulk + 𝔸_dev : 𝕂 reduced on the Walpole basis — 𝔸_dev : 𝕁 has the form
# of 𝔸_bulk, which it corrects by (d₁, g/r + G₄) / 3.
function _local_alv(f::LayeredSphereALVFields{4}, r, n, k::Int)
    p = f.profiles[k]
    f_r, f′ = _mode_sum(p.f, _RADIAL_POWERS, r)
    U_r, U′ = _mode_sum(p.U, _SHEAR_POWERS, r)
    g_r, g′ = _mode_sum(p.W, _SHEAR_POWERS, r)
    h_r = U_r ./ 2 .- g_r
    h′ = U′ ./ 2 .- g′
    G₂ = g′ .- g_r .+ 2 .* h_r
    d₁ = g_r .+ G₂ .+ (h′ .- 3 .* h_r) .+ h_r
    P = (f′ .- d₁) ./ 3
    Q = (f_r .- g_r .- h_r) ./ 3
    s = sqrt(2 * one(eltype(P)))
    ℓ = (d₁ .+ P, g_r .+ 2 .* Q, s .* P, s .* (h_r .+ Q), g_r, g_r .+ G₂ ./ 2)
    return sum(map(kron, ℓ, _walpole_mandel(n)))
end

# 𝐀(x) = f′ n⊗n + (f/r)(𝟏 − n⊗n).
function _local_alv(f::LayeredSphereALVFields{2}, r, n, k::Int)
    f_r, f′ = _mode_sum(f.profiles[k].f, _RADIAL_POWERS, r)
    p = [nᵢ * nⱼ for nᵢ in n, nⱼ in n]
    return kron(f′, p) + kron(f_r, I - p)
end

"""
    local_gradient_gradient_concentration_alv(f::LayeredSphereALVFields, point...; side = :outer, layer = nothing) -> Matrix
    local_gradient_gradient_concentration_alv(sphere, ref, times, point...; side, layer) -> Matrix

Volterra block matrix (``6n\\times 6n`` in elasticity, ``3n\\times 3n`` in
conduction, time-major, canonical frame) of the pointwise concentration of a
[`LayeredSphere`](@ref): it maps the history of the remote strain (gradient)
to the history of the strain (gradient) at the point, in any layer and in the
matrix. The point is a Cartesian vector or its spherical coordinates
``(r, \\theta, \\varphi)``; `side` and `layer` have the meaning they have for the
elastic [`local_gradient_gradient_concentration`](@ref), whose tensor this is
at every instant for a non-aging problem.

In elasticity, with ``\\underline n = \\underline x/r`` and the Walpole tensors
``\\mathbb W_i(\\underline n)``,
``\\widetilde{\\mathbb A}(\\underline x) = \\sum_i \\ell_i(r)\\otimes\\mathbb W_i(\\underline n)``:
the six ``\\ell_i`` are the combinations of ``f/r``, ``f'``, ``g/r``, ``g'``, ``h/r``,
``h'`` of the elastic tensor, each an ``n\\times n`` Volterra matrix. In
conduction, ``\\widetilde{\\boldsymbol A}(\\underline x) = f'\\otimes\\underline n\\otimes\\underline n +
(f/r)\\otimes(\\boldsymbol 1 - \\underline n\\otimes\\underline n)``.
"""
function local_gradient_gradient_concentration_alv(
        f::LayeredSphereALVFields, point...; side::Symbol = :outer, layer = nothing
    )
    r, n, k = _region_at(f, point, side, layer)
    return _local_alv(f, r, n, k)
end

_flux_sign(::LayeredSphereALVFields{4}) = 1
_flux_sign(::LayeredSphereALVFields{2}) = -1

"""
    local_flux_gradient_concentration_alv(f::LayeredSphereALVFields, point...; side = :outer, layer = nothing) -> Matrix
    local_flux_gradient_concentration_alv(sphere, ref, times, point...; side, layer) -> Matrix

Volterra block matrix of the pointwise flux of a [`LayeredSphere`](@ref) per
unit remote gradient history: the relaxation operator of the region applied
**on the left** of [`local_gradient_gradient_concentration_alv`](@ref),
``\\widetilde{\\mathbb C}(\\underline x)\\circ\\widetilde{\\mathbb A}(\\underline x)``, the stress per
unit remote strain. In conduction it is the flux
``\\underline q = -k\\,\\nabla T``, ``-\\widetilde k(\\underline x)\\circ\\widetilde{\\boldsymbol A}(\\underline x)``,
as in the elastic [`local_flux_gradient_concentration`](@ref).
"""
function local_flux_gradient_concentration_alv(
        f::LayeredSphereALVFields, point...; side::Symbol = :outer, layer = nothing
    )
    r, n, k = _region_at(f, point, side, layer)
    return _flux_sign(f) * (f.relaxations[k] * _local_alv(f, r, n, k))
end

# The 3 × 6 matrix of ε ↦ ε·n on the Mandel components of ε.
function _mandel_dot(n::NTuple{3})
    s = sqrt(2 * one(eltype(n)))
    z = zero(eltype(n))
    return [
        n[1] z z z n[3] / s n[2] / s
        z n[2] z n[3] / s z n[1] / s
        z z n[3] n[2] / s n[1] / s z
    ]
end

"""
    local_displacement_alv(f::LayeredSphereALVFields, point...; side = :outer, layer = nothing) -> Matrix
    local_displacement_alv(sphere, ref, times, point...; side, layer) -> Matrix

``3n\\times 6n`` Volterra block matrix of the displacement at a point of a
[`LayeredSphere`](@ref) per unit remote strain history (Mandel components),

```math
\\underline u = \\frac{\\mathrm{tr}\\,\\boldsymbol\\varepsilon^{\\infty}}{3}\\,f\\,\\underline n
  + g\\,(\\boldsymbol\\varepsilon^{\\infty\\mathrm d}\\cdot\\underline n)
  + h\\,(\\underline n\\cdot\\boldsymbol\\varepsilon^{\\infty\\mathrm d}\\cdot\\underline n)\\,\\underline n ,
```

``f``, ``g``, ``h`` being ``n\\times n`` Volterra matrices: the operator form of the
elastic [`local_displacement`](@ref). It jumps across a
[`SpringInterface`](@ref); `side` selects the limit.
"""
function local_displacement_alv(
        f::LayeredSphereALVFields{4}, point...; side::Symbol = :outer, layer = nothing
    )
    r, n, k = _region_at(f, point, side, layer)
    p = f.profiles[k]
    f_r, _ = _mode_sum(p.f, _RADIAL_POWERS, r)
    U_r, _ = _mode_sum(p.U, _SHEAR_POWERS, r)
    g_r, _ = _mode_sum(p.W, _SHEAR_POWERS, r)
    nv = collect(n)
    tr3 = [1 1 1 0 0 0] ./ 3                       # ε̂ ↦ tr ε / 3
    dev_n = _mandel_dot(n) * (I - 3 .* transpose(tr3) * tr3)     # ε̂ ↦ εᵈ·n
    return kron(r .* f_r, nv * tr3) + kron(r .* g_r, dev_n) +
        kron(r .* (U_r ./ 2 .- g_r), nv * (transpose(nv) * dev_n))
end

"""
    local_temperature_alv(f::LayeredSphereALVFields, point...; side = :outer, layer = nothing) -> Matrix
    local_temperature_alv(sphere, ref, times, point...; side, layer) -> Matrix

``n\\times 3n`` Volterra block matrix of the temperature at a point of a
[`LayeredSphere`](@ref) per unit remote gradient history,
``T = f\\,(\\underline n\\cdot\\nabla T^{\\infty})``: the operator form of the elastic
[`local_temperature`](@ref). It jumps across a [`KapitzaInterface`](@ref);
`side` selects the limit.
"""
function local_temperature_alv(
        f::LayeredSphereALVFields{2}, point...; side::Symbol = :outer, layer = nothing
    )
    r, n, k = _region_at(f, point, side, layer)
    f_r, _ = _mode_sum(f.profiles[k].f, _RADIAL_POWERS, r)
    return kron(r .* f_r, transpose(collect(n)))
end

# One-off evaluations: each runs the recurrences again.
for name in (
        :local_gradient_gradient_concentration_alv, :local_flux_gradient_concentration_alv,
        :local_displacement_alv, :local_temperature_alv,
    )
    @eval $name(
        sphere::LayeredSphere, ref::_ALVReference, times::AbstractVector{<:Real}, point...; kw...
    ) = $name(LayeredSphereALVFields(sphere, ref, times), point...; kw...)
end
