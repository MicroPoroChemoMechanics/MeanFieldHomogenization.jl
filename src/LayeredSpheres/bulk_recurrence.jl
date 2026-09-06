# =============================================================================
#  bulk_recurrence.jl — isotropic `n`-layer spherical inclusion, bulk
#  state-vector recurrence for the layered-sphere Eshelby problem.
#
#  Under a remote strain `ε∞ = ε_v J + ε_d K` the displacement field
#  decouples into
#
#   1. a **spherical (bulk) part**  `u_r(r) = A_k r + B_k / r²`,
#      propagated via a 2×2 transfer on the state vector
#      `s(r) = (u_r, σ_rr)`.  This formulation is regular when
#      `κ → ∞` (incompressible phase) — see below.
#
#   2. a **deviatoric (shear) part** with 4 elementary modes per layer;
#      the `L(r; κ, μ)` fundamental matrix connects the four
#      amplitudes `(a, b, c, d)` to the 4-component state vector
#      `(U, V, σ_rr, σ_rθ)`.  A 4×4 transfer propagates across
#      interfaces (perfect + imperfect).
#
#  Incompressibility-robust formulation
#  ------------------------------------
#  The intra-layer transfer  `s(r_out) = T · s(r_in)` for a layer
#  `(κ, μ)` is derived from `M(r_out) · M(r_in)⁻¹` where
#  `M(r) = [r 1/r²; 3κ -4μ/r³]`.  The resulting 2×2 matrix factorizes
#  via the bounded ratios `α = 4μ/(3κ+4μ) ∈ [0,1]` and
#  `β = 3κ/(3κ+4μ) ∈ [0,1]`, so every element stays finite as
#  `κ → ∞` (or `μ → ∞`).  The entry-point at `r = 0⁺` of the core
#  layer is written with the "pressure amplitude" `P_1 = 3κ_1 A_1` so
#  that `u(r_1⁻) = (r_1 / (3κ_1)) P_1` tends to zero for an
#  incompressible core while `σ(r_1⁻) = P_1` stays finite.
# =============================================================================

# ── ISO moduli extraction from TensISO stiffness tensor ─────────────────────

"""
    _iso_bulk_shear(C) -> (κ, μ)

Extract (bulk modulus, shear modulus) from an isotropic stiffness
`TensISO{4,3}`.  TensND stores `(3κ, 2μ)` internally.
"""
function _iso_bulk_shear(C::TensND.TensISO{4, 3})
    α, β = C.data
    return α / 3, β / 2
end

"""
    _iso_scalar(K) -> k

Extract the scalar conductivity from an isotropic `TensISO{2,3}`.
"""
_iso_scalar(K::TensND.TensISO{2, 3}) = MFH_Core.extract_iso_conductivity(K)

# =============================================================================
#  Bulk (spherical) part — 2×2 transfer matrix in the (u, σ) state vector.
# =============================================================================

"""
    _bulk_layer_transfer(r_out, r_in, κ, μ) -> Matrix(2×2)

Intra-layer transfer matrix propagating the state vector
`s = (u_r, σ_rr)` from radius `r_in` to `r_out` in an isotropic layer
`(κ, μ)`.  Regular in the incompressibility limit `κ → ∞` (the matrix
elements are written in the factorized `α, β` form with `α + β = 1`).
"""
@inline function _bulk_layer_transfer(r_out, r_in, κ, μ)
    T = promote_type(typeof(r_out), typeof(r_in), typeof(κ), typeof(μ))
    Tκ = T(κ); Tμ = T(μ)
    S = 3 * Tκ + 4 * Tμ
    α = 4 * Tμ / S
    β = 3 * Tκ / S

    ro = T(r_out); ri = T(r_in)
    ρ = ri / ro               # ≤ 1 (inner over outer)
    ρ² = ρ * ρ
    ρ³ = ρ² * ρ
    ro_ri = ro / ri               # ≥ 1 — reciprocal of ρ (stored, not divided)
    ri_inv = 1 / ri
    # M[2,1] coefficient:  4μ β (1/ri − ri²/ro³) = (4μ β / ri) · (1 − ρ³).
    fourμβ_over_ri = 4 * Tμ * β * ri_inv
    # M[1,2] coefficient:  (ro − ri³/ro²) / S = (ro / S) · (1 − ρ³).
    ro_over_S = ro / S

    one_minus_ρ³ = one(T) - ρ³
    M = Matrix{T}(undef, 2, 2)
    M[1, 1] = α * ro_ri + β * ρ²
    M[1, 2] = ro_over_S * one_minus_ρ³
    M[2, 1] = fourμβ_over_ri * one_minus_ρ³
    M[2, 2] = α * ρ³ + β
    return M
end

"""
    _bulk_seed_state(r_1, κ_1) -> Vector(2)

Seed state vector at `r = r_1⁻` for layer 1 under a unit "pressure
amplitude" `P_1 = 1`:  `u_r = r_1 / (3κ_1)`, `σ_rr = 1`.  Finite for any
`κ_1` including `κ_1 = ∞` (gives `u = 0`).
"""
@inline function _bulk_seed_state(r_1, κ_1)
    T = promote_type(typeof(r_1), typeof(κ_1))
    return T[T(r_1) / (3 * T(κ_1)), one(T)]
end

"""
    _bulk_extract_AB(r, κ, μ, u, σ) -> (A, B)

Given the layer moduli `(κ, μ)` and a state `(u, σ)` at radius `r`,
return the coefficients `(A, B)` of the layer's local expansion
`u_r(ρ) = A ρ + B / ρ²`.  Regular for `κ → ∞` (`A → 0`,
`B → u r²`).
"""
@inline function _bulk_extract_AB(r, κ, μ, u, σ)
    T = promote_type(typeof(r), typeof(κ), typeof(μ), typeof(u), typeof(σ))
    S = 3 * T(κ) + 4 * T(μ)
    A = (4 * T(μ) * T(u) + T(r) * T(σ)) / (T(r) * S)
    B = (3 * T(κ) * T(u) - T(r) * T(σ)) * T(r)^2 / S
    return A, B
end

"""
    _bulk_state_seq(sphere, κ₀, μ₀) -> NTuple{N, (u, σ)⁻}, (u, σ)⁺_N

Propagate the bulk state vector from the core outward.  Returns the
sequence of states at each interface `r_k` **on the inside** (layer
side) for every `k = 1..N`, and the state on the **outside** of `r_N`
(matrix side).  Interface jumps are applied via
`_bulk_interface_T(interface, κ, μ, r)` (defined in
`interface_transfer.jl`).  The seed state corresponds to a unit
pressure amplitude `P_1 = 1`.
"""
# Cached tuple of per-layer `(κ, μ)` moduli pairs.
@inline function _bulk_layer_moduli(sphere::LayeredSphere{T, N}) where {T, N}
    return ntuple(k -> _iso_bulk_shear(layer_modulus(sphere, k)), Val(N))
end

# `κμ` is deliberately typed `Tuple{Vararg{Any, N}}` and NOT `NTuple{N, <:Any}`:
# the latter is `Tuple{S, S, …}` with ONE element type, so it fails to match the
# moment the per-layer moduli differ in type — which is precisely what
# `ForwardDiff` produces when a single layer's modulus is the differentiation
# variable and the others stay `Float64`. Before this signature was widened,
# `ForwardDiff.derivative(p -> strain_strain_loc(sphere_with_one_Dual_layer, …), p)`
# died in a `MethodError`, while making *every* layer dual worked — an AD gap
# invisible to any test that differentiates the whole stack at once.
@inline function _bulk_promote(
        ::LayeredSphere{T, N}, κμ::Tuple{Vararg{Any, N}},
        κ₀, μ₀
    ) where {T, N}
    return promote_type(
        T, typeof(κ₀), typeof(μ₀),
        ntuple(k -> typeof(κμ[k][1]), N)...,
        ntuple(k -> typeof(κμ[k][2]), N)...
    )
end

function _bulk_state_seq(sphere::LayeredSphere{T, N}, κ₀, μ₀) where {T, N}
    κμ = _bulk_layer_moduli(sphere)
    TP = _bulk_promote(sphere, κμ, κ₀, μ₀)
    radii = sphere.radii

    κ_1, _ = κμ[1]
    s = _bulk_seed_state(TP(radii[1]), TP(κ_1))

    inside_states = Vector{Vector{TP}}(undef, N)
    for k in 1:N
        inside_states[k] = copy(s)
        intf = layer_interface(sphere, k)
        (κk, μk) = κμ[k]
        Tint = _bulk_interface_T(intf, TP(κk), TP(μk), TP(radii[k]))
        s = Tint * s
        if k < N
            (κk1, μk1) = κμ[k + 1]
            Tlay = _bulk_layer_transfer(
                TP(radii[k + 1]), TP(radii[k]),
                TP(κk1), TP(μk1)
            )
            s = Tlay * s
        end
    end
    return inside_states, s
end

"""
    _bulk_localization(sphere, κ₀, μ₀) -> NTuple{N, TP}

Per-layer bulk localization `α_k = A_k / A_∞` for the composite
sphere.  Regular in the limit `κ_k → ∞` (gives `α_k → 0`, no
volumetric strain in incompressible layer).
"""
function _bulk_localization(sphere::LayeredSphere{T, N}, κ₀, μ₀) where {T, N}
    κμ = _bulk_layer_moduli(sphere)
    TP = _bulk_promote(sphere, κμ, κ₀, μ₀)
    inside_states, s_matrix = _bulk_state_seq(sphere, κ₀, μ₀)
    radii = sphere.radii

    A_inf, _ = _bulk_extract_AB(
        TP(radii[N]), TP(κ₀), TP(μ₀),
        s_matrix[1], s_matrix[2]
    )
    inv_A_inf = one(TP) / A_inf

    return ntuple(N) do k
        (κk, μk) = κμ[k]
        (A_k, _) = _bulk_extract_AB(
            TP(radii[k]), TP(κk), TP(μk),
            inside_states[k][1], inside_states[k][2]
        )
        A_k * inv_A_inf
    end
end

"""
    _effective_bulk(sphere, κ₀, μ₀) -> κ_eff

Effective bulk modulus of the composite sphere under hydrostatic
loading:  `κ_eff = Σ_k f_k κ_k α_k` with
`f_k = (r_k³ - r_{k-1}³) / r_N³`.
"""
function _effective_bulk(sphere::LayeredSphere{T, N}, κ₀, μ₀) where {T, N}
    α = _bulk_localization(sphere, κ₀, μ₀)
    κμ = _bulk_layer_moduli(sphere)
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    return sum(f[k] * κμ[k][1] * α[k] for k in 1:N)
end

# =============================================================================
#  Deviatoric (shear) part — full 4×4 recurrence lives in
#  `shear_recurrence.jl`, including both the N=1 fast path (delegation to
#  the `Ellipsoid` Eshelby machinery) and the N≥2 state-vector recurrence.
# =============================================================================
