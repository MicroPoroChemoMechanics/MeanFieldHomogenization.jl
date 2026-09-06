# =============================================================================
#  localfields.jl — pointwise reconstruction of the temperature,
#  gradient and flux fields inside/around an `N`-layer confocal
#  spheroid, from the per-layer series coefficients
#  ([`spheroid_state_sequence`](@ref)).
#
#  Scope: fields are evaluated for the two CANONICAL remote loadings
#  the transfer-matrix recurrence directly solves — axial
#  (`H = H_axial·axis`) and transverse (`H = H_trans·ê₁`, `ê₁ ⟂ axis`
#  at `φ = 0`) — and their linear superposition (exact: the axial/
#  transverse harmonic subspaces are decoupled, eq:Tsol). A general
#  in-plane direction follows by evaluating at a shifted `φ`
#  (axisymmetry: `Hᵀ = H₁ê₁ + H₂ê₂` at azimuth `φ` behaves as a single
#  transverse magnitude `√(H₁²+H₂²)` at azimuth `φ - atan2(H₂,H₁)`).
#
#  Coordinates `(q, p, φ)` are in the spheroid's OWN principal frame
#  (revolution axis ≡ local `ê₃`); results are returned in that same
#  local frame. Rotate by `s.axis` (TensND) if the caller needs the
#  global frame and `s.axis ≠ (0,0,1)`.
# =============================================================================

"""
    get_layer(s::LayeredSpheroid, q; side = :outer) -> Int

Index of the region containing the confocal coordinate `q`: `1, …, N` for
the layers, `N+1` for the surrounding matrix (`|q| > |q_N|`).

Exactly **on** an interface `|q| = |q_k|` the field has two limits and `side`
picks one — `:outer` (default) the limit from outside, region `k+1`;
`:inner` the limit from inside, region `k`. Same convention and same keyword
as [`get_layer`](@ref)`(::LayeredSphere, r)`; it matters for a
[`KapitzaInterface`](@ref), across which the temperature itself jumps.
"""
function get_layer(s::LayeredSpheroid{T, N}, q; side::Symbol = :outer) where {T, N}
    side === :outer || side === :inner ||
        throw(ArgumentError("get_layer: `side` must be :outer or :inner, got $(side)"))
    # Same refusal as the sphere: an undecidable comparison would land silently
    # in region 1 rather than raise.  The predicate is on `abs(q)`, NOT on `q`:
    # an OBLATE spheroid carries a genuinely COMPLEX confocal coordinate, whose
    # modulus is an ordinary real and orders perfectly well — which is exactly
    # what the loop below compares.  Guarding `typeof(q)` instead would refuse
    # every oblate particle.
    is_hard_numeric(typeof(abs(q))) || throw(
        ArgumentError(
            "get_layer: a symbolic coordinate cannot be located by comparison " *
                "(got $(typeof(q)))"
        )
    )
    k = 1
    if side === :outer
        while k ≤ N && abs(q) ≥ abs(s.q[k])
            k += 1
        end
    else
        while k ≤ N && abs(q) > abs(s.q[k])
            k += 1
        end
    end
    return k
end

@inline function _base_fond(q::Qx, p, φ) where {Qx}
    qb = sqrt(q^2 - 1)
    pb = sqrt(1 - p^2)
    qp = sqrt(q^2 - p^2)
    eR = (cos(φ), sin(φ), zero(φ))
    ephi = (-sin(φ), cos(φ), zero(φ))
    ez = (zero(φ), zero(φ), one(φ))
    e_p = ((-p * qb) .* eR .+ (q * pb) .* ez) ./ qp
    e_q = ((pb * q) .* eR .+ (p * qb) .* ez) ./ qp
    return e_q, e_p, ephi
end

@inline function _metric(c::Qx, q, p) where {Qx}
    qb = sqrt(q^2 - 1)
    pb = sqrt(1 - p^2)
    qp = sqrt(q^2 - p^2)
    return c * qp / qb, c * qp / pb, c * qb * pb   # h_q, h_p, h_φ
end

"""
    _spheroid_field_coeffs(s, k₀, q, p; N_only = nothing)
        -> (layer, k_layer, Aa, Ba, At, Bt)

The layer containing `q`, its conductivity (matrix `k₀` if `q` lies
outside the spheroid), and the axial/transverse coefficient
sub-vectors valid there.
"""
function _spheroid_field_coeffs(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    Xa = spheroid_state_sequence(s, k₀, false)
    Xt = spheroid_state_sequence(s, k₀, true)
    return Xa, Xt
end

"""
    local_temperature(s, k₀, q, p, φ; H_axial = 1.0, H_trans = 0.0) -> T

Temperature at the spheroidal point `(q, p, φ)` (own frame) under a
remote gradient `H_axial·axis + H_trans·ê₁` (superposition of the
axial and transverse canonical problems, eq:Taxi/eq:Ttrans).
"""
function local_temperature(
        s::LayeredSpheroid{T, N}, k₀, q, p, φ; H_axial = 1.0, H_trans = 0.0,
    ) where {T, N}
    Xa, Xt = _spheroid_field_coeffs(s, k₀)
    return _local_T_spheroid(s, Xa, Xt, q, p, φ, H_axial, H_trans)
end

function _local_T_spheroid(
        s::LayeredSpheroid{T, N}, Xa, Xt, q, p, φ, H_axial, H_trans
    ) where {T, N}
    𝒩 = s.Nseries
    lay = get_layer(s, q)
    Aa, Ba = Xa[lay][1:𝒩], Xa[lay][(𝒩 + 1):(2𝒩)]
    At, Bt = Xt[lay][1:𝒩], Xt[lay][(𝒩 + 1):(2𝒩)]

    P0p, _ = legendre_odd(:P0, p, 𝒩)
    P0q, _ = legendre_odd(:P0, q, 𝒩)
    Q0q, _ = legendre_odd(:Q0, q, 𝒩)
    P1p, _ = legendre_odd(:P1p, p, 𝒩)
    P1q, _ = legendre_odd(:P1, q, 𝒩)
    Q1q, _ = legendre_odd(:Q1, q, 𝒩)

    Ta = s.c * sum(P0p[r] * (Aa[r] * P0q[r] + Ba[r] * Q0q[r]) for r in 1:𝒩)
    Tt = s.c * sum(P1p[r] * (At[r] * P1q[r] + Bt[r] * Q1q[r]) for r in 1:𝒩)
    return real(H_axial * Ta + H_trans * Tt * cos(φ))
end

"""
    local_gradient(s, k₀, q, p, φ; H_axial = 1.0, H_trans = 0.0) -> (g₁, g₂, g₃)

Temperature gradient `∇T` at `(q, p, φ)` (own frame, real Cartesian
triple), under the same remote loading as [`local_temperature`](@ref).

Valid on the revolution axis (`|p| = 1`) as well, where the `(q, p, φ)` chart
itself degenerates: the two `0/0` the naive expression carries there are
removed exactly, using `P¹ₙ = -√(1-p²) P′ₙ` and the Legendre equation, so the
value on the axis is the limit of the values around it and — as it must be —
independent of the azimuth, which is undefined there.
"""
function local_gradient(
        s::LayeredSpheroid{T, N}, k₀, q, p, φ; H_axial = 1.0, H_trans = 0.0,
    ) where {T, N}
    Xa, Xt = _spheroid_field_coeffs(s, k₀)
    return _local_grad_spheroid(s, Xa, Xt, q, p, φ, H_axial, H_trans)
end

function _local_grad_spheroid(
        s::LayeredSpheroid{T, N}, Xa, Xt, q, p, φ, H_axial, H_trans
    ) where {T, N}
    𝒩 = s.Nseries
    lay = get_layer(s, q)
    Aa, Ba = Xa[lay][1:𝒩], Xa[lay][(𝒩 + 1):(2𝒩)]
    At, Bt = Xt[lay][1:𝒩], Xt[lay][(𝒩 + 1):(2𝒩)]

    P0p, dP0p = legendre_odd(:P0, p, 𝒩)
    P0q, dP0q = legendre_odd(:P0, q, 𝒩)
    Q0q, dQ0q = legendre_odd(:Q0, q, 𝒩)
    P1p, _ = legendre_odd(:P1p, p, 𝒩)
    P1q, dP1q = legendre_odd(:P1, q, 𝒩)
    Q1q, dQ1q = legendre_odd(:Q1, q, 𝒩)

    dTa_dq = s.c * sum(P0p[r] * (Aa[r] * dP0q[r] + Ba[r] * dQ0q[r]) for r in 1:𝒩)
    dTa_dp = s.c * sum(dP0p[r] * (Aa[r] * P0q[r] + Ba[r] * Q0q[r]) for r in 1:𝒩)
    # The transverse radial factor, shared by all three transverse terms.
    Wt = ntuple(r -> At[r] * P1q[r] + Bt[r] * Q1q[r], 𝒩)
    dTt_dq = s.c * sum(P1p[r] * (At[r] * dP1q[r] + Bt[r] * dQ1q[r]) for r in 1:𝒩)

    e_q, e_p, e_φ = _base_fond(q, p, φ)
    qb = sqrt(q^2 - 1)
    pb = sqrt(1 - p^2)
    qp = sqrt(q^2 - p^2)
    h_q = s.c * qp / qb
    # `∂T/∂p / h_p` is written as `∂T/∂p · p̄ / (c q̄ₚ)`: algebraically the same
    # thing, since `h_p = c q̄ₚ / p̄`, but regular on the revolution axis.  There
    # `p̄ = 0`, so `h_p` is infinite; dividing a REAL number by `Inf` gives the
    # correct `0`, but an OBLATE spheroid carries a complex `q`, and Julia's
    # complex division `z / (Inf + 0im)` returns `NaN + NaN im`.  That is what
    # made every on-axis evaluation of an oblate particle a silent `NaN`.
    inv_hp = pb / (s.c * qp)

    g_axial = (dTa_dq / h_q) .* e_q .+ (dTa_dp * inv_hp) .* e_p

    # ── The transverse part, regular on the revolution axis ─────────────────
    #
    # At `|p| = 1` the chart degenerates: `h_φ = c q̄ p̄` vanishes and the
    # azimuth is undefined.  Written naively the transverse field carries two
    # `0/0` there, `T_t / h_φ` and `∂T_t/∂p · p̄`.  Both are removable exactly,
    # with no asymptotic expansion, because the `P¹` table is seeded
    # `P₁¹(p) = -√(1-p²)` — the Condon-Shortley convention
    #
    #     P¹ₙ(p) = -p̄ P′ₙ(p)
    #
    # — so the `p̄` of the numerator cancels the `p̄` of `h_φ` identically; and
    # because the Legendre equation `(1-p²) P″ₙ = 2p P′ₙ - n(n+1) Pₙ` removes
    # the second derivative from the other term:
    #
    #     T_t / h_φ           = -(1/q̄)  Σ_r P′ₙ(p) W_r,
    #     ∂T_t/∂p · p̄/(c q̄ₚ) =  (1/q̄ₚ) Σ_r [n(n+1) Pₙ(p) - p P′ₙ(p)] W_r.
    #
    # Both need only the ORDINARY Legendre table, already computed above, and
    # both are finite at every `p`.  The rearrangement is not merely tidier: on
    # the axis it is what makes the answer independent of the azimuth `φ`,
    # which is undefined there.  `P′ₙ(±1) = ±ⁿ⁺¹ n(n+1)/2` makes the two
    # coefficients exactly opposite, and the `e_q` term vanishes because
    # `P¹ₙ(±1) = 0`; a formulation that got either factor wrong would return a
    # field on the axis that depended on how the point was addressed.
    tt_over_hφ = -sum(dP0p[r] * Wt[r] for r in 1:𝒩) / qb
    dtt_dp_over_hp = sum(
        ((2r - 1) * 2r * P0p[r] - p * dP0p[r]) * Wt[r] for r in 1:𝒩
    ) / qp

    g_trans_r = (dTt_dq / h_q) .* e_q .+ dtt_dp_over_hp .* e_p
    g_trans = cos(φ) .* g_trans_r .- (sin(φ) * tt_over_hφ) .* e_φ

    g = H_axial .* g_axial .+ H_trans .* g_trans
    return real(g[1]), real(g[2]), real(g[3])
end

"""
    local_flux(s, k₀, q, p, φ; H_axial = 1.0, H_trans = 0.0) -> (u₁, u₂, u₃)

Heat/mass flux `u = -k(x)·∇T` at `(q, p, φ)` (own frame), `k(x)` the
conductivity of the layer containing the point (or `k₀` outside the
spheroid).
"""
function local_flux(
        s::LayeredSpheroid{T, N}, k₀, q, p, φ; H_axial = 1.0, H_trans = 0.0,
    ) where {T, N}
    Xa, Xt = _spheroid_field_coeffs(s, k₀)
    return _local_flux_spheroid(s, k₀, Xa, Xt, q, p, φ, H_axial, H_trans)
end

function _local_flux_spheroid(
        s::LayeredSpheroid{T, N}, k₀, Xa, Xt, q, p, φ, H_axial, H_trans
    ) where {T, N}
    lay = get_layer(s, q)
    k_layers = _spheroid_layer_moduli(s)
    k_here = lay ≤ N ? k_layers[lay] : _as_scalar_k(k₀)
    g1, g2, g3 = _local_grad_spheroid(s, Xa, Xt, q, p, φ, H_axial, H_trans)
    return -k_here * g1, -k_here * g2, -k_here * g3
end


# =============================================================================
#  Harmonization with `LayeredSphere` — a cached solution object, a remote
#  gradient given as a VECTOR, and the four `local_*_*_loc` couplings.
#
#  The sphere and the spheroid solve different harmonic problems and will
#  never share an implementation, but they answer the same questions and
#  should be asked them the same way.  What follows mirrors, name for name,
#  `LayeredSpheres/localfields.jl`; the canonical axial/transverse entry
#  points above stay exactly as they were.
# =============================================================================

"""
    LayeredSpheroidTransportFields(s, k₀)

Precomputed pointwise transport solution of a [`LayeredSpheroid`](@ref) in
the isotropic matrix `k₀`: the axial and transverse confocal-harmonic
coefficient sequences, solved once.

Build it once and pass it wherever `(s, k₀)` is accepted. The `(s, k₀)`
forms re-run **both** state sequences on every call, which a field map over
thousands of points pays for in full.

Transport twin of
[`LayeredSphereTransportFields`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphereTransportFields).
"""
struct LayeredSpheroidTransportFields{S, K, XA, XT}
    spheroid::S
    k₀::K
    Xa::XA
    Xt::XT
end

function LayeredSpheroidTransportFields(s::LayeredSpheroid, k₀)
    Xa, Xt = _spheroid_field_coeffs(s, k₀)
    return LayeredSpheroidTransportFields(s, k₀, Xa, Xt)
end

get_layer(f::LayeredSpheroidTransportFields, q; side::Symbol = :outer) =
    get_layer(f.spheroid, q; side)

local_temperature(
    f::LayeredSpheroidTransportFields, q, p, φ; H_axial = 1.0, H_trans = 0.0
) = _local_T_spheroid(f.spheroid, f.Xa, f.Xt, q, p, φ, H_axial, H_trans)

local_gradient(
    f::LayeredSpheroidTransportFields, q, p, φ; H_axial = 1.0, H_trans = 0.0
) = _local_grad_spheroid(f.spheroid, f.Xa, f.Xt, q, p, φ, H_axial, H_trans)

local_flux(
    f::LayeredSpheroidTransportFields, q, p, φ; H_axial = 1.0, H_trans = 0.0
) = _local_flux_spheroid(f.spheroid, f.k₀, f.Xa, f.Xt, q, p, φ, H_axial, H_trans)

"""
    _split_remote_gradient(G) -> (H_axial, H_trans, φ₀)

Decompose a remote gradient given in the spheroid's own frame into the two
canonical loadings the recurrence solves: the axial magnitude `G₃`, the
transverse magnitude `√(G₁²+G₂²)`, and the azimuth `φ₀ = atan(G₂, G₁)` the
transverse one points along.

By axisymmetry the transverse solution for a general in-plane direction is
the canonical `ê₁` one evaluated at the shifted azimuth `φ − φ₀` and rotated
back by `φ₀` — which is also true, vacuously, of the axial part, so a single
shifted evaluation carries both.
"""
@inline function _split_remote_gradient(G)
    g = TensND._extract_vec(G)
    return g[3], hypot(g[1], g[2]), atan(g[2], g[1])
end

@inline _rotz(v, φ₀) = (
    cos(φ₀) * v[1] - sin(φ₀) * v[2],
    sin(φ₀) * v[1] + cos(φ₀) * v[2],
    v[3],
)

"""
    local_temperature(s_or_fields, [k₀,] q, p, φ, ∇T∞) -> Number

Temperature at `(q, p, φ)` under a remote uniform gradient `∇T∞` given as a
**vector in the spheroid's own frame** (revolution axis ≡ local `ê₃`).

Vector form of the axial/transverse entry point above, harmonized with
[`local_temperature`](@ref MeanFieldHomogenization.LayeredSpheres.local_temperature)`(::LayeredSphereTransportFields, x, ∇T∞)`.
"""
local_temperature(s::LayeredSpheroid, k₀, q, p, φ, ∇T∞) =
    local_temperature(LayeredSpheroidTransportFields(s, k₀), q, p, φ, ∇T∞)

function local_temperature(f::LayeredSpheroidTransportFields, q, p, φ, ∇T∞)
    Ha, Ht, φ₀ = _split_remote_gradient(∇T∞)
    return local_temperature(f, q, p, φ - φ₀; H_axial = Ha, H_trans = Ht)
end

"""
    local_gradient(s_or_fields, [k₀,] q, p, φ, ∇T∞) -> NTuple{3}

Temperature gradient at `(q, p, φ)` under a remote uniform gradient `∇T∞`
given as a vector in the spheroid's own frame; the result is a Cartesian
triple in that same frame.
"""
local_gradient(s::LayeredSpheroid, k₀, q, p, φ, ∇T∞) =
    local_gradient(LayeredSpheroidTransportFields(s, k₀), q, p, φ, ∇T∞)

function local_gradient(f::LayeredSpheroidTransportFields, q, p, φ, ∇T∞)
    Ha, Ht, φ₀ = _split_remote_gradient(∇T∞)
    return _rotz(local_gradient(f, q, p, φ - φ₀; H_axial = Ha, H_trans = Ht), φ₀)
end

"""
    local_flux(s_or_fields, [k₀,] q, p, φ, ∇T∞) -> NTuple{3}

Flux `u = −k(x)·∇T` at `(q, p, φ)` under a remote uniform gradient `∇T∞`
given as a vector in the spheroid's own frame.
"""
local_flux(s::LayeredSpheroid, k₀, q, p, φ, ∇T∞) =
    local_flux(LayeredSpheroidTransportFields(s, k₀), q, p, φ, ∇T∞)

function local_flux(f::LayeredSpheroidTransportFields, q, p, φ, ∇T∞)
    Ha, Ht, φ₀ = _split_remote_gradient(∇T∞)
    return _rotz(local_flux(f, q, p, φ - φ₀; H_axial = Ha, H_trans = Ht), φ₀)
end

"""
    local_gradient_gradient_loc(s_or_fields, [k₀,] q, p, φ) -> Tens{2,3}

Pointwise gradient-gradient localization `𝐀(q,p,φ)`, so that
`∇T = 𝐀 · ∇T∞` for any remote uniform gradient, in the spheroid's own frame.

Unlike the sphere's, this tensor is **not** transversely isotropic about the
local radial direction — a confocal spheroid is not rotation-invariant about
the field point — so it comes back as a general `Tens{2,3}`, built column by
column from the three canonical remote gradients.

Spheroid twin of
[`local_gradient_gradient_loc`](@ref MeanFieldHomogenization.LayeredSpheres.local_gradient_gradient_loc).
"""
local_gradient_gradient_loc(s::LayeredSpheroid, k₀, q, p, φ) =
    local_gradient_gradient_loc(LayeredSpheroidTransportFields(s, k₀), q, p, φ)

function local_gradient_gradient_loc(f::LayeredSpheroidTransportFields, q, p, φ)
    cols = ntuple(j -> local_gradient(f, q, p, φ, ntuple(i -> i == j ? 1.0 : 0.0, 3)), 3)
    return TensND.Tens([cols[j][i] for i in 1:3, j in 1:3])
end

"""
    local_flux_gradient_loc(s_or_fields, [k₀,] q, p, φ) -> Tens{2,3}

Pointwise flux-gradient localization `−k(x) 𝐀(q,p,φ)`: `q = ... · ∇T∞` with
the Fourier / Fick sign convention.
"""
local_flux_gradient_loc(s::LayeredSpheroid, k₀, q, p, φ) =
    local_flux_gradient_loc(LayeredSpheroidTransportFields(s, k₀), q, p, φ)

function local_flux_gradient_loc(f::LayeredSpheroidTransportFields, q, p, φ)
    s = f.spheroid
    N = layer_count(s)
    lay = get_layer(s, q)
    k_here = lay ≤ N ? _spheroid_layer_moduli(s)[lay] : _as_scalar_k(f.k₀)
    return -k_here * local_gradient_gradient_loc(f, q, p, φ)
end

"""
    local_gradient_flux_loc(s_or_fields, [k₀,] q, p, φ) -> Tens{2,3}

Pointwise gradient-flux localization `𝐀 · K₀⁻¹`: the local gradient under a
remote uniform **flux** `q∞`.
"""
local_gradient_flux_loc(s::LayeredSpheroid, k₀, q, p, φ) =
    local_gradient_flux_loc(LayeredSpheroidTransportFields(s, k₀), q, p, φ)

local_gradient_flux_loc(f::LayeredSpheroidTransportFields, q, p, φ) =
    -local_gradient_gradient_loc(f, q, p, φ) / _as_scalar_k(f.k₀)

"""
    local_flux_flux_loc(s_or_fields, [k₀,] q, p, φ) -> Tens{2,3}

Pointwise flux-flux localization: the local flux under a remote uniform
**flux** `q∞`.
"""
local_flux_flux_loc(s::LayeredSpheroid, k₀, q, p, φ) =
    local_flux_flux_loc(LayeredSpheroidTransportFields(s, k₀), q, p, φ)

local_flux_flux_loc(f::LayeredSpheroidTransportFields, q, p, φ) =
    -local_flux_gradient_loc(f, q, p, φ) / _as_scalar_k(f.k₀)
