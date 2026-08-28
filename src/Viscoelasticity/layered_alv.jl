# =============================================================================
#  layered_alv.jl — n-layer composite sphere in an ALV matrix.
#
#  Extends the elastic Hervé-Zaoui recurrence
#  ([@LayeredSpheres/bulk_recurrence.jl]) to the ageing linear
#  viscoelastic setting by replacing every scalar modulus (κ, μ) with
#  its `(n×n)` trapezoidal Volterra matrix.  Each scalar transfer-
#  matrix entry of the elastic 2×2 transfer becomes a Volterra
#  combination of those `n×n` matrices ; the full bulk transfer is
#  thus a `(2n × 2n)` block matrix.
#
#  Reference : Hervé & Zaoui 1993 (elastic) ; ECHOES manual ch07
#  §"n-layer ALV" ; sanahuja2013.
#
#  This file delivers BOTH the **bulk (Y₀ harmonic)** and the
#  **shear (Y₂ harmonic)** ALV recurrences, plus the corresponding
#  per-layer localization matrices and the ALV interface transfers.
#  Both are exercised (incl. an N=2 cross-check against ECHOES Python)
#  in `test/Viscoelasticity/test_layered_alv.jl`.
# =============================================================================

# Imports for `LayeredSphere`, `layer_modulus`, `layer_interface`,
# `PerfectInterface`, etc. are pulled in by the module root
# (`Viscoelasticity.jl`).

# ── Helpers : extract scalar (κ, μ) Volterra matrices for each layer ────────

"""
    _ALVReference

Reference medium a layered-sphere ALV kernel is evaluated against:
either the matrix `ViscoLaw` — the usual case, where the reference is
fixed and known symbolically — or an already-discretized `(6n × 6n)`
relaxation matrix.  The latter is what the self-consistent and
differential schemes need: their reference is the *running* effective
medium, which exists only as a block matrix.  Same role as the `_at`
suffix on the crack kernels (`stiffness_contribution_alv_at`).
"""
const _ALVReference = Union{ViscoLaw, AbstractMatrix}

"""
    _bulk_layer_moduli_alv(sphere, C0_ref, times) -> (NTuple{N,(M_κ, M_μ)}, M_κ_0, M_μ_0)

Trapezoidal `(n×n)` matrices of `κ(t,t')` and `μ(t,t')` for every
layer of `sphere` plus the matrix reference `C0_ref` (whose iso
parameters are extracted from its trapezoidal block matrix, or read
directly off it when the reference is already a block matrix).

The per-layer kernels must be `ViscoLaw`s returning iso 4-tensors
(`TensISO{4,3}`) or stored as elastic `TensISO{4,3}` (auto-wrapped in a
Heaviside law).
"""
function _bulk_layer_moduli_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    n = length(times)
    # Matrix kernel : iso scalar matrices.
    R0 = C0_law isa ViscoLaw ? trapezoidal_matrix(C0_law, times) : C0_law
    α0, β0 = iso_params_from_blocks(R0)
    M_κ_0 = α0 ./ 3
    M_μ_0 = β0 ./ 2

    # Per-layer kernels.
    layers = ntuple(k -> _layer_iso_volterra(layer_modulus(sphere, k), times), N)
    return layers, M_κ_0, M_μ_0
end

# Convert a per-layer modulus value (either a `TensISO{4,3}` or a
# `ViscoLaw` returning `TensISO{4,3}`) to scalar `(M_κ, M_μ)` n×n
# Volterra matrices.  An elastic `TensISO{4,3}` is implicitly wrapped
# in a Heaviside law.
function _layer_iso_volterra(C, times::AbstractVector{<:Real})
    law = C isa ViscoLaw ? C : heaviside_law(C)
    R = trapezoidal_matrix(law, times)
    α, β = iso_params_from_blocks(R)
    return (α ./ 3, β ./ 2)   # (M_κ, M_μ)
end

# ── Interface parameter promotion (scalar OR ViscoLaw) ─────────────────────

"""
    _iface_param_volterra(p, times, n) -> Matrix{T}

Promote an interface parameter to its `n × n` Volterra block.  Scalar
parameters (constant in time, the elastic limit) become `p · I_n`;
genuinely viscoelastic parameters (`p::ViscoLaw` returning a scalar)
become their trapezoidal `n × n` matrix.  Lets every interface model
stack on top of an ageing matrix without code duplication.
"""
function _iface_param_volterra(p::Real, times::AbstractVector, n::Int)
    T = typeof(p)
    return p * Matrix{T}(I, n, n)
end

function _iface_param_volterra(p::ViscoLaw, times::AbstractVector, n::Int)
    return trapezoidal_matrix(p, times)
end

# ── Bulk (2n × 2n, mode-major) ALV interface transfer matrices ─────────────
#
# State vector (mode-major) : s = (u_r-block ; σ_rr-block) of size 2n × m.
# Mirror of `LayeredSpheres._bulk_interface_T` with each scalar entry of
# the elastic 2×2 jump matrix promoted to an n × n Volterra block.

"""
    _bulk_interface_T_alv(intf, M_κ, M_μ, r, times, n) -> Matrix{T}

`(2n × 2n)` jump matrix for the **bulk** (Y₀-harmonic) ALV state at
the interface of type `intf` located at radius `r`.  The adjacent
layer Volterra moduli `(M_κ, M_μ)` are passed in for type promotion.

Supports the same interface types as the elastic counterpart:
[`PerfectInterface`](@ref), [`SpringInterface`](@ref) (primal,
displacement jump driven by `kn`), and [`MembraneInterface`](@ref)
(dual, traction jump driven by `κs`).  Each elastic scalar parameter
may also be a [`ViscoLaw`](@ref) — in that case the jump is itself
ageing and the corresponding block is the parameter's trapezoidal
matrix.
"""
function _bulk_interface_T_alv(
        ::PerfectInterface, M_κ, M_μ, r,
        times::AbstractVector, n::Int
    )
    T = promote_type(eltype(M_κ), eltype(M_μ))
    return Matrix{T}(I, 2 * n, 2 * n)
end

function _bulk_interface_T_alv(
        intf::SpringInterface, M_κ, M_μ, r,
        times::AbstractVector, n::Int
    )
    M_kn = _iface_param_volterra(intf.kn, times, n)
    T = promote_type(eltype(M_κ), eltype(M_μ), eltype(M_kn))
    Id = Matrix{T}(I, n, n)
    M = zeros(T, 2 * n, 2 * n)
    M[1:n, 1:n] = Id
    M[1:n, (n + 1):(2n)] = T.(M_kn)
    M[(n + 1):(2n), (n + 1):(2n)] = Id
    return M
end

function _bulk_interface_T_alv(
        intf::MembraneInterface, M_κ, M_μ, r,
        times::AbstractVector, n::Int
    )
    M_κs = _iface_param_volterra(intf.κs, times, n)
    T = promote_type(eltype(M_κ), eltype(M_μ), eltype(M_κs))
    Id = Matrix{T}(I, n, n)
    fourκs_over_r² = T(4 / (r * r))
    M = zeros(T, 2 * n, 2 * n)
    M[1:n, 1:n] = Id
    M[(n + 1):(2n), 1:n] = fourκs_over_r² .* T.(M_κs)
    M[(n + 1):(2n), (n + 1):(2n)] = Id
    return M
end

# ── Closed-form bulk (Y₀) transition at an interface ───────────────────────
#
# At an interface located at radius R between adjacent layers (κ_a, μ_a)
# and (κ_b, μ_b), the bulk amplitudes (A, B) — coefficients of the two
# fundamental modes `u_r = A r` and `u_r = B / r²` — transform as
#     [A_b]                            [A_a]
#     [B_b]   =   T_interface(R) ·     [B_a]
#
# where `T_interface(R)` is a 2×2 matrix derived from continuity of
# (u_r, σ_rr) across the interface (perfect) or from the chosen jump
# law (spring / membrane).  Each entry has the form
# `(linear-in-κ, μ) / (3κ_b + 4μ_b)`, so for any non-degenerate
# moduli — *including* near-zero soft phases — the transition stays
# numerically well-conditioned.  This avoids the catastrophic
# cancellation that arises when computing
# `M(r_out; κ, μ) · M(r_in; κ, μ)^{-1}` on a fundamental matrix
# whose `σ`-rows are O(μ) and `(U, V)`-rows are O(1).
#
# The denominator `3κ_b + 4μ_b` is `> 0` for any non-vacuum layer and
# is inverted via [`volterra_divide`](@ref) so that, for soft phases,
# we never form an explicit `(3κ_b + 4μ_b)^{-vol}` whose entries
# would scale as `1 / κ`.

"""
    _bulk_transition_alv(intf, M_κ_a, M_μ_a, M_κ_b, M_μ_b, R, times, n)
        -> NTuple{4, Matrix}

Closed-form `(2 × 2)` block bulk transition at the interface located
at radius `R` between the inner layer `(M_κ_a, M_μ_a)` and the outer
layer `(M_κ_b, M_μ_b)`.  Returns the four `n × n` Volterra blocks
`(T11, T12, T21, T22)` of the amplitude transition

    [A_b]   [T11 T12]   [A_a]
    [B_b] = [T21 T22] · [B_a]

Numerically stable for arbitrary modulus contrasts (soft pores in a
solid matrix, step-activated `ViscoLaw`s, etc.).
"""
function _bulk_transition_alv(
        ::PerfectInterface,
        M_κ_a, M_μ_a, M_κ_b, M_μ_b,
        R::Real, times::AbstractVector, n::Int
    )
    R³ = R^3
    Sb = 3 .* M_κ_b .+ 4 .* M_μ_b   # 3κ_b + 4μ_b (denominator)
    # Hervé–Zaoui closed-form: T = M_b^{-1} · M_a → inverse on the LEFT.
    # T11 = U_b · (3κ_a + 4μ_b)
    T11 = volterra_left_divide(Sb, 3 .* M_κ_a .+ 4 .* M_μ_b; block_size = 1)
    # T12 = (4/R³) U_b · (μ_b - μ_a)
    T12 = volterra_left_divide(Sb, (4 / R³) .* (M_μ_b .- M_μ_a); block_size = 1)
    # T21 = 3R³ U_b · (κ_b - κ_a)
    T21 = volterra_left_divide(Sb, (-3 * R³) .* (M_κ_a .- M_κ_b); block_size = 1)
    # T22 = U_b · (3κ_b + 4μ_a)
    T22 = volterra_left_divide(Sb, 3 .* M_κ_b .+ 4 .* M_μ_a; block_size = 1)
    return (T11, T12, T21, T22)
end

function _bulk_transition_alv(
        intf::SpringInterface,
        M_κ_a, M_μ_a, M_κ_b, M_μ_b,
        R::Real, times::AbstractVector, n::Int
    )
    R² = R^2
    R³ = R^3
    R⁴ = R^4
    Sb = 3 .* M_κ_b .+ 4 .* M_μ_b
    M_kn = _iface_param_volterra(intf.kn, times, n)
    # Numerators (`u_b = u_a + kn σ_a`, `σ_b = σ_a` → augmented bulk transition).
    num11 = 3 .* M_κ_a .+ 4 .* M_μ_b .+ (12 / R) .* (M_μ_b * (M_κ_a * M_kn))
    num12 = (4 / R³) .* (M_μ_b .- M_μ_a) .- (16 / R⁴) .* (M_μ_b * (M_μ_a * M_kn))
    num21 = (-3 * R³) .* (M_κ_a .- M_κ_b) .+ (9 * R²) .* (M_κ_a * (M_κ_b * M_kn))
    num22 = 3 .* M_κ_b .+ 4 .* M_μ_a .- (12 / R) .* (M_κ_b * (M_μ_a * M_kn))
    return (
        volterra_left_divide(Sb, num11; block_size = 1),
        volterra_left_divide(Sb, num12; block_size = 1),
        volterra_left_divide(Sb, num21; block_size = 1),
        volterra_left_divide(Sb, num22; block_size = 1),
    )
end

function _bulk_transition_alv(
        intf::MembraneInterface,
        M_κ_a, M_μ_a, M_κ_b, M_μ_b,
        R::Real, times::AbstractVector, n::Int
    )
    R² = R^2
    R³ = R^3
    R⁴ = R^4
    Sb = 3 .* M_κ_b .+ 4 .* M_μ_b
    M_κs = _iface_param_volterra(intf.κs, times, n)
    # Numerators (`u_b = u_a`, `σ_b = σ_a + 4 κ_s/R² u_a`).
    num11 = 3 .* M_κ_a .+ 4 .* M_μ_b .+ (4 / R) .* M_κs
    num12 = (4 / R³) .* (M_μ_b .- M_μ_a) .+ (4 / R⁴) .* M_κs
    num21 = (-3 * R³) .* (M_κ_a .- M_κ_b) .- (4 * R²) .* M_κs
    num22 = 3 .* M_κ_b .+ 4 .* M_μ_a .- (4 / R) .* M_κs
    return (
        volterra_left_divide(Sb, num11; block_size = 1),
        volterra_left_divide(Sb, num12; block_size = 1),
        volterra_left_divide(Sb, num21; block_size = 1),
        volterra_left_divide(Sb, num22; block_size = 1),
    )
end

# ── Bulk recurrence in amplitude space ──────────────────────────────────────

"""
    bulk_amplitude_seq_alv(sphere, C0_law, times)
        -> (NTuple{N, (A_k, B_k)}, A_M, B_M)

Forward-propagate the bulk mode amplitudes `(A_k, B_k)` (each an
`n × n` Volterra matrix) layer by layer using closed-form interface
transitions.  Core regularity sets `B_1 = 0` and the amplitude seed
is `A_1 = I_n`.  Returns the inner-layer amplitudes plus the
matrix-side `(A_M, B_M)`.

Supports `PerfectInterface`, `SpringInterface` (primal) and
`MembraneInterface` (dual) — same set as the σ-state recurrence —
but with no matrix inversion of a fundamental `M(r; κ, μ)`.  The
only inversions performed are `volterra_divide(_, 3κ + 4μ)` which
remain stable for any non-degenerate modulus.
"""
function bulk_amplitude_seq_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    layers, M_κ_0, M_μ_0 = _bulk_layer_moduli_alv(sphere, C0_law, times)
    n = length(times)
    radii = sphere.radii

    # Over EVERY layer, not just the first: the amplitudes are propagated
    # outward, so a `Dual` in any layer's modulus (autodiff with respect to
    # one layer of the coating) reaches `A` / `B` before the recurrence ends.
    Telt = promote_type(
        eltype(M_κ_0), eltype(M_μ_0),
        (eltype(m) for lay in layers for m in lay)...
    )
    Id = Matrix{Telt}(I, n, n)
    Z = zeros(Telt, n, n)

    A = Vector{Matrix{Telt}}(undef, N)
    B = Vector{Matrix{Telt}}(undef, N)
    A[1] = copy(Id)
    B[1] = copy(Z)

    # Walk through the N interfaces.  Interface k (radius radii[k]) sits
    # between layer k and layer k+1 (or the matrix when k == N).
    A_curr = A[1]
    B_curr = B[1]
    for k in 1:N
        intf = layer_interface(sphere, k)
        (M_κ_a, M_μ_a) = layers[k]
        if k < N
            (M_κ_b, M_μ_b) = layers[k + 1]
        else
            (M_κ_b, M_μ_b) = (M_κ_0, M_μ_0)
        end
        T11, T12, T21, T22 = _bulk_transition_alv(
            intf,
            M_κ_a, M_μ_a, M_κ_b, M_μ_b, radii[k], times, n
        )
        A_new = T11 * A_curr + T12 * B_curr
        B_new = T21 * A_curr + T22 * B_curr
        A_curr = A_new
        B_curr = B_new
        if k < N
            A[k + 1] = copy(A_curr)
            B[k + 1] = copy(B_curr)
        end
    end
    # `A_curr`, `B_curr` now hold matrix-side amplitudes at r_N⁺.
    return (ntuple(k -> (A[k], B[k]), N), A_curr, B_curr)
end

"""
    bulk_localization_alv(sphere, C0_law, times) -> NTuple{N, Matrix}

Per-layer bulk localization matrices `α_k(t,t')` (`n × n` each), such
that `<ε_v>_layer_k = ⟨α_k⟩ ∘ ε_v_∞` in the Volterra sense.

For the bulk Y₀ harmonic, the volume-averaged volumetric strain in
layer `k` is exactly `3 A_k` (the mode-2 amplitude `B_k` has
`u_r = B_k/r²` ⇒ traceless contribution).  Therefore
`α_k = A_k · A_M^{-1}` with `A_M` the matrix-side mode-1 amplitude.

Implementation : amplitude-space recurrence with closed-form
interface transitions ([`bulk_amplitude_seq_alv`](@ref)).  Stable
for arbitrary modulus contrasts (pores, step-activated layers …).

Reference : Hervé-Zaoui 1993 (elastic) ; ECHOES manual ch07
§"n-layer ALV bulk recurrence" ; sanahuja2013.
"""
function bulk_localization_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    inside_amps, A_M, _ = bulk_amplitude_seq_alv(sphere, C0_law, times)
    A_M_inv = volterra_inverse(A_M; block_size = 1)
    return ntuple(N) do k
        A_k, _ = inside_amps[k]
        A_k * A_M_inv
    end
end

"""
    bulk_state_seq_alv(sphere, C0_law, times)
        -> (inside_states::NTuple{N, Matrix}, s_matrix::Matrix)

Backwards-compatible wrapper that exposes the σ-state form of the
bulk recurrence: at each layer `k`, reconstruct the `2n × n` block
`(u_r ; σ_rr)` from the per-layer amplitudes via
`u_r = A_k r_k + B_k / r_k²` and `σ_rr = 3 κ_k A_k - 4 μ_k B_k / r_k³`.

Most users should rely on [`bulk_localization_alv`](@ref) directly.
This helper is retained for the existing test that asserts the
matrix-side state matches the analytical Hervé-Zaoui boundary
condition.
"""
function bulk_state_seq_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    layers, M_κ_0, M_μ_0 = _bulk_layer_moduli_alv(sphere, C0_law, times)
    inside_amps, A_M, B_M = bulk_amplitude_seq_alv(sphere, C0_law, times)
    n = length(times)
    radii = sphere.radii
    Telt = eltype(A_M)

    function pack_state(r, M_κ, M_μ, A, B)
        s = zeros(Telt, 2 * n, n)
        s[1:n, 1:n] = r * A + (1 / r^2) * B
        s[(n + 1):(2n), 1:n] = 3 * (M_κ * A) - (4 / r^3) * (M_μ * B)
        return s
    end

    inside = ntuple(N) do k
        (M_κ_k, M_μ_k) = layers[k]
        A_k, B_k = inside_amps[k]
        pack_state(radii[k], M_κ_k, M_μ_k, A_k, B_k)
    end
    s_matrix = pack_state(radii[N], M_κ_0, M_μ_0, A_M, B_M)
    return inside, s_matrix
end

# =============================================================================
#  Shear (Y₂-harmonic) ALV recurrence — 4n × 4n state vector.
#
#  The deviatoric problem in an isotropic ALV layer has the same four
#  fundamental modes as in elasticity (radial dependencies r, r³, 1/r⁴,
#  1/r²) — only the *amplitude* of each mode and the local moduli κ, μ
#  become Volterra n×n matrices.  We adopt **time-major** indexing for
#  the (4n × 4n) fundamental matrix so it is block-lower-triangular with
#  4×4 diagonal blocks, allowing `volterra_inverse(_; block_size = 4)`
#  to invert it via block forward-substitution.
#
#  Reference : herve1993 (elastic) ; sanahuja2013 ALV
#  generalization ; ECHOES manual ch07 §"Layered sphere ALV".
# =============================================================================

"""
    _shear_M_matrix_alv(r, M_κ, M_μ, n) -> Matrix{T}  (4n × 4n, time-major)

ALV fundamental matrix for the deviatoric Y₂ harmonic, in
**τ-scaling** : the state vector is `(U, V, τ_rr, τ_rθ)` with
`τ = σ / μ` (per-layer normalization).  Rows 3 and 4 of the matrix
no longer carry an explicit `μ` factor; all entries are functions of
the modulus ratio `M_x = M_κ ∘ M_μ^{-vol}` and the radius only.
This keeps every entry `O(1)` for **any** physically-admissible
modulus (including pores `κ ≈ μ ≈ 0`), so the 4×4 diagonal blocks
in the time-major layout are well-conditioned and
`volterra_inverse(_; block_size = 4)` is stable in Float64.

The price for this stability is a non-trivial perfect-interface
jump: continuity of the *physical* `σ_rr` translates to
`τ_rr_+ = (μ_-/μ_+) τ_rr_-` across the interface.  The interface
helper [`_shear_interface_T_alv`](@ref) handles that conversion
via [`volterra_divide`](@ref).

Each scalar entry of the elastic 4×4 matrix becomes an `n × n`
Volterra matrix; entries are arranged in **time-major** layout
(row `(t-1)·4 + i`, col `(s-1)·4 + j` carries the `(t, s)` Volterra
entry of the `(i, j)` block), so the resulting `4n × 4n` matrix is
block-lower-triangular with 4×4 diagonal blocks.
"""
function _shear_M_matrix_alv(
        r::Real, M_κ::AbstractMatrix, M_μ::AbstractMatrix,
        n::Int
    )
    T = promote_type(eltype(M_κ), eltype(M_μ))
    Tr = T(r)
    r² = Tr * Tr
    r³ = r² * Tr
    inv_r² = one(T) / r²
    inv_r³ = inv_r² / Tr
    inv_r⁴ = inv_r² * inv_r²
    inv_r⁵ = inv_r⁴ / Tr
    Id = Matrix{T}(I, n, n)
    Mκ = Matrix{T}(M_κ); Mμ = Matrix{T}(M_μ)
    # σ-form fundamental matrix (state vector (U, V, σ_rr, σ_rθ)) in the
    # **Christensen–Lo (SymPy) convention** — same form as the elastic
    # state-space recurrence in `LayeredSpheres/shear_recurrence.jl`.
    # Mode-2 and mode-4 entries in the U row pick up `μ^{-1}` on the LEFT
    # (natural form from the Lamé operator) — the only Volterra inverse
    # of `μ`; rows 3 and 4 are clean polynomials in `(κ, μ)`.
    #
    # The closed-form inverse in `_shear_M_inverse_alv` derives from the
    # ECHOES C++ formula via the diagonal conjugation
    #   `M_C++ = D_row · M_Christ–Lo · D_col`
    # with `D_row = diag(1/2, 1, 1/2, 1)`, `D_col = diag(1, 1, 2, 2)`.

    blocks = Array{Matrix{T}, 2}(undef, 4, 4)

    # Mode 1 — uniform deviatoric strain. (U, V) = (2r, r).
    blocks[1, 1] = (2 * Tr) * Id
    blocks[2, 1] = Tr * Id
    blocks[3, 1] = 4 * Mμ
    blocks[4, 1] = 2 * Mμ

    # Mode 2 — r³ profile.  α₂ = 6(3x − 2), γ₂ = 15x + 11 with x = κ/μ.
    blocks[1, 2] = (6 * r³) * volterra_left_divide(
        Mμ, 3 .* Mκ .- 2 .* Mμ;
        block_size = 1
    )
    blocks[2, 2] = r³ * volterra_left_divide(
        Mμ, 15 .* Mκ .+ 11 .* Mμ;
        block_size = 1
    )
    blocks[3, 2] = r² * (12 .* Mμ .- 18 .* Mκ)         # 6(2μ − 3κ) r²
    blocks[4, 2] = r² * (48 .* Mκ .+ 10 .* Mμ)         # 2(24κ + 5μ) r²

    # Mode 3 — 1/r⁴ profile. (U, V) = (3, -1).
    blocks[1, 3] = 3 * inv_r⁴ * Id
    blocks[2, 3] = -inv_r⁴ * Id
    blocks[3, 3] = -24 * inv_r⁵ * Mμ
    blocks[4, 3] = 8 * inv_r⁵ * Mμ

    # Mode 4 — 1/r² profile. α₄ = 3(x + 1).
    blocks[1, 4] = (3 * inv_r²) * volterra_left_divide(
        Mμ, Mκ .+ Mμ;
        block_size = 1
    )
    blocks[2, 4] = inv_r² * Id
    blocks[3, 4] = -2 * inv_r³ * (9 .* Mκ .+ 4 .* Mμ)
    blocks[4, 4] = 3 * inv_r³ * Mκ

    # Assemble into the 4n × 4n time-major matrix.
    M = zeros(T, 4 * n, 4 * n)
    @inbounds for i in 1:4, j in 1:4
        Aij = blocks[i, j]
        for s in 1:n
            cs = (s - 1) * 4 + j
            for t in s:n     # Volterra causality: t ≥ s.
                M[(t - 1) * 4 + i, cs] = Aij[t, s]
            end
        end
    end
    return M
end

# ── Shear (4n × 4n, time-major) ALV interface transfer matrices ────────────
#
# State (time-major) : at row (t-1)·4 + i, the i-th component
# (U, V, σ_rr, σ_rθ) at time t.  Mirror of
# `LayeredSpheres._shear_interface_T` with each entry of the elastic
# 4×4 jump matrix promoted to a Volterra n × n block — for scalar
# (constant-in-time) parameters the resulting (4n × 4n) is block-
# diagonal in the 4×4 sense; for `ViscoLaw` parameters the off-diagonal
# (in time) blocks of the parameter's trapezoidal matrix populate the
# corresponding entries.

# Helper: assemble a `(4n × 4n)` time-major block-lower-triangular
# matrix from a 4×4 array of (n × n) Volterra blocks.  Off-diagonal
# 4×4 blocks (between modes/state-components) sit on the **diagonal**
# in time; the within-block n×n Volterra structure handles the time
# coupling.
function _assemble_4n_time_major(
        blocks::AbstractArray{<:AbstractMatrix, 2},
        n::Int
    )
    T = mapreduce(eltype, promote_type, blocks)
    M = zeros(T, 4 * n, 4 * n)
    @inbounds for i in 1:4, j in 1:4
        Aij = blocks[i, j]
        for s in 1:n
            cs = (s - 1) * 4 + j
            for t in s:n
                M[(t - 1) * 4 + i, cs] = Aij[t, s]
            end
        end
    end
    return M
end

"""
    _shear_interface_T_alv(intf, M_κ, M_μ, r, times, n) -> Matrix{T}

`(4n × 4n)` jump matrix for the **shear** (Y₂-harmonic) ALV state at
the interface of type `intf` located at radius `r`.  Time-major
layout, block-lower-triangular with 4×4 diagonal blocks.

For a scalar (elastic) interface the (4n × 4n) matrix is block-
diagonal in the 4×4 sense (the diagonal blocks repeat the elastic 4×4
jump for every time step).  For an ageing interface (parameters
`::ViscoLaw`) the corresponding entries also populate sub-diagonal
4×4 blocks, encoding the convolution.
"""
function _shear_interface_T_alv(
        ::PerfectInterface,
        M_κ_a, M_μ_a, M_κ_b, M_μ_b,
        r, times::AbstractVector, n::Int
    )
    # σ-state perfect interface: identity on (U, V, σ_rr, σ_rθ).
    T = promote_type(eltype(M_μ_a), eltype(M_μ_b))
    return Matrix{T}(I, 4 * n, 4 * n)
end

function _shear_interface_T_alv(
        intf::SpringInterface,
        M_κ_a, M_μ_a, M_κ_b, M_μ_b,
        r, times::AbstractVector, n::Int
    )
    M_kn = _iface_param_volterra(intf.kn, times, n)
    M_kt = _iface_param_volterra(intf.kt, times, n)
    T = promote_type(
        eltype(M_μ_a), eltype(M_μ_b),
        eltype(M_kn), eltype(M_kt)
    )
    Id = Matrix{T}(I, n, n)
    Z = zeros(T, n, n)
    blocks = Matrix{Matrix{T}}(undef, 4, 4)
    @inbounds for i in 1:4, j in 1:4
        blocks[i, j] = Z
    end
    blocks[1, 1] = Id
    blocks[2, 2] = Id
    blocks[3, 3] = Id
    blocks[4, 4] = Id
    # σ-state: U⁺ = U⁻ + kn · σ_rr⁻ ; V⁺ = V⁻ + kt · σ_rθ⁻.
    blocks[1, 3] = T.(M_kn)
    blocks[2, 4] = T.(M_kt)
    return _assemble_4n_time_major(blocks, n)
end

function _shear_interface_T_alv(
        intf::MembraneInterface,
        M_κ_a, M_μ_a, M_κ_b, M_μ_b,
        r, times::AbstractVector, n::Int
    )
    M_κs = _iface_param_volterra(intf.κs, times, n)
    M_μs = _iface_param_volterra(intf.μs, times, n)
    T = promote_type(
        eltype(M_μ_a), eltype(M_μ_b),
        eltype(M_κs), eltype(M_μs)
    )
    Id = Matrix{T}(I, n, n)
    Z = zeros(T, n, n)
    inv_r² = one(T) / T(r * r)
    blocks = Matrix{Matrix{T}}(undef, 4, 4)
    @inbounds for i in 1:4, j in 1:4
        blocks[i, j] = Z
    end
    blocks[1, 1] = Id
    blocks[2, 2] = Id
    blocks[3, 3] = Id
    blocks[4, 4] = Id
    # Gurtin–Murdoch surface-elasticity Y₂ traction jump — identical to the
    # elastic `_shear_interface_T(::MembraneInterface, ...)` (same
    # Christensen–Lo state convention, so the coefficients carry over
    # verbatim).  Derived symbolically from `[σ·n] = −divₛσˢ` and validated
    # to machine precision against Echoes' `DUALDISC` (`κs = λs + μs`):
    #   σ_rr⁺ = σ_rr⁻ + ( 4κs U − 12κs W) / r²
    #   σ_rθ⁺ = σ_rθ⁻ + (−2κs U + (6κs + 4μs) W) / r²
    blocks[3, 1] = 4 * inv_r² .* T.(M_κs)
    blocks[3, 2] = -12 * inv_r² .* T.(M_κs)
    blocks[4, 1] = -2 * inv_r² .* T.(M_κs)
    blocks[4, 2] = inv_r² .* (6 .* T.(M_κs) .+ 4 .* T.(M_μs))
    return _assemble_4n_time_major(blocks, n)
end

"""
    _shear_M_inverse_alv(r, M_κ, M_μ, n) -> Matrix{T}    (4n × 4n)

Closed-form `M(r; κ, μ)^{-1}` for the σ-form deviatoric (Y₂)
fundamental matrix.  Mirrors the reference C++ `set_visco_inv_matrix_dev`
formula.  The only `n × n` Volterra
inverses required are `U = (3κ + 4μ)^{-vol}` and `μ^{-vol}` — both
guaranteed regular for any non-vacuum modulus.  This avoids inverting
the full `4 × 4` diagonal block of `M(r)`, whose `det` collapses with
`μ → 0` (soft phases, step-activated layers).
"""
function _shear_M_inverse_alv(
        r::Real, M_κ::AbstractMatrix, M_μ::AbstractMatrix,
        n::Int
    )
    T = promote_type(eltype(M_κ), eltype(M_μ))
    Tr = T(r)
    R² = Tr * Tr
    R³ = R² * Tr
    R⁴ = R³ * Tr
    R⁵ = R⁴ * Tr
    Mκ = Matrix{T}(M_κ); Mμ = Matrix{T}(M_μ)
    # n × n Volterra inverses (the only ones needed).
    U = volterra_inverse(3 .* Mκ .+ 4 .* Mμ; block_size = 1)
    invMμ = volterra_inverse(Mμ; block_size = 1)
    # Composite combinations (all with the inverse on the LEFT, matching
    # the C++ `mult_array(U, ...)` convention).
    k9mu4 = 9 .* Mκ .+ 4 .* Mμ
    k3mmu2 = 3 .* Mκ .- 2 .* Mμ
    k15mu11 = 15 .* Mκ .+ 11 .* Mμ
    k24mu5 = 24 .* Mκ .+ 5 .* Mμ
    kmu = Mκ .+ Mμ
    Uk = U * Mκ
    Uk9mu4 = U * k9mu4
    Uk3mmu2 = U * k3mmu2
    Uk15mu11 = U * k15mu11
    Uk24mu5 = U * k24mu5
    Uμ = U * Mμ
    U_kmu_invμ = U * (kmu * invMμ)
    U_k3mmu2_invμ = U * (k3mmu2 * invMμ)
    U_k15mu11_invμ = U * (k15mu11 * invMμ)

    # ECHOES C++ closed-form M^{-1} (Christensen–Lo derivative).
    blocks_cpp = Array{Matrix{T}, 2}(undef, 4, 4)
    one70 = T(1 // 70)
    blocks_cpp[1, 1] = one70 * (28 / Tr) * Uk9mu4
    blocks_cpp[1, 2] = one70 * (-126 / Tr) * Uk
    blocks_cpp[1, 3] = one70 * 42 * U_kmu_invμ
    blocks_cpp[1, 4] = one70 * 42 * U
    blocks_cpp[2, 1] = one70 * (-16 / R³) * Uμ
    blocks_cpp[2, 2] = one70 * (16 / R³) * Uμ
    blocks_cpp[2, 3] = one70 * (-2 / R²) * U
    blocks_cpp[2, 4] = one70 * (2 / R²) * U
    blocks_cpp[3, 1] = one70 * (2 * R⁴) * Uk3mmu2
    blocks_cpp[3, 2] = one70 * (-2 * R⁴) * Uk24mu5
    blocks_cpp[3, 3] = one70 * (2 * R⁵) * U_k3mmu2_invμ
    blocks_cpp[3, 4] = one70 * R⁵ * U_k15mu11_invμ
    blocks_cpp[4, 1] = one70 * (28 * R²) * Uμ
    blocks_cpp[4, 2] = one70 * (42 * R²) * Uμ
    blocks_cpp[4, 3] = one70 * (-14 * R³) * U
    blocks_cpp[4, 4] = one70 * (-21 * R³) * U

    # Conjugate to the Christensen–Lo / SymPy convention used throughout
    # `_shear_M_matrix_alv` (and by the elastic state-space recurrence):
    #     M_C++ = D_row · M_CL · D_col,  with
    #     D_row = diag(1/2, 1, 1/2, 1),  D_col = diag(1, 1, 2, 2),
    # ⇒   M_CL^{-1} = D_col · M_C++^{-1} · D_row.
    # Per-(i,j) block scaling: factor[i, j] = D_col[i] · D_row[j].
    D_row = (T(1 // 2), T(1), T(1 // 2), T(1))
    D_col = (T(1), T(1), T(2), T(2))
    blocks = Array{Matrix{T}, 2}(undef, 4, 4)
    @inbounds for i in 1:4, j in 1:4
        blocks[i, j] = (D_col[i] * D_row[j]) .* blocks_cpp[i, j]
    end
    return _assemble_4n_time_major(blocks, n)
end

"""
    _shear_layer_transfer_alv(r_out, r_in, M_κ, M_μ, n) -> Matrix{T}

`(4n × 4n)` intra-layer field-to-field transfer
`S(r_out) = T · S(r_in)` for the deviatoric (Y₂) problem in an ALV
layer with Volterra moduli `(M_κ, M_μ)`.  Uses the closed-form
[`_shear_M_inverse_alv`](@ref) so the only Volterra inverses are the
n × n `(3κ + 4μ)^{-vol}` and `μ^{-vol}` — the dense `M(r; κ, μ)^{-1}`
is never formed (its `4 × 4` diagonal blocks collapse for soft phases).
"""
function _shear_layer_transfer_alv(
        r_out::Real, r_in::Real,
        M_κ::AbstractMatrix, M_μ::AbstractMatrix,
        n::Int
    )
    M_out = _shear_M_matrix_alv(r_out, M_κ, M_μ, n)
    M_in_inv = _shear_M_inverse_alv(r_in, M_κ, M_μ, n)
    return M_out * M_in_inv
end

"""
    _shear_seed_states_alv(r_1, M_κ_1, M_μ_1, n) -> (probe_a, probe_b)

Two `(4n × n)` probe state matrices at `r = r_1⁻` corresponding to
amplitudes `(a, b) = (I_n, 0)` and `(0, I_n)` in the core layer (with
the singular amplitudes `c = d = 0` enforced by finiteness at the
origin).  Each probe is the appropriate "block column" of
`M(r_1; M_κ_1, M_μ_1)` extracted in time-major form.
"""
function _shear_seed_states_alv(
        r_1::Real,
        M_κ_1::AbstractMatrix, M_μ_1::AbstractMatrix,
        n::Int
    )
    M_1 = _shear_M_matrix_alv(r_1, M_κ_1, M_μ_1, n)
    # In time-major form, mode-j columns are at positions j, 4+j, 8+j, …
    probe_a = Matrix(M_1[:, 1:4:end])
    probe_b = Matrix(M_1[:, 2:4:end])
    return probe_a, probe_b
end

"""
    _shear_amp_blocks_alv(r, M_κ, M_μ, n, state) -> (a, b)

Given a `(4n × m)` state matrix in time-major layout and the Volterra
moduli of the layer at radius `r`, solve `M(r) · x = state` (Volterra
inverse with `block_size = 4`) and extract the "mode 1" and "mode 2"
amplitude blocks (`n × m` each).  Modes 3 and 4 are not returned (they
are not needed for layered-sphere localization since `c = d = 0` for the
core probe construction and the matrix-side normalization only fixes
`a` and `b`).
"""
function _shear_amp_blocks_alv(
        r::Real, M_κ::AbstractMatrix, M_μ::AbstractMatrix,
        n::Int, state::AbstractMatrix
    )
    # Closed-form M(r)^{-1} (the only n × n Volterra inverses are
    # `(3κ+4μ)^{-vol}` and `μ^{-vol}`, both stable for any non-vacuum
    # modulus — see `_shear_M_inverse_alv`).
    M_r_inv = _shear_M_inverse_alv(r, M_κ, M_μ, n)
    amp = M_r_inv * state                            # (4n × m)
    m = size(state, 2)
    T = eltype(amp)
    a = zeros(T, n, m)
    b = zeros(T, n, m)
    @inbounds for t in 1:n
        row_a = (t - 1) * 4 + 1
        row_b = (t - 1) * 4 + 2
        for s in 1:m
            a[t, s] = amp[row_a, s]
            b[t, s] = amp[row_b, s]
        end
    end
    return a, b
end

"""
    _shear_solve_far_field_alv(a_a, a_b, b_a, b_b, n) -> (λ_a, λ_b)

Solve the `2n × 2n` Volterra block system

    a_a ∘ λ_a + a_b ∘ λ_b = I_n
    b_a ∘ λ_a + b_b ∘ λ_b = 0

for the two `n × n` Volterra matrices `λ_a`, `λ_b` that combine the two
probes so the matrix-side amplitudes match unit far-field
`(a, b) = (I_n, 0)`.  Built as a single time-major
`block_size = 2` Volterra inversion.
"""
function _shear_solve_far_field_alv(
        a_a::AbstractMatrix, a_b::AbstractMatrix,
        b_a::AbstractMatrix, b_b::AbstractMatrix,
        n::Int
    )
    T = promote_type(eltype(a_a), eltype(a_b), eltype(b_a), eltype(b_b))
    M_sys = zeros(T, 2 * n, 2 * n)
    @inbounds for s in 1:n
        c1 = (s - 1) * 2 + 1
        c2 = (s - 1) * 2 + 2
        for t in s:n
            r1 = (t - 1) * 2 + 1
            r2 = (t - 1) * 2 + 2
            M_sys[r1, c1] = a_a[t, s]
            M_sys[r1, c2] = a_b[t, s]
            M_sys[r2, c1] = b_a[t, s]
            M_sys[r2, c2] = b_b[t, s]
        end
    end
    M_sys_inv = volterra_inverse(M_sys; block_size = 2)
    # RHS is (I_n; 0) packed in time-major form: rhs[(t-1)*2+1, t] = 1.
    rhs = zeros(T, 2 * n, n)
    @inbounds for t in 1:n
        rhs[(t - 1) * 2 + 1, t] = one(T)
    end
    λ = M_sys_inv * rhs                              # (2n × n)
    λ_a = zeros(T, n, n)
    λ_b = zeros(T, n, n)
    @inbounds for t in 1:n
        r1 = (t - 1) * 2 + 1
        r2 = (t - 1) * 2 + 2
        for s in 1:n
            λ_a[t, s] = λ[r1, s]
            λ_b[t, s] = λ[r2, s]
        end
    end
    return λ_a, λ_b
end

# ── Forward propagation of the two probes through the layers ────────────────

"""
    _shear_state_seq_alv(sphere, layers, times)
        -> (inside_a, inside_b, s_a, s_b)

Forward-propagate the two deviatoric probe states from the core
outward through every layer (perfect interfaces only).  Returns one
`(4n × n)` matrix per layer at `r_k⁻` (just inside the k-th
interface) plus the matrix-side states `s_a`, `s_b` at `r_N⁺`.

`layers` must be the same `(M_κ_k, M_μ_k)` tuple produced by
`_bulk_layer_moduli_alv`.

`layers` is annotated `Tuple`, NOT `NTuple{N, <:Tuple}`: `NTuple` requires
every element to have the *same* type, and `_bulk_layer_moduli_alv` builds it
with `ntuple`, so differentiating with respect to ONE layer's modulus yields a
heterogeneous tuple (that layer's blocks `Dual`, the others `Float64`) which
the `NTuple` signature rejects outright.
"""
function _shear_state_seq_alv(
        sphere::LayeredSphere{T, N},
        layers::Tuple,
        M_κ_0, M_μ_0,
        times::AbstractVector{<:Real}
    ) where {T, N}
    n = length(times)
    radii = sphere.radii

    M_κ_1, M_μ_1 = layers[1]
    sa, sb = _shear_seed_states_alv(radii[1], M_κ_1, M_μ_1, n)

    # The propagated state picks up the eltype of every layer it crosses, so
    # the seed's own eltype (layer 1) is only a floor — a `Dual` in any OUTER
    # layer, or in the matrix reference, widens `sa` / `sb` further down the
    # loop.  Type the containers from the promotion over all of them.
    T_state = promote_type(
        eltype(sa), eltype(sb), eltype(M_κ_0), eltype(M_μ_0),
        (eltype(m) for lay in layers for m in lay)...
    )
    inside_a = Vector{Matrix{T_state}}(undef, N)
    inside_b = Vector{Matrix{T_state}}(undef, N)
    @inbounds for k in 1:N
        inside_a[k] = sa
        inside_b[k] = sb
        intf = layer_interface(sphere, k)
        (M_κ_a, M_μ_a) = layers[k]
        if k < N
            (M_κ_b, M_μ_b) = layers[k + 1]
        else
            (M_κ_b, M_μ_b) = (M_κ_0, M_μ_0)
        end
        # `PerfectInterface` is the identity jump on (U,V,σ_rr,σ_rθ) — skip
        # building and applying the `(4n×4n)` identity matrix entirely
        # rather than pay for a materialized `Matrix{T}(I,4n,4n)` and a
        # matmul that both reduce to a no-op.
        if !(intf isa PerfectInterface)
            T_intf = _shear_interface_T_alv(
                intf,
                M_κ_a, M_μ_a, M_κ_b, M_μ_b, radii[k], times, n
            )
            sa = T_intf * sa
            sb = T_intf * sb
        end
        if k < N
            T_layer = _shear_layer_transfer_alv(
                radii[k + 1], radii[k],
                M_κ_b, M_μ_b, n
            )
            sa = T_layer * sa
            sb = T_layer * sb
        end
    end
    return inside_a, inside_b, sa, sb
end

"""
    shear_localization_alv(sphere, C0_law, times) -> NTuple{N, Matrix}

Per-layer deviatoric ALV localization matrices `β_k(t,t')` (`n × n`
each), defined by `<ε_d>_layer_k = ⟨β_k⟩ ∘ ε_d_∞` in the Volterra
sense — the `n × n` Volterra matrix that maps a unit deviatoric remote
strain to the volume-averaged deviatoric strain in layer `k`.

The Y₂-harmonic recurrence uses a `(4n × 4n)` time-major fundamental
matrix per layer, two probe states with seed `(a, b) = (I, 0)` and
`(0, I)` propagated outward, and a final `(2n × 2n)` Volterra solve
that picks the linear combination matching unit far-field
`(a_{N+1}, b_{N+1}) = (I, 0)`.  Per-layer `β_k` is the mode-1
amplitude block extracted from the combined inside state.

Reference : ECHOES manual ch07 §"n-layer ALV shear recurrence" ;
Hervé-Zaoui 1993 generalized to ALV via [sanahuja2013](@cite).
"""
function shear_localization_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    n = length(times)
    layers, M_κ_0, M_μ_0 = _bulk_layer_moduli_alv(sphere, C0_law, times)

    inside_a, inside_b, s_mat_a, s_mat_b = _shear_state_seq_alv(
        sphere, layers, M_κ_0, M_μ_0, times
    )
    radii = sphere.radii

    # Matrix-side amplitudes of each probe at r_N⁺. The closed-form
    # `M(r_N)^{-1}` inside `_shear_amp_blocks_alv` is the same for both
    # probes — concatenate them into one call instead of computing that
    # inverse twice.
    state_ab = hcat(s_mat_a, s_mat_b)
    a_ab, b_ab = _shear_amp_blocks_alv(radii[N], M_κ_0, M_μ_0, n, state_ab)
    a_a, a_b = a_ab[:, 1:n], a_ab[:, (n + 1):(2n)]
    b_a, b_b = b_ab[:, 1:n], b_ab[:, (n + 1):(2n)]

    # Linear combination giving (a_{N+1}, b_{N+1}) = (I_n, 0).
    λ_a, λ_b = _shear_solve_far_field_alv(a_a, a_b, b_a, b_b, n)

    return ntuple(N) do k
        (M_κ_k, M_μ_k) = layers[k]
        inside_combo = inside_a[k] * λ_a + inside_b[k] * λ_b   # (4n × n)
        a_k, b_k = _shear_amp_blocks_alv(radii[k], M_κ_k, M_μ_k, n, inside_combo)
        # Layer bounds (innermost layer has r_a = 0).
        r_a = (k == 1) ? zero(eltype(radii)) : radii[k - 1]
        r_b = radii[k]
        # Mode-2 contribution to the layer-volume-averaged deviatoric strain.
        # `(21/5) · μ^{-vol} · (3κ + μ) · (r_b⁵ − r_a⁵)/(r_b³ − r_a³)`
        # (Christensen–Lo mode-2 angular integral with the C++ mode
        # normalization in `_shear_M_matrix_alv`).
        geom = (r_b^5 - r_a^5) / (r_b^3 - r_a^3)
        F_k = (21 / 5) * geom * volterra_left_divide(
            M_μ_k, 3 .* M_κ_k .+ M_μ_k;
            block_size = 1
        )
        a_k .+ F_k * b_k
    end
end

# =============================================================================
#  Public ALV-contribution / localization assembly for the layered sphere.
#
#  The combination of bulk α_k and shear β_k localization matrices into
#  isotropic 6n×6n strain-strain localization and stiffness contribution
#  tensors mirrors the elastic helpers in `LayeredSpheres.jl`:
#     A_loc = α 𝕁 + β 𝕂                (mean strain-strain in the sphere)
#     N     = Σ_k f_k (C_k − C_0) ∘ A_k  (size-independent contribution).
# =============================================================================

"""
    strain_strain_loc_alv(sphere, C0_law, times) -> Matrix{T}

`(6n × 6n)` block matrix describing the volume-averaged strain-strain
localization across the **entire** layered sphere under a unit
Volterra far-field strain.  In iso form this is
`A_avg = ⟨α⟩ 𝕁 + ⟨β⟩ 𝕂` with
`⟨α⟩ = Σ_k f_k α_k(t,t')` and `⟨β⟩ = Σ_k f_k β_k(t,t')` (Volterra
products).

This is the analog used by the ALV dilute / MT / Maxwell schemes
when the inclusion phase is a `LayeredSphere`.
"""
function strain_strain_loc_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    α_k = bulk_localization_alv(sphere, C0_law, times)
    β_k = shear_localization_alv(sphere, C0_law, times)
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    α_avg = sum(f[k] * α_k[k] for k in 1:N)
    β_avg = sum(f[k] * β_k[k] for k in 1:N)
    return iso_blocks_from_params(α_avg, β_avg)
end

"""
    stiffness_contribution_alv(sphere, C0_law, times) -> Matrix{T}

`(6n × 6n)` size-independent ALV stiffness contribution of a layered
sphere relative to its iso ALV matrix `C0_law`.  Iso parameters
(α-, β-blocks of the assembled matrix) are
   `α = 3 Σ_k f_k (M_κ_k − M_κ_0) ∘ α_k`,
   `β = 2 Σ_k f_k (M_μ_k − M_μ_0) ∘ β_k`,
where `α_k`, `β_k` are the per-layer localization matrices and `M_κ_k`,
`M_μ_k` the per-layer Volterra moduli.

The dilute-scheme effective stiffness with this inclusion at volume
fraction `f` is `C̃_eff = C̃_0 + f · stiffness_contribution_alv(sphere, …)`.
"""
function stiffness_contribution_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    layers, M_κ_0, M_μ_0 = _bulk_layer_moduli_alv(sphere, C0_law, times)
    α_k = bulk_localization_alv(sphere, C0_law, times)
    β_k = shear_localization_alv(sphere, C0_law, times)
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    n = length(times)

    Tα = promote_type(eltype(M_κ_0), eltype(α_k[1]), eltype(layers[1][1]))
    Tβ = promote_type(eltype(M_μ_0), eltype(β_k[1]), eltype(layers[1][2]))
    N_bulk = zeros(Tα, n, n)
    N_shear = zeros(Tβ, n, n)
    @inbounds for k in 1:N
        (M_κ_k, M_μ_k) = layers[k]
        N_bulk .+= f[k] .* ((M_κ_k - M_κ_0) * α_k[k])
        N_shear .+= f[k] .* ((M_μ_k - M_μ_0) * β_k[k])
    end
    a_surf, b_surf = _membrane_surface_stress_alv(sphere, C0_law, times)
    return iso_blocks_from_params(3 .* N_bulk .+ a_surf, 2 .* N_shear .+ b_surf)
end

"""
    stiffness_contribution_alv_at(sphere::LayeredSphere, C_ref::AbstractMatrix, times)

Layered-sphere counterpart of
[`stiffness_contribution_alv_at`](@ref MeanFieldHomogenization.Viscoelasticity.stiffness_contribution_alv_at)
for cracks: the same size-independent ALV contribution, but against a
pre-discretized `(6n × 6n)` reference — the *running* effective medium
of the differential (or self-consistent) ODE rather than the matrix law.

The reference must be isotropic, as everywhere in the layered-sphere ALV
recurrences.
"""
function stiffness_contribution_alv_at(
        sphere::LayeredSphere,
        C_ref::AbstractMatrix,
        times::AbstractVector{<:Real}
    )
    return stiffness_contribution_alv(sphere, C_ref, times)
end

"""
    _membrane_surface_stress_alv(sphere, C0_law, times) -> (a_surf, b_surf)

ALV analog of `_membrane_surface_stress`: the Gurtin–Murdoch surface-stress
contribution of the dual ([`MembraneInterface`](@ref)) interfaces to the
volume-averaged stress of the layered sphere, as `(n × n)` Volterra blocks in
the bulk (`𝕁`) and shear (`𝕂`) parts.  The surface moduli `(κs, μs)` are
elastic (constant in time); they multiply the time-dependent interface
displacement amplitudes `u_r(r)` (bulk) and `U(r), W(r)` (shear).  See the
elastic derivation for the coefficients

```
bulk :  4 κs · u_r(r)·r / R³ ,   shear:  (2/5)(−κs U + 3κs W + 6μs W)·r / R³ .
```
"""
function _membrane_surface_stress_alv(
        sphere::LayeredSphere{T, N},
        C0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    n = length(times)
    radii = sphere.radii
    R³ = radii[N]^3

    if !any(k -> layer_interface(sphere, k) isa MembraneInterface, 1:N)
        Z = zeros(T, n, n)
        return Z, Z
    end
    layers, M_κ_0, M_μ_0 = _bulk_layer_moduli_alv(sphere, C0_law, times)

    # Bulk displacement amplitudes u_r(r_k), normalized by the far-field A∞.
    inside_amps, A_M, _ = bulk_amplitude_seq_alv(sphere, C0_law, times)
    A_M_inv = volterra_inverse(A_M; block_size = 1)
    Tb = eltype(A_M)
    a_surf = zeros(Tb, n, n)
    b_surf = zeros(Tb, n, n)

    # Deviatoric interface displacement amplitudes (U, W)(r_k), normalized to a
    # unit remote deviatoric far field (same λ combination as the localization).
    inside_a, inside_b, s_a, s_b = _shear_state_seq_alv(sphere, layers, M_κ_0, M_μ_0, times)
    a_ab, b_ab = _shear_amp_blocks_alv(radii[N], M_κ_0, M_μ_0, n, hcat(s_a, s_b))
    λ_a, λ_b = _shear_solve_far_field_alv(
        a_ab[:, 1:n], a_ab[:, (n + 1):(2n)],
        b_ab[:, 1:n], b_ab[:, (n + 1):(2n)], n
    )

    for k in 1:N
        intf = layer_interface(sphere, k)
        intf isa MembraneInterface || continue
        κs = intf.κs; μs = intf.μs
        r = radii[k]
        A_k, B_k = inside_amps[k]
        u_r = (r .* A_k .+ (1 / r^2) .* B_k) * A_M_inv     # normalized u_r(r_k)
        a_surf .+= (4 * κs * r / R³) .* u_r
        combo = inside_a[k] * λ_a + inside_b[k] * λ_b       # (4n × n) state
        U = zeros(Tb, n, n); W = zeros(Tb, n, n)
        @inbounds for t in 1:n, s in 1:n
            U[t, s] = combo[(t - 1) * 4 + 1, s]
            W[t, s] = combo[(t - 1) * 4 + 2, s]
        end
        b_surf .+= (r / R³) .* ((2 / 5) .* (-κs .* U .+ 3κs .* W .+ 6μs .* W))
    end
    return a_surf, b_surf
end
