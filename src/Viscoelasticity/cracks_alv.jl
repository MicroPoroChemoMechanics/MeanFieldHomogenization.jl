# =============================================================================
#  cracks_alv.jl — pure penny crack in an iso ALV matrix.
#
#  Penny cracks (η = 1) in an **isotropic ALV matrix**, traction-free or with
#  the `(Rn(t,t'), Rt(t,t'))` interface stiffness laws.
#
#  ── Time-space decoupling ─────────────────────────────────────────────────
#
#  In iso elasticity, a penny crack with normal n̂ has a diagonal COD
#  tensor in the crack basis (n̂, t̂₁, t̂₂):
#       B_nn = 16 (1−ν²) / (3π E)
#       B_t  = 32 (1−ν²) / (3π E (2−ν))
#
#  Rewriting in (α = 3K, β = 2μ):
#       B_nn = (8 / (3π))  · (α + 2β) / (β · (α + β/2))
#       B_t  = (32 / (9π)) · (α + 2β) / (β · (α + β))
#
#  In iso ALV every "/" becomes a Volterra inverse and every "·" a Volterra
#  product on `n × n` matrices, and the ORDER of the factors is not free: an
#  aging matrix makes α and β non-commuting. It is fixed by the flat limit of
#  a void spheroid, whose Hill kernel combines (k + 4μ/3)^{-vol} and μ^{-vol}
#  alone (see `_penny_cod_alv`):
#       B̃_nn = (8 / (3π))  · (α + β/2)^{-vol} ∘ (α + 2β) ∘ β^{-vol}
#       B̃_t  = (32 / (9π)) · (α + β)^{-vol}   ∘ (α + 2β) ∘ β^{-vol}
#
#  The compliance contribution H̃ = (3/4) · n̂ ⊗ˢ B̃ ⊗ˢ n̂ is, in the
#  canonical crack-aligned axis n̂ = e₃ + Mandel basis :
#       H[3,3] = (3/4)  · B̃_n      (Walpole ℓ₁)
#       H[4,4] = H[5,5] = (3/8) · B̃_t   (Walpole ℓ₆)
#       all other entries vanish
#
#  i.e. H̃ is a **TI block matrix** in the crack normal axis with Walpole
#  coefficients `ℓ = (3 B̃_n / 4, 0, 0, 0, 0, 3 B̃_t / 8)`.  This plugs
#  directly into the existing TI ALV fast path without any extra
#  infrastructure.
#
#  ── Public API ────────────────────────────────────────────────────────────
#
#  The two functions below mirror the elastic API:
#       cod_kernel_alv(crack, C_M_law, times)         -> Matrix{T}
#       compliance_contribution_alv(crack, C_M_law, times) -> Matrix{T}
#
#  Apply [`delta_compliance_alv`](@ref) (= `(4π/3) · ε · H̃`) to convert
#  from the size-independent contribution to a fractional compliance
#  correction `ΔJ̃ = (4π/3) ε³ᵈ · H̃`.
#
#  Interface-stiffness cracks (`Rn(t,t'), Rt(t,t')`) and TI / aniso
#  matrices are deferred to v0.6.2.
# =============================================================================

# Detect whether a crack normal coincides with the canonical axis e_3.
# We restrict to canonical-axis penny cracks for the iso ALV fast path —
# arbitrary orientation will require a 6×6 Mandel rotation per (i,j) block.
@inline function _crack_axis_is_e3(crack)
    n̂ = TensND.get_array(TensND.tens_basis(crack_basis(crack), 3))
    return abs(n̂[1]) < 1.0e-10 && abs(n̂[2]) < 1.0e-10 && abs(n̂[3] - 1) < 1.0e-10
end

"""
    cod_kernel_alv(crack::EllipticCrack, C_M_law::ViscoLaw, times;
                   Rn = nothing, Rt = nothing) -> NamedTuple

Discrete ALV COD-tensor data for a penny crack ``\\eta = 1`` in an isotropic
ALV matrix.  Returns the named tuple `(B_n = …, B_t = …)` of two
``n\\times n`` scalar Volterra matrices (``n`` = `length(times)`).  Each
``[\\widetilde{B}]_{ij}`` approximates the COD coefficient at the time pair
``(t_i, t_j)``.

# Interface stiffness (Sevostianov-style spring-like interface)

When the crack carries finite **interface stiffness** kernels ``k_n(t,t')``
(normal) and ``k_t(t,t')`` (tangential), pass them as scalar `ViscoLaw`s
through the `Rn` / `Rt` keyword arguments.  The traction-free COD
matrices ``\\widetilde{B}_n``, ``\\widetilde{B}_t`` are then post-corrected via the algebraic identity

```math
\\widetilde{B}^{\\mathrm{int}}_{\\alpha}
= \\bigl(b\\,\\widetilde{k}_{\\alpha} + \\widetilde{B}_{\\alpha}^{-\\circ}\\bigr)^{-\\circ}
= \\widetilde{B}_{\\alpha}\\circ\\bigl(H + b\\,\\widetilde{k}_{\\alpha}\\circ\\widetilde{B}_{\\alpha}\\bigr)^{-\\circ},
\\qquad \\alpha \\in \\{n, t\\},
```

[sevostianovIJSS2007, barthelemyIJES2019](@cite), where ``b`` is the semi-minor
axis (`semi_minor`) of the elliptic crack.  Limits :

* `Rn / Rt = nothing` (default) → traction-free penny limit, recovers
  the existing `B_n`, `B_t`.
* ``k_n, k_t \\to \\infty`` (rigid bonding) → ``\\widetilde{B}^{\\mathrm{int}}_n, \\widetilde{B}^{\\mathrm{int}}_t \\to 0`` (no opening).

Throws if the matrix law is not iso or the crack is not a penny.
"""
function cod_kernel_alv(
        crack::MFH_Core.AbstractCrack, C_M_law::ViscoLaw,
        times::AbstractVector{<:Real};
        Rn::Union{Nothing, ViscoLaw} = nothing,
        Rt::Union{Nothing, ViscoLaw} = nothing
    )
    # Build the matrix relaxation block matrix (invert if law is :creep)
    # and check iso.
    C_M = _trapezoidal_relaxation(C_M_law, times, 6)
    _is_iso_block(C_M) ||
        throw(ArgumentError("cod_kernel_alv: matrix law is not iso (only iso ALV is supported)"))
    α, β = _iso_pair(C_M)

    # Penny check (η = 1).
    η = aspect_ratio(crack)
    isapprox(η, 1.0; atol = 1.0e-12) ||
        throw(ArgumentError("cod_kernel_alv: only penny cracks (η = 1) are currently supported"))

    # Volterra rationals for B̃_n and B̃_t — traction-free penny limit.
    B_n, B_t = _penny_cod_alv(α, β)

    # Interface-stiffness post-correction.
    if Rn !== nothing || Rt !== nothing
        B_n, B_t = _apply_interface_stiffness_alv(
            B_n, B_t, Rn, Rt, times,
            semi_minor(crack)
        )
    end
    return (B_n = B_n, B_t = B_t)
end

"""
    _penny_cod_alv(α, β) -> (B_n, B_t)

The ``n\\times n`` Volterra COD coefficients of a traction-free penny crack in
an isotropic matrix whose iso blocks are ``\\alpha = 3k``, ``\\beta = 2\\mu``:

```math
\\widetilde B_n = \\frac{8}{3\\pi}\\,(\\alpha + \\tfrac12\\beta)^{-\\circ}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-\\circ},
\\qquad
\\widetilde B_t = \\frac{32}{9\\pi}\\,(\\alpha + \\beta)^{-\\circ}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-\\circ}.
```

The order of the three factors is that of the flat limit of a void spheroid,
whose Hill kernel is a linear combination of ``(k + \\tfrac43\\mu)^{-\\circ}`` and
``\\mu^{-\\circ}`` alone; it is also Echoes' (`compute_visco_crack_compliance`).
For an aging matrix the factors do not commute, and the elastic expression
read with the inverse of ``\\beta\\circ(\\alpha + \\tfrac12\\beta)`` on the left is off by a
percent.
"""
function _penny_cod_alv(α::AbstractMatrix, β::AbstractMatrix)
    α_p_2β = α .+ 2β
    β_inv = volterra_inverse(β; block_size = 1)
    B_n = (8 / (3π)) .* (volterra_left_divide(α .+ β ./ 2, α_p_2β; block_size = 1) * β_inv)
    B_t = (32 / (9π)) .* (volterra_left_divide(α .+ β, α_p_2β; block_size = 1) * β_inv)
    return B_n, B_t
end

"""
    _apply_interface_stiffness_alv(B_n, B_t, Rn, Rt, times, b)

Apply the interface-stiffness post-correction
``\\widetilde{B}^{\\mathrm{int}} = \\widetilde{B}\\circ(H + b\\,\\widetilde{k}\\circ\\widetilde{B})^{-\\circ}``
to the traction-free COD matrices `B_n`, `B_t`.  When one of the two
interface laws is `nothing`, the corresponding component is left
untouched (modeling the traction-free direction).
"""
function _apply_interface_stiffness_alv(
        B_n::AbstractMatrix, B_t::AbstractMatrix,
        Rn::Union{Nothing, ViscoLaw},
        Rt::Union{Nothing, ViscoLaw},
        times::AbstractVector{<:Real},
        b::Real
    )
    n = size(B_n, 1)
    Iₙ = Matrix{eltype(B_n)}(LinearAlgebra.I, n, n)
    if Rn !== nothing
        K_n = _trapezoidal_relaxation_scalar(Rn, times)
        KB = K_n * B_n                        # b·K_n·B_n  (Volterra product)
        @. KB *= b
        @. KB += Iₙ                            # 𝟙 + b·K_n·B_n
        B_n = B_n * volterra_inverse(KB; block_size = 1)
    end
    if Rt !== nothing
        K_t = _trapezoidal_relaxation_scalar(Rt, times)
        KB = K_t * B_t
        @. KB *= b
        @. KB += Iₙ
        B_t = B_t * volterra_inverse(KB; block_size = 1)
    end
    return B_n, B_t
end

# Build the scalar (n × n) trapezoidal matrix of an interface ViscoLaw,
# inverting if the law is in `:creep` mode (so the user can pass a creep
# kernel and the algebra still expects a relaxation matrix).
function _trapezoidal_relaxation_scalar(
        law::ViscoLaw,
        times::AbstractVector{<:Real}
    )
    M = trapezoidal_matrix(law, times)
    visco_mode(law) === :creep && return volterra_inverse(M; block_size = 1)
    return M
end

"""
    compliance_contribution_alv(crack, C_M_law::ViscoLaw, times) -> Matrix{T}

Discrete ``6n\\times 6n`` size-independent compliance contribution ``\\widetilde{\\mathbb{H}}`` of a
penny crack in an isotropic ALV matrix.  Computed via the time-space
decoupling formula

```math
\\widetilde{\\mathbb{H}} = \\tfrac{3}{4}\\,\\widetilde{B}_n\\,\\mathbb{W}_1(\\underline{n})
  + \\tfrac{3}{8}\\,\\widetilde{B}_t\\,\\mathbb{W}_6(\\underline{n})
```

where ``\\mathbb{W}_1``, ``\\mathbb{W}_6`` are the canonical Walpole basis tensors of the crack
normal axis.  When ``\\underline{n} = \\underline{e}_3`` the result is in TI form and routes
through the existing TI ALV fast path; arbitrary orientation requires
a ``6\\times 6`` Mandel rotation per ``(i, j)`` block (not yet implemented).

Convention: same as the elastic [`compliance_contribution`](@ref MeanFieldHomogenization.Core.compliance_contribution) — the
Budiansky-O'Connell density factor is applied separately via
[`delta_compliance_alv`](@ref).
"""
function compliance_contribution_alv(
        crack::MFH_Core.AbstractCrack,
        C_M_law::ViscoLaw,
        times::AbstractVector{<:Real};
        Rn::Union{Nothing, ViscoLaw} = nothing,
        Rt::Union{Nothing, ViscoLaw} = nothing
    )
    _crack_axis_is_e3(crack) ||
        throw(ArgumentError("compliance_contribution_alv: only crack normal n̂ = e_3 is currently supported"))
    cod = cod_kernel_alv(crack, C_M_law, times; Rn = Rn, Rt = Rt)
    n = size(cod.B_n, 1)
    T = promote_type(eltype(cod.B_n), eltype(cod.B_t))
    Z = zeros(T, n, n)
    ℓ₁ = (T(3) / T(4)) .* cod.B_n
    ℓ₆ = (T(3) / T(8)) .* cod.B_t
    return ti_blocks_from_params((ℓ₁, copy(Z), copy(Z), copy(Z), copy(Z), ℓ₆))
end

"""
    delta_compliance_alv(crack, H̃, ε) -> Matrix

Apply the Budiansky-O'Connell crack density factor to the
size-independent compliance contribution ``\\widetilde{\\mathbb{H}}`` produced by
[`compliance_contribution_alv`](@ref), giving the fractional
compliance correction ``\\Delta\\widetilde{\\mathbb{L}} = \\tfrac{4\\pi}{3}\\,\\varepsilon^{3\\mathrm{d}}\\,\\widetilde{\\mathbb{H}}`` (penny / elliptic
geometry, ``\\varepsilon^{3\\mathrm{d}} = N\\,a\\,b^{2}``) — same pre-factor as the elastic case.
"""
function delta_compliance_alv(
        crack::MFH_Core.AbstractCrack,
        H̃::AbstractMatrix, ε::Real
    )
    if crack isa EllipticCrack
        return (4π / 3) * ε .* H̃
    elseif crack isa RibbonCrack
        return Float64(π) * ε .* H̃
    else
        throw(ArgumentError("delta_compliance_alv: unsupported crack type $(typeof(crack))"))
    end
end

"""
    stiffness_contribution_alv(crack, C_ref, times) -> Matrix{T}

Discrete ``6n\\times 6n`` crack **stiffness** contribution
``\\widetilde{\\mathbb{N}} = -\\widetilde{\\mathbb{C}}_0\\circ\\widetilde{\\mathbb{H}}\\circ\\widetilde{\\mathbb{C}}_0``,
mirror of the elastic [`stiffness_contribution(crack, C₀)`] formula, with
``\\widetilde{\\mathbb{C}}_0`` the discretized `C_ref`.
`C_ref` may be a `ViscoLaw` (the matrix law — relaxation auto-built
through [`_trapezoidal_relaxation`](@ref)) or a pre-discretized
``6n\\times 6n`` reference matrix (used by SC iterations against the
running estimate `C_n`).
"""
function stiffness_contribution_alv(
        crack::MFH_Core.AbstractCrack,
        C_M_law::ViscoLaw,
        times::AbstractVector{<:Real}
    )
    H̃ = compliance_contribution_alv(crack, C_M_law, times)
    C̃_ref = _trapezoidal_relaxation(C_M_law, times, 6)
    return -(C̃_ref * H̃ * C̃_ref)
end

"""
    stiffness_contribution_alv_at(crack, C_ref::AbstractMatrix) -> Matrix

Variant that takes a pre-discretized ``6n\\times 6n`` reference matrix.
The compliance contribution is recomputed from the iso parameters of
`C_ref` (only iso ALV matrices are currently supported by
[`compliance_contribution_alv`](@ref)).
"""
function stiffness_contribution_alv_at(
        crack::MFH_Core.AbstractCrack,
        C_ref::AbstractMatrix;
        Rn_mat::Union{Nothing, AbstractMatrix} = nothing,
        Rt_mat::Union{Nothing, AbstractMatrix} = nothing
    )
    # Wrap C_ref in a synthetic ViscoLaw for compliance_contribution_alv —
    # it only inspects the iso (α, β) parameters of the trapezoidal of the
    # law, so we just need a "dummy" law whose trapezoidal equals C_ref.
    # The fastest route is to extract (α, β) directly here.
    _is_iso_block(C_ref) ||
        throw(ArgumentError("stiffness_contribution_alv_at: only iso reference is supported"))
    α, β = _iso_pair(C_ref)
    B_n, B_t = _penny_cod_alv(α, β)
    # Optional Sevostianov interface-stiffness correction.  Caller
    # supplies the **already-discretized** scalar interface matrices
    # `Rn_mat`, `Rt_mat` (n × n Volterra) — the iteration of SC against
    # the running estimate does not need to re-trapezoidalize the
    # interface laws each pass.
    if Rn_mat !== nothing || Rt_mat !== nothing
        n_t = size(α, 1)
        Iₙ = Matrix{eltype(α)}(LinearAlgebra.I, n_t, n_t)
        b = semi_minor(crack)
        if Rn_mat !== nothing
            KB = Rn_mat * B_n; @. KB *= b; @. KB += Iₙ
            B_n = B_n * volterra_inverse(KB; block_size = 1)
        end
        if Rt_mat !== nothing
            KB = Rt_mat * B_t; @. KB *= b; @. KB += Iₙ
            B_t = B_t * volterra_inverse(KB; block_size = 1)
        end
    end
    n = size(α, 1)
    T = eltype(α)
    Z = zeros(T, n, n)
    ℓ₁ = (T(3) / T(4)) .* B_n
    ℓ₆ = (T(3) / T(8)) .* B_t
    H̃ = ti_blocks_from_params((ℓ₁, copy(Z), copy(Z), copy(Z), copy(Z), ℓ₆))
    return -(C_ref * H̃ * C_ref)
end

"""
    delta_stiffness_alv(crack, Ñ, ε) -> Matrix

Apply the Budiansky-O'Connell crack density factor to the
size-independent stiffness contribution ``\\widetilde{\\mathbb{N}}`` produced by
[`stiffness_contribution_alv`](@ref), giving the dilute stiffness
correction ``\\Delta\\widetilde{\\mathbb{C}} = \\tfrac{4\\pi}{3}\\,\\varepsilon^{3\\mathrm{d}}\\,\\widetilde{\\mathbb{N}}`` (penny / elliptic) or
``\\pi\\,\\varepsilon^{2\\mathrm{d}}\\,\\widetilde{\\mathbb{N}}`` (ribbon).  Same pre-factors as the elastic case.
"""
function delta_stiffness_alv(
        crack::MFH_Core.AbstractCrack,
        Ñ::AbstractMatrix, ε::Real
    )
    if crack isa EllipticCrack
        return (4π / 3) * ε .* Ñ
    elseif crack isa RibbonCrack
        return Float64(π) * ε .* Ñ
    else
        throw(ArgumentError("delta_stiffness_alv: unsupported crack type $(typeof(crack))"))
    end
end
