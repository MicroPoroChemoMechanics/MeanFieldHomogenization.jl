# =============================================================================
#  cracks_alv.jl — elliptic cracks in an isotropic ALV matrix.
#
#  Elliptic cracks of any aspect ratio η = b/a ≤ 1 (the penny, η = 1, included)
#  and any orientation, in an **isotropic ALV matrix**, traction-free or held by
#  the interface stiffness laws `Rn(t, t')`, `Rt(t, t')`.
#
#  ── Time-space decoupling ─────────────────────────────────────────────────
#
#  In isotropic elasticity the COD tensor of an elliptic crack is diagonal in
#  its frame (ℓ along the semi-axis a, m along b, n̂ the normal), with
#  α = 3k, β = 2μ and the elliptic functions 𝒞, 𝒮, ℰ of η (`_elliptic_CS`):
#       B_ℓℓ = 8/3 · (α + 2β) / (β [(α + 2β) 𝒞 + η² 𝒮 (2α + β)])
#       B_mm = 8/3 · (α + 2β) / (β [(α + 2β) η² 𝒮 + 𝒞 (2α + β)])
#       B_nn = 8/(3ℰ) · (α + 2β) / (β (2α + β))
#  In ALV every "/" becomes a Volterra inverse and every product a Volterra
#  product of n × n matrices, and their ORDER is not free: an aging matrix makes
#  α and β non-commuting. It is fixed by the flat limit of a void ellipsoid,
#  whose Hill kernel combines (k + 4μ/3)^{-∘} and μ^{-∘} alone, and it is
#  Echoes' (`compute_visco_crack_compliance`):
#       B̃ = 8/3 · L̃^{-∘} ∘ (α + 2β) ∘ β^{-∘},
#  L̃ the linear combination of α and β of each direction above.
#
#  The compliance contribution H̃ = (3/4) n̂ ⊗ˢ B̃ ⊗ˢ n̂ is stored in the class of
#  the crack: transversely isotropic about its normal for a penny, orthotropic
#  in its frame otherwise. Its blocks are those of the unit compliances of its
#  directions, rotated once as tensors (`_crack_compliance_operator`).
#
#  Apply [`delta_inverse_property_alv`](@ref) (= `(4π/3) · ε · H̃`) to convert
#  from the size-independent contribution to a fractional compliance
#  correction `ΔJ̃ = (4π/3) ε³ᵈ · H̃`.
# =============================================================================

"""
    cod_kernel_alv(crack::EllipticCrack, C_M_law::ALVProperty, times;
                   Rn = nothing, Rt = nothing) -> NamedTuple

Discrete ALV COD-tensor data of an elliptic crack (the penny included) in an
isotropic ALV matrix: the named tuple `(B_ℓ = …, B_m = …, B_n = …)` of the
``n\\times n`` Volterra matrices of its components in the crack frame, ``\\ell``
along the semi-axis ``a``, ``m`` along ``b``, ``n`` the normal (``n`` =
`length(times)` instants). For a penny ``\\widetilde{B}_\\ell = \\widetilde{B}_m``.

```math
\\widetilde{B}_{\\ell} = \\tfrac83\\,\\bigl((\\alpha + 2\\beta)\\,\\mathcal{C} + \\eta^2\\mathcal{S}\\,(2\\alpha + \\beta)\\bigr)^{-\\circ}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-\\circ},
\\qquad
\\widetilde{B}_{m} = \\tfrac83\\,\\bigl((\\alpha + 2\\beta)\\,\\eta^2\\mathcal{S} + \\mathcal{C}\\,(2\\alpha + \\beta)\\bigr)^{-\\circ}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-\\circ},
\\qquad
\\widetilde{B}_{n} = \\tfrac{8}{3\\,\\mathcal{E}}\\,(2\\alpha + \\beta)^{-\\circ}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-\\circ},
```

with ``\\alpha = 3k``, ``\\beta = 2\\mu`` the isotropic blocks of the matrix, and
``\\mathcal{C}``, ``\\mathcal{S}``, ``\\mathcal{E}`` the elliptic functions of
``\\eta = b/a`` of the elastic closed form: the elastic COD tensor with every
quotient a Volterra inverse, in the order of the flat limit of a void ellipsoid.

# Interface stiffness (Sevostianov-style spring-like interface)

When the crack carries finite **interface stiffness** kernels ``k_n(t,t')``
(normal) and ``k_t(t,t')`` (tangential), pass them as scalar `ViscoLaw`s
through the `Rn` / `Rt` keyword arguments.  The traction-free COD
matrices are then post-corrected via the algebraic identity

```math
\\widetilde{B}^{\\mathrm{int}}_{\\alpha}
= \\bigl(b\\,\\widetilde{k}_{\\alpha} + \\widetilde{B}_{\\alpha}^{-\\circ}\\bigr)^{-\\circ}
= \\widetilde{B}_{\\alpha}\\circ\\bigl(H + b\\,\\widetilde{k}_{\\alpha}\\circ\\widetilde{B}_{\\alpha}\\bigr)^{-\\circ},
```

[sevostianovIJSS2007, barthelemyIJES2019](@cite), with ``k_n`` on the normal
component and ``k_t`` on the two tangential ones, ``b`` the semi-minor axis
(`semi_minor`) of the crack.  Limits :

* `Rn / Rt = nothing` (default) → traction-free crack.
* ``k_n, k_t \\to \\infty`` (rigid bonding) → no opening.

Throws if the matrix law is not isotropic, or for a ribbon crack, whose ALV
opening is not implemented.
"""
function cod_kernel_alv(
        crack::MFH_Core.AbstractCrack, C_M_law::ALVProperty,
        times::AbstractVector{<:Real};
        Rn::Union{Nothing, ViscoLaw} = nothing,
        Rt::Union{Nothing, ViscoLaw} = nothing
    )
    α, β = _iso_blocks_alv(_relaxation_operator(C_M_law, TimeGrid(times)), "cod_kernel_alv")
    return _interface_corrected_cod(
        _cod_alv(crack, α, β),
        _trapezoidal_relaxation_scalar(Rn, times), _trapezoidal_relaxation_scalar(Rt, times),
        semi_minor(crack)
    )
end

# The isotropic blocks (α, β) of the matrix the crack opens in: the opening is
# known in an isotropic matrix only.
_iso_blocks_alv(C::VolterraOperator{<:TensISO{4, 3}}, who) = C.blocks
function _iso_blocks_alv(C::AbstractMatrix, who)
    _is_iso_block(C) ||
        throw(ArgumentError("$who: the crack opening is known in an isotropic ALV matrix only; this one is not"))
    return iso_params_from_blocks(C)
end

"""
    _cod_alv(crack, α, β) -> (B_ℓ, B_m, B_n)

The ``n\\times n`` Volterra COD coefficients of a traction-free elliptic crack in
an isotropic matrix of iso blocks ``\\alpha = 3k``, ``\\beta = 2\\mu``, in the crack
frame (see [`cod_kernel_alv`](@ref)). One expression for every aspect ratio: the
elliptic functions of `_elliptic_CS` are regular at the circle, where they give
the penny, ``\\widetilde B_\\ell = \\widetilde B_m``.
"""
function _cod_alv(crack::EllipticCrack, α::AbstractMatrix, β::AbstractMatrix)
    T = promote_type(eltype(α), eltype(β))
    η = Cracks._ratio_in(T, aspect_ratio(crack))
    𝒞, 𝒮, ℰ = Cracks._elliptic_CS(η)
    a₂ = α .+ 2β                    # α + 2β, i.e. 3(k + 4μ/3)
    b₂ = 2α .+ β                    # 2α + β, i.e. 6(k + μ/3)
    R = a₂ * volterra_inverse(β; block_size = 1)
    c = 8 * one(T) / 3
    return (
        B_ℓ = c .* volterra_left_divide(𝒞 .* a₂ .+ (η^2 * 𝒮) .* b₂, R; block_size = 1),
        B_m = c .* volterra_left_divide((η^2 * 𝒮) .* a₂ .+ 𝒞 .* b₂, R; block_size = 1),
        B_n = (c / ℰ) .* volterra_left_divide(b₂, R; block_size = 1),
    )
end
_cod_alv(crack::MFH_Core.AbstractCrack, α, β) = throw(
    ArgumentError("the ALV crack opening is implemented for elliptic cracks (the penny included); got a $(nameof(typeof(crack)))")
)

# The interface corrections of the three components: the normal stiffness on the
# opening, the tangential one on the two slidings.
_interface_corrected_cod(B, Rn, Rt, b) = (
    B_ℓ = _interface_corrected(B.B_ℓ, Rt, b),
    B_m = _interface_corrected(B.B_m, Rt, b),
    B_n = _interface_corrected(B.B_n, Rn, b),
)


# B̃ ∘ (𝟙 + b K̃ ∘ B̃)^{-∘}: the COD coefficient of a crack whose faces an interface
# of discretized stiffness K̃ holds; without one (`nothing`) the direction is
# traction-free and B̃ is left as it is.
_interface_corrected(B::AbstractMatrix, ::Nothing, b) = B
_interface_corrected(B::AbstractMatrix, K::AbstractMatrix, b) =
    B * volterra_inverse(LinearAlgebra.I + b * (K * B); block_size = 1)

# Build the scalar (n × n) trapezoidal matrix of an interface ViscoLaw,
# inverting if the law is in `:creep` mode (so the user can pass a creep
# kernel and the algebra still expects a relaxation matrix); none for none.
_trapezoidal_relaxation_scalar(::Nothing, ::AbstractVector{<:Real}) = nothing
_trapezoidal_relaxation_scalar(R::AbstractMatrix, times::AbstractVector{<:Real}) =
    _trapezoidal_relaxation(R, times, 1)
function _trapezoidal_relaxation_scalar(
        law::AbstractViscoLaw,
        times::AbstractVector{<:Real}
    )
    M = trapezoidal_matrix(law, times)
    visco_mode(law) === :creep && return volterra_inverse(M; block_size = 1)
    return M
end

"""
    inverse_property_contribution_alv(crack, C_M_law::ALVProperty, times; Rn, Rt) -> VolterraOperator

Size-independent compliance contribution ``\\widetilde{\\mathbb{H}}`` of an
elliptic crack (the penny included), of any orientation, in an isotropic ALV
matrix, from its COD coefficients ([`cod_kernel_alv`](@ref)):

```math
\\widetilde{\\mathbb{H}} = \\tfrac{3}{4}\\,\\underline{n}\\stackrel{s}{\\otimes}\\widetilde{\\boldsymbol{B}}\\stackrel{s}{\\otimes}\\underline{n},
```

stored in the class of the crack: transversely isotropic about its normal for a
penny, orthotropic in its frame otherwise.

Convention: same as the elastic [`inverse_property_contribution`](@ref MeanFieldHomogenization.Core.inverse_property_contribution) — the
Budiansky-O'Connell density factor is applied separately via
[`delta_inverse_property_alv`](@ref).
"""
function inverse_property_contribution_alv(
        crack::MFH_Core.AbstractCrack,
        C_M_law::ALVProperty,
        times::AbstractVector{<:Real};
        Rn::Union{Nothing, ViscoLaw} = nothing,
        Rt::Union{Nothing, ViscoLaw} = nothing
    )
    grid = TimeGrid(times)
    α, β = _iso_blocks_alv(_relaxation_operator(C_M_law, grid), "inverse_property_contribution_alv")
    return _crack_compliance_alv(
        crack, α, β, _trapezoidal_relaxation_scalar(Rn, times), _trapezoidal_relaxation_scalar(Rt, times), grid
    )
end

# H̃ of a crack against the isotropic blocks (α, β), its faces held by the
# interfaces of discretized stiffnesses `Rn`, `Rt` (`nothing`: traction-free).
_crack_compliance_alv(crack, α, β, Rn, Rt, grid) =
    _crack_compliance_operator(crack, _interface_corrected_cod(_cod_alv(crack, α, β), Rn, Rt, semi_minor(crack)), grid)

# H̃ = (3/4) n ⊗ˢ B̃ ⊗ˢ n in the class of the crack, as the sum of each component of
# B̃ times the blocks of its unit compliance: the unit compliances are tensors
# written in the crack frame, rotated once by TensND, not block by block.
# A penny: transversely isotropic about its normal, its two slidings equal.
function _crack_compliance_operator(crack::EllipticCrack{<:Number, Cracks.Penny}, B, grid)
    n̂ = Tuple(TensND.vecbasis(crack_basis(crack), :cov)[:, 3])
    c = TensTI{4}(1.0, 1.0, 0.0, 1.0, 1.0, n̂)
    return _operator_from_units(c, grid, (B.B_ℓ, _unit_compliance(crack, 1, 1, 0)), (B.B_n, _unit_compliance(crack, 0, 0, 1)))
end
# Any other ellipse: orthotropic in its frame.
function _crack_compliance_operator(crack::EllipticCrack{<:Number, Cracks.EllipticShape}, B, grid)
    c = TensOrtho{4, Float64}(ntuple(_ -> 1.0, 9), crack_basis(crack))
    return _operator_from_units(
        c, grid,
        (B.B_ℓ, _unit_compliance(crack, 1, 0, 0)), (B.B_m, _unit_compliance(crack, 0, 1, 0)), (B.B_n, _unit_compliance(crack, 0, 0, 1))
    )
end

# The compliance (3/4) n ⊗ˢ B ⊗ˢ n of the COD tensor diag(x, y, z) in the crack frame.
_unit_compliance(crack, x, y, z) =
    Cracks.compliance_from_cod(Cracks._diagonal_cod(float(x), float(y), float(z), crack_basis(crack)), crack)

# Σₖ Bₖ ⊗ (the blocks of the unit tensor Wₖ in the class c): an operator whose value
# at each pair of instants is Σₖ Bₖ[i, j] Wₖ.
function _operator_from_units(c, grid, pairs...)
    terms = map(((B, W),) -> map(w -> kron(B, Matrix(w)), _value_blocks(c, W)), pairs)
    return VolterraOperator(c, reduce((x, y) -> map(+, x, y), terms), grid)
end

"""
    delta_inverse_property_alv(crack, H̃, ε) -> Matrix

Apply the Budiansky-O'Connell crack density factor to the
size-independent compliance contribution ``\\widetilde{\\mathbb{H}}`` produced by
[`inverse_property_contribution_alv`](@ref), giving the fractional
compliance correction ``\\Delta\\widetilde{\\mathbb{L}} = \\tfrac{4\\pi}{3}\\,\\varepsilon^{3\\mathrm{d}}\\,\\widetilde{\\mathbb{H}}`` (penny / elliptic
geometry, ``\\varepsilon^{3\\mathrm{d}} = N\\,a\\,b^{2}``) — same pre-factor as the elastic case.
"""
function delta_inverse_property_alv(
        crack::MFH_Core.AbstractCrack,
        H̃::AbstractMatrix, ε::Real
    )
    # A product by a number, not a broadcast: an operator stored by blocks keeps them.
    return (Cracks.crack_density_factor(crack) * ε) * H̃
end

"""
    property_contribution_alv(crack, C_ref, times) -> Matrix{T}

Discrete ``6n\\times 6n`` crack **stiffness** contribution
``\\widetilde{\\mathbb{N}} = -\\widetilde{\\mathbb{C}}_0\\circ\\widetilde{\\mathbb{H}}\\circ\\widetilde{\\mathbb{C}}_0``,
mirror of the elastic [`property_contribution(crack, C₀)`] formula, with
``\\widetilde{\\mathbb{C}}_0`` the discretized `C_ref`.
`C_ref` may be a `ViscoLaw` (the matrix law — relaxation auto-built
through [`_trapezoidal_relaxation`](@ref)) or a pre-discretized
``6n\\times 6n`` reference matrix (used by SC iterations against the
running estimate `C_n`).
"""
function property_contribution_alv(
        crack::MFH_Core.AbstractCrack,
        C_M_law::ALVProperty,
        times::AbstractVector{<:Real}
    )
    H̃ = inverse_property_contribution_alv(crack, C_M_law, times)
    C̃_ref = _relaxation_operator(C_M_law, H̃.grid)
    return -(C̃_ref * H̃ * C̃_ref)
end

"""
    property_contribution_alv_at(crack, C_ref::AbstractMatrix; Rn_mat, Rt_mat) -> AbstractMatrix

The crack **stiffness** contribution
``\\widetilde{\\mathbb{N}} = -\\widetilde{\\mathbb{C}}_0\\circ\\widetilde{\\mathbb{H}}\\circ\\widetilde{\\mathbb{C}}_0``
against an isotropic reference given as its operator or its ``6n\\times 6n``
matrix (a running estimate), the interfaces given by their discretized
stiffnesses `Rn_mat`, `Rt_mat` (``n\\times n``, `nothing` for a traction-free
direction).
"""
function property_contribution_alv_at(
        crack::MFH_Core.AbstractCrack,
        C_ref::AbstractMatrix;
        Rn_mat::Union{Nothing, AbstractMatrix} = nothing,
        Rt_mat::Union{Nothing, AbstractMatrix} = nothing
    )
    α, β = _iso_blocks_alv(C_ref, "property_contribution_alv_at")
    H̃ = _crack_compliance_alv(crack, α, β, Rn_mat, Rt_mat, _operator_grid((C_ref,)))
    return -(C_ref * H̃ * C_ref)
end

"""
    delta_property_alv(crack, Ñ, ε) -> Matrix

Apply the Budiansky-O'Connell crack density factor to the
size-independent stiffness contribution ``\\widetilde{\\mathbb{N}}`` produced by
[`property_contribution_alv`](@ref), giving the dilute stiffness
correction ``\\Delta\\widetilde{\\mathbb{C}} = \\tfrac{4\\pi}{3}\\,\\varepsilon^{3\\mathrm{d}}\\,\\widetilde{\\mathbb{N}}`` (penny / elliptic) or
``\\pi\\,\\varepsilon^{2\\mathrm{d}}\\,\\widetilde{\\mathbb{N}}`` (ribbon).  Same pre-factors as the elastic case.
"""
function delta_property_alv(
        crack::MFH_Core.AbstractCrack,
        Ñ::AbstractMatrix, ε::Real
    )
    return (Cracks.crack_density_factor(crack) * ε) * Ñ
end
