# =============================================================================
#  cracks_alv.jl — elliptic and ribbon cracks in an isotropic ALV matrix.
#
#  Elliptic cracks of any aspect ratio η = b/a ≤ 1 (the penny, η = 1, included)
#  and ribbons (their limit η → 0), of any orientation, in an **isotropic ALV
#  matrix**, traction-free or held by the interface stiffness laws `Rn(t, t')`,
#  `Rt(t, t')`.
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
#  The compliance contribution H̃ = k n̂ ⊗ˢ B̃ ⊗ˢ n̂ (k = 3/4 for an ellipse, 2/π
#  for a ribbon) is stored in the class of the crack: transversely isotropic
#  about its normal for a penny, orthotropic in its frame otherwise. Its blocks are those of the unit compliances of its
#  directions, rotated once as tensors (`_crack_compliance_operator`).
#
#  Apply [`delta_inverse_property_alv`](@ref) (= `(4π/3) · ε · H̃`) to convert
#  from the size-independent contribution to a fractional compliance
#  correction `ΔJ̃ = (4π/3) ε³ᵈ · H̃`.
# =============================================================================

"""
    cod_kernel_alv(crack::EllipticCrack, C_M_law::ALVProperty, times;
                   Rn = nothing, Rt = nothing) -> NamedTuple

Discrete ALV COD-tensor data of an elliptic crack (the penny included) or of a
ribbon in an isotropic ALV matrix: the named tuple `(B_ℓ = …, B_m = …, B_n = …)` of the
``n\\times n`` Volterra matrices of its components in the crack frame, ``\\ell``
along the semi-axis ``a``, ``m`` along ``b``, ``n`` the normal (``n`` =
`length(times)` instants). For a penny ``\\widetilde{B}_\\ell = \\widetilde{B}_m``.

```math
\\widetilde{B}_{\\ell} = \\tfrac83\\,\\bigl((\\alpha + 2\\beta)\\,\\mathcal{C} + \\eta^2\\mathcal{S}\\,(2\\alpha + \\beta)\\bigr)^{-1^{\\circ}}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-1^{\\circ}},
\\qquad
\\widetilde{B}_{m} = \\tfrac83\\,\\bigl((\\alpha + 2\\beta)\\,\\eta^2\\mathcal{S} + \\mathcal{C}\\,(2\\alpha + \\beta)\\bigr)^{-1^{\\circ}}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-1^{\\circ}},
\\qquad
\\widetilde{B}_{n} = \\tfrac{8}{3\\,\\mathcal{E}}\\,(2\\alpha + \\beta)^{-1^{\\circ}}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-1^{\\circ}},
```

with ``\\alpha = 3k``, ``\\beta = 2\\mu`` the isotropic blocks of the matrix, and
``\\mathcal{C}``, ``\\mathcal{S}``, ``\\mathcal{E}`` the elliptic functions of
``\\eta = b/a`` of the elastic closed form: the elastic COD tensor with every
quotient a Volterra inverse, in the order of the flat limit of a void ellipsoid.

# Interface stiffness (spring-like interface, approximate)

When the crack carries finite **interface stiffness** kernels ``k_n(t,t')``
(normal) and ``k_t(t,t')`` (tangential), pass them as scalar `ViscoLaw`s
through the `Rn` / `Rt` keyword arguments.  The traction-free COD
matrices are then post-corrected via the algebraic identity

```math
\\widetilde{B}^{\\mathrm{int}}_{\\alpha}
= \\bigl(b\\,\\widetilde{k}_{\\alpha} + \\widetilde{B}_{\\alpha}^{-1^{\\circ}}\\bigr)^{-1^{\\circ}}
= \\widetilde{B}_{\\alpha}\\circ\\bigl(H + b\\,\\widetilde{k}_{\\alpha}\\circ\\widetilde{B}_{\\alpha}\\bigr)^{-1^{\\circ}},
```

with ``k_n`` on the normal component and ``k_t`` on the two tangential ones,
``b`` the semi-minor axis (`semi_minor`) of the crack: the compliance of the springs
added to that of the faces, as if they saw the average opening. An approximation,
exact in its limits but not derived from the integral equation of a crack with
spring-like faces. Limits:

* `Rn / Rt = nothing` (default) → traction-free crack.
* ``k_n, k_t \\to \\infty`` (rigid bonding) → no opening.

# Ribbon

For a [`RibbonCrack`](@ref) (``\\ell`` the unbounded direction, ``m`` across the
half-width ``b``), the ribbon limit of the same closed form:

```math
\\widetilde{B}_{\\ell} = \\pi\\,\\beta^{-1^{\\circ}},
\\qquad
\\widetilde{B}_{m} = \\widetilde{B}_{n} = \\pi\\,(2\\alpha + \\beta)^{-1^{\\circ}}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-1^{\\circ}}.
```

Throws if the matrix law is not isotropic.
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
"""
    _cod_alv(crack::RibbonCrack, α, β) -> (B_ℓ, B_m, B_n)

The ribbon (``\\ell`` the unbounded direction, ``m`` across its half-width ``b``):
the elastic closed form ``B_{\\ell\\ell} = \\pi/(2\\mu)``,
``B_{mm} = B_{nn} = \\pi(1-\\nu)/(2\\mu)``, that is, with ``1-\\nu = (\\alpha+2\\beta)/(2\\alpha+\\beta)``,

```math
\\widetilde{B}_{\\ell} = \\pi\\,\\beta^{-1^{\\circ}},
\\qquad
\\widetilde{B}_{m} = \\widetilde{B}_{n} = \\pi\\,(2\\alpha + \\beta)^{-1^{\\circ}}\\circ(\\alpha + 2\\beta)\\circ\\beta^{-1^{\\circ}},
```

the limit ``\\eta \\to 0`` of the elliptic expressions times ``3\\pi/8`` (the ratio of
the density factors ``\\pi`` and ``4\\pi/3`` and of the normalizations of ``B``),
in the same order of the factors.
"""
function _cod_alv(::RibbonCrack, α::AbstractMatrix, β::AbstractMatrix)
    T = promote_type(eltype(α), eltype(β))
    J_β = volterra_inverse(β; block_size = 1)
    B_n = T(π) .* volterra_left_divide(2α .+ β, (α .+ 2β) * J_β; block_size = 1)
    return (B_ℓ = T(π) .* J_β, B_m = B_n, B_n = B_n)
end
# A flowing crack opens as the elliptic crack it wraps.
_cod_alv(crack::Cracks.ConductiveCrack, α::AbstractMatrix, β::AbstractMatrix) = _cod_alv(crack.crack, α, β)
_cod_alv(crack::MFH_Core.AbstractCrack, α, β) = throw(
    ArgumentError("the ALV crack opening is implemented for elliptic cracks (the penny included) and ribbons; got a $(nameof(typeof(crack)))")
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
    inverse_property_contribution_alv(crack, C_M_law::ALVProperty, times;
                                      Rn, Rt, α_interface) -> VolterraOperator

Size-independent compliance contribution ``\\widetilde{\\mathbb{H}}`` of an
elliptic crack (the penny included) or of a ribbon, of any orientation, in an
isotropic ALV matrix, from its COD coefficients ([`cod_kernel_alv`](@ref)):

```math
\\widetilde{\\mathbb{H}} = k\\,\\underline{n}\\stackrel{s}{\\otimes}\\widetilde{\\boldsymbol{B}}\\stackrel{s}{\\otimes}\\underline{n},
```

with ``k = 3/4`` for an ellipse and ``k = 2/\\pi`` for a ribbon, as in elasticity
([`compliance_from_cod`](@ref MeanFieldHomogenization.Cracks.compliance_from_cod)), stored in the class of the crack: transversely
isotropic about its normal for a penny, orthotropic in its frame otherwise.

Convention: same as the elastic [`inverse_property_contribution`](@ref MeanFieldHomogenization.Core.inverse_property_contribution) — the
Budiansky-O'Connell density factor is applied separately via
[`delta_inverse_property_alv`](@ref).

# Conduction

In an isotropic conductor of ALV conductivity ``\\widetilde{k}_0\\,\\boldsymbol{1}`` (a
matrix law of order 2), the crack is insulating and the contribution is the
**resistivity** contribution, transversely isotropic about the normal,

```math
\\widetilde{\\boldsymbol{R}} = k\\,\\widetilde{b}\\;\\underline{n}\\otimes\\underline{n},
\\qquad
\\widetilde{b} = \\frac{4}{3\\,\\mathcal{E}}\\,\\widetilde{k}_0^{-1^{\\circ}}\\ \\ (\\text{ellipse}),
\\qquad
\\widetilde{b} = \\frac{\\pi}{2}\\,\\widetilde{k}_0^{-1^{\\circ}}\\ \\ (\\text{ribbon}),
```

the elastic thermal COD ``b`` of [`cod_tensor`](@ref MeanFieldHomogenization.Cracks.cod_tensor) with ``1/k_0`` the Volterra
inverse of the conductivity kernel. A Kapitza-like interface of discretized
conductance `α_interface` (a scalar `ViscoLaw`) corrects it as in elasticity,
``\\widetilde{b}\\circ(\\mathbb{1} + b\\,\\widetilde{\\alpha}\\circ\\widetilde{b})^{-1^{\\circ}}``.
`Rn` and `Rt` are elastic interfaces and `α_interface` a conduction one: each is
refused at the other order.
"""
function inverse_property_contribution_alv(
        crack::MFH_Core.AbstractCrack,
        C_M_law::ALVProperty,
        times::AbstractVector{<:Real};
        Rn::Union{Nothing, ViscoLaw} = nothing,
        Rt::Union{Nothing, ViscoLaw} = nothing,
        α_interface::Union{Nothing, ViscoLaw} = nothing
    )
    C_0 = _relaxation_operator(C_M_law, TimeGrid(times))
    interfaces = map(R -> _trapezoidal_relaxation_scalar(R, times), (; Rn, Rt, α_interface))
    return _crack_inverse_alv(_alv_order(C_0, times), crack, C_0, interfaces, "inverse_property_contribution_alv")
end

# The crack contribution against an isotropic reference `C_0` (its operator or its
# full matrix) of either order, the interfaces discretized (`nothing` for none):
# the compliance H̃ in elasticity, the resistivity R̃ in conduction.
function _crack_inverse_alv(::Val{4}, crack, C_0, interfaces, who)
    interfaces.α_interface === nothing ||
        throw(ArgumentError("$who: α_interface is the interface of a crack in conduction; in elasticity it is Rn and Rt"))
    α, β = _iso_blocks_alv(C_0, who)
    return _crack_compliance_alv(crack, α, β, interfaces.Rn, interfaces.Rt, _operator_grid((C_0,)))
end
function _crack_inverse_alv(::Val{2}, crack, C_0, interfaces, who)
    interfaces.Rn === interfaces.Rt === nothing ||
        throw(ArgumentError("$who: Rn and Rt are the interfaces of a crack in elasticity; in conduction it is α_interface"))
    return _crack_resistivity_alv(crack, _iso_order2_block_alv(C_0, who), interfaces.α_interface, _operator_grid((C_0,), 3))
end

# The conductivity kernel λ of the isotropic conductor the crack opens in.
_iso_order2_block_alv(K::VolterraOperator{<:TensISO{2, 3}}, who) = only(K.blocks)
function _iso_order2_block_alv(K::AbstractMatrix, who)
    _is_iso_order2_block(K) ||
        throw(ArgumentError("$who: the crack resistivity is known in an isotropic ALV conductor only; this one is not"))
    return iso_order2_params_from_blocks(K)
end

# R̃ = k b̃ n ⊗ n, transversely isotropic about the normal, b̃ held by the interface
# of discretized conductance `A` (`nothing`: insulating faces).
function _crack_resistivity_alv(crack, λ::AbstractMatrix, A, grid)
    b̃ = _interface_corrected(_cod_alv2(crack, λ), A, semi_minor(crack))
    n̂ = Tuple(TensND.vecbasis(crack_basis(crack), :cov)[:, 3])
    return _operator_from_units(TensTI{2}(1.0, 1.0, n̂), grid, (b̃, Cracks._resistivity_from_b(crack, 1.0, nothing)))
end

# The scalar thermal COD b̃ of an insulating crack: the elastic b with 1/k₀ the
# Volterra inverse of the conductivity kernel.
function _cod_alv2(crack::EllipticCrack, λ::AbstractMatrix)
    T = eltype(λ)
    ℰ = Cracks._elliptic_CS(Cracks._ratio_in(T, aspect_ratio(crack)))[3]
    return (4 / (3ℰ)) .* volterra_inverse(λ; block_size = 1)
end
_cod_alv2(::RibbonCrack, λ::AbstractMatrix) = (eltype(λ)(π) / 2) .* volterra_inverse(λ; block_size = 1)
_cod_alv2(crack::MFH_Core.AbstractCrack, λ) = throw(
    ArgumentError(
        "the ALV crack resistivity is implemented for insulating elliptic cracks (the penny included) and " *
            "ribbons; got a $(nameof(typeof(crack)))"
    )
)

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
# Any other ellipse, and a ribbon: orthotropic in its frame.
function _crack_compliance_operator(crack::Union{EllipticCrack{<:Number, Cracks.EllipticShape}, RibbonCrack}, B, grid)
    c = TensOrtho{4, Float64}(ntuple(_ -> 1.0, 9), crack_basis(crack))
    return _operator_from_units(
        c, grid,
        (B.B_ℓ, _unit_compliance(crack, 1, 0, 0)), (B.B_m, _unit_compliance(crack, 0, 1, 0)), (B.B_n, _unit_compliance(crack, 0, 0, 1))
    )
end

_crack_compliance_operator(crack::Cracks.ConductiveCrack, B, grid) = _crack_compliance_operator(crack.crack, B, grid)

# The compliance k n ⊗ˢ B ⊗ˢ n of the COD tensor diag(x, y, z) in the crack frame.
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
geometry, ``\\varepsilon^{3\\mathrm{d}} = N\\,a\\,b^{2}``) or ``\\pi\\,\\varepsilon^{2\\mathrm{d}}\\,\\widetilde{\\mathbb{H}}`` (ribbon,
``\\varepsilon^{2\\mathrm{d}} = N\\,b^{2}`` per unit area) — same pre-factors as the elastic case.
"""
function delta_inverse_property_alv(
        crack::MFH_Core.AbstractCrack,
        H̃::AbstractMatrix, ε::Real
    )
    # A product by a number, not a broadcast: an operator stored by blocks keeps them.
    return (Cracks.crack_density_factor(crack) * ε) * H̃
end

"""
    property_contribution_alv(crack, C_ref, times) -> VolterraOperator

Crack **stiffness** contribution
``\\widetilde{\\mathbb{N}} = -\\widetilde{\\mathbb{C}}_0\\stackrel{\\circ}{:}\\widetilde{\\mathbb{H}}\\stackrel{\\circ}{:}\\widetilde{\\mathbb{C}}_0``,
mirror of the elastic `property_contribution(crack, C₀)` formula, with
``\\widetilde{\\mathbb{C}}_0`` the discretized `C_ref` (a law, a relaxation
operator or an elastic tensor) and ``\\widetilde{\\mathbb{H}}`` from
[`inverse_property_contribution_alv`](@ref); in conduction, the conductivity
contribution ``-\\widetilde{\\boldsymbol{K}}_0\\stackrel{\\circ}{\\cdot}\\widetilde{\\boldsymbol{R}}\\stackrel{\\circ}{\\cdot}\\widetilde{\\boldsymbol{K}}_0``
of an insulating crack.
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
``\\widetilde{\\mathbb{N}} = -\\widetilde{\\mathbb{C}}_0\\stackrel{\\circ}{:}\\widetilde{\\mathbb{H}}\\stackrel{\\circ}{:}\\widetilde{\\mathbb{C}}_0``
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
