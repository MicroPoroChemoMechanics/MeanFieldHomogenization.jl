# =============================================================================
#  alv_order.jl — one function per notion in aging viscoelasticity, whatever
#  the tensor order, and the names of elasticity and of conduction.
#
#  A reference of the ALV pipeline does not carry its order in its type: a
#  `ViscoLaw` may return a fourth-order tensor, a second-order one or a Mandel
#  matrix, and a Volterra matrix is a plain matrix. `_alv_order` reads the
#  order once — from the law at the first instant, from the size of the matrix —
#  and the generic functions dispatch on it, behind that function barrier.
# =============================================================================

_alv_order(law::AbstractViscoLaw, times::AbstractVector{<:Real}) = Val(_alv_property_order(law, first(times)))
_alv_order(::TensND.AbstractTens{N}, ::AbstractVector{<:Real}) where {N} = Val(N)
function _alv_order(M::AbstractMatrix, times::AbstractVector{<:Real})
    n = length(times)
    size(M, 1) == 6n && return Val(4)
    size(M, 1) == 3n && return Val(2)
    throw(
        ArgumentError(
            "an ALV reference matrix has 6n rows in elasticity and 3n in conduction " *
                "(n = $(n) instants); this one has $(size(M, 1))"
        )
    )
end

"""
    gradient_gradient_concentration_alv(sphere, ref, times; external = true) -> Matrix
    gradient_gradient_concentration_alv(sphere, ref, times; layer = k, external = false, internal = false) -> Matrix

Volume-averaged gradient concentration of a whole [`LayeredSphere`](@ref) in an
isotropic aging reference `ref` (a `ViscoLaw`, or the Volterra matrix of a running
estimate), under a unit Volterra far-field gradient. The order of `ref` selects
the physics.

In elasticity, the ``6n\\times 6n`` block matrix of the strain concentration,
``\\langle\\widetilde{\\mathbb{A}}\\rangle = \\langle\\alpha\\rangle\\,\\mathbb{J} + \\langle\\beta\\rangle\\,\\mathbb{K}`` with
``\\langle\\alpha\\rangle = \\sum_k f_k\\,\\alpha_k(t,t') + \\Delta\\alpha`` and
``\\langle\\beta\\rangle = \\sum_k f_k\\,\\beta_k(t,t') + \\Delta\\beta`` (Volterra products),
``\\Delta\\alpha``, ``\\Delta\\beta`` being the displacement jumps of the spring
interfaces. In conduction, the ``3n\\times 3n`` gradient concentration
``\\sum_k f_k\\,\\alpha_k + \\Delta\\alpha``, ``\\Delta\\alpha`` adding the temperature jumps
``[\\![T]\\!] = -\\rho\\,q_n`` of the Kapitza interfaces. `external` decides whether the
outer interface belongs to the sphere, as in the elastic
[`gradient_gradient_concentration`](@ref).

With `layer = k` the operator is that of layer ``k``,
``\\widetilde{\\mathbb A}_k = \\alpha_k\\,\\mathbb J + \\beta_k\\,\\mathbb K`` (in conduction
``\\alpha_k\\,\\boldsymbol 1``): the average over its material by default, plus the
jump across its outer boundary ``r_k`` when `external` and across its inner one
``r_{k-1}`` when `internal`, each divided by the volume of the layer — nonzero
across a spring (Kapitza) interface only, as in the elastic
[`gradient_gradient_concentration`](@ref)`(sphere, C₀; layer)`. With
`external = true` on every layer the shells partition the sphere:
``\\sum_k f_k\\,\\widetilde{\\mathbb A}_k`` is the operator of the whole sphere.

This is what the ALV dilute / Mori–Tanaka / Maxwell schemes use when an
inclusion phase is a `LayeredSphere`. Explicit name in elasticity:
[`strain_strain_concentration_alv`](@ref).
"""
gradient_gradient_concentration_alv(
    sphere::_LayeredALV, ref::_ALVReference, times::AbstractVector{<:Real}; kw...
) = _gradient_gradient_concentration_alv(_alv_order(ref, times), sphere, ref, times; kw...)

"""
    property_contribution_alv(sphere, ref, times; external = true) -> Matrix

Size-independent ALV contribution of an inclusion to the property of an aging
reference `ref`: the dilute effective property at volume fraction ``f`` is
``\\widetilde{\\mathbb{C}}_0 + f\\,\\widetilde{\\mathbb{N}}``.

For a [`LayeredSphere`](@ref) the order of `ref` selects the physics. In
elasticity the ``6n\\times 6n`` stiffness contribution has the iso parameters

```math
\\begin{aligned}
\\alpha &= 3\\sum_k f_k\\,(\\widetilde{k}_k - \\widetilde{k}_0)\\circ\\alpha_k - 3\\,\\widetilde{k}_0\\circ\\Delta\\alpha,
\\\\
\\beta &= 2\\sum_k f_k\\,(\\widetilde{\\mu}_k - \\widetilde{\\mu}_0)\\circ\\beta_k - 2\\,\\widetilde{\\mu}_0\\circ\\Delta\\beta,
\\end{aligned}
```

``\\alpha_k``, ``\\beta_k`` being the per-layer concentration matrices, ``\\widetilde{k}_k``,
``\\widetilde{\\mu}_k`` the per-layer Volterra moduli and ``\\Delta\\alpha``, ``\\Delta\\beta`` the
displacement jumps of the spring interfaces, plus the surface stress of the
membranes. In conduction the ``3n\\times 3n`` conductivity contribution is
``\\sum_k f_k\\,(k_k - k_0)\\circ\\alpha_k - k_0\\circ\\Delta\\alpha`` plus the surface flux of the
surface-conductive interfaces. `ref` may also be the Volterra matrix of a
running estimate, which is how the self-consistent and differential schemes
call it; `external` has the meaning it has in
[`gradient_gradient_concentration_alv`](@ref).

The crack method, documented next, is the ALV contribution of a crack family,
its stiffness contribution in elasticity and its conductivity contribution in
conduction. Explicit names: [`stiffness_contribution_alv`](@ref) and
[`conductivity_contribution_alv`](@ref).
"""
property_contribution_alv(
    sphere::_LayeredALV, ref::_ALVReference, times::AbstractVector{<:Real}; kw...
) = _property_contribution_alv(_alv_order(ref, times), sphere, ref, times; kw...)

"""
    flux_gradient_concentration_alv(sphere::LayeredSphere, ref, times; layer = nothing, external = true) -> Matrix

Volterra block matrix of the average flux of a [`LayeredSphere`](@ref) per unit
remote gradient history, against the isotropic aging reference `ref`: in
elasticity the average stress per unit remote strain, in conduction the average
of ``k\\,\\nabla T`` per unit remote gradient. Without `layer`, the whole sphere,
``\\sum_k f_k\\,\\widetilde{\\mathbb C}_k\\circ\\widetilde{\\mathbb A}_k`` plus the surface stress
(flux) of its membranes (conducting surfaces), that of the outer one when
`external`: [`property_contribution_alv`](@ref) is this operator minus
``\\widetilde{\\mathbb C}_0\\circ\\langle\\widetilde{\\mathbb A}\\rangle``. With `layer = k`, the
material of layer ``k``: its relaxation operator applied **on the left** of its
average gradient, ``\\widetilde{\\mathbb C}_k\\circ\\widetilde{\\mathbb A}_k``.
"""
function flux_gradient_concentration_alv(
        sphere::LayeredSphere, ref::_ALVReference, times::AbstractVector{<:Real};
        layer::Union{Nothing, Int} = nothing, external::Bool = true,
    )
    return _flux_gradient_concentration_alv(_alv_order(ref, times), sphere, ref, times, layer, external)
end

# ── The names of elasticity and of conduction ────────────────────────────────

# The order of the reference, or an error naming the generic function.
function _alv_checked_order(::Val{N}, alias::Symbol, generic::Symbol, ref, times) where {N}
    _alv_order(ref, times) === Val(N) || throw(
        ArgumentError(
            "`$(alias)` is the name of `$(generic)` for references of order $(N), and " *
                "this one is not of that order; call `$(generic)`, which serves every order"
        )
    )
    return nothing
end

"""
    strain_strain_concentration_alv(sphere, C0_law, times; external = true) -> Matrix

Name in elasticity of [`gradient_gradient_concentration_alv`](@ref), for a
fourth-order reference only.
"""
function strain_strain_concentration_alv(sphere, ref, times; kw...)
    _alv_checked_order(Val(4), :strain_strain_concentration_alv, :gradient_gradient_concentration_alv, ref, times)
    return gradient_gradient_concentration_alv(sphere, ref, times; kw...)
end

"""
    stiffness_contribution_alv(sphere_or_crack, C0_law, times; kw...) -> Matrix

Name in elasticity of [`property_contribution_alv`](@ref), for a fourth-order
reference only.
"""
function stiffness_contribution_alv(incl, ref, times; kw...)
    _alv_checked_order(Val(4), :stiffness_contribution_alv, :property_contribution_alv, ref, times)
    return property_contribution_alv(incl, ref, times; kw...)
end

"""
    conductivity_contribution_alv(sphere_or_crack, K0_law, times; kw...) -> Matrix

Name in conduction of [`property_contribution_alv`](@ref), for a second-order
reference only.
"""
function conductivity_contribution_alv(sphere, ref, times; kw...)
    _alv_checked_order(Val(2), :conductivity_contribution_alv, :property_contribution_alv, ref, times)
    return property_contribution_alv(sphere, ref, times; kw...)
end

"""
    stiffness_contribution_alv_at(sphere_or_crack, C_ref, args...; kw...) -> Matrix

Name in elasticity of [`property_contribution_alv_at`](@ref), whose methods are
all elastic.
"""
stiffness_contribution_alv_at(args...; kw...) = property_contribution_alv_at(args...; kw...)

"""
    compliance_contribution_alv(crack, C_M_law, times; Rn, Rt) -> Matrix

Name in elasticity of [`inverse_property_contribution_alv`](@ref), for a
fourth-order reference only.
"""
function compliance_contribution_alv(crack, ref, times; kw...)
    _alv_checked_order(Val(4), :compliance_contribution_alv, :inverse_property_contribution_alv, ref, times)
    return inverse_property_contribution_alv(crack, ref, times; kw...)
end

"""
    resistivity_contribution_alv(crack, K_M_law, times; α_interface) -> Matrix

Name in conduction of [`inverse_property_contribution_alv`](@ref), for a
second-order reference only.
"""
function resistivity_contribution_alv(crack, ref, times; kw...)
    _alv_checked_order(Val(2), :resistivity_contribution_alv, :inverse_property_contribution_alv, ref, times)
    return inverse_property_contribution_alv(crack, ref, times; kw...)
end

"""
    delta_stiffness_alv(crack, Ñ, ε) -> Matrix

Name in elasticity of [`delta_property_alv`](@ref), which applies the density
factor at either order.
"""
delta_stiffness_alv(args...) = delta_property_alv(args...)

"""
    delta_compliance_alv(crack, H̃, ε) -> Matrix

Name in elasticity of [`delta_inverse_property_alv`](@ref), which applies the
density factor at either order.
"""
delta_compliance_alv(args...) = delta_inverse_property_alv(args...)
