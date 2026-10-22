# =============================================================================
#  moduli.jl
#
#  Extractors for the engineering moduli used by the analytical formulas in
#  the sub-modules `Elasticity`, `Cracks` and `Conductivity`.  The
#  routines only *read* components of the TensND tensor — they do not
#  perform any homogenization step.
# =============================================================================

"""
    extract_iso_moduli(C₀::TensISO{4,3}) -> (E, ν)

Extract Young's modulus `E` and Poisson's ratio `ν` from an isotropic
4th-order stiffness `TensISO{4,3}`.  The internal TensND convention is
``\\mathbb{C}_0 = 3k\\,\\mathbb{J} + 2\\mu\\,\\mathbb{K}`` i.e. `C₀.data = (3k, 2μ)`.
"""
function extract_iso_moduli(C₀::TensND.TensISO{4, 3})
    α, β = C₀.data           # α = 3k, β = 2μ
    k = α / 3
    μ = β / 2
    E = 9k * μ / (3k + μ)
    ν = (3k - 2μ) / (2 * (3k + μ))
    return E, ν
end

"""
    extract_iso_moduli(C₀::TensISO{4,2}) -> (E, ν)

2D plane-strain counterpart of [`extract_iso_moduli`](@ref) for
`TensISO{4,2}`.  The same 3D formulas are used because the TensND
storage is dimension-agnostic (`TensISO{4,d}` stores the same
``(\\alpha, \\beta) = (3k, 2\\mu)`` pair).
"""
function extract_iso_moduli(C₀::TensND.TensISO{4, 2})
    α, β = C₀.data
    k = α / 3
    μ = β / 2
    E = 9k * μ / (3k + μ)
    ν = (3k - 2μ) / (2 * (3k + μ))
    return E, ν
end

"""
    extract_ti_moduli(C₀, n̂) -> (E, H, ν₁, ν₂, Γ)

Read the five TI compliance moduli out of a stiffness tensor `C₀`
whose axis of symmetry is `n̂`.  Used by the closed-form COD formulas
of the `Cracks` sub-module.

The moduli are defined through the compliance tensor
``\\mathbb{S} = \\mathbb{C}_0^{-1}``, read in a frame whose third axis is the axis:

* ``E = 1/S_{1111}``
* ``H = 1/(S_{3333}\\,E)``
* ``\\nu_1 = -E\\,S_{1122}``
* ``\\nu_2 = -E\\,S_{1133}``
* ``\\Gamma = (1+\\nu_1)/(2\\,E\\,S_{2323})``

They do not depend on the axis: TensND's `arg_TI_Hoenig` reads them off the
Walpole coefficients of ``\\mathbb{S}``, exactly and with no change of basis (it
holds for symbolic moduli too). A `C₀` that is not a `TensTI` is first projected on
the transversely isotropic tensors about `n̂` (`best_fit_ti`).
"""
extract_ti_moduli(C₀::TensND.TensTI{4}, n̂) = _ti_moduli(C₀)
extract_ti_moduli(C₀::TensND.AbstractTens{4, 3}, n̂) =
    _ti_moduli(TensND.best_fit_ti(C₀, collect(TensND._extract_vec(n̂))))

function _ti_moduli(C₀::TensND.TensTI{4})
    E, ν₁, ν₂, H, Γ = TensND.arg_TI_Hoenig(inv(C₀))
    return (E = E, H = H, ν₁ = ν₁, ν₂ = ν₂, Γ = Γ)
end

"""
    extract_iso_conductivity(K₀::TensISO{2,d}) -> k

Extract the (scalar) conductivity coefficient of an isotropic 2nd-order
transport tensor `TensISO{2,d}` (``\\boldsymbol{K}_0 = k_0\\,\\boldsymbol{1}``).
"""
extract_iso_conductivity(K₀::TensND.TensISO{2, 3}) = K₀.data[1]
extract_iso_conductivity(K₀::TensND.TensISO{2, 2}) = K₀.data[1]
