# =============================================================================
#  fracture_compliance.jl — the compliance of linear slip of a family of
#  cracks, read on the compliance that a scheme gives it.
#
#  A set of parallel fractures of normal n adds to the compliance of the
#  medium the change of a linear-slip interface (Schoenberg and Sayers 1995):
#  the strain of the fractures is the symmetric part of n ⊗ [u], the jump
#  [u] = Z·(σ·n) proportional to the traction,
#
#       Δ𝕊 = sym(n ⊗ Z ⊗ n),   ΔS_ijkl = ¼(n_i Z_jk n_l + n_j Z_ik n_l + n_i Z_jl n_k + n_j Z_il n_k),
#
#  with Z symmetric. A family of flat cracks contributes exactly such a change,
#  whatever the scheme: its share of the macroscopic compliance
#  (`crack_family_compliances`) is the Δ𝕊 from which Z is read. With
#  M = n·Δ𝕊·n,
#
#       M = ¼(Z + n ⊗ Zn + Zn ⊗ n + Z_N n ⊗ n),   Z_N = n·Z·n = n·M·n ,
#       Zn = 2 M·n − Z_N n ,   Z = 4M − n ⊗ Zn − Zn ⊗ n − Z_N n ⊗ n .
# =============================================================================

"""
    slip_compliance(Δ𝕊, n) → Z

The compliance ``\\boldsymbol{Z}`` of the linear-slip interface of normal `n`
whose change of compliance is ``\\Delta\\mathbb{S}``,
``\\Delta\\mathbb{S} = \\operatorname{sym}(\\underline{n}\\otimes\\boldsymbol{Z}\\otimes\\underline{n})``:
the jump of displacement across the fractures is
``\\boldsymbol{Z}\\cdot\\boldsymbol{\\sigma}\\cdot\\underline{n}``. Read with
``\\boldsymbol{M} = \\underline{n}\\cdot\\Delta\\mathbb{S}\\cdot\\underline{n}``,
``Z_N = \\underline{n}\\cdot\\boldsymbol{M}\\cdot\\underline{n}`` and
``\\boldsymbol{Z}\\cdot\\underline{n} = 2\\boldsymbol{M}\\cdot\\underline{n} - Z_N\\underline{n}``.
`n` is a TensND vector or the components of a unit vector. Numbers, dual
numbers and symbols.
"""
slip_compliance(ΔS::TensND.AbstractTens{4, 3}, n::TensND.AbstractTens{1, 3}) =
    Tens(_slip_compliance(minorsymmetric(Tensor{4, 3}(Array(canonical_components(ΔS)))), _vec(n)))
slip_compliance(ΔS::TensND.AbstractTens{4, 3}, n::AbstractVector) = slip_compliance(ΔS, Tens(_unit(n)))

# On the canonical components, with the kernels of Tensors.jl: `dotdot`
# contracts the first and fourth indices, the same as the second and fourth
# for the minor-symmetric Δ𝕊.
function _slip_compliance(ΔS::SymmetricTensor{4, 3}, n::Vec{3})
    M = dotdot(n, ΔS, n)
    Z_N = n ⋅ M ⋅ n
    a = 2 * (M ⋅ n) - Z_N * n
    return 4 * M - otimes(n, a) - otimes(a, n) - Z_N * otimes(n)
end

"""
    fracture_compliance(rve, scheme, family; property = :C, C_hom = homogenize(rve, scheme, property), kw...)
        → (Z, Z_N, Z_T, normal)

The compliance of linear slip of the crack `family` of `rve`, as `scheme` sees
it: its share of the macroscopic compliance
([`crack_family_compliances`](@ref)) written as the change of a linear-slip
interface ([`slip_compliance`](@ref)). `Z` is the second-order tensor, `Z_N`
its normal part ``\\underline{n}\\cdot\\boldsymbol{Z}\\cdot\\underline{n}``, `Z_T` the
mean of its two tangential eigenvalues, ``(\\operatorname{tr}\\boldsymbol{Z} - Z_N)/2``,
the tangential compliance of a family invariant by rotation about its normal,
and `normal` the normal of the family. `C_hom` is the estimate of `scheme` for
`rve`; it is computed when not given. The schemes are those of
[`crack_family_compliances`](@ref).
"""
function fracture_compliance(
        rve::RVE, scheme::HomogenizationScheme, family::Symbol; property::Symbol = :C,
        C_hom = homogenize(rve, scheme, property), kw...
    )
    shares = crack_family_compliances(rve, scheme, C_hom; property, kw...).families
    haskey(shares, family) || throw(
        ArgumentError("no crack family :$(family) in this cell; its crack phases are $(collect(keys(shares)))")
    )
    geometry = rve.phases[family].geometry
    n = crack_normal(geometry)
    Z = slip_compliance(delta_inverse_property(geometry, shares[family], crack_density(rve, family)), n)
    Z_N = n ⋅ Z ⋅ n
    return (Z = Z, Z_N = Z_N, Z_T = (tr(Z) - Z_N) / 2, normal = n)
end
