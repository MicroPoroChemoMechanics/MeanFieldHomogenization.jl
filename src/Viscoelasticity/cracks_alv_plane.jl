# =============================================================================
#  cracks_alv_plane.jl — the crack of a two-dimensional cell in an isotropic
#  aging matrix of the plane.
#
#  The crack of a cell of the plane is the segment of half-length b that a
#  ribbon unbounded along the normal to the plane traces in it
#  (`RibbonCrack(b; angle)`, Cracks/plane.jl). In the isotropic elastic plane
#  α 𝕁 + β 𝕂 (α = 2k, β = 2μ) its COD tensor is χ 𝟏 (`Cracks._plane_cod`),
#
#      χ = (π/4)(1/μ + 1/k) = (π/2)(1/α + 1/β),
#
#  a sum of two inverses: in aging viscoelasticity each is a Volterra inverse,
#  and a sum leaves no order of factors to choose,
#
#      χ̃ = (π/2)(α^{-∘} + β^{-∘}).
#
#  That this is the flat limit of a void ellipse in the aging matrix, whose Hill
#  kernel is exact, is checked numerically by the tests, not proved here.
#  Springs on the faces hold the opening and the sliding as for the cracks of
#  space (`_interface_corrected`), and the compliance contribution is
#
#      ℍ̃ = χ̃ₙ ⊗ 𝕎ₙ + χ̃ₜ ⊗ 𝕎ₜ,
#
#  𝕎ₙ and 𝕎ₜ the elastic contributions (2/π) n ⊗ˢ 𝐁 ⊗ˢ n of the unit openings
#  𝐁 = n ⊗ n and 𝐁 = m ⊗ m (`Cracks._plane_compliance`), n the normal and m the
#  direction of the segment, χ̃ₙ and χ̃ₜ the opening and the sliding held by the
#  normal and the tangential springs.
# =============================================================================

# ℍ̃ of the segment against the isotropic blocks (α, β) of the plane, its faces held
# by the springs of discretized stiffnesses `Rn`, `Rt` (`nothing`: traction-free).
function _segment_compliance_alv(crack::RibbonCrack, α::AbstractMatrix, β::AbstractMatrix, Rn, Rt, grid)
    T = promote_type(eltype(α), eltype(β))
    χ = (T(π) / 2) .* (volterra_inverse(α; block_size = 1) .+ volterra_inverse(β; block_size = 1))
    n, m = Cracks._plane_normal(crack), Cracks._plane_direction(crack)
    Wₙ = Cracks._plane_compliance(crack, TensND.Tens(n * transpose(n)))
    Wₜ = Cracks._plane_compliance(crack, TensND.Tens(m * transpose(m)))
    b = semi_minor(crack)
    return _operator_from_units(_class_of(Wₙ), grid, (_interface_corrected(χ, Rn, b), Wₙ), (_interface_corrected(χ, Rt, b), Wₜ))
end
_segment_compliance_alv(crack::MFH_Core.AbstractCrack, α, β, Rn, Rt, grid) = Cracks._not_a_plane_crack(crack)
