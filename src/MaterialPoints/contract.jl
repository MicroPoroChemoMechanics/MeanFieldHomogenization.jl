# =============================================================================
#  contract.jl — a law is a material of the Gauss-point contract.
#
#  `material_response` takes and returns TensND tensors in the canonical frame
#  and plain scalars, by name, the language of every caller of the contract;
#  `point_update` works on the flat static vectors the algorithm differentiates.
#  The conversion is done here and nowhere else: a size-6 field is a symmetric
#  second-order tensor in Kelvin–Mandel components, a size-1 field a scalar.
#  The tangent comes back as the blocks the contract names flux then gradient
#  (σε, σp, φε, φp), each converted by its shape.
# =============================================================================

"""
    material_response(law::AbstractPointLaw, gradients, prev, Δt; algorithm = LocalNewton(), external = (;), cache = nothing)

The Gauss-point contract for a law of
[`MaterialPoints`](@ref MeanFieldHomogenization.MaterialPoints):
[`point_update`](@ref) with the gradients given and the fluxes returned by name,
tensors as TensND tensors in the canonical frame and scalars as numbers, and
the tangent as the blocks `σε`, `σp`, `φε`, … (flux then gradient). `external`
gives the variables the law reads without returning their flux, as for
[`point_update`](@ref). `cache` is
accepted for the contract and unused: a law has nothing to memoize across
points.
"""
function Constitutive.material_response(
        law::AbstractPointLaw, gradients::NamedTuple, prev::PointState, Δt::Real;
        algorithm::AbstractLocalAlgorithm = LocalNewton(), external::NamedTuple = (;), cache = nothing
    )
    g = _flat(map(_field_components, NamedTuple{keys(law.gradients)}(gradients)), law.gradients)
    F, 𝕂, state = _point_update(_with_external(law, external), g, prev, Δt, algorithm)
    fluxes = map(r -> _field_value(F[r]), law.fluxes)
    blocks = _tangent_blocks(𝕂, law.fluxes, law.gradients)
    return MaterialResponse(fluxes, blocks, state)
end

# A field given to the contract, as the algorithm reads it.
_field_components(x::Number) = x
_field_components(t::TensND.AbstractTens{2, 3}) = _mandel(t)
_field_components(v::StaticVector) = v

# A field returned by the contract.
_field_value(x::Number) = x
_field_value(v::StaticVector{6}) = _tensor(v)
_field_value(v::StaticVector) = v

# The block of the tangent of flux f with respect to gradient g, named fg.
function _tangent_blocks(𝕂, fluxes::NamedTuple{KF}, gradients::NamedTuple{KG}) where {KF, KG}
    names = Tuple(Symbol(f, g) for f in KF for g in KG)
    blocks = Tuple(_block_value(𝕂[rf, rg]) for rf in values(fluxes) for rg in values(gradients))
    return NamedTuple{names}(blocks)
end

_block_value(x::Number) = x
_block_value(A::StaticMatrix{6, 6}) = _tensor(A)
# A tensor flux against a scalar gradient, or the converse.
_block_value(v::StaticVector{6}) = _tensor(v)
_block_value(A::StaticArray) = A

# The checker's probe, field by field: the probe strain for the strain, a small
# value for a scalar field, a small vector for any other one.
Constitutive.probe_gradients(law::AbstractPointLaw, ε) = map(law.gradients) do r
    n = _width(r)
    n == 6 ? ε : n == 1 ? 1.0e-3 : SVector(ntuple(_ -> 1.0e-3, n))
end
