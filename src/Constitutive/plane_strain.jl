# =============================================================================
#  plane_strain.jl — driving a 3-D material from a 2-D element loop.
#
#  A homogenized material is intrinsically three-dimensional: its microstructure
#  is. A 2-D finite-element computation, however, hands over a
#  `SymmetricTensor{2,2}` and expects one back. Plane strain is the case where
#  that reduction is *exact* rather than an approximation:
#
#      ε₁₃ = ε₂₃ = ε₃₃ = 0
#
#  so embedding the in-plane strain with zeros loses nothing, and the in-plane
#  block of the 3-D tangent is the 2-D tangent. No static condensation is
#  involved — which is exactly why plane STRESS is not offered here (see below).
# =============================================================================

"""
    plane_strain_response(m, ε₂, state_old, Δt; cache = nothing)

Drive a three-dimensional [`AbstractMFHMaterial`](@ref) from a **plane-strain**
two-dimensional element loop.

`ε₂` is the in-plane strain as a `Tensors.SymmetricTensor{2,2}` — what a 2-D
Ferrite/Gridap element produces. Returns a `NamedTuple`:

| field | |
|:--|:--|
| `σ` | in-plane stress, `SymmetricTensor{2,2}` |
| `C` | in-plane tangent, `SymmetricTensor{4,2}` |
| `σ₃₃` | out-of-plane stress — **not zero** in plane strain |
| `state` | the updated internal state |

everything in the global frame, ready to assemble.

The reduction is exact: plane strain means ``\\varepsilon_{13} =
\\varepsilon_{23} = \\varepsilon_{33} = 0``, so the 3-D strain is the in-plane
one padded with zeros, and the 2-D tangent is the in-plane block
``\\mathbb{C}_{ijkl},\\ i,j,k,l \\in \\{1,2\\}`` of the 3-D one.

```julia
r = plane_strain_response(mat, ε₂, states[c][q], Δt; cache = cache)
Ke .+= ... r.C ...
states[c][q] = r.state
```

!!! warning "Plane strain only — not plane stress"
    Plane *stress* (``\\sigma_{33} = 0``) requires condensing the out-of-plane
    strain out of the law, which for a general anisotropic
    ``\\mathbb{C}^{\\rm hom}`` couples all six components and, for a material
    with internal state, has to be solved at every quadrature point. Reusing
    this function for plane stress would silently impose ``\\varepsilon_{33} =
    0`` instead of ``\\sigma_{33} = 0``, which is a different problem — note the
    `σ₃₃` this returns is generally non-zero.

!!! note "Anisotropy is not checked"
    A microstructure whose axes are not aligned with the plane produces a
    ``\\mathbb{C}^{\\rm hom}`` coupling in-plane and out-of-plane components
    (``\\mathbb{C}_{1123}`` and friends). Plane strain remains exact — those
    couplings only feed `σ₃₃` and the out-of-plane shears, which the 2-D
    momentum balance does not see — but the resulting plane problem is then not
    the one a 2-D intuition expects.
"""
function plane_strain_response(
        m::AbstractMFHMaterial, ε₂::Tensors.SymmetricTensor{2, 2},
        state_old::AbstractMaterialState, Δt::Real; cache = nothing
    )
    T = eltype(ε₂)
    ε₃ = from_tensors(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) -> (i <= 2 && j <= 2) ? ε₂[i, j] : zero(T)
        )
    )
    r = material_response(m, ε₃, state_old, Δt; cache = cache)
    σ₃ = to_tensors(stress(r))
    C₃ = to_tensors(tangent(r))
    return (
        σ = Tensors.SymmetricTensor{2, 2}((i, j) -> σ₃[i, j]),
        C = Tensors.SymmetricTensor{4, 2}((i, j, k, l) -> C₃[i, j, k, l]),
        σ₃₃ = σ₃[3, 3],
        state = state(r),
    )
end
