# =============================================================================
#  material_backend.jl — driving a Gauss-point MFH material from Ferrite.
#
#  This file is deliberately thin, and that is the design: the material contract
#  in `src/Constitutive/` knows nothing about any finite-element library, so
#  coupling to Ferrite needs no adapter layer — only the two pieces of
#  bookkeeping every Ferrite element loop repeats.
#
#  Ferrite has no notion of per-quadrature-point material state (there is no
#  `state` machinery in its API), so the convention here is the one its own
#  plasticity tutorials use: a nested vector indexed `[cell][qp]`, kept twice —
#  `states` being written during the Newton iteration, `states_old` frozen at the
#  last converged step. `material_response` never mutates its argument, so a
#  rejected iteration is undone by simply not swapping.
# =============================================================================

"""
    mfh_states(material, ncells, nqp) -> Vector{Vector{S}}
    mfh_states(material, dh::DofHandler, cv::CellValues) -> Vector{Vector{S}}

Allocate one [`initial_state`](@ref MeanFieldHomogenization.Constitutive.initial_state)
per quadrature point, as the nested `[cell][qp]` vector Ferrite element loops
expect.

Allocate **two** of them — one written during the Newton iteration, one frozen
at the last converged step — and swap only once a step has converged.

```julia
states     = mfh_states(mat, dh, cv)
states_old = mfh_states(mat, dh, cv)
```
"""
mfh_states(material, ncells::Integer, nqp::Integer) =
    [[MeanFieldHomogenization.initial_state(material) for _ in 1:nqp] for _ in 1:ncells]

mfh_states(material, dh::Ferrite.DofHandler, cv::Ferrite.CellValues) =
    mfh_states(material, Ferrite.getncells(Ferrite.get_grid(dh)), Ferrite.getnquadpoints(cv))

"""
    mfh_element!(Ke, re, cv, material, ue, states, states_old, Δt; cache = nothing)

Assemble the element stiffness `Ke` and internal-force vector `re` of a
**small-strain, plane-strain** element whose behavior comes from an MFH
material, and write the updated quadrature-point states into `states`.

`ue` is the element displacement vector, `states`/`states_old` the slices for
this cell. `cache` is a
[`MaterialCache`](@ref MeanFieldHomogenization.Constitutive.MaterialCache)
shared across the mesh — pass one, or every quadrature point re-runs the
homogenization scheme.

```julia
for cell in CellIterator(dh)
    reinit!(cv, cell)
    fill!(Ke, 0); fill!(re, 0)
    eldofs = celldofs(cell)
    mfh_element!(Ke, re, cv, mat, u[eldofs],
                 states[cellid(cell)], states_old[cellid(cell)], Δt; cache)
    assemble!(assembler, eldofs, Ke, re)
end
```

The residual is the internal force ``\\int_\\Omega \\boldsymbol\\sigma :
\\nabla^{\\rm s}\\hat{\\mathbf u}\\,{\\rm d}\\Omega``; external loads are the
caller's business.

!!! note "Plane strain"
    The 2-D reduction is
    [`plane_strain_response`](@ref MeanFieldHomogenization.Constitutive.plane_strain_response),
    which is exact — see its docstring for why plane *stress* is a different
    problem.
"""
function mfh_element!(
        Ke::AbstractMatrix, re::AbstractVector, cv::Ferrite.CellValues,
        material, ue::AbstractVector, states::AbstractVector,
        states_old::AbstractVector, Δt::Real; cache = nothing
    )
    n_basefuncs = Ferrite.getnbasefunctions(cv)
    for qp in 1:Ferrite.getnquadpoints(cv)
        dΩ = Ferrite.getdetJdV(cv, qp)
        ε = Ferrite.function_symmetric_gradient(cv, qp, ue)
        r = MeanFieldHomogenization.plane_strain_response(
            material, ε, states_old[qp], Δt; cache = cache
        )
        states[qp] = r.state
        for i in 1:n_basefuncs
            δε = Ferrite.shape_symmetric_gradient(cv, qp, i)
            re[i] += (δε ⊡ r.σ) * dΩ
            for j in 1:n_basefuncs
                Ke[i, j] += (δε ⊡ r.C ⊡ Ferrite.shape_symmetric_gradient(cv, qp, j)) * dΩ
            end
        end
    end
    return Ke, re
end

"""
    annulus_grid(Ri, Ro, nr, nθ; θmax = π/2) -> Ferrite.Grid

A structured quadrilateral mesh of an annular sector, built **without gmsh** by
bending a rectangle in ``(\\rho, \\theta)``.

The facet sets inherited from `generate_grid` keep their meaning after the
mapping: `"left"` is the inner radius ``R_i``, `"right"` the outer radius
``R_o``, `"bottom"` the ``\\theta = 0`` edge and `"top"` the
``\\theta = \\theta_{\\max}`` one.

Avoiding gmsh is what lets a thick-cylinder tutorial run inside a documentation
build: no binary artifact, no meshing time.
"""
function annulus_grid(Ri::Real, Ro::Real, nr::Integer, nθ::Integer; θmax::Real = π / 2)
    grid = Ferrite.generate_grid(
        Ferrite.Quadrilateral, (nr, nθ),
        Ferrite.Vec{2}((0.0, 0.0)), Ferrite.Vec{2}((1.0, 1.0))
    )
    Ferrite.transform_coordinates!(
        grid, x -> begin
            r = Ri + (Ro - Ri) * x[1]
            θ = θmax * x[2]
            Ferrite.Vec{2}((r * cos(θ), r * sin(θ)))
        end
    )
    return grid
end
