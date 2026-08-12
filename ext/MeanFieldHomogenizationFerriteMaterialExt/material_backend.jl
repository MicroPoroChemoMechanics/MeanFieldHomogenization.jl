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

The residual is the internal force ``\\int_\\Omega \\boldsymbol{\\sigma} :
\\nabla^{\\rm s}\\delta\\underline{u}\\,{\\rm d}\\Omega``; external loads are the
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
    mfh_poro_element!(Ke, re, cvu, cvp, material, ue, pe, states, states_old, Δt,
                      mobility; u_range, p_range, cache = nothing)

Assemble one **coupled ``(\\underline{u}, p)``** element of a transient
poroelastic problem, backward Euler, whose behavior comes from a two-gradient
MFH material such as [`FracturedPoroelasticRock`](@ref MeanFieldHomogenization.FracturedPoroelasticRock).

The residual of the step is the momentum balance dualized by
``\\delta\\underline{u}`` and the fluid mass balance dualized by ``\\delta p``:

```math
\\int_\\Omega \\boldsymbol{\\Sigma} : \\nabla^{\\rm s}\\delta\\underline{u}
   \\,{\\rm d}\\Omega ,
\\qquad
\\int_\\Omega \\Delta\\varphi\\,\\delta p\\,{\\rm d}\\Omega
 + \\Delta t \\int_\\Omega \\nabla\\delta p \\cdot
   \\left(\\frac{\\boldsymbol{K}}{\\mu}\\cdot\\nabla p\\right){\\rm d}\\Omega ,
```

external loads, imposed flow rates and well terms being the caller's business.
The Jacobian is the four-block operator the material declares — ``\\mathbb{C}^{\\rm hom}``,
``-\\boldsymbol{B}``, ``\\boldsymbol{B}``, ``1/M`` — plus the Darcy term.

- `cvu`, `cvp` — cell values of the displacement and pressure interpolations,
  **sharing one quadrature rule**.
- `u_range`, `p_range` — `dof_range(dh, :u)` and `dof_range(dh, :p)`.
- `mobility` — the mobility ``\\boldsymbol{K}/\\mu`` at each quadrature point, as
  a `SymmetricTensor{2,3}` per point.

!!! note "The mobility is an argument, not an output"
    ``\\boldsymbol{K}`` follows the apertures, so it belongs to the state; but
    differentiating it would couple the flow block to the mechanics through a
    self-consistent solve. Drivers evaluate it once per step from the last
    converged state — [`transport_property`](@ref MeanFieldHomogenization.transport_property) —
    and pass it here. The scheme is then implicit in ``(\\underline{u}, p)`` and
    explicit in ``\\boldsymbol{K}``, which is the usual reservoir practice and
    what the element's signature makes visible.
"""
function mfh_poro_element!(
        Ke::AbstractMatrix, re::AbstractVector,
        cvu::Ferrite.CellValues, cvp::Ferrite.CellValues, material,
        ue::AbstractVector, pe::AbstractVector, states::AbstractVector,
        states_old::AbstractVector, Δt::Real, mobility;
        u_range, p_range, cache = nothing
    )
    nu, np = length(u_range), length(p_range)
    for qp in 1:Ferrite.getnquadpoints(cvu)
        dΩ = Ferrite.getdetJdV(cvu, qp)
        ε = Ferrite.function_symmetric_gradient(cvu, qp, ue)
        p = Ferrite.function_value(cvp, qp, pe)
        ∇p = Ferrite.function_gradient(cvp, qp, pe)

        r = MeanFieldHomogenization.material_response(
            material, (; ε = MeanFieldHomogenization.from_tensors(ε), p = p),
            states_old[qp], Δt; cache = cache
        )
        states[qp] = MeanFieldHomogenization.state(r)

        σ = MeanFieldHomogenization.to_tensors(r.fluxes.σ)
        ℂ = MeanFieldHomogenization.to_tensors(r.tangents.σε)
        B = MeanFieldHomogenization.to_tensors(r.tangents.φε)
        invM = r.tangents.φp
        Δφ = r.fluxes.φ - MeanFieldHomogenization.fluid_content(
            material, states_old[qp]; cache = cache
        )
        𝕄 = mobility[qp]

        for i in 1:nu
            δε = Ferrite.shape_symmetric_gradient(cvu, qp, i)
            re[u_range[i]] += (δε ⊡ σ) * dΩ
            for j in 1:nu
                Ke[u_range[i], u_range[j]] +=
                    (δε ⊡ ℂ ⊡ Ferrite.shape_symmetric_gradient(cvu, qp, j)) * dΩ
            end
            for b in 1:np
                Ke[u_range[i], p_range[b]] -=
                    (δε ⊡ B) * Ferrite.shape_value(cvp, qp, b) * dΩ
            end
        end
        for a in 1:np
            Na, ∇Na = Ferrite.shape_value(cvp, qp, a), Ferrite.shape_gradient(cvp, qp, a)
            re[p_range[a]] += (Δφ * Na + Δt * (∇Na ⋅ (𝕄 ⋅ ∇p))) * dΩ
            for j in 1:nu
                Ke[p_range[a], u_range[j]] +=
                    Na * (B ⊡ Ferrite.shape_symmetric_gradient(cvu, qp, j)) * dΩ
            end
            for b in 1:np
                Ke[p_range[a], p_range[b]] += (
                    Na * invM * Ferrite.shape_value(cvp, qp, b) +
                        Δt * (∇Na ⋅ (𝕄 ⋅ Ferrite.shape_gradient(cvp, qp, b)))
                ) * dΩ
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

"""
    cylinder_sector_grid(Ri, Ro, H, nr, nθ, nz; θmax = π/2, grading = :log)

A structured hexahedral mesh of a cylindrical sector, built **without gmsh** by
bending a box in ``(\\rho, \\theta, z)`` — the three-dimensional twin of
[`annulus_grid`](@ref).

`grading = :log` spaces the radial layers geometrically, which is what a well
problem needs: the pressure drop is logarithmic in ``\\rho``, so uniform layers
would waste every element far from the well and resolve none near it.

The facet sets of `generate_grid` keep their meaning after the mapping:
`"left"` is the inner radius, `"right"` the outer one, `"front"` the
``\\theta = 0`` plane, `"back"` the ``\\theta = \\theta_{\\max}`` one, `"bottom"`
and `"top"` the two horizontal faces.
"""
function cylinder_sector_grid(
        Ri::Real, Ro::Real, H::Real, nr::Integer, nθ::Integer, nz::Integer;
        θmax::Real = π / 2, grading::Symbol = :log
    )
    grid = Ferrite.generate_grid(
        Ferrite.Hexahedron, (nr, nθ, nz),
        Ferrite.Vec{3}((0.0, 0.0, 0.0)), Ferrite.Vec{3}((1.0, 1.0, 1.0))
    )
    Ferrite.transform_coordinates!(
        grid, x -> begin
            r = grading === :log ? Ri * (Ro / Ri)^x[1] : Ri + (Ro - Ri) * x[1]
            θ = θmax * x[2]
            Ferrite.Vec{3}((r * cos(θ), r * sin(θ), H * x[3]))
        end
    )
    return grid
end
