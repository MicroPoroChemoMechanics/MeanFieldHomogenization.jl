# =============================================================================
#  material_points.jl — the Ferrite adapter of the coupling of MaterialPoints.
#
#  Everything that does not depend on Ferrite is in the package: the states of
#  the quadrature points, the modeling hypotheses and the quasi-static driver
#  (src/MaterialPoints/coupling.jl). What is left here is what only Ferrite
#  knows: its cell values, its assembler, its constraint handler, its VTK files.
# =============================================================================

const MP = MeanFieldHomogenization.MaterialPoints

# The position of a quadrature point, where the hypothesis needs it (the hoop
# strain and the volume weight of an axisymmetric body).
_point_x(::MP.ModelingHypothesis, cv, qp, coords) = nothing
_point_x(::MP.Axisymmetric, cv, qp, coords) = Ferrite.spatial_coordinate(cv, qp, coords)
_point_x(::MP.Axisymmetric, cv, qp, ::Nothing) = throw(
    ArgumentError("Axisymmetric elements need the cell coordinates: coords = getcoordinates(cell)")
)

# A mechanical element serves a material of the strain alone.
function _check_mechanical(material)
    names = MeanFieldHomogenization.gradient_names(material)
    names == (:ε,) || throw(
        ArgumentError(
            "material_element! assembles a mechanical problem; $(nameof(typeof(material))) " *
                "has the gradients $names and needs an element of its coupled fields"
        )
    )
    return nothing
end

function MP.material_element!(
        Ke::AbstractMatrix, re::AbstractVector, cv::Ferrite.CellValues, material,
        hypothesis::MP.ModelingHypothesis, ue::AbstractVector, states::MP.QuadratureStates,
        cell::Integer, Δt::Real; algorithm::MP.AbstractLocalAlgorithm = MP.LocalNewton(),
        external::NamedTuple = (;), cache = nothing, coords = nothing
    )
    _check_mechanical(material)
    n = Ferrite.getnbasefunctions(cv)
    # The virtual strains of the shape functions at a point, in Mandel form.
    δ = Vector{MP.Mandel{float(eltype(ue))}}(undef, n)
    for qp in 1:Ferrite.getnquadpoints(cv)
        x = _point_x(hypothesis, cv, qp, coords)
        ε = MP.mandel_strain(
            hypothesis, Ferrite.function_symmetric_gradient(cv, qp, ue), Ferrite.function_value(cv, qp, ue), x
        )
        σ, 𝕂, st = MP._mandel_update(material, ε, states.old[cell][qp], Δt; algorithm, external, cache)
        states.new[cell][qp] = st
        w = Ferrite.getdetJdV(cv, qp) * MP.volume_weight(hypothesis, x)
        for i in 1:n
            δ[i] = MP.mandel_strain(
                hypothesis, Ferrite.shape_symmetric_gradient(cv, qp, i), Ferrite.shape_value(cv, qp, i), x
            )
        end
        for i in 1:n
            re[i] += dot(δ[i], σ) * w
            𝕂ᵀδᵢ = transpose(𝕂) * δ[i]
            for j in 1:n
                Ke[i, j] += dot(𝕂ᵀδᵢ, δ[j]) * w
            end
        end
    end
    return Ke, re
end

function MP.material_system(
        dh::Ferrite.DofHandler, ch::Ferrite.ConstraintHandler, cv::Ferrite.CellValues, material,
        hypothesis::MP.ModelingHypothesis, states::MP.QuadratureStates;
        external_forces! = nothing, external::NamedTuple = (;),
        algorithm::MP.AbstractLocalAlgorithm = MP.LocalNewton(), cache = nothing
    )
    K = Ferrite.allocate_matrix(dh, ch)
    r = zeros(Ferrite.ndofs(dh))
    nd = Ferrite.ndofs_per_cell(dh)
    Ke, re = zeros(nd, nd), zeros(nd)
    ext_of_time = map(MP._of_time, external)
    function system!(u, t, Δt)
        asm = Ferrite.start_assemble(K, r)
        ext = map(f -> f(t), ext_of_time)
        for cell in Ferrite.CellIterator(dh)
            Ferrite.reinit!(cv, cell)
            fill!(Ke, 0)
            fill!(re, 0)
            dofs = Ferrite.celldofs(cell)
            MP.material_element!(
                Ke, re, cv, material, hypothesis, view(u, dofs), states, Ferrite.cellid(cell), Δt;
                algorithm, external = ext, cache, coords = Ferrite.getcoordinates(cell)
            )
            Ferrite.assemble!(asm, dofs, Ke, re)
        end
        # The external forces are subtracted: the residual is internal minus external.
        external_forces! === nothing || external_forces!(r, t)
        Ferrite.apply_zero!(K, r, ch)
        return K, r
    end
    function prescribe!(u, t)
        Ferrite.update!(ch, t)
        Ferrite.apply!(u, ch)
        return u
    end
    return system!, prescribe!
end

function MP.write_material_vtk(path::AbstractString, dh::Ferrite.DofHandler, u::AbstractVector, material::MP.AbstractPointLaw, states::MP.QuadratureStates)
    ncells = length(states.old)
    # Cell averages of the committed states: the fluxes recomputed from the
    # gradients and the internal variables (no integration: the state is final),
    # and every internal variable by name.
    mean_of(f) = [sum(f, states.old[c]) / length(states.old[c]) for c in 1:ncells]
    fluxes = mean_of(st -> MP._fluxes(material, st.g, st.y))
    gradients = mean_of(st -> st.g)
    Ferrite.VTKGridFile(path, dh) do vtk
        Ferrite.write_solution(vtk, dh, u)
        for (fields, values) in ((material.fluxes, fluxes), (material.gradients, gradients))
            _write_fields(vtk, fields, values)
        end
        _write_fields(vtk, material.layout, mean_of(st -> st.y))
    end
    return path
end

# Each entry of a layout: a scalar field, or one cell field per tensor component
# (`σ11`, …, the √2 of the Mandel shears removed).
function _write_fields(vtk, layout, values)
    for (name, range) in pairs(layout)
        n = MP._width(range)
        for (k, component) in enumerate(MP._component_names(name, n))
            i = n == 1 ? range : range[k]
            factor = MP._mandel_factors(n)[k]
            Ferrite.write_cell_data(vtk, [v[i] / factor for v in values], String(component))
        end
    end
    return nothing
end
