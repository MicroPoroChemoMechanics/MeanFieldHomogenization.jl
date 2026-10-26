# The finite-element coupling of MaterialPoints with Ferrite. Every structure is
# loaded so that its fields are homogeneous: the finite-element response must
# then be the response of one material point, computed by the point driver,
# whatever the hypothesis (3-D, plane strain, axisymmetric). The global Newton
# iteration must converge quadratically, which only a consistent tangent gives.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
import Ferrite
import Tensors

const MP_FE = MeanFieldHomogenization.MaterialPoints
const J2_FE = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, hardening = p -> 250.0 + 200.0 * (1 - exp(-300.0 * p)))
const TIMES_FE = range(0, 1; length = 9)

@testset "states of the quadrature points" begin
    s = QuadratureStates(J2_FE, 2, 3)
    @test length(s.old) == 2 && length(s.new[1]) == 3
    st = point_update(J2_FE, SVector(4.0e-3, 0, 0, 0, 0, 0), s.old[1][1], 0.0)[3]
    s.new[1][1] = st
    rollback!(s)
    @test s.new[1][1] == s.old[1][1] == initial_state(J2_FE)
    s.new[1][1] = st
    commit!(s)
    @test s.old[1][1] == st
end

@testset "modeling hypotheses" begin
    ε3 = Tensors.SymmetricTensor{2, 3}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    @test mandel_strain(ThreeD(), ε3) == Tensors.tomandel(SArray, ε3)
    ε2 = Tensors.SymmetricTensor{2, 2}((1.0, 2.0, 3.0))
    @test mandel_strain(PlaneStrain(), ε2) ≈ SVector(1.0, 3.0, 0, 0, 0, sqrt(2) * 2.0)
    @test mandel_strain(Axisymmetric(), ε2, (0.5, 0.0), (2.0, 1.0)) ≈ SVector(1.0, 3.0, 0.25, 0, 0, sqrt(2) * 2.0)
    @test volume_weight(Axisymmetric(), (2.0, 1.0)) == 2.0
    @test volume_weight(PlaneStrain(), nothing) * 3.5 === 3.5
end

# A box meshed by `cell` with `n` elements, its dofs, its constraints and its
# cell values; `dirichlet` lists (facet set, function of (x, t), components).
function _fe_box(cell, refshape, dim, n, lo, hi, dirichlet)
    grid = Ferrite.generate_grid(cell, n, Ferrite.Vec(lo), Ferrite.Vec(hi))
    ip = Ferrite.Lagrange{refshape, 1}()^dim
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip)
    Ferrite.close!(dh)
    ch = Ferrite.ConstraintHandler(dh)
    for (set, f, comps) in dirichlet
        Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(grid, set), f, comps))
    end
    Ferrite.close!(ch)
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{refshape}(2), ip)
    return (; grid, dh, ch, cv, ip)
end

# Run a structure along the times; returns the displacement, the states and the
# Newton history.
function _fe_run(fe, law, hypothesis; times = TIMES_FE, kw...)
    states = QuadratureStates(law, Ferrite.getncells(fe.grid), Ferrite.getnquadpoints(fe.cv))
    sys_kw = haskey(kw, :system_kw) ? kw[:system_kw] : (;)
    run_kw = Base.structdiff(NamedTuple(kw), (; system_kw = nothing))
    system!, prescribe! = material_system(fe.dh, fe.ch, fe.cv, law, hypothesis, states; sys_kw...)
    u = zeros(Ferrite.ndofs(fe.dh))
    history = quasistatic!(system!, prescribe!, u, states, times; run_kw...)
    return u, states, history
end

# Cell-and-point average of a component of the fluxes or of the gradients of a
# law, from the committed states.
_mean(f, states) = sum(sum(f, c) / length(c) for c in states.old) / length(states.old)
_stress(law, states, i) = _mean(st -> MP_FE._fluxes(law, st.g, st.y)[i], states)
_strain(states, i) = _mean(st -> st.g[i], states)

# The highest order of convergence read on the Newton histories.
function _convergence_order(history)
    orders = Float64[]
    for (_, norms) in history
        e = norms ./ norms[1]
        for k in 2:(length(e) - 1)
            1.0e-12 < e[k + 1] < e[k] < 0.5 && push!(orders, log(e[k + 1]) / log(e[k]))
        end
    end
    return maximum(orders; init = 0.0)
end

const CUBE = (Ferrite.Hexahedron, Ferrite.RefHexahedron, 3, (2, 1, 1), (0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
const SYMMETRY_3D = [("left", (x, t) -> 0.0, [1]), ("front", (x, t) -> 0.0, [2]), ("bottom", (x, t) -> 0.0, [3])]

@testset "3-D: a cube in uniaxial tension, displacement controlled" begin
    fe = _fe_box(CUBE..., vcat(SYMMETRY_3D, [("right", (x, t) -> 2.0 * 4.0e-3 * t, [1])]))
    u, states, history = _fe_run(fe, J2_FE, ThreeD())
    ref = point_response(J2_FE, LoadingPath(TIMES_FE; ε11 = t -> 4.0e-3 * t))
    @test _stress(J2_FE, states, 1) ≈ ref[:σ11][end] rtol = 1.0e-9
    @test _strain(states, 2) ≈ ref[:ε22][end] rtol = 1.0e-9
    @test abs(_stress(J2_FE, states, 2)) < 1.0e-6
    @test all(((_, n),) -> n[end] <= 1.0e-8 * n[1], history)
    @test _convergence_order(history) > 1.8                  # quadratic: a consistent tangent
end

@testset "3-D: the same cube, force controlled" begin
    fe = _fe_box(CUBE..., SYMMETRY_3D)
    # The nodal forces of a unit traction along e₁ on the right face.
    fv = Ferrite.FacetValues(Ferrite.FacetQuadratureRule{Ferrite.RefHexahedron}(2), fe.ip)
    f_unit = zeros(Ferrite.ndofs(fe.dh))
    fₑ = zeros(Ferrite.ndofs_per_cell(fe.dh))
    for facet in Ferrite.FacetIterator(fe.dh, Ferrite.getfacetset(fe.grid, "right"))
        Ferrite.reinit!(fv, facet)
        fill!(fₑ, 0)
        for q in 1:Ferrite.getnquadpoints(fv), i in eachindex(fₑ)
            fₑ[i] += Ferrite.shape_value(fv, q, i)[1] * Ferrite.getdetJdV(fv, q)
        end
        Ferrite.assemble!(f_unit, Ferrite.celldofs(facet), fₑ)
    end
    traction(t) = 330.0 * t
    u, states, _ = _fe_run(fe, J2_FE, ThreeD(); system_kw = (; external_forces! = (r, t) -> (r .-= traction(t) .* f_unit)))
    ref = point_response(J2_FE, LoadingPath(TIMES_FE; σ11 = traction))
    @test _strain(states, 1) ≈ ref[:ε11][end] rtol = 1.0e-8
    @test _stress(J2_FE, states, 1) ≈ 330.0 rtol = 1.0e-8
end

@testset "plane strain: a plate in tension" begin
    fe = _fe_box(
        Ferrite.Quadrilateral, Ferrite.RefQuadrilateral, 2, (2, 2), (0.0, 0.0), (1.0, 1.0),
        [("left", (x, t) -> 0.0, [1]), ("bottom", (x, t) -> 0.0, [2]), ("right", (x, t) -> 4.0e-3 * t, [1])]
    )
    _, states, history = _fe_run(fe, J2_FE, PlaneStrain())
    ref = point_response(J2_FE, LoadingPath(TIMES_FE; ε11 = t -> 4.0e-3 * t, ε33 = 0, ε13 = 0, ε23 = 0))
    @test _stress(J2_FE, states, 1) ≈ ref[:σ11][end] rtol = 1.0e-9
    @test _stress(J2_FE, states, 3) ≈ ref[:σ33][end] rtol = 1.0e-9          # σ₃₃ ≠ 0 in plane strain
    @test _convergence_order(history) > 1.8
end

@testset "axisymmetric: a hollow cylinder stretched along its axis" begin
    fe = _fe_box(
        Ferrite.Quadrilateral, Ferrite.RefQuadrilateral, 2, (2, 2), (0.5, 0.0), (1.5, 1.0),
        [("bottom", (x, t) -> 0.0, [2]), ("top", (x, t) -> 4.0e-3 * t, [2])]
    )
    _, states, history = _fe_run(fe, J2_FE, Axisymmetric())
    # Axes of the material (r, z, θ): the axial stress is the second component,
    # and the radial and hoop strains are equal.
    ref = point_response(J2_FE, LoadingPath(TIMES_FE; ε22 = t -> 4.0e-3 * t))
    @test _stress(J2_FE, states, 2) ≈ ref[:σ22][end] rtol = 1.0e-8
    @test _strain(states, 3) ≈ ref[:ε11][end] rtol = 1.0e-8
    @test _strain(states, 1) ≈ _strain(states, 3) rtol = 1.0e-8
    @test _convergence_order(history) > 1.8
end

@testset "steps cut, materials of the contract, external variables" begin
    # A law whose local equations are singular on steps longer than 0.3: one
    # step from 0 to 1 is cut twice, and ȧ = 1 integrates exactly.
    J = identity_mandel() * transpose(identity_mandel()) / 3
    ℂ = 3 * 160.0e3 * J + 2 * 80.0e3 * (one(SMatrix{6, 6, Float64}) - J)
    clock = ResidualLaw(;
        stress = (ε, y, p) -> p.ℂ * ε, layout = (a = 1,), p = (; ℂ),
        residual = (y, g, prev, Δt, p) -> (a = Δt > 0.3 ? zero(y.a) : y.a - prev.y.a - Δt,),
    )
    fe = _fe_box(CUBE..., vcat(SYMMETRY_3D, [("right", (x, t) -> 1.0e-4 * t, [1])]))
    _, states, history = _fe_run(fe, clock, ThreeD(); times = [0.0, 1.0])
    @test all(c -> all(st -> st.y[1] ≈ 1.0, c), states.old)
    @test length(history) == 4                                   # four quarter steps
    @test_throws LocalIntegrationError _fe_run(fe, clock, ThreeD(); times = [0.0, 1.0], max_cuts = 1)
    # The global iteration limited to one correction on a curved plastic branch.
    fe_t = _fe_box(CUBE..., vcat(SYMMETRY_3D, [("right", (x, t) -> 2.0 * 4.0e-3 * t, [1])]))
    err = (@test_throws GlobalNewtonError _fe_run(fe_t, J2_FE, ThreeD(); times = [0.0, 1.0], maxiters = 1, max_cuts = 0)).value
    @test occursin("equilibrium", sprint(showerror, err))
    # A homogenized material of the contract, in the same element.
    he = HomogenizedElastic(TensISO{3}(3 * 30.0e3, 2 * 18.0e3))
    _, states_he, _ = _fe_run(fe_t, he, ThreeD())
    @test states_he.old[1][1] isa NoState
    # A temperature read by a thermoelastic law: free expansion α ΔT.
    thermal(g, p) = g.ε - p.α * (p.T - 293.0) * identity_mandel()
    thermo = PotentialLaw(; energy = (g, a, p) -> sum(thermal(g, p) .* (p.ℂ * thermal(g, p))) / 2, p = (; ℂ, α = 1.0e-5))
    fe_free = _fe_box(CUBE..., SYMMETRY_3D)
    _, states_th, _ = _fe_run(fe_free, thermo, ThreeD(); system_kw = (; external = (T = t -> 293.0 + 100.0 * t,)))
    @test _strain(states_th, 1) ≈ 1.0e-3 rtol = 1.0e-10
    # Refusals: a coupled material in a mechanical element, an axisymmetric
    # element without coordinates.
    biot = BiotPoroelasticity(; E = 10.0e3, ν = 0.25, b = 0.8, M = 5.0e3)
    @test_throws ArgumentError _fe_run(fe_t, biot, ThreeD())
    n = Ferrite.ndofs_per_cell(fe_t.dh)
    @test_throws ArgumentError material_element!(
        zeros(n, n), zeros(n), fe_t.cv, J2_FE, Axisymmetric(), zeros(n), QuadratureStates(J2_FE, 1, 8), 1, 0.0
    )
end

@testset "VTK output" begin
    fe = _fe_box(CUBE..., vcat(SYMMETRY_3D, [("right", (x, t) -> 2.0 * 4.0e-3 * t, [1])]))
    u, states, _ = _fe_run(fe, J2_FE, ThreeD())
    mktempdir() do dir
        path = write_material_vtk(joinpath(dir, "cube"), fe.dh, u, J2_FE, states)
        file = path * ".vtu"
        @test isfile(file)
        content = read(file, String)
        @test all(name -> occursin(name, content), ("σ11", "ε22", "εᵖ11", "\"p\""))
    end
end
