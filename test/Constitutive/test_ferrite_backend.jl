# MeanFieldHomogenizationFerriteMaterialExt: an MFH material as the Gauss-point
# law of a Ferrite computation, with its state at each quadrature point, its two
# element routines and its two meshers. Each is checked against a reference that
# does not go through it: Lamé's solution for the plane-strain element, and for
# the coupled element the integrals it must assemble, which a field linear in
# the coordinates reproduces exactly on a trilinear element.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import Ferrite

const FMX = Base.get_extension(MeanFieldHomogenization, :MeanFieldHomogenizationFerriteMaterialExt)

@testset "plane-strain element: a thick cylinder against Lamé" begin
    # Stiff spherical inclusions: an isotropic composite, so that Lamé's solution
    # of a cylinder under internal pressure holds with its homogenized moduli.
    Ri, Ro, P = 0.1, 1.0, 20.0
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(90.0, 36.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(360.0, 160.0)); fraction = 0.25)
    mat = HomogenizedElastic(rve, MoriTanaka())
    k, μ = k_mu(stiffness(mat))
    E, ν = 9k * μ / (3k + μ), (3k - 2μ) / (2 * (3k + μ))
    # Plane-strain Lamé: u_r = (1 + ν) P Ri² / (E (Ro² - Ri²)) ((1 - 2ν) r + Ro²/r).
    ur_lame(ρ) = (1 + ν) * P * Ri^2 / (E * (Ro^2 - Ri^2)) * ((1 - 2ν) * ρ + Ro^2 / ρ)
    cache = MaterialCache()
    ip = Ferrite.Lagrange{Ferrite.RefQuadrilateral, 1}()^2
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{Ferrite.RefQuadrilateral}(2), ip)
    fv = Ferrite.FacetValues(Ferrite.FacetQuadratureRule{Ferrite.RefQuadrilateral}(2), ip)
    nbf = Ferrite.getnbasefunctions(cv)

    # The largest nodal error on u_r, relative to u_r(Ri), on an nr × nθ mesh of
    # a quarter of the cylinder.
    function lame_error(nr, nθ)
        grid = FMX.annulus_grid(Ri, Ro, nr, nθ)
        dh = Ferrite.DofHandler(grid)
        Ferrite.add!(dh, :u, ip)
        Ferrite.close!(dh)
        # No normal displacement on the two straight edges.
        ch = Ferrite.ConstraintHandler(dh)
        Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
        Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(grid, "top"), (x, t) -> 0.0, 1))
        Ferrite.close!(ch)
        states, states_old = FMX.mfh_states(mat, dh, cv), FMX.mfh_states(mat, dh, cv)
        @test length(states) == Ferrite.getncells(grid) &&
            all(s -> length(s) == Ferrite.getnquadpoints(cv), states)
        # Linear elastic: one assembly at u = 0 and one solve.
        n = Ferrite.ndofs(dh)
        K, fint = Ferrite.allocate_matrix(dh, ch), zeros(n)
        Ke, re = zeros(nbf, nbf), zeros(nbf)
        u = zeros(n)
        asm = Ferrite.start_assemble(K, fint)
        for cell in Ferrite.CellIterator(dh)
            Ferrite.reinit!(cv, cell)
            fill!(Ke, 0.0)
            fill!(re, 0.0)
            ed = Ferrite.celldofs(cell)
            c = Ferrite.cellid(cell)
            FMX.mfh_element!(Ke, re, cv, mat, u[ed], states[c], states_old[c], 0.0; cache)
            Ferrite.assemble!(asm, ed, Ke, re)
        end
        # The pressure on r = Ri, as consistent nodal forces.
        fext = zeros(n)
        for fc in Ferrite.FacetIterator(dh, Ferrite.getfacetset(grid, "left"))
            Ferrite.reinit!(fv, fc)
            fe = zeros(nbf)
            for q in 1:Ferrite.getnquadpoints(fv)
                nrm, dΓ = Ferrite.getnormal(fv, q), Ferrite.getdetJdV(fv, q)
                for i in 1:nbf
                    fe[i] += (Ferrite.shape_value(fv, q, i) ⋅ (-P * nrm)) * dΓ
                end
            end
            fext[Ferrite.celldofs(fc)] .+= fe
        end
        r = fint .- fext
        Ferrite.apply_zero!(K, r, ch)
        u .-= K \ r
        Ferrite.apply!(u, ch)
        xs = [grid.nodes[i].x for i in 1:Ferrite.getnnodes(grid)]
        un = Ferrite.evaluate_at_grid_nodes(dh, u, :u)
        ur = [(un[i] ⋅ xs[i]) / norm(xs[i]) for i in eachindex(xs)]
        return maximum(abs.(ur .- ur_lame.(norm.(xs)))) / ur_lame(Ri)
    end

    # Bilinear elements converge at second order in the displacement: halving the
    # mesh divides the error by four asymptotically (3.4 measured from 12 × 6,
    # where the chords of the inner arc still weigh), by two at first order.
    e_coarse, e_fine = lame_error(12, 6), lame_error(24, 12)
    @test e_fine < 2.0e-2
    @test e_coarse / e_fine > 3
    @test FMX.mfh_states(mat, 3, 5) isa Vector && size.(FMX.mfh_states(mat, 3, 5)) == fill((5,), 3)
end

@testset "coupled (u, p) element: the integrals it assembles" begin
    rve = RVE()
    props = Dict(:C => TensISO{3}(90.0, 36.0), :K => TensISO{3}(1.0e-18))
    add_phase!(rve, :M, Ellipsoid(1.0), props; fraction = :rest)
    add_phase!(rve, :F, ConductiveCrack(1.0; conductivity = 4.0e-18), props; density = 0.1)
    mat = FracturedPoroelasticRock(rve, MoriTanaka(); ω₀ = (1.0e-3,), k_matrix = 1.0e-18)

    # The material at the uniform state of the test, ε = 0 and p = p₀: its stress,
    # its four tangent blocks and the change of its fluid content.
    p₀ = 2.0
    s₀ = initial_state(mat)
    ε₀ = from_tensors(zero(TensND.Tensors.SymmetricTensor{2, 3, Float64}))
    rm = material_response(mat, (; ε = ε₀, p = p₀), s₀, 0.0)
    σ = MeanFieldHomogenization.to_tensors(rm.fluxes.σ)
    ℂ = MeanFieldHomogenization.to_tensors(rm.tangents.σε)
    B = MeanFieldHomogenization.to_tensors(rm.tangents.φε)
    invM = rm.tangents.φp
    Δφ = rm.fluxes.φ - MeanFieldHomogenization.fluid_content(mat, s₀)
    𝟏 = one(TensND.Tensors.SymmetricTensor{2, 3, Float64})
    kmob = 3.0
    mob = TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> i == j ? kmob : 0.0)

    for grading in (:log, :linear)
        grid = FMX.cylinder_sector_grid(0.1, 1.0, 0.5, 1, 1, 1; grading)
        ipu = Ferrite.Lagrange{Ferrite.RefHexahedron, 1}()^3
        ipp = Ferrite.Lagrange{Ferrite.RefHexahedron, 1}()
        qr = Ferrite.QuadratureRule{Ferrite.RefHexahedron}(2)
        cvu, cvp = Ferrite.CellValues(qr, ipu), Ferrite.CellValues(qr, ipp)
        dh = Ferrite.DofHandler(grid)
        Ferrite.add!(dh, :u, ipu)
        Ferrite.add!(dh, :p, ipp)
        Ferrite.close!(dh)
        ur, pr = Ferrite.dof_range(dh, :u), Ferrite.dof_range(dh, :p)
        cell = first(Ferrite.CellIterator(dh))
        Ferrite.reinit!(cvu, cell)
        Ferrite.reinit!(cvp, cell)
        nq = Ferrite.getnquadpoints(cvu)
        V = sum(q -> Ferrite.getdetJdV(cvu, q), 1:nq)
        xn = Ferrite.getcoordinates(cell)

        # δu = x: its strain is the identity exactly, the element being isoparametric.
        ux = [xn[a][c] for a in eachindex(xn) for c in 1:3]
        @test all(q -> Ferrite.function_symmetric_gradient(cvu, q, ux) ≈ 𝟏, 1:nq)
        tx = [c == 1 ? 1.0 : 0.0 for a in eachindex(xn) for c in 1:3]   # a translation
        o = ones(length(pr))

        function element(Δt, pe)
            nd = Ferrite.ndofs_per_cell(dh)
            Ke, re = zeros(nd, nd), zeros(nd)
            states = FMX.mfh_states(mat, 1, nq)[1]
            states_old = FMX.mfh_states(mat, 1, nq)[1]
            FMX.mfh_poro_element!(
                Ke, re, cvu, cvp, mat, zeros(length(ur)), pe, states, states_old, Δt,
                fill(mob, nq); u_range = ur, p_range = pr,
            )
            return Ke, re, states
        end
        Ke, re, states = element(0.5, fill(p₀, length(pr)))
        # Each point's new state is the material's at (ε = 0, p₀), up to the
        # rounding of the pressure interpolated there.
        @test all(s -> MeanFieldHomogenization.fluid_content(mat, s) ≈ rm.fluxes.φ, states)
        # Momentum: ∫σ:∇ˢδu with δu = x is V tr σ, and a translation does no work.
        @test ux ⋅ re[ur] ≈ V * tr(σ) rtol = 1.0e-10
        @test abs(tx ⋅ re[ur]) <= 1.0e-10 * abs(V * tr(σ))
        # Fluid mass at a uniform pressure: V Δφ, the Darcy term vanishing.
        @test sum(re[pr]) ≈ V * Δφ rtol = 1.0e-10
        # The four blocks, read with δu = x and a uniform δp.
        @test ux ⋅ (Ke[ur, ur] * ux) ≈ V * (𝟏 ⊡ ℂ ⊡ 𝟏) rtol = 1.0e-10
        @test ux ⋅ (Ke[ur, pr] * o) ≈ -V * (𝟏 ⊡ B) rtol = 1.0e-10
        @test o ⋅ (Ke[pr, ur] * ux) ≈ V * (B ⊡ 𝟏) rtol = 1.0e-10
        @test o ⋅ (Ke[pr, pr] * o) ≈ V * invM rtol = 1.0e-10
        # Darcy: what Δt multiplies, read on a pressure linear in x, is V g·𝕄·g.
        g = [1.0, -2.0, 0.5]
        px = [g ⋅ xn[a] for a in eachindex(xn)]
        K1, _, _ = element(1.0, px)
        K2, _, _ = element(2.0, px)
        @test px ⋅ ((K2[pr, pr] - K1[pr, pr]) * px) ≈ V * kmob * (g ⋅ g) rtol = 1.0e-10
    end
end
