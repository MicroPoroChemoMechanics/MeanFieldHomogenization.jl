# The truncated cell around a supershape: geometry and meshing only, no solve.
#
# Needs gmsh, so the caller guards it. Kept at level 2, which is a few thousand
# tetrahedra and a second of wall time -- the point here is that the topology is
# right and the curved boundary is where it should be, not that the mesh is fine
# enough for a converged answer.

using Test
using MeanFieldHomogenization
using Gmsh: gmsh

const FE = MeanFieldHomogenization.FiniteElements

@testset "supershape cell: geometry and meshing" begin

    @testset "the estimator refuses before gmsh has to" begin
        s = Supersphere(1.0, 0.3)
        e = fe_cell_size_estimate(s, FECellMeshOptions(; level = 2, radius_ratio = 3.0))
        @test e.ntets > 0
        @test e.dofs_vector == 3 * e.nnodes_p2
        @test e.R ≈ 3.0                       # p < 1, so the bounding radius is a
        # Refinement costs, and the estimate must say so monotonically.
        e3 = fe_cell_size_estimate(s, FECellMeshOptions(; level = 3, radius_ratio = 3.0))
        @test e3.ntets > e.ntets
        # An elongated superspheroid has its outer boundary set from the
        # bounding radius, not from `a`.
        q = Superspheroid(1.0, 3.0, 0.8)
        @test fe_cell_size_estimate(q, FECellMeshOptions(; radius_ratio = 3.0)).R >
            3.0 * 1.0
    end

    @testset "the budget guard refuses rather than crashes the session" begin
        @test fe_available_gb() > 0
        @test_throws ErrorException FE._fe_check_budget(
            10^9; max_dofs = 1000, min_free_gb = 0.0
        )
        @test_throws ErrorException FE._fe_check_budget(
            10; max_dofs = 10^9, min_free_gb = 1.0e9
        )
        @test FE._fe_check_budget(10; max_dofs = 10^9, min_free_gb = 0.0) === nothing
    end

    for (name, shape, limits) in (
            ("sphere", Supersphere(1.0, 1.0), false),
            ("mildly concave supersphere", Supersphere(1.0, 0.45), false),
            ("strongly concave supersphere", Supersphere(1.0, 0.35), true),
            ("superspheroid", Superspheroid(1.0, 2.0, 0.8), false),
        )
        @testset "meshing a $name" begin
            opts = FECellMeshOptions(;
                level = 2, outer_level = 3, radius_ratio = 3.0,
                relax = 20
            )
            gmsh.initialize()
            local built, ntet, nnode, Vflat, Vcurved, Vmesh
            try
                gmsh.option.setNumber("General.Terminal", 0)
                built = FE._build_gmsh_cell_model(gmsh, shape, opts)
                _, etags, _ = gmsh.model.mesh.getElements(3, FE.CELL_TAG_MATRIX)
                ntet = sum(length, etags)
                ntags, _, _ = gmsh.model.mesh.getNodes()
                nnode = length(ntags)
                Vflat = mesh_volume(built.inner)
                Vcurved = fe_cell_curved_volume(gmsh, FE.CELL_TAG_INCLUSION)
                Vmesh = fe_cell_meshed_volume(gmsh)
            finally
                gmsh.finalize()
            end

            @test ntet > 100
            @test nnode > ntet ÷ 4
            @test built.R ≈ 3.0 * bounding_radius(shape)

            # The meshed region is the shell, so its volume is the ball minus
            # the inclusion. That is the check that the two surface loops were
            # given in the right order -- reversing them yields the ball.
            Vball = 4π * built.R^3 / 3
            @test Vmesh ≈ Vball - shape_volume(shape) rtol = 0.05
            @test Vmesh < Vball

            # The curved boundary is closer to the exact shape than the flat one
            # it was built from. This is the *whole* purpose of the snapping,
            # and it cannot be seen by `mesh_volume`, which is why there are two
            # volume functions.
            Vex = shape_volume(shape)
            @test abs(Vcurved - Vex) < abs(Vflat - Vex)

            @test built.snap.maxd > 0            # it really moved something
            # Whether the snapping had to back off is a property of the shape,
            # not of the level: see the `limited` discussion in
            # `_snap_cell_surface_to_shape!`. Only a strongly concave shape does.
            if limits
                @test built.snap.limited > 0
                @test built.snap.min_blend < 1
                @test built.snap.limited < 0.1 * built.snap.total
            else
                @test built.snap.limited == 0
                @test built.snap.min_blend == 1.0
            end
        end
    end

    @testset "the octahedron needs no snapping at all" begin
        # p = 1/2 has edges and vertices everywhere and yet nothing to snap: its
        # faces are flat, so `setOrder(2)`'s mid-edge nodes already sit exactly
        # on the surface. This is what separates sharpness, which is harmless
        # here, from unbounded curvature, which is not.
        gmsh.initialize()
        local snap
        try
            gmsh.option.setNumber("General.Terminal", 0)
            b = FE._build_gmsh_cell_model(
                gmsh, Supersphere(1.0, 0.5),
                FECellMeshOptions(;
                    level = 3, outer_level = 3, radius_ratio = 3.0,
                    relax = 20
                ),
            )
            snap = b.snap
        finally
            gmsh.finalize()
        end
        @test snap.maxd < 1.0e-14
        @test snap.limited == 0
        @test snap.min_blend == 1.0
    end

    @testset "order 1 leaves a straight boundary" begin
        shape = Supersphere(1.0, 1.0)
        opts = FECellMeshOptions(;
            level = 2, outer_level = 2, radius_ratio = 3.0,
            relax = 0, order = 1
        )
        gmsh.initialize()
        local built, thrown
        try
            gmsh.option.setNumber("General.Terminal", 0)
            built = FE._build_gmsh_cell_model(gmsh, shape, opts)
            # No second-order triangles, so the curved measure must refuse
            # rather than return a wrong number.
            thrown = try
                fe_cell_curved_volume(gmsh, FE.CELL_TAG_INCLUSION)
                false
            catch err
                err isa ArgumentError
            end
        finally
            gmsh.finalize()
        end
        @test built.snap === nothing
        @test thrown
        @test_throws ArgumentError FE._build_gmsh_cell_model(
            gmsh, shape, FECellMeshOptions(; order = 0)
        )
    end
end
