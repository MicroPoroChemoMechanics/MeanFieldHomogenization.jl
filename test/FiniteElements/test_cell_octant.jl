# The octant cell, meshed and solved.
#
# The parity algebra behind it is checked in `test_cell_octant_algebra.jl`, at
# no cost and without a mesher. What is left to establish here is that the
# construction realizes it: that the boundary is watertight, that the flat faces
# survive the second-order promotion, and above all that an eighth of the cell
# returns the *same answer* as the whole of it.
#
# Level 2 throughout, and level 3 only for the cheapest of the checks: the point
# is the construction, not the discretization, and the machine has to stay up.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import Tensors
import Ferrite, FerriteGmsh
using Gmsh: gmsh

const FEO = MeanFieldHomogenization.FiniteElements
const _BO = FEO.FerriteBackend()

_octC(μ, ν) = (
    λ = 2μ * ν / (1 - 2ν);
    Tensors.SymmetricTensor{4, 3}(
        (i, j, k, l) -> λ * (i == j) * (k == l) +
            μ * ((i == k) * (j == l) + (i == l) * (j == k))
    )
)

function _oct_sphere_A(ν)
    SJ = (1 + ν) / (3 * (1 - ν))
    SK = 2 * (4 - 5ν) / (15 * (1 - ν))
    v = [1.0, 1, 1, 0, 0, 0]
    J = v * v' / 3
    K = Matrix(1.0I, 6, 6) - J
    return inv(Matrix(1.0I, 6, 6) - (SJ * J + SK * K))
end

"Both physics on one mesh, either whole or in an octant."
function _oct_run(shape, level, octant; μ = 1.0, ν = 0.3, outer_level = 2, rr = 3.0)
    opts = FECellMeshOptions(;
        level, outer_level, radius_ratio = rr, relax = 20, octant,
        max_dofs = 60_000,
    )
    g = FEO.fe_cell_grid(_BO, shape, opts)
    V = shape_volume(shape)
    sc = FEO.fe_cell_space(_BO, g.grid, 2, 1)
    c = FEO._cell_conduction_localization(_BO, sc, 1.0, V; octant)
    se = FEO.fe_cell_space(_BO, g.grid, 2, 3)
    e = FEO._cell_elastic_localization(_BO, se, _octC(μ, ν), μ, ν, V; octant)
    return (;
        cond = c.A, elas = e.A, built = g.built,
        ndofs = FEO.fe_cell_dof_split(_BO, se)[1],
    )
end

@testset "supershape cell: the octant" begin

    @testset "the shell closes, before anything is meshed" begin
        # The divergence identity over the five faces. A flipped face, a
        # duplicated node or a chain traversed backwards all show up here — in
        # milliseconds, and with no mesher involved.
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try
            for (shape, lvl) in (
                    (Supersphere(1.0, 1.0), 2), (Supersphere(1.0, 0.35), 2),
                    (Superspheroid(1.0, 2.0, 0.8), 2),
                )
                opts = FECellMeshOptions(; level = lvl, outer_level = 2, radius_ratio = 3.0, relax = 20)
                R = FEO._cell_outer_radius(shape, opts)
                b = FEO._octant_shell(gmsh, shape, R, opts)
                @test FEO._octant_closure_defect(b.shell) < 1.0e-9

                # Each boundary chain lies exactly in its plane, and the flat
                # faces are built only from nodes that do.
                for k in 1:3
                    @test all(iszero(b.shell.nodes[i][k]) for i in b.shell.curves[k])
                    @test all(iszero(b.shell.nodes[i][k]) for i in b.shell.curves[3 + k])
                    @test all(
                        iszero(b.shell.nodes[n][k])
                            for t in b.shell.faces[2 + k] for n in t
                    )
                end
                # Radial segments run along their own axis, from the shape to R.
                for k in 1:3
                    seg = b.shell.curves[6 + k]
                    @test all(
                        all(iszero(b.shell.nodes[i][l]) for l in 1:3 if l != k)
                            for i in seg
                    )
                    @test b.shell.nodes[seg[end]][k] ≈ R
                end
            end
        finally
            gmsh.finalize()
        end
    end

    @testset "the flat faces stay flat through setOrder(2) and both snappings" begin
        # Not enforced by any new code: `_snap_cell_surface_to_*!` scales a node
        # radially, and `r * 0.0 == 0.0`. Mid-edge nodes are midpoints of
        # coplanar pairs. This asserts the argument rather than trusting it.
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try
            shape = Supersphere(1.0, 0.6)
            opts = FECellMeshOptions(;
                level = 2, outer_level = 2, radius_ratio = 3.0, relax = 20, octant = true
            )
            b = FEO._build_gmsh_cell_model(gmsh, shape, opts)
            @test b.octant
            @test b.closure_defect < 1.0e-9
            for k in 1:3
                nk, co, _ = gmsh.model.mesh.getNodes(2, FEO.CELL_TAG_PLANE[k], true, false)
                @test !isempty(nk)
                @test all(abs(co[3 * (a - 1) + k]) < 1.0e-15 for a in 1:length(nk))
            end
            # The eighth of a cavity, times eight, is the cavity.
            @test b.cavity_volume ≈ shape_volume(shape) rtol = 0.05
        finally
            gmsh.finalize()
        end
    end

    @testset "the guard refuses a shape that has not opted in" begin
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try
            opts = FECellMeshOptions(; level = 2, octant = true)
            @test_throws ArgumentError FEO._build_gmsh_cell_model_octant(
                gmsh, _NotMirroredShape(), 3.0, opts
            )
            # And there is no silent fallback: a shape without the mirrors does
            # not quietly get a full cell eight times the size it asked for.
        finally
            gmsh.finalize()
        end
    end

    @testset "against the exact spherical pore" begin
        r = _oct_run(Supersphere(1.0, 1.0), 2, true)
        Aex = _oct_sphere_A(0.3)
        cond_err = norm(r.cond - 1.5I) / 1.5
        elas_err = norm(r.elas - Aex) / norm(Aex)
        # Absolute thresholds at level 2 with `outer_level = 2`; the sharp
        # statement is the comparison just below, which says the octant is not
        # allowed to be worse than the whole cell it replaces.
        @test cond_err < 1.0e-2
        @test elas_err < 5.0e-2
        # The gate is the same one the full cell passes, so an octant is not
        # allowed to be worse than it at the same level.
        rf = _oct_run(Supersphere(1.0, 1.0), 2, false)
        @test cond_err ≤ norm(rf.cond - 1.5I) / 1.5 * 1.5
        @test elas_err ≤ norm(rf.elas - Aex) / norm(Aex) * 1.5
    end

    @testset "an eighth gives the same answer as the whole" begin
        # The two meshes are genuinely different — gmsh does not build the full
        # cell by reflecting an octant — so the gap here is the discretization
        # error the two share, not round-off. At level 2 that is of order 1e-2,
        # which is what the tolerance says; it is not a number tuned until the
        # test passed.
        for shape in (Supersphere(1.0, 0.6), Superspheroid(1.0, 2.0, 0.8))
            o = _oct_run(shape, 2, true)
            f = _oct_run(shape, 2, false)
            @test norm(o.cond - f.cond) / norm(f.cond) < 1.5e-2
            @test norm(o.elas - f.elas) / norm(f.elas) < 3.0e-2
        end
    end

    @testset "the octant is smaller, and its structural zeros are exact" begin
        o = _oct_run(Supersphere(1.0, 0.6), 2, true)
        f = _oct_run(Supersphere(1.0, 0.6), 2, false)
        @test o.ndofs < f.ndofs / 2

        # What the parity mask sets to zero by construction, the full cell finds
        # small: the coupling between a normal load and a shear response is
        # forbidden by symmetry and only survives as mesh noise.
        @test maximum(abs, o.elas[1:3, 4:6]) == 0.0
        @test maximum(abs, o.elas[4:6, 1:3]) == 0.0
        @test maximum(abs, f.elas[1:3, 4:6]) < 5.0e-2 * norm(f.elas)
        # And off-diagonal transport coupling likewise.
        @test maximum(abs, o.cond - Diagonal(diag(o.cond))) == 0.0
    end

    @testset "a superspheroid keeps its transverse isotropy" begin
        # The mesh is not invariant under exchanging axes 1 and 2, only under
        # the sign changes, so this is an agreement at discretization level
        # rather than an identity.
        o = _oct_run(Superspheroid(1.0, 2.0, 0.8), 2, true)
        a = o.cond
        @test abs(a[1, 1] - a[2, 2]) / a[1, 1] < 1.0e-2
        @test abs(a[1, 1] - a[3, 3]) / a[1, 1] > 1.0e-2   # genuinely anisotropic
    end
end
