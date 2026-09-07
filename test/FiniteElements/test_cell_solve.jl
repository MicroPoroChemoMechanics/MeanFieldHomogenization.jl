# The corrected solve on the three-dimensional cell.
#
# The gate is the **spherical pore**, whose localization is exact in both
# physics: `A = 3/2` in conduction and `[I - S]^-1` with Eshelby's `S` in
# elasticity. A supersphere at `p = 1` *is* a sphere, so this exercises the
# whole chain -- discrete surface, curved boundary, corrected condition, surface
# average -- against a closed form, and not merely a scalar.
#
# Everything runs at level 2, a few thousand tetrahedra: what is being tested is
# that the construction is right, not that the mesh is fine.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import Tensors
import Ferrite, FerriteGmsh
using Gmsh: gmsh

const FE = MeanFieldHomogenization.FiniteElements
const _B = FE.FerriteBackend()

# Exact strain localization of a spherical pore, Kelvin-Mandel.
function _sphere_pore_A(ν)
    SJ = (1 + ν) / (3 * (1 - ν))
    SK = 2 * (4 - 5ν) / (15 * (1 - ν))
    v = [1.0, 1, 1, 0, 0, 0]
    J = v * v' / 3
    K = Matrix(1.0I, 6, 6) - J
    return inv(Matrix(1.0I, 6, 6) - (SJ * J + SK * K))
end

function _iso_C(μ, ν)
    λ = 2μ * ν / (1 - 2ν)
    return Tensors.SymmetricTensor{4, 3}(
        (i, j, k, l) -> λ * (i == j) * (k == l) +
            μ * ((i == k) * (j == l) + (i == l) * (j == k))
    )
end

"Solve both physics on one mesh and report the errors against the exact sphere."
function _sphere_gate(level, radius_ratio; μ = 1.0, ν = 0.3)
    shape = Supersphere(1.0, 1.0)
    opts = FECellMeshOptions(;
        level, outer_level = 3, radius_ratio, relax = 20
    )
    g = FE.fe_cell_grid(_B, shape, opts)
    V = shape_volume(shape)

    sc = FE.fe_cell_space(_B, g.grid, 2, 1)
    c = FE._cell_conduction_localization(_B, sc, 1.0, V)
    cbad = FE._cell_conduction_localization(_B, sc, 1.0, V; sign = +1)

    se = FE.fe_cell_space(_B, g.grid, 2, 3)
    e = FE._cell_elastic_localization(_B, se, _iso_C(μ, ν), μ, ν, V)

    Aex = _sphere_pore_A(ν)
    return (;
        cond_err = norm(c.A - 1.5I) / 1.5,
        cond_raw = norm(c.A_uncorrected - 1.5I) / 1.5,
        cond_bad = norm(cbad.A - 1.5I) / 1.5,
        cond_dip = c.dipole_norm,
        elas_err = norm(e.A - Aex) / norm(Aex),
        elas_raw = norm(e.A_uncorrected - Aex) / norm(Aex),
        elas_dip = e.dipole_norm,
        A_elastic = e.A,
        ndofs_scalar = FE.fe_cell_dof_split(_B, sc)[1],
        ndofs_vector = FE.fe_cell_dof_split(_B, se)[1],
    )
end

@testset "supershape cell: the corrected solve" begin

    r2 = _sphere_gate(2, 3.0)
    r3 = _sphere_gate(3, 3.0)

    @testset "the spherical pore, against its closed form" begin
        @test r3.cond_err < 2.0e-3
        @test r3.elas_err < 2.0e-3
        @test r3.ndofs_scalar > 10^4
        @test r3.ndofs_vector == 3 * r3.ndofs_scalar
        # And it converges, at a useful rate.
        @test r3.cond_err < r2.cond_err / 5
        @test r3.elas_err < r2.elas_err / 5
    end

    @testset "the correction is what buys the accuracy" begin
        # An order of magnitude and more, in both physics.
        @test r3.cond_raw > 20 * r3.cond_err
        @test r3.elas_raw > 20 * r3.elas_err
        # And this is the point worth pinning: the *uncorrected* error is a
        # truncation bias, not a discretization error, so refining the mesh
        # barely touches it while the corrected one falls by an order of
        # magnitude on the same two meshes.
        @test r3.cond_raw > 0.5 * r2.cond_raw
        @test r3.elas_raw > 0.5 * r2.elas_raw
    end

    @testset "the corrected answer no longer depends on where the cell is cut" begin
        # The whole claim of the method, in one test. The uncorrected answer
        # improves steadily with R -- that is the O((a/R)^3) bias going away by
        # brute force -- while the corrected one is already flat.
        rs = (2.5, 3.0, 4.0, 5.0)
        g = [_sphere_gate(2, R) for R in rs]
        raw = [x.cond_raw for x in g]
        cor = [x.cond_err for x in g]
        @test raw[1] > 3 * raw[end]                       # the bias really is there
        @test maximum(cor) < 1.5 * minimum(cor)           # and the correction removes it
        @test maximum(cor) < 0.4 * raw[1]

        # `dipole_norm` is the diagnostic: the dipole term is O((a/R)^3), so the
        # log-log slope against R must come out near -3.
        d = [x.cond_dip for x in g]
        slope = log(d[end] / d[1]) / log(rs[end] / rs[1])
        @test -3.3 < slope < -2.6
    end

    @testset "the wrong sign doubles the bias instead of removing it" begin
        # The signature to recognize. It approaches a factor of two only where
        # the truncation bias dominates the discretization floor, which is why
        # this is read at the *smallest* R rather than the largest.
        g = _sphere_gate(2, 2.5)
        @test g.cond_bad > g.cond_raw          # worse than not correcting at all
        @test 1.4 < g.cond_bad / g.cond_raw < 2.2
        @test g.cond_bad > 5 * g.cond_err
    end

    @testset "a supersphere comes out cubic, as group theory says it must" begin
        # Not a sphere any more, so the answer is genuinely three-constant. What
        # is checked is the *class*: the relative distance of the raw 6x6 to the
        # cubic class is discretization error and nothing else, because the
        # shape and the matrix leave the octahedral group invariant.
        shape = Supersphere(1.0, 0.6)
        opts = FECellMeshOptions(;
            level = 2, outer_level = 3, radius_ratio = 3.0,
            relax = 20
        )
        g = FE.fe_cell_grid(_B, shape, opts)
        V = shape_volume(shape)
        μ, ν = 1.0, 0.3
        se = FE.fe_cell_space(_B, g.grid, 2, 3)
        e = FE._cell_elastic_localization(_B, se, _iso_C(μ, ν), μ, ν, V)
        A = TensND.Tens(TensND.inv_KM(e.A))
        @test cubic_residual(A) < 5.0e-3
        # And it is *not* isotropic: that is the third constant the literature's
        # two-constant fits discard.
        @test abs(cubic_anisotropy(A)) > 10 * cubic_residual(A)

        sc = FE.fe_cell_space(_B, g.grid, 2, 1)
        c = FE._cell_conduction_localization(_B, sc, 1.0, V)
        # At order two the cubic class *is* the isotropic class, so conduction
        # carries no anisotropy signal whatever the shape does in elasticity.
        @test norm(c.A - (tr(c.A) / 3) * I) / abs(tr(c.A) / 3) < 5.0e-3
        @test tr(c.A) / 3 > 1.5      # a concave pore blocks more than a sphere
    end
end
