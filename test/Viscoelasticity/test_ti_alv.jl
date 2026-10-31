using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_ti_alv.jl — transversely isotropic ALV operators: the schemes computed
#  in the TI class (a 2×2 block and two scalars per pair of instants) against
#  the same schemes on the dense 6n×6n matrices, and the precision of the
#  element type kept through the class.
# =============================================================================

const _TI_AXIS = (0.0, 0.0, 1.0)

@testset "ti_alv — schemes in the TI class match the dense matrices" begin
    n = 4
    times = collect(range(0.0, 1.0; length = n))
    C_M_law = heaviside_law(TensTI{4}(4.0, 6.0, 1.0, 1.0, 2.0, _TI_AXIS))
    C_I_law = heaviside_law(TensISO{3}(3 * 10.0, 2 * 4.0))
    C_M, C_I = trapezoidal_matrix(C_M_law, times), trapezoidal_matrix(C_I_law, times)
    R_M, R_I = VolterraOperator(C_M_law, times), VolterraOperator(C_I_law, times)
    @test R_M.class isa TensTI && R_I.class isa TensISO
    f_I = 0.3
    # Voigt and Reuss, on the operators (the isotropic one joins the TI class).
    voigt = (1 - f_I) * R_M + f_I * R_I
    @test voigt.class isa TensTI
    @test Matrix(voigt) ≈ voigt_alv([C_M, C_I], [1 - f_I, f_I]) rtol = 1.0e-13
    reuss = inv((1 - f_I) * inv(R_M) + f_I * inv(R_I))
    @test reuss.class isa TensTI
    @test Matrix(reuss) ≈ reuss_alv([C_M, C_I], [1 - f_I, f_I]) rtol = 1.0e-12
end

@testset "ti_alv — homogenize_alv computes in the TI class for a TI matrix" begin
    n = 6
    times = collect(range(0.0, 2.0; length = n))
    C_M_law = heaviside_law(TensTI{4}(4.0, 6.0, 1.0, 1.5, 2.0, _TI_AXIS))
    C_I_law = heaviside_law(TensISO{3}(3 * 10.0, 2 * 4.0))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M_law); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_I_law); fraction = 0.2)
    C_eff = homogenize_alv(rve, Voigt(), :C; times = times)
    @test C_eff.class isa TensTI
    @test size(C_eff) == (6n, 6n)
    C_ref = voigt_alv([trapezoidal_matrix(C_M_law, times), trapezoidal_matrix(C_I_law, times)], [0.8, 0.2])
    @test C_eff ≈ C_ref rtol = 1.0e-13
end

@testset "TI blocks keep the precision of their element type" begin
    # The adapted basis takes its √2 in the element type: a BigFloat coupling
    # comes back to the precision of BigFloat.
    C = TensTI{4}(big(4.0), big(6.0), big(1.0), big(1.5), big(2.0), (big(0.0), big(0.0), big(1.0)))
    R = VolterraOperator(C, [0.0, 1.0])
    @test eltype(R) == BigFloat
    @test Matrix(R)[1:6, 1:6] ≈ mandel(canonical_components(C); symmetrize = true) rtol = 1.0e-70
    # Integer parameters give float blocks: the entries are divided.
    @test iso_blocks_from_params(fill(3, 1, 1), fill(2, 1, 1))[1, 1] ≈ 7 / 3
end
