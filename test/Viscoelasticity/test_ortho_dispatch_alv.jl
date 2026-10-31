using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_ortho_dispatch_alv.jl — the ORTHO fast paths of
#  `_homogenize_alv_dispatch` (`src/Viscoelasticity/homogenize_alv.jl`).
#
#  A genuinely orthotropic RVE (neither iso nor TI): its operators are stored,
#  and its schemes computed, in the orthotropic class, and the results are those
#  of the dense 6n×6n matrices.
# =============================================================================

const _ORTHO_FRAME = TensND.CanonicalBasis{3, Float64}()

# A genuinely orthotropic tensor: the three normal moduli and the three shear
# moduli are all distinct, so it is neither isotropic nor TI about any axis.
_ortho_tensor(s) = TensND.TensOrtho(
    20.0s, 8.0s, 6.0s,       # C11, C12, C13
    30.0s, 7.0s, 40.0s,      # C22, C23, C33
    5.0s, 6.0s, 7.0s,        # C44, C55, C66
    _ORTHO_FRAME
)

function _ortho_rve(; fraction = 0.25)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(_ortho_tensor(1.0))); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(_ortho_tensor(2.0)));
        fraction = fraction
    )
    return rve
end

@testset "ortho dispatch — le RVE est bien hors des raccourcis iso et TI" begin
    times = collect(0.0:0.5:1.0)
    R_M = VolterraOperator(heaviside_law(_ortho_tensor(1.0)), times)
    R_I = VolterraOperator(heaviside_law(_ortho_tensor(2.0)), times)
    # Precondition of this test: the operators are orthotropic, not narrower.
    @test R_M.class isa TensOrtho && R_I.class isa TensOrtho
    @test (R_M * R_I).class isa TensOrtho
    # A matrix out of the ortho form cannot be stored in it.
    M_bad = Matrix(R_M)
    M_bad[1, 4] += 1.0
    @test_throws "does not belong" VolterraOperator(M_bad, times; like = _ortho_tensor(1.0))
    # Iso ⊂ TI ⊂ ortho: an isotropic operator stays isotropic, and joins the ortho class.
    R_iso = VolterraOperator(heaviside_law(TensISO{3}(3.0, 2.0)), times)
    @test R_iso.class isa TensISO
    @test (R_iso + R_M).class isa TensOrtho
end

@testset "ortho dispatch — Voigt et Reuss passent par le chemin ortho" begin
    times = collect(0.0:0.5:1.5)
    n = length(times)
    f = 0.25
    rve = _ortho_rve(fraction = f)

    M_M = trapezoidal_matrix(heaviside_law(_ortho_tensor(1.0)), times)
    M_I = trapezoidal_matrix(heaviside_law(_ortho_tensor(2.0)), times)
    fr = [1 - f, f]

    C_voigt = homogenize_alv(rve, Voigt(), :C; times = times)
    @test size(C_voigt) == (6n, 6n)
    @test C_voigt.class isa TensOrtho            # on est bien resté ortho
    @test isapprox(C_voigt, voigt_alv([M_M, M_I], fr); atol = 1.0e-10)

    C_reuss = homogenize_alv(rve, Reuss(), :C; times = times)
    @test C_reuss.class isa TensOrtho
    @test isapprox(C_reuss, reuss_alv([M_M, M_I], fr); atol = 1.0e-8)

    # Encadrement de Voigt-Reuss sur les termes diagonaux.
    for i in 1:(6n)
        @test C_reuss[i, i] ≤ C_voigt[i, i] + 1.0e-9
    end
end

@testset "ortho dispatch — Dilute, Mori-Tanaka et Maxwell restent ortho" begin
    times = collect(0.0:0.5:1.5)
    n = length(times)
    rve = _ortho_rve(fraction = 0.2)

    for scheme in (Dilute(), MoriTanaka(), Maxwell())
        C = homogenize_alv(rve, scheme, :C; times = times)
        @test size(C) == (6n, 6n)
        @test all(isfinite, C)
        # These schemes preserve the ortho form: coaxial ortho phases give an
        # ortho result in the same material frame.
        @test C.class isa TensOrtho
    end
end

@testset "ortho dispatch — fraction nulle redonne la matrice" begin
    times = collect(0.0:0.5:1.0)
    rve = _ortho_rve(fraction = 0.0)
    M_M = trapezoidal_matrix(heaviside_law(_ortho_tensor(1.0)), times)

    for scheme in (Voigt(), Reuss(), Dilute(), MoriTanaka())
        C = homogenize_alv(rve, scheme, :C; times = times)
        @test isapprox(C, M_M; atol = 1.0e-8)
    end
end

@testset "ortho dispatch — round-trip through the orthotropic class" begin
    times = collect(0.0:0.5:1.5)
    M = trapezoidal_matrix(heaviside_law(_ortho_tensor(1.3)), times)
    R = VolterraOperator(M, times; like = _ortho_tensor(1.0))
    @test R.class isa TensOrtho
    @test length(R.blocks) == 4
    @test isapprox(Matrix(R), M; atol = 1.0e-12)
end
