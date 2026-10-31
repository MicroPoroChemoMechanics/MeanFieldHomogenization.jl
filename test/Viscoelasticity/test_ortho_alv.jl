using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random: Xoshiro

# =============================================================================
#  test_ortho_alv.jl — orthotropic ALV operators: the algebra and the schemes
#  in the orthotropic class (a 3×3 block and three scalars per pair of
#  instants), in a randomly drawn frame, against the same computations on the
#  dense 6n×6n matrices; and an ellipsoid whose result is orthotropic in its
#  own frame.
# =============================================================================

const _ORTHO_FRAME = random_basis(Xoshiro(20261031))

# A general orthotropic law (twelve constants, not major-symmetric) in the frame
# above, aging and relaxing.
function _ortho_law(seed)
    c = (6.0, 7.0, 8.0, 1.0 + 0.1seed, 1.5, 2.0, 1.2, 2.2, 3.1, 0.8, 1.3, 1.9 - 0.1seed)
    return ViscoLaw((t, tp) -> (1 + exp(-(t - tp) / (1 + seed))) * TensOrtho{4, Float64}(c, _ORTHO_FRAME))
end

@testset "ortho_alv — the algebra in a random frame matches the dense matrices" begin
    times = collect(range(0.0, 3.0; length = 5))
    a, b, p = _ortho_law(1), _ortho_law(2), _ortho_law(3)
    Ra, Rb, Rp = (VolterraOperator(l, times) for l in (a, b, p))
    Ma, Mb, Mp = (trapezoidal_matrix(l, times) for l in (a, b, p))
    @test all(R -> R.class isa TensOrtho, (Ra, Rb, Rp))
    f_a, f_b = 0.7, 0.3
    @test Matrix(f_a * Ra + f_b * Rb) ≈ voigt_alv([Ma, Mb], [f_a, f_b]) rtol = 1.0e-13
    @test Matrix(inv(f_a * inv(Ra) + f_b * inv(Rb))) ≈ reuss_alv([Ma, Mb], [f_a, f_b]) rtol = 1.0e-11
    A = inv(I + Rp * (Ra - Rb))
    @test (A).class isa TensOrtho
    @test Matrix(A) ≈ dilute_concentration_alv(Ma, Mb, Mp) rtol = 1.0e-11
    @test Matrix((Ra - Rb) * A) ≈ dilute_contribution_alv(Ma, Mb, Mp) rtol = 1.0e-11
end

@testset "ortho_alv — a tilted triaxial ellipsoid gives a result orthotropic in its frame" begin
    times = collect(range(0.0, 2.0; length = 6))
    matrix = maxwell_iso(10.0, 6.0, 0.8, 2.5)
    inclusion = heaviside_law(TensISO{3}(3 * 30.0, 2 * 20.0))
    angles = (0.4, 0.9, 0.2)
    function cell(shape)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => matrix); fraction = :rest)
        add_phase!(r, :I, shape, Dict(:C => inclusion); fraction = 0.2)
        return r
    end
    for scheme in (Dilute(), MoriTanaka())
        tilted = homogenize_alv(cell(Ellipsoid(2.0, 1.0, 0.4; euler_angles = angles)), scheme, :C; times)
        @test tilted.class isa TensOrtho
        # The result in the canonical frame, rotated block by block.
        aligned = Matrix(homogenize_alv(cell(Ellipsoid(2.0, 1.0, 0.4)), scheme, :C; times))
        Q6 = TensND.mandel_rotation(vecbasis(RotatedBasis(angles...)))
        Q = kron(Matrix(1.0I, length(times), length(times)), Q6)
        @test Matrix(tilted) ≈ Q * aligned * transpose(Q) rtol = 1.0e-12
    end
end

@testset "ortho_alv — isotropic phases stay isotropic" begin
    n = 6
    times = collect(range(0.0, 2.0; length = n))
    C_M_law = heaviside_law(TensISO{3}(3 * 5.0, 2 * 2.0))
    C_I_law = heaviside_law(TensISO{3}(3 * 10.0, 2 * 4.0))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M_law); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_I_law); fraction = 0.2)
    C_eff = homogenize_alv(rve, Voigt(), :C; times = times)
    @test C_eff.class isa TensISO
    C_ref = voigt_alv([trapezoidal_matrix(C_M_law, times), trapezoidal_matrix(C_I_law, times)], [0.8, 0.2])
    @test C_eff ≈ C_ref rtol = 1.0e-13
end
