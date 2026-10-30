# =============================================================================
#  test_structured_phases.jl — every linear scheme with phases stored as
#  structured tensors on frames and axes of their own, against the same scheme
#  with the same phases stored as generic tensors.
#
#  A scheme must not depend on how a phase is stored. The matrix is a cubic
#  crystal on a rotated cube, the two families of inclusions are transversely
#  isotropic about a random axis and orthotropic on another rotated frame: the
#  schemes add and subtract such tensors, and the self-consistent scheme used
#  to fail on the sum of two cubic tensors on different cubes (TensND refused
#  it). The generic tensors are their canonical components, a reference
#  independent of the structured algebra.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random

_sp_rng = MersenneTwister(20261029)
_sp_frame() = TensND.random_basis(_sp_rng)
_sp_generic(C) = TensND.from_mandel(mandel(TensND.canonical(C)))

function _sp_rve(M, I, J)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => M); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(2.0, 1.0, 1.0), Dict(:C => I); fraction = 0.15)
    add_phase!(rve, :J, Ellipsoid(1.0), Dict(:C => J); fraction = 0.1)
    return rve
end

@testset "linear schemes: structured phases as their generic tensors" begin
    M = TensND.cubic_from_constants(168.4e3, 121.4e3, 75.4e3, _sp_frame())
    I = TensND.ti_from_constants(1.0e5, 3.0e4, 2.5e4, 1.2e5, 2.0e4, normalize(randn(_sp_rng, 3)))
    J = TensND.TensOrtho(9.0e4, 8.0e4, 7.0e4, 1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.0e4, _sp_frame())
    structured, generic = _sp_rve(M, I, J), _sp_rve(_sp_generic(M), _sp_generic(I), _sp_generic(J))
    for scheme in (Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(), SelfConsistent(), AsymmetricSelfConsistent())
        Cs = mandel(TensND.canonical(homogenize(structured, scheme, :C; reltol = 1.0e-10)))
        Cg = mandel(TensND.canonical(homogenize(generic, scheme, :C; reltol = 1.0e-10)))
        @test Cs ≈ Cg rtol = 1.0e-8
    end
    # Two cubic phases on different cubes, the case that failed.
    cubic = _sp_rve(M, TensND.cubic_from_constants(250.0e3, 150.0e3, 120.0e3, _sp_frame()), TensND.cubic_from_constants(100.0e3, 60.0e3, 40.0e3, _sp_frame()))
    plain = _sp_rve(_sp_generic(M), map(φ -> _sp_generic(MeanFieldHomogenization.Core.cell_container_property(cubic, φ, :C)), (:I, :J))...)
    for scheme in (MoriTanaka(), SelfConsistent())
        @test mandel(TensND.canonical(homogenize(cubic, scheme, :C; reltol = 1.0e-10))) ≈
            mandel(TensND.canonical(homogenize(plain, scheme, :C; reltol = 1.0e-10))) rtol = 1.0e-8
    end
end
