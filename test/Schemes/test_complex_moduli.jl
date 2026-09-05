# =============================================================================
#  test_complex_moduli.jl — cross-cutting frequency-domain compatibility.
#
#  Sweeps a Maxwell-model viscoelastic 2-phase RVE over a range of angular
#  frequencies and verifies that every scheme:
#   * produces a `Complex{Float64}` result (eltype propagation),
#   * agrees with the real-modulus result in the limit Im(modulus) → 0,
#   * preserves causality (Im(C_eff[1111]) ≥ -tol).
#
#  Complex moduli need NO declaration on the RVE: the volume fraction stays
#  real, the element types are promoted where the values meet.  `RVE()`,
#  `RVE{ComplexF64}()` and `RVE(; T = ComplexF64)` must all give the same
#  numbers here — that equivalence is asserted below.
#
#  Assertions are deliberately NOT wrapped in try/catch: a scheme that stops
#  working in the complex plane must break this file, not disappear into a
#  `@test_broken`.  The single known gap is documented at the bottom.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND

const ATOL_FREQ = 1.0e-10

# Every scheme that must work with complex moduli.
complex_schemes() = [
    "Voigt" => Voigt(),
    "Reuss" => Reuss(),
    "Dilute" => Dilute(),
    "DiluteDual" => DiluteDual(),
    "MoriTanaka" => MoriTanaka(),
    "Maxwell" => Maxwell(),
    "PonteCastanedaWillis" => PonteCastanedaWillis(),
    "SelfConsistent" => SelfConsistent(; abstol = 1.0e-10, maxiters = 200),
    "AsymmetricSelfConsistent" =>
        AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 200),
    "DifferentialScheme" => DifferentialScheme(; nsteps = 50),
]

function _two_phase_rve(C_m, C_i, f; kw...)
    # The scheme list includes Maxwell and PCW, which read the distribution
    # shape and no longer default it. A unit sphere is what they used to get.
    rve = RVE(; distribution_shape = Ellipsoid(1.0), kw...)
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = f)
    return rve
end

@testset "Schemes — Complex moduli sweep" begin
    f_inc = 0.3
    # Sweep ω with Maxwell-model loss factor δ = 0.05 ω / (1 + ω²)
    for ω in (0.01, 0.1, 1.0, 10.0)
        δ = 0.05 * ω / (1 + ω^2)
        C_m = TensISO{3}(30.0 + δ * im, 10.0 + 0.5δ * im)
        C_i = TensISO{3}(60.0 + δ * im, 20.0 + 0.5δ * im)
        # No `T = …`: the fraction stays real, the moduli carry the complex part.
        rve_c = _two_phase_rve(C_m, C_i, f_inc)

        @testset "ω = $ω — $name" for (name, sch) in complex_schemes()
            Cs = homogenize(rve_c, sch)
            @test eltype(Cs) <: Complex
            @test all(isfinite, get_array(Cs))
            # Causality (loose, allow numerical noise)
            @test imag(get_array(Cs)[1, 1, 1, 1]) ≥ -1.0e-6
        end
    end
end

@testset "Schemes — complex moduli need no RVE declaration" begin
    # The three construction forms must be strictly interchangeable.
    δ = 0.05
    C_m = TensISO{3}(30.0 + δ * im, 10.0 + 0.5δ * im)
    C_i = TensISO{3}(60.0 + δ * im, 20.0 + 0.5δ * im)
    f_inc = 0.3

    rve_plain = _two_phase_rve(C_m, C_i, f_inc)                       # no `T` declared
    rve_param = let r = RVE{ComplexF64}(; distribution_shape = Ellipsoid(1.0))  # parametric form
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = f_inc)
        r
    end
    rve_kwarg = _two_phase_rve(C_m, C_i, ComplexF64(f_inc); T = ComplexF64)  # legacy form

    @test rve_param isa RVE{ComplexF64}
    @test rve_kwarg isa RVE{ComplexF64}
    @test eltype(rve_plain) === Float64        # the amount really stays real
    @test eltype(rve_param) === ComplexF64     # the floor widens it

    @testset "$name" for (name, sch) in complex_schemes()
        C_plain = get_array(homogenize(rve_plain, sch))
        C_param = get_array(homogenize(rve_param, sch))
        C_kwarg = get_array(homogenize(rve_kwarg, sch))
        @test maximum(abs.(C_plain .- C_param)) < ATOL_FREQ
        @test maximum(abs.(C_plain .- C_kwarg)) < ATOL_FREQ
    end
end

@testset "Schemes — Im → 0 limit consistency" begin
    f_inc = 0.3
    rve_re = _two_phase_rve(TensISO{3}(30.0, 10.0), TensISO{3}(60.0, 20.0), f_inc)
    rve_0 = _two_phase_rve(
        TensISO{3}(30.0 + 0im, 10.0 + 0im), TensISO{3}(60.0 + 0im, 20.0 + 0im), f_inc
    )

    @testset "$name" for (name, sch) in complex_schemes()
        C_re = get_array(homogenize(rve_re, sch))
        C_0 = get_array(homogenize(rve_0, sch))
        @test maximum(abs.(real.(C_0) .- C_re)) < ATOL_FREQ
        @test maximum(abs.(imag.(C_0))) < ATOL_FREQ
    end
end

@testset "Schemes — known complex-plane gap: SC Newton solver" begin
    # `SelfConsistent(algorithm = NewtonDefault())` builds its Jacobian with
    # ForwardDiff, which requires a real scalar type.  The default Anderson
    # solver (asserted above) is the complex-capable path.  Documented in
    # docs/src/manual/schemes.md § Complex-valued workflows.
    δ = 0.05
    rve_c = _two_phase_rve(
        TensISO{3}(30.0 + δ * im, 10.0 + 0.5δ * im),
        TensISO{3}(60.0 + δ * im, 20.0 + 0.5δ * im), 0.3
    )
    sc_newton = SelfConsistent(; algorithm = NewtonDefault(), abstol = 1.0e-10)
    @test_throws ArgumentError homogenize(rve_c, sc_newton)
end
