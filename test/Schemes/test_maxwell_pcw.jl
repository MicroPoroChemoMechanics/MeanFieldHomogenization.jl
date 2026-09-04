# =============================================================================
#  test_maxwell_pcw.jl — Maxwell and Ponte-Castañeda-Willis schemes.
#
#  Coverage:
#   1. Single-phase RVE returns the matrix property.
#   2. With UniformDistribution, Maxwell and PCW agree (single-shape case).
#   3. Bracketed by Voigt/Reuss bounds.
#   4. Reduce to MoriTanaka when the distribution shape coincides with the
#      inclusion shape.
#   5. Distribution-shape sensitivity: oblate distribution induces TI
#      anisotropy in the effective tensor.
#   6. Conductivity (`property = :K`).
#   7. ForwardDiff sensitivity to f.
#   8. Complex moduli (frequency-domain).
#   9. Symbol shortcuts.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const ATOL_MX = 1.0e-10
const RTOL_MX = 1.0e-9

@testset "Maxwell / PCW — sanity (single-phase)" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    @test homogenize(rve, Maxwell()) ≈ C_m
    @test homogenize(rve, PonteCastanedaWillis()) ≈ C_m
end

@testset "Maxwell ≡ PCW with UniformDistribution" begin
    rve = RVE(; distribution_shape = Ellipsoid(1.0, 1.0, 0.5))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 0.5), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 0.5),
        Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.3
    )
    @test homogenize(rve, Maxwell()) ≈ homogenize(rve, PonteCastanedaWillis())
end

@testset "Maxwell — bracketed by Voigt/Reuss" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )

    Vv = get_array(homogenize(rve, Voigt()))[1, 1, 1, 1]
    Vr = get_array(homogenize(rve, Reuss()))[1, 1, 1, 1]
    Vm = get_array(homogenize(rve, Maxwell()))[1, 1, 1, 1]
    Vp = get_array(homogenize(rve, PonteCastanedaWillis()))[1, 1, 1, 1]
    @test Vr - RTOL_MX * abs(Vr) ≤ Vm ≤ Vv + RTOL_MX * abs(Vv)
    @test Vr - RTOL_MX * abs(Vr) ≤ Vp ≤ Vv + RTOL_MX * abs(Vv)
end

@testset "Maxwell ≡ MT when distribution shape = inclusion shape (iso sphere)" begin
    # When all inclusions and the outer distribution have the same spherical
    # shape, Maxwell coincides with Mori-Tanaka by construction.
    rve = RVE(; distribution_shape = Ellipsoid(1.0))   # sphere outer
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    @test homogenize(rve, Maxwell()) ≈ homogenize(rve, MoriTanaka())
end

@testset "Maxwell — distribution shape induces anisotropy" begin
    # Oblate outer (1, 1, 0.3) + iso phases → effective tensor shows TI
    # symmetry (axial vs transverse components differ).
    rve = RVE(; distribution_shape = Ellipsoid(1.0, 1.0, 0.3))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    Cm = get_array(homogenize(rve, Maxwell()))
    # transverse [1111] and axial [3333] no longer equal
    @test !isapprox(Cm[1, 1, 1, 1], Cm[3, 3, 3, 3]; atol = 1.0e-3)
end

@testset "Maxwell / PCW — conductivity" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(2.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(8.0));
        fraction = 0.3
    )
    Kv = get_array(homogenize(rve, Voigt(); property = :K))[1, 1]
    Kr = get_array(homogenize(rve, Reuss(); property = :K))[1, 1]
    Km = get_array(homogenize(rve, Maxwell(); property = :K))[1, 1]
    Kp = get_array(homogenize(rve, PonteCastanedaWillis(); property = :K))[1, 1]
    @test Kr - RTOL_MX * abs(Kr) ≤ Km ≤ Kv + RTOL_MX * abs(Kv)
    @test Kr - RTOL_MX * abs(Kr) ≤ Kp ≤ Kv + RTOL_MX * abs(Kv)
end

@testset "Maxwell — ForwardDiff sensitivity" begin
    f_max(f) = begin
        DT = typeof(f)
        rve = RVE(; T = DT)
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
            fraction = f
        )
        get_array(homogenize(rve, Maxwell()))[1, 1, 1, 1]
    end
    df = ForwardDiff.derivative(f_max, 0.3)
    @test isfinite(df)
    @test df > 0
end

@testset "Maxwell / PCW — Complex moduli" begin
    δ = 0.05
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0 + δ * im, 10.0 + 0.5δ * im)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(60.0 + δ * im, 20.0 + 0.5δ * im));
        fraction = 0.3
    )

    for sch in (Maxwell(), PonteCastanedaWillis())
        Cs = homogenize(rve, sch)
        @test eltype(Cs) <: Complex
        @test all(isfinite, get_array(Cs))
    end

    # Im → 0 limit
    rve_re = RVE()
    add_phase!(rve_re, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve_re, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    rve_0 = RVE()
    add_phase!(rve_0, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0 + 0im, 10.0 + 0im)); fraction = :rest)
    add_phase!(
        rve_0, :I, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(60.0 + 0im, 20.0 + 0im)); fraction = 0.3
    )
    for sch in (Maxwell(), PonteCastanedaWillis())
        C_re = get_array(homogenize(rve_re, sch))
        C_0 = get_array(homogenize(rve_0, sch))
        @test maximum(abs.(real.(C_0) .- C_re)) < ATOL_MX
        @test maximum(abs.(imag.(C_0))) < ATOL_MX
    end
end

@testset "Maxwell / PCW — Symbol shortcuts (lowercase canonical)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )

    @test homogenize(rve, :maxwell) ≈ homogenize(rve, Maxwell())
    @test homogenize(rve, :Maxwell) ≈ homogenize(rve, Maxwell())
    @test homogenize(rve, :MAX) ≈ homogenize(rve, Maxwell())
    @test homogenize(rve, :pcw) ≈ homogenize(rve, PonteCastanedaWillis())
    @test homogenize(rve, :PCW) ≈ homogenize(rve, PonteCastanedaWillis())
    @test homogenize(rve, :ponte_castaneda_willis) ≈
        homogenize(rve, PonteCastanedaWillis())
end
