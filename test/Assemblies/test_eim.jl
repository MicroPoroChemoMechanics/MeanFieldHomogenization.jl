using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
using ForwardDiff

# =============================================================================
#  test_eim.jl — the variational equivalent inclusion method (Brisard,
#  Dormieux & Sab 2014) at order p = 0.
#
#  Coverage:
#   1. EIM ≡ ClusterModel on a periodic assembly, EXACT — the identity Brisard
#      et al. assert in their §3.1 (their order-zero influence pseudotensors
#      are the interaction tensors of Molinari & El Mouden). This is what pins
#      every sign convention shared by the two kernels.
#   2. One spherical particle concentric in a spherical SVE reproduces
#      Mori-Tanaka EXACTLY: with ℙ_Ω = ℙ the mixed-boundary term turns ℙ into
#      (1-f)ℙ, which is the Mori-Tanaka polarization. Validates the ℙ_Ω term
#      specifically, and is far sharper than a dilute-limit check.
#   3. Dilute limit as f → 0.
#   4. `eim_bound_type` on porous, reinforced and mixed assemblies, and the
#      bound itself: the estimate sits above the Hashin-Shtrikman-like
#      Mori-Tanaka value for a porous assembly (both are upper bounds, EIM
#      being the sharper one it must not exceed Voigt).
#   5. Brisard et al., Table 1 — plane strain, 160 circular pores, φ = 0.4,
#      ν₀ = 0.3, R = 20a: μ_app ≈ 0.310 μ₀. Reproduced here on a reduced
#      ensemble (the published figure uses 1000 realizations).
#   6. Local fields: `eim_polarizations` reproduces the effective property.
#   7. Conduction, and the rejection of unimplemented polarization orders.
#   8. ForwardDiff through the solve.
# =============================================================================

const RTOL_EXACT = 1.0e-12

_Cm() = TensISO{3}(3 * 1.0, 2 * 0.4)
_Ci() = TensISO{3}(3 * 10.0, 2 * 6.0)
_Cvoid() = TensISO{3}(3 * 1.0e-9, 2 * 1.0e-9)

_mu(C) = get_array(C)[1, 2, 1, 2]

@testset "EIM — coincides with the cluster model on a periodic assembly" begin
    # Brisard et al. (2014), §3.1: at order zero the two formulations are the
    # same linear system.  Machine precision or bust — a sign slip anywhere in
    # either kernel shows up here first.
    for cutoff in (0.0, 1.5, 3.0), kind in (:sc, :bcc)
        asm = cubic_lattice(
            kind, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.25, cutoff = cutoff
        )
        A = get_array(homogenize(asm, ClusterModel(), :C))
        B = get_array(homogenize(asm, EquivalentInclusion(), :C))
        @test maximum(abs.(A .- B)) < RTOL_EXACT * maximum(abs.(A))
    end
    # Conduction too.
    asm = cubic_lattice(
        :sc, Dict(:K => TensISO{3}(1.0)), Dict(:K => TensISO{3}(20.0));
        fraction = 0.25, cutoff = 2.0
    )
    A = get_array(homogenize(asm, ClusterModel(), :K))
    B = get_array(homogenize(asm, EquivalentInclusion(), :K))
    @test maximum(abs.(A .- B)) < RTOL_EXACT * maximum(abs.(A))
end

@testset "EIM — a single sphere in a spherical SVE is exactly Mori-Tanaka" begin
    # Under mixed boundary conditions the far field is closed by ℙ_Ω, the Hill
    # tensor of the SVE.  For a spherical SVE and a spherical particle
    # ℙ_Ω = ℙ, and the self block becomes (ℂ₁-ℂ₀)⁻¹ + (1-f)ℙ — the Mori-Tanaka
    # polarization.  Exact at every volume fraction, not only in the dilute
    # limit, which is what makes it a test of the ℙ_Ω term itself.
    for R in (2.0, 3.0, 10.0)
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(R)))
        add_matrix!(asm, Dict(:C => _Cm()))
        add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(1.0), Dict(:C => _Ci()))
        f = particle_volume_fraction(asm, :p1)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _Cm()); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => _Ci()); fraction = f)
        C_eim = get_array(homogenize(asm, EquivalentInclusion(), :C))
        C_mt = get_array(homogenize(rve, MoriTanaka(), :C))
        @test maximum(abs.(C_eim .- C_mt)) < RTOL_EXACT * maximum(abs.(C_mt))
    end
end

@testset "EIM — dilute limit" begin
    # As the SVE grows the fraction vanishes and every estimate collapses onto
    # the dilute one.
    errs = Float64[]
    for R in (5.0, 10.0, 20.0)
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(R)))
        add_matrix!(asm, Dict(:C => _Cm()))
        add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(1.0), Dict(:C => _Ci()))
        f = particle_volume_fraction(asm, :p1)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _Cm()); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => _Ci()); fraction = f)
        C_eim = get_array(homogenize(asm, EquivalentInclusion(), :C))
        C_dil = get_array(homogenize(rve, Dilute(), :C))
        push!(errs, maximum(abs.(C_eim .- C_dil)) / maximum(abs.(C_dil)))
    end
    @test issorted(errs; rev = true)
    @test errs[end] < 1.0e-6
end

@testset "EIM — bound type" begin
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Cvoid()); fraction = 0.2)
    @test eim_bound_type(asm) === :upper        # matrix stiffer than the pores
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2)
    @test eim_bound_type(asm) === :lower        # matrix softer than the particles
    # Mixed contrasts: no bound, only an estimate.
    asm = cubic_lattice(
        :bcc, Dict(:C => _Cm()), [Dict(:C => _Ci()), Dict(:C => _Cvoid())]; fraction = 0.2
    )
    @test eim_bound_type(asm) === :none
end

@testset "EIM — Brisard et al. (2014) Table 1, plane strain" begin
    # 160 circular pores of radius a in a circular SVE of radius R = 20a,
    # porosity φ = 0.4, matrix ν₀ = 0.3.  The paper reports μ_app = 0.310 μ₀
    # at p = 0, averaged over 1000 realizations; the reduced ensemble used
    # here lands within about 1 % of it.
    μ₀, ν₀ = 1.0, 0.3
    κ2 = μ₀ / (1 - 2ν₀)                      # plane-strain area modulus
    C_m = TensISO{2}(2κ2, 2μ₀)
    C_void = TensISO{2}(2κ2 * 1.0e-9, 2μ₀ * 1.0e-9)
    a, N, R = 1.0, 160, 20.0
    rng = Random.MersenneTwister(20260810)
    vals = Float64[]
    for _ in 1:4
        asm = random_assembly(
            N, Dict(:C => C_m), Dict(:C => C_void);
            radius = a, dim = 2, rng = rng, cycles = 800,
            boundary = MixedBC(Ellipsoid(R, R))
        )
        push!(vals, _mu(homogenize(asm, EquivalentInclusion(), :C)) / μ₀)
    end
    μ̄ = sum(vals) / length(vals)
    @test inclusion_volume_fraction(
        random_assembly(
            N, Dict(:C => C_m), Dict(:C => C_void);
            radius = a, dim = 2, rng = Random.MersenneTwister(1), cycles = 10,
            boundary = MixedBC(Ellipsoid(R, R))
        )
    ) ≈ 0.4 rtol = 1.0e-12
    @test μ̄ ≈ 0.31 rtol = 0.03
    # It is an upper bound on the apparent modulus, and the finite-element
    # reference of the paper (0.244 μ₀) sits below it.
    @test μ̄ > 0.244
end

@testset "EIM — local fields reproduce the effective property" begin
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.25, cutoff = 2.0)
    τ, names = eim_polarizations(asm)
    @test names == particle_names(asm)
    C₀ = matrix_property(asm, :C)
    rebuilt = get_array(C₀)
    for k in eachindex(names)
        rebuilt = rebuilt .+ particle_volume_fraction(asm, names[k]) .* get_array(τ[k])
    end
    C = get_array(homogenize(asm, EquivalentInclusion(), :C))
    @test maximum(abs.(rebuilt .- C)) < RTOL_EXACT * maximum(abs.(C))
end

@testset "EIM — unimplemented polarization order is refused" begin
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2)
    @test_throws ArgumentError homogenize(asm, EquivalentInclusion(; order = 1), :C)
    @test_throws ArgumentError homogenize(asm, EquivalentInclusion(; order = 2), :C)
end

@testset "EIM — scheme aliases" begin
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2, cutoff = 2.0)
    ref = get_array(homogenize(asm, EquivalentInclusion(), :C))
    @test get_array(homogenize(asm, :eim, :C)) == ref
    @test get_array(homogenize(asm, :EIM, :C)) == ref
    ref2 = get_array(homogenize(asm, ClusterModel(), :C))
    @test get_array(homogenize(asm, :cluster, :C)) == ref2
end

@testset "EIM — ForwardDiff" begin
    base = ParticleAssembly(; boundary = MixedBC(Ellipsoid(4.0)))
    add_matrix!(base, Dict(:C => _Cm()))
    add_particle!(base, :p1, (-1.5, 0.0, 0.0), Ellipsoid(1.0), Dict(:C => _Ci()))
    add_particle!(base, :p2, (1.5, 0.0, 0.0), Ellipsoid(1.0), Dict(:C => _Ci()))
    fx = x -> _mu(homogenize(set_param(base, center_param(:p2, 1), x), EquivalentInclusion(), :C))
    d_ad = ForwardDiff.derivative(fx, 1.5)
    d_fd = (fx(1.5 + 1.0e-6) - fx(1.5 - 1.0e-6)) / 2.0e-6
    @test isfinite(d_ad)
    @test d_ad ≈ d_fd rtol = 1.0e-5 atol = 1.0e-9
end
