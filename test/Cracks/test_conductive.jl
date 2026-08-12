using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const CANON_CD = TensND.CanonicalBasis{3, Float64}()
_g(t) = get_array(TensND.change_tens(t, CANON_CD))

@testset "conductive crack — closed form and limits" begin
    a, k₀, C = 1.0, 1.0e-3, 2.0e-2
    cr = ConductiveCrack(a; conductivity = C)
    @test fracture_conductivity(cr) == C

    # 𝕂 = γ/(1 + πγ/(4k₀)) (δ − n̂⊗n̂),  γ = C/2a
    γ = C / (2a)
    κ = γ / (1 + π * γ / (4k₀))
    K = _g(conductivity_contribution(cr, TensISO{3}(k₀)))
    @test K[1, 1] ≈ κ rtol = 1.0e-12
    @test K[2, 2] ≈ κ rtol = 1.0e-12
    @test K[3, 3] ≈ 0.0 atol = 1.0e-18      # purely in-plane
    @test K[1, 2] ≈ 0.0 atol = 1.0e-18

    # Ideal fracture C → ∞ : 𝕂 → (4k₀/π)(δ − n̂⊗n̂)
    ideal = _g(conductivity_contribution(ConductiveCrack(a; conductivity = 1.0e8), TensISO{3}(k₀)))
    @test ideal[1, 1] ≈ 4k₀ / π rtol = 1.0e-6
end

@testset "conductive crack — against the spheroid limit" begin
    # The independent gate: the same quantity as the ω → 0 limit of the
    # package's own (untouched) highly-conductive spheroid contribution, with
    # k_f = C/(2aω) so that C stays finite.
    a, k₀, C = 1.0, 1.0e-3, 2.0e-2
    cr = ConductiveCrack(a; conductivity = C)
    closed = _g(conductivity_contribution(cr, TensISO{3}(k₀)))[1, 1]
    prev = Inf
    for ω in (1.0e-3, 1.0e-4, 1.0e-5)
        N = conductivity_contribution(Ellipsoid(a, a, a * ω), TensISO{3}(C / (2a * ω)), TensISO{3}(k₀))
        err = abs(ω * _g(N)[1, 1] - closed)
        @test err < prev                      # converging to the closed form
        prev = err
    end
    @test prev / closed < 1.0e-4
end

@testset "conductive crack — anisotropic reference" begin
    # No closed form there; the limit is taken numerically and Richardson
    # extrapolated. Check it against a much smaller ω computed directly.
    a, C = 1.0, 2.0e-2
    cr = ConductiveCrack(a; conductivity = C)
    Kani = Tens([1.0e-3 0.0 0.0; 0.0 2.0e-3 0.0; 0.0 0.0 4.0e-4])
    K = _g(conductivity_contribution(cr, Kani))
    ref = _g(MeanFieldHomogenization.Cracks._flat_spheroid_limit(cr, Kani, 1.0e-6))
    @test K[1, 1] ≈ ref[1, 1] rtol = 1.0e-4
    @test K[2, 2] ≈ ref[2, 2] rtol = 1.0e-4
    @test abs(K[3, 3]) < 1.0e-3 * abs(K[1, 1])

    # An isotropic medium in generic storage must reproduce the closed form —
    # the two independent routes meeting.
    k₀ = 1.0e-3
    K_gen = _g(MeanFieldHomogenization.Cracks._conductive_crack_K(cr, Tens(get_array(TensISO{3}(k₀)))))
    K_iso = _g(conductivity_contribution(cr, TensISO{3}(k₀)))
    @test K_gen[1, 1] ≈ K_iso[1, 1] rtol = 1.0e-7
end

@testset "conductive crack — elasticity is inherited" begin
    # A flowing crack is mechanically an ordinary open crack.
    C₀ = TensISO{3}(3 * 30.0, 2 * 18.0)
    plain = PennyCrack(1.0; euler_angles = (π / 4, 0.0))
    flow = ConductiveCrack(plain, 1.0e-3)
    @test _g(cod_tensor(flow, C₀)) ≈ _g(cod_tensor(plain, C₀)) rtol = 1.0e-14
    @test _g(compliance_contribution(flow, C₀)) ≈ _g(compliance_contribution(plain, C₀)) rtol = 1.0e-14
    @test MeanFieldHomogenization.Cracks.crack_density_factor(flow) ≈ 4π / 3
end

@testset "conductive crack — guards" begin
    @test_throws ArgumentError ConductiveCrack(EllipticCrack(2.0, 1.0), 1.0e-3)
    @test_throws ArgumentError ConductiveCrack(1.0; conductivity = -1.0)
    cr = ConductiveCrack(1.0; conductivity = 1.0e-3)
    @test fracture_conductivity(with_conductivity(cr, 5.0e-4)) == 5.0e-4
end

@testset "fracture_permeability" begin
    ks, a, C = 1.0e-6, 1.0, 1.0e-3
    cr = ConductiveCrack(a; conductivity = C)

    # No fractures: the matrix.
    @test _g(fracture_permeability(ks, (cr,), (0.0,)))[1, 1] ≈ ks rtol = 1.0e-12

    # Dilute: one iteration of the additive condition.
    d = 1.0e-4
    γ = C / (2a); κ = γ / (1 + π * γ / (4ks))
    K = _g(fracture_permeability(ks, (cr,), (d,)))
    @test K[1, 1] ≈ ks + (4π / 3) * d * κ rtol = 1.0e-3
    @test K[3, 3] ≈ ks rtol = 1.0e-9          # no gain normal to the fractures

    # Two vertical families (normals e₁ and e₂): e₃ lies in BOTH planes, so it
    # must conduct most.
    fams = (
        ConductiveCrack(a; conductivity = C, euler_angles = (π / 2, 0.0)),
        ConductiveCrack(a; conductivity = C, euler_angles = (π / 2, π / 2)),
    )
    A = _g(fracture_permeability(ks, fams, (0.05, 0.05)))
    @test A[3, 3] > A[1, 1] > ks
    @test A[1, 1] ≈ A[2, 2] rtol = 1.0e-2     # symmetric pair

    # Monotone growth with density — the self-consistent estimate stiffens
    # sharply as the network connects.
    prev = ks
    for dd in (0.01, 0.05, 0.1)
        k33 = _g(fracture_permeability(ks, fams, (dd, dd)))[3, 3]
        @test k33 > prev
        prev = k33
    end

    @test_throws ArgumentError fracture_permeability(ks, (cr,), (0.1, 0.2))
end

@testset "the estimate really is self-consistent" begin
    # THE regression test for the fixed point. A conductivity of 10⁻¹⁸ m² is
    # the physical setting (a tight rock), and it is exactly where a fixed
    # absolute tolerance silently turns the estimate into the DILUTE one: the
    # first iterate is already within 10⁻¹⁰ of the second, the loop exits, and
    # the answer comes out linear in `d` and independent of the fracture
    # conductivity. Both closed forms below discriminate the two.
    ks = 1.0e-18

    # Quasi-uniform orientations (Fibonacci sphere), so the family average is
    # the isotropic one, ⟨𝟏 − n̲⊗n̲⟩ = (2/3)𝟏.
    n = 96
    ϕ = π * (3 - sqrt(5.0))
    dirs = [(acos(1 - 2(k - 0.5) / n), ϕ * k) for k in 1:n]
    fams = Tuple(
        ConductiveCrack(1.0; conductivity = 1.0e-6, euler_angles = e) for e in dirs
    )

    # A flowing crack saturates at 4k₀/π in a reference of conductivity k₀, so
    # each family contributes (4π/3)dᵢ(4k₀/π)⟨𝟏 − n̲⊗n̲⟩ = (32/9)dᵢ k₀ 𝟏 and the
    # self-consistent condition K = k_s + (32/9) d K closes in closed form:
    #
    #     K/k_s = 1/(1 − 32d/9),   percolating at d_c = 9/32.
    #
    # The dilute estimate would give 1 + 32d/9 instead — the two part company
    # by a factor 3 already at d = 0.2.
    for dtot in (0.05, 0.15, 0.25)
        K = _g(fracture_permeability(ks, fams, Tuple(fill(dtot / n, n)); maxiters = 5000))
        ratio = (K[1, 1] + K[2, 2] + K[3, 3]) / (3ks)
        @test ratio ≈ 1 / (1 - 32dtot / 9) rtol = 5.0e-3
        @test ratio > 1 + 32dtot / 9                 # NOT the dilute estimate
    end

    # Past the percolation threshold the estimate does not diverge: the
    # flowing-crack contribution saturates at the fracture conductivity itself,
    # so the network conducts at the γ = C/2a scale rather than the matrix one.
    two = (
        ConductiveCrack(1.0; conductivity = 6.7e-11, euler_angles = (π / 2, π / 8)),
        ConductiveCrack(1.0; conductivity = 6.7e-11, euler_angles = (π / 2, 7π / 8)),
    )
    below = _g(fracture_permeability(ks, two, (0.05, 0.05)))
    above = _g(fracture_permeability(ks, two, (0.37, 0.37); maxiters = 5000))
    @test below[2, 2] < 10ks                          # still a tight rock
    @test above[2, 2] > 1.0e5 * ks                    # …and a reservoir past it

    # Bounded above by the saturated contribution of every family, since
    # γ/(1 + πγ/4k₀) ≤ γ whatever the reference: with n̲ᵢ horizontal at 22.5°
    # and 157.5°, Σᵢ(𝟏 − n̲ᵢ⊗n̲ᵢ)₂₂ = 2 − 2sin²(22.5°).
    γ = 6.7e-11 / 2
    bound = ks + (4π / 3) * 0.37 * γ * (2 - 2 * sind(22.5)^2)
    @test above[2, 2] < bound
end

@testset "fracture_permeability from an RVE" begin
    ks, C₀ = 1.0e-6, TensISO{3}(3 * 30.0, 2 * 18.0)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀, :K => TensISO{3}(ks)))
    add_phase!(
        rve, :F, ConductiveCrack(1.0; conductivity = 1.0e-3),
        Dict(:C => C₀, :K => TensISO{3}(ks)); density = 0.05
    )
    K = _g(fracture_permeability(rve))
    @test K[1, 1] > ks
    @test K[3, 3] ≈ ks rtol = 1.0e-6

    rve_plain = RVE(:M)
    add_matrix!(rve_plain, Ellipsoid(1.0), Dict(:C => C₀, :K => TensISO{3}(ks)))
    add_phase!(rve_plain, :F, PennyCrack(1.0), Dict(:C => C₀, :K => TensISO{3}(ks)); density = 0.05)
    @test_throws ArgumentError fracture_permeability(rve_plain)
end
