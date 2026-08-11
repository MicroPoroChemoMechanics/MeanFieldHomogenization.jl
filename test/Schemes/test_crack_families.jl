using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# The gate of the whole per-family decomposition:
#
#     S_hom  ==  S_solid + Σ_i (4π/3) d_i 𝕊_i
#
# It is an identity of each scheme at its own fixed point, so it is satisfied by
# exactly one choice of the concentration correction. A wrong post-factor still
# reproduces the dilute limit, which is why this — and not a dilute check — is
# the test that pins the formula down.

const CS_FAM = TensISO{3}(3 * 30.0, 2 * 18.0)
const CANON_FAM = TensND.CanonicalBasis{3, Float64}()

# Components in the CANONICAL frame. Indexing (and `get_array`) return a TensND
# tensor's OWN-basis components; the scheme-corrected `𝕊_i` and the estimate
# `C_hom` are carried in bases inherited from the crack orientations, so any
# comparison between them — or against a canonically-built reference — has to
# put both in one frame first. Comparing raw arrays silently compares two
# different frames and, for a family at 45°, merely permutes the axes.
_g(t) = get_array(TensND.change_tens(t, CANON_FAM))

# Three non-coaxial penny families: the reference medium is genuinely
# anisotropic, so the correction factor cannot be hidden by an isotropic
# `C_hom` commuting with everything.
function _fractured_rve(; densities = (0.06, 0.05, 0.04), C_s = CS_FAM)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_s))
    # ZYZ Euler angles: θ tilts the crack normal away from e₃, φ is the azimuth.
    add_phase!(rve, :F1, PennyCrack(1.0; euler_angles = (0.0, 0.0)), Dict(:C => C_s); density = densities[1])
    add_phase!(rve, :F2, PennyCrack(1.0; euler_angles = (π / 4, 0.0)), Dict(:C => C_s); density = densities[2])
    add_phase!(rve, :F3, PennyCrack(1.0; euler_angles = (π / 4, π)), Dict(:C => C_s); density = densities[3])
    return rve
end

_assemble(rve, dec) = begin
    acc = dec.solid
    for (name, S_i) in dec.families
        acc = acc + delta_compliance(rve.phases[name].geometry, S_i, crack_density(rve, name))
    end
    acc
end

@testset "SelfConsistent — the decomposition identity" begin
    rve = _fractured_rve()
    sc = SelfConsistent(; abstol = 1.0e-14, reltol = 1.0e-14, maxiters = 5000)
    C_hom = homogenize(rve, sc)

    # The reference medium must really be anisotropic, otherwise the test is
    # vacuous.
    @test !(C_hom isa TensND.TensISO)

    dec = crack_family_compliances(rve, sc, C_hom)
    @test Set(keys(dec.families)) == Set((:F1, :F2, :F3))

    S_hom = inv(C_hom)
    @test _g(_assemble(rve, dec)) ≈ _g(S_hom) rtol = 1.0e-9
    @test crack_family_residual(rve, sc, C_hom) < 1.0e-9

    # The solid part of a single-matrix RVE is the matrix compliance.
    @test _g(dec.solid) ≈ _g(inv(CS_FAM)) rtol = 1.0e-9
end

@testset "SelfConsistent — the correction factor is not the identity" begin
    # Guards against the trivial mistake `𝕊_i = ℍ_i(C_hom)`: if the post-factor
    # were the identity the residual would be large, so this test proves the
    # identity test above has teeth.
    rve = _fractured_rve()
    sc = SelfConsistent(; abstol = 1.0e-14, reltol = 1.0e-14, maxiters = 5000)
    C_hom = homogenize(rve, sc)
    dec = crack_family_compliances(rve, sc, C_hom)

    naive = Dict(
        name => compliance_contribution(rve.phases[name].geometry, C_hom)
            for name in (:F1, :F2, :F3)
    )
    acc_naive = inv(CS_FAM)
    for (name, H) in naive
        acc_naive = acc_naive + delta_compliance(rve.phases[name].geometry, H, crack_density(rve, name))
    end
    res_naive = sqrt(
        sum(abs2, _g(inv(C_hom)) .- _g(acc_naive)) / sum(abs2, _g(inv(C_hom)))
    )
    @test res_naive > 1.0e-3
    # …and the corrected one is orders of magnitude better.
    @test crack_family_residual(rve, sc, C_hom) < res_naive / 1000
end

@testset "MoriTanaka — the decomposition identity" begin
    rve = _fractured_rve()
    C_hom = homogenize(rve, MoriTanaka())
    dec = crack_family_compliances(rve, MoriTanaka(), C_hom)

    # Mori-Tanaka on a matrix + cracks RVE needs no correction at all, and the
    # contributions are evaluated in the matrix.
    for name in (:F1, :F2, :F3)
        H = compliance_contribution(rve.phases[name].geometry, CS_FAM)
        @test _g(dec.families[name]) ≈ _g(H) rtol = 1.0e-12
    end
    @test _g(_assemble(rve, dec)) ≈ _g(inv(C_hom)) rtol = 1.0e-12
    @test crack_family_residual(rve, MoriTanaka(), C_hom) < 1.0e-12
end

@testset "a single family reproduces the total" begin
    # With one family the decomposition must return the whole crack part.
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => CS_FAM))
    add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => CS_FAM); density = 0.07)
    sc = SelfConsistent(; abstol = 1.0e-14, reltol = 1.0e-14, maxiters = 5000)
    C_hom = homogenize(rve, sc)
    dec = crack_family_compliances(rve, sc, C_hom)
    @test length(dec.families) == 1
    @test crack_family_residual(rve, sc, C_hom) < 1.0e-9
end

@testset "vanishing density degenerates to the matrix" begin
    rve = _fractured_rve(; densities = (1.0e-9, 1.0e-9, 1.0e-9))
    sc = SelfConsistent(; abstol = 1.0e-14, reltol = 1.0e-14, maxiters = 5000)
    C_hom = homogenize(rve, sc)
    dec = crack_family_compliances(rve, sc, C_hom)
    # In the dilute limit the correction factor tends to the identity, so the
    # per-family compliance tends to the bare ℍ evaluated in the matrix.
    H_ref = compliance_contribution(rve.phases[:F1].geometry, CS_FAM)
    @test _g(dec.families[:F1]) ≈ _g(H_ref) rtol = 1.0e-6
end

@testset "guards" begin
    C_s = CS_FAM

    # Symmetrized families have no single normal: refuse rather than mislead.
    rve_sym = RVE(:M)
    add_matrix!(rve_sym, Ellipsoid(1.0), Dict(:C => C_s))
    add_phase!(
        rve_sym, :F, PennyCrack(1.0), Dict(:C => C_s);
        density = 0.05, symmetrize = :iso
    )
    C_sym = homogenize(rve_sym, MoriTanaka())
    @test_throws ArgumentError crack_family_compliances(rve_sym, MoriTanaka(), C_sym)

    # Mori-Tanaka with a solid inclusion beside the cracks is out of scope.
    rve_mixed = RVE(:M)
    add_matrix!(rve_mixed, Ellipsoid(1.0), Dict(:C => C_s))
    add_phase!(rve_mixed, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 60.0, 2 * 30.0)); fraction = 0.2)
    add_phase!(rve_mixed, :F, PennyCrack(1.0), Dict(:C => C_s); density = 0.05)
    C_mixed = homogenize(rve_mixed, MoriTanaka())
    @test_throws ArgumentError crack_family_compliances(rve_mixed, MoriTanaka(), C_mixed)

    # …but the self-consistent decomposition handles it, since its solid part
    # is assembled from the accumulators rather than assumed to be `s_s`.
    sc = SelfConsistent(; abstol = 1.0e-14, reltol = 1.0e-14, maxiters = 5000)
    C_sc = homogenize(rve_mixed, sc)
    @test crack_family_residual(rve_mixed, sc, C_sc) < 1.0e-8

    # An unsupported scheme must say so.
    @test_throws ArgumentError crack_family_compliances(rve_mixed, Dilute(), C_mixed)
end
