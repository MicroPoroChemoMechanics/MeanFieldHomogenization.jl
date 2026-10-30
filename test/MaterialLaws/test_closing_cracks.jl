# Nonlinear poroelasticity of a rock whose cracks close under the effective
# stress: with every crack open the law is the linear poroelastic solid of the
# fractured-rock material; the moduli at an effective stress are those of the
# solid with the open densities; a loading integrates the rate equations (checked
# against a quadrature of them); the tangent is the derivative of the update,
# with respect to the gradients and to a parameter.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_CC = MeanFieldHomogenization.MaterialLaws
const SC_CC = MeanFieldHomogenization.Schemes
const C0_CC = TensISO{3}(3 * 40.0e3, 2 * 25.0e3)
# Mandel components in the canonical basis: an anisotropic result of homogenize
# may be held in a basis of its own, which `KM(t)` alone would read.
_canon(t) = MP_CC._mandel(TensND.canonical(t))

function _three_families(; d = 0.1)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0_CC); fraction = :rest)
    add_phase!(rve, :F3, PennyCrack(1.0), Dict(:C => C0_CC); density = d)
    add_phase!(rve, :F1, PennyCrack(1.0; euler_angles = (π / 2, 0.0)), Dict(:C => C0_CC); density = d)
    add_phase!(rve, :F2, PennyCrack(1.0; euler_angles = (π / 2, π / 2)), Dict(:C => C0_CC); density = d)
    return rve
end

_cc_law(; σc = 20.0) = ClosingCrackPoroelasticity(_three_families(); closure = ExponentialClosure(σc), porosity = 0.05)
_hydro(P) = LoadingPath(range(0, 1; length = 41); σ11 = t -> -P * t, σ22 = t -> -P * t, σ33 = t -> -P * t, σ23 = 0, σ13 = 0, σ12 = 0, p = 0)

@testset "the open fraction" begin
    f = ExponentialClosure(20.0)
    @test f(5.0) == 1 && f(0.0) == 1
    @test f(-20.0) ≈ exp(-1) rtol = 1.0e-14
    @test ForwardDiff.derivative(f, -10.0) ≈ exp(-0.5) / 20 rtol = 1.0e-14
    @test_throws ArgumentError ExponentialClosure(0.0)
end

@testset "every crack open: the linear fractured rock" begin
    law = _cc_law()
    rock = FracturedPoroelasticRock(_three_families(), MoriTanaka(); ω₀ = (1.0e-3, 1.0e-3, 1.0e-3), porosity_ref = 0.05)
    # A dilation: every normal traction is tensile, every crack stays open.
    g = (ε = SVector(1.0e-4, 1.2e-4, 0.8e-4, 0, 0, 0), p = 1.0)
    r = material_response(law, g, initial_state(law), 1.0)
    ref = material_response(rock, (ε = TensND.from_mandel(g.ε), p = g.p), initial_state(rock), 1.0)
    @test _canon(r.tangents.σε) ≈ _canon(ref.tangents.σε) rtol = 1.0e-10
    @test _canon(r.tangents.σp) ≈ _canon(ref.tangents.σp) rtol = 1.0e-10
    @test r.tangents.φp ≈ ref.tangents.φp rtol = 1.0e-10
    @test r.fluxes.φ ≈ ref.fluxes.φ rtol = 1.0e-10
end

@testset "the Biot parameters in Mandel components equal the tensor ones" begin
    # An anisotropic drained stiffness: cracks of an arbitrary orientation.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0_CC); fraction = :rest)
    add_phase!(rve, :F, PennyCrack(1.0; euler_angles = (0.7, 2.1)), Dict(:C => C0_CC); density = 0.2)
    C = TensND.canonical(homogenize(rve, MoriTanaka(), :C))
    ref = MeanFieldHomogenization.Poromechanics.poroelastic_parameters(C, C0_CC, 0.03)
    par = MeanFieldHomogenization.Poromechanics.poroelastic_parameters(_canon(C), _canon(C0_CC), 0.03)
    @test par.B ≈ _canon(ref.B) rtol = 1.0e-12
    @test par.inverse_modulus ≈ ref.inverse_modulus rtol = 1.0e-12
    @test par.modulus ≈ ref.modulus rtol = 1.0e-12
end

@testset "the moduli are those of the open densities" begin
    law = _cc_law()
    P = 30.0
    m = MP_CC._cc_moduli(-P * identity_mandel(), law.p)
    rve = _three_families()
    for φ in (:F1, :F2, :F3)
        rve = SC_CC.set_param(rve, SC_CC.AmountParameter(φ), 0.1 * exp(-P / 20.0))
    end
    @test m.C ≈ _canon(homogenize(rve, MoriTanaka(), :C)) rtol = 1.0e-12
    # Under a uniaxial effective compression only the family normal to it closes.
    m1 = MP_CC._cc_moduli(SVector(0, 0, -P, 0, 0, 0), law.p)
    rve1 = SC_CC.set_param(_three_families(), SC_CC.AmountParameter(:F3), 0.1 * exp(-P / 20.0))
    @test m1.C ≈ _canon(homogenize(rve1, MoriTanaka(), :C)) rtol = 1.0e-10
    @test m1.C[3, 3] > m1.C[1, 1] ≈ m1.C[2, 2]           # the closed cracks stiffen the axis
end

@testset "a hydrostatic compression integrates the rate equations" begin
    law = _cc_law()
    P = 60.0
    r = point_response(law, _hydro(P))
    # The rock stiffens as its cracks close.
    Δε = diff(r[:ε11])
    @test all(<(0), Δε) && issorted(-Δε; rev = true)
    @test r[:Σ′11][end] ≈ -P rtol = 1.0e-10
    # Reference: ε = ∫ 𝕊(Σ′) dΣ′ and φ = ∫ B : dε along Σ′ = −P′ 𝟏, by a composite
    # Gauss–Legendre quadrature of the moduli. The backward Euler steps of the law
    # are first order: halving the step halves the error, and the Richardson
    # extrapolation of two step sizes meets the quadrature.
    nodes, weights = [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664],
        [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]
    εref, φref = 0.0, 0.0
    for (x, w) in zip(nodes, weights), k in 1:8
        Pk = P * ((k - 1) + (x + 1) / 2) / 8
        mk = MP_CC._cc_moduli(-Pk * identity_mandel(), law.p)
        dε = -(mk.C \ identity_mandel())
        εref += w / 2 * P / 8 * dε[1]
        φref += w / 2 * P / 8 * (transpose(mk.B) * dε)
    end
    coarse = point_response(law, LoadingPath(range(0, 1; length = 21); σ11 = t -> -P * t, σ22 = t -> -P * t, σ33 = t -> -P * t, σ23 = 0, σ13 = 0, σ12 = 0, p = 0))
    for (name, ref) in ((:ε11, εref), (:φ, φref))
        e_fine, e_coarse = r[name][end] - ref, coarse[name][end] - ref
        @test e_fine / e_coarse ≈ 0.5 rtol = 0.05
        @test 2 * r[name][end] - coarse[name][end] ≈ ref rtol = 2.0e-3
    end
end

@testset "the pore pressure reopens the cracks" begin
    law = _cc_law()
    # The total stress held at -60 MPa, the pore pressure raised to 40 MPa: the
    # effective stress falls to -20 MPa, the cracks reopen, the drained stiffness drops.
    r = point_response(
        law, LoadingPath(
            vcat(range(0, 1; length = 21), range(1.05, 2; length = 20));
            σ11 = t -> -60.0 * min(t, 1), σ22 = t -> -60.0 * min(t, 1), σ33 = t -> -60.0 * min(t, 1),
            σ23 = 0, σ13 = 0, σ12 = 0, p = t -> 40.0 * max(t - 1, 0)
        )
    )
    @test r[:Σ′11][21] ≈ -60 rtol = 1.0e-10
    @test r[:Σ′11][end] ≈ -20 rtol = 1.0e-10
    @test r[:ε11][end] > r[:ε11][21]                       # the rock swells back
end

@testset "consistent tangent, and the derivative along a parameter" begin
    # The tangent of every block against a difference of the fluxes (the nested
    # derivative of a whole update is tested on the damage law, which shares the
    # machinery and compiles in a fraction of the time).
    @test check_material_interface(_cc_law(); verbose = false)
    # The strain under a hydrostatic compression, differentiated along σc.
    ε_end(σc) = point_response(_cc_law(; σc), _hydro(40.0))[:ε11][end]
    @test ForwardDiff.derivative(ε_end, 20.0) ≈ (ε_end(20.0 + 1.0e-3) - ε_end(20.0 - 1.0e-3)) / 2.0e-3 rtol = 1.0e-5
end

@testset "refusals" begin
    rve = _three_families()
    @test_throws ArgumentError ClosingCrackPoroelasticity(rve; closure = ExponentialClosure(20.0), families = (:M,))
    plain = RVE()
    add_phase!(plain, :M, Ellipsoid(1.0), Dict(:C => C0_CC); fraction = :rest)
    @test_throws ArgumentError ClosingCrackPoroelasticity(plain; closure = ExponentialClosure(20.0))
    iso = RVE()
    add_phase!(iso, :M, Ellipsoid(1.0), Dict(:C => C0_CC); fraction = :rest)
    add_phase!(iso, :F, PennyCrack(1.0), Dict(:C => C0_CC); density = 0.1, symmetrize = :iso)
    @test_throws ArgumentError ClosingCrackPoroelasticity(iso; closure = ExponentialClosure(20.0))
    # One closure per family.
    law = ClosingCrackPoroelasticity(rve; closure = (F1 = ExponentialClosure(10.0), F2 = ExponentialClosure(20.0), F3 = ExponentialClosure(40.0)))
    @test law.p.closures.F3.σc == 40.0
end
