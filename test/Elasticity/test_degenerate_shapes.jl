# =============================================================================
#  test_degenerate_shapes.jl
#
#  The shapes at which the closed forms used to fail, each checked against a
#  reference that shares no code with them — the nested cubature at tight
#  tolerances — and, for derivatives, against central finite differences of
#  the closed form as well:
#   1. a triaxial ellipsoid whose two semi-axes nearly coincide, whose elliptic
#      integrals and their divided differences were up to 5e-5 off;
#   2. the derivative with respect to a semi-axis where it meets another one,
#      which the merged shape (a sphere, a spheroid) no longer depended on:
#      100 % to 550 % off, without a word;
#   3. the same derivative for conduction, whose anisotropic path stopped
#      with a `DomainError` on a NaN error estimate;
#   4. the transversely isotropic closed form of a needle, whose Walpole
#      combinations canceled as ω².
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

@testset "Degenerate shapes — triaxial ellipsoid close to a spheroid" begin
    canon = P -> TensND.get_array(TensND.change_tens_canon(P))
    C_iso = TensISO{3}(3 * 30.0, 2 * 18.0)
    C_full = TensND.change_tens_canon(C_iso)
    tight = (abstol = 1.0e-14, reltol = 1.0e-13)
    # Both sides of the 2 % gap where the expansion around the spheroid hands
    # over to the elliptic integrals; a flat one; and a near sphere.
    for δ in (1.0e-9, 1.0e-6, 1.0e-4, 9.0e-3, 1.1e-2),
            ell in (Ellipsoid(1.0, 1 - δ, 0.5), Ellipsoid(1.0, 0.5 + δ, 0.5), Ellipsoid(1.0, 1 - δ, 1.0e-3), Ellipsoid(1.0, 1 - δ, 1 - 2δ))
        P = canon(hill_tensor(ell, C_iso))
        R = canon(hill_tensor(ell, C_full; method = :nestedquadgk, tight...))
        @test maximum(abs, P .- R) / maximum(abs, R) < 1.0e-12
    end
    # The potentials themselves: the sum rule and the identities
    # 3 I_aa + I_ab + I_ac = 4π/a², which the expansion does not impose.
    for (a, b, c) in ((1.0, 1 - 1.0e-7, 0.5), (1.0, 0.5 + 1.0e-7, 0.5), (1.0, 1 - 1.0e-7, 1 - 2.0e-7))
        Iv, IIv = MeanFieldHomogenization.Core.newton_potential_3d(a, b, c)
        Iaa, Ibb, Icc, Ibc, Ica, Iab = IIv
        @test sum(Iv) ≈ 4π rtol = 1.0e-14
        @test 3Iaa + Iab + Ica ≈ 4π / a^2 rtol = 1.0e-13
        @test 3Ibb + Iab + Ibc ≈ 4π / b^2 rtol = 1.0e-13
        @test 3Icc + Ica + Ibc ≈ 4π / c^2 rtol = 1.0e-13
    end
end

@testset "Degenerate shapes — derivative with respect to a semi-axis where two meet" begin
    canon = P -> TensND.get_array(TensND.change_tens_canon(P))
    rel = (A, B) -> maximum(abs, A .- B) / maximum(abs, B)
    C_iso = TensISO{3}(3 * 30.0, 2 * 18.0)
    C_full = TensND.change_tens_canon(C_iso)
    tAD = (abstol = 1.0e-13, reltol = 1.0e-12)
    cases = (
        (b -> Ellipsoid(1.0, b, 0.5), 1.0),           # two semi-axes meet
        (b -> Ellipsoid(1.0, b, 0.5), 1 - 1.0e-7),    # within the old tolerance
        (c -> Ellipsoid(1.0, 1.0, c), 1.0),           # a sphere, stretched along c
        (a -> Ellipsoid(a, 1.0, 1.0), 1.0),           # a sphere, stretched along a
        (c -> Ellipsoid(1.0, 0.5, c), 0.5),
        (b -> Ellipsoid(1.0, b, 0.01), 1.0),          # a flat one
    )
    for (mk, x0) in cases
        f = x -> canon(hill_tensor(mk(x), C_iso))
        d = ForwardDiff.derivative(f, x0)
        d_cub = ForwardDiff.derivative(x -> canon(hill_tensor(mk(x), C_full; method = :nestedquadgk, tAD...)), x0)
        h = 1.0e-6
        d_fd = (f(x0 + h) .- f(x0 - h)) ./ (2h)
        @test rel(d, d_cub) < 1.0e-12
        @test rel(d, d_fd) < 1.0e-7
    end
    # A dual semi-axis is merged with another one only when their derivatives
    # coincide too: the shape keeps the class its derivatives allow.
    x = ForwardDiff.Dual(1.0, 1.0)
    @test Ellipsoid(1.0, x, 0.5) isa Ellipsoid{3, Triaxial}
    @test Ellipsoid(1.0, 1.0, x) isa Ellipsoid{3, <:Union{Oblate, Prolate}}
    @test Spheroid(ForwardDiff.Dual(0.5, 1.0)) isa Ellipsoid{3, Oblate}
    @test Ellipsoid(x, x, x) isa Ellipsoid{3, Spherical}
    # Nested duals (second derivatives): merged when no derivative of any
    # order separates the semi-axes, kept apart when only the mixed second
    # derivative does.
    D = ForwardDiff.Dual
    xx = D{:out}(D{:in}(1.0, 1.0), D{:in}(1.0, 0.0))
    xy = D{:out}(D{:in}(1.0, 1.0), D{:in}(1.0, 1.0))
    @test Ellipsoid(xx, xx, xx) isa Ellipsoid{3, Spherical}
    @test Ellipsoid(xx, xy, 0.5) isa Ellipsoid{3, Triaxial}
end

@testset "Degenerate shapes — conduction tensor of a sphere, differentiated" begin
    canon = P -> TensND.get_array(TensND.change_tens_canon(P))
    rel = (A, B) -> maximum(abs, A .- B) / maximum(abs, B)
    k = 5.0
    K = TensISO{3}(k)
    f = c -> canon(hill_tensor(Ellipsoid(1.0, 1.0, c), K))
    d = ForwardDiff.derivative(f, 1.0)
    # ∂N/∂c at the sphere, N the depolarization tensor: diag(2, 2, −4)/15.
    @test d ≈ [2 0 0; 0 2 0; 0 0 -4] ./ (15k) rtol = 1.0e-13
    # The anisotropic path, a cubature under ForwardDiff, stopped on a NaN
    # error estimate at the sphere.
    d_aniso = ForwardDiff.derivative(c -> canon(hill_tensor(Ellipsoid(1.0, 1.0, c), TensND.change_tens_canon(K))), 1.0)
    @test rel(d_aniso, d) < 1.0e-12
    d_fd = (f(1 + 1.0e-6) .- f(1 - 1.0e-6)) ./ 2.0e-6
    @test rel(d_fd, d) < 1.0e-7
    # Second order, through nested duals, against the exact value: with
    # N₃(c) = (c/2) ∫₀^∞ ds / ((s + 1)(s + c²)^{3/2}) and
    # ∫₀^∞ (s + 1)^{-1-n} ds = 1/n, ∂²N₃/∂c² = 12/35 at the sphere, and
    # N₁ = N₂ = (1 − N₃)/2.
    d2 = ForwardDiff.derivative(c -> ForwardDiff.derivative(f, c), 1.0)
    @test d2 ≈ [-6 0 0; 0 -6 0; 0 0 12] ./ (35k) rtol = 1.0e-13
end

@testset "Degenerate shapes — transversely isotropic needle" begin
    canon = P -> TensND.get_array(TensND.change_tens_canon(P))
    rel = (A, B) -> maximum(abs, A .- B) / maximum(abs, B)
    n_axis = [0.0, 0.0, 1.0]
    C_TI = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)
    for ω in (1.0e2, 1.0e3)
        P = canon(hill_tensor(Spheroid(ω), C_TI))
        R = canon(hill_tensor(Spheroid(ω), TensND.change_tens_canon(C_TI); method = :nestedquadgk, abstol = 1.0e-14, reltol = 1.0e-13))
        @test rel(P, R) < 1.0e-11
    end
    # Isotropic values in a TI type against the isotropic builder, exact for
    # any aspect ratio.
    C_iso = TensISO{3}(3 * 30.0, 2 * 18.0)
    C_ti = fromISO(C_iso, n_axis)
    for ω in (1.0e4, 1.0e6)
        @test rel(canon(hill_tensor(Spheroid(ω), C_ti)), canon(hill_tensor(Spheroid(ω), C_iso))) < 1.0e-13
    end
end
