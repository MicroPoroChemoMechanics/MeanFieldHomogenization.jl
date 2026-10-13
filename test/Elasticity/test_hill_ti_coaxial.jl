# =============================================================================
#  test_hill_ti_coaxial.jl
#
#  Validates the closed-form analytical Hill tensor for a TI matrix coaxial
#  with a spheroid (Barthélémy 2020) by:
#   1. Reducing to the isotropic case and matching the analytical isotropic
#      builder (consistency with `_hill_3d_iso`).
#   2. Cross-validating against the residue algorithm (`:residues`) for an
#      oblate spheroid (where the residue path is reliable).
#   3. Cross-validating against the DECUHR algorithm (`:decuhr`) for a
#      prolate spheroid, where the residue path can be numerically fragile.
#   4. Verifying that the analytical algorithm is the dispatcher's default
#      (`:auto`) for a TI-coaxial setup.
#   5. Verifying that the dispatcher falls back to `Residue` when the TI axis
#      is not aligned with the spheroid axis.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const ATOL_ANA_ISO = 1.0e-8
const ATOL_ANA_RES = 1.0e-12
const ATOL_ANA_DEC = 1.0e-8

@testset "Hill TI coaxial — sphere recovers isotropic builder" begin
    # Build an isotropic stiffness, then re-express it as a (degenerate) TI
    # tensor with axis e₃: the analytical TI builder should reproduce the
    # isotropic Hill tensor to numerical precision.
    n_axis = [0.0, 0.0, 1.0]
    ell = Ellipsoid(1.0, 1.0, 1.0)                     # sphere
    C_iso = TensISO{3}(30.0, 10.0)                     # 3k=30, 2μ=10
    C_ti = fromISO(C_iso, n_axis)                     # TensTI{4,T,5}
    @test C_ti isa TensND.TensTI{4, Float64, 5}

    P_iso = hill_tensor(ell, C_iso)                    # isotropic builder
    P_ana = hill_tensor(ell, C_ti)                     # auto → :analytical
    @test P_ana isa TensND.TensTI{4, Float64, 5}
    @test maximum(abs, components_canon(P_ana - P_iso)) < ATOL_ANA_ISO
end

@testset "Hill TI coaxial — oblate, cross-validation vs residue" begin
    # Sevostianov-Yilmaz-Kushch-Levin (2005) test stiffness from the
    # Barthélémy 2020 paper figure 1.
    n_axis = [0.0, 0.0, 1.0]
    C_TI = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    # All ratios < 1 keep the spheroid axis along e₃ (oblate) — coaxial
    # with the matrix axis n_axis.
    for ratio in (0.1, 0.3, 0.5, 0.7, 0.9)
        ell = Ellipsoid(1.0, 1.0, ratio)
        @test ell isa Ellipsoid{3, MeanFieldHomogenization.Oblate}
        P_ana = hill_tensor(ell, C_TI)                   # analytical
        P_res = hill_tensor(ell, C_TI; method = :residues)
        diff = maximum(abs, components_canon(P_ana - P_res))
        @test diff < ATOL_ANA_RES
    end
end

@testset "Hill TI coaxial — prolate, cross-validation vs DECUHR" begin
    n_axis = [1.0, 0.0, 0.0]                            # TI axis = e₁
    C_TI = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    for ratio in (1.5, 3.0, 8.0)
        ell = Ellipsoid(ratio, 1.0, 1.0)                # prolate, axis e₁
        P_ana = hill_tensor(ell, C_TI)                   # analytical
        P_dec = hill_tensor(ell, C_TI; method = :decuhr)
        diff = maximum(abs, components_canon(P_ana - P_dec))
        @test diff < ATOL_ANA_DEC
    end
end

@testset "Hill TI coaxial — dispatcher selects Analytical by default" begin
    n_axis = [0.0, 0.0, 1.0]
    ell = Ellipsoid(1.0, 1.0, 0.4)
    C_TI = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    algo = MeanFieldHomogenization.Core._resolve_algo(Val(:auto), ell, C_TI)
    @test algo isa MeanFieldHomogenization.Core.Analytical
end

@testset "Hill TI coaxial — dispatcher falls back when non-coaxial" begin
    # TI axis = e₁ but spheroid axis = e₃ (oblate) — not coaxial.
    n_axis = [1.0, 0.0, 0.0]
    ell = Ellipsoid(1.0, 1.0, 0.4)                       # oblate, axis e₃
    C_TI = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    # Default fallback for non-coaxial is a CUBATURE (DECUHR when its
    # extension is loaded, else NestedQuadGK) — not the residue algorithm,
    # which degenerates on an anisotropically-typed isotropic-valued
    # reference and is therefore reachable on `:residues` only.
    algo = MeanFieldHomogenization.Core._resolve_algo(Val(:auto), ell, C_TI)
    @test algo isa Union{MeanFieldHomogenization.Core.DECUHR, MeanFieldHomogenization.Core.NestedQuadGK}
    @test MeanFieldHomogenization.Core._resolve_algo(Val(:residues), ell, C_TI) isa
        MeanFieldHomogenization.Core.Residue

    # And the residue / DECUHR path should still produce a finite answer.
    P_dec = hill_tensor(ell, C_TI; method = :decuhr)
    @test all(isfinite, P_dec)
end

@testset "Hill TI coaxial — ForwardDiff compatibility" begin
    n_axis = [0.0, 0.0, 1.0]
    ell = Ellipsoid(1.0, 1.0, 0.3)
    C0 = [2.179, 0.579, 0.689, 10.345, 1.0]

    # Differentiability w.r.t. the 5 elastic constants
    f = C -> begin
        C_TI = tens_TI(C[1], C[2], C[3], C[4], C[5], n_axis)
        get_array(hill_tensor(ell, C_TI))[3, 3, 3, 3]
    end
    grad_C = ForwardDiff.gradient(f, C0)
    @test length(grad_C) == 5
    @test all(isfinite, grad_C)

    # Differentiability w.r.t. the spheroid aspect ratio
    g = c_axial -> begin
        ell2 = Ellipsoid(1.0, 1.0, c_axial)
        C_TI = tens_TI(C0..., n_axis)
        get_array(hill_tensor(ell2, C_TI))[3, 3, 3, 3]
    end
    dval_dc = ForwardDiff.derivative(g, 0.3)
    @test isfinite(dval_dc)

    # The same derivative through the nested cubature, which shares no code
    # with the closed form.
    g_cub = c_axial -> begin
        C_full = TensND.change_tens_canon(tens_TI(C0..., n_axis))
        ell2 = Ellipsoid(1.0, 1.0, c_axial)
        P = hill_tensor(ell2, C_full; method = :nestedquadgk, abstol = 1.0e-13, reltol = 1.0e-12)
        get_array(P)[3, 3, 3, 3]
    end
    @test isapprox(dval_dc, ForwardDiff.derivative(g_cub, 0.3); rtol = 1.0e-9)
end

@testset "Hill TI coaxial — complex moduli (frequency-domain viscoelasticity)" begin
    # The analytical formula must accept complex stiffness values to support
    # harmonic / viscoelastic problems (ECHOES C++ library is templated on
    # `T = double | complex<double>` for the same reason).
    n_axis = [0.0, 0.0, 1.0]
    ell = Ellipsoid(1.0, 1.0, 0.3)

    # 1. eltype propagation — complex inputs ⇒ complex output.
    δ = 0.05
    C_c = tens_TI(
        2.179 + δ * im, 0.579 + 0.0im, 0.689 + 0.0im,
        10.345 + δ * im, 1.0 + 0.5δ * im, n_axis
    )
    @test eltype(C_c) <: Complex
    P_c = hill_tensor(ell, C_c; method = :auto)
    @test eltype(P_c) <: Complex
    @test all(isfinite, get_array(P_c))

    # 2. Limit Im → 0 must reproduce the real-modulus result exactly.
    C_r = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)
    C_0 = tens_TI(
        2.179 + 0.0im, 0.579 + 0.0im, 0.689 + 0.0im,
        10.345 + 0.0im, 1.0 + 0.0im, n_axis
    )
    P_r = hill_tensor(ell, C_r)
    P_0 = hill_tensor(ell, C_0)
    @test maximum(abs.(real.(get_array(P_0)) .- get_array(P_r))) < 1.0e-14
    @test maximum(abs.(imag.(get_array(P_0)))) < 1.0e-14
end

@testset "Hill TI coaxial — explicit :residues / :decuhr still work on coaxial" begin
    # The user can override the analytical default with a numerical method.
    n_axis = [0.0, 0.0, 1.0]
    ell = Ellipsoid(1.0, 1.0, 0.5)
    C_TI = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    P_ana = hill_tensor(ell, C_TI; method = :auto)
    P_res = hill_tensor(ell, C_TI; method = :residues)
    P_dec = hill_tensor(ell, C_TI; method = :decuhr)

    @test maximum(abs, components_canon(P_ana - P_res)) < ATOL_ANA_RES
    @test maximum(abs, components_canon(P_ana - P_dec)) < ATOL_ANA_DEC
end

@testset "Hill TI coaxial — isotropic values near the sphere and at extreme ratios" begin
    # An isotropic matrix stored as TI has coincident acoustic roots, and a
    # near-spherical spheroid puts them at η ≈ 1, where the closed forms cancel:
    # the relative error reached 6 to 23 at ω = 1 ± 1e-6, and 1e-5 at ω = 0.9999.
    # Reference: the nested cubature at tight tolerances, independent of both
    # closed forms; the isotropic builder must agree as well.
    n_axis = [0.0, 0.0, 1.0]
    C_iso = TensISO{3}(3 * 30.0, 2 * 18.0)
    C_ti = fromISO(C_iso, n_axis)
    C_full = TensND.change_tens_canon(C_iso)
    canon(P) = get_array(TensND.change_tens_canon(P))
    for ω in (1.0e-6, 1.0e-3, 0.9999, 1 - 1.0e-6, 1 + 1.0e-6, 1.0001, 10.0)
        R = canon(hill_tensor(Spheroid(ω), C_full; method = :nestedquadgk, abstol = 1.0e-14, reltol = 1.0e-13))
        for C in (C_ti, C_iso)
            P = canon(hill_tensor(Spheroid(ω), C))
            @test maximum(abs, P .- R) / maximum(abs, R) < 1.0e-12
        end
    end
end

@testset "Hill TI coaxial — flat spheroid, and the scale of the moduli" begin
    # Two tests had an absolute floor: the coincidence of the roots, which
    # took any two small roots for equal ones on a flat spheroid (an error of
    # about 0.1 ω), and the isotropy of the matrix, which took moduli below
    # about 1e-4 for isotropic ones. Reference: the nested cubature of the same
    # tensor at tight tolerances.
    n_axis = [0.0, 0.0, 1.0]
    C_TI = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)
    C_full = TensND.change_tens_canon(C_TI)
    tight = (abstol = 1.0e-14, reltol = 1.0e-12)
    for ω in (1.0e-5, 1.0e-3)
        P = get_array(hill_tensor(Spheroid(ω), C_TI))
        R = get_array(hill_tensor(Spheroid(ω), C_full; method = :nestedquadgk, tight...))
        @test maximum(abs, P .- R) / maximum(abs, R) < 1.0e-10
    end
    for s in (1.0e-6, 1.0e6), ω in (0.3, 1.0, 3.0)
        P = get_array(hill_tensor(Spheroid(ω), C_TI))
        Ps = s .* get_array(hill_tensor(Spheroid(ω), tens_TI(s .* (2.179, 0.579, 0.689, 10.345, 1.0)..., n_axis)))
        @test maximum(abs, Ps .- P) / maximum(abs, P) < 1.0e-13
    end
end

@testset "Hill TI coaxial — derivatives at an isotropic point" begin
    # At coincident roots each root has an infinite derivative. Merging them
    # there lost the term in the squared root difference, and with it every
    # derivative along a direction that breaks isotropy (24 % to a factor 14).
    # Reference: ForwardDiff through the nested cubature, which has no
    # coincidence to handle.
    n_axis = [0.0, 0.0, 1.0]
    λ, μ = 18.0, 18.0
    C0 = [λ + 2μ, λ, λ, λ + 2μ, μ]
    tight = (abstol = 1.0e-13, reltol = 1.0e-12)
    for ω in (0.3, 0.999, 3.0), dir in ([0.0, 0, 1, 0, 0], [0.0, 0, 0, 1, 0], [0.0, 0, 0, 0, 1])
        # Canonical components: a prolate spheroid's cubature is returned in
        # the spheroid's own basis.
        canon(P) = get_array(TensND.change_tens_canon(P))
        f = t -> canon(hill_tensor(Spheroid(ω), tens_TI((C0 .+ t .* dir)..., n_axis)))
        g = t -> canon(
            hill_tensor(
                Spheroid(ω), TensND.change_tens_canon(tens_TI((C0 .+ t .* dir)..., n_axis));
                method = :nestedquadgk, tight...
            )
        )
        d_f = ForwardDiff.derivative(f, 0.0)
        d_g = ForwardDiff.derivative(g, 0.0)
        @test maximum(abs, d_f .- d_g) / maximum(abs, d_g) < 1.0e-8
    end
end
