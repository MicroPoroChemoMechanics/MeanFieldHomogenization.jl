using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const LSel = MeanFieldHomogenization.LayeredSpheroids

# =============================================================================
#  test_elasticity.jl — the elastic n-layer confocal spheroid, case I.
#
#  The gate is Eshelby: a SINGLE homogeneous spheroid must reproduce the
#  analytic strain concentration, and its interior series must collapse to the
#  two coefficients a uniform strain can carry. Everything else here builds on
#  that — a shell given the core's own moduli must change nothing, and the
#  over-determined system must be redundant rather than conflicting.
# =============================================================================

_C(κ, μ) = TensISO{3}(3κ, 2μ)

"Prolate spheroid of aspect ratio `ω`, unit axis semi-axis, `N` confocal layers."
function _spheroid(ω, moduli; axes = nothing, 𝒩 = 5)
    a = 1.0
    b = a / ω
    c = sqrt(a^2 - b^2)
    as = axes === nothing ? (a,) : axes
    all(x -> x > c, as) || error("test setup: a layer falls below the focal distance")
    bs = map(x -> sqrt(x^2 - c^2), as)
    return LayeredSpheroid(as, bs, moduli; Nseries = 𝒩)
end

"Eshelby interior strain for a homogeneous spheroid, in this frame."
function _eshelby(ω, C₁, C₀, εa, εt)
    # MFH sorts an Ellipsoid's semi-axes, longest onto ê₁, so the loading is
    # permuted into that frame and read back.
    A = strain_strain_loc(Ellipsoid(ω, 1.0, 1.0), C₁, C₀)
    Ep = TensND.Tens(
        TensND.SymmetricTensor{2, 3}((i, j) -> i != j ? 0.0 : (i == 1 ? εa : εt))
    )
    e = get_array(A ⊡ Ep)
    return e[1, 1], e[2, 2]
end

@testset "Elastic confocal spheroid — case I" begin
    κ₀, μ₀ = 1.0, 0.6
    κ₁, μ₁ = 5.0, 3.0
    C₀, C₁ = _C(κ₀, μ₀), _C(κ₁, μ₁)
    εa, εt = 1.0, 0.3

    @testset "a homogeneous spheroid is Eshelby (ω = $ω)" for ω in (1.5, 2.0, 5.0, 20.0)
        s = _spheroid(ω, (C₁,))
        εa_in, εt_in = LSel.spheroid_core_strain(s, C₀, εa, εt)
        ea_ref, et_ref = _eshelby(ω, C₁, C₀, εa, εt)
        @test εa_in ≈ ea_ref rtol = 1.0e-11
        @test εt_in ≈ et_ref rtol = 1.0e-11
    end

    @testset "the interior series collapses to two coefficients" begin
        # A uniform interior strain has exactly two amplitudes to carry it:
        # degree 2 of φ₀ and degree 1 of φ₃. Everything else must come out zero
        # — this is what makes the single-inclusion case an exact oracle rather
        # than a converged approximation.
        s = _spheroid(3.0, (C₁,); 𝒩 = 6)
        r = LSel.spheroid_elastic_coefficients(s, C₀, εa, εt)
        core_modes, core_amp = r.modes[1], r.amplitudes[1]
        scale = maximum(abs, core_amp)
        for (m, v) in zip(core_modes, core_amp)
            if m === (:A, 2) || m === (:C, 1)
                @test abs(v) > 1.0e-3 * scale
            else
                @test abs(v) < 1.0e-10 * scale
            end
        end
    end

    @testset "the over-determined system is redundant, not conflicting" begin
        # One row per interface is a combination of the others; the residual
        # says so. A residual that stopped being negligible would mean the four
        # conditions had become inconsistent.
        for ω in (1.5, 4.0), 𝒩 in (3, 5, 8)
            s = _spheroid(ω, (C₁,); 𝒩)
            r = LSel.spheroid_elastic_coefficients(s, C₀, εa, εt)
            @test r.residual < 1.0e-10
        end
    end

    @testset "a shell with the core's moduli changes nothing" begin
        # The sharpest test of the chaining: inserting a fictitious interface
        # must be invisible.
        ω = 1.5
        one_layer = _spheroid(ω, (C₁,))
        two_layer = _spheroid(ω, (C₁, C₁); axes = (0.9, 1.0))
        e1 = LSel.spheroid_core_strain(one_layer, C₀, εa, εt)
        e2 = LSel.spheroid_core_strain(two_layer, C₀, εa, εt)
        @test e2[1] ≈ e1[1] rtol = 1.0e-10
        @test e2[2] ≈ e1[2] rtol = 1.0e-10
    end

    @testset "a coating shifts the core strain, monotonically in its stiffness" begin
        ω = 1.5
        core = _C(5.0, 3.0)
        soft = _spheroid(ω, (core, _C(0.5, 0.3)); axes = (0.9, 1.0))
        stiff = _spheroid(ω, (core, _C(20.0, 12.0)); axes = (0.9, 1.0))
        bare = _spheroid(ω, (core,))
        ea_soft, _ = LSel.spheroid_core_strain(soft, C₀, εa, εt)
        ea_stiff, _ = LSel.spheroid_core_strain(stiff, C₀, εa, εt)
        ea_bare, _ = LSel.spheroid_core_strain(bare, C₀, εa, εt)
        # A compliant coating lets the stiff core strain more than a rigid one.
        @test ea_soft > ea_bare > ea_stiff
    end

    @testset "converged in the truncation" begin
        # A single inclusion is exact at any `𝒩`; a genuinely layered one is
        # not, and the banding argument is what bounds what truncation
        # discards. Measured rate: the residual falls by roughly a factor 3 per
        # unit of `𝒩`, so the tolerances below are the observed behavior with
        # margin, not an aspiration.
        ω = 2.0
        mods = (_C(5.0, 3.0), _C(0.5, 0.3))
        run(𝒩) = LSel.spheroid_elastic_coefficients(
            _spheroid(ω, mods; axes = (0.95, 1.0), 𝒩), C₀, εa, εt
        )
        val(𝒩) = LSel.spheroid_core_strain(
            _spheroid(ω, mods; axes = (0.95, 1.0), 𝒩), C₀, εa, εt
        )[1]
        v4, v6, v9, v12 = val(4), val(6), val(9), val(12)
        @test v6 ≈ v4 rtol = 1.0e-2
        @test v9 ≈ v6 rtol = 3.0e-4
        @test v12 ≈ v9 rtol = 1.0e-5
        # …and the residual, which measures the same thing from the inside.
        rs = [run(𝒩).residual for 𝒩 in (4, 6, 9, 12)]
        @test issorted(rs; rev = true)
        @test rs[end] < 1.0e-6
    end

    @testset "Float64 runs out before the truncation does" begin
        # The interesting limit, and it is a precision limit rather than a
        # defect: up to `𝒩 ≈ 12` the `Float64` answer tracks a 256-bit one to
        # the last bits, and past that the least-squares conditioning takes
        # over — the residual stops falling while `BigFloat` keeps going. The
        # conduction side of this package records the same thing after
        # Barthélémy & Bignonnet's appendix C; the elastic blocks are wider, so
        # it arrives sooner.
        setprecision(BigFloat, 256) do
            ω = 2.0
            f64(𝒩) = LSel.spheroid_core_strain(
                _spheroid(ω, (_C(5.0, 3.0), _C(0.5, 0.3)); axes = (0.95, 1.0), 𝒩),
                C₀, εa, εt
            )[1]
            function big𝒩(𝒩)
                c = sqrt(one(BigFloat) - (one(BigFloat) / 2)^2)
                as = (big(95) / 100, one(BigFloat))
                s = LayeredSpheroid(
                    as, map(x -> sqrt(x^2 - c^2), as),
                    (_C(big(5), big(3)), _C(big(1) / 2, big(3) / 10)); Nseries = 𝒩
                )
                return LSel.spheroid_core_strain(
                    s, _C(big(1), big(6) / 10), one(BigFloat), big(3) / 10
                )[1]
            end
            @test abs(f64(8) - big𝒩(8)) / abs(big𝒩(8)) < 1.0e-14
            @test abs(f64(12) - big𝒩(12)) / abs(big𝒩(12)) < 1.0e-13
            # Past that the two part company — pinned so the day it stops being
            # true, someone notices.
            @test abs(f64(16) - big𝒩(16)) / abs(big𝒩(16)) > 1.0e-11
        end
    end

    @testset "linearity in the remote strain" begin
        s = _spheroid(2.5, (C₁, _C(2.0, 1.0)); axes = (0.95, 1.0))
        a1 = LSel.spheroid_core_strain(s, C₀, 1.0, 0.0)
        a2 = LSel.spheroid_core_strain(s, C₀, 0.0, 1.0)
        mix = LSel.spheroid_core_strain(s, C₀, 2.0, -3.0)
        @test mix[1] ≈ 2 * a1[1] - 3 * a2[1] rtol = 1.0e-10
        @test mix[2] ≈ 2 * a1[2] - 3 * a2[2] rtol = 1.0e-10
    end

    @testset "ForwardDiff through the solve" begin
        # The whole chain — geometry, Legendre tables, quadrature, least
        # squares — has to carry a derivative.
        function _f(κ)
            s = _spheroid(2.0, (_C(κ, 3.0),))
            return LSel.spheroid_core_strain(s, C₀, εa, εt)[1]
        end
        κ = 5.0
        d = ForwardDiff.derivative(_f, κ)
        fd = (_f(κ + 1.0e-6) - _f(κ - 1.0e-6)) / 2.0e-6
        @test d ≈ fd rtol = 1.0e-5
        @test !iszero(d)

        # …and with respect to the geometry, one semi-axis at a time.
        function _g(a₁)
            c = sqrt(1 - (1 / 1.5)^2)
            s = LayeredSpheroid(
                (a₁, 1.0), (sqrt(a₁^2 - c^2), sqrt(1 - c^2)),
                (C₁, _C(2.0, 1.0)); Nseries = 5
            )
            return LSel.spheroid_core_strain(s, C₀, εa, εt)[1]
        end
        d2 = ForwardDiff.derivative(_g, 0.95)
        fd2 = (_g(0.95 + 1.0e-7) - _g(0.95 - 1.0e-7)) / 2.0e-7
        @test d2 ≈ fd2 rtol = 1.0e-4
    end

    @testset "BigFloat" begin
        ω = 2.0
        a = big(1.0)
        b = a / ω
        c = sqrt(a^2 - b^2)
        s = LayeredSpheroid((a,), (b,), (_C(big(5.0), big(3.0)),); Nseries = 5)
        ea, et = LSel.spheroid_core_strain(s, _C(big(1.0), big(0.6)), big(1.0), big(0.3))
        @test ea isa BigFloat
        ea_ref, et_ref = _eshelby(2.0, C₁, C₀, εa, εt)
        @test Float64(ea) ≈ ea_ref rtol = 1.0e-11
        @test Float64(et) ≈ et_ref rtol = 1.0e-11
    end

    @testset "what is refused, and says why" begin
        # Oblate: the confocal parameter is complex there and nothing in this
        # solver has been checked against a reference for it.
        obl = LayeredSpheroid((1.0,), (sqrt(1.0 + 0.25),), (C₁,); Nseries = 4)
        @test_throws ArgumentError LSel.spheroid_elastic_coefficients(obl, C₀, εa, εt)

        # An internal invariant, unreachable through the public API but worth
        # keeping: it is what will catch a typo when cases II and III are wired
        # in beside `:core`, `:shell` and `:matrix`.
        @test_throws ArgumentError LSel._case1_modes(4, :nonsense)

        # Imperfect interfaces need their jump terms added to the four
        # conditions; refusing beats returning the perfect-interface answer.
        c = sqrt(1 - (1 / 1.5)^2)
        imp = LayeredSpheroid(
            (1.0,), (sqrt(1 - c^2),), (C₁,);
            interfaces = (SpringInterface(10.0, 5.0),), Nseries = 4
        )
        @test_throws ArgumentError LSel.spheroid_elastic_coefficients(imp, C₀, εa, εt)
    end
end
