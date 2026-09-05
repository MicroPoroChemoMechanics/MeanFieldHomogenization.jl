# =============================================================================
#  test_self_consistent.jl — SelfConsistent + AsymmetricSelfConsistent.
#
#  Coverage:
#   1. Single-phase RVE returns the matrix property exactly.
#   2. Iso 2-phase composite : SC bracketed by Voigt/Reuss.
#   3. SC fixed-point self-consistency : `step(C_eff) ≈ C_eff` to abstol.
#   4. ASC ≡ SC when the matrix is the soft phase (stiffness-form path).
#   5. ASC handles inclusion-soft RVE through the compliance-form path.
#   6. Conductivity (`property = :K`) — same recipes via gradient_gradient_loc.
#   7. ForwardDiff sensitivity through the volume fraction.
#   8. NewtonDefault (built-in, dependency-free) agrees with AndersonDefault.
#   9. Symbol shortcuts.
#  10. ASC compliance branch: `⟨C:A⟩` of an ANISOTROPIC phase under an
#      isotropic orientation average — the average does not commute with the
#      tensor product, so the result must come back isotropic and
#      major-symmetric.
#  11. ASC ≡ SC only when every phase shares one Hill tensor (`Σ f_r A_r = 𝟙`);
#      equal-but-oblate shapes agree, unequal shapes differ by percents.
#      Item 4's agreement is a consequence of this, not of the branch taken.
#
#  NonlinearSolve.jl-backed algorithms (NewtonRaphson, TrustRegion, …)
#  and AutoNonlinear are covered separately in
#  `test_self_consistent_nls.jl`.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const ATOL_SC = 1.0e-9
const RTOL_SC = 1.0e-8

@testset "SelfConsistent — sanity (single-phase)" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    @test homogenize(rve, SelfConsistent()) ≈ C_m
    @test homogenize(rve, AsymmetricSelfConsistent()) ≈ C_m
end

@testset "SelfConsistent — bracketed by Voigt/Reuss" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )

    Vv = get_array(homogenize(rve, Voigt()))[1, 1, 1, 1]
    Vr = get_array(homogenize(rve, Reuss()))[1, 1, 1, 1]
    Vsc = get_array(homogenize(rve, SelfConsistent()))[1, 1, 1, 1]
    @test Vr - RTOL_SC * abs(Vr) ≤ Vsc ≤ Vv + RTOL_SC * abs(Vv)
end

@testset "SelfConsistent — fixed-point self-consistency" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    C_eff = homogenize(rve, SelfConsistent(; abstol = 1.0e-12, maxiters = 200))
    # Verify : one more SC step on C_eff itself should return ≈ C_eff
    step_once = MeanFieldHomogenization.Schemes._sc_step(rve, C_eff, :C)
    @test maximum(abs.(get_array(step_once) .- get_array(C_eff))) < 1.0e-9
end

@testset "AsymmetricSelfConsistent — matches SC when matrix is soft" begin
    # ASC ≡ SC here because BOTH phases are spheres, i.e. share one Hill
    # tensor — not because the stiffness form was selected. See the testset
    # "ASC ≡ SC iff every phase shares one Hill tensor" below.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    @test homogenize(rve, AsymmetricSelfConsistent()) ≈
        homogenize(rve, SelfConsistent())
end

@testset "AsymmetricSelfConsistent — uses compliance form when matrix is stiff" begin
    # Soft inclusion in stiff matrix → ASC switches to compliance-form
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 1.0));
        fraction = 0.3
    )
    C_asc = homogenize(rve, AsymmetricSelfConsistent())
    # Bracketed by Voigt/Reuss
    Vv = get_array(homogenize(rve, Voigt()))[1, 1, 1, 1]
    Vr = get_array(homogenize(rve, Reuss()))[1, 1, 1, 1]
    Va = get_array(C_asc)[1, 1, 1, 1]
    @test Vr - RTOL_SC * abs(Vr) ≤ Va ≤ Vv + RTOL_SC * abs(Vv)
end

@testset "SelfConsistent — conductivity" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(2.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(8.0));
        fraction = 0.3
    )
    Kv = get_array(homogenize(rve, Voigt(); property = :K))[1, 1]
    Kr = get_array(homogenize(rve, Reuss(); property = :K))[1, 1]
    Ksc = get_array(homogenize(rve, SelfConsistent(); property = :K))[1, 1]
    @test Kr - RTOL_SC * abs(Kr) ≤ Ksc ≤ Kv + RTOL_SC * abs(Kv)
end

@testset "SelfConsistent — ForwardDiff sensitivity to f" begin
    f_sc(f) = begin
        DT = typeof(f)
        rve = RVE(; T = DT)
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
            fraction = f
        )
        get_array(homogenize(rve, SelfConsistent()))[1, 1, 1, 1]
    end
    df = ForwardDiff.derivative(f_sc, 0.3)
    @test isfinite(df)
    @test df > 0
end

@testset "SelfConsistent — NewtonDefault works out of the box" begin
    # Since v0.7.0 ForwardDiff is a strong dependency and the built-in
    # `NewtonDefault` SC solver ships with the package — quadratic
    # convergence on iso / TI / ortho canonical components.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    C_anderson = homogenize(rve, SelfConsistent(; algorithm = AndersonDefault()))
    C_newton = homogenize(rve, SelfConsistent(; algorithm = NewtonDefault()))
    @test isapprox(C_anderson, C_newton; atol = 1.0e-6, rtol = 1.0e-6)
end

@testset "SelfConsistent — NewtonDefault ForwardDiff sensitivity (non-matrix phase)" begin
    # Regression test: a `Float64` seed can meet `Dual` phase properties
    # while `step(x0)` promotes to `Dual` internally, whenever the
    # differentiated parameter lives on a phase OTHER than the one `x0`
    # is built from (an inclusion modulus, or any volume fraction). This
    # used to crash `_solve_sc(::NewtonDefault, …)` (`Tref` was derived
    # from `eltype(p0)` alone); it must now match a central finite
    # difference for every such parameter.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    idxC = C -> get_array(C)[1, 1, 1, 1]

    d_incl_modulus = derivative(
        rve, SelfConsistent(; algorithm = NewtonDefault()),
        property(:I, :C, :bulk); indexer = idxC
    )
    d_fraction = derivative(
        rve, SelfConsistent(; algorithm = NewtonDefault()),
        amount(:I); indexer = idxC
    )

    function f_modulus(K_I)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
        add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(K_I, 20.0)); fraction = 0.3)
        return idxC(homogenize(r, SelfConsistent(; algorithm = NewtonDefault())))
    end
    function f_fraction(f)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
        add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = f)
        return idxC(homogenize(r, SelfConsistent(; algorithm = NewtonDefault())))
    end
    h = 1.0e-5
    d_fd_modulus = (f_modulus(60.0 + h) - f_modulus(60.0 - h)) / (2h)
    d_fd_fraction = (f_fraction(0.3 + h) - f_fraction(0.3 - h)) / (2h)

    @test isapprox(d_incl_modulus, d_fd_modulus; rtol = 1.0e-5)
    @test isapprox(d_fraction, d_fd_fraction; rtol = 1.0e-5)
end

@testset "SelfConsistent / ASC — Symbol shortcuts" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    @test homogenize(rve, :sc) ≈ homogenize(rve, SelfConsistent())
    @test homogenize(rve, :SC) ≈ homogenize(rve, SelfConsistent())
    @test homogenize(rve, :self_consistent) ≈ homogenize(rve, SelfConsistent())
    @test homogenize(rve, :asc) ≈ homogenize(rve, AsymmetricSelfConsistent())
end

# =============================================================================
#  The orientation average does NOT commute with the tensor product, so
#  `⟨C:A⟩` has to be built from the RAW localization tensor and averaged
#  afterwards. The ASC compliance branch used to assemble it as
#  `C_i ⊡ ⟨A⟩` — the raw (anisotropic) phase stiffness times the
#  already-averaged concentration — which is a different tensor unless `C_i`
#  is isotropic. An iso-symmetrized phase then came back transversely
#  isotropic instead of isotropic, and not even major-symmetric.
#
#  The conjunction is what makes it rare: an ANISOTROPIC phase property AND
#  `symmetrize` on AND the compliance branch. The `_asc_use_stiffness`
#  assertion is load-bearing — without it the RVE could drift to the
#  stiffness branch and stop covering the bug silently.
# =============================================================================
@testset "ASC — iso orientation average of an anisotropic phase" begin
    C_m = TensISO{3}(3 * 1.0, 2 * 0.4)            # soft isotropic matrix
    # A transversely isotropic aggregate from engineering constants, aligned
    # with e₃ in its own frame. Built this way it is stored as the 5-component
    # (major-symmetric) flavor, so any asymmetry downstream was created by the
    # scheme rather than inherited from the input.
    C_i = TensND.tens_TI_eng(2.0, 1.0, 0.25, 0.2, 0.45, (0.0, 0.0, 1.0))
    @test C_i isa TensND.TensTI{4, Float64, 5}

    function build(sym)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(
            rve, :agg, Ellipsoid(1.0), Dict(:C => C_i);
            fraction = 0.3, symmetrize = sym
        )
        return rve
    end

    # The branch this test exists to cover.
    @test MeanFieldHomogenization.Schemes._asc_use_stiffness(build(:iso), :C, :M) == false

    tol = (abstol = 1.0e-13, maxiters = 600)
    C_asc = homogenize(build(:iso), AsymmetricSelfConsistent(; tol...), :C)
    C_sc = homogenize(build(:iso), SelfConsistent(; tol...), :C)

    # Averaging over all orientations of the aggregate leaves nothing
    # anisotropic behind.
    @test C_asc isa TensND.TensISO
    # …and the result is an admissible stiffness.
    KM = collect(TensND.KM(C_asc))
    @test maximum(abs, KM - KM') / maximum(abs, KM) < 1.0e-12
    # Both formulations share one fixed point; only the dynamics differ.
    @test k_mu(C_asc)[1] ≈ k_mu(C_sc)[1] rtol = 1.0e-8
    @test k_mu(C_asc)[2] ≈ k_mu(C_sc)[2] rtol = 1.0e-8

    # Control: with nothing to average, the aggregate's own anisotropy must
    # survive — the fix must not isotropize unconditionally.
    C_raw = homogenize(build(:none), AsymmetricSelfConsistent(; tol...), :C)
    @test !(C_raw isa TensND.TensISO)
    @test C_raw isa TensND.TensTI
end

# =============================================================================
#  ASC and SC are two SCHEMES, not two solvers for one answer. Their fixed
#  points coincide exactly when `Σ_r f_r A_r = 𝟙`, which "every phase shares
#  one Hill tensor" guarantees (derivation atop src/Schemes/self_consistent.jl).
#  Sphericity is only the usual such case — what matters is that the shapes be
#  EQUAL, not that they be spheres. The code used to claim the fixed points
#  always coincide.
# =============================================================================
@testset "ASC ≡ SC iff every phase shares one Hill tensor" begin
    C_m, C_i = TensISO{3}(3 * 1.0, 2 * 0.4), TensISO{3}(3 * 8.0, 2 * 3.0)
    tol = (abstol = 1.0e-13, maxiters = 800)

    function build(mshape, ishape; sym = :none)
        r = RVE()
        add_phase!(r, :M, mshape, Dict(:C => C_m); fraction = :rest)
        add_phase!(r, :I, ishape, Dict(:C => C_i); fraction = 0.3, symmetrize = sym)
        return r
    end

    # `Σ_r f_r A_r`, matrix INCLUDED — the quantity the equivalence turns on.
    function sum_fA(rve, C)
        tot = zero(C)
        for name in keys(rve.phases)
            # One accessor for every phase: the closure has already resolved
            # the complement, so no phase needs a case of its own.
            f = volume_fraction(rve, name)
            tot += f * MeanFieldHomogenization.Schemes._phase_dilute_concentration(rve, name, :C, C)
        end
        return tot
    end
    dev_I(T) = maximum(abs, collect(TensND.KM(T)) - Matrix{Float64}(I, 6, 6))

    @testset "equal shapes — spheres" begin
        rve = build(Ellipsoid(1.0), Ellipsoid(1.0))
        C_sc = homogenize(rve, SelfConsistent(; tol...), :C)
        @test dev_I(sum_fA(rve, C_sc)) < 1.0e-7
        @test homogenize(rve, AsymmetricSelfConsistent(; tol...), :C) ≈ C_sc rtol = 1.0e-6
    end

    @testset "equal shapes — both strongly oblate, so NOT about sphericity" begin
        for ω in (0.2, 0.05)
            rve = build(Spheroid(ω), Spheroid(ω))
            C_sc = homogenize(rve, SelfConsistent(; tol...), :C)
            @test dev_I(sum_fA(rve, C_sc)) < 1.0e-7
            @test homogenize(rve, AsymmetricSelfConsistent(; tol...), :C) ≈ C_sc rtol = 1.0e-6
        end
    end

    @testset "unequal shapes disagree, and the gap tracks ‖Σ f A − 𝟙‖" begin
        gaps, devs = Float64[], Float64[]
        for ω in (0.2, 0.05)
            rve = build(Ellipsoid(1.0), Spheroid(ω); sym = :iso)
            C_sc = homogenize(rve, SelfConsistent(; tol...), :C)
            C_asc = homogenize(rve, AsymmetricSelfConsistent(; tol...), :C)
            push!(devs, dev_I(sum_fA(rve, C_sc)))
            push!(gaps, abs(k_mu(C_asc)[1] - k_mu(C_sc)[1]) / k_mu(C_sc)[1])
        end
        # Percent-level: genuinely different effective media, not solver noise.
        @test all(>(1.0e-2), gaps)
        @test all(>(1.0e-2), devs)
        # Flatter inclusion → identity further from 𝟙 → wider gap.
        @test devs[2] > devs[1]
        @test gaps[2] > gaps[1]
    end
end

@testset "SelfConsistent — select_best tracks the best iterate, on every solver" begin
    # `select_best` returns the smallest-residual iterate seen rather than the
    # last one. It is a Picard notion, but `NewtonDefault` implements it too and
    # nothing exercised that pair: the branch was reachable and untested, and a
    # caller may legitimately write it.
    C_s = iso_stiffness(30.0, 20.0)
    C_w = iso_stiffness(1.0e-6, 1.0e-6)
    rve = RVE()
    add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => C_s); fraction = :rest)
    add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => C_w); fraction = 0.3)

    ref = homogenize(rve, SelfConsistent(; abstol = 0.0, reltol = 1.0e-14, maxiters = 5_000), :C)
    for algo in (AndersonDefault(), NewtonDefault())
        # Converged: best and last iterate agree with the reference.
        C = homogenize(
            rve, SelfConsistent(; algorithm = algo, select_best = true, maxiters = 200), :C
        )
        @test k_mu(C)[1] ≈ k_mu(ref)[1] rtol = 1.0e-6

        # Cut short on purpose: the loop cannot converge, so the returned value
        # is the best iterate rather than whatever the last step happened to be.
        C_short = homogenize(
            rve, SelfConsistent(;
                algorithm = algo, abstol = 0.0, reltol = 1.0e-14,
                maxiters = 3, select_best = true
            ), :C
        )
        @test all(isfinite, get_array(C_short))
        @test k_mu(C_short)[1] > 0
    end
end

@testset "SelfConsistent — the isometric parametrization and its fallback" begin
    # `abstol` bounds the Frobenius norm of the tensor for every solver, and
    # what converts one into the other is the weight vector below. The identity
    # holds only if the class basis is orthogonal, which the code checks rather
    # than assumes — so both halves of that contract are pinned here.
    S = MeanFieldHomogenization.Schemes
    frob(t) = sqrt(sum(abs2, get_array(t)))

    C_iso = iso_stiffness(30.0, 20.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_iso); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.2), Dict(:C => iso_stiffness(120.0, 80.0)); fraction = 0.2)
    C_ti = homogenize(rve, MoriTanaka(), :C)

    for T in (C_iso, C_ti)
        w, isometric = S._sc_param_weights(T)
        @test isometric
        L = length(TensND.get_data(T))
        @test length(w) == L
        @test all(>(0), w)
        # ‖w ⊙ p‖₂ == ‖rebuild(p)‖_F, on a vector aligned with no single axis.
        q = Float64[1 / (i + 1) for i in 1:L]
        @test sqrt(sum(abs2, q .* w)) ≈ frob(S._rebuild_from_data(T, q)) rtol = 1.0e-12
    end

    # The deviatoric basis tensor is not a unit tensor: an isotropic stiffness
    # weights (α, β) by (1, √5). This is the whole reason the plain Euclidean
    # norm of the components understated the tensor norm.
    w_iso, _ = S._sc_param_weights(C_iso)
    @test w_iso[1] ≈ 1.0 rtol = 1.0e-12
    @test w_iso[2] ≈ sqrt(5) rtol = 1.0e-12

    # The fallback: a coordinate vector the class cannot be rebuilt from is a
    # normal answer (`nothing`), not an exception — that is what lets the
    # caller degrade to measuring the tensor directly instead of failing.
    @test S._sc_basis_norm(C_ti, [1.0, 2.0, 3.0]) === nothing
    @test S._sc_basis_norm(C_ti, collect(1.0:length(TensND.get_data(C_ti)))) isa Real
end
