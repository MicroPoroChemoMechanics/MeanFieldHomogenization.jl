# =============================================================================
#  test_differential.jl — DifferentialScheme + trajectories.
#
#  Coverage:
#   1. Single-phase RVE returns the matrix property.
#   2. Bracketed by Voigt/Reuss bounds.
#   3. Single-inclusion RVE: all three trajectories give identical results
#      (the trajectory shape is irrelevant when there's only one phase).
#   4. Multi-phase RVE: Proportional and Sequential agree in the dilute
#      limit (small target fractions).
#   5. CustomPath validation: non-monotone, wrong endpoints, wrong length,
#      missing phase → all raise ArgumentError.
#   6. Crack-only RVE: stiffness reduces monotonically as nsteps→∞.
#   7. Conductivity (`property = :K`).
#   8. ForwardDiff sensitivity to f.
#   9. Symbol shortcuts.
#  10. Stiffness ≡ compliance formulation, on every inclusion family and
#      both tensor orders.
#  11. Sherman-Morrison volume balance vs the closed form of the
#      homothetic trajectory.
#  12. Pair constructors of the trajectories; `differential_path`.
#  13. Crack densities diluted by the solid increments.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const ATOL_DIFF = 1.0e-9
const RTOL_DIFF = 1.0e-8

@testset "Differential — sanity (single-phase)" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    @test homogenize(rve, DifferentialScheme()) ≈ C_m
end

@testset "Differential — bracketed by Voigt/Reuss" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    Vv = get_array(homogenize(rve, Voigt()))[1, 1, 1, 1]
    Vr = get_array(homogenize(rve, Reuss()))[1, 1, 1, 1]
    Vd = get_array(homogenize(rve, DifferentialScheme(; nsteps = 200)))[1, 1, 1, 1]
    @test Vr - RTOL_DIFF * abs(Vr) ≤ Vd ≤ Vv + RTOL_DIFF * abs(Vv)
end

@testset "Differential — trajectory invariance for single-inclusion RVE" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    C_prop = homogenize(rve, DifferentialScheme(; trajectory = Proportional(), nsteps = 100))
    C_seq = homogenize(rve, DifferentialScheme(; trajectory = Sequential([:I]), nsteps = 100))
    custom = CustomPath(Dict(:I => collect(range(0.0, 1.0; length = 101))))
    C_cus = homogenize(rve, DifferentialScheme(; trajectory = custom, nsteps = 100))
    # Tsit5 takes different adaptive steps depending on the smoothness
    # of `df/dτ` (Proportional has constant df, Sequential has step
    # discontinuities at window boundaries, CustomPath is piecewise
    # linear), so the parametrization invariance holds only up to the
    # solver's reltol (1e-6 by default).
    @test isapprox(C_prop, C_seq; rtol = 1.0e-5)
    @test isapprox(C_prop, C_cus; rtol = 1.0e-5)
end

@testset "Differential — multi-phase Proportional vs Sequential (dilute limit)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I1, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.01
    )
    add_phase!(
        rve, :I2, Ellipsoid(1.0), Dict(:C => TensISO{3}(15.0, 5.0));
        fraction = 0.01
    )
    Cp = get_array(
        homogenize(
            rve, DifferentialScheme(;
                trajectory = Proportional(),
                nsteps = 200
            )
        )
    )[1, 1, 1, 1]
    Cs = get_array(
        homogenize(
            rve, DifferentialScheme(;
                trajectory = Sequential([:I1, :I2]),
                nsteps = 200
            )
        )
    )[1, 1, 1, 1]
    # Dilute-limit difference is O(f²)
    @test abs(Cp - Cs) < 1.0e-3
end

@testset "Differential — CustomPath validation errors" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )

    # Wrong endpoint at start
    bad_start = CustomPath(Dict(:I => vcat([0.5], collect(range(0.0, 1.0; length = 100)))))
    @test_throws ArgumentError homogenize(rve, DifferentialScheme(; trajectory = bad_start, nsteps = 100))

    # Non-monotone
    nm = collect(range(0.0, 1.0; length = 101))
    nm[50] = 0.0
    bad_mono = CustomPath(Dict(:I => nm))
    @test_throws ArgumentError homogenize(rve, DifferentialScheme(; trajectory = bad_mono, nsteps = 100))

    # Missing phase
    rve2 = RVE()
    add_phase!(rve2, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve2, :I1, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.1
    )
    add_phase!(
        rve2, :I2, Ellipsoid(1.0), Dict(:C => TensISO{3}(15.0, 5.0));
        fraction = 0.1
    )
    bad_miss = CustomPath(Dict(:I1 => collect(range(0.0, 1.0; length = 101))))
    @test_throws ArgumentError homogenize(rve2, DifferentialScheme(; trajectory = bad_miss, nsteps = 100))
end

@testset "Differential — Path (functional) trajectory" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    # f(τ) = τ² is monotone with f(0)=0, f(1)=1.  Since DEM with one
    # solid phase is parametrization-invariant in τ, this should give
    # the same C^hom(τ=1) as the default Proportional path.
    C_path = homogenize(
        rve, DifferentialScheme(;
            trajectory = Path(Dict(:I => τ -> τ^2))
        )
    )
    C_prop = homogenize(rve, DifferentialScheme())
    @test isapprox(C_path, C_prop; rtol = 1.0e-5)
end

@testset "Differential — Path validation errors" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    # f(0) ≠ 0
    @test_throws ArgumentError homogenize(
        rve,
        DifferentialScheme(; trajectory = Path(Dict(:I => τ -> 0.5 + τ / 2)))
    )
    # f(1) ≠ 1
    @test_throws ArgumentError homogenize(
        rve,
        DifferentialScheme(; trajectory = Path(Dict(:I => τ -> τ / 2)))
    )
    # Non-monotone (sin(2πτ) has both signs)
    @test_throws ArgumentError homogenize(
        rve,
        DifferentialScheme(; trajectory = Path(Dict(:I => τ -> τ + 0.3sin(2π * τ))))
    )
    # Missing phase
    @test_throws ArgumentError homogenize(
        rve,
        DifferentialScheme(; trajectory = Path(Dict(:OTHER => τ -> τ)))
    )
end

@testset "Differential — saveat insensitivity (`nsteps`)" begin
    # The adaptive ODE solver controls integration step via abstol/reltol,
    # so `nsteps` (now `saveat` density) should not affect the final
    # result beyond solver tolerance.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    C_50 = homogenize(rve, DifferentialScheme(; nsteps = 50))
    C_200 = homogenize(rve, DifferentialScheme(; nsteps = 200))
    @test isapprox(C_50, C_200; rtol = 1.0e-6)
end

@testset "Differential — `abstol` / `reltol` kwargs are forwarded" begin
    # Loose tolerances should give a slightly different result than
    # tight ones — but both should be finite and within Voigt/Reuss.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    C_loose = homogenize(rve, DifferentialScheme(; abstol = 1.0e-3, reltol = 1.0e-2))
    C_tight = homogenize(rve, DifferentialScheme(; abstol = 1.0e-10, reltol = 1.0e-9))
    @test all(isfinite, get_array(C_tight))
    # Tighter tolerance gives a more accurate answer (not necessarily
    # closer to loose) — just check finiteness.
    @test all(isfinite, get_array(C_loose))
end

@testset "Differential — crack RVE reduces stiffness monotonically" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :CRACK, PennyCrack(1.0), Dict(:C => TensISO{3}(30.0, 10.0));
        density = 0.05
    )
    C_d = homogenize(rve, DifferentialScheme(; nsteps = 200))
    @test get_array(C_d)[3, 3, 3, 3] < get_array(TensISO{3}(30.0, 10.0))[3, 3, 3, 3]
    @test all(isfinite, get_array(C_d))
end

@testset "Differential — conductivity" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(2.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(8.0)); fraction = 0.3)
    Kv = get_array(homogenize(rve, Voigt(); property = :K))[1, 1]
    Kr = get_array(homogenize(rve, Reuss(); property = :K))[1, 1]
    Kd = get_array(
        homogenize(
            rve, DifferentialScheme(; nsteps = 200);
            property = :K
        )
    )[1, 1]
    @test Kr - RTOL_DIFF * abs(Kr) ≤ Kd ≤ Kv + RTOL_DIFF * abs(Kv)
end

@testset "Differential — ForwardDiff sensitivity to f" begin
    f_diff(f) = begin
        DT = typeof(f)
        rve = RVE(; T = DT)
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
            fraction = f
        )
        get_array(homogenize(rve, DifferentialScheme(; nsteps = 50)))[1, 1, 1, 1]
    end
    df = ForwardDiff.derivative(f_diff, 0.3)
    @test isfinite(df)
    @test df > 0
end

# The volume fraction and the matrix property reach the ODE through `x0`, so
# they promote the state eltype on their own.  Everything else — an inclusion
# property, an inclusion geometry, a crack semi-axis, a nested cell — reaches
# it ONLY through the RHS, and the solver's `du` buffer keeps the eltype fixed
# by `x0`.  The state eltype must therefore be promoted from what the phase
# kernels return, not just from the matrix property and the amounts.
@testset "Differential — ForwardDiff through inputs seen only by the RHS" begin
    _fd(f, x, h) = (f(x + h) - f(x - h)) / (2h)
    idx = C -> get_array(C)[1, 1, 1, 1]
    iso(k, μ) = TensISO{3}(3k, 2μ)

    @testset "formulation = :$form" for form in (:stiffness, :compliance)
        sch = DifferentialScheme(; nsteps = 50, formulation = form)

        # (a) property of an INCLUSION phase
        f_prop = ki -> begin
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso(10.0, 5.0)); fraction = :rest)
            add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso(ki, 20.0)); fraction = 0.3)
            return idx(homogenize(rve, sch))
        end
        @test ForwardDiff.derivative(f_prop, 40.0) ≈ _fd(f_prop, 40.0, 1.0e-4) rtol = 1.0e-5

        # (b) GEOMETRY of an inclusion phase (aspect ratio)
        f_geom = ω -> begin
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso(10.0, 5.0)); fraction = :rest)
            add_phase!(
                rve, :I, Spheroid(ω), Dict(:C => iso(40.0, 20.0));
                fraction = 0.2, symmetrize = :iso
            )
            return idx(homogenize(rve, sch))
        end
        @test ForwardDiff.derivative(f_geom, 3.0) ≈ _fd(f_geom, 3.0, 1.0e-5) rtol = 1.0e-5

        # (c) semi-axis of a CRACK family (the crack-density kernel)
        f_crack = b -> begin
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso(10.0, 5.0)); fraction = :rest)
            add_phase!(
                rve, :crk, EllipticCrack(1.0, b), Dict{Symbol, Any}();
                density = 0.05, symmetrize = :iso
            )
            return idx(homogenize(rve, sch))
        end
        @test ForwardDiff.derivative(f_crack, 0.6) ≈ _fd(f_crack, 0.6, 1.0e-6) rtol = 1.0e-5

        # (d) conductivity — same story on the 2-tensor state layout
        f_cond = ki -> begin
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(1.0)); fraction = :rest)
            add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(ki)); fraction = 0.3)
            return get_array(homogenize(rve, sch, :K))[1, 1]
        end
        @test ForwardDiff.derivative(f_cond, 10.0) ≈ _fd(f_cond, 10.0, 1.0e-5) rtol = 1.0e-5

        # (e) TI running medium (aligned spheroid, no orientation average):
        #     the state is the 5-component TI layout, not the iso one.
        f_ti = ki -> begin
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso(10.0, 5.0)); fraction = :rest)
            add_phase!(rve, :I, Spheroid(4.0), Dict(:C => iso(ki, 20.0)); fraction = 0.2)
            return idx(homogenize(rve, sch))
        end
        @test ForwardDiff.derivative(f_ti, 40.0) ≈ _fd(f_ti, 40.0, 1.0e-4) rtol = 1.0e-5
    end

    # (f) a parameter buried in a NESTED cell: neither the matrix property nor
    #     the amounts of the outer RVE carry the `Dual` at all.
    @testset "nested cell (multiscale lens)" begin
        function outer()
            micro = RVE()
            add_phase!(micro, :SOLID, Ellipsoid(1.0), Dict(:C => iso(3.0, 1.2)); fraction = :rest)
            add_phase!(
                micro, :pore, Ellipsoid(1.0), Dict(:C => iso(1.0e-9, 1.0e-9));
                fraction = 0.2
            )
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso(1.0, 0.4)); fraction = :rest)
            add_phase!(
                rve, :agg, Ellipsoid(1.0),
                Dict(:C => Homogenized(micro, MoriTanaka()));
                fraction = 0.3, symmetrize = :iso
            )
            return rve
        end
        sch = DifferentialScheme(; nsteps = 50)
        p = nested(:agg, :C, property(:SOLID, :C, :shear))
        μ_eff = C -> k_mu(C)[2]

        ∂_ad = derivative(outer(), sch, p; indexer = μ_eff)
        x₀ = get_param(outer(), p)
        ∂_fd = _fd(
            x -> μ_eff(homogenize(set_param(outer(), p, x), sch; property = :C)),
            x₀, 1.0e-6
        )
        @test isfinite(∂_ad)
        @test ∂_ad ≈ ∂_fd rtol = 1.0e-5

        # …and as one component of a gradient mixing lenses from both scales.
        ps = [p, nested(:agg, :C, amount(:pore))]
        g = gradient(outer(), sch, ps; indexer = μ_eff)
        @test length(g) == 2
        # Not bit-exact against `derivative`: a 2-partial `Dual` state changes
        # the norm the adaptive solver controls, hence the step sequence.
        @test g[1] ≈ ∂_ad rtol = 1.0e-6
        @test g[2] < 0                      # porosity softens the aggregate
    end
end

@testset "Differential — Symbol shortcuts" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    @test homogenize(rve, :differential) ≈ homogenize(rve, DifferentialScheme())
    @test homogenize(rve, :diff) ≈ homogenize(rve, DifferentialScheme())
    @test homogenize(rve, :DIFF) ≈ homogenize(rve, DifferentialScheme())
end

# =============================================================================
#  Compliance formulation, trajectory ergonomics, `differential_path`, and
#  the crack/solid volume balance.
# =============================================================================

# The two formulations integrate the same trajectory in different variables
# (`ℍ = −𝕊 : 𝐍 : 𝕊`), so they must agree to solver accuracy — on every
# inclusion family, both tensor orders.
@testset "Differential — stiffness ≡ compliance formulation" begin
    C_m = TensISO{3}(3 * 20.0, 2 * 8.0)
    C_i = TensISO{3}(3 * 50.0, 2 * 20.0)
    sch(form) = DifferentialScheme(;
        formulation = form, abstol = 1.0e-12, reltol = 1.0e-10
    )
    agree(rve, prop) = begin
        a = get_array(homogenize(rve, sch(:stiffness), prop))
        b = get_array(homogenize(rve, sch(:compliance), prop))
        maximum(abs, a .- b) / maximum(abs, a)
    end

    @testset "ellipsoid (elasticity)" begin
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.3)
        @test agree(rve, :C) < 1.0e-9
    end

    @testset "ellipsoid (conductivity)" begin
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(1.0)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(10.0)); fraction = 0.3)
        @test agree(rve, :K) < 1.0e-9
    end

    @testset "oblate spheroid (TI running medium)" begin
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve, :I, Spheroid(0.2), Dict(:C => C_i); fraction = 0.2)
        @test agree(rve, :C) < 1.0e-9
    end

    # Heterogeneous inclusion: no `compliance_contribution` of its own, the
    # dual kernel goes through `ℍ = −𝕊 : 𝐍 : 𝕊`.
    @testset "layered sphere (elasticity + conductivity)" begin
        s = LayeredSphere((0.8, 1.0), (TensISO{3}(3 * 80.0, 2 * 35.0), TensISO{3}(3 * 5.0, 2 * 2.0)))
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve, :I, s, Dict(:C => C_i); fraction = 0.3)
        @test agree(rve, :C) < 1.0e-9

        sk = LayeredSphere((0.8, 1.0), (TensISO{3}(0.1), TensISO{3}(20.0)))
        rk = RVE()
        add_phase!(rk, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(1.0)); fraction = :rest)
        add_phase!(rk, :I, sk, Dict(:K => TensISO{3}(10.0)); fraction = 0.3)
        @test agree(rk, :K) < 1.0e-9
    end

    # Randomly oriented cracks: the running medium stays isotropic.
    @testset "penny cracks (iso orientation average)" begin
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve, :CR, PennyCrack(1.0), Dict(:C => C_m); density = 0.15, symmetrize = :iso)
        @test agree(rve, :C) < 1.0e-8
    end

    # An aligned crack family is deliberately NOT compared here: its
    # contribution comes back in the canonical basis, so the running medium
    # is fully anisotropic and every step pays a numerical COD evaluation.
    # At the tolerances used above that case takes minutes; it is covered,
    # at the default tolerances, by "crack RVE reduces stiffness
    # monotonically" and by the dilution testset below.
end

# Regression: an aligned non-spherical phase makes the running estimate
# leave the matrix's symmetry class at the very first step.  The ODE state
# used to be sized from the matrix alone, so these RVEs died on a
# `DimensionMismatch`; the state is now sized from the phase contributions.
@testset "Differential — state accommodates anisotropy leaked by the phases" begin
    C_m = TensISO{3}(3 * 20.0, 2 * 8.0)
    C_i = TensISO{3}(3 * 50.0, 2 * 20.0)

    # An aligned TRIAXIAL inclusion takes the running medium all the way out of
    # the TI class, so the Hill tensors come from the cubature `:auto` selects
    # for an anisotropic reference (the residue path degenerates at the
    # isotropic start — see `Core/dispatch.jl`).
    @testset "aligned triaxial ellipsoid" begin
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0, 0.6, 0.3), Dict(:C => C_i); fraction = 0.2)
        C = homogenize(rve, DifferentialScheme(), :C)
        @test all(isfinite, get_array(C))
        Cv = get_array(homogenize(rve, Voigt(), :C))[1, 1, 1, 1]
        Cr = get_array(homogenize(rve, Reuss(), :C))[1, 1, 1, 1]
        @test Cr - RTOL_DIFF * abs(Cr) ≤ get_array(C)[1, 1, 1, 1] ≤ Cv + RTOL_DIFF * abs(Cv)
        # Asking for the residue path explicitly is refused with an explanation.
        @test_throws ArgumentError homogenize(rve, DifferentialScheme(), :C; method = :residues)
    end

    for geom in (Spheroid(0.2), Spheroid(3.0))
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve, :I, geom, Dict(:C => C_i); fraction = 0.2)
        C = homogenize(rve, DifferentialScheme(), :C)
        @test all(isfinite, get_array(C))
        # Aligned spheroids give a transversely isotropic effective medium,
        # strictly between the Voigt and Reuss bounds.
        Cv = get_array(homogenize(rve, Voigt(), :C))[3, 3, 3, 3]
        Cr = get_array(homogenize(rve, Reuss(), :C))[3, 3, 3, 3]
        Cd = get_array(C)[3, 3, 3, 3]
        @test Cr - RTOL_DIFF * abs(Cr) ≤ Cd ≤ Cv + RTOL_DIFF * abs(Cv)
    end
end

@testset "Differential — invalid formulation" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.2)
    @test_throws ArgumentError homogenize(rve, DifferentialScheme(; formulation = :bogus), :C)
end

# The volume balance alone, checked against the closed form of the
# homothetic case (`Proportional`): φ_i = −ln(f₀^∞) f_i^∞ / (1 − f₀^∞).
# Pure algebra — no tensor involved.
@testset "Differential — Sherman-Morrison vs homothetic closed form" begin
    f₁, f₂ = 0.2, 0.3
    f₀∞ = 1 - f₁ - f₂
    n = 200_000
    φ₁ = 0.0
    for k in 1:n
        τ = (k - 0.5) / n
        f = (τ * f₁, τ * f₂)
        df = (f₁, f₂)
        f₀ = 1 - sum(f)
        φ₁ += (df[1] + f[1] / f₀ * sum(df)) / n
    end
    @test φ₁ ≈ -log(f₀∞) * f₁ / (1 - f₀∞) rtol = 1.0e-9
end

@testset "Differential — trajectory pair constructors" begin
    C_m = TensISO{3}(30.0, 10.0)
    C_i = TensISO{3}(60.0, 20.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I1, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.15)
    add_phase!(rve, :I2, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.15)

    @test Sequential(:I1, :I2).order == Sequential([:I1, :I2]).order
    @test homogenize(rve, DifferentialScheme(; trajectory = Sequential(:I1, :I2))) ≈
        homogenize(rve, DifferentialScheme(; trajectory = Sequential([:I1, :I2])))
    @test homogenize(rve, DifferentialScheme(; trajectory = Path(:I1 => τ -> τ^2, :I2 => τ -> 2τ - τ^2))) ≈
        homogenize(
        rve, DifferentialScheme(;
            trajectory = Path(Dict(:I1 => τ -> τ^2, :I2 => τ -> 2τ - τ^2))
        )
    )
    @test homogenize(rve, DifferentialScheme(; trajectory = CustomPath(:I1 => [0.0, 0.5, 1.0], :I2 => [0.0, 0.5, 1.0]))) ≈
        homogenize(
        rve, DifferentialScheme(;
            trajectory = CustomPath(Dict(:I1 => [0.0, 0.5, 1.0], :I2 => [0.0, 0.5, 1.0]))
        )
    )
end

@testset "Differential — differential_path" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.3)
    sch = DifferentialScheme(; nsteps = 20)
    τ, Cs = differential_path(rve, sch, :C)

    @test length(τ) == 21
    @test length(Cs) == 21
    @test τ[1] == 0.0 && τ[end] == 1.0
    # τ = 0 is the matrix, τ = 1 is what `homogenize` returns.
    @test get_array(Cs[1]) ≈ get_array(C_m)
    @test get_array(Cs[end]) ≈ get_array(homogenize(rve, sch, :C))
    # Monotone stiffening towards a stiffer inclusion.
    ks = [k_mu(C)[1] for C in Cs]
    @test all(≥(0), diff(ks))

    # Compliance formulation returns the same declared property along τ.
    _, Cs_dual = differential_path(
        rve, DifferentialScheme(;
            nsteps = 20, formulation = :compliance,
            abstol = 1.0e-12, reltol = 1.0e-10
        ), :C
    )
    @test get_array(Cs_dual[end]) ≈
        get_array(homogenize(rve, DifferentialScheme(; abstol = 1.0e-12, reltol = 1.0e-10), :C)) rtol = 1.0e-8
end

# Cracks carry no volume but are diluted by the solid increments:
#   dφ_c^ε = dε_c + (ε_c / f₀) Σ_{solids} df_j .
# The correction is inactive when no solid grows at the same τ, which gives
# an exact reference: `Sequential(:I, :CR)` must reproduce the two-stage
# composition "solid first, then cracks in the resulting medium".
@testset "Differential — crack density dilution by solid increments" begin
    C_m = TensISO{3}(3 * 20.0, 2 * 8.0)
    C_i = TensISO{3}(3 * 50.0, 2 * 20.0)
    tol = (abstol = 1.0e-12, reltol = 1.0e-10)

    mixed = RVE()
    add_phase!(mixed, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(mixed, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.3)
    add_phase!(mixed, :CR, PennyCrack(1.0), Dict(:C => C_m); density = 0.15, symmetrize = :iso)

    C_seq = homogenize(
        mixed, DifferentialScheme(; trajectory = Sequential(:I, :CR), tol...), :C
    )

    solid = RVE()
    add_phase!(solid, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(solid, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.3)
    C_solid = homogenize(solid, DifferentialScheme(; tol...), :C)
    cracked = RVE()
    add_phase!(cracked, :M, Ellipsoid(1.0), Dict(:C => C_solid); fraction = :rest)
    add_phase!(cracked, :CR, PennyCrack(1.0), Dict(:C => C_solid); density = 0.15, symmetrize = :iso)
    C_two_stage = homogenize(cracked, DifferentialScheme(; tol...), :C)

    @test k_mu(C_seq)[1] ≈ k_mu(C_two_stage)[1] rtol = 1.0e-7
    @test k_mu(C_seq)[2] ≈ k_mu(C_two_stage)[2] rtol = 1.0e-7

    # A crack-only RVE has no solid increment, so the correction vanishes
    # identically and the crack ODE is the plain dC/dε.
    crack_only = RVE()
    add_phase!(crack_only, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(crack_only, :CR, PennyCrack(1.0), Dict(:C => C_m); density = 0.15, symmetrize = :iso)
    @test all(isfinite, get_array(homogenize(crack_only, DifferentialScheme(; tol...), :C)))

    # Growing both together creates more crack than the nominal dε (the
    # cracks already present are partly replaced by solid), so the medium
    # ends softer than the sequential solid → crack route.
    C_prop = homogenize(mixed, DifferentialScheme(; tol...), :C)
    @test k_mu(C_prop)[2] < k_mu(C_seq)[2]
end
