# =============================================================================
#  test_neural_inclusion.jl — the neural-surrogate inclusion contract.
#
#  Coverage:
#   1. The committed models load, and their provenance is complete enough for a
#      tolerance to be derived from it.
#   2. Exactness that must hold *regardless* of how well the network is trained:
#      `ℂ₁ = ℂ₀ ⟹ 𝔸 = 𝕀`, homogeneity `ℙ(λℂ₀) = ℙ(ℂ₀)/λ`, the symmetry class and
#      the major symmetry of the returned tensor, frame indifference.
#   3. Accuracy against the analytic Hill tensor, at a tolerance **derived from
#      the model's own recorded validation error** — never a literal, so a
#      retraining cannot silently loosen the threshold.
#   4. Every scheme, elasticity and conduction: `Ellipsoid` versus
#      `NeuralHillInclusion`.
#   5. Orientation averaging costs the surrogate nothing.
#   6. ForwardDiff through a *morphology* parameter — the capability the
#      finite-element inclusions refuse.
#   7. The affine factorization is materially exact: it reproduces ℙ at Poisson
#      ratios no label was ever generated at.
#   8. `save_surrogate` / `load_surrogate` round trip.
#   9. The domain guard.
#  10. Gate B plumbing (`NeuralLocalizationInclusion`) and its guard rails.
#  11. Constructor guard rails.
#
#  Tests 2, 8, 9, 10 and 11 use *untrained* networks where the values are
#  irrelevant: `glorot_mlp` needs no training dependency, and what is under test
#  there is the plumbing and the algebra, not the fit.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
using ForwardDiff

const NI = MeanFieldHomogenization.NeuralInclusions

# ── Helpers ──────────────────────────────────────────────────────────────────

const NN_C_M = iso_stiffness(30.0, 10.0)
const NN_C_I = iso_stiffness(60.0, 20.0)
const NN_K_M = TensISO{3}(2.0)
const NN_K_I = TensISO{3}(7.0)

_nn_c1111(C) = get_array(C)[1, 1, 1, 1]
_nn_k11(K) = get_array(K)[1, 1]

"Two-phase RVE whose inclusion geometry is `geom`."
function _nn_rve(geom, prop_m, prop_i, key; kwargs...)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(key => prop_m))
    add_phase!(rve, :I, geom, Dict(key => prop_i); kwargs...)
    return rve
end

"A spheroid of aspect ratio `ω` = distinct/equal, in the sorted convention."
_nn_spheroid(ω; kw...) =
    ω ≥ 1 ? Ellipsoid(ω, 1.0, 1.0; kw...) : Ellipsoid(1.0, 1.0, ω; kw...)

"An untrained surrogate of the given specification over the given box."
function _nn_untrained(spec, box; seed = 42)
    net = NI.glorot_mlp(
        Random.Xoshiro(seed), NI.network_widths(NI.TrainingOptions(), box, spec)
    )
    nz = NI.noutputs(spec)
    scaling = (;
        x_shift = zeros(length(box)), x_scale = ones(length(box)),
        y_kind = fill(:identity, nz), y_shift = zeros(nz), y_scale = ones(nz),
    )
    return NI.assemble_surrogate(net, spec, box, scaling, NI.Provenance())
end

const _NN_SPHEROID_BOX4 =
    NI.SampleBox([:log_aspect, :nu0], [-log(20), 0.0], [log(20), 0.49])
const _NN_SPHEROID_BOX2 = NI.SampleBox([:log_aspect], [-log(20)], [log(20)])

# The committed models. Loading them is itself the first assertion: the pilot is
# worthless if the artifacts are missing or stale.
const NN_ELASTIC = NI.load_surrogate(NI.model_path("spheroid_hill_iso_elastic"))
const NN_CONDUCTION = NI.load_surrogate(NI.model_path("spheroid_hill_iso_conduction"))
const NN_TRIAXIAL = NI.load_surrogate(NI.model_path("triaxial_hill_iso_elastic"))
const NN_AFFINE = NI.load_surrogate(NI.model_path("spheroid_hill_iso_affine"))

# A scheme composes several surrogate evaluations, so the tolerance it deserves
# is looser than one tensor's worth of error. This factor is the only piece of
# slack in the file, and it multiplies a *measured* error rather than replacing
# it.
const NN_SCHEME_SLACK = 20.0

@testset "NeuralInclusions — the committed models" begin
    for (name, s) in (
            ("elastic", NN_ELASTIC), ("conduction", NN_CONDUCTION),
            ("triaxial", NN_TRIAXIAL), ("affine", NN_AFFINE),
        )
        @test s isa NI.NeuralSurrogate
        @test NI.n_in(s.net) == length(s.features)
        @test NI.n_out(s.net) == NI.noutputs(s.output)
        # A model with no recorded error would silently disable every tolerance
        # derived from it, so the provenance must be populated.
        @test isfinite(NI.worst_error(s.provenance))
        @test NI.worst_error(s.provenance) < 0.05
        @test s.provenance.nvalidation > 0
        @test !isempty(s.provenance.teacher)
        @test all(s.domain_lo .< s.domain_hi)
    end
    @test NI.tensor_order(NN_ELASTIC) == 4
    @test NI.tensor_order(NN_CONDUCTION) == 2
    @test NI.hill_class(NN_TRIAXIAL) isa NI.HillOrtho
    @test NI.nterms(NN_AFFINE.output) == 2
    @test :nu0 ∉ NN_AFFINE.features        # the whole point of the affine spec
    @test :nu0 ∈ NN_ELASTIC.features
end

@testset "NeuralInclusions — exactness independent of the fit" begin
    # Deliberately an *untrained* network: these properties are structural, so
    # they must hold with random weights just as with trained ones.
    s = _nn_untrained(NI.DimensionlessHill(NI.HillTI()), _NN_SPHEROID_BOX4)
    incl = NeuralHillInclusion((1.0, 1.0, 0.4); elastic = s, guard = :error)

    @testset "zero contrast gives the identity to machine precision" begin
        A = strain_strain_loc(incl, NN_C_M, NN_C_M)
        @test get_array(A) ≈ get_array(TensISO{3}(1.0, 1.0)) atol = 1.0e-14
    end

    @testset "homogeneity in the reference moduli" begin
        P = hill_tensor(incl, NN_C_M)
        for λ in (0.25, 3.0, 17.0)
            @test λ .* get_array(hill_tensor(incl, λ * NN_C_M)) ≈ get_array(P) atol = 1.0e-14
        end
    end

    @testset "symmetry class and major symmetry are structural" begin
        P = hill_tensor(incl, NN_C_M)
        @test P isa TensND.TensTI{4}
        @test length(TensND.get_data(P)) == 5      # the major-symmetric form
        A = get_array(P)
        @test A ≈ permutedims(A, (3, 4, 1, 2)) atol = 1.0e-14
    end

    @testset "frame indifference: rotating the inclusion rotates ℙ" begin
        ang = (0.3, 0.7, 0.2)
        rot = NeuralHillInclusion(
            (1.0, 1.0, 0.4); elastic = s, euler_angles = ang, guard = :error
        )
        # Same five Walpole components, a different axis: the network never sees
        # the orientation, so the components cannot depend on it.
        @test collect(TensND.get_data(hill_tensor(rot, NN_C_M))) ≈
            collect(TensND.get_data(hill_tensor(incl, NN_C_M))) atol = 1.0e-14
        @test TensND.axis(hill_tensor(rot, NN_C_M)) ≠ TensND.axis(hill_tensor(incl, NN_C_M))
    end

    @testset "the sorted convention is applied, whatever the input order" begin
        a = NeuralHillInclusion((1.0, 0.4, 1.0); elastic = s, guard = :error)
        b = NeuralHillInclusion((0.4, 1.0, 1.0); elastic = s, guard = :error)
        for x in (a, b)
            @test collect(x.semi_axes) == [1.0, 1.0, 0.4]
        end
    end
end

@testset "NeuralInclusions — accuracy against the analytic Hill tensor" begin
    @testset "spheroid, elasticity" begin
        tol = NN_SCHEME_SLACK * NI.worst_error(NN_ELASTIC.provenance)
        for ω in (0.06, 0.2, 0.5, 0.9, 1.1, 2.0, 5.0, 18.0), nu in (0.05, 0.2, 0.45)
            C₀ = iso_stiffness_E_nu(1.0, nu)
            incl = NeuralHillInclusion(
                _nn_spheroid(ω).semi_axes; elastic = NN_ELASTIC, guard = :error
            )
            P_nn = get_array(hill_tensor(incl, C₀))
            P_ex = get_array(hill_tensor(_nn_spheroid(ω), C₀))
            @test maximum(abs, P_nn .- P_ex) ≤ tol * maximum(abs, P_ex)
        end
    end

    @testset "spheroid, conduction" begin
        tol = NN_SCHEME_SLACK * NI.worst_error(NN_CONDUCTION.provenance)
        for ω in (0.06, 0.3, 0.8, 1.5, 4.0, 15.0)
            incl = NeuralHillInclusion(
                _nn_spheroid(ω).semi_axes; transport = NN_CONDUCTION, guard = :error
            )
            P_nn = get_array(hill_tensor(incl, NN_K_M))
            P_ex = get_array(hill_tensor(_nn_spheroid(ω), NN_K_M))
            @test maximum(abs, P_nn .- P_ex) ≤ tol * maximum(abs, P_ex)
        end
    end

    @testset "triaxial ellipsoid" begin
        tol = NN_SCHEME_SLACK * NI.worst_error(NN_TRIAXIAL.provenance)
        for (r2, r3) in ((0.8, 0.5), (0.5, 0.1), (0.9, 0.85), (0.2, 0.08)),
                nu in (0.1, 0.4)
            C₀ = iso_stiffness_E_nu(1.0, nu)
            ell = Ellipsoid(1.0, r2, r3)
            incl = NeuralHillInclusion(
                ell.semi_axes; elastic = NN_TRIAXIAL, guard = :error
            )
            P_nn = get_array(hill_tensor(incl, C₀))
            P_ex = get_array(hill_tensor(ell, C₀))
            @test maximum(abs, P_nn .- P_ex) ≤ tol * maximum(abs, P_ex)
        end
    end
end

@testset "NeuralInclusions — the affine factorization is materially exact" begin
    # Labels were generated at two Poisson ratios only. If the decode is right,
    # the accuracy at every *other* ν₀ is the same — the material dependence is
    # algebra, not interpolation.
    tol = NN_SCHEME_SLACK * NI.worst_error(NN_AFFINE.provenance)
    errs = Float64[]
    # No ω = 1: that is a sphere, whose analytic Hill tensor is a `TensISO`, and a
    # :ti surrogate rightly refuses to describe it.
    for ω in (0.1, 0.4, 0.95, 3.0, 12.0), nu in (0.02, 0.15, 0.3, 0.42, 0.48)
        C₀ = iso_stiffness_E_nu(1.0, nu)
        incl = NeuralHillInclusion(
            _nn_spheroid(ω).semi_axes; elastic = NN_AFFINE, guard = :error
        )
        P_ex = get_array(hill_tensor(_nn_spheroid(ω), C₀))
        e = maximum(abs, get_array(hill_tensor(incl, C₀)) .- P_ex) / maximum(abs, P_ex)
        push!(errs, e)
        @test e ≤ tol
    end
    # No ν₀-dependent drift: the spread across Poisson ratios at fixed shape is
    # what would betray a fitted material dependence.
    @test maximum(errs) / max(minimum(errs), 1.0e-16) < 1.0e4
end

@testset "NeuralInclusions — every scheme, versus the native Ellipsoid" begin
    ω = 0.4
    ell = _nn_spheroid(ω; euler_angles = (0.3, 0.7, 0.0))
    incl = NeuralHillInclusion(
        ell.semi_axes; basis = MeanFieldHomogenization.inclusion_basis(ell),
        elastic = NN_ELASTIC, transport = NN_CONDUCTION, guard = :error
    )

    @testset "elasticity, one-shot schemes" begin
        # These evaluate the inclusion in the reference medium supplied, which is
        # isotropic, so the surrogate's domain is respected as is.
        tol = NN_SCHEME_SLACK * NI.worst_error(NN_ELASTIC.provenance)
        for (name, sch) in (
                "Dilute" => Dilute(), "DiluteDual" => DiluteDual(),
                "MoriTanaka" => MoriTanaka(), "Maxwell" => Maxwell(),
                "PCW" => PonteCastanedaWillis(),
            )
            ref = _nn_c1111(homogenize(_nn_rve(ell, NN_C_M, NN_C_I, :C; fraction = 0.25), sch, :C))
            got = _nn_c1111(homogenize(_nn_rve(incl, NN_C_M, NN_C_I, :C; fraction = 0.25), sch, :C))
            @test abs(got - ref) ≤ tol * abs(ref)
        end
    end

    @testset "elasticity, iterative schemes under IsoSymmetrize" begin
        # `SelfConsistent`, `AsymmetricSelfConsistent` and `DifferentialScheme`
        # re-evaluate the inclusion in their own running estimate, which for a
        # single tilted spheroid is transversely isotropic — outside the domain of
        # a surrogate trained for an isotropic reference. `IsoSymmetrize` makes the
        # scheme hand the kernel a pre-projected isotropic medium at every
        # iteration; both columns get it, so the comparison is like for like.
        tol = NN_SCHEME_SLACK * NI.worst_error(NN_ELASTIC.provenance)
        opts = (; fraction = 0.25, symmetrize = IsoSymmetrize())
        for (name, sch) in (
                "SelfConsistent" => SelfConsistent(),
                "ASC" => AsymmetricSelfConsistent(),
                "Differential" => DifferentialScheme(),
            )
            ref = _nn_c1111(homogenize(_nn_rve(ell, NN_C_M, NN_C_I, :C; opts...), sch, :C))
            got = _nn_c1111(homogenize(_nn_rve(incl, NN_C_M, NN_C_I, :C; opts...), sch, :C))
            @test abs(got - ref) ≤ tol * abs(ref)
        end
    end

    @testset "an anisotropic running reference is refused, not extrapolated" begin
        # The silent-failure case this guard exists for.
        @test_throws ArgumentError homogenize(
            _nn_rve(incl, NN_C_M, NN_C_I, :C; fraction = 0.25), SelfConsistent(), :C
        )
    end

    @testset "conduction" begin
        tol = NN_SCHEME_SLACK * NI.worst_error(NN_CONDUCTION.provenance)
        for (name, sch) in (
                "Dilute" => Dilute(), "DiluteDual" => DiluteDual(),
                "MoriTanaka" => MoriTanaka(), "Maxwell" => Maxwell(),
            )
            ref = _nn_k11(homogenize(_nn_rve(ell, NN_K_M, NN_K_I, :K; fraction = 0.25), sch, :K))
            got = _nn_k11(homogenize(_nn_rve(incl, NN_K_M, NN_K_I, :K; fraction = 0.25), sch, :K))
            @test abs(got - ref) ≤ tol * abs(ref)
        end
    end

    @testset "the surrogate reports gate A" begin
        @test check_inclusion_interface(incl; verbose = false)
        @test check_inclusion_interface(incl; physics = :conduction, verbose = false)
    end
end

@testset "NeuralInclusions — orientation averaging is free" begin
    tol = NN_SCHEME_SLACK * NI.worst_error(NN_ELASTIC.provenance)
    ω = 0.35
    function oriented(build, nbins, f)
        rve = RVE(:M)
        add_matrix!(rve, Ellipsoid(1.0), Dict(:C => NN_C_M))
        for (i, bin) in enumerate(polar_orientation_bins(nbins))
            add_phase!(
                rve, Symbol(:I, i), build(bin.θ), Dict(:C => NN_C_I);
                fraction = f * bin.weight, symmetrize = TISymmetrize((0.0, 0.0, 1.0))
            )
        end
        return rve
    end
    ell_at(θ) = _nn_spheroid(ω; euler_angles = (θ, 0.0, 0.0))
    nn_at(θ) = let e = ell_at(θ)
        NeuralHillInclusion(
            e.semi_axes; basis = MeanFieldHomogenization.inclusion_basis(e),
            elastic = NN_ELASTIC, guard = :error
        )
    end
    C_ref = homogenize(oriented(ell_at, 8, 0.2), MoriTanaka(), :C)
    C_nn = homogenize(oriented(nn_at, 8, 0.2), MoriTanaka(), :C)
    @test typeof(C_nn).name.name === typeof(C_ref).name.name
    @test maximum(abs, get_array(C_nn) .- get_array(C_ref)) ≤
        tol * maximum(abs, get_array(C_ref))
end

@testset "NeuralInclusions — ForwardDiff reaches the morphology" begin
    # The headline capability, and the one a finite-element inclusion refuses:
    # `FiniteElements._no_fe_sensitivity` errors on exactly this request.
    idx = C -> get_array(C)[1, 1, 1, 1]
    ω = 0.5
    build(a3) = NeuralHillInclusion(
        (1.0, 1.0, a3); elastic = NN_ELASTIC, guard = :none
    )
    function rve_of(a3)
        rve = RVE(:M)
        add_matrix!(rve, Ellipsoid(1.0), Dict(:C => NN_C_M))
        add_phase!(rve, :I, build(a3), Dict(:C => NN_C_I); fraction = 0.2)
        return rve
    end

    @testset "the sensitivity API finds the geometric field" begin
        d = derivative(
            rve_of(ω), Dilute(), geometry(:I, :semi_axes, 3); indexer = idx
        )
        h = 1.0e-6
        fd = (
            idx(homogenize(rve_of(ω + h), Dilute())) -
                idx(homogenize(rve_of(ω - h), Dilute()))
        ) / (2h)
        @test d ≈ fd rtol = 1.0e-4
        # Not accidentally zero — which is how a broken sensitivity presents.
        @test abs(d) > 1.0e-3
    end

    @testset "the derivative tracks the analytic one" begin
        # The surrogate's *derivative* is a harder ask than its value, so the
        # tolerance is looser; what matters is that it is right to a few percent
        # rather than merely finite.
        an = ForwardDiff.derivative(
            a3 -> begin
                rve = RVE(:M)
                add_matrix!(rve, Ellipsoid(1.0), Dict(:C => NN_C_M))
                add_phase!(rve, :I, Ellipsoid(1.0, 1.0, a3), Dict(:C => NN_C_I); fraction = 0.2)
                idx(homogenize(rve, Dilute()))
            end, ω
        )
        nn = derivative(rve_of(ω), Dilute(), geometry(:I, :semi_axes, 3); indexer = idx)
        @test nn ≈ an rtol = 0.05
    end

    @testset "second derivatives exist — the activation is smooth" begin
        f = a3 -> idx(homogenize(rve_of(a3), Dilute()))
        d2 = ForwardDiff.derivative(a -> ForwardDiff.derivative(f, a), ω)
        @test isfinite(d2)
    end
end

@testset "NeuralInclusions — serialization round trip" begin
    s = _nn_untrained(NI.DimensionlessHill(NI.HillTI()), _NN_SPHEROID_BOX4; seed = 7)
    path = joinpath(mktempdir(), "surrogate.json")
    @test NI.save_surrogate(path, s) == path
    s2 = NI.load_surrogate(path)

    @test s2.features == s.features
    @test s2.y_kind == s.y_kind
    @test s2.domain_lo == s.domain_lo
    @test NI.layer_widths(s2.net) == NI.layer_widths(s.net)
    @test NI.layer_activations(s2.net) == NI.layer_activations(s.net)
    # Bit-for-bit, not approximately: a lossy round trip would make a committed
    # model behave differently from the one that was validated.
    for x in ([-1.0, 0.1], [0.0, 0.3], [2.5, 0.48])
        @test NI.predict_components(s2, x) == NI.predict_components(s, x)
    end

    @testset "a corrupt or foreign file is refused" begin
        bad = joinpath(mktempdir(), "bad.json")
        write(bad, "{\"not\": \"a surrogate\"}")
        @test_throws ArgumentError NI.load_surrogate(bad)
        @test_throws ArgumentError NI.load_surrogate(joinpath(mktempdir(), "absent.json"))
        @test_throws ArgumentError NI.model_path("no_such_model")
    end
end

@testset "NeuralInclusions — the domain guard" begin
    s = _nn_untrained(NI.DimensionlessHill(NI.HillTI()), _NN_SPHEROID_BOX4)
    inside = NeuralHillInclusion((1.0, 1.0, 0.5); elastic = s, guard = :error)
    @test hill_tensor(inside, NN_C_M) isa TensND.AbstractTens

    outside = NeuralHillInclusion((1.0, 1.0, 1.0e-4); elastic = s, guard = :error)
    @test_throws ArgumentError hill_tensor(outside, NN_C_M)

    warned = NeuralHillInclusion((1.0, 1.0, 1.0e-4); elastic = s, guard = :warn)
    @test_logs (:warn,) hill_tensor(warned, NN_C_M)

    quiet = NeuralHillInclusion((1.0, 1.0, 1.0e-4); elastic = s, guard = :none)
    @test hill_tensor(quiet, NN_C_M) isa TensND.AbstractTens

    @test_throws ArgumentError NeuralHillInclusion(
        (1.0, 1.0, 0.5); elastic = s, guard = :maybe
    )
end

@testset "NeuralInclusions — gate B plumbing" begin
    # Untrained networks: what is under test is that the four methods dispatch,
    # that the heterogeneous branch of the contributions is taken, and that the
    # bounds appear only when the internal fractions are supplied. The numbers
    # are meaningless by construction.
    s4 = _nn_untrained(NI.DimensionlessHill(NI.HillTI()), _NN_SPHEROID_BOX4)
    s4b = _nn_untrained(NI.DimensionlessHill(NI.HillTI()), _NN_SPHEROID_BOX4; seed = 99)
    s2 = _nn_untrained(NI.DimensionlessHill(NI.HillTI2()), _NN_SPHEROID_BOX2)
    s2b = _nn_untrained(NI.DimensionlessHill(NI.HillTI2()), _NN_SPHEROID_BOX2; seed = 99)

    incl = NeuralLocalizationInclusion(
        (1.0, 1.0, 0.4); strain = s4, stress = s4b,
        gradient = s2, flux = s2b, guard = :none
    )

    @test !is_homogeneous_inclusion(incl)
    @test MeanFieldHomogenization.shape_trait(incl) === NI.NeuralShape
    @test MeanFieldHomogenization.dimension(incl) == 3

    @testset "the four localization methods answer" begin
        @test strain_strain_loc(incl, NN_C_I, NN_C_M) isa TensND.AbstractTens{4, 3}
        @test stress_strain_loc(incl, NN_C_I, NN_C_M) isa TensND.AbstractTens{4, 3}
        @test gradient_gradient_loc(incl, NN_K_I, NN_K_M) isa TensND.AbstractTens{2, 3}
        @test flux_gradient_loc(incl, NN_K_I, NN_K_M) isa TensND.AbstractTens{2, 3}
    end

    @testset "the contributions take the exact heterogeneous branch" begin
        A = strain_strain_loc(incl, NN_C_I, NN_C_M)
        B = stress_strain_loc(incl, NN_C_I, NN_C_M)
        N = stiffness_contribution(incl, NN_C_I, NN_C_M)
        @test get_array(N) ≈ get_array(B) - get_array(NN_C_M ⊡ A) atol = 1.0e-12
        # And *not* the homogeneous formula, which the stress-side surrogate
        # deliberately disagrees with.
        @test !isapprox(get_array(N), get_array((NN_C_I - NN_C_M) ⊡ A); atol = 1.0e-8)
    end

    @testset "gate B is reported, with no missing stress side" begin
        @test check_inclusion_interface(incl; verbose = false)
        @test check_inclusion_interface(incl; physics = :conduction, verbose = false)
    end

    @testset "bounds follow the internal fractions" begin
        @test !MeanFieldHomogenization.Schemes.has_layer_average(incl)
        bounded = NeuralLocalizationInclusion(
            (1.0, 1.0, 0.4); strain = s4, stress = s4b, guard = :none,
            fractions = (0.3, 0.7), properties = (NN_C_I, NN_C_M)
        )
        @test MeanFieldHomogenization.Schemes.has_layer_average(bounded)
        v = MeanFieldHomogenization.Schemes._layer_voigt(bounded, NN_C_M)
        @test get_array(v) ≈ get_array(0.3 * NN_C_I + 0.7 * NN_C_M) atol = 1.0e-12
        r = MeanFieldHomogenization.Schemes._layer_reuss(bounded, NN_C_M)
        @test get_array(r) ≈ get_array(0.3 * inv(NN_C_I) + 0.7 * inv(NN_C_M)) atol = 1.0e-12
    end

    @testset "half a pair is refused" begin
        # The silent-failure case the contract warns about: with only the strain
        # side, `Dilute` and `MoriTanaka` would stay right while the
        # self-consistent schemes drifted.
        @test_throws ArgumentError NeuralLocalizationInclusion(
            (1.0, 1.0, 0.4); strain = s4
        )
        @test_throws ArgumentError NeuralLocalizationInclusion(
            (1.0, 1.0, 0.4); stress = s4b
        )
        @test_throws ArgumentError NeuralLocalizationInclusion((1.0, 1.0, 0.4))
        @test_throws ArgumentError NeuralLocalizationInclusion(
            (1.0, 1.0, 0.4); strain = s4, stress = s4b, fractions = (0.5, 0.5)
        )
        @test_throws ArgumentError NeuralLocalizationInclusion(
            (1.0, 1.0, 0.4); strain = s4, stress = s4b,
            fractions = (0.5, 0.4), properties = (NN_C_I, NN_C_M)
        )
    end
end

@testset "NeuralInclusions — the localization classes carry six components" begin
    # A localization tensor is not major-symmetric, so it needs the 6-component
    # transversely isotropic form where a Hill tensor needs 5 — and not the
    # 8-component one, whose `ℓ₇`, `ℓ₈` are antisymmetric in an index pair and
    # vanish for anything mapping symmetric strains to symmetric stresses.
    axis = (0.0, 0.0, 1.0)
    ell = Ellipsoid(1.0, 1.0, 0.4)
    P = hill_tensor(ell, NN_C_M)
    A = strain_strain_loc(ell, NN_C_I, NN_C_M)

    _major_defect(t) = let M = Array(TensND.components_canon(t))
        maximum(abs, M .- permutedims(M, (3, 4, 1, 2))) / maximum(abs, M)
    end

    @testset "ℙ is major-symmetric, 𝔸_εε is not" begin
        @test _major_defect(P) < 1.0e-14
        @test _major_defect(A) > 1.0e-3          # ~11 % for this contrast
    end

    @testset "component counts" begin
        @test NI.ncomponents(NI.StrainLocTI()) == 6
        @test NI.ncomponents(NI.StressLocTI()) == 6
        @test NI.tensor_order(NI.StrainLocTI()) == 4
        @test length(NI.components(NI.StrainLocTI(), A, axis)) == 6
        # The 5-component form genuinely loses something.
        @test_throws ArgumentError NI.components(NI.HillTI(), A, axis)
    end

    @testset "build ∘ components is the identity on a TI localization tensor" begin
        c = NI.components(NI.StrainLocTI(), A, axis)
        rebuilt = NI.build(NI.StrainLocTI(), collect(c), axis)
        @test Array(TensND.components_canon(rebuilt)) ≈
            Array(TensND.components_canon(A)) atol = 1.0e-12
        @test length(TensND.get_data(rebuilt)) == 6
    end

    @testset "the antisymmetric couplings vanish" begin
        for t in (P, A, stress_strain_loc(ell, NN_C_I, NN_C_M))
            l8 = TensND.get_ℓ8(MeanFieldHomogenization.transverse_isotropify(t, axis))
            @test abs(l8[7]) < 1.0e-14
            @test abs(l8[8]) < 1.0e-14
        end
    end

    @testset "the two differ only by their dimension" begin
        # 𝔸_εε is of degree 0 in the moduli, 𝔸_σε of degree +1.
        @test NI.dimensionless_scale(NI.StrainLocTI(), NN_C_M) == 1
        @test NI.dimensionless_scale(NI.StressLocTI(), NN_C_M) ≈
            inv(TensND.get_data(NN_C_M)[2])
    end

    @testset "no affine factorization for a localization tensor" begin
        # `ℙ = d·𝕌ᴬ + 𝕎ᴬ/μ₀` belongs to the Hill tensor, not to 𝔸.
        @test_throws ArgumentError NI.material_coeffs(NI.StrainLocTI(), NN_C_M)
    end
end

@testset "NeuralInclusions — morphology parameters as features" begin
    box = NI.SampleBox(
        [:eccentricity, :core_fraction, :log_mu_ratio_2],
        [0.0, 0.2, log(0.1)], [0.8, 0.7, log(2.0)]
    )
    sA = _nn_untrained(NI.DimensionlessHill(NI.StrainLocTI()), box)
    sB = _nn_untrained(NI.DimensionlessHill(NI.StressLocTI()), box; seed = 5)
    C2 = iso_stiffness(4.0, 1.5)

    incl = NeuralLocalizationInclusion(
        (1.0, 1.0, 1.0); strain = sA, stress = sB,
        shape_params = (; eccentricity = 0.4, core_fraction = 0.5),
        fractions = (0.5, 0.5), properties = (NN_C_I, C2), guard = :none
    )

    @testset "the features are read off the morphology and the contrast" begin
        x = NI.raw_features(incl, sA, NN_C_M)
        @test x[1] ≈ 0.4
        @test x[2] ≈ 0.5
        @test x[3] ≈ log(k_mu(C2)[2] / k_mu(NN_C_M)[2])
    end

    @testset "an unnamed parameter is refused" begin
        bad = NI.SampleBox([:not_a_parameter], [0.0], [1.0])
        s = _nn_untrained(NI.DimensionlessHill(NI.StrainLocTI()), bad)
        i2 = NeuralLocalizationInclusion(
            (1.0, 1.0, 1.0); strain = s, stress = s, guard = :none
        )
        @test_throws ArgumentError strain_strain_loc(i2, NN_C_I, NN_C_M)
    end

    @testset "a contrast feature needs the constituents" begin
        i3 = NeuralLocalizationInclusion(
            (1.0, 1.0, 1.0); strain = sA, stress = sB,
            shape_params = (; eccentricity = 0.4, core_fraction = 0.5), guard = :none
        )
        @test_throws ArgumentError strain_strain_loc(i3, NN_C_I, NN_C_M)
    end

    @testset "the pair answers, and the contributions take the exact branch" begin
        A = strain_strain_loc(incl, NN_C_I, NN_C_M)
        B = stress_strain_loc(incl, NN_C_I, NN_C_M)
        @test length(TensND.get_data(A)) == 6
        @test length(TensND.get_data(B)) == 6
        N = stiffness_contribution(incl, NN_C_I, NN_C_M)
        @test get_array(N) ≈ get_array(B - NN_C_M ⊡ A) atol = 1.0e-12
        @test check_inclusion_interface(incl; verbose = false)
    end

    @testset "ForwardDiff reaches a morphology parameter" begin
        # The capability the finite-element inclusion refuses. `shape_params`
        # carries its own type parameter precisely so that perturbing it does not
        # have to perturb the semi-axes too.
        idx = C -> get_array(C)[1, 1, 1, 1]
        function rve_of(α)
            i = NeuralLocalizationInclusion(
                (1.0, 1.0, 1.0); strain = sA, stress = sB,
                shape_params = (; eccentricity = α, core_fraction = 0.5),
                fractions = (0.5, 0.5), properties = (NN_C_I, C2), guard = :none
            )
            r = RVE(:M)
            add_matrix!(r, Ellipsoid(1.0), Dict(:C => NN_C_M))
            add_phase!(r, :I, i, Dict(:C => NN_C_M); fraction = 0.3)
            return r
        end
        d = derivative(
            rve_of(0.4), MoriTanaka(), geometry(:I, :shape_params, 1); indexer = idx
        )
        h = 1.0e-6
        fd = (
            idx(homogenize(rve_of(0.4 + h), MoriTanaka())) -
                idx(homogenize(rve_of(0.4 - h), MoriTanaka()))
        ) / (2h)
        @test d ≈ fd rtol = 1.0e-5
        @test abs(d) > 1.0e-6                    # not accidentally zero
    end

    @testset "shape parameters must be numbers" begin
        @test_throws ArgumentError NeuralLocalizationInclusion(
            (1.0, 1.0, 1.0); strain = sA, stress = sB,
            shape_params = (; eccentricity = "a lot"), guard = :none
        )
    end
end

@testset "NeuralInclusions — constructor guard rails" begin
    s4 = _nn_untrained(NI.DimensionlessHill(NI.HillTI()), _NN_SPHEROID_BOX4)
    s2 = _nn_untrained(NI.DimensionlessHill(NI.HillTI2()), _NN_SPHEROID_BOX2)

    @test_throws ArgumentError NeuralHillInclusion((1.0, 1.0, 0.5))
    # An order-2 surrogate in the elastic slot, and the reverse.
    @test_throws ArgumentError NeuralHillInclusion((1.0, 1.0, 0.5); elastic = s2)
    @test_throws ArgumentError NeuralHillInclusion((1.0, 1.0, 0.5); transport = s4)

    @testset "the class must describe the geometry" begin
        # A :ti surrogate on a triaxial ellipsoid, and on a sphere: in both cases
        # the analytic teacher returns a different class, so the components would
        # mean something else.
        @test_throws ArgumentError NeuralHillInclusion((1.0, 0.7, 0.4); elastic = s4)
        @test_throws ArgumentError NeuralHillInclusion((1.0, 1.0, 1.0); elastic = s4)
        # And an :ortho surrogate on a spheroid.
        so = _nn_untrained(
            NI.DimensionlessHill(NI.HillOrtho()),
            NI.SampleBox(
                [:log_r2, :log_r32, :nu0],
                [-log(20), -log(20), 0.0], [-log(1.05), -log(1.05), 0.49]
            )
        )
        @test_throws ArgumentError NeuralHillInclusion((1.0, 1.0, 0.4); elastic = so)
        @test NeuralHillInclusion((1.0, 0.7, 0.4); elastic = so) isa NeuralHillInclusion
    end

    @testset "asking for the physics a surrogate does not serve" begin
        incl = NeuralHillInclusion((1.0, 1.0, 0.5); elastic = s4, guard = :none)
        @test_throws ArgumentError hill_tensor(incl, NN_K_M)
    end

    @testset "a surrogate whose declarations disagree with its network" begin
        net = NI.glorot_mlp(Random.Xoshiro(3), [2, 8, 5])
        @test_throws DimensionMismatch NI.NeuralSurrogate(;
            net, features = [:log_aspect], output = NI.DimensionlessHill(NI.HillTI())
        )
        @test_throws DimensionMismatch NI.NeuralSurrogate(;
            net, features = [:log_aspect, :nu0],
            output = NI.DimensionlessHill(NI.HillOrtho())
        )
    end

    @testset "an unknown feature name is refused, not ignored" begin
        net = NI.glorot_mlp(Random.Xoshiro(3), [1, 8, 5])
        s = NI.NeuralSurrogate(;
            net, features = [:not_a_feature],
            output = NI.DimensionlessHill(NI.HillTI()),
            domain_lo = [-1.0], domain_hi = [1.0]
        )
        incl = NeuralHillInclusion((1.0, 1.0, 0.5); elastic = s, guard = :none)
        @test_throws ArgumentError hill_tensor(incl, NN_C_M)
    end

    @testset "anisotropic reference media are refused, not silently projected" begin
        incl = NeuralHillInclusion((1.0, 1.0, 0.5); elastic = NN_ELASTIC, guard = :none)
        C_ti = hoenig_stiffness(30.0, 1.2, 0.2, 0.25, 1.1, (0.0, 0.0, 1.0))
        @test_throws ArgumentError hill_tensor(incl, C_ti)
    end
end

@testset "NeuralInclusions — sampling and datasets" begin
    box = NI.SampleBox([:log_aspect], [-1.0], [1.0])

    @testset "Halton sampling covers the box and is deterministic" begin
        X = NI.sample_box(box, 200)
        @test size(X) == (1, 200)
        @test all(-1.0 .≤ X .≤ 1.0)
        @test X == NI.sample_box(box, 200)
        # A held-out set drawn as the continuation of the sequence is disjoint.
        V = NI.sample_box(box, 50; offset = 200)
        @test isempty(intersect(vec(X), vec(V)))
    end

    @testset "a log-scaled feature is sampled in the logarithm" begin
        lb = NI.SampleBox([:log_aspect], [1.0e-3], [1.0e3]; scale = :log)
        X = vec(NI.sample_box(lb, 400))
        @test all(1.0e-3 .≤ X .≤ 1.0e3)
        # Half the points below 1 if the sampling is really logarithmic.
        @test 0.4 < count(<(1.0), X) / length(X) < 0.6
    end

    @testset "grid sampling hits the corners" begin
        g = NI.grid_box(NI.SampleBox([:a, :b], [0.0, 0.0], [1.0, 1.0]), 4)
        @test size(g) == (2, 16)
        @test [0.0, 0.0] ∈ collect(eachcol(g))
        @test [1.0, 1.0] ∈ collect(eachcol(g))
    end

    @testset "box guard rails" begin
        @test_throws DimensionMismatch NI.SampleBox([:a, :b], [0.0], [1.0])
        @test_throws ArgumentError NI.SampleBox([:a], [1.0], [0.0])
        @test_throws ArgumentError NI.SampleBox([:a], [0.0], [1.0]; scale = :sqrt)
        @test_throws ArgumentError NI.SampleBox([:a], [-1.0], [1.0]; scale = :log)
        @test_throws ArgumentError NI.feature_index(box, :absent)
    end

    @testset "the specification and the box must agree about ν₀" begin
        geometry(x) = _nn_spheroid(exp(x[1]))
        response(g, P₀) = hill_tensor(g, P₀)
        # A :dimensionless order-4 spec learns ν₀, so it must be in the box.
        @test_throws ArgumentError NI.generate_dataset(
            geometry, response, NI.DimensionlessHill(NI.HillTI()), box, 4
        )
        # An :affine spec reproduces it exactly, so it must not be.
        @test_throws ArgumentError NI.generate_dataset(
            geometry, response, NI.AffineHill(NI.HillTI()), _NN_SPHEROID_BOX4, 4
        )
    end

    @testset "a class mismatch is caught when the labels are read" begin
        # A spheroid teacher against an :ortho spec: the projection residual is
        # large, which is exactly what `components` checks.
        response(g, P₀) = hill_tensor(g, P₀)
        @test_throws ArgumentError NI.generate_dataset(
            x -> Ellipsoid(1.0, 0.6, 0.3), response,
            NI.DimensionlessHill(NI.HillTI()), box, 4
        )
    end

    @testset "fit_scaling standardizes and picks the transforms" begin
        geometry(x) = _nn_spheroid(exp(x[1]))
        response(g, P₀) = hill_tensor(g, P₀)
        train, val = NI.generate_dataset(
            geometry, response, NI.DimensionlessHill(NI.HillTI()),
            _NN_SPHEROID_BOX4, 120; nvalidation = 30
        )
        sc = NI.fit_scaling(train)
        @test length(sc.x_shift) == 2
        @test length(sc.y_kind) == 5
        @test all(k -> k in NI.TRANSFORMS, sc.y_kind)
        @test all(>(0), sc.x_scale)
        @test all(>(0), sc.y_scale)
        @test NI.nsamples(val) == 30
    end
end

@testset "NeuralInclusions — the MLP itself" begin
    @testset "widths, activations and the forward pass" begin
        m = NI.glorot_mlp(Random.Xoshiro(1), [3, 5, 2]; hidden = :softplus)
        @test NI.layer_widths(m) == [3, 5, 2]
        @test NI.layer_activations(m) == [:softplus, :identity]
        @test NI.nparams(m) == 3 * 5 + 5 + 5 * 2 + 2
        y = m([0.1, -0.2, 0.3])
        @test length(y) == 2
        @test all(isfinite, y)
    end

    @testset "generic in the element type — Dual in, Dual out" begin
        m = NI.glorot_mlp(Random.Xoshiro(1), [1, 6, 1])
        g = ForwardDiff.derivative(x -> only(m([x])), 0.4)
        @test isfinite(g)
        # Smooth activations: the second derivative exists too.
        h = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> only(m([y])), x), 0.4)
        @test isfinite(h)
    end

    @testset "softplus is stable at both extremes" begin
        @test NI.softplus(-800.0) ≈ 0.0 atol = 1.0e-300
        @test NI.softplus(800.0) ≈ 800.0
        @test NI.softplus(0.0) ≈ log(2)
    end

    @testset "guard rails" begin
        @test_throws DimensionMismatch NI.NNDense(zeros(3, 2), zeros(2), tanh)
        @test_throws DimensionMismatch NI.MLP(
            NI.NNDense(zeros(3, 2), zeros(3), tanh),
            NI.NNDense(zeros(2, 5), zeros(2), identity),
        )
        @test_throws ArgumentError NI.activation(:relu)
        @test_throws ArgumentError NI.glorot_mlp(Random.Xoshiro(1), [3])
        @test_throws ArgumentError NI.hill_class(:nope)
        @test_throws ArgumentError NI.output_spec(:nope, :ti)
    end
end

@testset "NeuralInclusions — training needs the extension" begin
    # Whether this errors or trains depends on whether Lux is loaded; both are
    # correct, and the point is that the *fallback message* is informative rather
    # than a `MethodError`.
    if !NN_HAS_LUX
        @test_throws ErrorException NI.train_surrogate()
    else
        @test hasmethod(
            NI.train_surrogate,
            Tuple{NI.AbstractOutputSpec, NI.SampleBox, NI.Dataset, NI.Dataset},
        )
    end
end

if NN_HAS_LUX
    @testset "NeuralInclusions — a short fit actually converges" begin
        # Cheap end-to-end proof of the training path: one input, two outputs,
        # a small net and a few hundred epochs. Not an accuracy claim — the
        # committed models carry that — but it does establish that the optimizer,
        # the Lux→MLP hand-off and the provenance all work.
        geometry(x) = _nn_spheroid(exp(x[1]))
        response(g, K₀) = hill_tensor(g, K₀)
        spec = NI.DimensionlessHill(NI.HillTI2())
        box = NI.SampleBox([:log_aspect], [-log(5)], [log(5)])
        train, val = NI.generate_dataset(geometry, response, spec, box, 400; nvalidation = 100)
        s = NI.train_surrogate(
            spec, box, train, val;
            options = NI.TrainingOptions(;
                hidden = [24, 24], epochs = 400, batchsize = 64,
                patience = 100, verbose = false
            ),
            teacher_name = "test", notes = "short fit"
        )
        @test s isa NI.NeuralSurrogate
        @test NI.worst_error(s.provenance) < 0.05
        @test s.provenance.nsamples == 400
        @test s.provenance.nvalidation == 100
        @test !isempty(s.provenance.created)
        # The extracted network is dependency-free Float64 and reproduces the
        # tensor through the ordinary inclusion path.
        @test eltype(s.net) === Float64
        incl = NeuralHillInclusion((1.0, 1.0, 0.5); transport = s, guard = :error)
        P_nn = get_array(hill_tensor(incl, NN_K_M))
        P_ex = get_array(hill_tensor(_nn_spheroid(0.5), NN_K_M))
        @test maximum(abs, P_nn .- P_ex) ≤ 0.05 * maximum(abs, P_ex)
    end
end
