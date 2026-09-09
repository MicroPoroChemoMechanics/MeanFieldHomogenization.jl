# `AnchoredHill`: predict the departure from a closed form instead of the answer.
#
# Two properties make it correct, and both are checked here rather than argued:
# **exactness** — on the face where the baseline is the answer, the label is the
# identity, so a network predicting the identity is exact — and the **round
# trip**, `decode ∘ encode = id`, without which training and prediction live in
# different spaces and every prediction is wrong with no error raised.
#
# What it is *not* is a component-wise ratio. `𝔸_b⁻¹ : 𝔸` is formed with `inv`
# and the double contraction on transversely isotropic tensors, so it stays in
# the Walpole basis; dividing entry by entry in Kelvin-Mandel mixes the
# transverse block and manufactures sign changes that are an artifact of the
# reading. Measured, and the reason this is a specification and not a division.

using Test
using Random
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const NIB = MeanFieldHomogenization.NeuralInclusions

const _AN_CAN = TensND.CanonicalBasis{3, Float64}()
const _AN_E3 = TensND.Tens(TensND.Vec(0.0, 0.0, 1.0), _AN_CAN)
_an_km(t) = Matrix(KM(TensND.change_tens(t, _AN_CAN)))
_an_arr(t) = Matrix(TensND.get_array(TensND.change_tens(t, _AN_CAN)))

# The exact cavity of revolution, built the way the finite-element suite builds
# its oracle — independently of the code under test.
_an_exact4(c, ν) = let C₀ = NIB._iso_ref(ν)
    inv(TensISO{3}(1.0, 1.0) - (hill_tensor(Spheroid(c), C₀) ⊡ C₀))
end
_an_exact2(c) = let K₀ = TensISO{3}(1.0)
    inv(TensISO{3}(1.0) - (hill_tensor(Spheroid(c), K₀) ⋅ K₀))
end

@testset "anchored output specification" begin

    @testset "it declares itself, and refuses an unknown baseline" begin
        sp = AnchoredHill(StrainLocTI(), :spheroid_cavity)
        @test NIB.spec_name(sp) === :anchored
        @test NIB.spec_baseline(sp) === :spheroid_cavity
        @test NIB.hill_class(sp) isa StrainLocTI
        @test NIB.nterms(sp) == 1
        @test NIB.noutputs(sp) == 6
        @test NIB.needs_nu(sp)
        @test !NIB.needs_nu(AnchoredHill(GradLocTI2(), :spheroid_cavity))
        @test :spheroid_cavity in NIB.anchor_baselines()
        @test_throws ArgumentError AnchoredHill(StrainLocTI(), :nonesuch)
        # The unanchored specifications have no baseline, and say so.
        @test NIB.spec_baseline(DimensionlessHill(StrainLocTI())) === nothing
    end

    @testset "it round-trips through the serialized names" begin
        sp = NIB.output_spec(:anchored, :loc_ti, "spheroid_cavity")
        @test sp isa AnchoredHill
        @test NIB.spec_baseline(sp) === :spheroid_cavity
        @test NIB.hill_class(sp) isa StrainLocTI
        # A file written before the field existed carries no baseline, and that
        # is an error rather than a guess.
        @test_throws ArgumentError NIB.output_spec(:anchored, :loc_ti, nothing)
        @test_throws ArgumentError NIB.output_spec(:nonesuch, :loc_ti)
    end

    @testset "exact on the face where the baseline is the answer" begin
        # A superspheroid at p = 1 *is* a spheroid, so the whole face p = 1 —
        # every aspect ratio, every ν₀, oblate through prolate — has the closed
        # form for its answer and the label must be the identity there.
        sp = AnchoredHill(StrainLocTI(), :spheroid_cavity)
        feats = [:log_aspect, :log_p, :nu0]
        zid = collect(NIB.components(StrainLocTI(), TensISO{3}(1.0, 1.0), _AN_E3))
        for c in (0.4, 0.7, 1.0, 1.6, 3.0), ν in (0.0, 0.3, 0.45)
            Aex = _an_exact4(c, ν)
            x = [log(c), 0.0, ν]
            z = NIB.encode(sp, Aex, NIB._iso_ref(ν), _AN_E3, x, feats; atol = 1.0e-6)
            @test z ≈ zid atol = 1.0e-10
            back = NIB.decode(sp, z, NIB._iso_ref(ν), _AN_E3, x, feats)
            @test norm(_an_km(back) - _an_km(Aex)) / norm(_an_km(Aex)) < 1.0e-12
        end
    end

    @testset "the prolate branch is the one that catches a frame error" begin
        # `Spheroid(ω)` sorts its semi-axes: the axis of revolution is `ez` when
        # oblate and `ex` when prolate. Reading the baseline in the localization
        # class's own convention — column 3 — is right for the morphology being
        # learned and wrong for the baseline, and it fails for prolate shapes
        # only. So an oblate-only check would pass over the bug.
        sp = AnchoredHill(StrainLocTI(), :spheroid_cavity)
        feats = [:log_aspect, :log_p, :nu0]
        for c in (1.5, 2.5, 4.0)
            Ab = NIB.anchor_tensor(sp, [log(c), 0.0, 0.3], feats, NIB._iso_ref(0.3), _AN_E3)
            # Same tensor as the closed form, in canonical components — and the
            # axis has to be e₃, not wherever sorting put it.
            @test norm(_an_km(Ab) - _an_km(_an_exact4(c, 0.3))) / norm(_an_km(Ab)) < 1.0e-12
            M = _an_km(Ab)
            @test M[3, 3] ≉ M[1, 1]          # genuinely transversely isotropic
            @test M[1, 1] ≈ M[2, 2]          # about e₃
        end
    end

    @testset "and in transport" begin
        sp = AnchoredHill(GradLocTI2(), :spheroid_cavity)
        f2 = [:log_aspect, :log_p]
        zid = collect(NIB.components(GradLocTI2(), TensISO{3}(1.0), _AN_E3))
        for c in (0.4, 1.0, 2.5)
            aex = _an_exact2(c)
            x = [log(c), 0.0]
            z = NIB.encode(sp, aex, TensISO{3}(1.0), _AN_E3, x, f2; atol = 1.0e-6)
            @test z ≈ zid atol = 1.0e-10
            back = NIB.decode(sp, z, TensISO{3}(1.0), _AN_E3, x, f2)
            @test norm(_an_arr(back) - _an_arr(aex)) < 1.0e-12
        end
    end

    @testset "the round trip is algebra, so it holds off the face too" begin
        sp = AnchoredHill(StrainLocTI(), :spheroid_cavity)
        feats = [:log_aspect, :log_p, :nu0]
        A = NIB.build(StrainLocTI(), [3.1, 5.7, -0.8, -0.6, 2.2, 1.9], _AN_E3)
        for c in (0.6, 1.0, 1.4), ν in (0.1, 0.35)
            x = [log(c), log(0.4), ν]
            P₀ = NIB._iso_ref(ν)
            z = NIB.encode(sp, A, P₀, _AN_E3, x, feats; atol = 1.0e-6)
            back = NIB.decode(sp, z, P₀, _AN_E3, x, feats)
            @test norm(_an_km(back) - _an_km(A)) / norm(_an_km(A)) < 1.0e-12
        end
    end

    @testset "the baseline must be computable from the features alone" begin
        # At prediction time there is no solve to read the geometry from, so a
        # feature set without an aspect ratio cannot be served — and it is
        # refused rather than defaulted to a sphere.
        sp = AnchoredHill(StrainLocTI(), :spheroid_cavity)
        @test_throws ArgumentError NIB.encode(
            sp, TensISO{3}(1.0, 1.0), NIB._iso_ref(0.3), _AN_E3, [0.4, 0.3], [:p, :nu0]
        )
        # `:aspect` is accepted beside `:log_aspect`, for a linearly sampled box.
        z = NIB.encode(
            sp, _an_exact4(0.8, 0.3), NIB._iso_ref(0.3), _AN_E3, [0.8, 0.3],
            [:aspect, :nu0]; atol = 1.0e-6
        )
        @test z ≈ collect(NIB.components(StrainLocTI(), TensISO{3}(1.0, 1.0), _AN_E3)) atol = 1.0e-10
    end

    @testset "a surrogate carries the specification through save and load" begin
        mktempdir() do dir
            sp = AnchoredHill(StrainLocTI(), :spheroid_cavity)
            net = NIB.glorot_mlp(Random.MersenneTwister(3), [3, 8, 6])
            s = NIB.NeuralSurrogate(;
                net, features = [:log_aspect, :log_p, :nu0], output = sp,
                domain_lo = [-1.0, -1.5, 0.0], domain_hi = [1.0, 0.5, 0.45],
            )
            path = NIB.save_surrogate(joinpath(dir, "anchored.json"), s)
            t = NIB.load_surrogate(path)
            @test NIB.spec_name(t.output) === :anchored
            @test NIB.spec_baseline(t.output) === :spheroid_cavity
            # And it still answers, which is what the baseline round trip is for.
            C₀ = NIB._iso_ref(0.3)
            x = [log(0.8), log(0.5), 0.3]
            @test s(x, C₀, _AN_E3; guard = :none) ≈ t(x, C₀, _AN_E3; guard = :none)
        end
    end
end

@testset "anchored spec — the layered-spheroid baseline" begin
    # The homogeneous spheroid at the layers' mean modulus. Its point is the
    # **face** it is exact on: a layered spheroid whose layers agree *is* a
    # homogeneous one, and `r₁ = r₂` is a three-dimensional face of the box the
    # shipped models are trained on, where `:spheroid_cavity` is exact at a
    # single point.
    ν = 0.2
    C₀ = NIB._iso_ref(ν)
    feats = [:log_aspect, :core_fraction, :log_mu_ratio_1, :log_mu_ratio_2]
    id4 = NIB._identity_4sym(Float64)
    own_axis(c) = let sph = Spheroid(c)
        MeanFieldHomogenization.Core._basis_col(
            MeanFieldHomogenization.Core.inclusion_basis(sph),
            NIB._spheroid_axis_index(NIB._axes(sph)),
        )
    end
    # The exact homogeneous-inclusion localization, built independently of the
    # code under test, and transversely isotropic about the spheroid's **own**
    # axis — which is `e₃` when oblate and `e₁` when prolate, since `Spheroid`
    # sorts its semi-axes. Hence the tests below pass that axis rather than
    # assuming `e₃`: assuming it would silently exercise the oblate branch only,
    # and the prolate one is where a frame mistake bites.
    exact_hom(c, r) = inv(id4 + (hill_tensor(Spheroid(c), C₀) ⊡ (r * C₀ - C₀)))

    @testset "it is declared" begin
        @test :layered_spheroid in NIB.anchor_baselines()
        @test NIB.spec_baseline(AnchoredHill(StrainLocTI(), :layered_spheroid)) ===
            :layered_spheroid
        @test NIB.output_spec(:anchored, :loc_ti, "layered_spheroid") isa AnchoredHill
    end

    @testset "exact wherever the layers agree, prolate and oblate" begin
        sp = AnchoredHill(StrainLocTI(), :layered_spheroid)
        for c in (2.5, 1.4, 0.8, 0.4), r in (0.6, 1.0, 3.0), w in (0.25, 0.7)
            axis = own_axis(c)
            x = [log(c), w, log(r), log(r)]                  # r₁ = r₂ = r
            z = NIB.encode(sp, exact_hom(c, r), C₀, axis, x, feats)
            @test z ≈ collect(NIB.components(StrainLocTI(), id4, axis)) atol = 1.0e-8
        end
    end

    @testset "the stress side carries the mean modulus, not the identity" begin
        sp = AnchoredHill(StressLocTI(), :layered_spheroid)
        for c in (1.8, 0.5), r in (0.8, 2.5), w in (0.3, 0.65)
            axis = own_axis(c)
            x = [log(c), w, log(r), log(r)]
            # A homogeneous inclusion's stress side *is* derivable: ℂ̄ : 𝔸.
            B = (r * C₀) ⊡ exact_hom(c, r)
            z = NIB.encode(sp, B, C₀, axis, x, feats)
            @test z ≈ collect(NIB.components(StressLocTI(), id4, axis)) atol = 1.0e-8
        end
        # And it is genuinely a different baseline from the strain side, or the
        # anchored target would still carry a modulus.
        strain = AnchoredHill(StrainLocTI(), :layered_spheroid)
        x = [log(2.0), 0.4, log(3.0), log(3.0)]
        ax = own_axis(2.0)
        @test !isapprox(
            _an_km(NIB.anchor_tensor(sp, x, feats, C₀, ax)),
            _an_km(NIB.anchor_tensor(strain, x, feats, C₀, ax)); atol = 1.0e-6
        )
    end

    @testset "the mean is the layer average, so w moves the baseline" begin
        sp = AnchoredHill(StrainLocTI(), :layered_spheroid)
        ax = own_axis(2.0)
        b(w) = _an_km(NIB.anchor_tensor(sp, [log(2.0), w, log(4.0), log(0.5)], feats, C₀, ax))
        @test !isapprox(b(0.25), b(0.7); atol = 1.0e-6)
        # and at r₁ = r₂ it does not, the average being r whatever the weights
        c(w) = _an_km(NIB.anchor_tensor(sp, [log(2.0), w, log(2.0), log(2.0)], feats, C₀, ax))
        @test c(0.25) ≈ c(0.7) atol = 1.0e-12
    end

    @testset "the frame asked for is the frame returned" begin
        # Decoding the identity returns the baseline itself, and it must be
        # transversely isotropic about the frame requested — `components`
        # measures the projection residual, so a frame mistake throws here
        # instead of passing silently.
        sp = AnchoredHill(StrainLocTI(), :layered_spheroid)
        for c in (2.5, 0.6)
            x = [log(c), 0.4, log(2.0), log(0.7)]
            z = collect(NIB.components(StrainLocTI(), id4, _AN_E3))
            Ab = NIB.decode(sp, z, C₀, _AN_E3, x, feats)
            @test length(collect(NIB.components(StrainLocTI(), Ab, _AN_E3; atol = 1.0e-10))) == 6
        end
    end

    @testset "round trip on a genuinely layered point" begin
        sp = AnchoredHill(StrainLocTI(), :layered_spheroid)
        x = [log(2.0), 0.4, log(3.0), log(0.6)]
        ax = own_axis(2.0)
        A = exact_hom(1.5, 2.0)                       # any tensor of the class
        z = NIB.encode(sp, A, C₀, ax, x, feats)
        @test _an_km(NIB.decode(sp, z, C₀, ax, x, feats)) ≈ _an_km(A) atol = 1.0e-10
    end

    @testset "it refuses features it cannot build the baseline from" begin
        sp = AnchoredHill(StrainLocTI(), :layered_spheroid)
        @test_throws ArgumentError NIB.encode(
            sp, id4, C₀, _AN_E3, [log(2.0)], [:log_aspect]
        )
        @test_throws ArgumentError NIB.encode(
            sp, id4, C₀, _AN_E3, [0.4, log(2.0), log(2.0)],
            [:core_fraction, :log_mu_ratio_1, :log_mu_ratio_2]
        )
    end

    @testset "component_labels answers for an anchored spec" begin
        # It did not: a shipped output specification could not be reported on.
        @test NIB.component_labels(AnchoredHill(StrainLocTI(), :layered_spheroid)) ==
            NIB.component_labels(StrainLocTI())
        @test NIB.component_labels(AnchoredHill(GradLocTI2(), :spheroid_cavity)) ==
            NIB.component_labels(GradLocTI2())
    end
end
