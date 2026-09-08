# A surrogate standing in for the finite-element cell of a cube-symmetric pore.
#
# The teacher here is **synthetic**: a closed-form cubic tensor, smooth in the
# shape exponent and in ν₀. That is deliberate and it is the same device
# `84_neural_inclusion_ellipsoid.jl` uses for gate A — it exercises the whole
# path (dataset, fit, decode, inclusion, schemes, sensitivity) against an
# exactly known answer, with no mesh and no solve anywhere. Training on the real
# teacher, `fe_cell_localization`, is a dataset of finite-element solves and
# belongs in a script, not in the suite.
#
# What the path has to deliver is two things a finite-element solve cannot:
# a response in microseconds, and a **derivative with respect to the
# morphology**. The second is the one worth testing.

using Test
using Random
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff

const NI = MeanFieldHomogenization.NeuralInclusions

@testset "cubic localization surrogate" begin

    can = TensND.CanonicalBasis{3, Float64}()

    # A stand-in for the cavity's strain localization: cubic, dimensionless,
    # smooth in (p, ν₀), and genuinely anisotropic (β ≠ γ) so that the class is
    # doing work.
    _teacher(p, ν) = TensND.TensCubic(1 + 2p, 1 + p + ν, 1 + p^2 / 2, can)

    # Named `_geom` and not `geometry`: `MeanFieldHomogenization.geometry` is the
    # sensitivity-parameter constructor, and shadowing it here would break the
    # very test that uses it.
    _geom(x) = FESupershapePore(Supersphere(1.0, x[1]))
    function response(g, C₀)
        _, ν = MeanFieldHomogenization.Core.extract_iso_moduli(C₀)
        return _teacher(g.shape.p, ν)
    end

    spec = NI.DimensionlessHill(StrainLocCubic())
    box = NI.SampleBox([:p, :nu0], [0.3, 0.0], [1.5, 0.45])

    @testset "the class round-trips through build and components" begin
        @test NI.ncomponents(StrainLocCubic()) == 3
        @test NI.tensor_order(StrainLocCubic()) == 4
        @test NI.class_name(StrainLocCubic()) == :loc_cubic
        @test NI.hill_class(spec) isa StrainLocCubic
        @test NI.noutputs(spec) == 3
        # Three components and not six: a localization tensor has no major
        # symmetry in general -- which is why `StrainLocTI` needs six -- but a
        # cubic one has it automatically, so there is nothing to drop.
        @test NI.ncomponents(StrainLocTI()) == 6

        C = _teacher(0.6, 0.3)
        c = NI.components(StrainLocCubic(), C, can)
        @test length(c) == 3
        @test all(isapprox.(c, TensND.get_data(C); atol = 1.0e-12))
        @test norm(KM(NI.build(StrainLocCubic(), c, can)) - KM(C)) < 1.0e-12

        # A tensor outside the class is refused loudly rather than trained on.
        ti = TensND.tens_TI(10.0, 3.0, 2.5, 12.0, 2.0, [0.0, 0.0, 1.0])
        @test_throws ArgumentError NI.components(StrainLocCubic(), ti, can)

        # The frame comes from the inclusion's own basis, all three columns of
        # it -- a cube has no distinguished direction to single out.
        pore = FESupershapePore(Supersphere(1.0, 0.6))
        @test NI._class_frame(StrainLocCubic(), pore) === inclusion_basis(pore)
    end

    @testset "the pore exposes its parameters as features" begin
        pore = FESupershapePore(Supersphere(1.0, 0.6))
        @test pore_shape_params(pore) == (; a = 1.0, p = 0.6)
        @test pore_shape_params(FESupershapePore(Superspheroid(1.0, 2.0, 0.6))) ==
            (; a = 1.0, c = 2.0, p = 0.6)
        @test !has_surrogate(pore, 4)
        @test !has_surrogate(pore, 2)
    end

    if NN_HAS_LUX
        train, val = NI.generate_dataset(_geom, response, spec, box, 600; nvalidation = 150)
        s = NI.train_surrogate(
            spec, box, train, val;
            teacher_name = "synthetic cubic localization",
            options = NI.TrainingOptions(; hidden = [24, 24], epochs = 1500, seed = 1),
        )

        @testset "it learns the teacher" begin
            # The teacher here is a closed form, so the fit is the only source
            # of error and 5 % is a generous ceiling on "the pipeline ran".
            @test NI.worst_error(s.provenance) < 5.0e-2
            # What the accuracy is checked against is the surrogate's *own*
            # reported error, not a literal — the repository's doctrine, and
            # what stops a retraining from silently loosening a threshold. The
            # factor of three is the margin between a worst case over the
            # held-out set and one at a point drawn from outside it.
            tol = 3 * NI.worst_error(s.provenance)
            C₀ = iso_stiffness(2.0, 0.75)
            _, ν = MeanFieldHomogenization.Core.extract_iso_moduli(C₀)
            for p in (0.4, 0.8, 1.2)
                pore = FESupershapePore(Supersphere(1.0, p); elastic = s)
                A = fe_cell_localization(pore, C₀)
                @test A isa TensND.TensCubic
                @test norm(KM(A) - KM(_teacher(p, ν))) < tol * norm(KM(_teacher(p, ν)))
                # No mesh was built and no system factorized.
                @test fe_assembly_count(pore) == 0
                @test has_surrogate(pore, 4)
            end
        end

        @testset "it reaches the schemes, with no backend loaded" begin
            C₀ = iso_stiffness(2.0, 0.75)
            pore = FESupershapePore(Supersphere(1.0, 0.7); elastic = s)
            rve = RVE()
            add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
            add_phase!(rve, :p, pore, Dict(:C => C₀); fraction = 0.05)
            Cd = homogenize(rve, Dilute(), :C)
            N = stiffness_contribution(pore, C₀, C₀)
            @test norm(KM(Cd) - KM(C₀ + 0.05 * N)) < 1.0e-12 * norm(KM(Cd))
            @test fe_assembly_count(pore) == 0
            # Still a cavity: the stress side is identically zero, so the
            # contribution identity holds exactly as it does for the meshed one.
            H = compliance_contribution(pore, C₀, C₀)
            @test norm(KM(H ⊡ C₀ + inv(C₀) ⊡ N)) < 1.0e-10 * norm(KM(H ⊡ C₀))
        end

        @testset "and it differentiates in the morphology, which the mesh cannot" begin
            C₀ = iso_stiffness(2.0, 0.75)
            build_rve = p -> begin
                r = RVE()
                add_phase!(r, :m, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
                add_phase!(
                    r, :p, FESupershapePore(Supersphere(1.0, p); elastic = s),
                    Dict(:C => C₀); fraction = 0.05,
                )
                r
            end
            d = derivative(build_rve(0.7), Dilute(), geometry(:p, :p))
            @test all(isfinite, KM(d))
            @test norm(KM(d)) > 1.0e-6                     # it really depends on p

            # Against a central difference over freshly built inclusions.
            h = 1.0e-5
            fd = (
                KM(homogenize(build_rve(0.7 + h), Dilute(), :C)) -
                    KM(homogenize(build_rve(0.7 - h), Dilute(), :C))
            ) ./ 2h
            @test norm(Matrix(KM(d)) - Matrix(fd)) < 1.0e-4 * norm(Matrix(fd))

            # The meshed pore refuses the same request rather than returning a
            # silent zero -- the contrast that is the whole point.
            plain = RVE()
            add_phase!(plain, :m, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
            add_phase!(
                plain, :p, FESupershapePore(Supersphere(1.0, 0.7)),
                Dict(:C => C₀); fraction = 0.05,
            )
            @test_throws ErrorException derivative(plain, Dilute(), geometry(:p, :p))
        end
    else
        @info "Lux / Optimisers / Zygote unavailable — skipping the cubic " *
            "surrogate fit."
    end
end

# The transport class of a cavity, and the construction-time guards. None of
# this needs a fit, so it runs whether or not Lux is installed -- which matters,
# because these are the checks that turn a silent wrong answer into an error.
@testset "the transport localization class of a cavity" begin
    NI = MeanFieldHomogenization.NeuralInclusions
    c = GradLocISO2()

    @testset "one component, and no scale divides out" begin
        # A second-order tensor invariant under the octahedral group is
        # isotropic, so a supersphere needs three constants in elasticity and
        # exactly one here.
        @test NI.ncomponents(c) == 1
        @test NI.tensor_order(c) == 2
        @test NI.class_name(c) == :grad_loc_iso2
        @test NI._class_frame(c, nothing) === nothing

        # The distinction from `HillISO2`, and the reason the class exists:
        # `𝑨_∇∇` is of degree 0 in `k₀`, the Hill tensor of degree −1.
        for k₀ in (1.0, 4.0, 0.25)
            K₀ = TensISO{3}(k₀)
            @test NI.dimensionless_scale(c, K₀) == 1
            @test NI.dimensionless_scale(HillISO2(), K₀) == k₀
        end

        A = NI.build(c, [1.5], nothing)
        @test A isa TensND.TensISO{2, 3}
        @test collect(NI.components(c, A, nothing)) ≈ [1.5]
        @test NI.hill_class(:grad_loc_iso2) isa GradLocISO2
        # A localization class has no affine decomposition on shape tensors.
        @test_throws ArgumentError NI.material_coeffs(c, TensISO{3}(1.0))
    end

    @testset "and it decodes without dividing by k₀" begin
        # The failure this class exists to prevent, made explicit: decoding the
        # same prediction under `HillISO2` scales with `k₀`, under
        # `GradLocISO2` it does not.
        z = [1.5]
        for k₀ in (1.0, 4.0)
            K₀ = TensISO{3}(k₀)
            a = TensND.get_data(NI.decode(NI.DimensionlessHill(c), z, K₀, nothing))[1]
            b = TensND.get_data(
                NI.decode(NI.DimensionlessHill(HillISO2()), z, K₀, nothing)
            )[1]
            @test a ≈ 1.5
            @test b ≈ 1.5 / k₀
        end
    end

    @testset "a reference medium exists for it, and needs no ν₀" begin
        box = NI.SampleBox([:p], [0.3], [1.5])
        @test NI._reference_medium(c, box, [0.7]) isa TensND.TensISO{2, 3}
        # The cavity restriction on `StrainLocTI` is about the morphology, not
        # the class, and it still has no method.
        @test_throws MethodError NI._reference_medium(StrainLocTI(), box, [0.7])
    end
end

@testset "a pore validates its surrogate when it is built, not when it solves" begin
    NI = MeanFieldHomogenization.NeuralInclusions
    shape = Supersphere(1.0, 0.6)
    net = NI.glorot_mlp(Random.MersenneTwister(1), [1, 4, 1])

    ok2 = NI.NeuralSurrogate(;
        net, features = [:p], output = NI.DimensionlessHill(GradLocISO2()),
        domain_lo = [0.3], domain_hi = [1.5],
    )
    # Right order, right class, feature the pore can supply.
    @test FESupershapePore(shape; transport = ok2) isa FESupershapePore

    # Wrong order: an order-2 surrogate in the elastic slot.
    @test_throws ArgumentError FESupershapePore(shape; elastic = ok2)

    # Wrong class: `HillISO2` has the same one component and divides by k₀.
    bad_class = NI.NeuralSurrogate(;
        net, features = [:p], output = NI.DimensionlessHill(HillISO2()),
        domain_lo = [0.3], domain_hi = [1.5],
    )
    @test_throws ArgumentError FESupershapePore(shape; transport = bad_class)

    # A feature the shape cannot supply: `:c` belongs to a superspheroid.
    bad_feat = NI.NeuralSurrogate(;
        net, features = [:c], output = NI.DimensionlessHill(GradLocISO2()),
        domain_lo = [0.3], domain_hi = [1.5],
    )
    @test_throws ArgumentError FESupershapePore(shape; transport = bad_feat)
    # The same surrogate is fine on a superspheroid, which has a `c`.
    @test FESupershapePore(Superspheroid(1.0, 2.0, 0.6); transport = bad_feat) isa
        FESupershapePore

    # The names come from the shape, so they can be asked for before a pore is
    # built -- which is what makes the check possible at construction.
    @test propertynames(
        MeanFieldHomogenization.FiniteElements.pore_shape_params(shape)
    ) == (:a, :p)
end

# The two models this package ships for the supershape cavity. Loading them
# needs no Lux and no backend, which is the whole point of committing them: a
# user gets the route without a training stack, and the suite gets a check on
# the committed artifacts rather than only on a freshly fitted one.
@testset "the shipped supershape models" begin
    NI = MeanFieldHomogenization.NeuralInclusions
    shipped = NI.shipped_models()

    @testset "they are there, and they declare what they are" begin
        for name in ("supershape_pore_conduction", "supershape_pore_elastic")
            @test name in shipped
        end
        sc = NI.load_surrogate(NI.model_path("supershape_pore_conduction"))
        se = NI.load_surrogate(NI.model_path("supershape_pore_elastic"))
        @test NI.hill_class(sc) isa GradLocISO2
        @test NI.hill_class(se) isa StrainLocCubic
        @test NI.tensor_order(sc) == 2
        @test NI.tensor_order(se) == 4
        @test sc.features == [:p]
        @test se.features == [:p, :nu0]
        # Teacher-limited, not fit-limited: the finite-element cell departs from
        # the symmetry class by about 1.5e-2 at the concave end, so these are the
        # honest figures rather than targets.
        @test NI.worst_error(sc.provenance) < 5.0e-3
        @test NI.worst_error(se.provenance) < 5.0e-2
    end

    @testset "a pore accepts them, and answers without meshing" begin
        sc = NI.load_surrogate(NI.model_path("supershape_pore_conduction"))
        se = NI.load_surrogate(NI.model_path("supershape_pore_elastic"))
        for p in (0.5, 1.0, 2.0)
            pore = FESupershapePore(
                Supersphere(1.0, p); elastic = se, transport = sc, guard = :error
            )
            @test has_surrogate(pore, 4)
            @test has_surrogate(pore, 2)

            A = fe_cell_localization(pore, iso_stiffness(1.0, 0.6))
            @test A isa TensND.TensCubic
            B = fe_cell_localization(pore, TensISO{3}(2.5))
            @test B isa TensND.TensISO{2, 3}
            @test fe_assembly_count(pore) == 0

            # Scale-free in k₀, which is what `GradLocISO2` encodes and what
            # `HillISO2` would have destroyed.
            C = fe_cell_localization(pore, TensISO{3}(1.0))
            @test collect(TensND.get_data(B)) ≈ collect(TensND.get_data(C))
        end
    end

    @testset "the sphere is the control" begin
        # At p = 1 the cavity is a sphere and both answers are known exactly.
        sc = NI.load_surrogate(NI.model_path("supershape_pore_conduction"))
        se = NI.load_surrogate(NI.model_path("supershape_pore_elastic"))
        pore = FESupershapePore(Supersphere(1.0, 1.0); elastic = se, transport = sc)

        a = only(TensND.get_data(fe_cell_localization(pore, TensISO{3}(1.0))))
        @test a ≈ 1.5 rtol = 3 * NI.worst_error(sc.provenance)

        ν = 0.3
        C₀ = iso_stiffness(1.0 / (3 * (1 - 2ν)), 1.0 / (2 * (1 + ν)))
        SJ = (1 + ν) / (3 * (1 - ν))
        SK = 2 * (4 - 5ν) / (15 * (1 - ν))
        v = [1.0, 1, 1, 0, 0, 0]
        J = v * v' / 3
        K = Matrix(1.0I, 6, 6) - J
        Aex = inv(Matrix(1.0I, 6, 6) - (SJ * J + SK * K))
        A = Matrix(KM(fe_cell_localization(pore, C₀)))
        @test norm(A - Aex) / norm(Aex) < 3 * NI.worst_error(se.provenance)
    end

    @testset "and the domain guard refuses extrapolation" begin
        # The models are trained on p ∈ [0.35, 2.5], the lower bound being where
        # the teacher stops being trustworthy rather than where the physics
        # stops. Below it the answer is refused, not extrapolated.
        sc = NI.load_surrogate(NI.model_path("supershape_pore_conduction"))
        pore = FESupershapePore(Supersphere(1.0, 0.2); transport = sc, guard = :error)
        @test_throws Exception fe_cell_localization(pore, TensISO{3}(1.0))
    end
end
