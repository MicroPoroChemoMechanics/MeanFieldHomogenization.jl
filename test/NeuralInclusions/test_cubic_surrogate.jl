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
            @test NI.worst_error(s.provenance) < 5.0e-2
            C₀ = iso_stiffness(2.0, 0.75)
            _, ν = MeanFieldHomogenization.Core.extract_iso_moduli(C₀)
            for p in (0.4, 0.8, 1.2)
                pore = FESupershapePore(Supersphere(1.0, p); elastic = s)
                A = fe_cell_localization(pore, C₀)
                @test A isa TensND.TensCubic
                @test norm(KM(A) - KM(_teacher(p, ν))) < 5.0e-2 * norm(KM(_teacher(p, ν)))
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
