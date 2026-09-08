# The surrogate side of the axisymmetric cavity: the transport class it needs,
# the reference medium it has to *state* rather than guess, the two knobs that
# turned a mediocre fit into a good one, and the two shipped models.
#
# Nothing here fits a network on a real teacher — that is a dataset of
# finite-element solves and belongs in `scripts/nn/train_supershape.jl`. What is
# checked is the machinery around it, all of which was a silent wrong answer
# before it was an error.

using Test
using Random
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const NIA = MeanFieldHomogenization.NeuralInclusions

@testset "the transport localization class of an axisymmetric cavity" begin
    c = GradLocTI2()

    @testset "two components, and no scale divides out" begin
        # A second-order tensor invariant under the rotations about one axis has
        # two constants, where the octahedral group leaves exactly one.
        @test NIA.ncomponents(c) == 2
        @test NIA.ncomponents(GradLocISO2()) == 1
        @test NIA.tensor_order(c) == 2
        @test NIA.class_name(c) == :grad_loc_ti2
        @test NIA.hill_class(:grad_loc_ti2) isa GradLocTI2

        # The distinction from `HillTI2`, which has the same two components and
        # a different dimension. Reusing it would divide every prediction by k₀.
        for k₀ in (1.0, 4.0, 0.25)
            K₀ = TensISO{3}(k₀)
            @test NIA.dimensionless_scale(c, K₀) == 1
            @test NIA.dimensionless_scale(HillTI2(), K₀) == k₀
        end
        @test_throws ArgumentError NIA.material_coeffs(c, TensISO{3}(1.0))
    end

    @testset "it builds a transversely isotropic tensor about the given axis" begin
        can = TensND.CanonicalBasis{3, Float64}()
        e₃ = TensND.Tens(TensND.Vec(0.0, 0.0, 1.0), can)
        A = NIA.build(c, [1.5, 2.5], e₃)
        @test A isa TensND.TensTI{2}
        @test collect(NIA.components(c, A, e₃)) ≈ [1.5, 2.5]
        # The components in canonical axes, so that which one is which is pinned
        # rather than left to a convention: `a` is transverse, `b` axial.
        M = Matrix(TensND.get_array(TensND.change_tens(A, can)))
        @test diag(M) ≈ [1.5, 1.5, 2.5]
        @test norm(M - Diagonal(diag(M))) < 1.0e-14
    end

    @testset "and it decodes without dividing by k₀" begin
        can = TensND.CanonicalBasis{3, Float64}()
        e₃ = TensND.Tens(TensND.Vec(0.0, 0.0, 1.0), can)
        z = [1.5, 2.5]
        for k₀ in (1.0, 4.0)
            K₀ = TensISO{3}(k₀)
            a = NIA.decode(NIA.DimensionlessHill(c), z, K₀, e₃)
            b = NIA.decode(NIA.DimensionlessHill(HillTI2()), z, K₀, e₃)
            @test collect(NIA.components(c, a, e₃)) ≈ z
            @test collect(NIA.components(HillTI2(), b, e₃)) ≈ z ./ k₀
        end
    end

    @testset "a reference medium exists for it, and needs no ν₀" begin
        box = NIA.SampleBox([:log_p], [log(0.25)], [log(1.5)])
        @test NIA._reference_medium(c, box, [0.0]) isa TensND.TensISO{2, 3}
    end
end

@testset "the reference medium of a cavity is stated, not guessed" begin
    # `StrainLocTI` has no `_reference_medium` method, deliberately: the same
    # class serves *heterogeneous* morphologies that carry their constituents
    # inside themselves, and guessing there would train on corrupted labels. A
    # cavity has no constituent, so the caller declares the medium.
    box = NIA.SampleBox([:log_p, :nu0], [log(0.25), 0.0], [log(1.5), 0.45])
    @test_throws MethodError NIA._reference_medium(StrainLocTI(), box, [0.0, 0.3])

    ref = (b, x) -> NIA._iso_ref(x[NIA.feature_index(b, :nu0)])
    C₀ = ref(box, [0.0, 0.3])
    @test C₀ isa TensND.TensISO{4, 3}
    _, ν = MeanFieldHomogenization.Elasticity.E_nu(C₀)
    @test ν ≈ 0.3

    # And an affine output refuses the keyword outright: its labels are the
    # coefficients of the reference dependence, so a fixed medium is meaningless.
    @test :reference in Base.kwarg_decl(
        only(methods(NIA.generate_dataset, Tuple{Any, Any, Any, Any, Integer}))
    )
end

@testset "a pore names its shape features, including the derived ones" begin
    shape = Superspheroid(1.0, 0.4, 0.4)
    pore = FEAxiSupershapePore(shape)
    K₀ = TensISO{3}(1.0)

    @test NIA._shape_param(pore, :p) == 0.4
    @test NIA._shape_param(pore, :c) == 0.4
    @test NIA._shape_param(pore, :a) == 1.0
    @test NIA._shape_param(pore, :nonesuch) === nothing

    # The two derived features, which exist because a `SampleBox` is linear and
    # the feature is therefore the sampling law.
    @test NIA._feature(Val(:log_p), pore, K₀) ≈ log(0.4)
    @test NIA._feature(Val(:log_aspect), pore, K₀) ≈ log(0.4)
    # `log_aspect` keeps the spheroid convention: zero at the equiaxed case,
    # positive prolate, negative oblate.
    @test NIA._feature(
        Val(:log_aspect), FEAxiSupershapePore(
            Superspheroid(1.0, 1.0, 0.8)
        ), K₀
    ) ≈ 0.0 atol = 1.0e-15
    @test NIA._feature(
        Val(:log_aspect), FEAxiSupershapePore(
            Superspheroid(1.0, 2.0, 0.8)
        ), K₀
    ) > 0
end

@testset "the log output scaling is a knob, and it fires where it should" begin
    # `fit_scaling` picks `:log` for a positive row whose dynamic range exceeds
    # the threshold. The default of 30 was calibrated on components that span
    # decades; `R₃₃` of a concave cavity spans a factor of nine and still needs
    # it, which is the whole reason the threshold is exposed.
    X = reshape(collect(range(-1.0, 1.0; length = 20)), 1, 20)
    Z = vcat(
        reshape(collect(range(1.0, 9.0; length = 20)), 1, 20),   # range 9
        reshape(collect(range(1.0, 1.5; length = 20)), 1, 20),   # range 1.5
        reshape(collect(range(-1.0, 1.0; length = 20)), 1, 20),  # not positive
    )
    train = NIA.Dataset(X, Z, [:log_p])

    s30 = NIA.fit_scaling(train)                        # the default
    s5 = NIA.fit_scaling(train; log_threshold = 5.0)
    @test s30.y_kind == [:identity, :identity, :identity]
    @test s5.y_kind == [:log, :identity, :identity]

    # A negative row is never logged, whatever the threshold.
    @test NIA.fit_scaling(train; log_threshold = 1.0).y_kind[3] == :identity

    # And the option carries it, so a script states it rather than patching a
    # default.
    @test NIA.TrainingOptions().log_threshold == 30.0
    @test NIA.TrainingOptions(; log_threshold = 5.0).log_threshold == 5.0
end

@testset "a component that crosses zero is flagged, not quietly reported" begin
    # `ℓ₃` and `ℓ₄` of a superspheroidal cavity pass within 2e-4 of zero inside
    # the training box, so their per-component *relative* error is not a
    # measure of anything. The report has to say so: a reader who is not told
    # will read the column.
    net = NIA.glorot_mlp(Random.MersenneTwister(7), [1, 4, 2])
    s = NIA.NeuralSurrogate(;
        net, features = [:log_p], output = NIA.DimensionlessHill(GradLocTI2()),
        domain_lo = [-1.0], domain_hi = [1.0],
    )
    X = reshape(collect(range(-1.0, 1.0; length = 8)), 1, 8)
    # First row strictly positive, second row straddling zero.
    Z = vcat(
        reshape(collect(range(1.0, 2.0; length = 8)), 1, 8),
        reshape(collect(range(-0.5, 0.5; length = 8)), 1, 8),
    )
    io = IOBuffer()
    NIA.report_surrogate(io, s, NIA.Dataset(X, Z, [:log_p]); labels = [:a, :b])
    out = String(take!(io))
    @test occursin("crosses zero", out)
    @test occursin("1 of 2", out)
    # And the positive row carries no flag.
    lines = split(out, '\n')
    arow = only(filter(l -> startswith(strip(l), "a "), lines))
    @test !occursin("crosses zero", arow)
end

@testset "validation reports the distribution, not only its maximum" begin
    # A maximum over a heavy-tailed error distribution is not a summary of it,
    # and it is not comparable between runs: a held-out set 2.2 times larger
    # reaches further into the tail, so the maximum rises while every rms falls.
    # That is measured behavior on the axisymmetric elastic pore, and it is why
    # `validate_surrogate` returns quantiles.
    net = NIA.glorot_mlp(Random.MersenneTwister(11), [1, 8, 2])
    s = NIA.NeuralSurrogate(;
        net, features = [:log_p], output = NIA.DimensionlessHill(GradLocTI2()),
        domain_lo = [-1.0], domain_hi = [1.0],
    )
    n = 50
    X = reshape(collect(range(-1.0, 1.0; length = n)), 1, n)
    Z = vcat(
        reshape(collect(range(1.0, 3.0; length = n)), 1, n),
        reshape(collect(range(2.0, 5.0; length = n)), 1, n),
    )
    v = NIA.validate_surrogate(s, NIA.Dataset(X, Z, [:log_p]))

    @test length(v.block_errors) == n
    @test all(≥(0), v.block_errors)
    # The ordering that makes the quantiles meaningful at all.
    @test v.block_median ≤ v.block_p90 ≤ v.block_p99 ≤ v.max_block_error
    @test v.block_rms ≤ v.max_block_error
    @test v.max_block_error == maximum(v.block_errors)
    # The median is a genuine order statistic of the same sample.
    @test v.block_median in v.block_errors
    # And the legacy names still say what they said.
    @test v.worst == v.max_block_error
end

@testset "the shipped axisymmetric models" begin
    shipped = NIA.shipped_models()
    for name in ("axi_supershape_pore_conduction", "axi_supershape_pore_elastic")
        @test name in shipped
    end
    sc = NIA.load_surrogate(NIA.model_path("axi_supershape_pore_conduction"))
    se = NIA.load_surrogate(NIA.model_path("axi_supershape_pore_elastic"))

    @testset "they declare what they are" begin
        @test NIA.hill_class(sc) isa GradLocTI2
        @test NIA.hill_class(se) isa StrainLocTI
        @test NIA.tensor_order(sc) == 2
        @test NIA.tensor_order(se) == 4
        @test sc.features == [:log_aspect, :log_p]
        @test se.features == [:log_aspect, :log_p, :nu0]
        # Fit-limited rather than teacher-limited, transverse isotropy being
        # structural in the Fourier cell. These bounds sit just above the
        # measured worst case, which for the elastic model is a *single* point
        # in the near-crack corner of the box — its rms is 6.4e-3 and its p90
        # 6.6e-3. Bounding the maximum still catches a regression; reading it as
        # the model's accuracy would be wrong, so the distribution is bounded
        # too.
        @test NIA.worst_error(sc.provenance) < 5.0e-3
        @test NIA.worst_error(se.provenance) < 7.0e-2
    end

    @testset "a pore accepts them and answers without meshing" begin
        for (c, p) in ((1.0, 1.0), (0.5, 0.5), (2.0, 0.8))
            pore = FEAxiSupershapePore(
                Superspheroid(1.0, c, p); elastic = se, transport = sc,
                guard = :error,
            )
            @test has_surrogate(pore, 4)
            @test has_surrogate(pore, 2)

            A = fe_axi_pore_localization(pore, iso_stiffness(1.0, 0.6))
            @test A isa TensND.TensTI{4}
            B = fe_axi_pore_localization(pore, TensISO{3}(2.5))
            @test B isa TensND.TensTI{2}
            @test fe_assembly_count(pore) == 0

            # Degree 0 in k₀, which is what `GradLocTI2` encodes.
            C = fe_axi_pore_localization(pore, TensISO{3}(1.0))
            @test collect(TensND.get_data(B)) ≈ collect(TensND.get_data(C))
        end
    end

    @testset "the sphere is the control" begin
        # `Superspheroid(1, 1, 1)` is a sphere and both answers are exact.
        pore = FEAxiSupershapePore(
            Superspheroid(1.0, 1.0, 1.0); elastic = se, transport = sc
        )
        can = TensND.CanonicalBasis{3, Float64}()

        a = Matrix(
            TensND.get_array(
                TensND.change_tens(
                    fe_axi_pore_localization(pore, TensISO{3}(1.0)), can
                )
            )
        )
        @test diag(a) ≈ [1.5, 1.5, 1.5] rtol = 4 * NIA.worst_error(sc.provenance)

        ν = 0.3
        C₀ = iso_stiffness(1.0 / (3 * (1 - 2ν)), 1.0 / (2 * (1 + ν)))
        SJ = (1 + ν) / (3 * (1 - ν))
        SK = 2 * (4 - 5ν) / (15 * (1 - ν))
        v = [1.0, 1, 1, 0, 0, 0]
        J = v * v' / 3
        K = Matrix(1.0I, 6, 6) - J
        Aex = inv(Matrix(1.0I, 6, 6) - (SJ * J + SK * K))
        A = Matrix(KM(fe_axi_pore_localization(pore, C₀)))
        @test norm(A - Aex) / norm(Aex) < 4 * NIA.worst_error(se.provenance)
    end

    @testset "and the domain guard refuses extrapolation" begin
        # Trained on p ∈ [0.25, 1.5] and c/a ∈ [0.5, 2]; outside, the answer is
        # refused rather than extrapolated — in either variable.
        for sh in (Superspheroid(1.0, 1.0, 0.15), Superspheroid(1.0, 4.0, 0.8))
            pore = FEAxiSupershapePore(sh; transport = sc, guard = :error)
            @test_throws Exception fe_axi_pore_localization(pore, TensISO{3}(1.0))
        end
    end
end
