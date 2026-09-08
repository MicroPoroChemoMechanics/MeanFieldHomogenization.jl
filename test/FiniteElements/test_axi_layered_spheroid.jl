# An `N`-layer spheroid by Fourier axisymmetric finite elements.
#
# The space of nested coaxial spheroids is crossed by **two** independent exact
# families, and this file leans on both:
#
#   * confocal layers, where `LayeredSpheroid` is the closed form — and both
#     branches of it, since the oblate case runs the whole computation in
#     complex arithmetic through the substitution `c → -ic̄`, `q → iτ`, and
#     relies on an exact cancellation of the imaginary part. A prolate-only
#     comparison validates the easy half;
#   * concentric spheres of **arbitrary** radii, where `LayeredSphere` is the
#     closed form — which tests the layer count and free radii without assuming
#     anything confocal.
#
# They meet only at the equal-radii sphere, so between them they pin the mesher,
# the per-layer material map, the `N`-region average and the dipole correction
# over a genuinely two-parameter family. What is left uncovered is a nest whose
# layers have *different* aspect ratios, and for that the checks are the closed
# form of each layer's volume and mesh convergence — stated, not implied.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import Ferrite, FerriteGmsh, Gmsh

const _ALS = MeanFieldHomogenization.FiniteElements

const _ALS_CAN = TensND.CanonicalBasis{3, Float64}()
_als_m(t) = Matrix(TensND.get_array(TensND.change_tens(t, _ALS_CAN)))
_als_k(t) = Matrix(KM(TensND.change_tens(t, _ALS_CAN)))
_als_opts(nr = 12, rr = 5.0) = FEAxiMeshOptions(; nradial = nr, radius_ratio = rr)
_als_C(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))

@testset "axisymmetric layered spheroid" begin

    @testset "the geometry is the analytic type's, not a transcription of it" begin
        # `confocal_layer_radii` is the single source: the same pair builds both
        # inclusions, so a cross-check compares two solvers on one body.
        for (ω, fr) in ((2.0, (0.3, 0.7)), (0.5, (0.4, 0.6)), (1.8, (0.2, 0.3, 0.5)))
            ar, dr = confocal_layer_radii(ω, 1.0, fr)
            n = length(fr)
            @test length(ar) == n && length(dr) == n
            # Confocal, to round-off, in both families.
            f2 = [ar[ℓ]^2 - dr[ℓ]^2 for ℓ in 1:n]
            @test all(x -> isapprox(x, f2[1]; rtol = 1.0e-10), f2)
            # Ascending, hence nested, hence meshable.
            @test issorted(ar) && issorted(dr)
            # And the outer aspect ratio is what was asked for.
            @test ar[n] / dr[n] ≈ ω rtol = 1.0e-10

            K = ntuple(_ -> TensISO{3}(1.0), n)
            fe = FEAxiLayeredSpheroid(ar, dr, K)
            @test layer_count(fe) == n
            # The volume fractions come back as the fractions requested — an
            # independent check that the confocal bisection and the closed-form
            # layer volume agree.
            @test collect(layer_fractions(fe)) ≈ collect(fr ./ sum(fr)) rtol = 1.0e-8
            @test sum(layer_volumes(fe)) ≈ 4π / 3 * dr[n]^2 * ar[n] rtol = 1.0e-12
        end
    end

    @testset "nesting is refused at construction, not discovered in the mesher" begin
        # A violation is not a wrong answer: it is a self-intersecting geometry
        # that gmsh rejects from inside its own pipeline, naming neither the
        # layer nor the semi-axis.
        @test check_nested_spheroids([0.5, 1.0], [0.3, 0.6]) === nothing
        # `a == c` *within* a layer is a sphere, and unrestricted: it is the
        # slice where `LayeredSphere` is the closed form.
        @test check_nested_spheroids([1.0], [1.0]) === nothing
        @test check_nested_spheroids([0.5, 1.0], [0.5, 1.0]) === nothing
        # But two *coinciding boundaries* carry no volume and cannot be meshed —
        # the element size is capped at a fraction of a layer's thickness, so a
        # zero-thickness layer asks gmsh for a zero-sized element. Strictly
        # ascending, therefore, and refused rather than meshed into a hang.
        @test_throws ArgumentError check_nested_spheroids([0.5, 0.5], [0.3, 0.3])
        @test_throws ArgumentError check_nested_spheroids([0.5, 1.0], [0.6, 1.0])
        @test_throws ArgumentError check_nested_spheroids([0.5, 1.4], [1.1, 1.0])
        @test_throws ArgumentError check_nested_spheroids([1.4, 0.5], [0.9, 1.0])
        @test_throws ArgumentError check_nested_spheroids([0.0, 1.0], [0.5, 1.0])
        @test_throws ArgumentError check_nested_spheroids([0.5, 1.0], [0.5, 1.0], 0.9)
        @test_throws DimensionMismatch check_nested_spheroids([0.5, 1.0], [0.5])
        @test_throws ArgumentError check_nested_spheroids(Float64[], Float64[])
    end

    @testset "the constructor guards what it cannot serve" begin
        ar, dr = confocal_layer_radii(2.0, 1.0, (0.3, 0.7))
        K = (TensISO{3}(5.0), TensISO{3}(2.0))
        # One modulus per layer.
        @test_throws ArgumentError FEAxiLayeredSpheroid(ar, dr, (TensISO{3}(1.0),))
        # One physics per object: mixing orders is refused, not promoted.
        @test_throws ArgumentError FEAxiLayeredSpheroid(
            ar, dr, (TensISO{3}(1.0), _als_C(1.0, 0.3))
        )
        # And an imperfect interface is refused **by name** rather than ignored,
        # because the axisymmetric formulation has no jump term at all.
        @test_throws ArgumentError FEAxiLayeredSpheroid(
            ar, dr, K;
            interfaces = (KapitzaInterface(0.1), PerfectInterface{Float64}()),
        )
        @test_throws ArgumentError FEAxiLayeredSpheroid(
            ar, dr, K;
            interfaces = (SpringInterface(1.0, 1.0), PerfectInterface{Float64}()),
        )
        # A wider core inside a narrower shell.
        @test_throws ArgumentError FEAxiLayeredSpheroid((0.5, 1.4), (1.1, 1.0), K)
    end

    @testset "a sensitivity is refused, not answered with a zero" begin
        # The solve runs in `Float64` and memoizes on the reference medium
        # alone, so a derivative with respect to a layer radius would come back
        # a silent zero. The type has to be named in the refusal union: without
        # that, the request falls through to a generic that hands one back.
        ar, dr = confocal_layer_radii(2.0, 1.0, (0.3, 0.7))
        fe = FEAxiLayeredSpheroid(ar, dr, (TensISO{3}(5.0), TensISO{3}(2.0)))
        Sch = MeanFieldHomogenization.Schemes
        @test_throws Exception Sch._replace_geom_field(fe, Val(:axis_radii), nothing, 1.0)
        @test_throws Exception Sch._replace_geom_field(fe, Val(:disk_radii), 1, 1.0)
    end

    @testset "the object declares itself" begin
        ar, dr = confocal_layer_radii(2.0, 1.0, (0.3, 0.7))
        K = (TensISO{3}(5.0), TensISO{3}(2.0))
        fe = FEAxiLayeredSpheroid(ar, dr, K)
        Core_ = MeanFieldHomogenization.Core
        @test Core_.dimension(fe) == 3
        # Heterogeneous, so gate B needs both sides and the stress side is not
        # `ℂ₁ : 𝔸_εε` for any single `ℂ₁`.
        @test !Core_.is_homogeneous_inclusion(fe)
        @test MeanFieldHomogenization.Schemes.has_layer_average(fe)
        # The shape tensor is the outer boundary, distinct semi-axis on the axis.
        D = diag(_als_m(Core_.shape_tensor(fe)))
        @test D[1] ≈ D[2] ≈ dr[end]
        @test D[3] ≈ ar[end]
        # The bounds read the closed-form internal fractions.
        V = MeanFieldHomogenization.Schemes._layer_voigt(fe, TensISO{3}(1.0))
        @test only(TensND.get_data(V)) ≈ 0.3 * 5.0 + 0.7 * 2.0
        Rs = MeanFieldHomogenization.Schemes._layer_reuss(fe, TensISO{3}(1.0))
        @test only(TensND.get_data(Rs)) ≈ 0.3 / 5.0 + 0.7 / 2.0
        # A bound of the other physics is refused rather than silently wrong.
        @test_throws ArgumentError MeanFieldHomogenization.Schemes._layer_voigt(
            fe, _als_C(1.0, 0.3)
        )
    end

    @testset "each meshed layer has the volume its semi-axes say" begin
        # `4πa²c/3` is closed form for *any* semi-axes, confocal or not, so
        # this is the one check available on a geometry with no analytic
        # solution at all. The residual is the linear triangle's chord
        # against a curved boundary, and it falls with refinement.
        for (ar, dr) in (
                confocal_layer_radii(2.0, 1.0, (0.3, 0.7)),
                confocal_layer_radii(0.5, 1.0, (0.4, 0.6)),
                ((0.4, 1.4), (0.9, 1.0)),        # aspect reverses: no closed form
                ((0.5, 0.8, 1.0), (0.5, 0.8, 1.0)),   # spheres
            )
            n = length(ar)
            K = ntuple(_ -> TensISO{3}(1.0), n)
            fe = FEAxiLayeredSpheroid(ar, dr, K; opts = _als_opts(16, 5.0))
            b = _ALS._resolve_backend(fe.backend)
            grid = _ALS.fe_axi_grid(b, fe)
            exact = layer_volumes(fe)
            for ℓ in 1:n
                v = _ALS.fe_axi_region_volume(b, grid, axi_layer_set(ℓ))
                @test v ≈ exact[ℓ] rtol = 8.0e-3
            end
        end
    end

    @testset "a small core is meshed, not swallowed by one element" begin
        # `h_in` is set from the *outer* semi-axis, so a core an order of
        # magnitude smaller than the shell has a generous gap to its
        # neighbor and would keep the full element size — one element wider
        # than the whole core, with every gap-based bound satisfied. The cap
        # is therefore bounded by the core's own size as well.
        radii = (0.08, 1.0)
        K = (TensISO{3}(6.0), TensISO{3}(2.0))
        K₀ = TensISO{3}(1.0)
        fe = FEAxiLayeredSpheroid(radii, radii, K; opts = _als_opts(16, 5.0))
        r = fe_axi_mesh_report(fe)
        # The core has cells of its own, and its volume is right — which a
        # core swallowed by one element would fail outright.
        @test r.ncells_by_layer[1] > 20
        @test r.volume_error < 2.0e-2
        # And the answer still matches the closed form.
        Aex = _als_m(gradient_gradient_loc(LayeredSphere(radii, K), K₀, K₀))
        A = _als_m(gradient_gradient_loc(fe, K₀, K₀))
        @test norm(A - Aex) / norm(Aex) < 2.0e-3
    end

    @testset "conduction, confocal, against LayeredSpheroid" begin
        # Prolate **and** oblate: the oblate branch of the analytic solution
        # is the complex substitution, and it is the delicate half.
        for (ω, fr, tol) in (
                (2.0, (0.3, 0.7), 3.0e-4),
                (0.5, (0.4, 0.6), 5.0e-4),
                (1.8, (0.2, 0.3, 0.5), 3.0e-4),
                (0.6, (0.25, 0.35, 0.4), 5.0e-4),
            )
            ar, dr = confocal_layer_radii(ω, 1.0, fr)
            n = length(fr)
            K = ntuple(i -> TensISO{3}(1.0 + 3.0 * i), n)
            K₀ = TensISO{3}(1.0)
            Aex = _als_m(gradient_gradient_loc(LayeredSpheroid(ar, dr, K), K₀, K₀))
            fe = FEAxiLayeredSpheroid(ar, dr, K; opts = _als_opts(12, 6.0))
            A = _als_m(gradient_gradient_loc(fe, K₀, K₀))
            @test norm(A - Aex) / norm(Aex) < tol
            # Transversely isotropic about the axis, structurally.
            @test A[1, 1] ≈ A[2, 2] atol = 1.0e-12
            @test norm(A - Diagonal(diag(A))) < 1.0e-12
        end
    end

    @testset "conduction, spheres of free radii, against LayeredSphere" begin
        # Not a confocal family at all: this is what covers arbitrary radii.
        for radii in ((0.6, 1.0), (0.4, 0.7, 1.0))
            n = length(radii)
            K = ntuple(i -> TensISO{3}(1.0 + 3.0 * i), n)
            K₀ = TensISO{3}(1.0)
            Aex = _als_m(gradient_gradient_loc(LayeredSphere(radii, K), K₀, K₀))
            fe = FEAxiLayeredSpheroid(radii, radii, K; opts = _als_opts(16, 6.0))
            A = _als_m(gradient_gradient_loc(fe, K₀, K₀))
            @test norm(A - Aex) / norm(Aex) < 5.0e-4
            # A sphere's answer is isotropic, and nothing enforces that.
            @test A[1, 1] ≈ A[3, 3] rtol = 2.0e-3
        end
    end

    @testset "elasticity, both families" begin
        for (ω, fr, tol) in ((2.0, (0.3, 0.7), 3.0e-4), (0.5, (0.4, 0.6), 5.0e-4))
            ar, dr = confocal_layer_radii(ω, 1.0, fr)
            Cs = (_als_C(4.0, 0.2), _als_C(1.5, 0.3))
            C₀ = _als_C(1.0, 0.25)
            Aex = _als_k(strain_strain_loc(LayeredSpheroid(ar, dr, Cs), C₀, C₀))
            fe = FEAxiLayeredSpheroid(ar, dr, Cs; opts = _als_opts(12, 6.0))
            A = _als_k(strain_strain_loc(fe, C₀, C₀))
            @test norm(A - Aex) / norm(Aex) < tol
            # The class identity, free and structural.
            @test A[6, 6] / 2 ≈ (A[1, 1] - A[1, 2]) / 2 atol = 1.0e-10
        end
        let radii = (0.6, 1.0), C₀ = _als_C(1.0, 0.25)
            Cs = (_als_C(3.0, 0.2), _als_C(1.5, 0.3))
            ana = LayeredSphere(radii, Cs)
            Aex = _als_k(strain_strain_loc(ana, C₀, C₀))
            Bex = _als_k(stress_strain_loc(ana, C₀, C₀))
            fe = FEAxiLayeredSpheroid(radii, radii, Cs; opts = _als_opts(16, 6.0))
            @test norm(_als_k(strain_strain_loc(fe, C₀, C₀)) - Aex) / norm(Aex) < 5.0e-4
            # The stress side is measured, not derived — and it is right.
            @test norm(_als_k(stress_strain_loc(fe, C₀, C₀)) - Bex) / norm(Bex) < 5.0e-4
        end
    end

    @testset "equal moduli give the homogeneous spheroid" begin
        # A coherence limit that needs no layered reference at all, and it
        # covers a geometry the confocal family does not contain.
        ar, dr = (0.6, 1.0), (0.36, 0.6)
        K₀, k1 = TensISO{3}(1.0), TensISO{3}(4.0)
        fe = FEAxiLayeredSpheroid(ar, dr, (k1, k1); opts = _als_opts(16, 6.0))
        A = _als_m(gradient_gradient_loc(fe, K₀, K₀))
        ex = _als_m(gradient_gradient_loc(Spheroid(ar[2] / dr[2]), k1, K₀))
        @test norm(A - ex) / norm(ex) < 5.0e-4
    end

    @testset "the mesh report is the check that always exists" begin
        # `4πa²c/3` per layer is closed form for any semi-axes, so this
        # works on the nest that has no analytic solution at all — which is
        # the geometry chosen here on purpose.
        fe = FEAxiLayeredSpheroid(
            (0.4, 1.4), (0.9, 1.0), (TensISO{3}(3.0), TensISO{3}(1.5));
            opts = _als_opts(16, 4.0),
        )
        r = fe_axi_mesh_report(fe)
        @test r.ncells > 0
        @test length(r.ncells_by_layer) == 2
        @test sum(r.ncells_by_layer) + r.ncells_matrix == r.ncells
        @test r.volume_layers_exact ≈ layer_volumes(fe)
        @test r.volume_error < 8.0e-3
        @test r.volume_cell ≈ r.volume_cell_exact rtol = 2.0e-2
    end

    @testset "the R/a sweep, which is the only proof of the dipole's sign" begin
        # A wrong sign leaves *exactly twice* the truncation bias instead of
        # none, and that reads as a mesh that will not converge rather than
        # as an algebra error. The signature is unmistakable, though: the
        # uncorrected answer drifts as `(a/R)³` while the corrected one does
        # not move.
        radii = (0.6, 1.0)
        K = (TensISO{3}(5.0), TensISO{3}(2.0))
        K₀ = TensISO{3}(1.0)
        # The exact answer of the same body, so "does not move" can be
        # measured against something rather than asserted.
        Aex = _als_m(gradient_gradient_loc(LayeredSphere(radii, K), K₀, K₀))
        corr, unco = Float64[], Float64[]
        for rr in (2.5, 4.0, 6.0)
            fe = FEAxiLayeredSpheroid(
                radii, radii, K; opts = _als_opts(14, rr)
            )
            b = fe_axi_breakdown(fe, K₀)
            push!(corr, norm(_als_m(b.A) - Aex) / norm(Aex))
            push!(unco, norm(_als_m(b.A_uncorrected) - Aex) / norm(Aex))
        end
        # The uncorrected error falls with the cell radius — it is the
        # truncation the correction removes — and the corrected one is
        # already at the mesh's floor at the smallest radius.
        @test unco[1] > unco[2] > unco[3]
        @test all(<(2.0e-3), corr)
        # And the correction is worth more than an order of magnitude where
        # the cell is tight, which is the point of having it.
        @test unco[1] / corr[1] > 10
    end

    @testset "both localization tensors come from one solve" begin
        radii = (0.6, 1.0)
        Cs = (_als_C(3.0, 0.2), _als_C(1.5, 0.3))
        C₀ = _als_C(1.0, 0.25)
        fe = FEAxiLayeredSpheroid(radii, radii, Cs; opts = _als_opts(10, 4.0))
        @test fe_assembly_count(fe) == 0
        A, B = fe_axi_localization(fe, C₀)
        @test fe_assembly_count(fe) == 1
        # Calling the generics separately costs nothing more: they share the
        # memoized solve, which is what makes a family of orientations free.
        @test strain_strain_loc(fe, C₀, C₀) ≈ A
        @test stress_strain_loc(fe, C₀, C₀) ≈ B
        @test fe_assembly_count(fe) == 1
        # A different reference medium is a different solve, and only one.
        strain_strain_loc(fe, _als_C(2.0, 0.3), _als_C(2.0, 0.3))
        @test fe_assembly_count(fe) == 2
        fe_reset!(fe)
        @test fe_assembly_count(fe) == 0
    end

    @testset "it arrives in a scheme, which is the point" begin
        ar, dr = (0.6, 1.0), (0.36, 0.6)
        C₀ = _als_C(1.0, 0.25)
        Cs = (_als_C(4.0, 0.2), _als_C(2.0, 0.3))
        fe = FEAxiLayeredSpheroid(ar, dr, Cs; opts = _als_opts(10, 4.0))
        rve = RVE()
        add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
        # `IsoSymmetrize` is not optional on the iterative schemes: they
        # re-evaluate the inclusion in their own running estimate, which for
        # an oriented spheroid is transversely isotropic, and the dipole
        # boundary condition has no closed form there.
        add_phase!(
            rve, :incl, fe, Dict(:C => C₀); fraction = 0.15,
            symmetrize = IsoSymmetrize(),
        )
        res = Dict(
            nameof(typeof(s)) => k_mu(homogenize(rve, s, :C)) for s in
                (Dilute(), MoriTanaka(), SelfConsistent(), Voigt(), Reuss())
        )
        # The bounds have to bracket the estimates, and the estimates have to
        # order themselves the way these schemes always do.
        for i in 1:2
            @test res[:Reuss][i] < res[:Dilute][i] < res[:MoriTanaka][i] <
                res[:SelfConsistent][i] < res[:Voigt][i]
        end
    end
end
