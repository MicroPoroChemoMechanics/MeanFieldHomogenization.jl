# =============================================================================
#  test_symmetrize.jl — round-trip tests for the orientation-distribution
#  projection helpers `_apply_symmetrize` and the RVE-level `symmetrize`
#  keyword.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import MeanFieldHomogenization.Schemes: _apply_symmetrize, _project_matrix
import ForwardDiff as FD

@testset "symmetrize" begin

    @testset "iso projection — identity on iso 4-tensor" begin
        C = TensISO{3}(216.0, 64.0)   # 216 J + 64 K
        Cp = _apply_symmetrize(C, IsoSymmetrize())
        @test Cp isa TensND.TensISO{4, 3}
        αp, βp = TensND.get_data(Cp)
        @test αp ≈ 216.0
        @test βp ≈ 64.0
    end

    @testset "iso projection — extracts (α, β) of an aniso 4-tensor" begin
        # Build a simple aniso tensor as a perturbation of an iso one.
        α0, β0 = 60.0, 24.0
        arr = zeros(3, 3, 3, 3)
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            J_iso = (i == j) * (k == l) / 3.0
            K_iso = ((i == k) * (j == l) + (i == l) * (j == k)) / 2.0 - J_iso
            arr[i, j, k, l] = α0 * J_iso + β0 * K_iso
        end
        # add an off-iso perturbation (orthorhombic-style scaling on the (3,3) axis)
        for i in 1:3, j in 1:3
            arr[i, j, 3, 3] += 1.5 * (i == j)
            arr[3, 3, i, j] += 1.5 * (i == j)
        end
        T = TensND.Tens(arr)
        Tp = _apply_symmetrize(T, IsoSymmetrize())
        @test Tp isa TensND.TensISO{4, 3}
        αp, βp = TensND.get_data(Tp)
        # Recompute analytically: α = (1/3) sum T[i,i,j,j], β = (full_trace - α)/5
        α_ref = sum(arr[i, i, j, j] for i in 1:3, j in 1:3) / 3.0
        full_trace = sum(arr[i, j, i, j] for i in 1:3, j in 1:3)
        β_ref = (full_trace - α_ref) / 5.0
        @test αp ≈ α_ref
        @test βp ≈ β_ref
    end

    @testset "iso projection — 2nd-order spherical part" begin
        K = TensND.Tens(Diagonal([1.5, 2.0, 2.5]))
        Kp = _apply_symmetrize(K, IsoSymmetrize())
        @test Kp isa TensND.TensISO{2, 3}
        @test TensND.get_data(Kp)[1] ≈ (1.5 + 2.0 + 2.5) / 3.0
    end

    @testset "TI average — identity on coaxial TI(ez) tensor" begin
        n = (0.0, 0.0, 1.0)
        # Build a TI(ez) tensor manually via TensND constructor.
        data = (1.0, 2.0, 0.5, 3.0, 4.0)   # (ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆)
        T = TensND.TensTI{4, Float64, 5}(data, n)
        Tp = _apply_symmetrize(T, TISymmetrize(n))
        # The exact azimuthal average now lives in the full 8-dim space,
        # with the extra components exactly zero for a TI(ez) input.
        @test Tp isa TensND.TensTI{4, Float64, 8}
        ℓ = TensND.get_ℓ8(Tp)
        ref = (1.0, 2.0, 0.5, 0.5, 3.0, 4.0, 0.0, 0.0)
        for (a, b) in zip(ℓ, ref)
            @test a ≈ b atol = 1.0e-10
        end
        # best-fit projection recovers the 5-component form
        bf = best_fit_ti(T, n)
        @test bf isa TensND.TensTI{4, Float64, 5}
        for (a, b) in zip(TensND.get_data(bf), data)
            @test a ≈ b atol = 1.0e-10
        end
    end

    @testset "TI average — 2nd-order axial / transverse split" begin
        n = (0.0, 0.0, 1.0)
        K = TensND.Tens(Diagonal([1.5, 2.0, 4.0]))
        Kp = _apply_symmetrize(K, TISymmetrize(n))
        @test Kp isa TensND.TensTI{2, Float64, 3}
        a, b, c = TensND.get_data(Kp)
        # axial b = K[3,3] = 4.0 ; transverse a = (K[1,1] + K[2,2]) / 2 = 1.75 ;
        # antisymmetric part c = 0 for a symmetric input
        @test b ≈ 4.0
        @test a ≈ 1.75
        @test abs(c) < 1.0e-14
        # non-symmetric input: the antisymmetric in-plane part is PRESERVED
        m = [1.5 0.6 0.0; -0.6 2.0 0.0; 0.0 0.0 4.0]
        Kp2 = _apply_symmetrize(TensND.Tens(m), TISymmetrize(n))
        a2, b2, c2 = TensND.get_data(Kp2)
        @test c2 ≈ (m[2, 1] - m[1, 2]) / 2
        @test TensND.get_array(Kp2)[1, 2] ≈ 0.6
        @test TensND.get_array(Kp2)[2, 1] ≈ -0.6
    end

    @testset "TISymmetrize — reference_projection option" begin
        sym_def = TISymmetrize((0.0, 0.0, 1.0))
        @test sym_def.reference_projection === :iso
        sym_none = TISymmetrize((0.0, 0.0, 1.0); reference_projection = :none)
        @test sym_none.reference_projection === :none
        @test_throws ArgumentError TISymmetrize((0.0, 0.0, 1.0); reference_projection = :foo)
        # _project_matrix behavior
        C_ti = TensND.TensTI{4, Float64, 5}((10.0, 6.0, 2.0, 3.0, 4.0), (0.0, 0.0, 1.0))
        @test _project_matrix(C_ti, sym_none) === C_ti
        @test _project_matrix(C_ti, sym_def) isa TensND.TensISO{4, 3}
        sym_ti = TISymmetrize((0.0, 0.0, 1.0); reference_projection = :ti)
        @test _project_matrix(C_ti, sym_ti) isa TensND.TensTI{4, Float64, 5}

        # The axis-free form: e₃ by default, and the keyword still honored.
        @test TISymmetrize().axis == (0.0, 0.0, 1.0)
        @test TISymmetrize(; reference_projection = :none).reference_projection === :none
        # And the vector form, which coerces to the tuple.
        @test TISymmetrize([0.0, 0.0, 1.0]).axis == (0.0, 0.0, 1.0)
    end

    @testset "symmetrize coercion accepts every documented spelling" begin
        S = MeanFieldHomogenization.Schemes
        @test S._to_symmetrize(nothing) isa NoSymmetrize
        @test S._to_symmetrize(:none) isa NoSymmetrize
        @test S._to_symmetrize(:iso) isa IsoSymmetrize
        @test S._to_symmetrize(:ISO) isa IsoSymmetrize
        @test S._to_symmetrize(:ti) isa TISymmetrize
        @test S._to_symmetrize(:TI) isa TISymmetrize
        @test S._to_symmetrize(IsoSymmetrize()) isa IsoSymmetrize
        @test_throws ArgumentError S._to_symmetrize(:bogus)
    end

    @testset "RVE-level symmetrize integration — porous oblate ⇒ iso C_eff" begin
        # Spherical-pore matrix with iso symmetrize on every phase: the
        # macroscopic stiffness should remain iso (TensISO) and match the
        # un-symmetrized result for sphere geometry (sanity check).
        C_s = TensISO{3}(216.0, 64.0)
        C_p = TensISO{3}(3.0e-6, 2.0e-6)
        rve = RVE()
        add_phase!(rve, :M, Spheroid(0.2), Dict(:C => C_s); fraction = :rest, symmetrize = :iso)
        add_phase!(
            rve, :PORE, Spheroid(0.2), Dict(:C => C_p);
            fraction = 0.3, symmetrize = :iso
        )
        C_eff = homogenize(rve, MoriTanaka(), :C)
        @test C_eff isa TensND.TensISO{4, 3}
        α, β = TensND.get_data(C_eff)
        @test α / 3 > 0
        @test β / 2 > 0
    end

    @testset "RVE-level symmetrize — :ti yields TI(ez) macroscopic tensor" begin
        # Oblate inclusions with TI(ez) symmetrize: the macroscopic
        # stiffness inherits a TI structure around ez (axisymmetric).
        C_m = TensISO{3}(216.0, 64.0)
        C_i = TensISO{3}(3.0e-6, 2.0e-6)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(
            rve, :I, Spheroid(0.2), Dict(:C => C_i);
            fraction = 0.2, symmetrize = TISymmetrize((0.0, 0.0, 1.0))
        )
        C_eff = homogenize(rve, MoriTanaka(), :C)
        # MT keeps the homogenized tensor in a TI subspace ; verify the
        # canonical/walpole representation has the expected axisymmetry by
        # checking C[1,1,1,1] == C[2,2,2,2] (transverse symmetry plane).
        a = TensND.get_array(C_eff)
        @test a[1, 1, 1, 1] ≈ a[2, 2, 2, 2] atol = 1.0e-8
        @test a[1, 1, 2, 2] ≈ a[2, 2, 1, 1] atol = 1.0e-8
    end

    @testset "TI(ez) average of a coaxial spheroid == NoSymmetrize (exactness)" begin
        # A spheroid whose revolution axis coincides with the TI axis is
        # already axially invariant : the exact azimuthal average must leave
        # its contribution strictly unchanged.
        C_m = TensISO{3}(3 * 20.0, 2 * 12.0)
        C_i = TensISO{3}(3 * 80.0, 2 * 50.0)
        rve1 = RVE()
        add_phase!(rve1, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve1, :I, Spheroid(5.0), Dict(:C => C_i); fraction = 0.2)
        C_ref = homogenize(rve1, MoriTanaka(), :C)
        rve2 = RVE()
        add_phase!(rve2, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(
            rve2, :I, Spheroid(5.0), Dict(:C => C_i);
            fraction = 0.2, symmetrize = TISymmetrize((0.0, 0.0, 1.0))
        )
        C_ti = homogenize(rve2, MoriTanaka(), :C)
        @test maximum(abs, TensND.get_array(C_ref) .- TensND.get_array(C_ti)) < 1.0e-10
    end

    @testset "θ-binned TI(ez) families converge to the ISO average (MT)" begin
        # Pichler-Hellmich-style discretization : tilted spheroids at polar
        # angles θ_k with solid-angle weights, each phase exactly averaged
        # about the GLOBAL axis ez (the azimuthal orbit of the orientation
        # family — echoes' `symmetrize=[TI]` semantics).  As nθ grows this
        # must converge (in O(Δθ²)) to the single ISO-symmetrized phase.
        C_m = TensISO{3}(3 * 20.0, 2 * 12.0)
        C_i = TensISO{3}(3 * 80.0, 2 * 50.0)
        rve_iso = RVE()
        add_phase!(rve_iso, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
        add_phase!(rve_iso, :I, Spheroid(5.0), Dict(:C => C_i); fraction = 0.15, symmetrize = :iso)
        Aiso = TensND.get_array(homogenize(rve_iso, MoriTanaka(), :C))
        ez = (0.0, 0.0, 1.0)
        errs = Float64[]
        for nθ in (10, 20)
            rve_bins = RVE()
            add_phase!(rve_bins, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
            edges = range(0, π / 2; length = nθ + 1)
            for k in 1:nθ
                θm, θp = edges[k], edges[k + 1]
                θ = (θm + θp) / 2
                w = cos(θm) - cos(θp)
                add_phase!(
                    rve_bins, Symbol(:B, k),
                    Spheroid(5.0; euler_angles = (θ, 0.0, 0.0)), Dict(:C => C_i);
                    fraction = 0.15 * w, symmetrize = TISymmetrize(ez)
                )
            end
            A = TensND.get_array(homogenize(rve_bins, MoriTanaka(), :C))
            push!(errs, maximum(abs, A .- Aiso) / maximum(abs, Aiso))
        end
        @test errs[1] < 2.0e-4
        @test errs[2] < 5.0e-5          # ~4× reduction: O(Δθ²) convergence
    end

    @testset "multi-axis TI phases : SC runs, ForwardDiff-compatible" begin
        # The formerly impossible configuration : several simultaneously
        # declared TI(axis_i) phases with non-coaxial axes inside the
        # generic SC iteration (used to hit a TensND axis assertion, then a
        # rank-deficient 9×9 inversion).
        C_i = TensISO{3}(3 * 80.0, 2 * 50.0)
        θs = (0.0, π / 4, π / 2)
        build = x -> begin
            rv = RVE()
            add_phase!(rv, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 20.0 * x, 2 * 12.0 * x)); fraction = :rest)
            for (k, θ) in enumerate(θs)
                add_phase!(
                    rv, Symbol(:I, k),
                    Spheroid(5.0; euler_angles = (θ, 0.0, 0.0)), Dict(:C => C_i);
                    fraction = 0.05,
                    symmetrize = TISymmetrize((sin(θ), 0.0, cos(θ)))
                )
            end
            rv
        end
        C_sc = homogenize(build(1.0), SelfConsistent(), :C)
        a = TensND.get_array(C_sc)
        @test all(isfinite, a)
        # stiffer than the matrix : C₃₃₃₃(matrix) = (α + 2β)/3 = (60 + 48)/3 = 36
        @test a[3, 3, 3, 3] > 36.0
        # ForwardDiff through the multi-axis SC (generic anisotropic running
        # estimate → NestedQuadGK Hill branch under Dual)
        f = x -> TensND.get_array(homogenize(build(x), SelfConsistent(), :C))[3, 3, 3, 3]
        g = FD.derivative(f, 1.0)
        h = 1.0e-5
        @test g ≈ (f(1.0 + h) - f(1.0 - h)) / 2h rtol = 1.0e-4
    end
end
