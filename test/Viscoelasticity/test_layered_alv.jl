using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_layered_alv.jl — n-layer composite sphere in an ALV matrix.
#
#  Covers (1) the bulk (Y₀) recurrence, (2) the shear (Y₂) recurrence,
#  and (3) ALV interface transfers (perfect, spring, membrane) for both
#  harmonics.
# =============================================================================

@testset "bulk_localization_alv — elastic limit (Heaviside law)" begin
    C_M = TensISO{3}(30.0, 8.0)   # k = 10, μ = 4 — matrix
    C_1 = TensISO{3}(60.0, 16.0)  # k = 20, μ = 8 — core
    C_2 = TensISO{3}(90.0, 24.0)  # k = 30, μ = 12 — shell
    sphere = LayeredSphere((0.5, 1.0), (C_1, C_2))
    times = collect(0.0:0.25:1.0)
    n = length(times)

    α_alv = bulk_localization_alv(sphere, heaviside_law(C_M), times)
    @test length(α_alv) == 2
    @test all(size(α) == (n, n) for α in α_alv)

    # Reference elastic α_k from the existing recurrence.
    α_elas = MeanFieldHomogenization.LayeredSpheres._bulk_localization(sphere, 10.0, 4.0)

    for k in 1:2
        # Diagonal entries match the elastic constant.
        diag_α = [α_alv[k][i, i] for i in 1:n]
        @test maximum(abs.(diag_α .- α_elas[k])) ≤ 1.0e-12
        # Off-diagonal entries are zero (no history coupling for elastic limit).
        for i in 1:n, j in 1:n
            if i != j
                @test α_alv[k][i, j] == 0.0
            end
        end
    end
end

@testset "bulk_localization_alv — single-layer sphere ≡ Eshelby bulk α" begin
    # Single layer = single sphere ; bulk α should equal the Eshelby
    # bulk localization factor for a sphere (k_inc) in matrix (k_0, μ_0).
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    sphere = LayeredSphere((1.0,), (C_1,))
    times = collect(0.0:0.5:1.0)

    α_alv = bulk_localization_alv(sphere, heaviside_law(C_M), times)
    α_elas = MeanFieldHomogenization.LayeredSpheres._bulk_localization(sphere, 10.0, 4.0)
    diag_α = [α_alv[1][i, i] for i in 1:length(times)]
    @test maximum(abs.(diag_α .- α_elas[1])) ≤ 1.0e-12
end

@testset "bulk_localization_alv — Maxwell relaxation kernel structure" begin
    # Matrix is a Maxwell relaxation ; layers are elastic.  At t = 0 the
    # response is elastic-instantaneous ; at later t' < t the kernel has
    # decayed and the localization factor changes accordingly.
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    sphere = LayeredSphere((0.5, 1.0), (C_1, C_2))
    times = collect(0.0:0.25:1.0)
    n = length(times)

    law = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    α_alv = bulk_localization_alv(sphere, law, times)
    @test length(α_alv) == 2
    @test all(size(α) == (n, n) for α in α_alv)
    # Lower-triangular structure (causality).
    for k in 1:2
        for i in 1:n, j in 1:n
            if j > i
                @test abs(α_alv[k][i, j]) ≤ 1.0e-12
            end
        end
    end
end

@testset "bulk_state_seq_alv — Spring + Perfect interface stack (elastic limit)" begin
    # Elastic limit : Heaviside matrix kernel, scalar-spring interface, perfect outer.
    # The ALV bulk α_k under a Spring interface must coincide diagonal-wise with
    # the elastic α_k computed by `_bulk_localization` of the same sphere.
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    intf = SpringInterface(0.05)
    sphere = LayeredSphere(
        (0.5, 1.0), (C_1, C_2);
        interfaces = (intf, PerfectInterface())
    )
    times = collect(0.0:0.5:1.0)

    α_alv = bulk_localization_alv(sphere, heaviside_law(C_M), times)
    α_elas = MeanFieldHomogenization.LayeredSpheres._bulk_localization(sphere, 10.0, 4.0)
    for k in 1:2
        diag_α = [α_alv[k][i, i] for i in 1:length(times)]
        @test maximum(abs.(diag_α .- α_elas[k])) ≤ 1.0e-12
    end
end

@testset "bulk_state_seq_alv — Membrane interface (elastic limit)" begin
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    intf = MembraneInterface(0.1, 0.05)
    sphere = LayeredSphere(
        (0.5, 1.0), (C_1, C_2);
        interfaces = (intf, PerfectInterface())
    )
    times = collect(0.0:0.5:1.0)

    α_alv = bulk_localization_alv(sphere, heaviside_law(C_M), times)
    α_elas = MeanFieldHomogenization.LayeredSpheres._bulk_localization(sphere, 10.0, 4.0)
    for k in 1:2
        diag_α = [α_alv[k][i, i] for i in 1:length(times)]
        @test maximum(abs.(diag_α .- α_elas[k])) ≤ 1.0e-12
    end
end

# ── Shear (Y₂) recurrence ──────────────────────────────────────────────────

@testset "shear_localization_alv — elastic limit (Heaviside law)" begin
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    sphere = LayeredSphere((0.5, 1.0), (C_1, C_2))
    times = collect(0.0:0.25:1.0)
    n = length(times)

    β_alv = MeanFieldHomogenization.Viscoelasticity.shear_localization_alv(
        sphere, heaviside_law(C_M), times
    )
    @test length(β_alv) == 2
    @test all(size(β) == (n, n) for β in β_alv)

    β_elas = MeanFieldHomogenization.LayeredSpheres._shear_localization(sphere, C_M)
    for k in 1:2
        diag_β = [β_alv[k][i, i] for i in 1:n]
        @test maximum(abs.(diag_β .- β_elas[k])) ≤ 1.0e-10
        # Off-diagonal entries vanish for the Heaviside (memory-less) limit.
        for i in 1:n, j in 1:n
            if i != j
                @test abs(β_alv[k][i, j]) ≤ 1.0e-10
            end
        end
    end
end

@testset "shear_localization_alv — single-layer sphere ≡ Eshelby shear β" begin
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    sphere = LayeredSphere((1.0,), (C_1,))
    times = collect(0.0:0.5:1.0)

    β_alv = MeanFieldHomogenization.Viscoelasticity.shear_localization_alv(
        sphere, heaviside_law(C_M), times
    )
    β_elas = MeanFieldHomogenization.LayeredSpheres._shear_localization(sphere, C_M)
    diag_β = [β_alv[1][i, i] for i in 1:length(times)]
    @test maximum(abs.(diag_β .- β_elas[1])) ≤ 1.0e-10
end

@testset "shear_localization_alv — Maxwell kernel triangular structure" begin
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    sphere = LayeredSphere((0.5, 1.0), (C_1, C_2))
    times = collect(0.0:0.25:1.0)
    n = length(times)
    law = maxwell_iso(10.0, 4.0, 1.0, 0.5)

    β_alv = MeanFieldHomogenization.Viscoelasticity.shear_localization_alv(sphere, law, times)
    @test length(β_alv) == 2
    @test all(size(β) == (n, n) for β in β_alv)
    for k in 1:2
        for i in 1:n, j in 1:n
            if j > i
                @test abs(β_alv[k][i, j]) ≤ 1.0e-10
            end
        end
    end
end

@testset "shear_localization_alv — Spring interface (elastic limit)" begin
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    intf = SpringInterface(0.05, 0.07)
    sphere = LayeredSphere(
        (0.5, 1.0), (C_1, C_2);
        interfaces = (intf, PerfectInterface())
    )
    times = collect(0.0:0.5:1.0)

    β_alv = MeanFieldHomogenization.Viscoelasticity.shear_localization_alv(
        sphere, heaviside_law(C_M), times
    )
    β_elas = MeanFieldHomogenization.LayeredSpheres._shear_localization(sphere, C_M)
    for k in 1:2
        diag_β = [β_alv[k][i, i] for i in 1:length(times)]
        @test maximum(abs.(diag_β .- β_elas[k])) ≤ 1.0e-10
    end
end

@testset "shear_localization_alv — Membrane interface (elastic limit)" begin
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    intf = MembraneInterface(0.1, 0.05)
    sphere = LayeredSphere(
        (0.5, 1.0), (C_1, C_2);
        interfaces = (intf, PerfectInterface())
    )
    times = collect(0.0:0.5:1.0)

    β_alv = MeanFieldHomogenization.Viscoelasticity.shear_localization_alv(
        sphere, heaviside_law(C_M), times
    )
    β_elas = MeanFieldHomogenization.LayeredSpheres._shear_localization(sphere, C_M)
    for k in 1:2
        diag_β = [β_alv[k][i, i] for i in 1:length(times)]
        @test maximum(abs.(diag_β .- β_elas[k])) ≤ 1.0e-10
    end
end

# ── stiffness_contribution_alv & strain_strain_loc_alv (composite assembly) ─

@testset "stiffness_contribution_alv — elastic limit ≡ stiffness_contribution" begin
    C_M = TensISO{3}(30.0, 8.0)    # κ = 10, μ = 4
    C_1 = TensISO{3}(60.0, 16.0)   # κ = 20, μ = 8
    C_2 = TensISO{3}(90.0, 24.0)   # κ = 30, μ = 12
    sphere = LayeredSphere((0.5, 1.0), (C_1, C_2))
    times = collect(0.0:0.25:1.0)
    n = length(times)

    N_alv = stiffness_contribution_alv(sphere, heaviside_law(C_M), times)
    @test size(N_alv) == (6 * n, 6 * n)

    # Reference elastic stiffness contribution.
    N_elas = stiffness_contribution(sphere, C_M)
    N_elas_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(N_elas)

    # Diagonal 6×6 blocks must match the elastic 6×6, off-diag must vanish.
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test isapprox(N_alv[rows, rows], N_elas_M; rtol = 1.0e-10, atol = 1.0e-10)
        for j in 1:(i - 1)
            cols = (6 * (j - 1) + 1):(6 * j)
            @test maximum(abs, N_alv[rows, cols]) ≤ 1.0e-10
        end
    end
end

@testset "stiffness_contribution_alv — Membrane (DUALDISC) surface stress" begin
    # The ALV stiffness contribution must include the Gurtin–Murdoch surface
    # stress of a dual interface; in the elastic (Heaviside) limit it must
    # equal the elastic `stiffness_contribution`, which is validated against
    # Echoes' `DUALDISC`.  Single-layer aggregate (the N=1 composite-sphere
    # path) + membrane, and a 2-layer sphere with an interior membrane.
    C_M = iso_stiffness_E_nu(30.0, 0.3)
    Cagg = iso_stiffness_E_nu(70.0, 0.2)
    C_2 = iso_stiffness_E_nu(40.0, 0.25)
    times = collect(0.0:0.5:1.0)
    n = length(times)
    spheres = (
        LayeredSphere((1.0,), (Cagg,); interfaces = (MembraneInterface(5.0, 3.0),)),
        LayeredSphere(
            (1.0, 1.5), (Cagg, C_2);
            interfaces = (MembraneInterface(4.0, 2.0), PerfectInterface())
        ),
    )
    for sphere in spheres
        N_alv = stiffness_contribution_alv(sphere, heaviside_law(C_M), times)
        N_elas_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(
            stiffness_contribution(sphere, C_M)
        )
        for i in 1:n
            rows = (6 * (i - 1) + 1):(6 * i)
            @test isapprox(N_alv[rows, rows], N_elas_M; rtol = 1.0e-9, atol = 1.0e-9)
        end
    end
end

# ── homogenize_alv with a LayeredSphere phase (Dilute / MT) ────────────────

@testset "homogenize_alv — LayeredSphere phase, Dilute, elastic limit" begin
    C_M = TensISO{3}(30.0, 8.0)
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    sphere = LayeredSphere((0.5, 1.0), (C_1, C_2))
    times = collect(0.0:0.25:1.0)
    n = length(times)
    f_I = 0.2

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_M)); fraction = :rest)
    add_phase!(
        rve, :I, sphere, Dict(:C => heaviside_law(C_M));
        fraction = f_I
    )

    C_alv = homogenize_alv(rve, Dilute(), :C; times = times)
    # Reference elastic dilute.
    N_elas = stiffness_contribution(sphere, C_M)
    C_eff_elas = C_M + f_I * N_elas
    C_eff_elas_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(C_eff_elas)

    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test isapprox(
            C_alv[rows, rows], C_eff_elas_M;
            rtol = 1.0e-10, atol = 1.0e-10
        )
        for j in 1:(i - 1)
            cols = (6 * (j - 1) + 1):(6 * j)
            @test maximum(abs, C_alv[rows, cols]) ≤ 1.0e-10
        end
    end
end

@testset "shear_localization_alv — N=2 cross-check vs ECHOES Python" begin
    # Reference values produced by `scripts/bench_echoes/bench_layered_alv.py`
    # (ECHOES Python `sphere_nlayers.layer_visco_eE` with `visco_paramsym(_, ISO)`).
    # Setup: Maxwell matrix, two elastic Heaviside layers.
    k0 = 1.0; mu0 = 0.5
    eta_k0 = 0.6; eta_mu0 = 2.0
    k1, mu1 = 2.0, 1.0   # core
    k2, mu2 = 3.0, 1.5   # shell
    sphere = LayeredSphere(
        (0.5^(1 / 3), 1.0),
        (
            heaviside_law(TensISO{3}(3 * k1, 2 * mu1)),
            heaviside_law(TensISO{3}(3 * k2, 2 * mu2)),
        )
    )
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    C0_law = maxwell_iso(k0, mu0, eta_k0, eta_mu0)

    α_jl = bulk_localization_alv(sphere, C0_law, times)
    β_jl = MeanFieldHomogenization.Viscoelasticity.shear_localization_alv(sphere, C0_law, times)

    # ECHOES Python diagonal (== single-time elastic limit) reference values.
    α_py_diag_core = [0.5952381, 0.4792996, 0.4792996, 0.4792996, 0.4792996]
    β_py_diag_core = [0.650383, 0.5938882, 0.5938882, 0.5938882, 0.5938882]
    α_py_diag_shell = [0.4761905, 0.3834397, 0.3834397, 0.3834397, 0.3834397]
    β_py_diag_shell = [0.5293026, 0.4816787, 0.4816787, 0.4816787, 0.4816787]

    for i in 1:5
        @test α_jl[1][i, i] ≈ α_py_diag_core[i]  rtol = 1.0e-6
        @test β_jl[1][i, i] ≈ β_py_diag_core[i]  rtol = 1.0e-6
        @test α_jl[2][i, i] ≈ α_py_diag_shell[i] rtol = 1.0e-6
        @test β_jl[2][i, i] ≈ β_py_diag_shell[i] rtol = 1.0e-6
    end
end

@testset "homogenize_alv — LayeredSphere phase, Maxwell relaxation kernel" begin
    # Maxwell-relaxation matrix + elastic layered sphere ; checks that the
    # effective stiffness is causal (lower-triangular block structure) and
    # has the right t = t = 0 limit.
    C_1 = TensISO{3}(60.0, 16.0)
    C_2 = TensISO{3}(90.0, 24.0)
    sphere = LayeredSphere((0.5, 1.0), (C_1, C_2))
    times = collect(0.0:0.25:1.0)
    n = length(times)

    law_M = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(rve, :I, sphere, Dict(:C => law_M); fraction = 0.2)

    C_alv = homogenize_alv(rve, Dilute(), :C; times = times)
    # Causality : upper-triangular blocks must vanish.
    for i in 1:n, j in (i + 1):n
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        @test maximum(abs, C_alv[rows, cols]) ≤ 1.0e-10
    end
    # t = t = 0 block ≡ elastic dilute with κ = 10, μ = 4.
    C_M_t0 = TensISO{3}(30.0, 8.0)
    N_elas_t0 = stiffness_contribution(sphere, C_M_t0)
    C_eff_t0 = C_M_t0 + 0.2 * N_elas_t0
    block11 = C_alv[1:6, 1:6]
    @test isapprox(
        block11,
        MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(C_eff_t0);
        rtol = 1.0e-10, atol = 1.0e-10
    )
end
