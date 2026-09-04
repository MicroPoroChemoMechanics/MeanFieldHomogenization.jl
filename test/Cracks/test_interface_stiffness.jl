using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_interface_stiffness.jl — Sevostianov spring-like interface
#  correction for flat cracks.  Three cases: elasticity, conductivity,
#  ALV.  Each case tests the algebraic identity, the K → 0 (free) and
#  K → ∞ (rigid bond) limits, and the dependence on the interface
#  stiffness magnitude.
# =============================================================================

@testset "Elastic crack — interface stiffness limits" begin
    C₀ = TensISO{3}(15.0, 4.0)
    crack = PennyCrack(1.0)
    B_TF = cod_tensor(crack, C₀)

    # K = 0 → recovers traction-free B
    B0 = cod_tensor(crack, C₀; K_interface = TensISO{3}(0.0))
    @test isapprox(TensND.get_array(B0), TensND.get_array(B_TF); atol = 1.0e-12)

    # K → ∞ → B_eff → 0
    B∞ = cod_tensor(crack, C₀; K_interface = TensISO{3}(1.0e12))
    @test maximum(abs, TensND.get_array(B∞)) < 1.0e-9

    # Intermediate K — strictly between 0 and B_TF
    B_K = cod_tensor(crack, C₀; K_interface = TensISO{3}(5.0))
    @test all(0 .< diag(TensND.get_array(B_K)) .< diag(TensND.get_array(B_TF)))

    # `compliance_contribution` propagates the kwarg
    H_TF = compliance_contribution(crack, C₀)
    H_K = compliance_contribution(crack, C₀; K_interface = TensISO{3}(5.0))
    H0 = compliance_contribution(crack, C₀; K_interface = TensISO{3}(0.0))
    @test isapprox(TensND.get_array(H0), TensND.get_array(H_TF); atol = 1.0e-12)
    # Interface stiffens → smaller H → smaller compliance
    @test all(abs.(TensND.get_array(H_K)) .≤ abs.(TensND.get_array(H_TF)) .+ 1.0e-12)
end

@testset "Conduction crack — interface conductance limits" begin
    K₀ = TensISO{3}(2.0)
    crack = PennyCrack(1.0)
    b_TF = cod_tensor(crack, K₀)

    @test cod_tensor(crack, K₀; α_interface = 0.0) ≈ b_TF atol = 1.0e-12
    b∞ = cod_tensor(crack, K₀; α_interface = 1.0e12)
    @test abs(b∞) < 1.0e-9
    b_α = cod_tensor(crack, K₀; α_interface = 1.0)
    @test 0 < b_α < b_TF

    R_TF = compliance_contribution(crack, K₀)
    R_α = compliance_contribution(crack, K₀; α_interface = 1.0)
    @test maximum(abs, TensND.get_array(R_α)) ≤
        maximum(abs, TensND.get_array(R_TF)) + 1.0e-12
end

@testset "ALV crack — interface ViscoLaw limits" begin
    times = collect(range(0.0, 2.0; length = 8))

    R_M(t, tp) = TensISO{3}(
        3 * (3.0 + 2.0 * exp(-(t - tp))),
        2 * (1.0 + exp(-(t - tp) / 0.5))
    )
    law_M = ViscoLaw(R_M, :relaxation)

    crack = PennyCrack(1.0)
    cod0 = MeanFieldHomogenization.Viscoelasticity.cod_kernel_alv(crack, law_M, times)

    # Rn = Rt = 0 → recovers traction-free
    law_zero = ViscoLaw((t, tp) -> 0.0, :relaxation)
    cod_zero = MeanFieldHomogenization.Viscoelasticity.cod_kernel_alv(
        crack, law_M, times;
        Rn = law_zero, Rt = law_zero
    )
    @test isapprox(cod_zero.B_n, cod0.B_n; atol = 1.0e-10)
    @test isapprox(cod_zero.B_t, cod0.B_t; atol = 1.0e-10)

    # Rn, Rt → ∞ → B_eff → 0
    law_huge = ViscoLaw((t, tp) -> 1.0e10, :relaxation)
    cod_huge = MeanFieldHomogenization.Viscoelasticity.cod_kernel_alv(
        crack, law_M, times;
        Rn = law_huge, Rt = law_huge
    )
    @test maximum(abs, cod_huge.B_n) < 1.0e-8
    @test maximum(abs, cod_huge.B_t) < 1.0e-8

    # Intermediate stiffness: 0 < B_n_eff < B_n_TF (componentwise on the diag)
    law_K = ViscoLaw((t, tp) -> 1.0, :relaxation)
    cod_K = MeanFieldHomogenization.Viscoelasticity.cod_kernel_alv(
        crack, law_M, times;
        Rn = law_K, Rt = law_K
    )
    for i in 1:length(times)
        @test 0 < abs(cod_K.B_n[i, i]) < abs(cod0.B_n[i, i])
    end

    # Algebraic identity check: (b·K + B^{-1})^{-vol} ≈ B ∘ (𝟙 + b·K·B)^{-vol}
    K_M = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation_scalar(law_K, times)
    B_n = cod0.B_n
    n = size(B_n, 1)
    Iₙ = Matrix{Float64}(LinearAlgebra.I, n, n)
    b = MeanFieldHomogenization.Cracks.semi_minor(crack)   # = 1.0 for penny
    # Form 1: (b·K + B^{-1})^{-vol}
    B_inv = volterra_inverse(B_n; block_size = 1)
    sum_form = b .* K_M .+ B_inv
    eff_1 = volterra_inverse(sum_form; block_size = 1)
    # Form 2: B ∘ (𝟙 + b·K·B)^{-vol}
    KB = K_M * B_n
    eff_2 = B_n * volterra_inverse(Iₙ .+ b .* KB; block_size = 1)
    @test isapprox(eff_1, eff_2; atol = 1.0e-10)
    @test isapprox(eff_2, cod_K.B_n; atol = 1.0e-10)
end

@testset "ALV crack interface — homogenize_alv end-to-end" begin
    times = collect(range(0.0, 2.0; length = 8))
    R_M(t, tp) = TensISO{3}(
        3 * (3.0 + 2.0 * exp(-(t - tp))),
        2 * (1.0 + exp(-(t - tp) / 0.5))
    )
    law_M = ViscoLaw(R_M, :relaxation)
    law_K = ViscoLaw((t, tp) -> 1.0 + 0.5 * exp(-(t - tp)), :relaxation)

    # Two RVEs with same crack density, one TF, one with finite Rn = Rt = law_K
    rve_TF = RVE()
    add_phase!(rve_TF, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        rve_TF, :CRACK, PennyCrack(1.0), Dict(:C => law_M);
        density = 0.05, symmetrize = :iso
    )

    rve_IS = RVE()
    add_phase!(rve_IS, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        rve_IS, :CRACK, PennyCrack(1.0),
        Dict(:C => law_M, :Rn => law_K, :Rt => law_K);
        density = 0.05, symmetrize = :iso
    )

    C_TF = homogenize_alv(rve_TF, MoriTanaka(), :C; times = times)
    C_IS = homogenize_alv(rve_IS, MoriTanaka(), :C; times = times)
    _, β_TF = iso_params_from_blocks(C_TF)
    _, β_IS = iso_params_from_blocks(C_IS)

    # Interface stiffens the composite : 2μ_eff(t_n, t_n) ≥ 2μ_TF
    @test β_IS[end, end] > β_TF[end, end]
    # Same for all (t, t') diagonal entries
    for i in 1:length(times)
        @test β_IS[i, i] ≥ β_TF[i, i] - 1.0e-12
    end

    # K → ∞ recovers the matrix without crack effect
    law_huge = ViscoLaw((t, tp) -> 1.0e10, :relaxation)
    rve_rigid = RVE()
    add_phase!(rve_rigid, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        rve_rigid, :CRACK, PennyCrack(1.0),
        Dict(:C => law_M, :Rn => law_huge, :Rt => law_huge);
        density = 0.05, symmetrize = :iso
    )
    C_rigid = homogenize_alv(rve_rigid, MoriTanaka(), :C; times = times)
    C_M = trapezoidal_matrix(law_M, times)
    @test isapprox(C_rigid, C_M; rtol = 1.0e-6)   # crack contribution → 0
end
