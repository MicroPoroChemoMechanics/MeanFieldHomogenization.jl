using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_order2_alv.jl — order-2 (vector-tensor) ALV homogenization:
#  Heaviside elastic limit must match `homogenize` on the conductivity API.
# =============================================================================

@testset "order2_alv — iso matrix + sphere inclusion (elastic limit)" begin
    α_M = 2.0; α_I = 5.0
    times = collect(range(0.0, 1.0; length = 4))
    K_M_t = TensISO{3}(α_M)
    K_I_t = TensISO{3}(α_I)
    law_M = heaviside_law(K_M_t)
    law_I = heaviside_law(K_I_t)

    # Elastic reference
    rve_el = RVE()
    add_phase!(rve_el, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M_t); fraction = :rest)
    add_phase!(rve_el, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_I_t); fraction = 0.2)
    K_el = TensND.get_array(homogenize(rve_el, MoriTanaka(), :K))

    # ALV
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => law_M); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => law_I); fraction = 0.2)
    K_alv = homogenize_alv_order2(rve, MoriTanaka(), :K; times = times)

    n = length(times)
    @test size(K_alv) == (3 * n, 3 * n)
    for i in 1:n
        rows = (3 * (i - 1) + 1):(3 * i)
        @test isapprox(K_alv[rows, rows], K_el; atol = 1.0e-12)
        for j in 1:(i - 1)
            cols = (3 * (j - 1) + 1):(3 * j)
            @test maximum(abs, K_alv[rows, cols]) ≤ 1.0e-12
        end
    end
end

@testset "order2_alv — iso matrix + spheroid (TI elastic limit)" begin
    α_M = 2.0; α_I = 5.0
    times = collect(range(0.0, 1.0; length = 3))
    K_M_t = TensISO{3}(α_M)
    K_I_t = TensISO{3}(α_I)
    law_M = heaviside_law(K_M_t)
    law_I = heaviside_law(K_I_t)
    omega = 0.5

    rve_el = RVE()
    add_phase!(rve_el, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M_t); fraction = :rest)
    add_phase!(rve_el, :I, Spheroid(omega), Dict(:K => K_I_t); fraction = 0.3)
    K_el = TensND.get_array(homogenize(rve_el, MoriTanaka(), :K))

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => law_M); fraction = :rest)
    add_phase!(rve, :I, Spheroid(omega), Dict(:K => law_I); fraction = 0.3)
    K_alv = homogenize_alv_order2(rve, MoriTanaka(), :K; times = times)
    @test isapprox(K_alv[1:3, 1:3], K_el; atol = 1.0e-12)
end

@testset "order2_alv — Voigt / Reuss / Dilute / Maxwell elastic limit" begin
    α_M = 2.0; α_I = 5.0
    times = collect(range(0.0, 1.0; length = 4))
    K_M_t = TensISO{3}(α_M)
    K_I_t = TensISO{3}(α_I)
    law_M = heaviside_law(K_M_t)
    law_I = heaviside_law(K_I_t)
    f_I = 0.25

    function _setup_alv()
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => law_M); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => law_I); fraction = f_I)
        return rve
    end
    function _setup_el()
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M_t); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_I_t); fraction = f_I)
        return rve
    end

    for sch in (Voigt(), Reuss(), Dilute(), MoriTanaka(), Maxwell())
        K_el = TensND.get_array(homogenize(_setup_el(), sch, :K))
        K_alv = homogenize_alv_order2(_setup_alv(), sch, :K; times = times)
        @test isapprox(K_alv[1:3, 1:3], K_el; atol = 1.0e-12)
    end
end

@testset "order2_alv — iso parameter round-trip" begin
    n = 4
    α = randn(n, n)
    M = iso_order2_blocks_from_params(α)
    @test size(M) == (3n, 3n)
    @test MeanFieldHomogenization.Viscoelasticity._is_iso_order2_block(M)
    α_back = iso_order2_params_from_blocks(M)
    @test isapprox(α, α_back; atol = 1.0e-14)
end
