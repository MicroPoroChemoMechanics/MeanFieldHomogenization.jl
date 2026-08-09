# =============================================================================
#  test_laminate_alv.jl — the periodic multilayer in ageing viscoelasticity.
#
#  The decisive oracle is the ELASTIC LIMIT: feeding each layer a Heaviside
#  law built on its elastic stiffness must reproduce, in every diagonal time
#  block, the effective stiffness of the elastic laminate. That single check
#  pins the whole Volterra transposition — the out-of-plane index selection
#  inside each time block, the `volterra_inverse` substitution for the 3×3
#  cofactor inverse, and the interface terms — against a result already
#  validated in closed form against Backus (1962).
#
#  Further coverage: N = 1 degeneracy, Voigt/Reuss ALV bracketing, genuine
#  ageing behavior (a creeping layer), conduction at order 2, and the
#  interaction with `has_visco_property`.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const ATOL_LALV = 1.0e-9

_isoa(k, μ) = TensISO{3}(3k, 2μ)

# Diagonal time block (t_i, t_i) of an ALV operator, as a 6×6 Mandel matrix.
_blk(M, i, nc = 6) = M[(nc * (i - 1) + 1):(nc * i), (nc * (i - 1) + 1):(nc * i)]

function _elastic_laminate(; kn = 0.0, κs = 0.0)
    lam = Laminate(; normal = (0, 0, 1))
    itf1 = kn == 0 ? PerfectInterface() : SpringInterface(kn, kn / 2)
    itf2 = κs == 0 ? PerfectInterface() : MembraneInterface(κs, κs / 2)
    add_layer!(lam, :A, Dict(:C => _isoa(2.0, 0.8)); thickness = 0.3, interface = itf1)
    add_layer!(lam, :B, Dict(:C => _isoa(0.5, 0.2)); thickness = 0.7, interface = itf2)
    return lam
end

function _heaviside_laminate(; kn = 0.0, κs = 0.0)
    lam = Laminate(; normal = (0, 0, 1))
    itf1 = kn == 0 ? PerfectInterface() : SpringInterface(kn, kn / 2)
    itf2 = κs == 0 ? PerfectInterface() : MembraneInterface(κs, κs / 2)
    add_layer!(
        lam, :A, Dict(:C => heaviside_law(_isoa(2.0, 0.8)));
        thickness = 0.3, interface = itf1
    )
    add_layer!(
        lam, :B, Dict(:C => heaviside_law(_isoa(0.5, 0.2)));
        thickness = 0.7, interface = itf2
    )
    return lam
end

@testset "ALV laminate — the elastic limit" begin
    times = [0.0, 1.0, 3.0, 7.0]
    C_el = Matrix(KM(homogenize(_elastic_laminate(), Laminated(), :C)))
    M = homogenize_alv(_heaviside_laminate(), Laminated(), :C; times = times)

    @test size(M) == (6 * length(times), 6 * length(times))
    # The package convention for an elastic limit (see `_check_alv_elastic`
    # in `test_schemes_alv.jl`): the Volterra operator is block-DIAGONAL,
    # each diagonal block being the elastic tensor — an elastic material has
    # no memory, so no earlier time contributes.
    for i in eachindex(times)
        @test _blk(M, i) ≈ C_el atol = ATOL_LALV
        for j in 1:(i - 1)
            @test maximum(
                abs, M[(6 * (i - 1) + 1):(6 * i), (6 * (j - 1) + 1):(6 * j)]
            ) ≤ ATOL_LALV
        end
    end
end

@testset "ALV laminate — the elastic limit with interfaces" begin
    times = [0.0, 1.0, 4.0]
    for (kn, κs) in ((5.0e-2, 0.0), (0.0, 0.07), (5.0e-2, 0.07))
        C_el = Matrix(KM(homogenize(_elastic_laminate(; kn = kn, κs = κs), Laminated(), :C)))
        M = homogenize_alv(_heaviside_laminate(; kn = kn, κs = κs), Laminated(), :C; times = times)
        for i in eachindex(times)
            @test _blk(M, i) ≈ C_el atol = ATOL_LALV
        end
    end
end

@testset "ALV laminate — N = 1 degeneracy" begin
    times = [0.0, 2.0, 5.0]
    law = maxwell_relaxation(_isoa(1.0, 0.4), [_isoa(1.0, 0.4)], [2.0])
    lam = Laminate()
    add_layer!(lam, :A, Dict(:C => law); thickness = 1.0)
    M = homogenize_alv(lam, Laminated(), :C; times = times)
    M1 = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation(law, times, 6)
    @test M ≈ M1 atol = ATOL_LALV
end

@testset "ALV laminate — genuine ageing, bounded by Voigt / Reuss" begin
    times = collect(range(0.0, 10.0; length = 6))
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(
        lam, :A,
        Dict(:C => maxwell_relaxation(_isoa(2.0, 0.8), [_isoa(1.0, 0.4)], [3.0]));
        thickness = 0.4
    )
    add_layer!(lam, :B, Dict(:C => heaviside_law(_isoa(0.5, 0.2))); thickness = 0.6)

    M = homogenize_alv(lam, Laminated(), :C; times = times)
    Mv = homogenize_alv(lam, Voigt(), :C; times = times)
    Mr = homogenize_alv(lam, Reuss(), :C; times = times)

    # The stack relaxes: the instantaneous response is stiffer than the
    # long-term one, on every component.
    @test _blk(M, 1)[3, 3] > _blk(M, length(times))[3, 3]
    @test _blk(M, 1)[6, 6] > _blk(M, length(times))[6, 6]

    for i in eachindex(times)
        # Bracketed by the ALV bounds …
        @test _blk(Mr, i)[3, 3] ≤ _blk(M, i)[3, 3] + ATOL_LALV
        @test _blk(M, i)[3, 3] ≤ _blk(Mv, i)[3, 3] + ATOL_LALV
        # … and, exactly as in the elastic case, the in-plane shear IS Voigt
        # and the out-of-plane response IS Reuss.
        @test _blk(M, i)[6, 6] ≈ _blk(Mv, i)[6, 6] atol = ATOL_LALV
        @test _blk(M, i)[3, 3] ≈ _blk(Mr, i)[3, 3] atol = ATOL_LALV
    end
end

@testset "ALV laminate — conduction (order 2)" begin
    times = [0.0, 1.0, 3.0]
    K_A, K_B = TensISO{3}(2.0), TensISO{3}(0.3)

    lam_el = Laminate(; normal = (0, 0, 1))
    add_layer!(lam_el, :A, Dict(:K => K_A); thickness = 0.3)
    add_layer!(lam_el, :B, Dict(:K => K_B); thickness = 0.7)
    K_el = Matrix(components(homogenize(lam_el, Laminated(), :K)))

    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:K => heaviside_law(K_A)); thickness = 0.3)
    add_layer!(lam, :B, Dict(:K => heaviside_law(K_B)); thickness = 0.7)
    M = homogenize_alv(lam, Laminated(), :K; times = times)

    @test size(M) == (3 * length(times), 3 * length(times))
    for i in eachindex(times)
        @test _blk(M, i, 3) ≈ K_el atol = ATOL_LALV
    end
    # The exact series law survives the Volterra transposition.
    @test 1 / _blk(M, 1, 3)[3, 3] ≈ 0.3 / 2.0 + 0.7 / 0.3 atol = ATOL_LALV
end

@testset "ALV laminate — Kapitza interface in the elastic limit" begin
    times = [0.0, 2.0]
    ρ = 0.11
    lam_el = Laminate(; normal = (0, 0, 1))
    add_layer!(
        lam_el, :A, Dict(:K => TensISO{3}(2.0)); thickness = 0.3,
        interface = KapitzaInterface(ρ)
    )
    add_layer!(lam_el, :B, Dict(:K => TensISO{3}(0.3)); thickness = 0.7)
    K_el = Matrix(components(homogenize(lam_el, Laminated(), :K)))

    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(
        lam, :A, Dict(:K => heaviside_law(TensISO{3}(2.0))); thickness = 0.3,
        interface = KapitzaInterface(ρ)
    )
    add_layer!(lam, :B, Dict(:K => heaviside_law(TensISO{3}(0.3))); thickness = 0.7)
    M = homogenize_alv(lam, Laminated(), :K; times = times)
    @test _blk(M, 1, 3) ≈ K_el atol = ATOL_LALV
    @test 1 / _blk(M, 1, 3)[3, 3] ≈ 0.3 / 2.0 + 0.7 / 0.3 + ρ / 1.0 atol = ATOL_LALV
end

@testset "ALV laminate — has_visco_property on a Laminate" begin
    lam = _heaviside_laminate()
    @test has_visco_property(lam, :C)
    @test !has_visco_property(_elastic_laminate(), :C)
    # ... and through a declaratively nested cell, without resolving it.
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => heaviside_law(_isoa(1.0, 0.4))))
    add_phase!(
        rve, :agg, Ellipsoid(1.0),
        Dict(:C => Homogenized(_heaviside_laminate(), Laminated())); fraction = 0.2
    )
    @test has_visco_property(rve, :C)
end
