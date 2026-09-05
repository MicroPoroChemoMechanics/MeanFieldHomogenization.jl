using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import MeanFieldHomogenization.Viscoelasticity: _is_ortho_block, _is_ti_block, _is_iso_block,
    _try_iso_pairs, _try_ti_tuples, _try_ortho_tuples, _ortho_pair, _ortho_blocks

# =============================================================================
#  test_ortho_dispatch_alv.jl — the ORTHO fast paths of
#  `_homogenize_alv_dispatch` (`src/Viscoelasticity/homogenize_alv.jl`).
#
#  `test_ortho_alv.jl` exercises the ortho primitives (`voigt_alv_ortho`, …) by
#  calling them directly, but every RVE it builds is isotropic: since
#  iso ⊂ TI ⊂ ortho, the dispatcher then takes the iso shortcut and the ortho
#  branches of `_homogenize_alv_dispatch` are never executed.
#
#  This file builds a genuinely orthotropic RVE (neither iso nor TI) to force
#  `_try_iso_pairs` and `_try_ti_tuples` to return `nothing`, so that the
#  dispatcher falls through to `_try_ortho_tuples`.
# =============================================================================

const _ORTHO_FRAME = TensND.CanonicalBasis{3, Float64}()

# A genuinely orthotropic tensor: the three normal moduli and the three shear
# moduli are all distinct, so it is neither isotropic nor TI about any axis.
_ortho_tensor(s) = TensND.TensOrtho(
    20.0s, 8.0s, 6.0s,       # C11, C12, C13
    30.0s, 7.0s, 40.0s,      # C22, C23, C33
    5.0s, 6.0s, 7.0s,        # C44, C55, C66
    _ORTHO_FRAME
)

function _ortho_rve(; fraction = 0.25)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(_ortho_tensor(1.0))); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(_ortho_tensor(2.0)));
        fraction = fraction
    )
    return rve
end

@testset "ortho dispatch — le RVE est bien hors des raccourcis iso et TI" begin
    times = collect(0.0:0.5:1.0)
    M_M = trapezoidal_matrix(heaviside_law(_ortho_tensor(1.0)), times)
    M_I = trapezoidal_matrix(heaviside_law(_ortho_tensor(2.0)), times)

    # Precondition of this test: without it the dispatcher would take the iso
    # or the TI path and the branches under test would stay dead.
    @test !_is_iso_block(M_M)
    @test !_is_ti_block(M_M)
    @test _is_ortho_block(M_M)

    @test _try_iso_pairs([M_M, M_I]) === nothing
    @test _try_ti_tuples([M_M, M_I]) === nothing

    o = _try_ortho_tuples([M_M, M_I])
    @test o !== nothing
    @test length(o) == 2
    @test length(o[1]) == 12

    # `_try_ortho_tuples` must return `nothing` as soon as a single matrix
    # leaves the ortho form.
    M_bad = copy(M_M)
    M_bad[1, 4] += 1.0
    @test !_is_ortho_block(M_bad)
    @test _try_ortho_tuples([M_M, M_bad]) === nothing

    # Cas vide : vecteur vide, pas `nothing`.
    empty_in = Matrix{Float64}[]
    @test _try_ortho_tuples(empty_in) == NTuple{12, Matrix{Float64}}[]
    @test _try_ti_tuples(empty_in) == NTuple{6, Matrix{Float64}}[]
    @test _try_iso_pairs(empty_in) == Tuple{Matrix{Float64}, Matrix{Float64}}[]
end

@testset "ortho dispatch — Voigt et Reuss passent par le chemin ortho" begin
    times = collect(0.0:0.5:1.5)
    n = length(times)
    f = 0.25
    rve = _ortho_rve(fraction = f)

    M_M = trapezoidal_matrix(heaviside_law(_ortho_tensor(1.0)), times)
    M_I = trapezoidal_matrix(heaviside_law(_ortho_tensor(2.0)), times)
    fr = [1 - f, f]

    C_voigt = homogenize_alv(rve, Voigt(), :C; times = times)
    @test size(C_voigt) == (6n, 6n)
    @test _is_ortho_block(C_voigt)
    @test !_is_ti_block(C_voigt)                 # on est bien resté ortho
    @test isapprox(C_voigt, voigt_alv([M_M, M_I], fr); atol = 1.0e-10)

    C_reuss = homogenize_alv(rve, Reuss(), :C; times = times)
    @test _is_ortho_block(C_reuss)
    @test isapprox(C_reuss, reuss_alv([M_M, M_I], fr); atol = 1.0e-8)

    # Encadrement de Voigt-Reuss sur les termes diagonaux.
    for i in 1:(6n)
        @test C_reuss[i, i] ≤ C_voigt[i, i] + 1.0e-9
    end
end

@testset "ortho dispatch — Dilute, Mori-Tanaka et Maxwell restent ortho" begin
    times = collect(0.0:0.5:1.5)
    n = length(times)
    rve = _ortho_rve(fraction = 0.2)

    for scheme in (Dilute(), MoriTanaka(), Maxwell())
        C = homogenize_alv(rve, scheme, :C; times = times)
        @test size(C) == (6n, 6n)
        @test all(isfinite, C)
        # These schemes preserve the ortho form: coaxial ortho phases give an
        # ortho result in the same material frame.
        @test _is_ortho_block(C)
        @test !_is_iso_block(C)
    end
end

@testset "ortho dispatch — fraction nulle redonne la matrice" begin
    times = collect(0.0:0.5:1.0)
    rve = _ortho_rve(fraction = 0.0)
    M_M = trapezoidal_matrix(heaviside_law(_ortho_tensor(1.0)), times)

    for scheme in (Voigt(), Reuss(), Dilute(), MoriTanaka())
        C = homogenize_alv(rve, scheme, :C; times = times)
        @test isapprox(C, M_M; atol = 1.0e-8)
    end
end

@testset "ortho dispatch — round-trip _ortho_pair / _ortho_blocks" begin
    times = collect(0.0:0.5:1.5)
    M = trapezoidal_matrix(heaviside_law(_ortho_tensor(1.3)), times)

    o = _ortho_pair(M)
    @test length(o) == 12
    @test isapprox(_ortho_blocks(o), M; atol = 1.0e-12)
end
