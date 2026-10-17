using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_alv_symmetry_paths.jl — every ALV scheme on each of the paths its
#  dispatch takes by symmetry: transversely isotropic about e₃, orthotropic in
#  the canonical frame, and general. Each is held to its elastic limit: with
#  phases that do not creep, the ALV operator is block diagonal and each block is
#  the stiffness the elastic scheme gives, which is computed by another pipeline.
# =============================================================================

const _alv_mandel = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

const _C_MAT = TensISO{3}(3 * 10.0, 2 * 4.0)
const _C_INC = TensISO{3}(3 * 20.0, 2 * 8.0)
const _CANON = TensND.CanonicalBasis{3, Float64}()

# The three classes, by what makes the dispatch take each path: the shape of the
# inclusion for the schemes that read concentration tensors, the stiffness of the
# inclusion for Voigt and Reuss, which read the phases alone.
const _ALV_CLASSES = (
    ti = (
        shape = Ellipsoid(1.0, 1.0, 0.3),
        C_inc = tens_TI(30.0, 12.0, 6.0, 14.0, 9.0, (0.0, 0.0, 1.0)),
    ),
    ortho = (
        shape = Ellipsoid(1.0, 0.6, 0.3),
        C_inc = TensOrtho(30.0, 26.0, 34.0, 9.0, 8.0, 7.0, 7.0, 8.0, 9.0, _CANON),
    ),
    general = (
        shape = Ellipsoid(1.0, 1.0, 0.3; euler_angles = (0.4, 0.7, 0.2)),
        C_inc = tens_TI(30.0, 12.0, 6.0, 14.0, 9.0, (0.6, 0.0, 0.8)),
    ),
)

function _alv_rve(shape, C_inc; alv::Bool)
    law(C) = alv ? heaviside_law(C) : C
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law(_C_MAT)); fraction = :rest)
    add_phase!(rve, :I, shape, Dict(:C => law(C_inc)); fraction = 0.2)
    return rve
end

function _check_elastic_limit(C_alv, M_ref, n)
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test C_alv[rows, rows] ≈ M_ref rtol = 1.0e-9 atol = 1.0e-9
        for j in 1:(i - 1)
            @test maximum(abs, C_alv[rows, (6 * (j - 1) + 1):(6 * j)]) <= 1.0e-9
        end
    end
    return
end

@testset "ALV symmetry paths — $(scheme) on the $(cls) path, elastic limit" for scheme in (
            Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(), Maxwell(), PonteCastanedaWillis(),
        ),
        cls in keys(_ALV_CLASSES)

    c = _ALV_CLASSES[cls]
    # Voigt and Reuss read the phases: the class is that of the inclusion's
    # stiffness. The others read concentration tensors: that of its shape.
    phases_only = scheme isa Voigt || scheme isa Reuss
    shape = phases_only ? Ellipsoid(1.0) : c.shape
    C_inc = phases_only ? c.C_inc : _C_INC
    times = [0.0, 0.5, 1.0]
    M_ref = _alv_mandel(homogenize(_alv_rve(shape, C_inc; alv = false), scheme, :C))
    C_alv = homogenize_alv(_alv_rve(shape, C_inc; alv = true), scheme, :C; times = times)
    _check_elastic_limit(C_alv, M_ref, length(times))
end

# A random lower-triangular n×n Volterra matrix, its diagonal kept away from zero.
function _alv_rand_lt(n; scale = 1.0)
    M = [i >= j ? scale * randn() : 0.0 for i in 1:n, j in 1:n]
    for i in 1:n
        M[i, i] += sign(M[i, i]) * 1.0 + (M[i, i] == 0)
    end
    return M
end

@testset "MT and Maxwell on random TI and orthotropic operators match the generic algebra" begin
    # Genuinely time-dependent operators: the structured fast paths have to
    # agree with the 6n×6n algebra both on the order of the tensor products,
    # whose normal blocks are 2×2 (TI) or 3×3 (orthotropic) matrices, and on the
    # order in time, along which Volterra operators do not commute either.
    V = MeanFieldHomogenization.Viscoelasticity
    n = 4
    for (np, blocks, mt_struct, mx_struct) in (
            (6, ti_blocks_from_params, V.mori_tanaka_alv_ti, V.maxwell_alv_ti),
            (12, ortho_blocks_from_params, V.mori_tanaka_alv_ortho, V.maxwell_alv_ortho),
        )
        rand_op(; scale = 1.0) = ntuple(_ -> _alv_rand_lt(n; scale), np)
        # The identity of each class, read off its block form.
        id = blocks === ti_blocks_from_params ? V._ti_identity(n, Float64) : V._ortho_identity(n, Float64)
        near_id() = ntuple(k -> id[k] .+ 0.1 .* _alv_rand_lt(n; scale = 0.3), np)
        C0, H0 = rand_op(), rand_op(; scale = 0.2)
        A = [near_id(), near_id()]
        N = [rand_op(), rand_op()]
        f, fM = [0.2, 0.3], 0.5
        mt = blocks(mt_struct(C0, A, N, f, fM))
        @test mt ≈ mori_tanaka_alv(blocks(C0), blocks.(A), blocks.(N), f, fM) rtol = 1.0e-9
        mx = blocks(mx_struct(C0, N, f, H0))
        @test mx ≈ maxwell_alv(blocks(C0), blocks.(N), f; H_0 = blocks(H0)) rtol = 1.0e-9
    end
end
