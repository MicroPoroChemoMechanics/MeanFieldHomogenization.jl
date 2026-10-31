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

const _alv_mandel = (t -> TensND.mandel(TensND.canonical_components(t); symmetrize = true))

const _C_MAT = TensISO{3}(3 * 10.0, 2 * 4.0)
const _C_INC = TensISO{3}(3 * 20.0, 2 * 8.0)
const _CANON = TensND.CanonicalBasis{3, Float64}()

# The three classes, by what makes the dispatch take each path: the shape of the
# inclusion for the schemes that read concentration tensors, the stiffness of the
# inclusion for Voigt and Reuss, which read the phases alone.
const _ALV_CLASSES = (
    ti = (
        shape = Ellipsoid(1.0, 1.0, 0.3),
        C_inc = ti_from_constants(30.0, 12.0, 6.0, 14.0, 9.0, (0.0, 0.0, 1.0)),
    ),
    ortho = (
        shape = Ellipsoid(1.0, 0.6, 0.3),
        C_inc = TensOrtho(30.0, 26.0, 34.0, 9.0, 8.0, 7.0, 7.0, 8.0, 9.0, _CANON),
    ),
    general = (
        shape = Ellipsoid(1.0, 1.0, 0.3; euler_angles = (0.4, 0.7, 0.2)),
        C_inc = ti_from_constants(30.0, 12.0, 6.0, 14.0, 9.0, (0.6, 0.0, 0.8)),
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

@testset "MT and Maxwell on random TI and orthotropic operators match the generic algebra" begin
    # Genuinely time-dependent operators in a tilted axis and a random frame: the
    # class algebra has to agree with the 6n×6n one both on the order of the tensor
    # products, whose normal blocks are 2×2 (TI) or 3×3 (orthotropic) matrices,
    # and on the order in time, along which Volterra operators do not commute either.
    V = MeanFieldHomogenization.Viscoelasticity
    times = collect(range(0.0, 3.0; length = 4))
    axis = Tuple(normalize([0.3, -0.5, 0.8]))
    frame = TensND.RotatedBasis(0.4, 0.9, 0.2)
    laws = (
        s -> ViscoLaw((t, tp) -> (1 + s * exp(tp - t)) * TensTI{4}(5.0 + s, 4.0, 1.0, 0.5 + 0.1s, 2.0, 1.5, axis)),
        s -> ViscoLaw((t, tp) -> (1 + s * exp(tp - t)) * TensOrtho{4, Float64}(ntuple(k -> 1.0 + 0.3k + 0.1s * isodd(k), 12), frame)),
    )
    for law in laws
        op(s) = VolterraOperator(law(s), times)
        C0, H0 = op(1.0), 0.2 * op(2.0)
        A = [I + 0.1 * op(3.0), I + 0.1 * op(4.0)]
        N = [op(5.0), op(6.0)]
        f, fM = [0.2, 0.3], 0.5
        a = V._OpALV{4}(C0.grid)
        mt = V._alg_mori_tanaka(a, C0, A, N, f, fM)
        @test typeof(mt.class) == typeof(C0.class)
        @test Matrix(mt) ≈ mori_tanaka_alv(Matrix(C0), Matrix.(A), Matrix.(N), f, fM) rtol = 1.0e-11
        mx = V._alg_maxwell(a, C0, N, f, H0)
        @test Matrix(mx) ≈ maxwell_alv(Matrix(C0), Matrix.(N), f; H_0 = Matrix(H0)) rtol = 1.0e-11
    end
end
