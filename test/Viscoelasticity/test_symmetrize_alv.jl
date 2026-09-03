# =============================================================================
#  test_symmetrize_alv.jl — block-wise orientation averages of ALV Volterra
#  matrices (`_iso_project_blocks`, `_ti_project_blocks`,
#  `_maybe_symmetrize_alv`) against the elastic Core implementations.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
import MeanFieldHomogenization.Viscoelasticity: _iso_project_blocks, _ti_project_blocks,
    _maybe_symmetrize_alv, _iso_project_mandel66,
    _iso_project_blocks3, _ti_project_blocks3, _maybe_symmetrize_alv2
const MCr = MeanFieldHomogenization.Core

function _rand_minor_mandel(rng)
    a = randn(rng, 3, 3, 3, 3)
    b = zeros(3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        b[i, j, k, l] = (a[i, j, k, l] + a[j, i, k, l] + a[i, j, l, k] + a[j, i, l, k]) / 4
    end
    return MCr.mandel66_minor(b)
end

@testset "ALV block-wise orientation averages" begin
    rng = MersenneTwister(42)
    nax = (0.36, -0.48, 0.8)
    ez = (0.0, 0.0, 1.0)

    @testset "single block == elastic Core implementation" begin
        M = _rand_minor_mandel(rng)
        # ISO — non-symmetric block: the full Σ M[i,j] must be used, not
        # a symmetric 2·M[1,2] shortcut (regression for Volterra blocks)
        α, β = _iso_project_mandel66(M)
        αc, βc = MCr.iso_average_mandel66(M)
        @test α ≈ αc
        @test β ≈ βc
        # TI about an arbitrary axis
        @test _ti_project_blocks(M, nax) ≈ MCr.ti_average_mandel66(M, nax) atol = 1.0e-12
    end

    @testset "multi-block matrix : block independence + idempotence" begin
        n = 3
        M = zeros(6n, 6n)
        blocks = [_rand_minor_mandel(rng) for _ in 1:n, _ in 1:n]
        for i in 1:n, j in 1:n
            M[(6i - 5):(6i), (6j - 5):(6j)] = blocks[i, j]
        end
        out = _ti_project_blocks(M, nax)
        for i in 1:n, j in 1:n
            @test out[(6i - 5):(6i), (6j - 5):(6j)] ≈
                MCr.ti_average_mandel66(blocks[i, j], nax) atol = 1.0e-12
        end
        # idempotence
        @test _ti_project_blocks(out, nax) ≈ out atol = 1.0e-12
        # ISO ∘ TI == ISO (block-wise)
        @test _iso_project_blocks(out) ≈ _iso_project_blocks(M) atol = 1.0e-12
    end

    @testset "_maybe_symmetrize_alv dispatch incl. TISymmetrize" begin
        M = zeros(12, 12)
        for i in 1:2, j in 1:2
            M[(6i - 5):(6i), (6j - 5):(6j)] = _rand_minor_mandel(rng)
        end
        @test _maybe_symmetrize_alv(M, NoSymmetrize()) === M
        @test _maybe_symmetrize_alv(M, IsoSymmetrize()) ≈ _iso_project_blocks(M)
        @test _maybe_symmetrize_alv(M, TISymmetrize(ez)) ≈ _ti_project_blocks(M, ez)
        @test _maybe_symmetrize_alv(M, TISymmetrize(nax)) ≈ _ti_project_blocks(M, nax)
    end

    # ── Order-2 (3 × 3 time blocks) ─────────────────────────────────────────
    #
    # The projectors above slice 6×6 Mandel blocks; the order-2 ALV pipeline
    # carries (3n × 3n) matrices of plain 2-tensor blocks and needs its own.
    # `n = 3` below is ODD on purpose: `3n = 9` is not divisible by 6, so the
    # order-4 projectors cannot even be called on it.
    @testset "order-2 block-wise averages (3×3 blocks)" begin
        n = 3
        M = randn(rng, 3n, 3n)
        blk(A, i, j) = A[(3i - 2):(3i), (3j - 2):(3j)]

        Pi = _iso_project_blocks3(M)
        Pt_z = _ti_project_blocks3(M, ez)
        Pt_n = _ti_project_blocks3(M, nax)

        @testset "iso == spherical part, block by block" begin
            for i in 1:n, j in 1:n
                B = blk(M, i, j)
                @test blk(Pi, i, j) ≈ (tr(B) / 3) * I(3) atol = 1.0e-12
            end
        end

        # Exact azimuthal average, checked against an explicit rotation average
        # about the axis (trapezoid over θ, which is exact for the finite
        # Fourier content of R·B·Rᵀ).
        function _num_azim_avg(B, axis; nθ = 2048)
            v = collect(axis) ./ sqrt(sum(abs2, collect(axis)))
            acc = zeros(3, 3)
            for m in 0:(nθ - 1)
                θ = 2π * m / nθ
                c, s = cos(θ), sin(θ)
                K = [0.0 -v[3] v[2]; v[3] 0.0 -v[1]; -v[2] v[1] 0.0]
                R = I(3) + s * K + (1 - c) * (K * K)     # Rodrigues
                acc .+= R * B * R'
            end
            return acc ./ nθ
        end

        @testset "TI == explicit numerical rotation average" begin
            for (P, axis) in ((Pt_z, ez), (Pt_n, nax)), i in 1:n, j in 1:n
                @test blk(P, i, j) ≈ _num_azim_avg(blk(M, i, j), axis) atol = 1.0e-8
            end
        end

        @testset "projector algebra" begin
            # Both are projectors: idempotent.
            @test _iso_project_blocks3(Pi) ≈ Pi atol = 1.0e-12
            @test _ti_project_blocks3(Pt_z, ez) ≈ Pt_z atol = 1.0e-12
            @test _ti_project_blocks3(Pt_n, nax) ≈ Pt_n atol = 1.0e-12
            # ISO ∘ TI == ISO (the coarser average absorbs the finer one).
            @test _iso_project_blocks3(Pt_z) ≈ Pi atol = 1.0e-12
            @test _iso_project_blocks3(Pt_n) ≈ Pi atol = 1.0e-12
            # Orientation averaging preserves the trace of every block.
            for i in 1:n, j in 1:n
                t = tr(blk(M, i, j))
                @test tr(blk(Pi, i, j)) ≈ t atol = 1.0e-12
                @test tr(blk(Pt_z, i, j)) ≈ t atol = 1.0e-12
                @test tr(blk(Pt_n, i, j)) ≈ t atol = 1.0e-12
            end
            # An already-isotropic matrix is a fixed point of both.
            Iso = _iso_project_blocks3(randn(rng, 3n, 3n))
            @test _ti_project_blocks3(Iso, nax) ≈ Iso atol = 1.0e-12
        end

        @testset "_maybe_symmetrize_alv2 dispatch" begin
            @test _maybe_symmetrize_alv2(M, NoSymmetrize()) === M
            @test _maybe_symmetrize_alv2(M, IsoSymmetrize()) ≈ Pi
            @test _maybe_symmetrize_alv2(M, TISymmetrize(ez)) ≈ Pt_z
            @test _maybe_symmetrize_alv2(M, TISymmetrize(nax)) ≈ Pt_n
        end

        @testset "size validation" begin
            @test_throws ArgumentError _iso_project_blocks3(randn(rng, 4, 4))
            @test_throws ArgumentError _iso_project_blocks3(randn(rng, 3, 6))
            @test_throws ArgumentError _ti_project_blocks3(randn(rng, 4, 4), ez)
        end
    end
end

# The order-2 ALV Hill kernel exists for an ISOTROPIC reference only.  Where
# Mori-Tanaka / dilute / Maxwell evaluate it against the fixed matrix, the
# differential scheme evaluates it against its running medium, so a
# non-spherical inclusion must be orientation-averaged for that medium to stay
# in the class.  This checks the projection actually lands: the effective
# order-2 result must come out isotropic block by block.
@testset "ALV order-2 — `symmetrize = :iso` makes the result isotropic" begin
    times = collect(range(0.0, 2.0; length = 6))
    law_K = ViscoLaw((t, tp) -> TensISO{3}(1.0 * (1.0 + exp(-(t - tp)))), :relaxation)

    function eff(sch, sym)
        rve = RVE(:M)
        add_matrix!(rve, Ellipsoid(1.0), Dict(:K => law_K))
        add_phase!(
            rve, :I, Spheroid(5.0), Dict(:K => heaviside_law(TensISO{3}(10.0)));
            fraction = 0.2, symmetrize = sym
        )
        return homogenize_alv(rve, sch, :K; times = times)
    end
    dev_from_iso(M) = maximum(
        let n = size(M, 1) ÷ 3
            [
                begin
                    B = M[(3i - 2):(3i), (3j - 2):(3j)]
                    maximum(abs, B - (tr(B) / 3) * I(3)) / max(abs(tr(B) / 3), 1.0e-12)
                end for i in 1:n, j in 1:n
            ]
        end
    )

    # Every order-2 scheme, differential included.
    for sch in (
            Voigt(), Reuss(), Dilute(), DiluteDual(),
            MoriTanaka(), Maxwell(), DifferentialScheme(; nsteps = 20),
        )
        @test dev_from_iso(eff(sch, :iso)) < 1.0e-10
    end

    # And the average is not a no-op for the shape-sensitive schemes: an
    # aligned spheroid gives a genuinely anisotropic result without it.
    # (Voigt / Reuss average the phase matrices directly — the geometry never
    # enters them, so there is nothing for the projection to change.)
    for sch in (Dilute(), DiluteDual(), MoriTanaka(), Maxwell())
        @test dev_from_iso(eff(sch, :none)) > 1.0e-3
        @test maximum(abs, eff(sch, :iso) - eff(sch, :none)) > 1.0e-8
    end
    for sch in (Voigt(), Reuss())
        @test eff(sch, :iso) ≈ eff(sch, :none)
    end
end
