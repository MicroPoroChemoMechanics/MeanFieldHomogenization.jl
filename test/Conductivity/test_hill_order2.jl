using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

@testset "Conductivity — order-2 Hill tensor" begin
    K = TensISO{3}(5.0)
    ell = Ellipsoid(1.0)
    P = hill_tensor(ell, K)
    @test P[1, 1] ≈ 1 / (3 * 5.0) atol = 1.0e-12

    # 2D circle
    K2 = TensISO{2}(3.0)
    ell2 = Ellipsoid(1.0; dim = 2)
    P2 = hill_tensor(ell2, K2)
    @test P2[1, 1] ≈ 1 / (2 * 3.0) atol = 1.0e-12
end

# ── Orientation ──────────────────────────────────────────────────────────────
#
# Regression guard. The isotropic 3-D kernels read the Newton-potential
# components in the *inclusion's own* basis; wrapping them straight into a
# canonical `Tens` used to drop the orientation, so `hill_tensor` of a rotated
# spheroid returned the tensor of the unrotated one — exactly, hence silently.
#
# The internal consistency check `s = P·K₀` in `test_eshelby.jl` cannot catch
# this: both sides drop the rotation identically. What catches it is
# equivariance, and a comparison against the anisotropic kernel, which always
# built the shape tensor with the rotation in it.

_rot(basis) = Matrix(TensND.vecbasis(basis, :cov))
_canon2(t) = Matrix(TensND.components_canon(t))

@testset "Conductivity — the order-2 Hill tensor rotates with the inclusion" begin
    K = TensISO{3}(1.7)
    ang = (0.3, 0.7, 0.2)

    @testset "ellipsoid" begin
        for axes in ((1.0, 1.0, 0.4), (2.5, 1.0, 1.0), (3.0, 2.0, 1.0))
            e0 = Ellipsoid(axes...)
            e1 = Ellipsoid(axes...; euler_angles = ang)
            R = _rot(MeanFieldHomogenization.inclusion_basis(e1))
            P0, P1 = _canon2(hill_tensor(e0, K)), _canon2(hill_tensor(e1, K))
            @test P1 ≈ R * P0 * transpose(R) atol = 1.0e-12
            # And it really is rotated: a non-spherical shape must *change*.
            @test !isapprox(P1, P0; atol = 1.0e-6)
        end
    end

    @testset "cylinder" begin
        c0 = Cylinder(1.0, 0.4)
        c1 = Cylinder(1.0, 0.4; euler_angles = ang)
        R = _rot(MeanFieldHomogenization.inclusion_basis(c1))
        P0, P1 = _canon2(hill_tensor(c0, K)), _canon2(hill_tensor(c1, K))
        @test P1 ≈ R * P0 * transpose(R) atol = 1.0e-12
        @test !isapprox(P1, P0; atol = 1.0e-6)
    end

    @testset "the isotropic kernel agrees with the anisotropic one" begin
        # The sharpest guard available: the anisotropic path is an independent
        # derivation (Giraud's K^(-1/2) change of variable) and was never
        # affected. Reaching for the private kernel is deliberate — the public
        # entry point dispatches `TensISO` to the analytic branch, so there is no
        # other way to compare the two on the same input.
        C = MeanFieldHomogenization.Conductivity
        K = TensISO{3}(1.7)
        for axes in ((1.0, 1.0, 0.4), (3.0, 2.0, 1.0))
            for angles in ((), (0.3, 0.7, 0.2))
                e = Ellipsoid(axes...; euler_angles = angles)
                Pi = _canon2(C._hill_order2_3d_iso(e, K))
                Pa = _canon2(C._hill_order2_3d_aniso(e, K))
                @test Pi ≈ Pa atol = 1.0e-10
            end
        end
    end
end

# ── Automatic differentiation of the anisotropic kernel ──────────────────────
#
# The anisotropic kernel went through LAPACK `eigen` and `svd`, which a dual
# cannot enter. Reference: a conductor whose principal axes are the
# inclusion's. The fictitious inclusion of the K^(-1/2) change of variable then
# has semi-axes aᵢ/√kᵢ along the same axes, and P = diag(Iᵢ(a/√k)/kᵢ), with I
# the Newton potential of the isotropic kernel, which differentiates as it
# stands. A rotated configuration is checked by equivariance, P(R·ε, R K Rᵀ) =
# R P(ε, K) Rᵀ, which brings the off-diagonal entries of K into play.

_diagK(k) = TensND.Tens(diagm(k))

@testset "Conductivity — the anisotropic order-2 Hill tensor differentiates" begin
    _P(ell, K) = _canon2(hill_tensor(ell, K))
    function P_ref(a, k)
        e = Ellipsoid(a[1] / sqrt(k[1]), a[2] / sqrt(k[2]), a[3] / sqrt(k[3]))
        IA = TensND.components_canon(tens_IA(e))
        return [IA[i, i] / k[i] for i in 1:3]
    end
    a = [2.0, 1.2, 0.7]
    for k in ([3.0, 1.5, 0.8], [2.0, 2.0, 0.8])     # the second, a repeated eigenvalue
        P_k(k) = diag(_P(Ellipsoid(a...), _diagK(k)))
        @test P_k(k) ≈ P_ref(a, k) rtol = 1.0e-12
        @test ForwardDiff.jacobian(P_k, k) ≈ ForwardDiff.jacobian(x -> P_ref(a, x), k) rtol = 1.0e-10
        @test ForwardDiff.jacobian(x -> diag(_P(Ellipsoid(x...), _diagK(k))), a) ≈
            ForwardDiff.jacobian(x -> P_ref(x, k), a) rtol = 1.0e-10
        # The value of a differentiated evaluation is the evaluation itself.
        @test ForwardDiff.value.(P_k(ForwardDiff.Dual.(k, 1.0))) == P_k(k)
    end
    # Turned: a derivative with respect to a conductivity, the conductor and
    # the ellipsoid turned together.
    ang = (0.3, 0.7, 0.2)
    R = _rot(MeanFieldHomogenization.inclusion_basis(Ellipsoid(a...; euler_angles = ang)))
    k = [3.0, 1.5, 0.8]
    turned(k) = _P(Ellipsoid(a...; euler_angles = ang), TensND.Tens(R * Matrix(get_array(_diagK(k))) * R'))
    aligned(k) = _P(Ellipsoid(a...), _diagK(k))
    @test turned(k) ≈ R * aligned(k) * R' rtol = 1.0e-12
    for j in 1:3
        e_j(t) = [i == j ? k[i] + t : k[i] for i in 1:3]
        @test ForwardDiff.derivative(t -> turned(e_j(t)), 0.0) ≈
            R * ForwardDiff.derivative(t -> aligned(e_j(t)), 0.0) * R' rtol = 1.0e-10
    end
    # The cylinder: P₁₁ = 0 and, with b' = b/√k₂, c' = c/√k₃,
    # P₂₂ = c'/((b' + c') k₂), P₃₃ = b'/((b' + c') k₃).
    b, c = 1.0, 0.4
    P_cyl(k) = diag(_P(Cylinder(b, c), _diagK(k)))
    function P_cyl_ref(k)
        b′, c′ = b / sqrt(k[2]), c / sqrt(k[3])
        return [zero(b′), c′ / ((b′ + c′) * k[2]), b′ / ((b′ + c′) * k[3])]
    end
    @test P_cyl(k) ≈ P_cyl_ref(k) atol = 1.0e-14
    @test ForwardDiff.jacobian(P_cyl, k) ≈ ForwardDiff.jacobian(P_cyl_ref, k) atol = 1.0e-12
end
