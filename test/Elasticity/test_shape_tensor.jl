using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

@testset "shape_tensor — Ellipsoid 3D (canonical basis)" begin
    ell = Ellipsoid(3.0, 2.0, 1.0)
    A = shape_tensor(ell)
    A_arr = Array(A)
    @test A_arr ≈ Diagonal([3.0, 2.0, 1.0])  rtol = 1.0e-12
end

@testset "shape_tensor — Ellipsoid 3D (rotated basis)" begin
    b = TensND.RotatedBasis(π / 3, π / 4, π / 5)
    R = Array(b)

    # Constructor with euler_angles and pass-through basis must give same tensor
    ell1 = Ellipsoid(3.0, 2.0, 1.0; euler_angles = (π / 3, π / 4, π / 5))
    ell2 = Ellipsoid(3.0, 2.0, 1.0, b)
    A1 = canonical(shape_tensor(ell1))
    A2 = canonical(shape_tensor(ell2))
    A1_arr = Array(A1)
    A2_arr = Array(A2)
    A_ref = R * Diagonal([3.0, 2.0, 1.0]) * R'

    @test A1_arr ≈ A_ref  rtol = 1.0e-12
    @test A2_arr ≈ A_ref  rtol = 1.0e-12
    @test A1_arr ≈ A2_arr rtol = 1.0e-12

    # Symmetry
    @test A1_arr ≈ A1_arr'  rtol = 1.0e-12

    # Eigenvalues must be the sorted semi-axes (descending)
    λ = sort(eigvals(Symmetric(A1_arr)); rev = true)
    @test λ ≈ [3.0, 2.0, 1.0]  rtol = 1.0e-12
end

@testset "shape_tensor — Ellipsoid 2D" begin
    ell = Ellipsoid(3.0, 2.0; angle = π / 4)
    A = canonical(shape_tensor(ell))
    A_arr = Array(A)

    c, s = cos(π / 4), sin(π / 4)
    R = [c -s; s c]
    A_ref = R * Diagonal([3.0, 2.0]) * R'

    @test size(A_arr) == (2, 2)
    @test A_arr ≈ A_ref        rtol = 1.0e-12
    @test A_arr ≈ A_arr'       rtol = 1.0e-12
end

@testset "shape_tensor — Cylinder (Inf on axis)" begin
    b = TensND.RotatedBasis(π / 3, π / 4, π / 5)
    cyl = Cylinder(2.0, 1.0, b)
    A = canonical(shape_tensor(cyl))
    A_arr = Array(A)

    # At least one entry is Inf (the Inf diagonal in the local frame, when
    # rotated, propagates Inf to all entries touched by the first row/column
    # of R).
    @test any(isinf, A_arr)

    # Along the cylinder axis (column 1 of the basis), the quadratic form
    # u^T A u should be Inf.
    axis = b[:, 1]
    @test isinf(axis' * A_arr * axis)

    # In the canonical-basis case the tensor is exactly diag(Inf, b, c).
    cyl0 = Cylinder(2.0, 1.0)
    A0 = shape_tensor(cyl0)
    A0_arr = Array(A0)
    @test isinf(A0_arr[1, 1])
    @test A0_arr[2, 2] == 2.0
    @test A0_arr[3, 3] == 1.0
    @test iszero(A0_arr[1, 2]) || isnan(A0_arr[1, 2])      # Inf*0 may give NaN
end

@testset "shape_tensor — EllipticCrack (zero on normal)" begin
    b = TensND.RotatedBasis(π / 3, π / 4, π / 5)
    crack = EllipticCrack(3.0, 2.0, b)
    A = canonical(shape_tensor(crack))
    A_arr = Array(A)

    # Along the crack normal n̂ (column 3), quadratic form vanishes.
    nhat = b[:, 3]
    @test isapprox(nhat' * A_arr * nhat, 0.0; atol = 1.0e-12)

    # Along l̂ and m̂, it returns a and b respectively.
    lhat = b[:, 1]
    mhat = b[:, 2]
    @test lhat' * A_arr * lhat ≈ 3.0  rtol = 1.0e-12
    @test mhat' * A_arr * mhat ≈ 2.0  rtol = 1.0e-12
end

@testset "shape_tensor — RibbonCrack (Inf on tunnel, zero on normal)" begin
    # The rotated ribbon-shape tensor has `Inf` in the (1,1) entry of the
    # local frame and `0` in (3,3). Rotating produces `Inf*0 = NaN`
    # everywhere `R[i,1]*R[i,3] ≠ 0`, so the rotated canonical-frame tensor
    # is genuinely ill-defined numerically. We therefore only assert
    # properties that survive: Inf along the tunnel axis, and the exact
    # diagonal shape in the canonical-basis case.

    # Canonical basis — diagonal is exactly (Inf, b, 0)
    ribbon0 = RibbonCrack(2.0)
    A0 = shape_tensor(ribbon0)
    A0_arr = Array(A0)
    @test isinf(A0_arr[1, 1])
    @test A0_arr[2, 2] == 2.0
    @test A0_arr[3, 3] == 0.0

    # Rotated: tunnel axis still yields Inf.
    b = TensND.RotatedBasis(π / 3, π / 4, π / 5)
    ribbon = RibbonCrack(2.0, b)
    A = canonical(shape_tensor(ribbon))
    A_arr = Array(A)
    lhat = b[:, 1]
    @test isinf(lhat' * A_arr * lhat)
    @test any(isinf, A_arr)
end

@testset "Input-order robustness — unsorted semi-axes rotate the basis" begin
    # Ellipsoid 3D: any permutation of (0.5, 3.0, 1.0) must produce the
    # same physical geometry in the canonical frame.
    for perm in (
            (0.5, 3.0, 1.0), (3.0, 0.5, 1.0), (1.0, 0.5, 3.0),
            (0.5, 1.0, 3.0), (3.0, 1.0, 0.5), (1.0, 3.0, 0.5),
        )
        ell = Ellipsoid(perm...)
        # Internal invariant — axes always sorted descending
        @test ell.semi_axes == (3.0, 1.0, 0.5)
        # Canonical-frame shape_tensor matches the user's input order
        A = canonical(shape_tensor(ell))
        @test A[1, 1] ≈ perm[1] rtol = 1.0e-12
        @test A[2, 2] ≈ perm[2] rtol = 1.0e-12
        @test A[3, 3] ≈ perm[3] rtol = 1.0e-12
    end

    # Ellipsoid 2D
    for (a_in, b_in) in ((3.0, 1.5), (1.5, 3.0))
        ell2 = Ellipsoid(a_in, b_in)
        @test ell2.semi_axes == (3.0, 1.5)
        A2 = canonical(shape_tensor(ell2))
        @test A2[1, 1] ≈ a_in rtol = 1.0e-12
        @test A2[2, 2] ≈ b_in rtol = 1.0e-12
    end

    # EllipticCrack with b > a must swap; canonical shape matches input.
    ec = EllipticCrack(0.5, 3.0)
    @test (ec.a, ec.b) == (3.0, 0.5)
    Aec = canonical(shape_tensor(ec))
    @test Aec[1, 1] ≈ 0.5 rtol = 1.0e-12    # user's a (first input)
    @test Aec[2, 2] ≈ 3.0 rtol = 1.0e-12    # user's b
    @test Aec[3, 3] ≈ 0.0 atol = 1.0e-12    # normal

    # Cylinder with c > b swaps transverse columns; canonical diagonal
    # b-direction (e₂) and c-direction (e₃) match user input.
    # (Inf×0 → NaN on off-diagonals is the known Cylinder limitation.)
    cyl = Cylinder(0.5, 2.0)
    @test cyl.semi_axes == (2.0, 0.5)
    A0 = shape_tensor(cyl)           # local frame, reliable
    @test A0[1, 1] == Inf            # cylinder axis stays on col 1
end

@testset "Input-order robustness with euler_angles" begin
    # With a non-trivial Euler rotation, the user's input order must
    # still define which axis gets which length: Ellipsoid(0.5, 3., 1.,
    # euler=(θ,)) means 0.5 along the rotated ẽ₁, 3.0 along ẽ₂, etc.
    θ = π / 5
    ell = Ellipsoid(0.5, 3.0, 1.0; euler_angles = (θ,))
    # Rebuild the reference geometry with explicit descending input.
    ell_ref = Ellipsoid(3.0, 1.0, 0.5; euler_angles = (θ,))
    # Shape tensors differ because the user's input order permutes axes
    # relative to the basis columns.  But the EIGENVALUES must be the
    # same sorted semi-axes, and the TRACE/DETERMINANT are identical.
    A = canonical(shape_tensor(ell))
    A_arr = Array(A)
    Aref = canonical(shape_tensor(ell_ref))
    Aref_arr = Array(Aref)
    @test sort(eigvals(Symmetric(A_arr)); rev = true) ≈ [3.0, 1.0, 0.5]
    @test sort(eigvals(Symmetric(Aref_arr)); rev = true) ≈ [3.0, 1.0, 0.5]
    @test tr(A_arr) ≈ tr(Aref_arr) rtol = 1.0e-12
end

@testset "euler_angles — flexibility (padding + mixed types)" begin
    # Padding with trailing zeros
    ell_pad1 = Ellipsoid(4.0, 2.0, 1.0; euler_angles = (π / 2,))
    ell_pad2 = Ellipsoid(4.0, 2.0, 1.0; euler_angles = (π / 2, 0.0, 0.0))
    @test canonical(shape_tensor(ell_pad1))[1, 1] ≈
        canonical(shape_tensor(ell_pad2))[1, 1] rtol = 1.0e-12

    ell_pad3 = Ellipsoid(4.0, 2.0, 1.0; euler_angles = ())
    ell_pad4 = Ellipsoid(4.0, 2.0, 1.0)
    A3 = canonical(shape_tensor(ell_pad3))
    A4 = canonical(shape_tensor(ell_pad4))
    @test A3 ≈ A4 atol = 1.0e-14

    # Mixed types — Int + Float
    ell_mix = Ellipsoid(4.0, 2.0, 1.0; euler_angles = (π / 2, 0, 0))
    @test canonical(shape_tensor(ell_mix))[1, 1] ≈
        canonical(shape_tensor(ell_pad2))[1, 1] rtol = 1.0e-12

    # Irrational (π directly) + Float
    ell_irr = Ellipsoid(4.0, 2.0, 1.0; euler_angles = (π, 0.0, 0.0))
    @test !(shape_tensor(ell_irr) === nothing)  # constructs without error

    # Coverage across the other inclusion types
    @test_nowarn Cylinder(1.0, 2.0; euler_angles = (π / 4,))
    @test_nowarn EllipticCrack(2.0, 1.0; euler_angles = (π / 4, π / 6))
    @test_nowarn RibbonCrack(1.0; euler_angles = (π / 3, 0, π / 4))
    @test_nowarn PennyCrack(1.0; euler_angles = (0.5,))

    # Length > 3 → ArgumentError
    @test_throws ArgumentError Ellipsoid(4.0, 2.0, 1.0; euler_angles = (0.1, 0.2, 0.3, 0.4))
end

@testset "shape_tensor — eltype propagation" begin
    ell_f = Ellipsoid(3.0, 2.0, 1.0)
    A_f = shape_tensor(ell_f)
    @test eltype(A_f) <: AbstractFloat

    ell_b = Ellipsoid(BigFloat(3), BigFloat(2), BigFloat(1))
    A_b = shape_tensor(ell_b)
    @test eltype(A_b) === BigFloat
end

@testset "Derivatives with respect to the orientation" begin
    # Semi-axes given out of order permute the columns of the basis, and the
    # permutation read the frame as Float64: a dual Euler angle threw, and so
    # did a dual rotation matrix handed to a constructor, which converted it
    # to Float64 before anything else. Reference: the rotation matrix of
    # TensND, differentiated as it stands.
    Q(θ) = Matrix(TensND.vecbasis(TensND.RotatedBasis(θ, 0.4, 0.2), :cov))
    frame(incl) = Matrix(TensND.vecbasis(inclusion_basis(incl), :cov))
    θ = 0.7
    D = Diagonal([0.5, 3.0, 1.0])
    dA_ref = ForwardDiff.derivative(t -> Q(t) * D * Q(t)', θ)
    for incl in (t -> Ellipsoid(0.5, 3.0, 1.0; euler_angles = (t, 0.4, 0.2)), t -> Ellipsoid(0.5, 3.0, 1.0, Q(t)))
        A(t) = Matrix(TensND.canonical_components(shape_tensor(incl(t))))
        @test A(θ) ≈ Q(θ) * D * Q(θ)' rtol = 1.0e-12
        @test ForwardDiff.derivative(A, θ) ≈ dA_ref rtol = 1.0e-12
    end
    # The transverse axes of a cylinder in ascending order swap columns 2 and
    # 3, the in-plane axes of a crack columns 1 and 2, the second negated.
    swap23(M) = hcat(M[:, 1], M[:, 3], -M[:, 2])
    swap12(M) = hcat(M[:, 2], -M[:, 1], M[:, 3])
    @test ForwardDiff.derivative(t -> frame(Cylinder(0.4, 1.0, Q(t))), θ) ≈
        ForwardDiff.derivative(t -> swap23(Q(t)), θ) rtol = 1.0e-12
    @test ForwardDiff.derivative(t -> frame(EllipticCrack(0.5, 2.0, Q(t))), θ) ≈
        ForwardDiff.derivative(t -> swap12(Q(t)), θ) rtol = 1.0e-12
end
