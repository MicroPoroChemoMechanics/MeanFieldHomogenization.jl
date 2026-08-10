using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_pair_tensors.jl — two-inclusion interaction tensors 𝕋^{ab}.
#
#  Coverage:
#   1. Green operator 𝔾⁰ against a finite difference of `green_gradient_iso`.
#   2. Zero isotropic part: T_iijj = T_ijij = 0 (elasticity), tr 𝕋 = 0
#      (conduction) — the invariant behind "cubic arrays keep the Mori-Tanaka
#      bulk modulus".
#   3. Closed form vs the product-quadrature oracle, in the four
#      physics × dimension combinations.
#   4. Multipole expansion reproduces the closed form exactly for ball pairs,
#      and converges at the expected rate for general ellipsoids.
#   5. Self term: 𝕋^{aa} = -ℙ, exactly.
#   6. Far-field decay R⁻³ (3D) / R⁻² (2D) and the ρ² finite-size correction.
#   7. Frame equivariance under a rigid rotation of the pair.
#   8. Asymmetric case: unequal radii, off-axis separation.
#   9. Error paths: overlapping regions, anisotropic reference, bad shape.
#  10. ForwardDiff through the kernel (separation and moduli).
# =============================================================================

const RTOL_QUAD = 1.0e-10
const ATOL_ZERO = 1.0e-12

# Reference media used throughout: k = 1, μ = 1 (elasticity), σ₀ = 2
# (conduction).  `TensISO{3}(α, β)` is 3k J + 2μ K with α = 3k, β = 2μ.
_C3() = TensISO{3}(3.0, 2.0)
_C2() = TensISO{2}(3.0, 2.0)
_K3() = TensISO{3}(2.0)
_K2() = TensISO{2}(2.0)

_ball(a) = Ellipsoid(a)
_disk(a) = Ellipsoid(a, a)

@testset "Green operator — consistency with the Green gradient" begin
    C₀ = _C3()
    x = [1.3, -0.7, 2.1]
    h = 1.0e-5
    # 𝔾⁰_ijkl = [∂²G_ik/∂x_j∂x_l]_{(ij)(kl)}: differentiate ∂G/∂x once more.
    dG = k -> begin
        xp = copy(x); xp[k] += h
        xm = copy(x); xm[k] -= h
        (green_gradient_iso(C₀, xp) .- green_gradient_iso(C₀, xm)) ./ (2h)
    end
    H = [dG(l)[i, k, j] for i in 1:3, k in 1:3, j in 1:3, l in 1:3]
    Γfd = [
        (H[i, k, j, l] + H[j, k, i, l] + H[i, l, j, k] + H[j, l, i, k]) / 4
            for i in 1:3, j in 1:3, k in 1:3, l in 1:3
    ]
    Γ = green_operator_iso(C₀, x)
    @test maximum(abs.(Γ .- Γfd)) < 1.0e-6 * maximum(abs.(Γ))
end

@testset "Green operator — conduction kernels are traceless" begin
    x3 = [1.0, -2.0, 0.5]
    @test tr(green_operator_iso(_K3(), x3)) ≈ 0.0 atol = ATOL_ZERO
    x2 = [1.0, -2.0]
    @test tr(green_operator_iso(_K2(), x2)) ≈ 0.0 atol = ATOL_ZERO
end

@testset "𝕋^{ab} — vanishing isotropic part (elasticity)" begin
    # Bieniek et al. (2024), App. B: the interaction tensor of two balls has a
    # strictly zero isotropic part.  This is why the effective bulk modulus of
    # a cubic array coincides exactly with the Mori-Tanaka estimate.
    C₀ = _C3()
    for r in ([0.0, 0.0, 5.0], [3.0, -4.0, 2.0], [1.0, 1.0, 1.0] .* 4)
        A = get_array(interaction_tensor(_ball(1.0), _ball(0.7), r, C₀))
        @test sum(A[i, i, j, j] for i in 1:3, j in 1:3) ≈ 0.0 atol = ATOL_ZERO
        @test sum(A[i, j, i, j] for i in 1:3, j in 1:3) ≈ 0.0 atol = ATOL_ZERO
    end
end

@testset "𝕋^{ab} — vanishing isotropic part (conduction)" begin
    @test tr(get_array(interaction_tensor(_ball(1.0), _ball(0.7), [1.0, 2.0, 3.0], _K3()))) ≈
        0.0 atol = ATOL_ZERO
    @test tr(get_array(interaction_tensor(_disk(1.0), _disk(0.7), [2.0, 3.0], _K2()))) ≈
        0.0 atol = ATOL_ZERO
end

@testset "𝕋^{ab} — closed form vs product quadrature" begin
    # The quadrature back-end integrates the Green operator over both regions
    # with no closed-form input, so agreement validates the analytical kernels
    # end to end (including the ρ² finite-size correction).
    # The 3D rule converges spectrally: at this separation the relative error
    # is 3.8e-9 with 8 radial/polar nodes, 1.6e-11 with 10 and 1.3e-13 with 12.
    # Ten nodes buys a 1e-10 gate at a quarter of the cost of twelve.
    cases = (
        ("3D elasticity", _ball(1.0), _ball(0.8), [1.0, 2.0, 3.0], _C3(), (10, 10, 20)),
        ("3D conduction", _ball(1.0), _ball(0.8), [1.0, 2.0, 3.0], _K3(), (10, 10, 20)),
        ("2D elasticity", _disk(1.0), _disk(0.8), [2.0, 3.0], _C2(), (10, 8, 32)),
        ("2D conduction", _disk(1.0), _disk(0.8), [2.0, 3.0], _K2(), (8, 8, 24)),
    )
    for (name, ia, ib, r, P₀, nodes) in cases
        @testset "$name" begin
            A = get_array(interaction_tensor(ia, ib, r, P₀))
            Q = get_array(interaction_tensor(ia, ib, r, P₀; method = :quadrature, nodes = nodes))
            @test maximum(abs.(A .- Q)) < RTOL_QUAD * maximum(abs.(A))
        end
    end
end

@testset "𝕋^{ab} — multipole is exact for ball pairs" begin
    # For balls the multipole series terminates at the second moment (the
    # elastic Green function is biharmonic), so `:multipole` must reproduce
    # the closed form to machine precision — not merely approximate it.
    for (ia, ib, r, P₀) in (
            (_ball(1.0), _ball(0.6), [1.0, 2.0, 3.0], _C3()),
            (_ball(1.0), _ball(0.6), [1.0, 2.0, 3.0], _K3()),
            (_disk(1.0), _disk(0.6), [2.0, 3.0], _C2()),
            (_disk(1.0), _disk(0.6), [2.0, 3.0], _K2()),
        )
        A = get_array(interaction_tensor(ia, ib, r, P₀))
        M = get_array(interaction_tensor(ia, ib, r, P₀; method = :multipole))
        @test maximum(abs.(A .- M)) < 1.0e-14 * maximum(abs.(A))
    end
end

@testset "𝕋^{ab} — multipole convergence for general ellipsoids" begin
    # Here the series does NOT terminate: order 2 must converge to the
    # quadrature value faster than order 0, and both must improve with
    # separation.
    C₀ = _C3()
    ea, eb = Ellipsoid(1.0, 0.6, 0.4), Ellipsoid(0.8, 0.8, 0.3)
    errs2 = Float64[]
    errs0 = Float64[]
    for R in (6.0, 24.0)
        r = [0.0, 0.0, R]
        Q = get_array(interaction_tensor(ea, eb, r, C₀; method = :quadrature, nodes = (8, 8, 16)))
        M2 = get_array(interaction_tensor(ea, eb, r, C₀; method = :multipole, order = 2))
        M0 = get_array(interaction_tensor(ea, eb, r, C₀; method = :multipole, order = 0))
        s = maximum(abs.(Q))
        push!(errs2, maximum(abs.(M2 .- Q)) / s)
        push!(errs0, maximum(abs.(M0 .- Q)) / s)
    end
    @test all(errs2 .< errs0)                 # order 2 beats order 0 everywhere
    @test issorted(errs2; rev = true)         # and both converge with distance
    @test issorted(errs0; rev = true)
    @test errs2[end] < 1.0e-4
end

@testset "𝕋^{aa} — self term is minus the Hill tensor" begin
    for (incl, P₀) in (
            (_ball(1.0), _C3()), (Ellipsoid(1.0, 0.5, 0.25), _C3()),
            (_ball(1.0), _K3()), (_disk(1.0), _C2()), (_disk(1.0), _K2()),
        )
        S = get_array(self_interaction_tensor(incl, P₀))
        P = get_array(hill_tensor(incl, P₀))
        @test maximum(abs.(S .+ P)) ≈ 0.0 atol = 1.0e-14
    end
end

@testset "𝕋^{ab} — far-field decay and finite-size correction" begin
    C₀ = _C3()
    # Leading order scales as V_b/R³: doubling R divides the tensor by 8.
    n(R) = maximum(abs.(get_array(interaction_tensor(_ball(0.1), _ball(0.1), [0.0, 0.0, R], C₀))))
    @test n(40.0) / n(80.0) ≈ 8.0 rtol = 1.0e-3
    # The ρ² correction is what separates two pairs of equal separation and
    # different radii — at fixed b the receiver radius must still matter.
    Γ_small = get_array(interaction_tensor(_ball(0.2), _ball(1.0), [0.0, 0.0, 5.0], C₀))
    Γ_big = get_array(interaction_tensor(_ball(1.5), _ball(1.0), [0.0, 0.0, 5.0], C₀))
    @test !isapprox(Γ_small[3, 3, 3, 3], Γ_big[3, 3, 3, 3]; rtol = 1.0e-3)
    # 2D conduction has no such correction: the receiver radius drops out.
    Γ2a = get_array(interaction_tensor(_disk(0.2), _disk(1.0), [5.0, 0.0], _K2()))
    Γ2b = get_array(interaction_tensor(_disk(1.5), _disk(1.0), [5.0, 0.0], _K2()))
    @test Γ2a ≈ Γ2b rtol = 1.0e-14
end

@testset "𝕋^{ab} — frame equivariance" begin
    # Rotating the pair rotates the tensor: Γ(Rr) = R⋆Γ(r).
    C₀ = _C3()
    r = [1.0, 2.0, 3.0]
    θ = 0.7
    Rot = [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]
    A = get_array(interaction_tensor(_ball(1.0), _ball(0.8), r, C₀))
    B = get_array(interaction_tensor(_ball(1.0), _ball(0.8), Rot * r, C₀))
    RA = [
        sum(
                Rot[i, p] * Rot[j, q] * Rot[k, s] * Rot[l, t] * A[p, q, s, t]
                for p in 1:3, q in 1:3, s in 1:3, t in 1:3
            ) for i in 1:3, j in 1:3, k in 1:3, l in 1:3
    ]
    @test maximum(abs.(B .- RA)) < 1.0e-12 * maximum(abs.(A))
end

@testset "𝕋^{ab} — asymmetric case: unequal radii, off-axis" begin
    # A symmetric configuration can hide an index or an axis mistake; this one
    # cannot.  Note that 𝕋^{ab} ≠ Γ^{ba} in general — only the *moment* form
    # V_a 𝕋^{ab} is symmetric under the exchange of the two inclusions.
    C₀ = _C3()
    a, b = 1.3, 0.4
    r = [2.0, -3.0, 4.5]
    Γab = get_array(interaction_tensor(_ball(a), _ball(b), r, C₀))
    Γba = get_array(interaction_tensor(_ball(b), _ball(a), -r, C₀))
    V_a, V_b = 4π * a^3 / 3, 4π * b^3 / 3
    @test maximum(abs.(V_a .* Γab .- V_b .* Γba)) < 1.0e-12 * maximum(abs.(V_a .* Γab))
    @test !isapprox(Γab, Γba; rtol = 1.0e-6)
    Q = get_array(interaction_tensor(_ball(a), _ball(b), r, C₀; method = :quadrature, nodes = (10, 10, 20)))
    @test maximum(abs.(Γab .- Q)) < RTOL_QUAD * maximum(abs.(Γab))
end

@testset "𝕋^{ab} — error paths" begin
    C₀ = _C3()
    # Overlapping regions
    @test_throws ArgumentError interaction_tensor(_ball(1.0), _ball(1.0), [0.0, 0.0, 1.5], C₀)
    # An anisotropic reference in 2D ELASTICITY is still refused: its Green
    # function needs the Stroh formalism. (3D elasticity and conduction accept
    # any anisotropy — see the anisotropic testset below.)
    C2aniso = Tens(get_array(TensISO{2}(3.0, 2.0)))
    @test_throws ArgumentError interaction_tensor(_disk(1.0), _disk(1.0), [5.0, 0.0], C2aniso)
    # Non-spherical geometry forced onto the closed-form kernel
    @test_throws ArgumentError interaction_tensor(
        Ellipsoid(1.0, 0.5, 0.25), _ball(1.0), [0.0, 0.0, 5.0], C₀; method = :analytical
    )
    # Unsupported multipole order
    @test_throws ArgumentError interaction_tensor(
        _ball(1.0), _ball(1.0), [0.0, 0.0, 5.0], C₀; method = :multipole, order = 1
    )
end

@testset "𝕋^{ab} — ForwardDiff" begin
    C₀ = _C3()
    fR = R -> get_array(interaction_tensor(_ball(1.0), _ball(1.0), [0.0, 0.0, R], C₀))[3, 3, 3, 3]
    @test ForwardDiff.derivative(fR, 5.0) ≈ (fR(5.0 + 1.0e-6) - fR(5.0 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-6

    fμ = μ -> get_array(
        interaction_tensor(_ball(1.0), _ball(1.0), [0.0, 0.0, 5.0], TensISO{3}(3.0, 2μ))
    )[3, 3, 3, 3]
    @test ForwardDiff.derivative(fμ, 1.0) ≈ (fμ(1.0 + 1.0e-6) - fμ(1.0 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-6

    # Radii too — this is what a sensitivity to a particle size goes through.
    fa = a -> get_array(interaction_tensor(_ball(a), _ball(1.0), [0.0, 0.0, 5.0], C₀))[3, 3, 3, 3]
    @test ForwardDiff.derivative(fa, 0.8) ≈ (fa(0.8 + 1.0e-6) - fa(0.8 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-6
end

@testset "Periodic images — spherical cutoff" begin
    # `periodic_images` must enumerate a BALL of images, not a box: the
    # convergence proof of Molinari & El Mouden (1996), App. B, relies on it.
    imgs = periodic_images([0.0, 0.0, 0.0], 1.0, 2.5; skip_self = true)
    @test all(norm(x) ≤ 2.5 + 1.0e-12 for x in imgs)
    @test all(norm(x) > 0 for x in imgs)
    # A simple cubic lattice of unit spacing has 6 nearest neighbors at 1.0,
    # 12 at √2 and 8 at √3 — 26 images within a radius of 1.8.
    close = periodic_images([0.0, 0.0, 0.0], 1.0, 1.8; skip_self = true)
    @test length(close) == 26
    # An empty cutoff yields no image at all (the cluster is the receiver).
    @test isempty(periodic_images([0.0, 0.0, 0.0], 1.0, 0.5; skip_self = true))
end

@testset "Lattice sum — cubic symmetry of the accumulated tensor" begin
    # Summed over a simple cubic lattice, Γ̄ must have cubic symmetry: the
    # three axial components coincide, and the isotropic part still vanishes.
    C₀ = _C3()
    Γ̄ = lattice_interaction_tensor(_ball(0.3), _ball(0.3), [0.0, 0.0, 0.0], C₀, 1.0, 3.0)
    A = get_array(Γ̄)
    @test A[1, 1, 1, 1] ≈ A[2, 2, 2, 2] rtol = 1.0e-12
    @test A[1, 1, 1, 1] ≈ A[3, 3, 3, 3] rtol = 1.0e-12
    @test A[1, 1, 2, 2] ≈ A[2, 2, 3, 3] rtol = 1.0e-12
    @test sum(A[i, i, j, j] for i in 1:3, j in 1:3) ≈ 0.0 atol = ATOL_ZERO
    @test sum(A[i, j, i, j] for i in 1:3, j in 1:3) ≈ 0.0 atol = ATOL_ZERO
    # No image inside the cutoff ⇒ exactly zero.
    Γ0 = lattice_interaction_tensor(_ball(0.3), _ball(0.3), [0.0, 0.0, 0.0], C₀, 1.0, 0.5)
    @test maximum(abs.(get_array(Γ0))) ≈ 0.0 atol = ATOL_ZERO
end


@testset "𝕋^{ab} — anisotropic reference (3D)" begin
    # Since `Core/green_aniso.jl` landed, an anisotropic reference is a
    # supported case rather than an error.
    #
    # Cost note: the quadrature oracle calls the anisotropic Green operator at
    # every node pair, and that operator is ~500x dearer than the isotropic
    # closed form. The node counts below are therefore deliberately small — the
    # pairs are well separated, so the integrand is smooth and a coarse rule is
    # still far more accurate than the multipole truncation being tested.
    ia, ib = _ball(1.0), _ball(0.8)
    C₀ = _C3()

    # (a) Isotropic values, generic type: the Barnett route must land back on
    #     the closed form. This is the sharpest check of the whole anisotropic
    #     chain, since the closed form is independently validated — and it is
    #     cheap, needing no quadrature at all.
    for r in ([1.0, 2.0, 3.0], [0.0, 0.0, 6.0])
        A = get_array(interaction_tensor(ia, ib, r, C₀))
        B = get_array(interaction_tensor(ia, ib, r, Tens(get_array(C₀))))
        @test maximum(abs.(A .- B)) < 1.0e-10 * maximum(abs.(A))
    end

    # (b) A genuinely anisotropic (cubic) reference against the quadrature.
    #     The multipole truncation is O((size/separation)^4), so the agreement
    #     must improve at that rate as the pair separates.
    arr = collect(get_array(C₀))
    for (i, j) in ((1, 2), (1, 3), (2, 3)), idx in (
                (i, j, i, j), (j, i, i, j), (i, j, j, i), (j, i, j, i),
            )
        arr[idx...] = 1.4
    end
    Ccub = Tens(arr)
    errs = Float64[]
    Qs = Any[]
    for R in (6.0, 12.0)
        r = [0.0, 0.0, R]
        M = get_array(interaction_tensor(ia, ib, r, Ccub))
        Q = get_array(interaction_tensor(ia, ib, r, Ccub; method = :quadrature, nodes = (4, 4, 8)))
        push!(errs, maximum(abs.(M .- Q)) / maximum(abs.(Q)))
        push!(Qs, Q)
    end
    @test errs[2] < errs[1]
    @test errs[end] < 1.0e-3

    # (c) With an anisotropic reference the isotropic part does NOT vanish —
    #     that invariant is specific to an isotropic Green operator. The
    #     quadrature value from (b) confirms it independently of the multipole.
    Q = Qs[end]
    @test abs(sum(Q[i, i, j, j] for i in 1:3, j in 1:3)) > 1.0e-3 * maximum(abs.(Q))

    # (d) Conduction with an anisotropic conductivity. Cheap: the anisotropic
    #     conduction Green operator is a closed form, not a quadrature.
    Kan = Tens([3.0 0.4 0.1; 0.4 2.0 -0.2; 0.1 -0.2 1.5])
    r = [0.0, 0.0, 8.0]
    M = get_array(interaction_tensor(ia, ib, r, Kan))
    Qk = get_array(interaction_tensor(ia, ib, r, Kan; method = :quadrature, nodes = (6, 6, 12)))
    @test maximum(abs.(M .- Qk)) < 1.0e-3 * maximum(abs.(Qk))
end
