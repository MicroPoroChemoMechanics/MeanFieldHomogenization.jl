using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_green_aniso.jl — real-space Green function and Green operator of an
#  anisotropic medium (Barnett line integral, ported from the exploratory
#  `Green.jl` of `echoes_cpp`).
#
#  The whole file rests on one idea: an isotropic stiffness *typed* as a
#  general tensor must go through the anisotropic machinery and come back with
#  the Kelvin closed form. That single cross-check validates the line integral,
#  its normalization, the nested-ForwardDiff second gradient and the index
#  bookkeeping at once — none of which has an independent oracle.
#
#  Coverage:
#   1. Barnett line integral reproduces the Kelvin Green function.
#   2. Green OPERATOR from the anisotropic route reproduces the isotropic
#      closed form; quadrature convergence with the node count.
#   3. Conduction: the anisotropic closed forms reduce to the isotropic ones in
#      2D and 3D, and satisfy `K₀ : G⁰ = 0`.
#   4. Genuine anisotropy: minor and major symmetries, homogeneity of degree -3,
#      and agreement with a finite difference of the Green function.
#   5. The dispatcher picks the closed form for an isotropic reference.
#   6. Error paths: the origin, and plane-strain elasticity (Stroh, not
#      implemented).
#   7. `ForwardDiff` through the anisotropic operator, and the memoized
#      Gauss-Legendre rule.
# =============================================================================

const RTOL_GA = 1.0e-12

_Ciso() = TensISO{3}(3 * 1.0, 2 * 0.4)          # k = 1, μ = 0.4 ⇒ ν = 0.3

# The same isotropic values, but typed as a general anisotropic tensor so that
# dispatch is forced through the Barnett route.
_Cgen() = Tens(get_array(_Ciso()))

# A genuinely anisotropic (cubic) stiffness: the isotropic one with its shear
# constant detuned, which is exactly what a cluster estimate on a cubic array
# produces.
function _Ccub(c44 = 0.55)
    # Typed on `c44` so the helper survives being handed a `ForwardDiff.Dual`:
    # `collect(get_array(...))` alone gives an `Array{Float64}`, into which a
    # dual cannot be stored.
    arr = promote_type(Float64, typeof(c44)).(collect(get_array(_Ciso())))
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        for idx in (
                (i, j, i, j), (j, i, i, j), (i, j, j, i), (j, i, j, i),
            )
            arr[idx...] = c44
        end
    end
    return Tens(arr)
end

@testset "Anisotropic Green function — reduces to Kelvin" begin
    C₀ = _Ciso()
    E, ν = MeanFieldHomogenization.Core.extract_iso_moduli(C₀)
    μ = E / (2 * (1 + ν))
    for x in ([1.3, -0.7, 2.1], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
        r = norm(x)
        n = x ./ r
        A = 1 / (16π * μ * (1 - ν))
        G_kelvin = A / r .* ((3 - 4ν) .* Matrix(I, 3, 3) .+ n * n')
        for nodes in (16, 32)
            G = green_function_aniso(C₀, x; nodes = nodes)
            @test maximum(abs.(G .- G_kelvin)) < 1.0e-13 * maximum(abs.(G_kelvin))
        end
    end
end

@testset "Anisotropic Green operator — reduces to the isotropic closed form" begin
    # For an ISOTROPIC stiffness the Christoffel inverse is a low-order
    # trigonometric polynomial, so the rule is exact almost immediately: the
    # operator is already at round-off by 16 nodes. Anything above that is
    # noise, so the assertion is an absolute floor rather than a trend.
    C₀ = _Ciso()
    for x in ([1.3, -0.7, 2.1], [2.0, 0.0, 0.0])
        Γiso = green_operator_iso(C₀, x)
        s = maximum(abs.(Γiso))
        for nodes in (16, 32, 64)
            @test maximum(abs.(green_operator_aniso(C₀, x; nodes = nodes) .- Γiso)) < 1.0e-12 * s
        end
        @test maximum(abs.(green_operator_aniso(C₀, x; nodes = 8) .- Γiso)) > 1.0e-12 * s
    end
end

@testset "Anisotropic Green operator — quadrature convergence" begin
    # A genuinely anisotropic reference is where the node count actually
    # matters: the error against a fine reference must fall monotonically, and
    # the package default (32) must sit well below the truncation error of the
    # multipole expansion that consumes it.
    C = _Ccub()
    x = [1.0, 2.0, 3.0]
    ref = green_operator_aniso(C, x; nodes = 128)
    s = maximum(abs.(ref))
    errs = [maximum(abs.(green_operator_aniso(C, x; nodes = n) .- ref)) / s for n in (16, 24, 32)]
    @test issorted(errs; rev = true)
    @test errs[end] < 1.0e-4
end

@testset "Anisotropic Green operator — conduction" begin
    x3 = [1.3, -0.7, 2.1]
    K₀ = TensISO{3}(2.0)
    @test maximum(abs.(green_operator_aniso(K₀, x3) .- green_operator_iso(K₀, x3))) ≈ 0.0 atol = 1.0e-15
    x2 = [1.3, -0.7]
    K2 = TensISO{2}(2.0)
    @test maximum(abs.(green_operator_aniso(K2, x2) .- green_operator_iso(K2, x2))) ≈ 0.0 atol = 1.0e-15

    # For an anisotropic reference the isotropic part does not simply vanish;
    # what survives is the K-weighted trace.
    Kan = Tens([3.0 0.4 0.1; 0.4 2.0 -0.2; 0.1 -0.2 1.5])
    G = green_operator_aniso(Kan, x3)
    Karr = get_array(Kan)
    @test sum(Karr[i, j] * G[i, j] for i in 1:3, j in 1:3) ≈ 0.0 atol = 1.0e-14
    @test G ≈ G' rtol = 1.0e-14
end

@testset "Anisotropic Green operator — genuine anisotropy" begin
    C = _Ccub()
    x = [1.0, 2.0, 3.0]
    Γ = green_operator_aniso(C, x; nodes = 64)
    @test maximum(abs.(Γ .- permutedims(Γ, (2, 1, 3, 4)))) ≈ 0.0 atol = 1.0e-14   # minor (ij)
    @test maximum(abs.(Γ .- permutedims(Γ, (1, 2, 4, 3)))) ≈ 0.0 atol = 1.0e-14   # minor (kl)
    @test maximum(abs.(Γ .- permutedims(Γ, (3, 4, 1, 2)))) ≈ 0.0 atol = 1.0e-14   # major
    # Homogeneous of degree -3: two derivatives of a degree -1 function.
    @test maximum(abs.(green_operator_aniso(C, 2 .* x; nodes = 64) .- Γ ./ 8)) <
        1.0e-13 * maximum(abs.(Γ))
    # And it really is anisotropic — otherwise the test above proves nothing.
    @test !isapprox(Γ, green_operator_aniso(_Ciso(), x; nodes = 64); rtol = 1.0e-2)

    # Independent check: finite-difference the Green function itself.
    h = 1.0e-4
    d2 = (k, l) -> begin
        e_k = [i == k ? h : 0.0 for i in 1:3]
        e_l = [i == l ? h : 0.0 for i in 1:3]
        (
            green_function_aniso(C, x .+ e_k .+ e_l; nodes = 64) .-
                green_function_aniso(C, x .+ e_k .- e_l; nodes = 64) .-
                green_function_aniso(C, x .- e_k .+ e_l; nodes = 64) .+
                green_function_aniso(C, x .- e_k .- e_l; nodes = 64)
        ) ./ (4h^2)
    end
    H = [d2(j, l)[i, k] for i in 1:3, k in 1:3, j in 1:3, l in 1:3]
    # 𝔾⁰ = -[∂²G]_{(ij)(kl)}: the minus is the Brisard convention, and the
    # anisotropic route must carry it exactly as the closed form does.
    Gfd = [
        -(H[i, k, j, l] + H[j, k, i, l] + H[i, l, j, k] + H[j, l, i, k]) / 4
            for i in 1:3, j in 1:3, k in 1:3, l in 1:3
    ]
    @test maximum(abs.(Γ .- Gfd)) < 1.0e-5 * maximum(abs.(Γ))
end

@testset "green_operator — dispatch" begin
    x = [1.3, -0.7, 2.1]
    C₀ = _Ciso()
    # An isotropic reference takes the closed form, bit for bit.
    @test green_operator(C₀, x) == green_operator_iso(C₀, x)
    @test green_operator(TensISO{3}(2.0), x) == green_operator_iso(TensISO{3}(2.0), x)
    # The same values typed generically go the long way round and land back.
    @test maximum(abs.(green_operator(_Cgen(), x) .- green_operator_iso(C₀, x))) <
        1.0e-10 * maximum(abs.(green_operator_iso(C₀, x)))
    # `green_nodes` reaches the anisotropic branch and is ignored by the
    # isotropic one, so both can be passed together with a quadrature `nodes`.
    @test green_operator(C₀, x; green_nodes = 8) == green_operator_iso(C₀, x)

    # The public API is tensor-valued throughout; the `SArray` kernels the hot
    # loops use are the `_`-prefixed twins, not these.
    @test green_operator(C₀, x) isa TensND.AbstractTens{4, 3}
    @test green_operator(_Cgen(), x) isa TensND.AbstractTens{4, 3}
    @test green_operator_aniso(TensISO{3}(2.0), x) isa TensND.AbstractTens{2, 3}
    @test green_function_aniso(C₀, x) isa TensND.AbstractTens{2, 3}
    K = MeanFieldHomogenization.Core._green_operator(C₀, x)
    @test K isa AbstractArray && !(K isa TensND.AbstractTens)
end

@testset "Anisotropic Green — error paths" begin
    @test_throws DomainError green_function_aniso(_Ciso(), [0.0, 0.0, 0.0])
    @test_throws DomainError green_operator_aniso(TensISO{3}(2.0), [0.0, 0.0, 0.0])
    # Plane-strain elasticity with an anisotropic reference needs Stroh.
    C2 = Tens(get_array(TensISO{2}(3.0, 2.0)))
    @test_throws ArgumentError green_operator(C2, [1.0, 2.0])
end

@testset "Anisotropic Green — ForwardDiff and the memoized rule" begin
    C = _Ccub()
    f = t -> green_operator_aniso(C, [1.0, 2.0, 3.0 + t]; nodes = 32)[1, 1, 1, 1]
    @test ForwardDiff.derivative(f, 0.0) ≈ (f(1.0e-6) - f(-1.0e-6)) / 2.0e-6 rtol = 1.0e-6

    # Differentiating with respect to a *modulus* also has to work — that is
    # what a sensitivity through an N-body scheme in an anisotropic reference
    # goes through.
    g = c44 -> green_operator_aniso(_Ccub(c44), [1.0, 2.0, 3.0]; nodes = 32)[1, 2, 1, 2]
    @test ForwardDiff.derivative(g, 0.55) ≈ (g(0.55 + 1.0e-6) - g(0.55 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-5

    # The Gauss-Legendre rule is memoized by node count; asking twice must give
    # the identical arrays, and the nodes must lie in the requested interval.
    t1, w1 = gauss_legendre_nodes(24, 0.0, 2π)
    t2, w2 = gauss_legendre_nodes(24, 0.0, 2π)
    @test t1 == t2 && w1 == w2
    @test all(0 .≤ t1 .≤ 2π)
    @test sum(w1) ≈ 2π rtol = 1.0e-14
end
