# =============================================================================
#  test_conformal_holes.jl — cylindrical cavities of conformal cross-section.
#
#  References, independent of the finite system the code solves:
#   1. the elliptic cylinder of the package (Hill tensor), on every component,
#      in elasticity and in conduction;
#   2. the boundary condition of the plane problem solved by collocation on the
#      circle, with long series for the two potentials;
#   3. closed forms of the hypotrochoid, and the published values of Ekneligoda
#      & Zimmerman (2006, 2008), data/literature/ekneligodaPRSA200{6,8}.json;
#   4. exact properties: symmetry of the compliance, rotation of the hole,
#      isotropy in the plane of a hole with a threefold axis.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

const CH = MeanFieldHomogenization.ConformalHoles

_ch_K, _ch_μ = 1.7, 0.6
_ch_ν = (3 * _ch_K - 2 * _ch_μ) / (2 * (3 * _ch_K + _ch_μ))
_ch_C₀ = iso_stiffness(_ch_K, _ch_μ)
_ch_k₀ = TensISO{3}(1.3)
_ch_A(hole, C₀ = _ch_C₀) = canonical_components(gradient_gradient_concentration(hole, 0 * C₀, C₀))
_ch_H(hole) = canonical_components(inverse_property_contribution(hole, 0 * _ch_C₀, _ch_C₀))
# Normalized compressibility and shear compliance of the plane-strain problem,
# in the plane (2, 3): G Cpc/(1-ν), the hydrostatic stress of the plane with
# its axial reaction ν (σ₂₂ + σ₃₃), and G H₂₃₂₃/(1-ν) in the engineering sense.
function _ch_compressibility(hole)
    H = _ch_H(hole)
    σ = (2 * _ch_ν, 1.0, 1.0)
    return sum(H[i, i, j, j] * σ[j] for i in 2:3, j in 1:3) * _ch_μ / (1 - _ch_ν)
end
_ch_shear(hole) = 2 * _ch_μ * _ch_H(hole)[2, 3, 2, 3] / (1 - _ch_ν)
# the same, averaged over the directions of the shear in the plane
function _ch_mean_shear(hole)
    H = _ch_H(hole)
    return _ch_μ * (H[2, 3, 2, 3] + (H[2, 2, 2, 2] - 2H[2, 2, 3, 3] + H[3, 3, 3, 3]) / 4) / (1 - _ch_ν)
end
_ch_literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))["tables"]

# The plane problem by collocation. φ = Γ ζ + Σ aₖ ζ⁻ᵏ and ψ = Γ′ ζ + Σ bₖ ζ⁻ᵏ,
# k ≤ M; the free surface of the hole reads, on the circle,
#   φ(σ) + ω(σ) conj(φ′(σ)) / conj(ω′(σ)) + conj(ψ(σ)) = c,
# written at P points on the real and imaginary parts of the unknowns and
# solved in the least-squares sense. Returns the aₖ.
function _ch_collocation(m::Vector{ComplexF64}, Γ, Γ′; M = 50, P = 500)
    ω(σ) = σ + sum(m[k] * σ^(-k) for k in eachindex(m))
    dω(σ) = 1 - sum(k * m[k] * σ^(-k - 1) for k in eachindex(m))
    A, rhs = zeros(2P, 4M + 2), zeros(2P)
    for (p, θ) in enumerate(range(0, 2π; length = P + 1)[1:P])
        σ = cis(θ)
        r = ω(σ) / conj(dω(σ))
        row = zeros(ComplexF64, 4M + 2)
        for k in 1:M
            s, t = σ^(-k), r * k * conj(σ^(-k - 1))
            row[k], row[M + k] = s - t, im * (s + t)                    # Re aₖ, Im aₖ
            row[2M + k], row[3M + k] = conj(s), -im * conj(s)           # Re bₖ, Im bₖ
        end
        row[4M + 1], row[4M + 2] = -1, -im                              # the constant
        b = -(Γ * σ + r * conj(Γ) + conj(Γ′ * σ))
        A[2p - 1, :], A[2p, :] = real.(row), imag.(row)
        rhs[2p - 1], rhs[2p] = real(b), imag(b)
    end
    x = A \ rhs
    return complex.(x[1:M], x[(M + 1):(2M)])
end

@testset "Conformal holes" begin

    @testset "the map" begin
        @test first.(polygon_map(4, 2)) == [3, 7]
        @test last.(polygon_map(4, 2)) ≈ [1 / 6, 1 / 56]       # ω′ = (1 - ζ⁻⁴)^{1/2}
        @test first.(polygon_map(3, 3)) == [2, 5, 8]
        @test last.(polygon_map(3, 2)) ≈ [1 / 3, 1 / 45]
        @test_throws ArgumentError polygon_map(2, 3)
        @test_throws ArgumentError polygon_map(4, 0)
        # the map of the polygon: Σ k|mₖ| tends to 1 from below
        @test sum(k * abs(m) for (k, m) in polygon_map(5, 200)) < 1
        # 1/ω′ as a series: (1 - m ζ⁻²)⁻¹ = Σ mˡ ζ⁻²ˡ
        g = CH._inverse_derivative_series(ComplexF64[0.3], 6)
        @test g ≈ [1, 0, 0.3, 0, 0.09, 0, 0.027]
        @test CH._hole_area(ComplexF64[0.2, 0.1im]) ≈ 1 - 0.04 - 2 * 0.01
    end

    @testset "the hole and its contract" begin
        h = ConformalHole((0.2, 0.0, 0.1))
        @test h isa ConformalHole{Float64}
        @test ConformalHole([0.2, 0.0, 0.1]).coefficients == h.coefficients
        @test ConformalHole(1 => 0.2, 3 => 0.1).coefficients == h.coefficients
        @test ConformalHole(3 => 0.1, 1 => 1 // 5).coefficients ≈ h.coefficients
        @test ConformalHole((0, 0)) isa ConformalHole{Float64}
        @test ConformalHole((0.1 + 0.2im,)).coefficients == [0.1 + 0.2im]
        @test ConformalHole(()).coefficients == [0.0 + 0.0im]           # the circle
        @test dimension(h) == 3
        @test shape_trait(h) isa ConformalShape
        @test !is_homogeneous_inclusion(h)
        @test inclusion_basis(h) isa TensND.AbstractBasis
        @test occursin("ConformalHole", sprint(show, h))
        @test check_inclusion_interface(h; verbose = false)
        @inferred gradient_gradient_concentration(h, 0 * _ch_C₀, _ch_C₀)
        @inferred gradient_gradient_concentration(h, 0 * _ch_k₀, _ch_k₀)
        # nothing passes through the cavity
        @test iszero(canonical_components(flux_gradient_concentration(h, 0 * _ch_C₀, _ch_C₀)))
        @test iszero(canonical_components(flux_gradient_concentration(h, 0 * _ch_k₀, _ch_k₀)))
        # the property of the phase is not read
        @test _ch_A(h) ≈ canonical_components(gradient_gradient_concentration(h, 3 * _ch_C₀, _ch_C₀))
    end

    @testset "the ellipse: the elliptic cylinder of the package" begin
        for (b, c) in ((1.0, 1.0), (1.0, 0.5), (1.0, 0.1), (0.4, 1.0))
            hole, cylinder = ConformalHole(((b - c) / (b + c),)), Cylinder(b, c)
            A = _ch_A(cylinder)
            @test _ch_A(hole) ≈ A rtol = 1.0e-11
            Ak = canonical_components(gradient_gradient_concentration(cylinder, 0 * _ch_k₀, _ch_k₀))
            @test canonical_components(gradient_gradient_concentration(hole, 0 * _ch_k₀, _ch_k₀)) ≈ Ak rtol = 1.0e-12
        end
        # another matrix, nearly incompressible
        C₀ = iso_stiffness(40.0, 0.3)
        @test _ch_A(ConformalHole((0.5,)), C₀) ≈ _ch_A(Cylinder(3.0, 1.0), C₀) rtol = 1.0e-10
        # the same in a tilted basis
        angles = (0.4, 1.1, -0.7)
        @test _ch_A(ConformalHole((0.5,); euler_angles = angles)) ≈ _ch_A(Cylinder(3.0, 1.0; euler_angles = angles)) rtol = 1.0e-10
        @test _ch_A(ConformalHole((0.5,); basis = inclusion_basis(Cylinder(3.0, 1.0; euler_angles = angles)))) ≈ _ch_A(Cylinder(3.0, 1.0; euler_angles = angles)) rtol = 1.0e-10
    end

    @testset "the potentials by collocation" begin
        # smooth contours, for which the series of ψ converges fast
        for coefficients in ((0.1 + 0.05im, 0.1, -0.04im, 0.03), (0.0, 0.15, 0.0, 0.0, 0.02), (-0.2, 0.05im, 0.04))
            m = ComplexF64[coefficients...]
            a, loads = CH._potential_coefficients(m)
            for (q, (Γ, Γ′)) in enumerate(loads)
                reference = _ch_collocation(m, Γ, Γ′)
                @test a[:, q] ≈ reference[1:length(m)] atol = 1.0e-7
                # the potential φ stops at the degree of the map
                @test maximum(abs, reference[(length(m) + 1):end]) < 1.0e-5
            end
        end
    end

    @testset "exact properties" begin
        hole = ConformalHole((0.2 + 0.1im, 0.05, -0.03im, 0.02))
        # the compliance contribution is symmetric and positive
        H = TensND.mandel(inverse_property_contribution(hole, 0 * _ch_C₀, _ch_C₀))
        @test H ≈ H' rtol = 1.0e-12
        @test minimum(eigvals(Symmetric(Matrix(H)))) > 0
        # turning the hole by θ about its axis multiplies mₖ by e^{i(k+1)θ}
        θ = 0.37
        turned = ConformalHole(Tuple(m * cis((k + 1) * θ) for (k, m) in pairs(hole.coefficients)))
        R = [1 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
        A = _ch_A(hole)
        @test _ch_A(turned) ≈ [sum(R[i, p] * R[j, q] * R[k, r] * R[l, s] * A[p, q, r, s] for p in 1:3, q in 1:3, r in 1:3, s in 1:3) for i in 1:3, j in 1:3, k in 1:3, l in 1:3] rtol = 1.0e-11
        Ak = canonical_components(gradient_gradient_concentration(hole, 0 * _ch_k₀, _ch_k₀))
        @test canonical_components(gradient_gradient_concentration(turned, 0 * _ch_k₀, _ch_k₀)) ≈ R * Ak * R' rtol = 1.0e-12
        # a threefold axis makes the plane isotropic; a fourfold one does not
        H = _ch_H(ConformalHole(polygon_map(3, 3)...))
        @test H[2, 2, 2, 2] ≈ H[3, 3, 3, 3] rtol = 1.0e-12
        @test 2H[2, 3, 2, 3] ≈ H[2, 2, 2, 2] - H[2, 2, 3, 3] rtol = 1.0e-12
        @test abs(H[2, 2, 2, 3]) < 1.0e-13
        H = _ch_H(ConformalHole(polygon_map(4, 3)...))
        @test H[2, 2, 2, 2] ≈ H[3, 3, 3, 3] rtol = 1.0e-12
        @test abs(2H[2, 3, 2, 3] - (H[2, 2, 2, 2] - H[2, 2, 3, 3])) > 0.1
        # the cavity is a prism: its axial strain is the remote one
        @test A[1, 1, 1, 1] ≈ 1
        @test maximum(abs, (A[1, 1, 2, 2], A[1, 1, 3, 3], A[1, 1, 2, 3], A[1, 1, 1, 2])) < 1.0e-13
    end

    @testset "hypotrochoids in closed form" begin
        # z = ζ + m ζ⁻ⁿ. Compressibility 2 (1 + n m²)/(1 - n m²), the formula
        # the articles recall; the shear compliance is 2/(1 - n m²) for n = 2
        # and 2/((1 - (n-2) m²)(1 - n m²)) for n ≥ 4, from the same system.
        for (n, m) in ((2, 0.3), (2, 0.5), (3, 0.2), (4, 0.1), (5, 0.15), (7, 1 / 7))
            hole = ConformalHole(n => m)
            @test _ch_compressibility(hole) ≈ 2 * (1 + n * m^2) / (1 - n * m^2) rtol = 1.0e-12
            n == 2 && @test _ch_shear(hole) ≈ 2 / (1 - n * m^2) rtol = 1.0e-12
            n ≥ 4 && @test _ch_shear(hole) ≈ 2 / ((1 - (n - 2) * m^2) * (1 - n * m^2)) rtol = 1.0e-12
        end
        # conduction of the insulating section: 2/(1 - n m²) in the plane
        Ak = canonical_components(gradient_gradient_concentration(ConformalHole(3 => 0.2), 0 * _ch_k₀, _ch_k₀))
        @test Ak ≈ Diagonal([1, 2 / (1 - 3 * 0.04), 2 / (1 - 3 * 0.04)])
    end

    @testset "published values (Ekneligoda & Zimmerman)" begin
        hole(row) = haskey(row, "m2") ? ConformalHole(2 => row["m1"], 5 => row["m2"]) : ConformalHole(2 => row["m1"])
        compressibility = _ch_literature("ekneligodaPRSA2006")
        for row in [compressibility["two_terms"]; compressibility["three_terms"]]
            @test _ch_compressibility(hole(row)) ≈ row["analytical"] atol = 1.0e-3
        end
        # regular polygons: the article extrapolates maps of three and four
        # terms, and the map converges here
        for row in compressibility["polygons"]
            converged = _ch_compressibility(ConformalHole(polygon_map(row["sides"], 300)...))
            @test converged ≈ row["extrapolated"] atol = 2.0e-3
            @test converged ≈ row["boundary_elements"] rtol = 3.0e-3
        end
        @test _ch_compressibility(ConformalHole(polygon_map(4, 300)...)) ≈ 1 + sqrt(2) atol = 1.0e-4
        shear = _ch_literature("ekneligodaPRSA2008")
        for row in shear["two_terms"]
            @test _ch_shear(hole(row)) ≈ row["analytical"] atol = 1.5e-4
        end
        # Of the three maps of three terms, the article and this solution agree
        # on the last; on the two others they differ by 1.3 % and 5.6 %, and the
        # collocation above confirms the potentials for maps of that family.
        rows = shear["three_terms"]
        @test _ch_shear(hole(rows[3])) ≈ rows[3]["analytical"] atol = 1.0e-4
        @test _ch_shear(hole(rows[1])) ≈ 2.6828 atol = 1.0e-4
        @test _ch_shear(hole(rows[2])) ≈ 2.4694 atol = 1.0e-4
        # quasi-polygons: the article counts the term ζ. The pentagon of two
        # terms is the hypotrochoid n = 4, m = 1/10 of the closed form above,
        # 2.1259, where the article prints 2.1321.
        for row in shear["quasi_polygons"]
            n = row["sides"]
            n == 5 || @test _ch_mean_shear(ConformalHole(polygon_map(n, 1)...)) ≈ row["two_terms"] atol = 1.5e-4
            @test _ch_mean_shear(ConformalHole(polygon_map(n, 2)...)) ≈ row["three_terms"] atol = 1.0e-4
        end
    end

    @testset "derivatives and number types" begin
        # with respect to the shape, against the elliptic cylinder
        ellipse(m) = _ch_A(ConformalHole((m,)))[2, 2, 2, 2]
        cylinder(m) = _ch_A(Cylinder(1 + m, 1 - m))[2, 2, 2, 2]
        @test ForwardDiff.derivative(ellipse, 0.3) ≈ ForwardDiff.derivative(cylinder, 0.3) rtol = 1.0e-9
        @test ForwardDiff.derivative(m -> ForwardDiff.derivative(ellipse, m), 0.3) ≈
            ForwardDiff.derivative(m -> ForwardDiff.derivative(cylinder, m), 0.3) rtol = 1.0e-7
        # with respect to a coefficient of a longer map, against the closed form
        @test ForwardDiff.derivative(m -> _ch_compressibility(ConformalHole(2 => 0.2, 4 => m)), 0.0) ≈ 0 atol = 1.0e-12
        @test ForwardDiff.derivative(m -> _ch_compressibility(ConformalHole(4 => m)), 0.1) ≈ 32 * 0.1 / (1 - 4 * 0.01)^2 rtol = 1.0e-10
        # with respect to the matrix
        stiffness(K) = canonical_components(gradient_gradient_concentration(ConformalHole(polygon_map(4, 2)...), 0 * _ch_C₀, iso_stiffness(K, _ch_μ)))[2, 2, 2, 2]
        setprecision(BigFloat, 256) do
            h = big"1.0e-25"
            @test ForwardDiff.derivative(stiffness, _ch_K) ≈ Float64((stiffness(big(_ch_K) + h) - stiffness(big(_ch_K) - h)) / (2h)) rtol = 1.0e-9
        end
        @inferred gradient_gradient_concentration(ConformalHole((ForwardDiff.Dual(0.3, 1.0),)), 0 * _ch_C₀, _ch_C₀)
        @test eltype(_ch_A(ConformalHole((big"0.3",)))) == BigFloat
        @test ConformalHole((0.3f0,)) isa ConformalHole{Float32}
    end

    @testset "in a scheme" begin
        f = 0.12
        cell(pore) = (rve = RVE(); add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => _ch_C₀); fraction = :rest); add_phase!(rve, :pores, pore, Dict(:C => 0 * _ch_C₀); fraction = f); rve)
        for scheme in (Dilute(), MoriTanaka())
            @test canonical_components(homogenize(cell(ConformalHole((0.5,))), scheme, :C)) ≈ canonical_components(homogenize(cell(Cylinder(3.0, 1.0)), scheme, :C)) rtol = 1.0e-9
        end
        # the dilute estimate from the tensor of the hole
        hole = ConformalHole(polygon_map(3, 4)...)
        A = gradient_gradient_concentration(hole, 0 * _ch_C₀, _ch_C₀)
        @test canonical_components(homogenize(cell(hole), Dilute(), :C)) ≈ canonical_components(_ch_C₀ - f * (_ch_C₀ ⊡ A)) rtol = 1.0e-10
    end

    @testset "what is refused" begin
        @test_throws DomainError ConformalHole(2 => 0.6)                # 2 × 0.6 > 1: the map folds
        @test_throws DomainError ConformalHole((0.5, 0.3))
        @test_throws DomainError ConformalHole((1.0,))                  # a slit has no area
        @test_throws ArgumentError ConformalHole(0 => 0.1)
        @test ConformalHole(2 => 0.5) isa ConformalHole                 # the cusped limit
        hole = ConformalHole((0.3,))
        Cti = TensND.TensTI{4}(5.0, 1.0, 1.2, 6.0, 1.5, (0.0, 0.0, 1.0))
        @test_throws ArgumentError gradient_gradient_concentration(hole, Cti, Cti)
        @test_throws ArgumentError flux_gradient_concentration(hole, Cti, Cti)
        Kti = TensND.TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))
        @test_throws ArgumentError gradient_gradient_concentration(hole, Kti, Kti)
        @test_throws ArgumentError flux_gradient_concentration(hole, Kti, Kti)
    end
end
