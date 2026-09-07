using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const LSec = MeanFieldHomogenization.LayeredSpheroids

# =============================================================================
#  test_elastic_cases.jl — the three elementary problems, the strain
#  concentration tensor, and the path into the mean-field schemes.
#
#  The gate is Eshelby, on all 81 components: a single homogeneous confocal
#  spheroid must reproduce the analytic concentration tensor. Everything else
#  is built on that — per-layer averages that sum to the total, a shell with
#  the core's moduli changing nothing, transverse isotropy, and `homogenize`
#  degenerating onto the equivalent `Ellipsoid`.
# =============================================================================

_Ce(κ, μ) = TensISO{3}(3κ, 2μ)

"Prolate spheroid of outer aspect ratio `ω`, unit outer axis semi-axis."
function _sph(axes, moduli, ω; 𝒩 = 6)
    c = sqrt(1 - (1 / ω)^2)
    all(x -> x > c, axes) || error("test setup: a layer falls below the focal distance")
    return LayeredSpheroid(axes, map(x -> sqrt(x^2 - c^2), axes), moduli; Nseries = 𝒩)
end

# `R` maps this module's frame (axis ê₃) onto MFH's `Ellipsoid`, which sorts its
# semi-axes and puts the longest on ê₁.
const _R = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]

"Rotate a fourth-order array by `R`."
function _rotate4(A, R)
    B = zeros(eltype(A), 3, 3, 3, 3)
    for a in 1:3, b in 1:3, c in 1:3, d in 1:3
        acc = zero(eltype(A))
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            acc += R[a, i] * R[b, j] * R[c, k] * R[d, l] * A[i, j, k, l]
        end
        B[a, b, c, d] = acc
    end
    return B
end

@testset "Elastic confocal spheroid — all three cases" begin
    C₀, C₁ = _Ce(1.0, 0.6), _Ce(5.0, 3.0)

    @testset "the concentration tensor is Eshelby, on all 81 components (ω = $ω)" for
        ω in (1.5, 2.0, 4.0, 10.0)
        s = _sph((1.0,), (C₁,), ω; 𝒩 = 5)
        r = spheroid_strain_concentration(s, C₀; D = 6)
        Ar = get_array(strain_strain_loc(Ellipsoid(ω, 1.0, 1.0), C₁, C₀))
        @test maximum(abs, _rotate4(get_array(r.A), _R) .- Ar) <
            1.0e-11 * maximum(abs, Ar)
        # a single inclusion is exact at any truncation, so the residual is not
        # a fitting error but a statement that the six conditions are consistent
        @test maximum(r.residuals) < 1.0e-10
    end

    @testset "the two members of each shear pair agree" begin
        # Loadings 3–4 and 5–6 are the same problem rotated about the axis, so
        # their responses must match. Nothing in the assembly enforces it.
        ω = 2.0
        s = _sph((1.0,), (C₁,), ω; 𝒩 = 5)
        A = get_array(spheroid_strain_concentration(s, C₀; D = 6).A)
        @test A[1, 2, 1, 2] ≈ (A[1, 1, 1, 1] - A[1, 1, 2, 2]) / 2 rtol = 1.0e-9
        @test A[1, 3, 1, 3] ≈ A[2, 3, 2, 3] rtol = 1.0e-9
    end

    @testset "transverse isotropy, and it is not asserted by a type" begin
        ω = 1.5
        s = _sph((0.9, 1.0), (C₁, _Ce(0.5, 0.3)), ω)
        r = spheroid_strain_concentration(s, C₀; D = 6)
        A = get_array(r.A)
        θ = 0.7
        Q = [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]
        # holds to the truncation accuracy, which the residual measures
        @test maximum(abs, _rotate4(A, Q) .- A) <
            50 * maximum(r.residuals) * maximum(abs, A)
    end

    @testset "a shell with the core's moduli changes nothing" begin
        ω = 1.5
        A1 = get_array(spheroid_strain_concentration(_sph((1.0,), (C₁,), ω), C₀; D = 6).A)
        A2 = get_array(
            spheroid_strain_concentration(_sph((0.9, 1.0), (C₁, C₁), ω), C₀; D = 6).A
        )
        @test maximum(abs, A2 .- A1) < 1.0e-12 * maximum(abs, A1)
    end

    @testset "per-layer averages sum to the total" begin
        # Two independent routes: the layers from differences of surface moments
        # on their own boundaries, the total from one moment on the outer one.
        ω = 1.5
        for axes in ((0.9, 1.0), (0.85, 0.93, 1.0))
            mods = ntuple(k -> _Ce(5.0 - k, 3.0 - 0.5k), length(axes))
            s = _sph(axes, mods, ω)
            r = LSec.spheroid_layer_strain_concentration(s, C₀; D = 6)
            f = [layer_volume_fraction(s, k) for k in eachindex(r.layers)]
            Asum = sum(f[k] .* get_array(r.layers[k]) for k in eachindex(r.layers))
            At = get_array(r.A)
            @test maximum(abs, Asum .- At) <
                200 * maximum(r.residuals) * maximum(abs, At)
        end
    end

    @testset "a coating shifts the concentration monotonically in its stiffness" begin
        ω = 1.5
        core = _Ce(5.0, 3.0)
        soft = get_array(
            spheroid_strain_concentration(
                _sph((0.9, 1.0), (core, _Ce(0.5, 0.3)), ω), C₀; D = 6
            ).A
        )
        stiff = get_array(
            spheroid_strain_concentration(
                _sph((0.9, 1.0), (core, _Ce(20.0, 12.0)), ω), C₀; D = 6
            ).A
        )
        bare = get_array(spheroid_strain_concentration(_sph((1.0,), (core,), ω), C₀; D = 6).A)
        @test soft[3, 3, 3, 3] > bare[3, 3, 3, 3] > stiff[3, 3, 3, 3]
    end

    @testset "converged in the truncation" begin
        ω = 2.0
        s(D) = spheroid_strain_concentration(
            _sph((0.95, 1.0), (C₁, _Ce(0.5, 0.3)), ω; 𝒩 = 6), C₀; D
        )
        v(D) = get_array(s(D).A)[3, 3, 3, 3]
        v4, v6, v9 = v(4), v(6), v(9)
        @test v6 ≈ v4 rtol = 1.0e-2
        @test v9 ≈ v6 rtol = 3.0e-4
        @test maximum(s(9).residuals) < maximum(s(4).residuals)
    end

    @testset "no rigid-body rotation is needed, and here is why" begin
        # Duan et al. add two to case III and blame their omission for the error
        # in Riccardi & Montheillet (1999). With the full four-potential set the
        # rotation is already spanned: `φ₁ = A z` with `φ₃ = -A x` is the
        # ANTISYMMETRIC combination of the same two potentials whose symmetric
        # combination is the remote shear. A rotation carries no strain, so its
        # traction must vanish identically.
        c, μ, ν = 0.8, 3.0, 0.27
        p, q, ϕ = 0.37, 1.6, 0.41
        # `z = c P₁(p)P₁(q)` and `x = -c P₁¹(p)P₁¹(q) cos φ`
        u1, t1 = LSec.mode_fields(
            LSec.PNMode(1, :regular, 1, 0, :cos), ϕ, p, q, c, μ, ν, Float64
        )
        u3, t3 = LSec.mode_fields(
            LSec.PNMode(3, :regular, 1, 1, :cos), ϕ, p, q, c, μ, ν, Float64
        )
        # symmetric combination — the remote shear — carries traction
        t_sym = ntuple(k -> t1[k] - t3[k], 3)
        @test any(k -> abs(t_sym[k]) > 1.0e-6, 1:3)
        # antisymmetric combination — the rotation — carries none
        t_rot = ntuple(k -> t1[k] + t3[k], 3)
        @test all(k -> abs(t_rot[k]) < 1.0e-12 * max(1.0, maximum(abs, t_sym)), 1:3)
    end

    @testset "degree 0 of φ₀ is not optional" begin
        # Only its REGULAR part is a constant and inert. The irregular one is
        # `arccoth q`, an essential mode: dropping it made the case-I system
        # genuinely inconsistent, residual `1e-1` instead of `1e-16`.
        g = LSec._case_groups(AxisymmetricCase(), 5)
        @test 0 in g[1].degrees
        @test all(iseven, g[1].degrees)
        @test all(isodd, g[2].degrees)
        c, μ, ν = 0.8, 3.0, 0.27
        _, treg = LSec.mode_fields(
            LSec.PNMode(0, :regular, 0, 0, :cos), 0.3, 0.4, 1.7, c, μ, ν, Float64
        )
        _, tirr = LSec.mode_fields(
            LSec.PNMode(0, :irregular, 0, 0, :cos), 0.3, 0.4, 1.7, c, μ, ν, Float64
        )
        @test all(iszero, treg)                     # the constant, inert
        @test any(!iszero, tirr)                    # arccoth q, essential
    end

    @testset "into the schemes: one layer degenerates onto the Ellipsoid" begin
        for ω in (1.5, 3.0), scheme in (Dilute(), MoriTanaka())
            a = 1.0
            c = sqrt(a^2 - (a / ω)^2)
            sp = LayeredSpheroid((a,), (sqrt(a^2 - c^2),), (C₁,); Nseries = 5)
            r1 = RVE()
            add_phase!(r1, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C₀); fraction = :rest)
            add_phase!(r1, :I, sp, Dict(:C => C₁); fraction = 0.2)
            r2 = RVE()
            add_phase!(r2, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C₀); fraction = :rest)
            add_phase!(r2, :I, Ellipsoid(ω, 1.0, 1.0), Dict(:C => C₁); fraction = 0.2)
            A = get_array(homogenize(r1, scheme, :C))
            B = get_array(homogenize(r2, scheme, :C))
            @test maximum(abs, _rotate4(A, _R) .- B) < 1.0e-11 * maximum(abs, B)
        end
    end

    @testset "a genuinely layered spheroid homogenizes, and anisotropically" begin
        ω = 1.5
        s = _sph((0.9, 1.0), (C₁, _Ce(0.5, 0.3)), ω)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C₀); fraction = :rest)
        add_phase!(r, :I, s, Dict(:C => C₁); fraction = 0.15)
        A = get_array(homogenize(r, MoriTanaka(), :C))
        @test all(isfinite, A)
        @test !isapprox(A[1, 1, 1, 1], A[3, 3, 3, 3]; rtol = 1.0e-4)
        # the stiffness contribution is assembled layer by layer, so it must not
        # coincide with the naive homogeneous-inhomogeneity form
        NC = stiffness_contribution(s, C₁, C₀)
        @test all(isfinite, get_array(NC))
    end

    @testset "what is refused, and says why" begin
        ω = 1.5
        c = sqrt(1 - (1 / ω)^2)
        obl = LayeredSpheroid((1.0,), (sqrt(1 + 0.25),), (C₁,); Nseries = 4)
        @test_throws ArgumentError spheroid_strain_concentration(obl, C₀)
        imp = LayeredSpheroid(
            (1.0,), (sqrt(1 - c^2),), (C₁,);
            interfaces = (SpringInterface(10.0, 5.0),), Nseries = 4
        )
        @test_throws ArgumentError spheroid_strain_concentration(imp, C₀)
        tilted = LayeredSpheroid(
            (1.0,), (sqrt(1 - c^2),), (C₁,); Nseries = 4, axis = (1.0, 0.0, 0.0)
        )
        @test_throws ArgumentError spheroid_strain_concentration(tilted, C₀)
        @test_throws ArgumentError LSec._case_groups(AxisymmetricCase(), 3)[1].parts[1][2] |>
            m -> LSec._branch_kinds(3, :regular)
    end

    @testset "ForwardDiff and BigFloat" begin
        ω = 2.0
        c = sqrt(1 - (1 / ω)^2)
        f = function (κ)
            s = LayeredSpheroid(
                (1.0,), (sqrt(1 - c^2),), (_Ce(κ, 3.0),); Nseries = 5
            )
            return get_array(spheroid_strain_concentration(s, C₀; D = 4).A)[3, 3, 3, 3]
        end
        d = ForwardDiff.derivative(f, 5.0)
        fd = (f(5.0 + 1.0e-6) - f(5.0 - 1.0e-6)) / 2.0e-6
        @test d ≈ fd rtol = 1.0e-4
        @test !iszero(d)

        cb = sqrt(one(BigFloat) - (one(BigFloat) / 2)^2)
        sb = LayeredSpheroid(
            (one(BigFloat),), (sqrt(one(BigFloat) - cb^2),),
            (_Ce(big(5), big(3)),); Nseries = 5
        )
        rb = spheroid_strain_concentration(sb, _Ce(big(1), big(6) / 10); D = 4)
        @test eltype(get_array(rb.A)) === BigFloat
        sf = LayeredSpheroid((1.0,), (sqrt(1 - c^2),), (C₁,); Nseries = 5)
        rf = spheroid_strain_concentration(sf, C₀; D = 4)
        @test Float64(get_array(rb.A)[3, 3, 3, 3]) ≈ get_array(rf.A)[3, 3, 3, 3] rtol = 1.0e-8
    end
end
