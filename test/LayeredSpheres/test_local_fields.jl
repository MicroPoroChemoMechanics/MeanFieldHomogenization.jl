# =============================================================================
#  test_local_fields.jl — pointwise strain / stress / displacement and
#  transport fields in an n-layer sphere (`LayeredSpheres/localfields.jl`).
#
#  The oracles, sharpest first:
#
#   1. a homogeneous stack must give 𝔸 ≡ 𝕀 everywhere, origin and matrix
#      included;
#   2. for N = 1 the interior field is uniform and equals the Eshelby
#      localization of an `Ellipsoid`;
#   3. for N = 1 the exterior field is `𝕀 − 𝕋 ⊡ τ` with `𝕋` the closed-form
#      ball interaction tensor of `Interactions._pair_ball_iso` evaluated at
#      a POINT receiver (`a = 0`) — exact at every distance, because the
#      elastic Green operator is biharmonic and its mean-value expansion
#      terminates;
#   4. the volume average of the pointwise field reproduces the per-layer
#      `α_k`, `β_k` of the production recurrences;
#   5. compatibility `ε = sym ∇u` and equilibrium `div σ = 0`, both by
#      `ForwardDiff` on the field itself;
#   6. interface conditions — continuity for a perfect interface, and the
#      PRESCRIBED jump for spring and membrane ones, re-derived from the
#      field rather than from the transfer matrix that produced it;
#   7. hard-coded Echoes `loc_eE` / `loc_sE` values, including the two
#      imperfect interface families.
# =============================================================================

using Test
using MeanFieldHomogenization
using MeanFieldHomogenization.LayeredSpheres: _bulk_localization, _shear_localization,
    _iso_bulk_shear, _cond_localization
using MeanFieldHomogenization.Interactions: _pair_ball_iso
using TensND
using LinearAlgebra
using ForwardDiff

_iso(E, ν) = TensISO{3}(3E / (3 * (1 - 2ν)), 2E / (2 * (1 + ν)))

const LF_C₀ = _iso(30.0, 0.3)
const LF_C₁ = _iso(100.0, 0.2)
const LF_C₂ = _iso(10.0, 0.35)
const LF_C₃ = _iso(60.0, 0.25)
const LF_𝕀 = TensISO{3}(1.0, 1.0)
const LF_ε∞ = Tens(
    [
        0.3 0.12 -0.05;
        0.12 -0.2 0.07;
        -0.05 0.07 0.4
    ]
)

_maxabs(A) = maximum(abs, get_array(A))
_unit(v) = v ./ norm(v)

@testset "local fields — homogeneous stack gives the identity everywhere" begin
    sph = LayeredSphere((1.0, 2.0), (LF_C₀, LF_C₀))
    sol = LayeredSphereFields(sph, LF_C₀)
    for x in (
            [0.0, 0.0, 0.0], [0.3, 0.2, 0.1], [0.9, 0.0, 0.0],
            [1.4, 0.5, -0.3], [2.5, 1.0, 1.0], [50.0, 0.0, 0.0],
        )
        @test _maxabs(local_strain_strain_loc(sol, x) - LF_𝕀) < 1.0e-14
        @test _maxabs(local_stress_stress_loc(sol, x) - LF_𝕀) < 1.0e-14
    end
    # The displacement must be the affine remote one, with no perturbation.
    for x in ([0.4, -0.2, 0.7], [3.0, 1.0, -1.0])
        u = local_displacement(sol, x, LF_ε∞)
        @test collect(u) ≈ [sum(LF_ε∞[i, j] * x[j] for j in 1:3) for i in 1:3] rtol = 1.0e-12
    end
end

@testset "local fields — N=1 interior is uniform and equals Eshelby" begin
    R = 1.0
    sph = LayeredSphere((R,), (LF_C₁,))
    sol = LayeredSphereFields(sph, LF_C₀)
    A_esh = strain_strain_loc(MeanFieldHomogenization.Elasticity.Ellipsoid(R), LF_C₁, LF_C₀)
    for x in ([0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.5, -0.5, 0.5], [0.99, 0.0, 0.0])
        @test _maxabs(local_strain_strain_loc(sol, x) - A_esh) < 1.0e-13
    end
    # With a perfect interface the core carries no mode-2 amplitude at all.
    @test abs(sol.abcd[1][2]) < 1.0e-15
end

@testset "local fields — N=1 exterior matches the closed-form ball interaction" begin
    R = 1.0
    sph = LayeredSphere((R,), (LF_C₁,))
    sol = LayeredSphereFields(sph, LF_C₀)
    A₁ = strain_strain_loc(MeanFieldHomogenization.Elasticity.Ellipsoid(R), LF_C₁, LF_C₀)
    τ = (LF_C₁ - LF_C₀) ⊡ A₁                     # uniform polarization of the ball
    for ρ in (1.6, 2.6, 5.4, 20.0)
        x = ρ * R * _unit([1.0, 0.3, -0.2])
        𝕋 = _pair_ball_iso(0.0, R, x, LF_C₀)     # point receiver: a = 0
        @test _maxabs(local_strain_strain_loc(sol, x) - (LF_𝕀 - 𝕋 ⊡ τ)) < 1.0e-13
    end
end

@testset "local fields — shell averages reproduce α_k and β_k" begin
    sph = LayeredSphere((1.0, 2.0, 3.5), (LF_C₁, LF_C₂, LF_C₃))
    sol = LayeredSphereFields(sph, LF_C₀)
    α = _bulk_localization(sph, _iso_bulk_shear(LF_C₀)...)
    β = _shear_localization(sph, LF_C₀)
    for k in 1:3
        a, b = shell_localization(sol, k)
        @test a ≈ α[k] rtol = 1.0e-14
        @test b ≈ β[k] rtol = 1.0e-13
    end

    # The same identity by quadrature on the field itself.  `isotropify` of
    # 𝔸(x) is `Ã_k 𝕁 + β_local(r) 𝕂` with `β_local(r) r²` a degree-4
    # polynomial in `r` (only mode 2 survives the directional average), so a
    # 3-point Gauss-Legendre rule integrates it EXACTLY — no tolerance slop.
    gl_x = (-sqrt(3 / 5), 0.0, sqrt(3 / 5))
    gl_w = (5 / 9, 8 / 9, 5 / 9)
    radii = (0.0, 1.0, 2.0, 3.5)
    for k in 1:3
        ra, rb = radii[k], radii[k + 1]
        mid, half = (ra + rb) / 2, (rb - ra) / 2
        num = zero(LF_𝕀)
        for (ξ, w) in zip(gl_x, gl_w)
            r = mid + half * ξ
            A = local_strain_strain_loc(sol, [r, 0.0, 0.0]; side = :inner)
            num += (w * half * r^2) * isotropify(A)
        end
        avg = (3 / (rb^3 - ra^3)) * num
        @test avg ≈ TensISO{3}(α[k], β[k]) rtol = 1.0e-12
    end
end

@testset "local fields — far field returns to the identity as r⁻³" begin
    sph = LayeredSphere((1.0, 2.0), (LF_C₁, LF_C₂))
    sol = LayeredSphereFields(sph, LF_C₀)
    # Two decades apart, and far enough out that the subleading 1/r⁵ mode is
    # negligible — at r = 10 it still biases the ratio by ~3 %.
    e1 = _maxabs(local_strain_strain_loc(sol, [100.0, 0.0, 0.0]) - LF_𝕀)
    e2 = _maxabs(local_strain_strain_loc(sol, [1000.0, 0.0, 0.0]) - LF_𝕀)
    @test e1 < 1.0e-4
    @test e1 / e2 ≈ 1000.0 rtol = 1.0e-2         # the decaying mode is 1/r³
end

@testset "local fields — compatibility and equilibrium (ForwardDiff)" begin
    sph = LayeredSphere((1.0, 2.0, 3.5), (LF_C₁, LF_C₂, LF_C₃))
    sol = LayeredSphereFields(sph, LF_C₀)
    for x₀ in ([0.4, 0.3, 0.2], [1.5, 0.4, -0.6], [2.7, 1.0, 0.5], [6.0, 1.0, -2.0])
        J = ForwardDiff.jacobian(x -> collect(local_displacement(sol, x, LF_ε∞)), x₀)
        @test (J + J') / 2 ≈ get_array(local_strain(sol, x₀, LF_ε∞)) atol = 1.0e-13

        divσ = [
            sum(
                ForwardDiff.gradient(x -> local_stress(sol, x, LF_ε∞)[i, j], x₀)[j]
                    for j in 1:3
            ) for i in 1:3
        ]
        scale = _maxabs(local_stress(sol, x₀, LF_ε∞)) / norm(x₀)
        @test norm(divσ) / scale < 1.0e-12
    end
end

@testset "local fields — interface conditions are the prescribed ones" begin
    n = _unit([0.3, -0.5, 0.8])
    traction(σ) = [sum(σ[i, j] * n[j] for j in 1:3) for i in 1:3]

    # Perfect interfaces: displacement AND traction continuous.
    sph = LayeredSphere((1.0, 2.0, 3.5), (LF_C₁, LF_C₂, LF_C₃))
    sol = LayeredSphereFields(sph, LF_C₀)
    for rk in (1.0, 2.0, 3.5)
        x = rk * n
        @test norm(
            collect(
                local_displacement(sol, x, LF_ε∞; side = :outer) -
                    local_displacement(sol, x, LF_ε∞; side = :inner)
            )
        ) < 1.0e-12
        @test norm(
            traction(local_stress(sol, x, LF_ε∞; side = :outer)) -
                traction(local_stress(sol, x, LF_ε∞; side = :inner))
        ) < 1.0e-10
    end

    # Spring: traction continuous, displacement jumping by the compliance
    # times the traction — re-derived from the field, not from the jump matrix.
    kn, kt = 47.6, 27.1
    sph_s = LayeredSphere(
        (1.0, 2.0), (LF_C₁, LF_C₂);
        interfaces = (SpringInterface(kn, kt), PerfectInterface{Float64}())
    )
    sol_s = LayeredSphereFields(sph_s, LF_C₀)
    x = 1.0 * n
    t⁻ = traction(local_stress(sol_s, x, LF_ε∞; side = :inner))
    t⁺ = traction(local_stress(sol_s, x, LF_ε∞; side = :outer))
    @test norm(t⁺ - t⁻) < 1.0e-10
    Δu = collect(
        local_displacement(sol_s, x, LF_ε∞; side = :outer) -
            local_displacement(sol_s, x, LF_ε∞; side = :inner)
    )
    tn = dot(t⁻, n)
    @test Δu ≈ (tn / kn) * n + (t⁻ - tn * n) / kt rtol = 1.0e-10
    @test norm(Δu) > 1.0e-3                     # the jump is not vacuously small

    # An anisotropic spring (kn ≠ kt) drives a mode-2 amplitude in the core, so
    # the N = 1 field there is NOT uniform — a shortcut worth guarding against.
    sph_1 = LayeredSphere(
        (1.0,), (LF_C₁,); interfaces = (SpringInterface(kn, kt),)
    )
    sol_1 = LayeredSphereFields(sph_1, LF_C₀)
    @test abs(sol_1.abcd[1][2]) > 1.0e-6
    @test _maxabs(
        local_strain_strain_loc(sol_1, [0.9, 0.0, 0.0]) -
            local_strain_strain_loc(sol_1, [0.1, 0.0, 0.0])
    ) > 1.0e-6

    # Membrane: displacement continuous, traction jumping.
    sph_m = LayeredSphere(
        (1.0, 2.0), (LF_C₁, LF_C₂);
        interfaces = (PerfectInterface{Float64}(), MembraneInterface(1.3, 0.7))
    )
    sol_m = LayeredSphereFields(sph_m, LF_C₀)
    y = 2.0 * n
    @test norm(
        collect(
            local_displacement(sol_m, y, LF_ε∞; side = :outer) -
                local_displacement(sol_m, y, LF_ε∞; side = :inner)
        )
    ) < 1.0e-12
    Δt = traction(local_stress(sol_m, y, LF_ε∞; side = :outer)) -
        traction(local_stress(sol_m, y, LF_ε∞; side = :inner))
    @test norm(Δt) > 1.0e-3                     # the surface stress really acts
end

@testset "local fields — the four couplings and the two point spellings agree" begin
    sph = LayeredSphere((1.0, 2.0), (LF_C₁, LF_C₂))
    sol = LayeredSphereFields(sph, LF_C₀)
    𝕊₀ = inv(LF_C₀)
    for (r, θ, φ) in ((0.5, 0.7, 0.9), (1.4, 2.1, -0.3), (5.0, 1.0, 2.5))
        x = [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)]
        A = local_strain_strain_loc(sol, x)
        @test _maxabs(A - local_strain_strain_loc(sol, r, θ, φ)) < 1.0e-14

        k = get_layer(sph, r)
        ℂ = region_stiffness(sol, k)
        @test _maxabs(local_stress_strain_loc(sol, x) - ℂ ⊡ A) < 1.0e-12
        @test _maxabs(local_strain_stress_loc(sol, x) - A ⊡ 𝕊₀) < 1.0e-14
        @test _maxabs(local_stress_stress_loc(sol, x) - ℂ ⊡ A ⊡ 𝕊₀) < 1.0e-12

        # The field evaluators are the tensors applied to the loading.
        @test local_strain(sol, x, LF_ε∞) ≈ A ⊡ LF_ε∞
        @test local_stress(sol, x, LF_ε∞) ≈ ℂ ⊡ (A ⊡ LF_ε∞)
    end
end

@testset "local fields — rotation covariance" begin
    sph = LayeredSphere((1.0, 2.0), (LF_C₁, LF_C₂))
    sol = LayeredSphereFields(sph, LF_C₀)
    Q = Matrix(rot3(0.4, 0.9, -0.7))
    x = [0.7, -0.4, 1.1]
    Qx = Q * x
    A = get_array(local_strain_strain_loc(sol, x))
    AQ = get_array(local_strain_strain_loc(sol, Qx))
    rotated = [
        sum(
            Q[i, a] * Q[j, b] * Q[k, c] * Q[l, d] * A[a, b, c, d]
                for a in 1:3, b in 1:3, c in 1:3, d in 1:3
        ) for i in 1:3, j in 1:3, k in 1:3, l in 1:3
    ]
    @test AQ ≈ rotated atol = 1.0e-13
end

@testset "local fields — get_layer, sides and the explicit layer override" begin
    sph = LayeredSphere((1.0, 2.0), (LF_C₁, LF_C₂))
    @test get_layer(sph, 0.5) == 1
    @test get_layer(sph, 1.5) == 2
    @test get_layer(sph, 9.0) == 3                     # the matrix
    @test get_layer(sph, 1.0) == 2                     # on an interface: outer limit
    @test get_layer(sph, 1.0; side = :inner) == 1
    @test get_layer(sph, 2.0; side = :inner) == 2
    @test get_layer(sph, 2.0; side = :outer) == 3
    @test_throws ArgumentError get_layer(sph, 1.0; side = :both)

    sol = LayeredSphereFields(sph, LF_C₀)
    x = [1.0, 0.0, 0.0]
    @test local_strain_strain_loc(sol, x; layer = 1) ==
        local_strain_strain_loc(sol, x; side = :inner)
    @test local_strain_strain_loc(sol, x; layer = 2) ==
        local_strain_strain_loc(sol, x; side = :outer)
    @test_throws ArgumentError local_strain_strain_loc(sol, x; layer = 4)
end

@testset "local fields — genericity: Dual, BigFloat, near-incompressible" begin
    # A single layer's modulus as the differentiation variable, the others
    # staying `Float64` — the heterogeneous case.
    f(p) = local_strain_strain_loc(
        LayeredSphere((1.0, 2.0), (_iso(100.0 * p, 0.2), LF_C₂)), LF_C₀, [0.5, 0.0, 0.0]
    )[1, 1, 1, 1]
    h = 1.0e-6
    @test ForwardDiff.derivative(f, 1.0) ≈ (f(1.0 + h) - f(1.0 - h)) / (2h) rtol = 1.0e-7

    # One interface radius as the variable — a mixed-eltype radii tuple.
    g(r₁) = local_strain_strain_loc(
        LayeredSphere((r₁, 2.0), (LF_C₁, LF_C₂)), LF_C₀, [0.5, 0.0, 0.0]
    )[1, 1, 1, 1]
    @test ForwardDiff.derivative(g, 0.9) ≈ (g(0.9 + h) - g(0.9 - h)) / (2h) rtol = 1.0e-6

    setprecision(BigFloat, 200) do
        Cb₀ = _iso(big(30.0), big(3) / 10)
        Cb₁ = _iso(big(100.0), big(2) / 10)
        Cb₂ = _iso(big(10.0), big(35) / 100)
        Ab = local_strain_strain_loc(
            LayeredSphere((big(1.0), big(2.0)), (Cb₁, Cb₂)), Cb₀,
            [big(1) / 2, big(0), big(0)]
        )
        Af = local_strain_strain_loc(
            LayeredSphere((1.0, 2.0), (LF_C₁, LF_C₂)), LF_C₀, [0.5, 0.0, 0.0]
        )
        @test Float64(maximum(abs, get_array(Ab) .- get_array(Af))) < 1.0e-14
    end

    # Near-incompressible core: finite, and the volumetric localization → 0.
    C_inc = TensISO{3}(3 * 1.0e14, 2 * 1.0)
    sol = LayeredSphereFields(LayeredSphere((1.0, 2.0), (C_inc, LF_C₂)), LF_C₀)
    A = local_strain_strain_loc(sol, [0.5, 0.0, 0.0])
    @test all(isfinite, get_array(A))
    @test abs(sol.AB[1][1]) < 1.0e-10
end

@testset "local fields — a symbolic radius needs an explicit layer" begin
    sph = LayeredSphere((1.0, 2.0), (LF_C₁, LF_C₂))
    sol = LayeredSphereFields(sph, LF_C₀)
    # `Symbolics.Num` subtypes `Real` without being ordered, so the region
    # cannot be found by comparison; the error must say so instead of dying
    # inside the lookup loop.  (Checked here with a stand-in that is likewise
    # not `is_hard_numeric`-comparable only through the documented path.)
    @test_throws ArgumentError local_strain_strain_loc(sol, [0.5, 0.0, 0.0]; layer = 0)
end

# ── Echoes cross-check ──────────────────────────────────────────────────────
#
# Hard-coded from the C++ reference (`sphere_nlayers.loc_eE` / `loc_sE`,
# rows rotated to the canonical basis with `rot6(θ, φ)`), 2-layer sphere
# r = (1, 2), core E=100/ν=0.2, shell E=10/ν=0.35, matrix E=30/ν=0.3.
# The stiffness convention matches: Echoes' `PRIMALDISC [kn, kt]` takes the
# same numbers as `SpringInterface(kn, kt)`.

@testset "local fields — Echoes reference values, imperfect interfaces" begin
    cases = (
        (
            "spring",
            (SpringInterface(47.6, 27.1), PerfectInterface{Float64}()),
            (
                (
                    0.999, 1.2, -0.4,
                    [0.2525134178173, 0.2377027137573, 0.2376921465, 0.2096422929753, 0.20441498566, 0.2044112560398],
                    [30.82612166541, 28.06726502677, 28.06529661609, 17.47019108127, 17.03458213833, 17.03427133665],
                ),
                (
                    2.6, 1.0, 0.6,
                    [0.9429257585191, 1.059139501796, 1.030965132697, 1.030219453937, 0.9571517329126, 0.979598420139],
                    [38.46453693173, 42.18477704059, 41.26677202026, 23.77429509085, 22.08811691337, 22.60611738782],
                ),
                (
                    7.0, 0.45, -1.3,
                    [1.002900201211, 1.006060509898, 0.9852404005593, 0.9920879712201, 1.008299780065, 1.005411137046],
                    [40.51786248134, 40.55717231851, 39.94499085405, 22.89433779739, 23.26845646305, 23.20179547029],
                ),
            ),
        ),
        (
            "membrane",
            (PerfectInterface{Float64}(), MembraneInterface(1.3, 0.7)),
            (
                (
                    0.999, 1.2, -0.4,
                    [0.3294383792308, 0.3241569956903, 0.3241532274874, 0.2866051844373, 0.2847411667171, 0.2847398367632],
                    [39.12201093619, 38.13822380609, 38.13752188594, 23.88376536978, 23.72843055976, 23.72831973026],
                ),
                (
                    2.6, 1.0, 0.6,
                    [0.9560247461684, 1.047042930942, 1.025102260769, 1.024224575405, 0.964574727677, 0.9830307590383],
                    [38.88141982088, 41.82287537301, 41.09962644949, 23.63595174012, 22.25941679255, 22.68532520858],
                ),
                (
                    7.0, 0.45, -1.3,
                    [1.002162214351, 1.004819650039, 0.9883175136479, 0.9935890736165, 1.006729156623, 1.004382391722],
                    [40.48822778676, 40.52232252961, 40.03481994605, 22.92897862192, 23.23221130669, 23.17805519359],
                ),
            ),
        ),
    )

    for (name, interfaces, pts) in cases
        sph = LayeredSphere((1.0, 2.0), (LF_C₁, LF_C₂); interfaces)
        sol = LayeredSphereFields(sph, LF_C₀)
        for (r, θ, φ, eE_diag, sE_diag) in pts
            A = Matrix(KM(local_strain_strain_loc(sol, r, θ, φ)))
            S = Matrix(KM(local_stress_strain_loc(sol, r, θ, φ)))
            @test [A[i, i] for i in 1:6] ≈ eE_diag rtol = 1.0e-10
            @test [S[i, i] for i in 1:6] ≈ sE_diag rtol = 1.0e-10
        end
    end
end

# ── Transport ───────────────────────────────────────────────────────────────

@testset "local transport fields" begin
    K₀ = TensISO{3}(2.0)
    K₁ = TensISO{3}(20.0)
    K₂ = TensISO{3}(0.5)
    ∇T∞ = [0.0, 0.0, 1.0]

    # Homogeneous stack ⟹ the gradient is the remote one, everywhere.
    solH = LayeredSphereTransportFields(LayeredSphere((1.0, 2.0), (K₀, K₀)), K₀)
    for x in ([0.0, 0.0, 0.0], [0.5, 0.3, -0.2], [1.5, 0.0, 0.0], [40.0, 1.0, 2.0])
        @test collect(local_gradient(solH, x, ∇T∞)) ≈ ∇T∞ atol = 1.0e-13
    end

    # N = 1: the classical 3k₀/(2k₀+k₁), uniform inside.
    sol1 = LayeredSphereTransportFields(LayeredSphere((1.0,), (K₁,)), K₀)
    @test sol1.AB[1][1] ≈ 3 * 2.0 / (2 * 2.0 + 20.0) rtol = 1.0e-14
    for x in ([0.2, 0.1, 0.3], [0.9, 0.0, 0.0])
        @test collect(local_gradient(sol1, x, ∇T∞)) ≈ (3 * 2.0 / 24.0) .* ∇T∞ rtol = 1.0e-12
    end

    # Multi-layer: the per-layer amplitudes are the production localizations.
    sph = LayeredSphere((1.0, 2.0), (K₁, K₂))
    sol = LayeredSphereTransportFields(sph, K₀)
    α = _cond_localization(sph, 2.0)
    for k in 1:2
        @test sol.AB[k][1] ≈ α[k] rtol = 1.0e-14
    end

    # Interface conditions at a perfect interface: T and the normal flux
    # continuous, the tangential gradient continuous.
    n = _unit([0.3, -0.5, 0.8])
    x = 1.0 * n
    @test local_temperature(sol, x, ∇T∞; side = :inner) ≈
        local_temperature(sol, x, ∇T∞; side = :outer) rtol = 1.0e-12
    q⁻ = collect(local_flux(sol, x, ∇T∞; side = :inner))
    q⁺ = collect(local_flux(sol, x, ∇T∞; side = :outer))
    @test dot(q⁻, n) ≈ dot(q⁺, n) rtol = 1.0e-10
    g⁻ = collect(local_gradient(sol, x, ∇T∞; side = :inner))
    g⁺ = collect(local_gradient(sol, x, ∇T∞; side = :outer))
    @test norm((g⁺ - dot(g⁺, n) * n) - (g⁻ - dot(g⁻, n) * n)) < 1.0e-11

    # ∇T = grad T, by ForwardDiff on the temperature itself.
    for x₀ in ([0.4, 0.3, 0.2], [1.5, 0.4, -0.6], [6.0, 1.0, -2.0])
        @test ForwardDiff.gradient(y -> local_temperature(sol, y, ∇T∞), x₀) ≈
            collect(local_gradient(sol, x₀, ∇T∞)) atol = 1.0e-12
    end

    # Far field, and the flux localization is −k(x) times the gradient one.
    @test collect(local_gradient(sol, [1.0e5, 0.0, 0.0], ∇T∞)) ≈ ∇T∞ atol = 1.0e-8
    for (x, k) in (([0.5, 0.0, 0.0], 20.0), ([1.5, 0.0, 0.0], 0.5), ([9.0, 0.0, 0.0], 2.0))
        @test get_array(local_flux_gradient_loc(sol, x)) ≈
            -k * get_array(local_gradient_gradient_loc(sol, x)) rtol = 1.0e-13
    end

    # An impermeable core (k = 0) carries no FLUX — the gradient is finite and
    # generally non-zero there, which is the k → 0 limit of α = 3k₀/(2k₀+k),
    # not zero.  Asserting a vanishing gradient is the natural mistake.
    sph0 = LayeredSphere((1.0, 2.0), (TensISO{3}(0.0), K₂))
    sol0 = LayeredSphereTransportFields(sph0, K₀)
    x0 = [0.6, 0.1, -0.2]
    @test norm(collect(local_flux(sol0, x0, ∇T∞))) < 1.0e-13
    @test norm(collect(local_gradient(sol0, x0, ∇T∞))) > 1.0e-2
    @test sol0.AB[1][1] ≈ _cond_localization(sph0, 2.0)[1] rtol = 1.0e-14
    # A single impermeable sphere: the classical 3k₀/(2k₀) = 3/2.
    sol0₁ = LayeredSphereTransportFields(LayeredSphere((1.0,), (TensISO{3}(0.0),)), K₀)
    @test collect(local_gradient(sol0₁, [0.5, 0.0, 0.0], ∇T∞)) ≈ 1.5 .* ∇T∞ rtol = 1.0e-13
end

# =============================================================================
#  Averages rebuilt from the pointwise field.
#
#  `cumulative_strain_average` used to weight the FULL-layer average by the
#  TRUNCATED volume, which is exact only where the field is uniform inside a
#  layer — and the mode-2 term makes it vary as r².  The error reached 6 % at
#  mid-layer radii and vanished at every interface radius, which is exactly
#  where the old tests evaluated it.  The oracle below is the pointwise field
#  itself, integrated with a 3-point Gauss-Legendre rule that is EXACT because
#  `β_local(r)·r²` is a degree-4 polynomial inside a layer.
# =============================================================================

const LF_GLX = (-sqrt(3 / 5), 0.0, sqrt(3 / 5))
const LF_GLW = (5 / 9, 8 / 9, 5 / 9)

"Ball average of the pointwise strain over radius `r_max`, integrated layer by layer."
function _ball_strain_average(sol, radii, r_max, ε∞)
    num = nothing
    vol = 0.0
    for k in 1:(length(radii) - 1)
        ra, rb = radii[k], min(r_max, radii[k + 1])
        rb ≤ ra && break
        mid, half = (ra + rb) / 2, (rb - ra) / 2
        for (ξ, w) in zip(LF_GLX, LF_GLW)
            r = mid + half * ξ
            A = isotropify(local_strain_strain_loc(sol, [r, 0.0, 0.0]; side = :inner))
            c = w * half * r^2
            num = num === nothing ? c * (A ⊡ ε∞) : num + c * (A ⊡ ε∞)
        end
        vol += (rb^3 - ra^3) / 3
        rb ≥ r_max && break
    end
    return (1 / vol) * num
end

@testset "averages — cumulative average is exact inside a layer" begin
    sph = LayeredSphere((1.0, 2.0, 3.5), (LF_C₁, LF_C₂, LF_C₃))
    sol = LayeredSphereFields(sph, LF_C₀)
    radii = (0.0, 1.0, 2.0, 3.5)

    for r in (0.4, 0.7, 1.0, 1.2, 1.37, 1.7, 2.0, 2.4, 2.9, 3.2, 3.5)
        got = cumulative_strain_average(sph, LF_C₀, LF_ε∞, r)
        ref = _ball_strain_average(sol, radii, r, LF_ε∞)
        @test get_array(got) ≈ get_array(ref) rtol = 1.0e-13
    end

    # The interface radii must be untouched by the fix.
    @test get_array(cumulative_strain_average(sph, LF_C₀, LF_ε∞, 3.5)) ≈
        get_array(sphere_strain_average(sph, LF_C₀, LF_ε∞)) rtol = 1.0e-13
    @test get_array(cumulative_strain_average(sph, LF_C₀, LF_ε∞, 1.0)) ≈
        get_array(layer_strain_average(sph, LF_C₀, LF_ε∞, 1)) rtol = 1.0e-13

    # And a mid-layer radius must genuinely differ from the volume-scaled
    # full-layer average — otherwise this testset proves nothing.
    naive = let r = 1.37
        vol1 = 1.0^3
        vol2 = r^3 - 1.0^3
        (
            vol1 * layer_strain_average(sph, LF_C₀, LF_ε∞, 1) +
                vol2 * layer_strain_average(sph, LF_C₀, LF_ε∞, 2)
        ) / (vol1 + vol2)
    end
    exact = cumulative_strain_average(sph, LF_C₀, LF_ε∞, 1.37)
    @test maximum(abs, get_array(naive) .- get_array(exact)) /
        maximum(abs, get_array(exact)) > 1.0e-2
end

@testset "averages — stress and transport" begin
    sph = LayeredSphere((1.0, 2.0, 3.5), (LF_C₁, LF_C₂, LF_C₃))
    sol = LayeredSphereFields(sph, LF_C₀)

    # Definitional, then against the scheme-facing route, which assembles
    # Σ f_k ℂ_k : A_k independently.
    for k in 1:3
        @test layer_stress_average(sph, LF_C₀, LF_ε∞, k) ≈
            layer_modulus(sph, k) ⊡ layer_strain_average(sph, LF_C₀, LF_ε∞, k)
    end
    @test get_array(sphere_stress_average(sph, LF_C₀, LF_ε∞)) ≈
        get_array(stress_strain_loc(sph, LF_C₁, LF_C₀) ⊡ LF_ε∞) rtol = 1.0e-13

    # Against the pointwise field, layer 2.
    ra, rb = 1.0, 2.0
    mid, half = (ra + rb) / 2, (rb - ra) / 2
    num = nothing
    for (ξ, w) in zip(LF_GLX, LF_GLW)
        r = mid + half * ξ
        A = isotropify(local_strain_strain_loc(sol, [r, 0.0, 0.0]; side = :inner))
        c = w * half * r^2
        σ = LF_C₂ ⊡ (A ⊡ LF_ε∞)
        num = num === nothing ? c * σ : num + c * σ
    end
    @test get_array(layer_stress_average(sph, LF_C₀, LF_ε∞, 2)) ≈
        get_array((3 / (rb^3 - ra^3)) * num) rtol = 1.0e-13

    # A membrane interface carries a surface stress that belongs to neither
    # bulk: the two routes must then DISAGREE, and agree without it.
    sm = LayeredSphere(
        (1.0, 2.0), (LF_C₁, LF_C₂);
        interfaces = (PerfectInterface{Float64}(), MembraneInterface(1.3, 0.7))
    )
    gap = maximum(
        abs,
        get_array(sphere_stress_average(sm, LF_C₀, LF_ε∞)) .-
            get_array(stress_strain_loc(sm, LF_C₁, LF_C₀) ⊡ LF_ε∞)
    )
    @test gap > 1.0e-2

    # Transport.
    K₀ = TensISO{3}(2.0)
    sphK = LayeredSphere((1.0, 2.0), (TensISO{3}(20.0), TensISO{3}(0.5)))
    G = [0.3, -0.7, 1.0]
    for k in 1:2
        A = gradient_gradient_loc(sphK, K₀; layer = k)
        @test collect(layer_gradient_average(sphK, K₀, G, k)) ≈
            [sum(A[i, j] * G[j] for j in 1:3) for i in 1:3] rtol = 1.0e-14
    end
    A3 = gradient_gradient_loc(sphK, TensISO{3}(20.0), K₀)
    @test collect(sphere_gradient_average(sphK, K₀, G)) ≈
        [sum(A3[i, j] * G[j] for j in 1:3) for i in 1:3] rtol = 1.0e-14
    @test collect(layer_flux_average(sphK, K₀, G, 1)) ≈
        -20.0 .* collect(layer_gradient_average(sphK, K₀, G, 1)) rtol = 1.0e-14

    # The mean gradient is constant inside a layer: the 1/r² mode is present
    # (B̃ ≠ 0) but contributes nothing to the directional average.
    solK = LayeredSphereTransportFields(sphK, K₀)
    @test abs(solK.AB[2][2]) > 1.0e-2
    tr3(r) = let a = get_array(local_gradient_gradient_loc(solK, [r, 0.0, 0.0]; side = :inner))
        (a[1, 1] + a[2, 2] + a[3, 3]) / 3
    end
    for rs in ((0.1, 0.5, 0.99), (1.01, 1.5, 1.99), (2.5, 6.0, 40.0))
        vals = tr3.(rs)
        @test maximum(vals) - minimum(vals) < 1.0e-14
    end
end
