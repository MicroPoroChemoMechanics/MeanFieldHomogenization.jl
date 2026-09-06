using Test
using MeanFieldHomogenization
using MeanFieldHomogenization.LayeredSpheroids: _base_fond
using TensND

# =============================================================================
#  test_local_fields.jl — pointwise temperature/gradient/flux
#  reconstruction (`localfields.jl`), checked against the two physical
#  continuity conditions at a perfect interface (normal flux and
#  tangential gradient continuous, NOT the full field vector) and the
#  far-field limit (recovers the imposed remote uniform gradient).
# =============================================================================

const K1 = TensISO{3}(5.0)
const K2 = TensISO{3}(20.0)
const K0 = TensISO{3}(2.0)

function _two_layer_spheroid(; Nseries = 8)
    focal2 = 8.0
    a_in = 2.9
    b_in = sqrt(a_in^2 - focal2)
    return LayeredSpheroid((a_in, 3.0), (b_in, 1.0), (K1, K2); Nseries)
end

@testset "local fields — continuity across a perfect interface" begin
    s = _two_layer_spheroid()
    q1 = real(s.q[1])
    p, φ = 0.4, 0.7
    ε = 1.0e-7

    e_q, e_p, e_φ = _base_fond(q1, p, φ)
    e_q, e_p, e_φ = real.(e_q), real.(e_p), real.(e_φ)

    u_below = local_flux(s, K0, q1 - ε, p, φ; H_axial = 1.0, H_trans = 0.3)
    u_above = local_flux(s, K0, q1 + ε, p, φ; H_axial = 1.0, H_trans = 0.3)
    un_below = sum(u_below .* e_q)
    un_above = sum(u_above .* e_q)
    @test un_below ≈ un_above rtol = 1.0e-4

    g_below = local_gradient(s, K0, q1 - ε, p, φ; H_axial = 1.0, H_trans = 0.3)
    g_above = local_gradient(s, K0, q1 + ε, p, φ; H_axial = 1.0, H_trans = 0.3)
    @test sum(g_below .* e_p) ≈ sum(g_above .* e_p) rtol = 1.0e-4
    @test sum(g_below .* e_φ) ≈ sum(g_above .* e_φ) rtol = 1.0e-4

    T_below = local_temperature(s, K0, q1 - ε, p, φ; H_axial = 1.0, H_trans = 0.3)
    T_above = local_temperature(s, K0, q1 + ε, p, φ; H_axial = 1.0, H_trans = 0.3)
    @test T_below ≈ T_above rtol = 1.0e-4
end

@testset "local fields — far field recovers the remote uniform gradient" begin
    s = _two_layer_spheroid()
    qbig = 1.0e3
    p, φ = 0.4, 0.7

    T = local_temperature(s, K0, qbig, p, φ; H_axial = 1.0, H_trans = 0.0)
    z = real(s.c * qbig * p)   # remote T ~ H₃·z, z = c·q·p (eq:xLeg, P₁(q)=q, P₁(p)=p)
    @test T ≈ z rtol = 1.0e-8

    g_axial = local_gradient(s, K0, qbig, p, φ; H_axial = 1.0, H_trans = 0.0)
    @test collect(g_axial) ≈ [0.0, 0.0, 1.0] atol = 1.0e-8

    g_trans = local_gradient(s, K0, qbig, 0.0, 0.0; H_axial = 0.0, H_trans = 1.0)
    @test collect(g_trans) ≈ [1.0, 0.0, 0.0] atol = 1.0e-8
end

@testset "local fields — accept both scalar and TensISO matrix conductivity" begin
    s = _two_layer_spheroid()
    T_tens = local_temperature(s, K0, 5.0, 0.2, 0.1)
    T_scalar = local_temperature(s, 2.0, 5.0, 0.2, 0.1)
    @test T_tens ≈ T_scalar
end

# =============================================================================
#  Harmonization with `LayeredSphere`: the cached solution object, the remote
#  gradient given as a vector, the `side` keyword, and the four `local_*_*_loc`
#  couplings.  Same names and same conventions as the sphere, on a problem
#  whose solution is structurally different.
# =============================================================================

@testset "local fields — cached solution object matches the direct form" begin
    s = _two_layer_spheroid()
    f = LayeredSpheroidTransportFields(s, K0)
    for (q, p, φ) in ((1.5, 0.4, 0.7), (3.0, -0.2, 2.0), (1.0e3, 0.6, 0.1))
        @test local_temperature(f, q, p, φ; H_axial = 1.0, H_trans = 0.3) ≈
            local_temperature(s, K0, q, p, φ; H_axial = 1.0, H_trans = 0.3)
        @test collect(local_gradient(f, q, p, φ; H_axial = 1.0, H_trans = 0.3)) ≈
            collect(local_gradient(s, K0, q, p, φ; H_axial = 1.0, H_trans = 0.3))
        @test collect(local_flux(f, q, p, φ; H_axial = 0.5, H_trans = 1.0)) ≈
            collect(local_flux(s, K0, q, p, φ; H_axial = 0.5, H_trans = 1.0))
    end
    @test get_layer(f, 1.5) == get_layer(s, 1.5)
end

@testset "local fields — remote gradient as a vector" begin
    s = _two_layer_spheroid()
    f = LayeredSpheroidTransportFields(s, K0)

    # A purely axial / purely ê₁-transverse loading must reproduce the
    # canonical entry points exactly.
    @test local_temperature(f, 1.5, 0.4, 0.7, [0.0, 0.0, 1.0]) ≈
        local_temperature(f, 1.5, 0.4, 0.7; H_axial = 1.0, H_trans = 0.0)
    @test local_temperature(f, 1.5, 0.4, 0.0, [0.3, 0.0, 1.0]) ≈
        local_temperature(f, 1.5, 0.4, 0.0; H_axial = 1.0, H_trans = 0.3)

    # A transverse loading along ê₂ is the ê₁ one seen from a shifted azimuth.
    @test local_temperature(f, 1.5, 0.4, π / 2, [0.0, 1.0, 0.0]) ≈
        local_temperature(f, 1.5, 0.4, 0.0; H_axial = 0.0, H_trans = 1.0)

    # Linearity in the loading, which is the whole content of a localization
    # tensor and is NOT automatic given the azimuth-shift construction.
    G₁ = [0.3, -0.7, 1.0]
    G₂ = [1.1, 0.2, -0.4]
    for (q, p, φ) in ((1.5, 0.4, 0.7), (3.0, -0.2, 2.0))
        g₁ = collect(local_gradient(f, q, p, φ, G₁))
        g₂ = collect(local_gradient(f, q, p, φ, G₂))
        g₁₂ = collect(local_gradient(f, q, p, φ, G₁ + 2 * G₂))
        @test g₁₂ ≈ g₁ + 2 * g₂ rtol = 1.0e-12
    end
end

@testset "local fields — the localization tensors" begin
    s = _two_layer_spheroid()
    f = LayeredSpheroidTransportFields(s, K0)
    G = [0.3, -0.7, 1.0]
    k₀ = 2.0

    # One point per region: inside the core (1 < q < q₁ — prolate coordinates
    # require |q| ≥ 1), inside the shell, and out in the matrix.
    q_core = (1 + real(s.q[1])) / 2
    for (q, p, φ) in ((q_core, -0.2, 2.0), (1.5, 0.4, 0.7), (3.0, 0.6, 0.1))
        A = local_gradient_gradient_loc(f, q, p, φ)
        g = collect(local_gradient(f, q, p, φ, G))
        @test [sum(A[i, j] * G[j] for j in 1:3) for i in 1:3] ≈ g rtol = 1.0e-12

        # `−k(x)` relates the flux and gradient localizations, `k(x)` being the
        # conductivity of the region containing the point.
        lay = get_layer(s, q)
        k_here = lay ≤ layer_count(s) ?
            MeanFieldHomogenization.LayeredSpheroids._spheroid_layer_moduli(s)[lay] : k₀
        @test get_array(local_flux_gradient_loc(f, q, p, φ)) ≈
            -k_here .* get_array(A) rtol = 1.0e-12
        @test get_array(local_gradient_flux_loc(f, q, p, φ)) ≈
            -get_array(A) ./ k₀ rtol = 1.0e-12
        @test get_array(local_flux_flux_loc(f, q, p, φ)) ≈
            k_here .* get_array(A) ./ k₀ rtol = 1.0e-12
    end

    # Far away the localization returns to the identity.
    @test get_array(local_gradient_gradient_loc(f, 1.0e5, 0.4, 0.7)) ≈
        Matrix(1.0I, 3, 3) atol = 1.0e-6

    # Unlike the sphere's, this tensor is genuinely non-symmetric: a confocal
    # spheroid is not rotation-invariant about the field point, so it cannot be
    # carried by a `TensTI` and a general `Tens` is required.
    A = get_array(local_gradient_gradient_loc(f, 1.5, 0.4, 0.7))
    @test maximum(abs, A - A') > 1.0e-3
end

@testset "local fields — the `side` keyword on an interface" begin
    s = _two_layer_spheroid()
    q₁ = real(s.q[1])
    @test get_layer(s, q₁; side = :inner) == 1
    @test get_layer(s, q₁; side = :outer) == 2
    @test get_layer(s, q₁) == 2                       # `:outer` is the default
    @test get_layer(s, 0.5 * q₁; side = :inner) == 1
    @test_throws ArgumentError get_layer(s, q₁; side = :both)
end
