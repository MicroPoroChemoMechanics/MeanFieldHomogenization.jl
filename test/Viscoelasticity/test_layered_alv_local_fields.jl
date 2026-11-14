using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_layered_alv_local_fields.jl — the pointwise operators of an n-layer
#  sphere in aging linear viscoelasticity, orders 4 and 2.
#
#  Echoes has no pointwise field of its n-layer sphere, so the references are
#  identities the operators must satisfy, on aging laws and a non-uniform grid:
#  the elastic limit; the averages over a shell (a volume quadrature of the
#  strain operator, and the surface integrals of the displacement and of the
#  traction, which are the average strain and — the stress being in
#  equilibrium — the average stress); compatibility; the interface laws read on
#  both sides; the far field; an interface replaced by a thin elastic layer.
# =============================================================================

module LayeredALVLocalData
    using MeanFieldHomogenization, TensND, LinearAlgebra
    const V = MeanFieldHomogenization.Viscoelasticity
    const times = [0.0, 0.3, 0.7, 1.2, 2.0]
    iso4(a, b) = (t, tp) -> t >= tp ? TensISO{3}(a(t, tp), b(t, tp)) : TensISO{3}(0.0, 0.0)
    const core = ViscoLaw(iso4((t, tp) -> 6.0 * (1 + 0.5tp) * exp(-(t - tp) / 0.8), (t, tp) -> 2.0 * (1 + tp) * exp(-(t - tp) / 1.5)))
    shell(p = 1.0) = ViscoLaw(
        (t, tp) -> t >= tp ?
            TensISO{3}(9.0 * exp(-(t - tp) / 2.0), 3.0 * p * (1 + 0.3tp) * exp(-(t - tp) / 0.7)) :
            TensISO{3}(zero(p), zero(p))
    )
    const outer = ViscoLaw(iso4((t, tp) -> 4.0 * exp(-(t - tp) / 1.1), (t, tp) -> 1.5 * exp(-(t - tp) / 0.9)))
    const matrix = ViscoLaw(iso4((t, tp) -> 3.0 * exp(-(t - tp) / 0.6), (t, tp) -> exp(-(t - tp) / 2.0)))
    iso2(a) = (t, tp) -> t >= tp ? TensISO{3}(a(t, tp)) : TensISO{3}(0.0)
    const K0 = ViscoLaw(iso2((t, tp) -> 2.0 * exp(-(t - tp) / 0.9)))
    K1(p = 1.0) = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(5.0 * p * (1 + 0.5tp) * exp(-(t - tp) / 1.4)) : TensISO{3}(zero(p)))
    const K2 = ViscoLaw(iso2((t, tp) -> (1 + tp) * exp(-(t - tp) / 0.7)))
    const K3 = ViscoLaw(iso2((t, tp) -> 3.0 * (1 + 0.2tp) * exp(-(t - tp) / 1.1)))

    # Mandel components of a symmetric 3 × 3 matrix, and back.
    mandel6(a) = [a[1, 1], a[2, 2], a[3, 3], √2 * a[2, 3], √2 * a[1, 3], √2 * a[1, 2]]
    unmandel(v) = [v[1] v[6] / √2 v[5] / √2; v[6] / √2 v[2] v[4] / √2; v[5] / √2 v[4] / √2 v[3]]
    # u ↦ sym(u ⊗ n) on Mandel components (6 × 3); its transpose is ε ↦ ε·n.
    symdyad(n) = reduce(hcat, [mandel6(((1:3 .== i) * n' + n * (1:3 .== i)') / 2) for i in 1:3])
    # A block of numbers acting at every instant.
    every(B, n = length(times)) = kron(Matrix(1.0I, n, n), B)

    # Quadrature on the unit sphere, weights summing to 1: 4 Gauss–Legendre nodes
    # in cos θ and 8 azimuths, exact for the polynomials of degree 7 in n.
    const gl = (
        (-0.8611363115940526, 0.3478548451374538), (-0.3399810435848563, 0.6521451548625461),
        (0.3399810435848563, 0.6521451548625461), (0.8611363115940526, 0.3478548451374538),
    )
    # A direction from its spherical angles. On an interface a point is given
    # by (r, θ, φ): the norm of r n may fall on either side of r.
    direction(θ, φ) = [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]
    const directions = [(acos(c), φ, w / 16) for (c, w) in gl for φ in (0.1 .+ (0:7) .* (π / 4))]
    # F is called with the direction and the angles (θ, φ).
    angular(F) = sum(w .* F(direction(θ, φ), (θ, φ)) for (θ, φ, w) in directions)
    # Average over the shell (a, b) of a field whose angular average is a
    # polynomial of degree ≤ 4 in r.
    radial(F, a, b) = sum(w * ((a + b) / 2 + x * (b - a) / 2)^2 .* F((a + b) / 2 + x * (b - a) / 2) for (x, w) in gl) *
        (b - a) / 2 * 3 / (b^3 - a^3)
end
const LL = LayeredALVLocalData

const llf_radii = (0.5, 0.8, 1.0)
const llf_angles = (0.7, 2.1)
const llf_dir = LL.direction(llf_angles...)
# Points in every region, at the origin, and on the interfaces from both sides,
# by their spherical coordinates.
const llf_points = [
    ((0.0, llf_angles...), :outer), ((0.2, llf_angles...), :outer), ((0.5, llf_angles...), :inner),
    ((0.5, llf_angles...), :outer), ((0.65, llf_angles...), :outer), ((0.8, llf_angles...), :inner),
    ((0.8, llf_angles...), :outer), ((0.93, llf_angles...), :outer), ((1.0, llf_angles...), :inner),
    ((1.0, llf_angles...), :outer), ((1.7, llf_angles...), :outer),
]

@testset "Pointwise ALV operators, order 4 — the elastic limit" begin
    times = [0.0, 0.4, 1.5]
    n = length(times)
    C₀ = TensISO{3}(3.0, 1.0)
    C = (TensISO{3}(6.0, 2.0), TensISO{3}(9.0, 3.0), TensISO{3}(4.0, 1.5))
    canonical(A) = TensND.mandel(A, TensND.CanonicalBasis{3, Float64}())
    for interfaces in (
            (PerfectInterface(), PerfectInterface(), PerfectInterface()),
            (SpringInterface(5.0, 2.0), SpringInterface(8.0, 3.0), SpringInterface(4.0, 6.0)),
            (MembraneInterface(1.3, 0.7), MembraneInterface(0.4, 0.9), MembraneInterface(0.8, 0.2)),
            (SpringInterface(5.0, 2.0), MembraneInterface(1.3, 0.7), SpringInterface(4.0, 6.0)),
        )
        s = LayeredSphere(llf_radii, C; interfaces)
        f = LayeredSphereALVFields(s, heaviside_law(C₀), times)
        fe = LayeredSphereFields(s, C₀)
        for (x, side) in llf_points
            @test local_gradient_gradient_concentration_alv(f, x...; side) ≈
                LL.every(canonical(local_gradient_gradient_concentration(fe, x...; side)), n) atol = 1.0e-12
            @test local_flux_gradient_concentration_alv(f, x...; side) ≈
                LL.every(canonical(local_flux_gradient_concentration(fe, x...; side)), n) atol = 1.0e-11
            U₀ = reduce(hcat, [collect(local_displacement(fe, x..., TensND.Tens(LL.unmandel(1:6 .== J)); side)) for J in 1:6])
            @test local_displacement_alv(f, x...; side) ≈ LL.every(U₀, n) atol = 1.0e-12
        end
        # Spherical coordinates, a named layer, and the one-off form.
        r, θ, φ = 0.65, 0.7, 2.1
        x = r .* [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]
        for op in (local_gradient_gradient_concentration_alv, local_flux_gradient_concentration_alv, local_displacement_alv)
            @test op(f, r, θ, φ) ≈ op(f, x) rtol = 1.0e-13
            @test op(s, heaviside_law(C₀), times, x) == op(f, x)
            @test op(f, 0.8, llf_angles...; layer = 2) == op(f, 0.8, llf_angles...; side = :inner)
        end
    end
end

@testset "Pointwise ALV operators, order 2 — the elastic limit" begin
    times = [0.0, 0.4, 1.5]
    n = length(times)
    K₀ = TensISO{3}(2.0)
    K = (TensISO{3}(5.0), TensISO{3}(1.0), TensISO{3}(3.0))
    canonical(A) = Matrix(TensND.canonical_components(A))
    for interfaces in (
            (PerfectInterface(), PerfectInterface(), PerfectInterface()),
            (KapitzaInterface(0.1), KapitzaInterface(0.25), KapitzaInterface(0.4)),
            (SurfaceConductiveInterface(0.2), SurfaceConductiveInterface(0.5), SurfaceConductiveInterface(0.3)),
            (SurfaceConductiveInterface(0.2), KapitzaInterface(0.25), SurfaceConductiveInterface(0.3)),
        )
        s = LayeredSphere(llf_radii, K; interfaces)
        f = LayeredSphereALVFields(s, heaviside_law(K₀), times)
        fe = LayeredSphereTransportFields(s, K₀)
        for (x, side) in llf_points
            @test local_gradient_gradient_concentration_alv(f, x...; side) ≈
                LL.every(canonical(local_gradient_gradient_concentration(fe, x...; side)), n) atol = 1.0e-13
            @test local_flux_gradient_concentration_alv(f, x...; side) ≈
                LL.every(canonical(local_flux_gradient_concentration(fe, x...; side)), n) atol = 1.0e-12
            T₀ = [local_temperature(fe, x..., 1.0 .* (1:3 .== j); side) for j in 1:3]
            @test local_temperature_alv(f, x...; side) ≈ LL.every(T₀', n) atol = 1.0e-13
        end
        r, θ, φ = 0.65, 0.7, 2.1
        x = r .* [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]
        for op in (local_gradient_gradient_concentration_alv, local_flux_gradient_concentration_alv, local_temperature_alv)
            @test op(f, r, θ, φ) ≈ op(f, x) rtol = 1.0e-13
            @test op(s, heaviside_law(K₀), times, x) == op(f, x)
        end
    end
end

@testset "Pointwise ALV operators, order 4 — averages over a shell, aging layers" begin
    times = LL.times
    radii = llf_radii
    relax = (
        LL.V._trapezoidal_relaxation(LL.core, times, 6), LL.V._trapezoidal_relaxation(LL.shell(), times, 6),
        LL.V._trapezoidal_relaxation(LL.outer, times, 6),
    )
    for interfaces in (
            (SpringInterface(5.0, 2.0), SpringInterface(8.0, 3.0), SpringInterface(4.0, 6.0)),
            (SpringInterface(5.0, 2.0), MembraneInterface(1.3, 0.7), MembraneInterface(0.8, 0.2)),
            (MembraneInterface(1.3, 0.7), PerfectInterface(), SpringInterface(4.0, 6.0)),
        )
        s = LayeredSphere(radii, (LL.core, LL.shell(), LL.outer); interfaces)
        f = LayeredSphereALVFields(s, LL.matrix, times)
        # r² ⟨sym(u ⊗ n)⟩ on the sphere of radius r: the strain it encloses, to
        # the factor 3/Δr³; r³ ⟨sym(n ⊗ σ·n)⟩: the stress.
        strain_flux(r, side) = iszero(r) ? zeros(6 * length(times), 6 * length(times)) :
            r^2 .* LL.angular((n, a) -> LL.every(LL.symdyad(n)) * local_displacement_alv(f, r, a...; side))
        stress_flux(r, side) = iszero(r) ? zeros(6 * length(times), 6 * length(times)) :
            r^3 .* LL.angular(
                (n, a) -> LL.every(LL.symdyad(n) * LL.symdyad(n)') * local_flux_gradient_concentration_alv(f, r, a...; side)
            )
        for k in 1:3
            a, b = k == 1 ? 0.0 : radii[k - 1], radii[k]
            w = 3 / (b^3 - a^3)
            material = gradient_gradient_concentration_alv(s, LL.matrix, times; layer = k)
            # Volume quadrature of the strain operator.
            @test LL.radial(r -> LL.angular((n, a) -> local_gradient_gradient_concentration_alv(f, r, a...; layer = k)), a, b) ≈
                material rtol = 1.0e-11
            # The displacement on its two faces: read outside an interface, its
            # jump is counted.
            for external in (false, true), internal in (false, true)
                @test w .* (strain_flux(b, external ? :outer : :inner) - strain_flux(a, internal ? :inner : :outer)) ≈
                    gradient_gradient_concentration_alv(s, LL.matrix, times; layer = k, external, internal) rtol = 1.0e-10
            end
            # The traction on its two faces, read inside the layer.
            flux = flux_gradient_concentration_alv(s, LL.matrix, times; layer = k)
            @test w .* (stress_flux(b, :inner) - stress_flux(a, :outer)) ≈ flux rtol = 1.0e-10
            @test flux ≈ relax[k] * material rtol = 1.0e-12
        end
        # The whole sphere, read on either side of its outer interface: the
        # traction outside it carries the surface stress of the membranes.
        for (external, side) in ((true, :outer), (false, :inner))
            @test 3 .* strain_flux(1.0, side) ≈ gradient_gradient_concentration_alv(s, LL.matrix, times; external) rtol = 1.0e-10
            @test 3 .* stress_flux(1.0, side) ≈ flux_gradient_concentration_alv(s, LL.matrix, times; external) rtol = 1.0e-10
        end
    end
end

@testset "Pointwise ALV operators, order 2 — averages over a shell, aging layers" begin
    times = LL.times
    radii = llf_radii
    for interfaces in (
            (KapitzaInterface(0.1), KapitzaInterface(0.25), KapitzaInterface(0.4)),
            (SurfaceConductiveInterface(0.2), KapitzaInterface(0.25), SurfaceConductiveInterface(0.3)),
        )
        s = LayeredSphere(radii, (LL.K1(), LL.K2, LL.K3); interfaces)
        f = LayeredSphereALVFields(s, LL.K0, times)
        zero_block = zeros(3 * length(times), 3 * length(times))
        # r² ⟨T n⟩: the gradient enclosed; r³ ⟨n (q·n)⟩: the flux enclosed.
        gradient_flux(r, side) = iszero(r) ? zero_block :
            r^2 .* LL.angular((n, a) -> LL.every(reshape(n, 3, 1)) * local_temperature_alv(f, r, a...; side))
        heat_flux(r, side) = iszero(r) ? zero_block :
            r^3 .* LL.angular((n, a) -> LL.every(n * n') * local_flux_gradient_concentration_alv(f, r, a...; side))
        for k in 1:3
            a, b = k == 1 ? 0.0 : radii[k - 1], radii[k]
            w = 3 / (b^3 - a^3)
            @test LL.radial(r -> LL.angular((n, a) -> local_gradient_gradient_concentration_alv(f, r, a...; layer = k)), a, b) ≈
                gradient_gradient_concentration_alv(s, LL.K0, times; layer = k) rtol = 1.0e-11
            for external in (false, true), internal in (false, true)
                @test w .* (gradient_flux(b, external ? :outer : :inner) - gradient_flux(a, internal ? :inner : :outer)) ≈
                    gradient_gradient_concentration_alv(s, LL.K0, times; layer = k, external, internal) rtol = 1.0e-10
            end
            # q = −k ∘ ∇T: the average of k ∘ ∇T is minus that of the flux.
            @test -w .* (heat_flux(b, :inner) - heat_flux(a, :outer)) ≈
                flux_gradient_concentration_alv(s, LL.K0, times; layer = k) rtol = 1.0e-10
        end
        for (external, side) in ((true, :outer), (false, :inner))
            @test 3 .* gradient_flux(1.0, side) ≈ gradient_gradient_concentration_alv(s, LL.K0, times; external) rtol = 1.0e-10
            @test -3 .* heat_flux(1.0, side) ≈ flux_gradient_concentration_alv(s, LL.K0, times; external) rtol = 1.0e-10
        end
    end
end

@testset "Pointwise ALV operators — interface laws between aging layers" begin
    times = LL.times
    n = llf_dir
    p = n * n'
    q = I - p
    # Elasticity: spring, membrane, perfect.
    interfaces = (SpringInterface(5.0, 2.0), MembraneInterface(1.3, 0.7), PerfectInterface())
    s = LayeredSphere(llf_radii, (LL.core, LL.shell(), LL.outer); interfaces)
    f = LayeredSphereALVFields(s, LL.matrix, times)
    u(r, side) = local_displacement_alv(f, r, llf_angles...; side)
    strain(r, side) = local_gradient_gradient_concentration_alv(f, r, llf_angles...; side)
    traction(r, side) = LL.every(LL.symdyad(n)') * local_flux_gradient_concentration_alv(f, r, llf_angles...; side)
    # Spring: [u] = (n⊗n / kₙ + (𝟏 − n⊗n) / kₜ)·t, the traction continuous.
    @test traction(0.5, :outer) ≈ traction(0.5, :inner) rtol = 1.0e-11
    @test u(0.5, :outer) - u(0.5, :inner) ≈ LL.every(p ./ 5.0 .+ q ./ 2.0) * traction(0.5, :inner) rtol = 1.0e-10
    @test norm(u(0.5, :outer) - u(0.5, :inner)) > 0.01 * norm(u(0.5, :inner))
    # Membrane: the displacement continuous, hence the tangential strain; the
    # normal traction jumps by tr σˢ / r = 2κˢ tr εˢ / r.
    @test u(0.8, :outer) ≈ u(0.8, :inner) rtol = 1.0e-11
    tangential = LL.every(reshape(LL.mandel6(q), 1, 6))
    @test tangential * strain(0.8, :outer) ≈ tangential * strain(0.8, :inner) rtol = 1.0e-10
    @test LL.every(n') * (traction(0.8, :outer) - traction(0.8, :inner)) ≈
        (2 * 1.3 / 0.8) .* (tangential * strain(0.8, :inner)) rtol = 1.0e-10
    # Perfect: both continuous, the strain not.
    @test u(1.0, :outer) ≈ u(1.0, :inner) rtol = 1.0e-11
    @test traction(1.0, :outer) ≈ traction(1.0, :inner) rtol = 1.0e-10
    @test !(strain(1.0, :outer) ≈ strain(1.0, :inner))

    # Conduction: resistance, conducting surface, perfect.
    interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3), PerfectInterface())
    s = LayeredSphere(llf_radii, (LL.K1(), LL.K2, LL.K3); interfaces)
    f = LayeredSphereALVFields(s, LL.K0, times)
    T(r, side) = local_temperature_alv(f, r, llf_angles...; side)
    qₙ(r, side) = LL.every(n') * local_flux_gradient_concentration_alv(f, r, llf_angles...; side)
    # Kapitza: [T] = −ρ qₙ, the normal flux continuous.
    @test qₙ(0.5, :outer) ≈ qₙ(0.5, :inner) rtol = 1.0e-11
    @test T(0.5, :outer) - T(0.5, :inner) ≈ -0.1 .* qₙ(0.5, :inner) rtol = 1.0e-10
    # Conducting surface: T continuous, [qₙ] = kˢ Δₛ T = −2kˢ T / r².
    @test T(0.8, :outer) ≈ T(0.8, :inner) rtol = 1.0e-11
    @test qₙ(0.8, :outer) - qₙ(0.8, :inner) ≈ (-2 * 0.3 / 0.8^2) .* T(0.8, :inner) rtol = 1.0e-10
    @test T(1.0, :outer) ≈ T(1.0, :inner) rtol = 1.0e-11
    @test qₙ(1.0, :outer) ≈ qₙ(1.0, :inner) rtol = 1.0e-10
end

@testset "Pointwise ALV operators — compatibility, far field, operator order" begin
    times = LL.times
    nt = length(times)
    interfaces = (SpringInterface(5.0, 2.0), MembraneInterface(1.3, 0.7), SpringInterface(4.0, 6.0))
    s = LayeredSphere(llf_radii, (LL.core, LL.shell(), LL.outer); interfaces)
    f = LayeredSphereALVFields(s, LL.matrix, times)
    e(j) = 1.0 .* (1:3 .== j)
    h = 1.0e-5
    # ε = sym ∇u, the gradient of the displacement operator by central
    # differences in space.
    for r in (0.2, 0.65, 0.93, 1.7)
        x = r .* llf_dir
        sym_grad = sum(
            LL.every(LL.symdyad(e(j))) * (local_displacement_alv(f, x .+ h .* e(j)) - local_displacement_alv(f, x .- h .* e(j))) ./ 2h
                for j in 1:3
        )
        @test sym_grad ≈ local_gradient_gradient_concentration_alv(f, x) rtol = 1.0e-7
    end
    # Far from the sphere the remote history is recovered.
    @test local_gradient_gradient_concentration_alv(f, 1.0e4 .* llf_dir) ≈ Matrix(1.0I, 6nt, 6nt) atol = 1.0e-10
    @test local_flux_gradient_concentration_alv(f, 1.0e4 .* llf_dir) ≈ LL.V._trapezoidal_relaxation(LL.matrix, times, 6) atol = 1.0e-9
    # The stress is the relaxation of the region on the LEFT of the strain; the
    # laws being aging, the other order differs.
    C = LL.V._trapezoidal_relaxation(LL.shell(), times, 6)
    A = local_gradient_gradient_concentration_alv(f, 0.65 .* llf_dir)
    @test local_flux_gradient_concentration_alv(f, 0.65 .* llf_dir) ≈ C * A rtol = 1.0e-13
    @test !(C * A ≈ A * C)
    # At the center the operator is isotropic, and finite.
    A₀ = local_gradient_gradient_concentration_alv(f, zeros(3))
    α, β = iso_params_from_blocks(A₀)
    @test A₀ ≈ iso_blocks_from_params(α, β) rtol = 1.0e-13
    @test A₀ ≈ local_gradient_gradient_concentration_alv(f, 1.0e-9 .* llf_dir) rtol = 1.0e-10
    @test iszero(local_displacement_alv(f, zeros(3)))

    # Conduction: ∇T from the temperature operator, the far field, the order.
    s = LayeredSphere(llf_radii, (LL.K1(), LL.K2, LL.K3); interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3), KapitzaInterface(0.4)))
    f = LayeredSphereALVFields(s, LL.K0, times)
    for r in (0.2, 0.65, 0.93, 1.7)
        x = r .* llf_dir
        grad = sum(
            LL.every(reshape(e(j), 3, 1)) * (local_temperature_alv(f, x .+ h .* e(j)) - local_temperature_alv(f, x .- h .* e(j))) ./ 2h
                for j in 1:3
        )
        @test grad ≈ local_gradient_gradient_concentration_alv(f, x) rtol = 1.0e-7
    end
    @test local_gradient_gradient_concentration_alv(f, 1.0e4 .* llf_dir) ≈ Matrix(1.0I, 3nt, 3nt) atol = 1.0e-10
    K = LL.V._trapezoidal_relaxation(LL.K2, times, 3)
    A = local_gradient_gradient_concentration_alv(f, 0.65 .* llf_dir)
    @test local_flux_gradient_concentration_alv(f, 0.65 .* llf_dir) ≈ -K * A rtol = 1.0e-13
    @test !(K * A ≈ A * K)
    @test iszero(local_temperature_alv(f, zeros(3)))
end

@testset "Pointwise ALV operators — an interface is a thin elastic layer" begin
    # An interface does not age: a spring of stiffnesses (kₙ, kₜ) is the limit
    # of an elastic layer of thickness h with k + 4μ/3 = kₙ h, μ = kₜ h; a
    # resistance ρ, of a layer of conductivity h/ρ; a conducting surface kˢ, of
    # a layer of conductivity kˢ/h. Between aging layers, away from the layer.
    times = LL.times
    h = 1.0e-5
    r = 0.6
    points = (0.3 .* llf_dir, 0.85 .* llf_dir, 1.4 .* llf_dir)
    kₙ, kₜ = 5.0, 2.0
    thin = TensISO{3}(3 * (kₙ - 4kₜ / 3) * h, 2 * kₜ * h)
    f = LayeredSphereALVFields(
        LayeredSphere((r, 1.0), (LL.core, LL.outer); interfaces = (SpringInterface(kₙ, kₜ), PerfectInterface())), LL.matrix, times
    )
    g = LayeredSphereALVFields(LayeredSphere((r - h / 2, r + h / 2, 1.0), (LL.core, thin, LL.outer)), LL.matrix, times)
    for x in points, op in (local_gradient_gradient_concentration_alv, local_flux_gradient_concentration_alv, local_displacement_alv)
        @test op(f, x) ≈ op(g, x) rtol = 2.0e-4
    end
    for (interface, k_thin) in ((KapitzaInterface(0.2), h / 0.2), (SurfaceConductiveInterface(0.3), 0.3 / h))
        f = LayeredSphereALVFields(
            LayeredSphere((r, 1.0), (LL.K1(), LL.K2); interfaces = (interface, PerfectInterface())), LL.K0, times
        )
        g = LayeredSphereALVFields(LayeredSphere((r - h / 2, r + h / 2, 1.0), (LL.K1(), TensISO{3}(k_thin), LL.K2)), LL.K0, times)
        for x in points, op in (local_gradient_gradient_concentration_alv, local_flux_gradient_concentration_alv, local_temperature_alv)
            @test op(f, x) ≈ op(g, x) rtol = 2.0e-4
        end
    end
end

@testset "Pointwise ALV operators — derivative with respect to a layer modulus" begin
    times = LL.times
    interfaces = (SpringInterface(5.0, 2.0), MembraneInterface(1.3, 0.7), SpringInterface(4.0, 6.0))
    fields(p) = LayeredSphereALVFields(LayeredSphere(llf_radii, (LL.core, LL.shell(p), LL.outer); interfaces), LL.matrix, times)
    h = 1.0e-5
    # The shear modulus of the shell, read in the core, in the shell, in the
    # outer layer and in the matrix.
    for r in (0.2, 0.65, 0.93, 1.7), op in (local_gradient_gradient_concentration_alv, local_flux_gradient_concentration_alv, local_displacement_alv)
        F(p) = op(fields(p), r .* llf_dir)
        @test ForwardDiff.derivative(F, 1.0) ≈ (F(1 + h) - F(1 - h)) / 2h rtol = 1.0e-6
    end
    fields2(p) = LayeredSphereALVFields(
        LayeredSphere(llf_radii, (LL.K1(p), LL.K2, LL.K3); interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3), PerfectInterface())),
        LL.K0, times
    )
    for r in (0.2, 0.65, 1.7), op in (local_gradient_gradient_concentration_alv, local_flux_gradient_concentration_alv, local_temperature_alv)
        F(p) = op(fields2(p), r .* llf_dir)
        @test ForwardDiff.derivative(F, 1.0) ≈ (F(1 + h) - F(1 - h)) / 2h rtol = 1.0e-6
    end
end

@testset "Pointwise ALV operators — impermeable shell, and what is refused" begin
    times = LL.times
    radii = (0.4, 0.6, 0.8, 1.0)
    interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3), KapitzaInterface(0.25), SurfaceConductiveInterface(0.2))
    fields(k) = LayeredSphereALVFields(LayeredSphere(radii, (LL.K1(), TensISO{3}(k), LL.K2, LL.K3); interfaces), LL.K0, times)
    f₀, fε = fields(0.0), fields(1.0e-10)
    for r in (0.2, 0.5, 0.7, 0.9, 1.5), op in (local_gradient_gradient_concentration_alv, local_flux_gradient_concentration_alv, local_temperature_alv)
        @test op(f₀, r .* llf_dir) ≈ op(fε, r .* llf_dir) rtol = 1.0e-8 atol = 1.0e-8
    end
    # No field inside the shell, no flux in it, and its temperature zero on its
    # inner face.
    @test iszero(local_gradient_gradient_concentration_alv(f₀, 0.2 .* llf_dir))
    @test iszero(local_flux_gradient_concentration_alv(f₀, 0.5 .* llf_dir))
    @test norm(local_gradient_gradient_concentration_alv(f₀, 0.5 .* llf_dir)) > 0.1
    @test norm(local_temperature_alv(f₀, 0.4, llf_angles...)) ≤ 1.0e-14

    f = LayeredSphereALVFields(LayeredSphere((0.6, 1.0), (LL.core, LL.outer)), LL.matrix, times)
    @test_throws "origin belongs to the core" local_gradient_gradient_concentration_alv(f, zeros(3); layer = 2)
    @test_throws "layer must be in 1..3" local_gradient_gradient_concentration_alv(f, 0.3 .* llf_dir; layer = 4)
    @test_throws "side" local_flux_gradient_concentration_alv(f, 0.3 .* llf_dir; side = :middle)
end
