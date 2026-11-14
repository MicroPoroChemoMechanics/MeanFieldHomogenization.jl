using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_layered_alv_layers.jl — the operators of ONE layer of an n-layer sphere
#  in aging linear viscoelasticity, orders 4 and 2, and the impermeable shell.
#
#  Echoes gives the operators of the whole sphere only (`sphere_visco_eE`), so
#  the references here are identities: the elastic limit, the partition of the
#  sphere into its shells, the interface conditions solved as one system
#  (conduction), the contribution written from the average stress, and the
#  limit of a vanishing conductivity. Aging laws on a non-uniform grid
#  throughout, where the Volterra matrices do not commute. The averages of the
#  pointwise operators over a shell are in `test_layered_alv_local_fields.jl`.
# =============================================================================

module LayeredALVLayersData
    using MeanFieldHomogenization, TensND
    const V = MeanFieldHomogenization.Viscoelasticity
    const times = [0.0, 0.3, 0.7, 1.2, 2.0]
    iso4(a, b) = (t, tp) -> t >= tp ? TensISO{3}(a(t, tp), b(t, tp)) : TensISO{3}(0.0, 0.0)
    # An aging core, an aging shell (scaled by `p`, the parameter of the
    # derivative test), a Maxwell outer layer and a Maxwell matrix.
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
    const K1 = ViscoLaw(iso2((t, tp) -> 5.0 * (1 + 0.5tp) * exp(-(t - tp) / 1.4)))
    const K2 = ViscoLaw(iso2((t, tp) -> (1 + tp) * exp(-(t - tp) / 0.7)))
    const K3 = ViscoLaw(iso2((t, tp) -> 3.0 * (1 + 0.2tp) * exp(-(t - tp) / 1.1)))
    kmat(law) = iso_order2_params_from_blocks(V._trapezoidal_relaxation(law, times, 3))
    relax4(law) = V._trapezoidal_relaxation(law, times, 6)
    fractions(s) = [MeanFieldHomogenization.LayeredSpheres.layer_volume_fraction(s, k) for k in 1:length(s.radii)]
end
const LD = LayeredALVLayersData

@testset "Layers of a sphere in ALV, order 4 — the elastic limit" begin
    # Heaviside laws: every operator is the elastic tensor at every instant.
    times = [0.0, 0.4, 1.5]
    n = length(times)
    Id = Matrix(1.0I, n, n)
    C₀ = TensISO{3}(3.0, 1.0)
    C = (TensISO{3}(6.0, 2.0), TensISO{3}(9.0, 3.0), TensISO{3}(4.0, 1.5))
    blocks(A) = iso_blocks_from_params((TensND.get_data(A) .* Ref(Id))...)
    for interfaces in (
            (PerfectInterface(), PerfectInterface(), PerfectInterface()),
            (SpringInterface(5.0, 2.0), SpringInterface(8.0, 3.0), SpringInterface(4.0, 6.0)),
            (MembraneInterface(1.3, 0.7), MembraneInterface(0.4, 0.9), MembraneInterface(0.8, 0.2)),
            (SpringInterface(5.0, 2.0), MembraneInterface(1.3, 0.7), SpringInterface(4.0, 6.0)),
        )
        s = LayeredSphere((0.5, 0.8, 1.0), C; interfaces)
        ref = heaviside_law(C₀)
        for k in 1:3
            for external in (false, true), internal in (false, true)
                @test gradient_gradient_concentration_alv(s, ref, times; layer = k, external, internal) ≈
                    blocks(gradient_gradient_concentration(s, C₀; layer = k, external, internal)) atol = 1.0e-13
            end
            @test flux_gradient_concentration_alv(s, ref, times; layer = k) ≈
                blocks(C[k] ⊡ gradient_gradient_concentration(s, C₀; layer = k)) atol = 1.0e-13
            # The name of elasticity forwards the keywords.
            @test strain_strain_concentration_alv(s, ref, times; layer = k, external = true) ==
                gradient_gradient_concentration_alv(s, ref, times; layer = k, external = true)
        end
        for external in (false, true)
            @test flux_gradient_concentration_alv(s, ref, times; external) ≈
                blocks(flux_gradient_concentration(s, C₀, C₀; external)) atol = 1.0e-13
        end
    end
end

@testset "Layers of a sphere in ALV, order 4 — the shells partition the sphere" begin
    times = LD.times
    for interfaces in (
            (SpringInterface(5.0, 2.0), SpringInterface(8.0, 3.0), SpringInterface(4.0, 6.0)),
            (SpringInterface(5.0, 2.0), MembraneInterface(1.3, 0.7), MembraneInterface(0.8, 0.2)),
            (MembraneInterface(1.3, 0.7), PerfectInterface(), SpringInterface(4.0, 6.0)),
        )
        s = LayeredSphere((0.5, 0.8, 1.0), (LD.core, LD.shell(), LD.outer); interfaces)
        f = LD.fractions(s)
        layer_sum(; kw...) = sum(f[k] .* gradient_gradient_concentration_alv(s, LD.matrix, times; layer = k, kw...) for k in 1:3)
        # Each interface given to the layer inside it: the whole sphere, outer
        # interface included. Given to the layer outside it: without the outer one.
        @test layer_sum(external = true) ≈ gradient_gradient_concentration_alv(s, LD.matrix, times) rtol = 1.0e-12
        @test layer_sum(internal = true) ≈
            gradient_gradient_concentration_alv(s, LD.matrix, times; external = false) rtol = 1.0e-12
        # By default a layer is its material: a jump is counted only when asked.
        material = [gradient_gradient_concentration_alv(s, LD.matrix, times; layer = k) for k in 1:3]
        α = bulk_concentration_alv(s, LD.matrix, times)
        β = shear_concentration_alv(s, LD.matrix, times)
        @test all(material[k] ≈ iso_blocks_from_params(α[k], β[k]) for k in 1:3)
        @test gradient_gradient_concentration_alv(s, LD.matrix, times; layer = 1, internal = true) == material[1]
        if interfaces[1] isa SpringInterface
            @test !(gradient_gradient_concentration_alv(s, LD.matrix, times; layer = 2, internal = true) ≈ material[2])
        end

        # The stress of a layer is its relaxation operator on the LEFT of its
        # strain; the other order differs, the laws being aging.
        C = (LD.relax4(LD.core), LD.relax4(LD.shell()), LD.relax4(LD.outer))
        flux = [flux_gradient_concentration_alv(s, LD.matrix, times; layer = k) for k in 1:3]
        @test all(flux[k] ≈ C[k] * material[k] for k in 1:3)
        @test !(flux[1] ≈ material[1] * C[1])
        # The contribution of the sphere is its average stress — that of its
        # layers and of its membranes — minus the reference applied to its
        # average strain, jumps included.
        for external in (false, true)
            whole = flux_gradient_concentration_alv(s, LD.matrix, times; external)
            @test property_contribution_alv(s, LD.matrix, times; external) ≈
                whole - LD.relax4(LD.matrix) * gradient_gradient_concentration_alv(s, LD.matrix, times; external) rtol = 1.0e-11
            membrane = any(i -> i isa MembraneInterface, interfaces[1:(external ? 3 : 2)])
            @test (whole ≈ sum(f[k] .* flux[k] for k in 1:3)) == !membrane
        end
    end
end

# The amplitudes A₁, A₂, B₂, B∞ (n × n each, for A∞ = 1) of a two-layer sphere
# in conduction, from the four interface conditions written as they stand —
# u = rA + B/r², s = k ∘ (A − 2B/r³), [T] = ρ s, [s] = 2kˢ u/r² — and solved
# together, without the closed-form transitions of the recurrence.
function lal_two_layer_conduction(r₁, r₂, interfaces)
    n = length(LD.times)
    k₀, k₁, k₂ = LD.kmat(LD.K0), LD.kmat(LD.K1), LD.kmat(LD.K2)
    Id = Matrix(1.0I, n, n)
    ρ(i) = interfaces[i] isa KapitzaInterface ? interfaces[i].resistance : 0.0
    kˢ(i) = interfaces[i] isa SurfaceConductiveInterface ? interfaces[i].conductance : 0.0
    M = zeros(4n, 4n)
    rhs = zeros(4n, n)
    add!(i, j, X) = (M[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)] .+= X)
    add!(1, 2, r₁ * Id); add!(1, 3, Id / r₁^2); add!(1, 1, -(r₁ * Id + ρ(1) * k₁))
    add!(2, 2, k₂); add!(2, 3, -2 / r₁^3 * k₂); add!(2, 1, -(k₁ + 2kˢ(1) / r₁ * Id))
    add!(3, 4, Id / r₂^2); add!(3, 2, -(r₂ * Id + ρ(2) * k₂)); add!(3, 3, -(Id / r₂^2 - 2ρ(2) / r₂^3 * k₂))
    rhs[(2n + 1):(3n), :] .= -r₂ * Id
    add!(4, 4, -2 / r₂^3 * k₀)
    add!(4, 2, -(k₂ + 2kˢ(2) / r₂ * Id)); add!(4, 3, -(-2 / r₂^3 * k₂ + 2kˢ(2) / r₂^4 * Id))
    rhs[(3n + 1):(4n), :] .= -k₀
    X = M \ rhs
    blk(i) = X[((i - 1) * n + 1):(i * n), :]
    return (; A₁ = blk(1), A₂ = blk(2), B₂ = blk(3), k₁, k₂, ρ, kˢ)
end

@testset "Layers of a sphere in ALV, order 2 — the interface conditions solved as one system" begin
    times = LD.times
    r₁, r₂ = 0.6, 1.0
    V₁, V₂ = r₁^3, r₂^3 - r₁^3
    for interfaces in (
            (KapitzaInterface(0.1), KapitzaInterface(0.25)),
            (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3)),
            (SurfaceConductiveInterface(0.2), KapitzaInterface(0.25)),
        )
        x = lal_two_layer_conduction(r₁, r₂, interfaces)
        # The temperature jumps r² ρ s of the two interfaces, s = k ∘ ∇T·n read
        # on the inner side.
        j₁ = r₁^2 * x.ρ(1) * (x.k₁ * x.A₁)
        j₂ = r₂^2 * x.ρ(2) * (x.k₂ * (x.A₂ - 2 / r₂^3 * x.B₂))
        s = LayeredSphere((r₁, r₂), (LD.K1, LD.K2); interfaces)
        A(k; kw...) = iso_order2_params_from_blocks(gradient_gradient_concentration_alv(s, LD.K0, times; layer = k, kw...))
        @test A(1) ≈ x.A₁ rtol = 1.0e-12
        @test A(2) ≈ x.A₂ rtol = 1.0e-12
        @test A(1; external = true) ≈ x.A₁ + j₁ / V₁ rtol = 1.0e-12
        @test A(1; external = true, internal = true) ≈ x.A₁ + j₁ / V₁ rtol = 1.0e-12
        @test A(2; internal = true) ≈ x.A₂ + j₁ / V₂ rtol = 1.0e-12
        @test A(2; external = true) ≈ x.A₂ + j₂ / V₂ rtol = 1.0e-12
        @test A(2; external = true, internal = true) ≈ x.A₂ + (j₁ + j₂) / V₂ rtol = 1.0e-12
        # The average of k ∘ ∇T in a layer: its conductivity on the left.
        B(k) = iso_order2_params_from_blocks(flux_gradient_concentration_alv(s, LD.K0, times; layer = k))
        @test B(1) ≈ x.k₁ * x.A₁ rtol = 1.0e-12
        @test B(2) ≈ x.k₂ * x.A₂ rtol = 1.0e-12
        @test !(B(1) ≈ x.A₁ * x.k₁)
        # The whole sphere: the layers, the surface flux 2kˢ u r / R³ of the
        # conducting surfaces, and the contribution that follows.
        f = LD.fractions(s)
        surface(external) = 2 * x.kˢ(1) * r₁^2 * x.A₁ / r₂^3 +
            external * 2 * x.kˢ(2) * r₂ * (r₂ * x.A₂ + x.B₂ / r₂^2) / r₂^3
        for external in (false, true)
            whole = flux_gradient_concentration_alv(s, LD.K0, times; external)
            @test iso_order2_params_from_blocks(whole) ≈ f[1] * B(1) + f[2] * B(2) + surface(external) rtol = 1.0e-12
            @test property_contribution_alv(s, LD.K0, times; external) ≈ whole -
                LD.V._trapezoidal_relaxation(LD.K0, times, 3) * gradient_gradient_concentration_alv(s, LD.K0, times; external) rtol = 1.0e-11
        end
        layer_sum(; kw...) = sum(f[k] .* gradient_gradient_concentration_alv(s, LD.K0, times; layer = k, kw...) for k in 1:2)
        @test layer_sum(external = true) ≈ gradient_gradient_concentration_alv(s, LD.K0, times) rtol = 1.0e-12
        @test layer_sum(internal = true) ≈ gradient_gradient_concentration_alv(s, LD.K0, times; external = false) rtol = 1.0e-12
    end
end

@testset "Layers of a sphere in ALV, order 2 — the elastic limit" begin
    times = [0.0, 0.4, 1.5]
    Id = Matrix(1.0I, 9, 9)
    K₀ = TensISO{3}(2.0)
    K = (TensISO{3}(5.0), TensISO{3}(1.0), TensISO{3}(3.0))
    for interfaces in (
            (KapitzaInterface(0.1), KapitzaInterface(0.25), KapitzaInterface(0.4)),
            (SurfaceConductiveInterface(0.2), KapitzaInterface(0.25), SurfaceConductiveInterface(0.3)),
            (PerfectInterface(), PerfectInterface(), PerfectInterface()),
        )
        s = LayeredSphere((0.5, 0.8, 1.0), K; interfaces)
        ref = heaviside_law(K₀)
        scalar(A) = TensND.get_data(A)[1]
        for k in 1:3
            for external in (false, true), internal in (false, true)
                @test gradient_gradient_concentration_alv(s, ref, times; layer = k, external, internal) ≈
                    scalar(gradient_gradient_concentration(s, K₀; layer = k, external, internal)) .* Id atol = 1.0e-13
            end
            @test flux_gradient_concentration_alv(s, ref, times; layer = k) ≈
                scalar(K[k]) * scalar(gradient_gradient_concentration(s, K₀; layer = k)) .* Id atol = 1.0e-13
        end
        for external in (false, true)
            @test flux_gradient_concentration_alv(s, ref, times; external) ≈
                scalar(flux_gradient_concentration(s, K₀, K₀; external)) .* Id atol = 1.0e-13
        end
    end
end

@testset "Layers of a sphere in ALV — derivative with respect to a layer modulus" begin
    times = LD.times
    # Order 4: the shear modulus of the shell, scaled by p; springs on both sides.
    interfaces = (SpringInterface(5.0, 2.0), SpringInterface(8.0, 3.0), MembraneInterface(0.8, 0.2))
    sphere(p) = LayeredSphere((0.5, 0.8, 1.0), (LD.core, LD.shell(p), LD.outer); interfaces)
    for op in (
            p -> gradient_gradient_concentration_alv(sphere(p), LD.matrix, times; layer = 2, external = true, internal = true),
            p -> flux_gradient_concentration_alv(sphere(p), LD.matrix, times; layer = 2),
            p -> flux_gradient_concentration_alv(sphere(p), LD.matrix, times),
            p -> gradient_gradient_concentration_alv(sphere(p), LD.matrix, times; layer = 3, internal = true),
        )
        h = 1.0e-5
        @test ForwardDiff.derivative(op, 1.0) ≈ (op(1 + h) - op(1 - h)) / 2h rtol = 1.0e-7
    end
    # Order 2: the conductivity of the core, scaled by p; a resistance on it.
    K1(p) = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(5.0 * p * (1 + 0.5tp) * exp(-(t - tp) / 1.4)) : TensISO{3}(zero(p)))
    sphere2(p) = LayeredSphere((0.6, 1.0), (K1(p), LD.K2); interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3)))
    for op in (
            p -> gradient_gradient_concentration_alv(sphere2(p), LD.K0, times; layer = 2, internal = true),
            p -> flux_gradient_concentration_alv(sphere2(p), LD.K0, times; layer = 1),
            p -> flux_gradient_concentration_alv(sphere2(p), LD.K0, times),
        )
        h = 1.0e-5
        @test ForwardDiff.derivative(op, 1.0) ≈ (op(1 + h) - op(1 - h)) / 2h rtol = 1.0e-7
    end
    # The interface parameters too: a resistance and a surface modulus.
    op_ρ(ρ) = gradient_gradient_concentration_alv(
        LayeredSphere((0.6, 1.0), (LD.K1, LD.K2); interfaces = (KapitzaInterface(ρ), PerfectInterface())),
        LD.K0, times; layer = 1, external = true
    )
    @test ForwardDiff.derivative(op_ρ, 0.1) ≈ (op_ρ(0.1 + 1.0e-5) - op_ρ(0.1 - 1.0e-5)) / 2.0e-5 rtol = 1.0e-7
    op_κ(κs) = flux_gradient_concentration_alv(
        LayeredSphere((0.6, 1.0), (LD.core, LD.outer); interfaces = (MembraneInterface(κs, 0.7), PerfectInterface())),
        LD.matrix, times
    )
    @test ForwardDiff.derivative(op_κ, 1.3) ≈ (op_κ(1.3 + 1.0e-5) - op_κ(1.3 - 1.0e-5)) / 2.0e-5 rtol = 1.0e-7
end

@testset "Layers of a sphere in ALV — what is refused" begin
    s = LayeredSphere((0.6, 1.0), (LD.core, LD.outer))
    @test_throws BoundsError gradient_gradient_concentration_alv(s, LD.matrix, LD.times; layer = 3)
    @test_throws BoundsError flux_gradient_concentration_alv(s, LD.matrix, LD.times; layer = 0)
    s2 = LayeredSphere((0.6, 1.0), (LD.K1, LD.K2))
    @test_throws BoundsError gradient_gradient_concentration_alv(s2, LD.K0, LD.times; layer = 3)
    @test_throws BoundsError flux_gradient_concentration_alv(s2, LD.K0, LD.times; layer = 0)
end

@testset "Impermeable shell in ALV, order 2" begin
    times = LD.times
    n = length(times)
    radii = (0.4, 0.6, 0.8, 1.0)
    Z = TensISO{3}(0.0)
    # An aging core, an impermeable shell, two aging layers; a resistance and a
    # conducting surface on the faces of the shell, a resistance further out.
    interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3), KapitzaInterface(0.25), SurfaceConductiveInterface(0.2))
    sphere(k_shell) = LayeredSphere(radii, (LD.K1, k_shell, LD.K2, LD.K3); interfaces)
    s₀ = sphere(Z)
    sε = sphere(TensISO{3}(1.0e-10))
    # The limit of a vanishing conductivity, on every operator.
    for external in (false, true)
        @test gradient_gradient_concentration_alv(s₀, LD.K0, times; external) ≈
            gradient_gradient_concentration_alv(sε, LD.K0, times; external) rtol = 1.0e-8
        @test property_contribution_alv(s₀, LD.K0, times; external) ≈
            property_contribution_alv(sε, LD.K0, times; external) rtol = 1.0e-8
        @test flux_gradient_concentration_alv(s₀, LD.K0, times; external) ≈
            flux_gradient_concentration_alv(sε, LD.K0, times; external) rtol = 1.0e-8
    end
    for k in 1:4, external in (false, true), internal in (false, true)
        @test gradient_gradient_concentration_alv(s₀, LD.K0, times; layer = k, external, internal) ≈
            gradient_gradient_concentration_alv(sε, LD.K0, times; layer = k, external, internal) rtol = 1.0e-8 atol = 1.0e-8
    end
    # The layers inside the shell are shielded, and the gradient of the shell
    # itself is not zero: its field vanishes on its inner face only.
    α = gradient_concentration_alv(s₀, LD.K0, times)
    @test iszero(α[1])
    @test all(>(0.1), diag(α[2]))
    @test iszero(flux_gradient_concentration_alv(s₀, LD.K0, times; layer = 2))
    # What is inside the shell does not matter.
    other = LayeredSphere(radii, (LD.K3, Z, LD.K2, LD.K3); interfaces = (PerfectInterface(), interfaces[2:4]...))
    @test property_contribution_alv(other, LD.K0, times) ≈ property_contribution_alv(s₀, LD.K0, times) rtol = 1.0e-13
    # A non-aging sphere: the elastic recurrence at every instant.
    K₀ = TensISO{3}(2.0)
    e = LayeredSphere(radii, (TensISO{3}(5.0), Z, TensISO{3}(1.0), TensISO{3}(3.0)); interfaces)
    Id = Matrix(1.0I, 3n, 3n)
    @test gradient_gradient_concentration_alv(e, heaviside_law(K₀), times) ≈
        TensND.get_data(gradient_gradient_concentration(e, K₀, K₀))[1] .* Id atol = 1.0e-13
    @test property_contribution_alv(e, heaviside_law(K₀), times) ≈
        TensND.get_data(MeanFieldHomogenization.property_contribution(e, K₀))[1] .* Id atol = 1.0e-13
    for k in 1:4
        @test gradient_gradient_concentration_alv(e, heaviside_law(K₀), times; layer = k, external = true) ≈
            TensND.get_data(gradient_gradient_concentration(e, K₀; layer = k, external = true))[1] .* Id atol = 1.0e-13
    end
    # An impermeable core and an impermeable shell further out.
    two = LayeredSphere(radii, (Z, LD.K2, Z, LD.K3); interfaces)
    twoε = LayeredSphere(radii, (TensISO{3}(1.0e-10), LD.K2, TensISO{3}(1.0e-10), LD.K3); interfaces)
    @test property_contribution_alv(two, LD.K0, times) ≈ property_contribution_alv(twoε, LD.K0, times) rtol = 1.0e-8
    # Two adjacent impermeable layers: the field of the outer one has no
    # conducting neighbor to be the limit of.
    @test_throws "both have a zero conductivity" gradient_gradient_concentration_alv(
        LayeredSphere(radii, (LD.K1, Z, Z, LD.K3)), LD.K0, times
    )
    @test_throws "both have a zero conductivity" property_contribution_alv(
        LayeredSphere(radii, (Z, Z, LD.K2, LD.K3)), LD.K0, times
    )
end
