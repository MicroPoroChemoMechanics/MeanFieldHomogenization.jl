using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_layered_alv_order2.jl — the n-layer sphere in conduction, in the aging
#  linear Volterra (ALV) setting.
#
#  Echoes parametrizes the viscoelastic reference of its n-layer sphere by a
#  bulk and a shear kernel, so it gives no reference in conduction. The
#  references here are exact identities, on an aging core and a non-uniform
#  grid, where the Volterra matrices do not commute: the elastic limit, a solve
#  of the interface conditions that does not use the closed-form transitions,
#  the equivalent sphere of a Kapitza or a surface-conductive interface, and the
#  one-layer sphere against the ellipsoid path, itself checked against Echoes.
# =============================================================================

const V2 = MeanFieldHomogenization.Viscoelasticity
const times2 = [0.0, 0.3, 0.7, 1.2, 2.0]
const K0_law = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(2.0 * exp(-(t - tp) / 0.9)) : TensISO{3}(0.0))
const K1_law = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(5.0 * (1 + 0.5tp) * exp(-(t - tp) / 1.4)) : TensISO{3}(0.0))
const K2_law = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}((1 + tp) * exp(-(t - tp) / 0.7)) : TensISO{3}(0.0))
kmat(law) = iso_order2_params_from_blocks(V2._trapezoidal_relaxation(law, times2, 3))

@testset "Layered sphere, order 2 — the elastic limit" begin
    # Heaviside laws: every time step is the elastic problem.
    times = [0.0, 0.4, 1.5]
    K₀ = TensISO{3}(2.0)
    Id = Matrix(1.0I, 9, 9)
    for interfaces in (
                (KapitzaInterface(0.1), KapitzaInterface(0.25)),
                (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3)),
                (PerfectInterface(), PerfectInterface()),
            ), external in (true, false)
        s = LayeredSphere((0.6, 1.0), (TensISO{3}(5.0), TensISO{3}(1.0)); interfaces)
        A = gradient_gradient_loc_alv(s, heaviside_law(K₀), times; external)
        N = conductivity_contribution_alv(s, heaviside_law(K₀), times; external)
        @test A ≈ TensND.get_data(gradient_gradient_loc(s, K₀, K₀; external))[1] .* Id atol = 1.0e-14
        @test N ≈ TensND.get_data(MeanFieldHomogenization.conductivity_contribution(s, K₀; external))[1] .* Id atol = 1.0e-14
    end
    # An impermeable core needs no Volterra inverse of its conductivity.
    s = LayeredSphere(
        (0.6, 1.0), (TensISO{3}(0.0), TensISO{3}(1.0));
        interfaces = (SurfaceConductiveInterface(0.3), PerfectInterface())
    )
    @test gradient_gradient_loc_alv(s, heaviside_law(K₀), times) ≈
        TensND.get_data(gradient_gradient_loc(s, K₀, K₀))[1] .* Id atol = 1.0e-14
    @test conductivity_contribution_alv(s, heaviside_law(K₀), times) ≈
        TensND.get_data(MeanFieldHomogenization.conductivity_contribution(s, K₀))[1] .* Id atol = 1.0e-14
end

@testset "Layered sphere, order 2 — the interface conditions solved as one system" begin
    # Unknowns A₁, A₂, B₂, B∞ (n × n each) for A∞ = 1, the four conditions at
    # the two interfaces written as they stand — u = rA + B/r², s = k ∘ (A − 2B/r³),
    # [T] = ρ s, [s] = 2kˢ u/r² — and solved together. The recurrence composes
    # the same operators in closed form; any misplaced factor shows here.
    n = length(times2)
    k₀, k₁, k₂ = kmat(K0_law), kmat(K1_law), kmat(K2_law)
    Id = Matrix(1.0I, n, n)
    for interfaces in (
            (KapitzaInterface(0.1), KapitzaInterface(0.25)),
            (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3)),
            (SurfaceConductiveInterface(0.2), KapitzaInterface(0.25)),
        )
        ρ(i) = interfaces[i] isa KapitzaInterface ? interfaces[i].resistance : 0.0
        kˢ(i) = interfaces[i] isa SurfaceConductiveInterface ? interfaces[i].conductance : 0.0
        r₁, r₂ = 0.6, 1.0
        M = zeros(4n, 4n)
        rhs = zeros(4n, n)
        add!(i, j, X) = (M[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)] .+= X)
        # r₁: the core (A₁, B₁ = 0) against the shell (A₂, B₂).
        add!(1, 2, r₁ * Id); add!(1, 3, Id / r₁^2); add!(1, 1, -(r₁ * Id + ρ(1) * k₁))
        add!(2, 2, k₂); add!(2, 3, -2 / r₁^3 * k₂); add!(2, 1, -(k₁ + 2kˢ(1) / r₁ * Id))
        # r₂: the shell against the reference (A∞ = 1, B∞).
        add!(3, 4, Id / r₂^2); add!(3, 2, -(r₂ * Id + ρ(2) * k₂)); add!(3, 3, -(Id / r₂^2 - 2ρ(2) / r₂^3 * k₂))
        rhs[(2n + 1):(3n), :] .= -r₂ * Id
        add!(4, 4, -2 / r₂^3 * k₀)
        add!(4, 2, -(k₂ + 2kˢ(2) / r₂ * Id)); add!(4, 3, -(-2 / r₂^3 * k₂ + 2kˢ(2) / r₂^4 * Id))
        rhs[(3n + 1):(4n), :] .= -k₀
        X = M \ rhs
        α = gradient_localization_alv(LayeredSphere((r₁, r₂), (K1_law, K2_law); interfaces), K0_law, times2)
        @test α[1] ≈ X[1:n, :] rtol = 1.0e-12
        @test α[2] ≈ X[(n + 1):(2n), :] rtol = 1.0e-12
    end
end

@testset "Layered sphere, order 2 — the equivalent sphere of an interface" begin
    # A Kapitza interface around a core of conductivity k₁ is a sphere of
    # conductivity k₁ ∘ (1 + ρ k₁/R)^{-∘}; a surface-conductive one, of k₁ + 2kˢ/R.
    # Their concentration and contribution are those of the ellipsoid path.
    n = length(times2)
    R = 0.8
    k₁ = kmat(K1_law)
    K_ref = V2._trapezoidal_relaxation(K0_law, times2, 3)
    P = hill_kernel_order2(Ellipsoid(1.0, 1.0, 1.0), K0_law, times2)
    for (interface, k_eq) in (
            (KapitzaInterface(0.2), k₁ * inv(Matrix(1.0I, n, n) + 0.2 * k₁ / R)),
            (SurfaceConductiveInterface(0.3), k₁ + 2 * 0.3 / R * I),
        )
        s = LayeredSphere((R,), (K1_law,); interfaces = (interface,))
        K_eq = iso_order2_blocks_from_params(Matrix(k_eq))
        @test gradient_gradient_loc_alv(s, K0_law, times2) ≈ dilute_concentration_alv_order2(K_eq, K_ref, P) rtol = 1.0e-12
        @test conductivity_contribution_alv(s, K0_law, times2) ≈ dilute_contribution_alv_order2(K_eq, K_ref, P) rtol = 1.0e-12
    end
    # Leaving the outer interface to the matrix drops its jump from the average,
    # not its effect on the field: what remains is the material average.
    s = LayeredSphere((R,), (K1_law,); interfaces = (KapitzaInterface(0.2),))
    @test gradient_gradient_loc_alv(s, K0_law, times2; external = false) ≈
        iso_order2_blocks_from_params(gradient_localization_alv(s, K0_law, times2)[1]) rtol = 1.0e-14
end

@testset "Layered sphere, order 2 — every scheme, one layer against the ellipsoid" begin
    function cell(geometry)
        r = RVE(; distribution_shape = Ellipsoid(1.0, 1.0, 1.0))
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K0_law); fraction = :rest)
        add_phase!(r, :I, geometry, Dict(:K => K1_law); fraction = 0.3)
        return r
    end
    for scheme in (
            Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(), Maxwell(),
            PonteCastanedaWillis(), SelfConsistent(), DifferentialScheme(),
        )
        @test homogenize_alv(cell(LayeredSphere((1.0,), (K1_law,))), scheme, :K; times = times2) ≈
            homogenize_alv(cell(Ellipsoid(1.0, 1.0, 1.0)), scheme, :K; times = times2) rtol = 1.0e-10
    end
    # A two-layer sphere with interfaces through a scheme: Mori–Tanaka is
    # Maxwell for a single family of spheres.
    s = LayeredSphere((0.6, 1.0), (K1_law, K2_law); interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3)))
    @test homogenize_alv(cell(s), MoriTanaka(), :K; times = times2) ≈
        homogenize_alv(cell(s), Maxwell(), :K; times = times2) rtol = 1.0e-12
end

@testset "Layered sphere, order 2 — what is refused" begin
    s = LayeredSphere((0.6, 1.0), (K1_law, K2_law); interfaces = (SpringInterface(5.0, 2.0), PerfectInterface()))
    @test_throws "elastic interface" gradient_gradient_loc_alv(s, K0_law, times2)
    hot = LayeredSphere((0.6, 1.0), (TensISO{3}(Inf), TensISO{3}(1.0)))
    @test_throws "infinite conductivity" gradient_gradient_loc_alv(hot, K0_law, times2)
    @test_throws "reference medium" gradient_gradient_loc_alv(
        LayeredSphere((1.0,), (K1_law,)), heaviside_law(TensISO{3}(Inf)), times2
    )
    @test_throws "no Hill kernel" V2.hill_kernel_order2_at(
        LayeredSphere((1.0,), (K1_law,)), V2._trapezoidal_relaxation(K0_law, times2, 3)
    )
end
