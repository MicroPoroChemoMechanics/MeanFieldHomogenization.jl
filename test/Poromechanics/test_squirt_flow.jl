using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# Quartz-like solid (moduli in GPa) and water, the values of Jakobsen et al. (2003).
const K0_SQ, MU0_SQ, KF_SQ = 37.0, 44.0, 2.2
const C0_SQ = TensISO{3}(3 * K0_SQ, 2 * MU0_SQ)

# Kelvin-Mandel components in the canonical basis (a matrix at order 4, a vector at order 2).
_km(t) = mandel(t, TensND.CanonicalBasis{3, Float64}())
_rel(A, B) = norm(A - B) / norm(B)

# Two flat crack families of different orientations and spherical pores, all
# filled with the stiffness Cf (ignored by `squirt_flow`, used for the sealed cell).
function _squirt_rve(Cf; ξ = 1.0e-3, φc = 6.0e-4, φp = 0.1)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0_SQ); fraction = :rest)
    add_phase!(rve, :c1, Spheroid(ξ), Dict(:C => Cf); fraction = φc)
    add_phase!(rve, :c2, Spheroid(ξ; euler_angles = (1.2, 0.4, 0.0)), Dict(:C => Cf); fraction = φc)
    φp > 0 && add_phase!(rve, :p, Ellipsoid(1.0), Dict(:C => Cf); fraction = φp)
    return rve
end
const DRY_SQ = TensISO{3}(0.0, 0.0)
const FLUID_SQ = [:c1, :c2, :p]

@testset "squirt flow: the sealed and the relaxed limits" begin
    rve = _squirt_rve(DRY_SQ)
    sealed_rve = _squirt_rve(TensISO{3}(3 * KF_SQ, 0.0))
    for (scheme, tol) in ((MoriTanaka(), 1.0e-13), (SelfConsistent(), 1.0e-8))
        # ωτ → ∞: no exchange, the fluid phases of modulus k_f
        hi = squirt_flow(rve, scheme, Inf; fluid = FLUID_SQ, k_f = KF_SQ)
        @test _rel(_km(hi.stiffness), _km(homogenize(sealed_rve, scheme))) < tol
        ct = concentration_tensors(sealed_rve, scheme)
        for r in FLUID_SQ
            @test norm(_km(hi.pressure[r] - pore_pressure_tensor(ct, r))) < tol
        end
        # ωτ = 0: one pressure in all the pores, Gassmann and Brown-Korringa
        C_d = homogenize(rve, scheme)
        par = poroelastic_parameters(C_d, C0_SQ, sum(volume_fraction(rve, r) for r in FLUID_SQ); k_f = KF_SQ)
        lo = squirt_flow(rve, scheme, 0.0; fluid = FLUID_SQ, k_f = KF_SQ)
        @test _rel(_km(lo.stiffness), _km(undrained_stiffness(C_d, par.B, par.modulus))) < tol
        # in between: reciprocity (major symmetry) and, the time dependence
        # being e^{iωt}, dissipation (Im C positive semidefinite, not zero)
        mid = squirt_flow(rve, scheme, 1.0; fluid = FLUID_SQ, k_f = KF_SQ)
        S, C = Matrix(_km(mid.compliance)), Matrix(_km(mid.stiffness))
        @test _rel(transpose(S), S) < tol
        @test C * S ≈ I
        λ = eigvals(Symmetric(imag(C)))
        @test minimum(λ) > -tol * maximum(λ)
        @test maximum(λ) > 1.0e-3 * opnorm(real(C))
    end
end

# Two families on exactly orthogonal axes: Mori-Tanaka adds transversely
# isotropic tensors of complex components on orthogonal axes.
@testset "squirt flow: families on orthogonal axes" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0_SQ); fraction = :rest)
    add_phase!(rve, :c1, Spheroid(1.0e-3), Dict(:C => DRY_SQ); fraction = 6.0e-4)
    add_phase!(rve, :c2, Spheroid(1.0e-3; euler_angles = (π / 2, 0.0, 0.0)), Dict(:C => DRY_SQ); fraction = 6.0e-4)
    sealed = RVE()
    add_phase!(sealed, :M, Ellipsoid(1.0), Dict(:C => C0_SQ); fraction = :rest)
    for (name, angles) in ((:c1, ()), (:c2, (π / 2, 0.0, 0.0)))
        add_phase!(sealed, name, Spheroid(1.0e-3; euler_angles = angles), Dict(:C => TensISO{3}(3 * KF_SQ, 0.0)); fraction = 6.0e-4)
    end
    hi = squirt_flow(rve, MoriTanaka(), Inf; fluid = [:c1, :c2], k_f = KF_SQ)
    @test _rel(_km(hi.stiffness), _km(homogenize(sealed, MoriTanaka()))) < 1.0e-13
    mid = squirt_flow(rve, MoriTanaka(), 1.0; fluid = [:c1, :c2], k_f = KF_SQ)
    @test maximum(abs, imag(Matrix(_km(mid.stiffness)))) > 0
end

@testset "squirt flow: one aligned family does not disperse" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0_SQ); fraction = :rest)
    add_phase!(rve, :c, Spheroid(1.0e-3), Dict(:C => DRY_SQ); fraction = 6.0e-4)
    C(ωτ) = _km(squirt_flow(rve, MoriTanaka(), ωτ; fluid = [:c], k_f = KF_SQ).stiffness)
    @test _rel(C(0.0), C(Inf)) < 1.0e-13
    @test _rel(C(3.0), C(Inf)) < 1.0e-13
end

# Guéguen and Sarout (2011, eq. 10): to first order in the crack density, thin
# cracks of normals n_r and densities ρ_r, compliance non-interacting,
#   S^hf − S^lf = h (1 − ν₀/2)/(1 + δ_f) (α ⊗ α / tr α − β),
# h = 32(1 − ν₀²)/(3(2 − ν₀)E₀), δ_f = π ξ E₀ (1/k_f − 1/k₀)/(4(1 − ν₀²)),
# α = Σ ρ_r n_r ⊗ n_r, β = Σ ρ_r n_r ⊗ n_r ⊗ n_r ⊗ n_r.
@testset "squirt flow: Guéguen and Sarout, thin cracks" begin
    ξ, ρ = 1.0e-5, 0.02
    φc = 4π / 3 * ρ * ξ                         # ρ = N a³/V for spheroids of aspect ratio ξ
    k_f = 2.0e-4                                # δ_f of order one at this aspect ratio
    rve = _squirt_rve(DRY_SQ; ξ, φc, φp = 0.0)
    fluid = [:c1, :c2]
    S_hf = _km(squirt_flow(rve, DiluteDual(), Inf; fluid, k_f).compliance)
    S_lf = _km(squirt_flow(rve, DiluteDual(), 0.0; fluid, k_f).compliance)
    E₀ = 9 * K0_SQ * MU0_SQ / (3 * K0_SQ + MU0_SQ)
    ν₀ = (3 * K0_SQ - 2 * MU0_SQ) / (2 * (3 * K0_SQ + MU0_SQ))
    h = 32 * (1 - ν₀^2) / (3 * (2 - ν₀) * E₀)
    δ_f = π * ξ * E₀ * (1 / k_f - 1 / K0_SQ) / (4 * (1 - ν₀^2))
    θ, ϕ = 1.2, 0.4                             # the axis of the second family (ZYZ angles)
    normals = ([0.0, 0.0, 1.0], [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)])
    nn(n) = Tens(n * transpose(n))
    α = sum(ρ * nn(n) for n in normals)
    β = sum(ρ * (nn(n) ⊗ nn(n)) for n in normals)
    ref = _km(h * (1 - ν₀ / 2) / (1 + δ_f) * ((α ⊗ α) / tr(get_array(α)) - β))
    @test _rel(S_hf - S_lf, ref) < 1.0e-3
end

@testset "squirt flow: derivative with respect to the fluid modulus" begin
    rve = _squirt_rve(DRY_SQ)
    C_d = homogenize(rve, MoriTanaka())
    φ = sum(volume_fraction(rve, r) for r in FLUID_SQ)
    via_squirt(k) = real(_km(squirt_flow(rve, MoriTanaka(), 0.0; fluid = FLUID_SQ, k_f = k).stiffness)[1, 1])
    via_biot(k) = (p = poroelastic_parameters(C_d, C0_SQ, φ; k_f = k); _km(undrained_stiffness(C_d, p.B, p.modulus))[1, 1])
    @test ForwardDiff.derivative(via_squirt, KF_SQ) ≈ ForwardDiff.derivative(via_biot, KF_SQ) rtol = 1.0e-10
end

@testset "squirt flow: refusals" begin
    rve = _squirt_rve(DRY_SQ)
    @test_throws DomainError squirt_flow(rve, MoriTanaka(), -1.0; fluid = FLUID_SQ, k_f = KF_SQ)
    @test_throws ArgumentError squirt_flow(rve, MoriTanaka(), 1.0; fluid = Symbol[], k_f = KF_SQ)
    @test_throws ArgumentError squirt_flow(rve, MoriTanaka(), 1.0; fluid = [:M, :c1], k_f = KF_SQ)
    @test_throws ArgumentError squirt_flow(rve, MoriTanaka(), 1.0; fluid = [:nope], k_f = KF_SQ)
    @test_throws ArgumentError squirt_flow(rve, MoriTanaka(), 1.0; fluid = [:c1, :c2], k_f = KF_SQ)
end
