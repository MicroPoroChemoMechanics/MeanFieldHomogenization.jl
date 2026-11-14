# =============================================================================
#  test_layered_alv_fe_oracle.jl — the layered sphere and the layered cylinder
#  in aging linear viscoelasticity against a radial finite-element reference.
#
#  The reference (`fe_oracle_layered.jl`) shares only the time discretization
#  with the package: the trapezoidal Volterra matrices of the laws. In space it
#  solves the weak form of equilibrium with cubic elements in r, the interfaces
#  as energies, the matrix by an exact exterior element, and forms the averages
#  from their definitions. Its error is that of the radial mesh: each comparison
#  is made on two meshes, and the error must fall at the rate of the scheme.
#
#  The laws age, differ from layer to layer and from modulus to modulus, and the
#  grid is not uniform, so that the Volterra matrices do not commute (checked
#  below): a product taken in the wrong order is an error of order 10⁻¹.
#
#  Part of the `visco-neural` part of the suite.
# =============================================================================
using Test, MeanFieldHomogenization, TensND, LinearAlgebra
isdefined(@__MODULE__, :FEOracleLayered) || include(joinpath(@__DIR__, "fe_oracle_layered.jl"))

const FEO = FEOracleLayered
const FEO_V = MeanFieldHomogenization.Viscoelasticity
const FEO_TIMES = [0.0, 0.3, 0.7, 1.2, 2.0]
const FEO_RADII = (0.5, 0.8, 1.0)
const FEO_EZ = (0.0, 0.0, 1.0)
const FEO_NELS = (10, 20)            # elements per unit length, coarse and fine

# Aging kernels, of distinct relaxation times and aging laws.
feo_g1(t, tp) = (1 + 2tp) * exp(-(t - tp) / 0.4)
feo_g2(t, tp) = exp(-(t - tp) / 3.0) / (1 + tp)
feo_g3(t, tp) = (1 + tp^2) * exp(-(t - tp) / 1.0)
feo_g4(t, tp) = (2 + sin(3tp)) * exp(-(t - tp) / 0.25)

feo_trap(law, b) = FEO_V._trapezoidal_relaxation(law, FEO_TIMES, b)
feo_iso(κ, μ, gk, gm) = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(3κ * gk(t, tp), 2μ * gm(t, tp)) : TensISO{3}(0.0, 0.0))
feo_k(k, g) = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(k * g(t, tp)) : TensISO{3}(0.0))
feo_hill(k, l, m, n, p) = ti_from_constants(k + m, k - m, l, n, p, FEO_EZ)
feo_ti(Ca, ga, Cb, gb) = ViscoLaw((t, tp) -> t >= tp ? ga(t, tp) * Ca + gb(t, tp) * Cb : 0.0 * Ca)
feo_kti(t, a, g) = ViscoLaw((τ, τp) -> τ >= τp ? g(τ, τp) * TensTI{2}(t, a, FEO_EZ) : 0.0 * TensTI{2}(t, a, FEO_EZ))

const FEO_SPH4 = (
    feo_iso(10.0, 6.0, feo_g1, feo_g3), feo_iso(2.0, 1.0, feo_g2, feo_g4), feo_iso(5.0, 3.0, feo_g3, feo_g1),
)
const FEO_SPH4_REF = feo_iso(3.0, 1.5, feo_g4, feo_g2)
const FEO_SPH2 = (feo_k(5.0, feo_g1), feo_k(1.0, feo_g2), feo_k(3.0, feo_g3))
const FEO_SPH2_REF = feo_k(2.0, feo_g4)
const FEO_CYL4 = (
    feo_ti(feo_hill(30.0, 12.0, 9.0, 150.0, 14.0), feo_g1, feo_hill(3.0, 1.0, 2.0, 5.0, 1.0), feo_g3),
    feo_ti(feo_hill(4.0, 1.5, 1.5, 6.0, 1.2), feo_g2, feo_hill(1.0, 0.2, 0.5, 1.0, 0.5), feo_g4),
    feo_ti(feo_hill(8.0, 3.0, 4.0, 12.0, 3.5), feo_g3, feo_hill(2.0, 0.5, 1.0, 2.0, 1.0), feo_g1),
)
const FEO_CYL4_REF = feo_ti(feo_hill(5.0, 1.0, 2.0, 9.0, 3.0), feo_g4, feo_hill(1.0, 0.3, 0.6, 1.5, 0.4), feo_g2)
const FEO_CYL2 = (feo_kti(5.0, 7.0, feo_g1), feo_kti(1.0, 2.0, feo_g2), feo_kti(3.0, 1.5, feo_g3))
const FEO_CYL2_REF = feo_kti(2.0, 3.0, feo_g4)

const FEO_P = FEO.OracleInterface(:perfect)
const FEO_S = FEO.OracleInterface(:spring, 20.0, 8.0)        # (kn, kt)
const FEO_M = FEO.OracleInterface(:membrane, 0.6, 0.25)      # (κs, μs)
const FEO_K = FEO.OracleInterface(:kapitza, 0.1)             # ρ
const FEO_C = FEO.OracleInterface(:surface, 0.3)             # ks

feo_pkg(i::FEO.OracleInterface) =
    i.kind === :perfect ? PerfectInterface() :
    i.kind === :spring ? SpringInterface(i.p[1], i.p[2]) :
    i.kind === :membrane ? MembraneInterface(i.p[1], i.p[2]) :
    i.kind === :kapitza ? KapitzaInterface(i.p[1]) : SurfaceConductiveInterface(i.p[1])

# The average stress (flux) of the material of each layer: a method of the sphere only.
feo_layer_flux(s::LayeredSphere, ref) =
    [flux_gradient_concentration_alv(s, ref, FEO_TIMES; layer = k) for k in eachindex(FEO_RADII)]
feo_layer_flux(::LayeredCylinder, ref) = nothing

# Every operator of `incl` against the oracle on the two meshes: the fine error below
# `tol`, and the error divided by at least `ratio` from one mesh to the next (the
# observed rate is close to 5 with cubic elements: a ratio of 30).
function feo_check(h, ctor, laws, ref, itfs, b; external = true, tol = 1.0e-7, ratio = 12)
    incl = ctor(FEO_RADII, laws; interfaces = map(feo_pkg, itfs))
    nl = length(FEO_RADII)
    pkg = (
        A = gradient_gradient_concentration_alv(incl, ref, FEO_TIMES; external),
        N = property_contribution_alv(incl, ref, FEO_TIMES; external),
        layer = [gradient_gradient_concentration_alv(incl, ref, FEO_TIMES; layer = k) for k in 1:nl],
        layer_ext = [gradient_gradient_concentration_alv(incl, ref, FEO_TIMES; layer = k, external = true) for k in 1:nl],
        layer_int = [gradient_gradient_concentration_alv(incl, ref, FEO_TIMES; layer = k, internal = true) for k in 1:nl],
        flux = feo_layer_flux(incl, ref),
    )
    errs = map(FEO_NELS) do nel
        r = FEO.oracle_solve(
            h, collect(FEO_RADII), [feo_trap(l, b) for l in laws], collect(itfs), feo_trap(ref, b);
            external, nel,
        )
        e = Dict(
            :A => FEO.compare(pkg.A, r.eps, h), :N => FEO.compare(pkg.N, r.N, h),
            :layer => maximum(FEO.compare(pkg.layer[k], r.eps_layer[k], h) for k in 1:nl),
            :layer_ext => maximum(FEO.compare(pkg.layer_ext[k], r.eps_layer_ext[k], h) for k in 1:nl),
            :layer_int => maximum(FEO.compare(pkg.layer_int[k], r.eps_layer_int[k], h) for k in 1:nl),
        )
        pkg.flux === nothing || (e[:flux] = maximum(FEO.compare(pkg.flux[k], r.sig_layer[k], h) for k in 1:nl))
        e
    end
    for key in keys(errs[2])
        coarse, fine = errs[1][key], errs[2][key]
        @test fine < tol
        @test coarse > ratio * fine
    end
    return errs
end

@testset "FE oracle — its own checks" begin
    # The kernels do not commute on this grid.
    M = [iso_order2_params_from_blocks(feo_trap(feo_k(1.0, g), 3)) for g in (feo_g1, feo_g2, feo_g3, feo_g4)]
    for a in 1:4, c in (a + 1):4
        @test norm(M[a] * M[c] - M[c] * M[a]) > 0.1 * norm(M[a] * M[c])
    end
    # A homogeneous inclusion: the exterior element and the remote load are exact.
    for (h, laws, ref, b) in (
            (FEO.sphere_elastic(), FEO_SPH4, FEO_SPH4_REF, 6), (FEO.cylinder_elastic(), FEO_CYL4, FEO_CYL4_REF, 6),
            (FEO.sphere_conduction(), FEO_SPH2, FEO_SPH2_REF, 3), (FEO.cylinder_conduction(), FEO_CYL2, FEO_CYL2_REF, 3),
        )
        R0 = feo_trap(ref, b)
        r = FEO.oracle_solve(h, collect(FEO_RADII), [R0, R0, R0], [FEO_P, FEO_P, FEO_P], R0; nel = 3)
        nt = length(FEO_TIMES)
        for ℓ in eachindex(h.loads), i in 1:nt, J in 1:nt
            @test r.eps[ℓ, i, J] ≈ (i == J ? h.loads[ℓ].E : zero(h.loads[ℓ].E)) atol = 1.0e-12
            @test norm(r.N[ℓ, i, J]) < 1.0e-12
        end
    end
    # One layer, bulk: the oracle picks the order of equilibrium,
    # α = (3k₁ + 4μ₀)⁻¹ ∘ (3k₀ + 4μ₀), and not the other one.
    K1, _ = iso_params_from_blocks(feo_trap(FEO_SPH4[1], 6))
    K0, M0 = iso_params_from_blocks(feo_trap(FEO_SPH4_REF, 6))      # 3k, 2μ
    h = FEO.sphere_elastic()
    r = FEO.oracle_solve(h, [1.0], [feo_trap(FEO_SPH4[1], 6)], [FEO_P], feo_trap(FEO_SPH4_REF, 6); nel = 4)
    α = [r.eps[1, i, J][1] for i in eachindex(FEO_TIMES), J in eachindex(FEO_TIMES)]
    @test α ≈ (K1 + 2M0) \ (K0 + 2M0) atol = 1.0e-12
    @test maximum(abs, α - (K0 + 2M0) / (K1 + 2M0)) > 0.05
    # The exterior element does not depend on where it starts.
    itfs = [FEO_S, FEO_M, FEO_S]
    laws = [feo_trap(l, 6) for l in FEO_SPH4]
    r0 = FEO.oracle_solve(h, collect(FEO_RADII), laws, itfs, feo_trap(FEO_SPH4_REF, 6); nel = 20)
    r1 = FEO.oracle_solve(h, collect(FEO_RADII), laws, itfs, feo_trap(FEO_SPH4_REF, 6); nel = 20, shell = 3.0, nel_shell = 20)
    scale = maximum(maximum(abs, r0.N[k]) for k in eachindex(r0.N))
    @test maximum(maximum(abs, r0.N[k] - r1.N[k]) for k in eachindex(r0.N)) < 1.0e-8 * scale
end

@testset "FE oracle — LayeredSphere, order 4" begin
    h = FEO.sphere_elastic()
    for itfs in (
            (FEO_P, FEO_P, FEO_P), (FEO_S, FEO_P, FEO_S), (FEO_P, FEO_M, FEO_M), (FEO_S, FEO_M, FEO_S), (FEO_M, FEO_S, FEO_M),
        )
        feo_check(h, LayeredSphere, FEO_SPH4, FEO_SPH4_REF, itfs, 6)
    end
    feo_check(h, LayeredSphere, FEO_SPH4, FEO_SPH4_REF, (FEO_M, FEO_S, FEO_S), 6; external = false)
    feo_check(h, LayeredSphere, FEO_SPH4, FEO_SPH4_REF, (FEO_S, FEO_S, FEO_M), 6; external = false)
end

@testset "FE oracle — LayeredSphere, order 2" begin
    h = FEO.sphere_conduction()
    for itfs in ((FEO_P, FEO_P, FEO_P), (FEO_K, FEO_P, FEO_K), (FEO_P, FEO_C, FEO_C), (FEO_K, FEO_C, FEO_K), (FEO_C, FEO_K, FEO_C))
        feo_check(h, LayeredSphere, FEO_SPH2, FEO_SPH2_REF, itfs, 3)
    end
    feo_check(h, LayeredSphere, FEO_SPH2, FEO_SPH2_REF, (FEO_C, FEO_K, FEO_K), 3; external = false)
end

@testset "FE oracle — LayeredCylinder, order 4" begin
    h = FEO.cylinder_elastic()
    for itfs in (
            (FEO_P, FEO_P, FEO_P), (FEO_S, FEO_P, FEO_S), (FEO_P, FEO_M, FEO_M), (FEO_S, FEO_M, FEO_S), (FEO_M, FEO_S, FEO_M),
        )
        feo_check(h, LayeredCylinder, FEO_CYL4, FEO_CYL4_REF, itfs, 6)
    end
    feo_check(h, LayeredCylinder, FEO_CYL4, FEO_CYL4_REF, (FEO_M, FEO_S, FEO_S), 6; external = false)
    feo_check(h, LayeredCylinder, FEO_CYL4, FEO_CYL4_REF, (FEO_S, FEO_S, FEO_M), 6; external = false)
end

@testset "FE oracle — LayeredCylinder, order 2" begin
    h = FEO.cylinder_conduction()
    for itfs in ((FEO_P, FEO_P, FEO_P), (FEO_K, FEO_P, FEO_K), (FEO_P, FEO_C, FEO_C), (FEO_K, FEO_C, FEO_K), (FEO_C, FEO_K, FEO_C))
        feo_check(h, LayeredCylinder, FEO_CYL2, FEO_CYL2_REF, itfs, 3)
    end
    feo_check(h, LayeredCylinder, FEO_CYL2, FEO_CYL2_REF, (FEO_C, FEO_K, FEO_K), 3; external = false)
end
