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
feo_trap(R::AbstractMatrix, b) = R                     # a reference given by its operator
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

# A cylinder reference without major symmetry: in every block, the Mandel slots (3, 1) and
# (3, 2) (σ₃₃ per in-plane strain) are 1.3 times the slots (1, 3), (2, 3). Each block stays
# transversely isotropic about e₃.
function feo_nonsym(R::AbstractMatrix)
    S = copy(R)
    for i in 1:(size(S, 1) ÷ 6), j in 1:(size(S, 2) ÷ 6)
        S[6(i - 1) + 3, 6(j - 1) .+ (1:2)] .= 1.3 .* R[6(i - 1) .+ (1:2), 6(j - 1) + 3]
    end
    return S
end

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
        flux = flux_gradient_concentration_alv(incl, ref, FEO_TIMES; external),
        flux_layer = [flux_gradient_concentration_alv(incl, ref, FEO_TIMES; layer = k) for k in 1:nl],
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
            :flux => FEO.compare(pkg.flux, r.sig, h),
            :flux_layer => maximum(FEO.compare(pkg.flux_layer[k], r.sig_layer[k], h) for k in 1:nl),
        )
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

@testset "FE oracle — LayeredCylinder, order 4, reference without major symmetry" begin
    h = FEO.cylinder_elastic()
    R0 = feo_nonsym(feo_trap(FEO_CYL4_REF, 6))
    @test norm(R0 - feo_trap(FEO_CYL4_REF, 6)) > 0.1
    for itfs in ((FEO_P, FEO_P, FEO_P), (FEO_S, FEO_M, FEO_S), (FEO_M, FEO_S, FEO_M))
        feo_check(h, LayeredCylinder, FEO_CYL4, R0, itfs, 6)
    end
    feo_check(h, LayeredCylinder, FEO_CYL4, R0, (FEO_S, FEO_S, FEO_M), 6; external = false)
    # The asymmetry changes σ₃₃ only, so N and not A: the oracle of the symmetric
    # reference is 10⁻² away from the package's N of the other one.
    c = LayeredCylinder(FEO_RADII, FEO_CYL4; interfaces = map(feo_pkg, (FEO_S, FEO_M, FEO_S)))
    sym = FEO.oracle_solve(
        h, collect(FEO_RADII), [feo_trap(l, 6) for l in FEO_CYL4], [FEO_S, FEO_M, FEO_S], feo_trap(FEO_CYL4_REF, 6); nel = 20,
    )
    @test FEO.compare(property_contribution_alv(c, R0, FEO_TIMES), sym.N, h) > 1.0e-3
end

# ── Pointwise operators ──────────────────────────────────────────────────────

# The points: inside each layer, on both sides of each interface, in the matrix, as
# (radius, region, side); the direction is generic.
const FEO_POINTS = [
    (0.31, 1, :outer), (0.67, 2, :outer), (0.93, 3, :outer), (1.4, 4, :outer), (3.0, 4, :outer),
    (0.5, 1, :inner), (0.5, 2, :outer), (0.8, 2, :inner), (0.8, 3, :outer), (1.0, 3, :inner), (1.0, 4, :outer),
]
const FEO_ANGLES = (0.7, 0.4)       # (θ, φ) on the sphere, (θ, z) on the cylinder

feo_point(::LayeredSphere, r, (θ, φ)) = r .* [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]
feo_point(::LayeredCylinder, r, (θ, z)) = [r * cos(θ), r * sin(θ), z]
feo_fields(s::LayeredSphere, ref) = LayeredSphereALVFields(s, ref, FEO_TIMES)
feo_fields(c::LayeredCylinder, ref) = LayeredCylinderALVFields(c, ref, FEO_TIMES)
# The value at the point: the displacement (temperature) of the sphere; the cylinder has none.
feo_value(F::LayeredSphereALVFields, h, args...; side) =
    h.elastic ? local_displacement_alv(F, args...; side) : local_temperature_alv(F, args...; side)
feo_value(::LayeredCylinderALVFields, h, args...; side) = nothing

# The normal component of a flux: the traction σ·n, or q·n.
feo_normal(v, n, elastic) = elastic ? FEO.mandel_to_matrix(v) * n : [dot(v, n)]

# Jump of the normal flux across r_k: the package's operators on both sides against the
# oracle's fields on both sides, per load, instant and loading instant.
function feo_jump_error(Pin, Pout, fin, fout, h, sign)
    nt = length(FEO_TIMES); b = h.b; n = fin.normal
    err = 0.0; sc = 0.0
    for ℓ in eachindex(h.loads), i in 1:nt, J in 1:nt
        rows, cols = ((i - 1) * b + 1):(i * b), ((J - 1) * b + 1):(J * b)
        E = h.loads[ℓ].E
        jp = feo_normal(Pout[rows, cols] * E, n, h.elastic) - feo_normal(Pin[rows, cols] * E, n, h.elastic)
        jo = sign .* (feo_normal(fout.flux[ℓ, i, J], n, h.elastic) - feo_normal(fin.flux[ℓ, i, J], n, h.elastic))
        err = max(err, maximum(abs, jp - jo)); sc = max(sc, maximum(abs, jp))
    end
    return err / sc, sc
end

# The pointwise gradient, flux and value at every point, and the jump of the normal flux
# across every membrane (conducting surface), on two meshes of degree p: the derivative
# of the amplitudes, which the pointwise fields need, is not superconvergent (rate about 5
# at p = 6). The values of the sphere reach the rounding errors on the coarse mesh already.
function feo_check_local(h, ctor, laws, ref, itfs, b; p = 6, nels = (10, 20), tol = 5.0e-7, ratio = 8)
    incl = ctor(FEO_RADII, laws; interfaces = map(feo_pkg, itfs))
    F = feo_fields(incl, ref)
    sign = h.elastic ? 1 : -1                       # the pointwise flux of conduction is q = -k·∇T
    pkg = map(FEO_POINTS) do (r, _, side)
        args = (r, FEO_ANGLES...)
        (
            grad = local_gradient_gradient_concentration_alv(F, args...; side),
            flux = local_flux_gradient_concentration_alv(F, args...; side),
            value = feo_value(F, h, args...; side),
        )
    end
    errs = map(nels) do nel
        res = FEO.oracle_solve(h, collect(FEO_RADII), [feo_trap(l, b) for l in laws], collect(itfs), feo_trap(ref, b); p, nel)
        pf = [FEO.point_fields(res, feo_point(incl, r, FEO_ANGLES), region) for (r, region, _) in FEO_POINTS]
        e = Dict(
            :grad => maximum(FEO.compare(pkg[k].grad, pf[k].grad, h) for k in eachindex(pf)),
            :flux => maximum(FEO.compare(pkg[k].flux, pf[k].flux, h; sign) for k in eachindex(pf)),
        )
        pkg[1].value === nothing ||
            (e[:value] = maximum(FEO.compare(pkg[k].value, pf[k].u, h; rb = h.elastic ? 3 : 1) for k in eachindex(pf)))
        for (k, itf) in enumerate(itfs)
            itf.kind in (:membrane, :surface) || continue
            kin = findfirst(P -> P[1] == FEO_RADII[k] && P[3] === :inner, FEO_POINTS)
            kout = findfirst(P -> P[1] == FEO_RADII[k] && P[3] === :outer, FEO_POINTS)
            ej, sc = feo_jump_error(pkg[kin].flux, pkg[kout].flux, pf[kin], pf[kout], h, sign)
            @test sc > 1.0e-3                       # the jump is not zero
            e[Symbol(:jump, k)] = ej
        end
        e
    end
    for key in keys(errs[2])
        @test errs[2][key] < tol
        @test errs[1][key] > ratio * errs[2][key] || errs[1][key] < 1.0e-10
    end
    return errs
end

@testset "FE oracle — pointwise operators" begin
    feo_check_local(FEO.sphere_elastic(), LayeredSphere, FEO_SPH4, FEO_SPH4_REF, (FEO_S, FEO_M, FEO_M), 6)
    feo_check_local(FEO.sphere_conduction(), LayeredSphere, FEO_SPH2, FEO_SPH2_REF, (FEO_K, FEO_C, FEO_C), 3)
    feo_check_local(FEO.cylinder_elastic(), LayeredCylinder, FEO_CYL4, FEO_CYL4_REF, (FEO_S, FEO_M, FEO_M), 6)
    feo_check_local(FEO.cylinder_conduction(), LayeredCylinder, FEO_CYL2, FEO_CYL2_REF, (FEO_K, FEO_C, FEO_C), 3)
    feo_check_local(
        FEO.cylinder_elastic(), LayeredCylinder, FEO_CYL4, feo_nonsym(feo_trap(FEO_CYL4_REF, 6)), (FEO_M, FEO_S, FEO_M), 6,
    )
end

# ── Derivatives ──────────────────────────────────────────────────────────────

# The oracle differentiated by a fourth-order central difference, on a mesh whose element
# counts do not change with the parameter.
function feo_fd(f, x, δ)
    a, b, c, d = f(x + 2δ), f(x + δ), f(x - δ), f(x - 2δ)
    return map((a, b, c, d) -> (-a + 8b - 8c + d) / (12δ), a, b, c, d)
end

@testset "FE oracle — ForwardDiff through the cylinder" begin
    h = FEO.cylinder_elastic()
    laws = [feo_trap(l, 6) for l in FEO_CYL4]; R0 = feo_trap(FEO_CYL4_REF, 6)
    pkg_itfs(kn) = (SpringInterface(20.0, 8.0), MembraneInterface(0.6, 0.25), SpringInterface(kn, oftype(kn, 8.0)))
    oracle_itfs(kn) = [FEO_S, FEO_M, FEO.OracleInterface(:spring, kn, 8.0)]
    cyl(r₁, kn) = LayeredCylinder((r₁, 0.8, 1.0), FEO_CYL4; interfaces = pkg_itfs(kn))
    # d/dk_n of the outer spring, and d/dr₁ of the core radius
    D_kn = ForwardDiff.derivative(kn -> property_contribution_alv(cyl(0.5, kn), FEO_CYL4_REF, FEO_TIMES), 20.0)
    D_r = ForwardDiff.derivative(r₁ -> property_contribution_alv(cyl(r₁, 20.0), FEO_CYL4_REF, FEO_TIMES), 0.5)
    DA_r = ForwardDiff.derivative(r₁ -> gradient_gradient_concentration_alv(cyl(r₁, 20.0), FEO_CYL4_REF, FEO_TIMES), 0.5)
    errs = map((1, 2)) do s
        nel_layers = s .* [5, 3, 2]
        sol(r₁, kn) = FEO.oracle_solve(h, [r₁, 0.8, 1.0], laws, oracle_itfs(kn), R0; nel_layers)
        (
            FEO.compare(D_kn, feo_fd(kn -> sol(0.5, kn).N, 20.0, 0.05), h),
            FEO.compare(D_r, feo_fd(r₁ -> sol(r₁, 20.0).N, 0.5, 1.0e-3), h),
            FEO.compare(DA_r, feo_fd(r₁ -> sol(r₁, 20.0).eps, 0.5, 1.0e-3), h),
        )
    end
    for q in 1:3
        @test errs[2][q] < 1.0e-6
        @test errs[1][q] > 8 * errs[2][q]
    end
end
