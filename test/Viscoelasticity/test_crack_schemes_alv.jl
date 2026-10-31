using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_crack_schemes_alv.jl — crack-aware ALV homogenization schemes:
#  Voigt / Reuss (cracks ignored), Dilute / DiluteDual / MT / Maxwell / PCW
#  (cracks add ΔC̃ = (4π/3) ε · property_contribution_alv to the
#  numerator), SC / ASC (cracks iterated against the running estimate).
# =============================================================================

const _to_mandel = (t -> TensND.mandel(TensND.canonical_components(t); symmetrize = true))

function _setup_crack_elastic(; k_M = 5.0, μ_M = 2.0, ε = 0.1, n_times = 4)
    times = collect(range(0.0, 1.0; length = n_times))
    C_M_t = TensISO{3}(3 * k_M, 2 * μ_M)
    return (;
        C_M_t, law_M = heaviside_law(C_M_t),
        crack = PennyCrack(1.0), ε, times,
    )
end

_build_alv(ctx) = let
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.law_M); fraction = :rest)
    add_phase!(
        rve, :CRACK, ctx.crack, Dict(:C => ctx.law_M);
        density = ctx.ε, symmetrize = :iso
    )
    rve
end

_build_el(ctx) = let
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.C_M_t); fraction = :rest)
    add_phase!(
        rve, :CRACK, ctx.crack, Dict(:C => ctx.C_M_t);
        density = ctx.ε, symmetrize = :iso
    )
    rve
end

@testset "Voigt / Reuss with cracks — match elastic (cracks ignored)" begin
    ctx = _setup_crack_elastic()
    n = length(ctx.times)
    for sch in (Voigt(), Reuss())
        ref = _to_mandel(homogenize(_build_el(ctx), sch, :C))
        R = homogenize_alv(_build_alv(ctx), sch, :C; times = ctx.times)
        for i in 1:n
            rows = (6 * (i - 1) + 1):(6 * i)
            @test isapprox(R[rows, rows], ref; atol = 1.0e-12)
        end
    end
end

@testset "Dilute / DiluteDual / MT / Maxwell / PCW with cracks — elastic limit" begin
    ctx = _setup_crack_elastic()
    n = length(ctx.times)
    for sch in (
            Dilute(), DiluteDual(), MoriTanaka(), Maxwell(),
            PonteCastanedaWillis(),
        )
        ref = _to_mandel(homogenize(_build_el(ctx), sch, :C))
        R = homogenize_alv(_build_alv(ctx), sch, :C; times = ctx.times)
        for i in 1:n
            rows = (6 * (i - 1) + 1):(6 * i)
            @test isapprox(R[rows, rows], ref; atol = 1.0e-10, rtol = 1.0e-10)
        end
    end
end

@testset "SC / ASC with cracks — Bristow-Budiansky-O'Connell consistency" begin
    # Three sanity checks for self-consistent ALV with cracks.
    ctx = _setup_crack_elastic()
    rve = _build_alv(ctx)

    # 1) At very low density, SC ≈ Dilute (perturbative).
    times = ctx.times
    rve_low = let
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.law_M); fraction = :rest)
        add_phase!(
            r, :CRACK, ctx.crack, Dict(:C => ctx.law_M);
            density = 0.001, symmetrize = :iso
        )
        r
    end
    R_dil_low = homogenize_alv(rve_low, Dilute(), :C; times = times)
    R_sc_low = homogenize_alv(
        rve_low, SelfConsistent(), :C; times = times,
        abstol = 1.0e-13, reltol = 1.0e-12, maxiters = 500
    )
    @test isapprox(R_dil_low[1:6, 1:6], R_sc_low[1:6, 1:6]; atol = 1.0e-3)

    # 2) SC stays stiffer than Dilute at moderate density (Dilute over-softens).
    R_dil = homogenize_alv(rve, Dilute(), :C; times = times)
    R_sc = homogenize_alv(
        rve, SelfConsistent(), :C; times = times,
        abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 500
    )
    α_dil = R_dil[1, 1] + 2 * R_dil[1, 2]
    α_sc = R_sc[1, 1] + 2 * R_sc[1, 2]
    @test α_sc > α_dil   # SC stiffer (less crack softening) than Dilute

    # 3) SC stays stiffer than ASC at moderate density.  The ECHOES
    # symmetric SC fixed point (with `strain_Stress_α = A_α·S_n` for
    # solids and `strain_Stress_c = H_c` for cracks) differs from the
    # ASC compliance-form fixed point — SC is on the matrix-stiff
    # branch and ASC on the matrix-distinguished compliance branch.
    R_asc = homogenize_alv(
        rve, AsymmetricSelfConsistent(), :C;
        times = times, abstol = 1.0e-12, maxiters = 500
    )
    α_sc_block = R_sc[1, 1] + 2 * R_sc[1, 2]
    α_asc_block = R_asc[1, 1] + 2 * R_asc[1, 2]
    @test α_sc_block ≥ α_asc_block
end

@testset "Crack stiffness contribution helper" begin
    ctx = _setup_crack_elastic()
    crack = ctx.crack
    times = ctx.times
    C_M = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation(ctx.law_M, times, 6)
    Ñ = MeanFieldHomogenization.Viscoelasticity.property_contribution_alv_at(crack, C_M)
    H̃ = inverse_property_contribution_alv(crack, ctx.law_M, times)
    # Ñ = -C̃·H̃·C̃ — round-trip identity at machine precision.
    @test isapprox(Ñ, -(C_M * H̃ * C_M); atol = 1.0e-12)
end

@testset "ALV penny crack — the flat limit of a void spheroid, aging matrix" begin
    # A penny crack of density ε is the limit of void spheroids of aspect ω at
    # volume fraction (4π/3) ω ε. The Hill kernel of the spheroid combines the
    # inverses of k + 4μ/3 and μ alone, so the limit fixes the order of the
    # Volterra factors of the COD tensor, which an aging matrix makes matter.
    V = MeanFieldHomogenization.Viscoelasticity
    times = [0.0, 0.3, 0.7, 1.2, 2.0]
    matrix = ViscoLaw(
        (t, tp) -> t >= tp ?
            TensISO{3}(6.0 * (1 + 0.5tp) * exp(-(t - tp) / 0.8), 2.0 * (1 + tp) * exp(-(t - tp) / 1.5)) :
            TensISO{3}(0.0, 0.0)
    )
    void = ViscoLaw((t, tp) -> TensISO{3}(0.0, 0.0))
    ε, ω = 0.1, 1.0e-5
    cracked = RVE()
    add_phase!(cracked, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(cracked, :F, PennyCrack(1.0), Dict(:C => matrix); density = ε, symmetrize = :iso)
    porous = RVE()
    add_phase!(porous, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(porous, :V, Spheroid(ω), Dict(:C => void); fraction = 4π / 3 * ω * ε, symmetrize = :iso)
    for (a, b) in zip(
            V.iso_params_from_blocks(homogenize_alv(cracked, Dilute(:M), :C; times)),
            V.iso_params_from_blocks(homogenize_alv(porous, Dilute(:M), :C; times)),
        )
        @test a ≈ b rtol = 1.0e-4
    end
end

@testset "ALV penny cracks in an aging matrix, self-consistent — Echoes reference" begin
    # The running estimate takes the COD tensor of `_penny_cod_alv` too, in the
    # Budiansky–O'Connell branch of the self-consistent scheme. Reference: Echoes
    # `homogenize_visco(…, scheme = SC)` with a crack of aspect 1e-7, whence the
    # tolerance.
    V = MeanFieldHomogenization.Viscoelasticity
    times = [0.0, 0.4, 1.5]
    matrix = ViscoLaw(
        (t, tp) -> t >= tp ?
            TensISO{3}(6.0 * (1 + 0.5tp) * exp(-(t - tp) / 0.8), 2.0 * (1 + tp) * exp(-(t - tp) / 1.5)) :
            TensISO{3}(0.0, 0.0)
    )
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => matrix); density = 0.1, symmetrize = :iso)
    α, β = V.iso_params_from_blocks(homogenize_alv(rve, SelfConsistent(), :C; times))
    lower(rows) = [i >= j ? rows[i][j] : 0.0 for i in 1:3, j in 1:3]
    @test α ≈ lower([[4.272689333671794], [-1.2841395313625679, 4.003516637323179], [-0.36990479296289147, -3.601174676535309, 4.755441449535003]]) rtol = 1.0e-6
    @test β ≈ lower([[1.7297859422284707], [-0.5474610920054662, 1.864948809282462], [-0.2589671145064161, -1.8299722030102268, 2.712448059696108]]) rtol = 1.0e-6
end

@testset "DiluteDual ALV — inclusions and cracks together, elastic limit" begin
    # The dual scheme adds the compliance contributions of the solid phases and
    # of the cracks: in the elastic limit each diagonal block is the elastic
    # DiluteDual stiffness, the off-diagonal ones vanish.
    mandel = (t -> TensND.mandel(TensND.canonical_components(t); symmetrize = true))
    C_M, C_I = TensISO{3}(3 * 10.0, 2 * 4.0), TensISO{3}(3 * 30.0, 2 * 12.0)
    times = [0.0, 0.5, 1.0]
    function rve_with(law)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law(C_M)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => law(C_I)); fraction = 0.1)
        add_phase!(rve, :CRACK, PennyCrack(1.0), Dict(:C => law(C_M)); density = 0.05, symmetrize = :iso)
        return rve
    end
    M_ref = mandel(homogenize(rve_with(identity), DiluteDual(), :C))
    C_alv = homogenize_alv(rve_with(heaviside_law), DiluteDual(), :C; times = times)
    for i in eachindex(times)
        rows = (6 * (i - 1) + 1):(6 * i)
        @test C_alv[rows, rows] ≈ M_ref rtol = 1.0e-9
        for j in 1:(i - 1)
            @test maximum(abs, C_alv[rows, (6 * (j - 1) + 1):(6 * j)]) <= 1.0e-9 * maximum(abs, M_ref)
        end
    end
end

@testset "has_visco_property — a phase without the property is passed over" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => heaviside_law(TensISO{3}(30.0, 8.0))); fraction = :rest)
    add_phase!(rve, :P, Ellipsoid(1.0), Dict(:K => TensISO{3}(1.0)); fraction = 0.1)
    @test has_visco_property(rve, :C)
    @test !has_visco_property(rve, :K)
end

@testset "Mori-Tanaka with cracks — differentiated by the crack density" begin
    # The crack term was accumulated into a buffer of the matrix's element
    # type, which a dual density could not enter. In the elastic limit the
    # derivative is the one of the elastic scheme.
    ctx = _setup_crack_elastic()
    alv(ε) = homogenize_alv(_build_alv(merge(ctx, (; ε))), MoriTanaka(), :C; times = ctx.times)[1:6, 1:6]
    el(ε) = _to_mandel(homogenize(_build_el(merge(ctx, (; ε))), MoriTanaka(), :C))
    @test ForwardDiff.derivative(alv, 0.1) ≈ ForwardDiff.derivative(el, 0.1) rtol = 1.0e-10
end

@testset "SC and ASC with cracks — the class route and the full-matrix route agree" begin
    # A one-layer sphere is the spherical inclusion, but it sends the scheme down
    # the full-matrix route, while the ellipsoid keeps it in the classes of the
    # operators: two independent computations of the same aging cracked cell,
    # with springs on the crack faces.
    times = collect(range(0.0, 3.0; length = 6))
    matrix = maxwell_iso(10.0, 6.0, 0.8, 2.5)
    inclusion = heaviside_law(TensISO{3}(3 * 30.0, 2 * 20.0))
    function cell(shape)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => matrix); fraction = :rest)
        add_phase!(r, :I, shape, Dict(:C => inclusion); fraction = 0.2)
        add_phase!(
            r, :CRACK, PennyCrack(1.0),
            Dict(:C => matrix, :Rn => ViscoLaw((t, tp) -> 3.0 * exp(-(t - tp) / 4)), :Rt => heaviside_law(1.5));
            density = 0.15, symmetrize = :iso
        )
        return r
    end
    for scheme in (SelfConsistent(reltol = 1.0e-13), AsymmetricSelfConsistent(reltol = 1.0e-13))
        in_classes = homogenize_alv(cell(Ellipsoid(1.0)), scheme, :C; times)
        dense = homogenize_alv(cell(LayeredSphere((1.0,), (inclusion,))), scheme, :C; times)
        @test in_classes.class isa TensISO
        @test Matrix(in_classes) ≈ Matrix(dense) rtol = 1.0e-11
    end
end

@testset "ALV elliptic crack of any orientation — the flat limit of a void ellipsoid, aging matrix" begin
    # An elliptic crack of semi-axes a ≥ b and density ε = N a b² is the limit of void
    # ellipsoids of semi-axes (a, b, ω b) at volume fraction (4π/3) ω ε, whose Hill
    # kernel, orthotropic in their frame, is computed independently of the closed
    # form of the crack: the limit checks the elliptic functions, the order of the
    # Volterra factors and the rotation together.
    times = [0.0, 0.3, 0.7, 1.2, 2.0]
    matrix = ViscoLaw(
        (t, tp) -> t >= tp ?
            TensISO{3}(6.0 * (1 + 0.5tp) * exp(-(t - tp) / 0.8), 2.0 * (1 + tp) * exp(-(t - tp) / 1.5)) :
            TensISO{3}(0.0, 0.0)
    )
    void = ViscoLaw((t, tp) -> TensISO{3}(0.0, 0.0))
    angles = (0.4, 0.9, 0.2)
    ε, ω = 0.1, 1.0e-6
    cracked = RVE()
    add_phase!(cracked, :M, Ellipsoid(1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(cracked, :F, EllipticCrack(1.0, 0.4; euler_angles = angles), Dict(:C => matrix); density = ε)
    porous = RVE()
    add_phase!(porous, :M, Ellipsoid(1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(porous, :V, Ellipsoid(1.0, 0.4, 0.4ω; euler_angles = angles), Dict(:C => void); fraction = 4π / 3 * ω * ε)
    C_crack = homogenize_alv(cracked, Dilute(:M), :C; times)
    @test C_crack.class isa TensOrtho
    @test Matrix(C_crack) ≈ Matrix(homogenize_alv(porous, Dilute(:M), :C; times)) rtol = 1.0e-5
    # A ribbon has no ALV opening, and an anisotropic matrix none either: both refused.
    @test_throws "implemented for elliptic cracks" inverse_property_contribution_alv(RibbonCrack(1.0), matrix, times)
    ti = heaviside_law(TensTI{4}(4.0, 6.0, 1.0, 1.5, 2.0, (0.0, 0.0, 1.0)))
    @test_throws "isotropic ALV matrix only" inverse_property_contribution_alv(PennyCrack(1.0), ti, times)
end

@testset "ALV elliptic cracks, iso-averaged, self-consistent — Echoes reference" begin
    # Echoes `homogenize_visco(…, scheme = SC)`, a crack of semi-axes (1, 0.4, 1e-12)
    # and Echoes density ε / η² (its density is N a³), averaged in orientation; the
    # isotropic parameters of the result on three instants.
    V = MeanFieldHomogenization.Viscoelasticity
    times = [0.0, 0.4, 1.5]
    g(tp) = 1 - 0.5 * exp(-tp)
    matrix = ViscoLaw((t, tp) -> g(tp) * TensISO{3}(3 * (0.2 + 2.0 * exp(-(t - tp) / 0.8)), 2 * (0.1 + exp(-(t - tp) / 1.5))))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(rve, :F, EllipticCrack(1.0, 0.4), Dict(:C => matrix); density = 0.1, symmetrize = :iso)
    α, β = V.iso_params_from_blocks(homogenize_alv(rve, SelfConsistent(reltol = 1.0e-13), :C; times))
    lower(rows) = [i >= j ? rows[i][j] : 0.0 for i in 1:3, j in 1:3]
    @test α ≈ lower([[2.1060270141195234], [-0.6843457431162296, 2.1159961600659187], [-0.23255240795355414, -1.5820142668242856, 2.3926740244309066]]) rtol = 1.0e-10
    @test β ≈ lower([[0.899289293076641], [-0.24595200073836723, 0.9498149450804234], [-0.12328865352225456, -0.6108357712745645, 1.1103729044170134]]) rtol = 1.0e-10
end

@testset "ALV elliptic crack — derivatives in its aspect ratio and orientation, elastic limit" begin
    # Under a constant kernel the dilute ALV estimate is the elastic one at every
    # instant: its derivatives in the aspect ratio and in an angle of the crack are
    # those of the elastic closed form, an independent code.
    C = TensISO{3}(3 * 5.0, 2 * 2.0)
    function estimates(η, θ)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => heaviside_law(C)); fraction = :rest)
        add_phase!(r, :F, EllipticCrack(1.0, η; euler_angles = (θ, 0.9, 0.2)), Dict(:C => heaviside_law(C)); density = 0.1)
        e = RVE()
        add_phase!(e, :M, Ellipsoid(1.0), Dict(:C => C); fraction = :rest)
        add_phase!(e, :F, EllipticCrack(1.0, η; euler_angles = (θ, 0.9, 0.2)), Dict(:C => C); density = 0.1)
        return Matrix(homogenize_alv(r, Dilute(:M), :C; times = [0.0, 1.0]))[7:12, 7:12], _to_mandel(homogenize(e, Dilute(:M), :C))
    end
    for (f, x) in ((η -> estimates(η, 0.4), 0.4), (θ -> estimates(0.4, θ), 0.4))
        d_alv = ForwardDiff.derivative(x -> first(f(x)), x)
        d_el = ForwardDiff.derivative(x -> last(f(x)), x)
        @test d_alv ≈ d_el rtol = 1.0e-10
        @test norm(d_el) > 0
    end
end

@testset "SC with cracks of one orientation — the anisotropic estimate is refused" begin
    # The self-consistent estimate must stay isotropic for the ALV Hill kernel against
    # it: a family of parallel cracks makes it transversely isotropic, which is refused
    # with the class it does not belong to, instead of being projected silently.
    times = [0.0, 0.5, 1.0]
    matrix = maxwell_iso(10.0, 6.0, 0.8, 2.5)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(r, :F, PennyCrack(1.0), Dict(:C => matrix); density = 0.1)
    @test_throws ArgumentError homogenize_alv(r, SelfConsistent(), :C; times)
end
