using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra

# =============================================================================
#  test_sensitivities_alv.jl — ALV pipeline differentiates correctly via
#  `ForwardDiff` (set_param lens for fractions, closure-captured material
#  parameters for the kernel).  Each AD derivative is compared against a
#  central finite difference at `rtol ≤ 1e-7`.
# =============================================================================

const TIMES = collect(range(0.0, 2.0; length = 8))

function _build_law_M(k_M, μ_M, τ_K = 1.0, τ_μ = 0.5)
    function R_iso(t, tp)
        α = 3 * k_M * (1.0 + 4.0 * exp(-(t - tp) / τ_K))
        β = 2 * μ_M * (0.5 + 1.5 * exp(-(t - tp) / τ_μ))
        return TensISO{3}(α, β)
    end
    return ViscoLaw(R_iso, :relaxation)
end

const _C_INC = TensISO{3}(3 * 10.0, 2 * 4.0)

function _eff_mu_final(rve, scheme)
    R̃ = homogenize_alv(rve, scheme, :C; times = TIMES)
    _, β = iso_params_from_blocks(R̃)
    return β[end, end] / 2
end

function _build_rve_base(f::Real)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _build_law_M(1.0, 1.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(_C_INC));
        fraction = f
    )
    return rve
end

@testset "ALV sensitivities — d/df via set_param lens" begin
    rve = _build_rve_base(0.2)
    f₀ = 0.2

    function eff_mu_vs_f(f, scheme)
        rve_f = set_param(rve, AmountParameter(:I), f)
        return _eff_mu_final(rve_f, scheme)
    end

    for sch in (Voigt(), Reuss(), Dilute(), MoriTanaka(), Maxwell())
        dμ_AD = ForwardDiff.derivative(f -> eff_mu_vs_f(f, sch), f₀)
        h = 1.0e-5
        dμ_FD = (eff_mu_vs_f(f₀ + h, sch) - eff_mu_vs_f(f₀ - h, sch)) / (2h)
        @test isapprox(dμ_AD, dμ_FD; rtol = 1.0e-7)
    end
end

@testset "ALV sensitivities — d/dμ_M via closure" begin
    function eff_mu_vs_μM(μ_M)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _build_law_M(1.0, μ_M)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(_C_INC));
            fraction = 0.2
        )
        return _eff_mu_final(rve, MoriTanaka())
    end

    μM₀ = 1.0
    AD = ForwardDiff.derivative(eff_mu_vs_μM, μM₀)
    h = 1.0e-5
    FD = (eff_mu_vs_μM(μM₀ + h) - eff_mu_vs_μM(μM₀ - h)) / (2h)
    @test isapprox(AD, FD; rtol = 1.0e-6)
end

@testset "ALV sensitivities — gradient over (f, k_M, μ_M)" begin
    function eff_mu_vs_fkμ(p::AbstractVector)
        f, k_M, μ_M = p
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _build_law_M(k_M, μ_M)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(_C_INC));
            fraction = 0.2
        )
        rve_f = set_param(rve, AmountParameter(:I), f)
        return _eff_mu_final(rve_f, MoriTanaka())
    end

    p₀ = [0.2, 1.0, 1.0]
    ∇AD = ForwardDiff.gradient(eff_mu_vs_fkμ, p₀)
    h = 1.0e-5
    for i in 1:3
        eᵢ = [j == i ? 1.0 : 0.0 for j in 1:3]
        FD = (eff_mu_vs_fkμ(p₀ .+ h .* eᵢ) - eff_mu_vs_fkμ(p₀ .- h .* eᵢ)) / (2h)
        @test isapprox(∇AD[i], FD; rtol = 1.0e-6)
    end
end

@testset "ALV sensitivities — d/dτ_K (relaxation time inside kernel)" begin
    function eff_mu_vs_τK(τ_K)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _build_law_M(1.0, 1.0, τ_K, 0.5)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(_C_INC));
            fraction = 0.2
        )
        return _eff_mu_final(rve, MoriTanaka())
    end

    τK₀ = 1.0
    AD = ForwardDiff.derivative(eff_mu_vs_τK, τK₀)
    h = 1.0e-5
    FD = (eff_mu_vs_τK(τK₀ + h) - eff_mu_vs_τK(τK₀ - h)) / (2h)
    @test isapprox(AD, FD; rtol = 1.0e-6)
end

# =============================================================================
#  Phase-2 extension : AD through the iterative / extra ALV schemes
#  (SelfConsistent, AsymmetricSelfConsistent, PonteCastanedaWillis,
#  DifferentialScheme) and through a GEOMETRY parameter (aspect ratio) —
#  previously blocked by hard-coded `Matrix{Float64}` containers.
# =============================================================================

@testset "ALV sensitivities — d/df through SC / ASC / PCW / DIFF" begin
    rve = _build_rve_base(0.2)
    f₀ = 0.2

    function eff_mu_vs_f(f, scheme)
        rve_f = set_param(rve, AmountParameter(:I), f)
        return _eff_mu_final(rve_f, scheme)
    end

    for sch in (
            SelfConsistent(), AsymmetricSelfConsistent(),
            PonteCastanedaWillis(), DifferentialScheme(),
        )
        dμ_AD = ForwardDiff.derivative(f -> eff_mu_vs_f(f, sch), f₀)
        h = 1.0e-5
        dμ_FD = (eff_mu_vs_f(f₀ + h, sch) - eff_mu_vs_f(f₀ - h, sch)) / (2h)
        @test isapprox(dμ_AD, dμ_FD; rtol = 1.0e-5)
    end
end

@testset "ALV sensitivities — d/dω geometry (aspect ratio) MT + SC" begin
    function eff_mu_vs_ω(ω, scheme; symmetrize = :none)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _build_law_M(1.0, 1.0)); fraction = :rest)
        add_phase!(
            rve, :I, Spheroid(ω), Dict(:C => heaviside_law(_C_INC));
            fraction = 0.2, symmetrize = symmetrize
        )
        return _eff_mu_final(rve, scheme)
    end

    ω₀ = 3.0
    h = 1.0e-5

    # Mori-Tanaka evaluates the ALV Hill kernel against the fixed, isotropic
    # MATRIX, so an aligned spheroid needs no orientation average.
    AD = ForwardDiff.derivative(ω -> eff_mu_vs_ω(ω, MoriTanaka()), ω₀)
    FD = (eff_mu_vs_ω(ω₀ + h, MoriTanaka()) - eff_mu_vs_ω(ω₀ - h, MoriTanaka())) / (2h)
    @test isapprox(AD, FD; rtol = 1.0e-5)

    # `SelfConsistent` evaluates it against its RUNNING estimate, which an
    # aligned spheroid takes out of the isotropic class — the only class the
    # ALV Hill kernel exists for, and the one whose `(α, β)` the Picard loop
    # reads off that estimate.  Refused rather than silently answered.
    @test_throws ArgumentError eff_mu_vs_ω(ω₀, SelfConsistent())

    # With the orientation average it is legitimate, and differentiable.
    sc_iso = ω -> eff_mu_vs_ω(ω, SelfConsistent(); symmetrize = :iso)
    AD_sc = ForwardDiff.derivative(sc_iso, ω₀)
    FD_sc = (sc_iso(ω₀ + h) - sc_iso(ω₀ - h)) / (2h)
    @test isfinite(AD_sc)
    @test isapprox(AD_sc, FD_sc; rtol = 1.0e-5)
end

# =============================================================================
#  Order-2 (conduction / diffusion) ALV: every kernel ALLOCATES its
#  accumulators before adding the per-phase blocks, so the element type has to
#  span those blocks and not only `K̃_0` and the fractions.  Differentiating
#  with respect to a phase property or an inclusion geometry makes
#  `contribs` / `A_duts` `Dual` while the matrix law stays `Float64` — which
#  used to raise `MethodError: Float64(::Dual)` in EVERY order-2 scheme.
# =============================================================================

_fd_c(f, x, h) = (f(x + h) - f(x - h)) / (2h)

_law_K(k) = ViscoLaw((t, tp) -> TensISO{3}(k * (1.0 + exp(-(t - tp)))), :relaxation)

const _ALV2_SCHEMES = (
    Voigt(), Reuss(), Dilute(), DiluteDual(),
    MoriTanaka(), Maxwell(), DifferentialScheme(; nsteps = 20),
)

@testset "ALV order-2 sensitivities — d/dK_phase (every order-2 scheme)" begin
    function eff_K(k, sch)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => _law_K(1.0)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:K => heaviside_law(TensISO{3}(k)));
            fraction = 0.2
        )
        return homogenize_alv(rve, sch, :K; times = TIMES)[end, end]
    end

    for sch in _ALV2_SCHEMES
        AD = ForwardDiff.derivative(k -> eff_K(k, sch), 10.0)
        FD = _fd_c(k -> eff_K(k, sch), 10.0, 1.0e-5)
        @test isfinite(AD)
        @test isapprox(AD, FD; rtol = 1.0e-5, atol = 1.0e-9)
    end
end

# `DifferentialScheme` is included: the order-2 ALV Hill kernel exists for an
# isotropic reference only, and the differential scheme evaluates it against
# its RUNNING medium, so a non-spherical inclusion needs the orientation
# average to keep that medium isotropic — which the order-2 projector now
# actually performs (see `_maybe_symmetrize_alv2`).
@testset "ALV order-2 sensitivities — d/dω geometry (every order-2 scheme)" begin
    function eff_K_ω(ω, sch)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => _law_K(1.0)); fraction = :rest)
        add_phase!(
            rve, :I, Spheroid(ω), Dict(:K => heaviside_law(TensISO{3}(10.0)));
            fraction = 0.2, symmetrize = :iso
        )
        return homogenize_alv(rve, sch, :K; times = TIMES)[end, end]
    end

    for sch in _ALV2_SCHEMES
        AD = ForwardDiff.derivative(ω -> eff_K_ω(ω, sch), 3.0)
        FD = _fd_c(ω -> eff_K_ω(ω, sch), 3.0, 1.0e-5)
        @test isfinite(AD)
        @test isapprox(AD, FD; rtol = 1.0e-5, atol = 1.0e-9)
    end
end

# A `LayeredSphere`'s per-layer moduli live in a type parameter of their own
# (`LayeredSphere{T,N,Cs,Is}` — `T` is the eltype of the RADII), and the ALV
# layer kernels build `layers` with `ntuple`.  Differentiating with respect to
# ONE layer therefore yields a heterogeneous tuple, which an `NTuple{N,<:Tuple}`
# signature rejects outright.  The OUTER layer additionally checks the
# propagation: the state crosses every layer, so typing the accumulators off
# `layers[1]` alone is not enough.
@testset "ALV sensitivities — d/dC_layer of a LayeredSphere inclusion" begin
    function eff_mu_layer(x, sch, which::Symbol)
        moduli = which === :inner ?
            (TensISO{3}(3x, 2 * 5.0), TensISO{3}(3 * 20.0, 2 * 8.0)) :
            (TensISO{3}(3 * 30.0, 2 * 5.0), TensISO{3}(3x, 2 * 8.0))
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _build_law_M(1.0, 1.0)); fraction = :rest)
        add_phase!(
            rve, :I, LayeredSphere((0.6, 1.0), moduli),
            Dict(:C => heaviside_law(_C_INC)); fraction = 0.2
        )
        return _eff_mu_final(rve, sch)
    end

    for sch in (MoriTanaka(), DifferentialScheme(; nsteps = 20)),
            which in (:inner, :outer)

        AD = ForwardDiff.derivative(x -> eff_mu_layer(x, sch, which), 25.0)
        FD = _fd_c(x -> eff_mu_layer(x, sch, which), 25.0, 1.0e-4)
        @test isfinite(AD)
        @test isapprox(AD, FD; rtol = 1.0e-4, atol = 1.0e-10)
    end
end
