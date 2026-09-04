# =============================================================================
#  test_sensitivities.jl — cross-check of finite differences against autodiff
#  on every scheme.
#
#  For each shipped scheme (Voigt, Reuss, Dilute, DiluteDual, MoriTanaka,
#  Maxwell, PCW, SC, ASC, Differential) and each applicable lens type, checks
#  that:
#   * derivative(rve, scheme, p; indexer)   agrees with a finite difference
#                                           of step h,
#   * gradient(rve, scheme, ps; indexer)    agrees with [derivative]_i,
#   * jacobian(rve, scheme, ps)             has the right shape.
#
#  The tolerance is loose for the iterative schemes: a fixed point solved to
#  abstol ≈ 1e-10 leaves a residual error of order abstol/h.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

const RTOL_CLOSED = 1.0e-6
const RTOL_ITER = 1.0e-4

# Centered finite-difference reference
_fd_centered(f, x, h) = (f(x + h) - f(x - h)) / (2h)

# Reference two-phase spherical RVE
function _ref_rve(; f = 0.25)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(60.0, 20.0)); fraction = f
    )
    return rve
end

# Canonical indexer: C[1,1,1,1]
const idxC = C -> get_array(C)[1, 1, 1, 1]

# Closed-form schemes (tight rtol)
const _CLOSED_SCHEMES = (
    Voigt(), Reuss(), Dilute(), DiluteDual(),
    MoriTanaka(), Maxwell(), PonteCastanedaWillis(),
)
# Iterative / differential schemes (loose rtol)
const _ITER_SCHEMES = (
    SelfConsistent(; abstol = 1.0e-12, maxiters = 200),
    AsymmetricSelfConsistent(; abstol = 1.0e-12, maxiters = 200),
    DifferentialScheme(; nsteps = 50),
)

@testset "derivative vs FD — AmountParameter (closed-form schemes)" begin
    for sch in _CLOSED_SCHEMES
        rve = _ref_rve(f = 0.25)
        p = amount(:I)
        ∂_ad = derivative(rve, sch, p; indexer = idxC)
        # FD reference: rebuild the RVE for each h
        f_eval = h -> begin
            rve2 = _ref_rve(f = 0.25 + h)
            return idxC(homogenize(rve2, sch))
        end
        ∂_fd = _fd_centered(f_eval, 0.0, 1.0e-6)
        @test isfinite(∂_ad)
        @test isapprox(∂_ad, ∂_fd; rtol = RTOL_CLOSED, atol = 1.0e-7)
    end
end

@testset "derivative vs FD — AmountParameter (iterative schemes)" begin
    for sch in _ITER_SCHEMES
        rve = _ref_rve(f = 0.25)
        p = amount(:I)
        ∂_ad = derivative(rve, sch, p; indexer = idxC)
        f_eval = h -> begin
            rve2 = _ref_rve(f = 0.25 + h)
            return idxC(homogenize(rve2, sch))
        end
        ∂_fd = _fd_centered(f_eval, 0.0, 1.0e-4)
        @test isfinite(∂_ad)
        @test isapprox(∂_ad, ∂_fd; rtol = RTOL_ITER, atol = 1.0e-5)
    end
end

@testset "derivative vs FD — PropertyParameter (closed-form)" begin
    for sch in _CLOSED_SCHEMES
        rve = _ref_rve()
        p = property(:I, :C, :bulk)
        ∂_ad = derivative(rve, sch, p; indexer = idxC)
        K0 = get_param(rve, p)
        f_eval = h -> begin
            rve2 = _ref_rve()
            rve2 = set_param(rve2, p, K0 + h)
            return idxC(homogenize(rve2, sch))
        end
        ∂_fd = _fd_centered(f_eval, 0.0, 1.0e-3)
        @test isfinite(∂_ad)
        @test isapprox(∂_ad, ∂_fd; rtol = RTOL_CLOSED, atol = 1.0e-7)
    end
end

# Same lens on the iterative/differential schemes.  An INCLUSION property is
# invisible to the differential scheme's initial state (only the matrix
# property and the amounts are read there) and reaches its ODE through the
# right-hand side alone — see `T_contrib` in `Schemes/differential.jl`.
@testset "derivative vs FD — PropertyParameter (iterative schemes)" begin
    for sch in _ITER_SCHEMES
        rve = _ref_rve()
        p = property(:I, :C, :bulk)
        ∂_ad = derivative(rve, sch, p; indexer = idxC)
        K0 = get_param(rve, p)
        f_eval = h -> begin
            rve2 = set_param(_ref_rve(), p, K0 + h)
            return idxC(homogenize(rve2, sch))
        end
        ∂_fd = _fd_centered(f_eval, 0.0, 1.0e-3)
        @test isfinite(∂_ad)
        @test isapprox(∂_ad, ∂_fd; rtol = RTOL_ITER, atol = 1.0e-7)
    end
end

@testset "derivative — Christensen 1990 closed form for ∂k_MT/∂f" begin
    # k_MT(f) = k_m + f·Δk·ζm / [ζm + (1-f)·Δk]   (Christensen 1990)
    # ⇒ ∂k_MT/∂f = Δk·ζm·(ζm + Δk) / [ζm + (1-f)·Δk]²
    k_m, μ_m = 10.0, 5.0
    k_i, μ_i = 40.0, 20.0
    f = 0.25
    ζm = k_m + 4 * μ_m / 3
    Δk = k_i - k_m
    D = ζm + (1 - f) * Δk
    ∂k_∂f_ref = Δk * ζm * (ζm + Δk) / D^2

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k_m, 2μ_m)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k_i, 2μ_i));
        fraction = f
    )

    # Bulk modulus extracted from the trace of the spherical projection :
    # 3K = (1/3) Σ_i Σ_j C[i,i,j,j]  for any 4th-order isotropic stiffness
    bulk_from_tens = C -> begin
        a = get_array(C)
        s = zero(eltype(a))
        for i in 1:3, j in 1:3
            s += a[i, i, j, j]
        end
        return s / 9
    end
    ∂k_∂f_ad = derivative(rve, MoriTanaka(), amount(:I); indexer = bulk_from_tens)
    @test isapprox(∂k_∂f_ad, ∂k_∂f_ref; rtol = 1.0e-10)
end

@testset "gradient = vector of derivatives" begin
    rve = _ref_rve()
    ps = [amount(:I), property(:I, :C, :bulk), property(:M, :C, :shear)]
    ∇ = gradient(rve, MoriTanaka(), ps; indexer = idxC)
    @test length(∇) == 3
    for (i, p) in enumerate(ps)
        ∂ = derivative(rve, MoriTanaka(), p; indexer = idxC)
        @test isapprox(∇[i], ∂; rtol = 1.0e-10)
    end
end

@testset "jacobian — shape & consistency with gradient" begin
    rve = _ref_rve()
    ps = [amount(:I), property(:I, :C, :bulk)]
    J = jacobian(rve, MoriTanaka(), ps)
    @test size(J) == (81, 2)             # 3^4 components × 2 params

    # Index (1,1,1,1) corresponds to flat index 1
    ∂_C1111_∂f = derivative(rve, MoriTanaka(), ps[1]; indexer = idxC)
    @test isapprox(J[1, 1], ∂_C1111_∂f; rtol = 1.0e-10)

    ∂_C1111_∂K = derivative(rve, MoriTanaka(), ps[2]; indexer = idxC)
    @test isapprox(J[1, 2], ∂_C1111_∂K; rtol = 1.0e-10)
end

@testset "single-parameter jacobian convenience" begin
    rve = _ref_rve()
    J = jacobian(rve, MoriTanaka(), amount(:I))
    @test size(J) == (81, 1)
end

@testset "sensitivity — closure fallback (auto kind)" begin
    # Scalar in / scalar out → derivative
    f1 = x -> x^3
    @test sensitivity(f1, 2.0) ≈ 12.0

    # Vector in / scalar out → gradient
    f2 = xs -> xs[1]^2 + 3 * xs[2]
    g = sensitivity(f2, [1.5, 0.5])
    @test g ≈ [3.0, 3.0]

    # Vector in / vector out → jacobian
    f3 = xs -> [xs[1] * xs[2], xs[1] + xs[2]]
    J = sensitivity(f3, [2.0, 3.0])
    @test J ≈ [3.0 2.0; 1.0 1.0]
end

@testset "sensitivity — multi-scale closure (anticipates Pichler script)" begin
    # 2 nested MT calls: inner ≡ a sub-RVE inside the matrix of an outer RVE
    function build_mortar(K_inner)
        # Inner RVE
        rve1 = RVE()
        add_phase!(rve1, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
        add_phase!(
            rve1, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(K_inner, 20.0));
            fraction = 0.3
        )
        return idxC(homogenize(rve1, MoriTanaka()))
    end
    # Single-pass autodiff through the nested call
    df = ForwardDiff.derivative(build_mortar, 60.0)
    @test isfinite(df)
    df_fd = _fd_centered(build_mortar, 60.0, 1.0e-3)
    @test isapprox(df, df_fd; rtol = 1.0e-6)
end

# =============================================================================
#  User-defined inclusion — showing that extensibility is automatic
# =============================================================================
#
#  A minimal user-defined inclusion type `Blob` is declared here, reusing the
#  `Sphere` kernel for its `hill_tensor`. Sensitivity with respect to the
#  `radius` field must work without any change to `parameters.jl` — only the
#  reflection through `_replace_geom_field` is needed.

# Extending `MeanFieldHomogenization.Schemes._replace_geom_field` is NOT needed
# here, because `Blob` has an auto-generated parametric constructor compatible
# with the @generated fallback.
struct Blob{T <: Number, B} <: MeanFieldHomogenization.AbstractEllipsoidalInclusion{3, T}
    radius::T
    basis::B
end

# Make a Blob act exactly like a sphere of the same radius for hill_tensor.
function MeanFieldHomogenization.hill_tensor(b::Blob, C₀::TensND.AbstractTens; kw...)
    return MeanFieldHomogenization.hill_tensor(Ellipsoid(b.radius, b.radius, b.radius), C₀; kw...)
end

# Required AbstractInclusion interface bits — defer to spherical equivalent
MeanFieldHomogenization.material_symmetry(b::Blob) =
    MeanFieldHomogenization.material_symmetry(Ellipsoid(b.radius, b.radius, b.radius))
MeanFieldHomogenization.dimension(b::Blob) = 3
MeanFieldHomogenization.shape_trait(b::Blob) =
    MeanFieldHomogenization.shape_trait(Ellipsoid(b.radius, b.radius, b.radius))
MeanFieldHomogenization.shape_tensor(b::Blob) =
    MeanFieldHomogenization.shape_tensor(Ellipsoid(b.radius, b.radius, b.radius))
MeanFieldHomogenization.inclusion_basis(b::Blob) = b.basis
MeanFieldHomogenization.eshelby_tensor(b::Blob, C₀; kw...) =
    MeanFieldHomogenization.eshelby_tensor(Ellipsoid(b.radius, b.radius, b.radius), C₀; kw...)

@testset "User-defined inclusion (Blob) — geometry derivative works without code change" begin
    rve = RVE()
    basis = TensND.CanonicalBasis{3, Float64}()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :B, Blob{Float64, typeof(basis)}(1.0, basis),
        Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.2
    )

    # GeometryParameter on user type: dispatch via @generated fallback
    p = geometry(:B, :radius)
    @test get_param(rve, p) ≈ 1.0

    # Round-trip
    rve2 = set_param(rve, p, 1.5)
    @test get_param(rve2, p) ≈ 1.5

    # Derivative through Dilute (uses hill_tensor of Blob = sphere)
    ∂ = derivative(rve, Dilute(), p; indexer = idxC)
    @test isfinite(∂)
    # For a sphere of radius r, all moduli are independent of r — derivative is 0
    @test isapprox(∂, 0.0; atol = 1.0e-10)
end
