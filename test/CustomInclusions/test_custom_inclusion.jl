# =============================================================================
#  test_custom_inclusion.jl — the user-defined-inclusion contract.
#
#  Coverage:
#   1. The three entry gates (Hill / localization / contribution) plug a
#      `CustomInclusion` into every scheme and are indistinguishable from the
#      native `Ellipsoid`, in elasticity and in conduction.
#   2. A flat custom inclusion registered with `density = ε` reproduces
#      `EllipticCrack` — i.e. the "amount × contribution" seam and its
#      geometric prefactor are traversed correctly.
#   3. `shape_trait`-based dispatch: a user crack type that only implements
#      `cod_tensor` inherits ℍ, ℕ, 𝐑, 𝐍_K and the four `delta_*`.
#   4. A subtype of `AbstractCustomInclusion` reaches `hill_tensor` through
#      the open `_kernel` table.
#   5. `is_homogeneous_inclusion = false` routes to the exact branch.
#   6. Orientation averaging (`IsoSymmetrize`) is applied by the scheme and
#      therefore costs the user nothing.
#   7. ForwardDiff sensitivity through a geometric field of a user struct.
#   8. `check_inclusion_interface` reports the right gate and the real gaps.
#   9. Constructor guard rails.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const CI_C_M = TensISO{3}(30.0, 10.0)
const CI_C_I = TensISO{3}(60.0, 20.0)
const CI_K_M = TensISO{3}(2.0)
const CI_K_I = TensISO{3}(7.0)

_ci_c1111(C) = get_array(C)[1, 1, 1, 1]
_ci_k11(K) = get_array(K)[1, 1]

"Build a two-phase RVE whose inclusion geometry is `geom`."
function _ci_rve_with(geom, prop_m, prop_i, key; kwargs...)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(key => prop_m); fraction = :rest)
    add_phase!(rve, :I, geom, Dict(key => prop_i); kwargs...)
    return rve
end

@testset "CustomInclusion — the three entry gates" begin
    # A prolate spheroid, so that the answer is anisotropic and a wrong
    # rotation or a dropped term cannot hide.
    ell = Ellipsoid(3.0, 1.0, 1.0; euler_angles = (0.3, 0.7, 0.0))

    gate_A = CustomInclusion(
        (3.0, 1.0, 1.0);
        basis = MeanFieldHomogenization.inclusion_basis(ell),
        hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...),
    )
    gate_B = CustomInclusion(
        (3.0, 1.0, 1.0);
        basis = MeanFieldHomogenization.inclusion_basis(ell),
        strain_strain_loc = (P₁, P₀; kw...) -> strain_strain_loc(ell, P₁, P₀; kw...),
        gradient_gradient_loc = (P₁, P₀; kw...) ->
        gradient_gradient_loc(ell, P₁, P₀; kw...),
    )
    gate_C = CustomInclusion(
        (3.0, 1.0, 1.0);
        basis = MeanFieldHomogenization.inclusion_basis(ell),
        stiffness_contribution = (P₁, P₀; kw...) ->
        stiffness_contribution(ell, P₁, P₀; kw...),
        compliance_contribution = (P₁, P₀; kw...) ->
        compliance_contribution(ell, P₁, P₀; kw...),
        conductivity_contribution = (P₁, P₀; kw...) ->
        conductivity_contribution(ell, P₁, P₀; kw...),
        resistivity_contribution = (P₁, P₀; kw...) ->
        resistivity_contribution(ell, P₁, P₀; kw...),
    )

    # Gate C supplies only the contribution tensors.  That is enough for every
    # scheme whose kernel consumes `N`/`H` alone, but not for the ones that
    # also need the dilute concentration tensor `A` (Mori-Tanaka and the two
    # self-consistent flavors) — see the warning in the `CustomInclusion`
    # docstring.
    all_schemes = (
        Dilute(), DiluteDual(), MoriTanaka(), Maxwell(),
        PonteCastanedaWillis(), SelfConsistent(),
        AsymmetricSelfConsistent(), DifferentialScheme(),
    )
    gate_C_schemes = (
        Dilute(), DiluteDual(), Maxwell(), PonteCastanedaWillis(),
        DifferentialScheme(),
    )

    @testset "elasticity — every scheme" begin
        ref = _ci_rve_with(ell, CI_C_M, CI_C_I, :C; fraction = 0.25)
        for (geom, schemes) in
            ((gate_A, all_schemes), (gate_B, all_schemes), (gate_C, gate_C_schemes))
            got_rve = _ci_rve_with(geom, CI_C_M, CI_C_I, :C; fraction = 0.25)
            for scheme in schemes
                @test get_array(homogenize(got_rve, scheme, :C)) ≈
                    get_array(homogenize(ref, scheme, :C)) rtol = 1.0e-10
            end
        end
    end

    @testset "conduction — every one-shot scheme" begin
        ref = _ci_rve_with(ell, CI_K_M, CI_K_I, :K; fraction = 0.25)
        for (geom, schemes) in (
                (gate_A, (Dilute(), DiluteDual(), MoriTanaka(), Maxwell())),
                (gate_B, (Dilute(), DiluteDual(), MoriTanaka(), Maxwell())),
                (gate_C, (Dilute(), DiluteDual(), Maxwell())),
            )
            got_rve = _ci_rve_with(geom, CI_K_M, CI_K_I, :K; fraction = 0.25)
            for scheme in schemes
                @test get_array(homogenize(got_rve, scheme, :K)) ≈
                    get_array(homogenize(ref, scheme, :K)) rtol = 1.0e-10
            end
        end
    end

    @testset "gate C without `A` is refused loudly by Mori-Tanaka" begin
        rve = _ci_rve_with(gate_C, CI_C_M, CI_C_I, :C; fraction = 0.25)
        @test_throws ArgumentError homogenize(rve, MoriTanaka(), :C)
    end

    @testset "gate A also yields the derived localization tensors" begin
        for f in (
                strain_strain_loc, stress_strain_loc,
                strain_stress_loc, stress_stress_loc,
            )
            @test get_array(f(gate_A, CI_C_I, CI_C_M)) ≈ get_array(f(ell, CI_C_I, CI_C_M))
        end
        for f in (
                gradient_gradient_loc, flux_gradient_loc,
                gradient_flux_loc, flux_flux_loc,
            )
            @test get_array(f(gate_A, CI_K_I, CI_K_M)) ≈ get_array(f(ell, CI_K_I, CI_K_M))
        end
    end
end

@testset "CustomInclusion — flat object with a density amount" begin
    a, b, ε = 1.0, 0.25, 0.08
    crack = EllipticCrack(a, b)

    flat = CustomInclusion(
        (a, b, 0.0);
        basis = MeanFieldHomogenization.inclusion_basis(crack),
        density_factor = 4π / 3,
        compliance_contribution = (P₀; kw...) -> compliance_contribution(crack, P₀; kw...),
        stiffness_contribution = (P₀; kw...) -> stiffness_contribution(crack, P₀; kw...),
        conductivity_contribution = (P₀; kw...) ->
        conductivity_contribution(crack, P₀; kw...),
    )

    @testset "elasticity" begin
        ref = _ci_rve_with(crack, CI_C_M, CI_C_M, :C; density = ε)
        got = _ci_rve_with(flat, CI_C_M, CI_C_M, :C; density = ε)
        for scheme in (Dilute(), DiluteDual(), MoriTanaka(), SelfConsistent())
            @test get_array(homogenize(got, scheme, :C)) ≈
                get_array(homogenize(ref, scheme, :C)) rtol = 1.0e-10
        end
    end

    @testset "conduction" begin
        ref = _ci_rve_with(crack, CI_K_M, CI_K_M, :K; density = ε)
        got = _ci_rve_with(flat, CI_K_M, CI_K_M, :K; density = ε)
        for scheme in (Dilute(), MoriTanaka())
            @test get_array(homogenize(got, scheme, :K)) ≈
                get_array(homogenize(ref, scheme, :K)) rtol = 1.0e-10
        end
    end

    @testset "the prefactor really is the seam" begin
        H = compliance_contribution(flat, CI_C_M)
        @test get_array(delta_compliance(flat, H, ε)) ≈
            get_array(delta_compliance(crack, H, ε))
        # A wrong prefactor must show up as a different effective compliance.
        wrong = CustomInclusion(
            (a, b, 0.0);
            basis = MeanFieldHomogenization.inclusion_basis(crack),
            density_factor = 1.0,
            compliance_contribution = (P₀; kw...) ->
            compliance_contribution(crack, P₀; kw...),
            stiffness_contribution = (P₀; kw...) ->
            stiffness_contribution(crack, P₀; kw...),
        )
        @test !isapprox(
            get_array(delta_compliance(wrong, H, ε)),
            get_array(delta_compliance(crack, H, ε))
        )
    end

    @testset "a density amount without a density_factor is refused" begin
        nofactor = CustomInclusion(
            (a, b, 0.0);
            basis = MeanFieldHomogenization.inclusion_basis(crack),
            compliance_contribution = (P₀; kw...) ->
            compliance_contribution(crack, P₀; kw...),
            stiffness_contribution = (P₀; kw...) ->
            stiffness_contribution(crack, P₀; kw...),
        )
        rve = _ci_rve_with(nofactor, CI_C_M, CI_C_M, :C; density = ε)
        @test_throws ArgumentError homogenize(rve, Dilute(), :C)
    end
end

# =============================================================================
#  A user crack type: only `cod_tensor` is implemented, everything else is
#  inherited through `shape_trait`.
# =============================================================================

struct DelegatingCrack{T, B <: TensND.AbstractBasis} <: MeanFieldHomogenization.AbstractCrack{T}
    a::T
    b::T
    basis::B
end

_as_elliptic(c::DelegatingCrack) = EllipticCrack(c.a, c.b, c.basis)

MeanFieldHomogenization.shape_trait(::DelegatingCrack) = MeanFieldHomogenization.EllipticShape
MeanFieldHomogenization.shape_tensor(c::DelegatingCrack) =
    MeanFieldHomogenization.shape_tensor(_as_elliptic(c))
# One method per tensor order: the generic `cod_tensor` is declared separately
# for `AbstractTens{4,3}` and `AbstractTens{2,3}`, so a single method typed on
# `AbstractTens` would be ambiguous with both.
MeanFieldHomogenization.Cracks.cod_tensor(
    c::DelegatingCrack, C₀::TensND.AbstractTens{4, 3}; kw...
) = cod_tensor(_as_elliptic(c), C₀; kw...)
MeanFieldHomogenization.Cracks.cod_tensor(
    c::DelegatingCrack, K₀::TensND.AbstractTens{2, 3}; kw...
) = cod_tensor(_as_elliptic(c), K₀; kw...)

@testset "user crack — `cod_tensor` alone unlocks the whole chain" begin
    a, b, ε = 1.0, 0.3, 0.05
    basis = TensND.CanonicalBasis{3, Float64}()
    mine = DelegatingCrack(a, b, basis)
    ref = EllipticCrack(a, b, basis)

    @test get_array(compliance_contribution(mine, CI_C_M)) ≈
        get_array(compliance_contribution(ref, CI_C_M))
    @test get_array(stiffness_contribution(mine, CI_C_M)) ≈
        get_array(stiffness_contribution(ref, CI_C_M))
    @test get_array(compliance_contribution(mine, CI_K_M)) ≈
        get_array(compliance_contribution(ref, CI_K_M))
    @test get_array(conductivity_contribution(mine, CI_K_M)) ≈
        get_array(conductivity_contribution(ref, CI_K_M))
    @test MeanFieldHomogenization.Cracks.crack_density_factor(mine) ≈ 4π / 3

    H = compliance_contribution(mine, CI_C_M)
    N = stiffness_contribution(mine, CI_C_M)
    @test get_array(delta_compliance(mine, H, ε)) ≈ get_array(delta_compliance(ref, H, ε))
    @test get_array(delta_stiffness(mine, N, ε)) ≈ get_array(delta_stiffness(ref, N, ε))

    for scheme in (Dilute(), DiluteDual(), MoriTanaka(), SelfConsistent())
        got = homogenize(_ci_rve_with(mine, CI_C_M, CI_C_M, :C; density = ε), scheme, :C)
        exp = homogenize(_ci_rve_with(ref, CI_C_M, CI_C_M, :C; density = ε), scheme, :C)
        @test get_array(got) ≈ get_array(exp) rtol = 1.0e-10
    end

    @testset "an unknown shape trait fails loudly, not silently" begin
        struct WeirdShape end
        weird = DelegatingCrack(a, b, basis)
        # Locally override the trait for a throw-away type.
        @test_throws ArgumentError MeanFieldHomogenization.Cracks._compliance_from_B(
            WeirdShape, weird, cod_tensor(ref, CI_C_M)
        )
    end
end

# =============================================================================
#  A subtype of `AbstractCustomInclusion` reaching `hill_tensor` through the
#  open `_kernel` table rather than by overriding `hill_tensor`.
# =============================================================================

struct KernelBlob{T, B <: TensND.AbstractBasis} <: MeanFieldHomogenization.AbstractCustomInclusion{T}
    radius::T
    basis::B
end

_blob_ell(b::KernelBlob) = Ellipsoid(b.radius, b.radius, b.radius)

MeanFieldHomogenization.dimension(::KernelBlob) = 3
MeanFieldHomogenization.inclusion_basis(b::KernelBlob) = b.basis
MeanFieldHomogenization.shape_trait(::KernelBlob) = MeanFieldHomogenization.Spherical
MeanFieldHomogenization.shape_tensor(b::KernelBlob) = MeanFieldHomogenization.shape_tensor(_blob_ell(b))

MeanFieldHomogenization.Elasticity._kernel(
    b::KernelBlob, C₀::TensND.AbstractTens, ::MeanFieldHomogenization.Analytical; kw...
) = hill_tensor(_blob_ell(b), C₀; kw...)

@testset "AbstractCustomInclusion subtype through the `_kernel` table" begin
    blob = KernelBlob(1.0, TensND.CanonicalBasis{3, Float64}())
    ell = Ellipsoid(1.0)

    @test get_array(hill_tensor(blob, CI_C_M)) ≈ get_array(hill_tensor(ell, CI_C_M))
    @test get_array(eshelby_tensor(blob, CI_C_M)) ≈ get_array(eshelby_tensor(ell, CI_C_M))
    @test get_array(hill_tensor(blob, CI_K_M)) ≈ get_array(hill_tensor(ell, CI_K_M))

    for scheme in (Dilute(), MoriTanaka(), SelfConsistent())
        got = homogenize(_ci_rve_with(blob, CI_C_M, CI_C_I, :C; fraction = 0.2), scheme, :C)
        exp = homogenize(_ci_rve_with(ell, CI_C_M, CI_C_I, :C; fraction = 0.2), scheme, :C)
        @test get_array(got) ≈ get_array(exp) rtol = 1.0e-10
    end

    @test check_inclusion_interface(blob; verbose = false)
end

@testset "heterogeneous inclusion needs BOTH localizations" begin
    # A heterogeneous inclusion has no single `C₁`, so the generic identity
    # `A_σε = C₁ : A_εε` does not hold: the stress-side localization must be
    # supplied too.  `LayeredSphere` is the reference heterogeneous inclusion.
    sphere = LayeredSphere((0.5, 1.0), (TensISO{3}(60.0, 20.0), TensISO{3}(90.0, 30.0)))
    C_i = TensISO{3}(60.0, 20.0)

    # Complete: both sides of the pair.
    het = CustomInclusion(;
        homogeneous = false,
        strain_strain_loc = (P₁, P₀; kw...) -> strain_strain_loc(sphere, P₁, P₀; kw...),
        stress_strain_loc = (P₁, P₀; kw...) -> stress_strain_loc(sphere, P₁, P₀; kw...),
        stiffness_contribution = (P₁, P₀; kw...) ->
        stiffness_contribution(sphere, P₁, P₀; kw...),
        compliance_contribution = (P₁, P₀; kw...) ->
        compliance_contribution(sphere, P₁, P₀; kw...),
    )
    # Incomplete: strain side only, as one is tempted to write.
    partial = CustomInclusion(;
        homogeneous = false,
        strain_strain_loc = (P₁, P₀; kw...) -> strain_strain_loc(sphere, P₁, P₀; kw...),
        stiffness_contribution = (P₁, P₀; kw...) ->
        stiffness_contribution(sphere, P₁, P₀; kw...),
        compliance_contribution = (P₁, P₀; kw...) ->
        compliance_contribution(sphere, P₁, P₀; kw...),
    )

    @test !MeanFieldHomogenization.is_homogeneous_inclusion(het)
    @test MeanFieldHomogenization.is_homogeneous_inclusion(
        CustomInclusion(; hill_tensor = (P₀; kw...) -> hill_tensor(Ellipsoid(1.0), P₀; kw...))
    )

    ref = _ci_rve_with(sphere, CI_C_M, C_i, :C; fraction = 0.2)
    ok = _ci_rve_with(het, CI_C_M, C_i, :C; fraction = 0.2)
    bad = _ci_rve_with(partial, CI_C_M, C_i, :C; fraction = 0.2)

    # `Dilute` and `MoriTanaka` consume only `(A, N)`, so they cannot see the
    # gap — which is exactly what makes the omission dangerous.
    for scheme in (Dilute(), MoriTanaka())
        @test get_array(homogenize(bad, scheme, :C)) ≈
            get_array(homogenize(ref, scheme, :C)) rtol = 1.0e-10
    end

    # `SelfConsistent` consumes the stress average: there the complete
    # inclusion reproduces `LayeredSphere` while the partial one is off by
    # several percent.
    @test get_array(homogenize(ok, SelfConsistent(), :C)) ≈
        get_array(homogenize(ref, SelfConsistent(), :C)) rtol = 1.0e-10
    @test !isapprox(
        get_array(homogenize(bad, SelfConsistent(), :C)),
        get_array(homogenize(ref, SelfConsistent(), :C)); rtol = 1.0e-3
    )

    # A heterogeneous inclusion with no layer-wise average has no property to
    # enter the bounds: a bound averages the *constituent* properties, which
    # takes the internal volume fractions the RVE does not carry.  Informative
    # error rather than a `MethodError` three frames down.
    for scheme in (Voigt(), Reuss())
        @test_throws ArgumentError homogenize(ok, scheme, :C)
        @test !MeanFieldHomogenization.Schemes.has_layer_average(het)
    end

    # `AsymmetricSelfConsistent`, on the other hand, must *not* inherit that
    # restriction: its iteration consumes the same two localization tensors as
    # `SelfConsistent`, and only its branch-selection heuristic ever looked at a
    # bound.  With none available it falls back to the dilute estimate and runs,
    # reaching the same fixed point as `SelfConsistent`.
    asc = homogenize(ok, AsymmetricSelfConsistent(), :C)
    @test get_array(asc) ≈
        get_array(homogenize(ref, AsymmetricSelfConsistent(), :C)) rtol = 1.0e-8
    @test get_array(asc) ≈ get_array(homogenize(ok, SelfConsistent(), :C)) rtol = 1.0e-6

    # The root cause, isolated: the two stress-side localizations disagree.
    @test !isapprox(
        get_array(stress_strain_loc(sphere, C_i, CI_C_M)),
        get_array(C_i ⊡ strain_strain_loc(sphere, C_i, CI_C_M)); rtol = 1.0e-3
    )

    # The checker is what protects the user from that silent gap.
    @test check_inclusion_interface(het; verbose = false)
    @test !check_inclusion_interface(partial; verbose = false)

    # A heterogeneous inclusion also needs the stress side for `stress_stress_loc`
    # to be right — it is derived from `stress_strain_loc`, not from `A_εε`.
    @test get_array(stress_stress_loc(het, C_i, CI_C_M)) ≈
        get_array(stress_strain_loc(sphere, C_i, CI_C_M) ⊡ inv(CI_C_M))
end

@testset "shape_tensor is optional" begin
    # A custom inclusion supplies its own response; it owes the package nothing
    # about an equivalent ellipsoidal envelope.
    ell = Ellipsoid(1.0)
    shapeless = CustomInclusion(; hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...))
    @test MeanFieldHomogenization.dimension(shapeless) == 3
    @test_throws ArgumentError MeanFieldHomogenization.shape_tensor(shapeless)
    @test check_inclusion_interface(shapeless; verbose = false)

    # ... and it works in the schemes all the same.
    r = _ci_rve_with(shapeless, CI_C_M, CI_C_I, :C; fraction = 0.2)
    ref = _ci_rve_with(ell, CI_C_M, CI_C_I, :C; fraction = 0.2)
    @test get_array(homogenize(r, MoriTanaka(), :C)) ≈
        get_array(homogenize(ref, MoriTanaka(), :C)) rtol = 1.0e-10

    # Given semi-axes, it does have one.
    shaped = CustomInclusion(
        (2.0, 1.0, 0.5); hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...)
    )
    @test diag(Matrix(get_array(MeanFieldHomogenization.shape_tensor(shaped)))) ≈ [2.0, 1.0, 0.5]
    @test_throws ArgumentError CustomInclusion(
        (2.0, 1.0); dim = 3, hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...)
    )
end

@testset "orientation averaging costs the user nothing" begin
    ell = Ellipsoid(4.0, 1.0, 1.0)
    custom = CustomInclusion(
        (4.0, 1.0, 1.0);
        basis = MeanFieldHomogenization.inclusion_basis(ell),
        hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...),
    )
    for sym in (IsoSymmetrize(), TISymmetrize((0.0, 0.0, 1.0)))
        ref = _ci_rve_with(ell, CI_C_M, CI_C_I, :C; fraction = 0.2, symmetrize = sym)
        got = _ci_rve_with(custom, CI_C_M, CI_C_I, :C; fraction = 0.2, symmetrize = sym)
        Cref = homogenize(ref, MoriTanaka(), :C)
        Cgot = homogenize(got, MoriTanaka(), :C)
        @test get_array(Cgot) ≈ get_array(Cref) rtol = 1.0e-10
    end

    # Isotropic averaging of an anisotropic inclusion must produce an
    # isotropic effective tensor, exactly as for the native type.
    C = homogenize(
        _ci_rve_with(custom, CI_C_M, CI_C_I, :C; fraction = 0.2, symmetrize = IsoSymmetrize()),
        MoriTanaka(), :C
    )
    @test C isa TensND.TensISO
end

# =============================================================================
#  Sensitivities through a geometric field of a user struct.
# =============================================================================

struct SensBlob{T, B <: TensND.AbstractBasis} <: MeanFieldHomogenization.AbstractCustomInclusion{T}
    aspect::T
    basis::B
end

MeanFieldHomogenization.dimension(::SensBlob) = 3
MeanFieldHomogenization.inclusion_basis(b::SensBlob) = b.basis
MeanFieldHomogenization.shape_trait(::SensBlob) = MeanFieldHomogenization.Prolate
MeanFieldHomogenization.shape_tensor(b::SensBlob) =
    MeanFieldHomogenization.shape_tensor(Ellipsoid(b.aspect, 1.0, 1.0))
MeanFieldHomogenization.Elasticity.hill_tensor(b::SensBlob, C₀::TensND.AbstractTens; kw...) =
    hill_tensor(Ellipsoid(b.aspect, one(b.aspect), one(b.aspect)), C₀; kw...)

@testset "ForwardDiff through a user geometric parameter" begin
    basis = TensND.CanonicalBasis{3, Float64}()
    rve = _ci_rve_with(SensBlob(3.0, basis), CI_C_M, CI_C_I, :C; fraction = 0.2)
    ∂ = derivative(rve, Dilute(), geometry(:I, :aspect); indexer = _ci_c1111)

    function f(δ)
        r = _ci_rve_with(SensBlob(3.0 + δ, basis), CI_C_M, CI_C_I, :C; fraction = 0.2)
        return _ci_c1111(homogenize(r, Dilute(), :C))
    end
    h = 1.0e-6
    @test ∂ ≈ (f(h) - f(-h)) / (2h) rtol = 1.0e-5
end

@testset "check_inclusion_interface" begin
    ell = Ellipsoid(1.0)

    @test check_inclusion_interface(ell; verbose = false)
    @test check_inclusion_interface(ell; physics = :conduction, verbose = false)

    # A crack has no volume fraction: no gate is available for `:fraction`.
    crack = EllipticCrack(1.0, 0.25)
    @test check_inclusion_interface(crack; amount = :density, verbose = false)
    @test !check_inclusion_interface(crack; amount = :fraction, verbose = false)

    # Gate C without a density factor is incomplete for a density amount.
    gate_C = CustomInclusion(
        (1.0, 1.0, 1.0);
        stiffness_contribution = (P₁, P₀; kw...) ->
        stiffness_contribution(ell, P₁, P₀; kw...),
        compliance_contribution = (P₁, P₀; kw...) ->
        compliance_contribution(ell, P₁, P₀; kw...),
    )
    @test check_inclusion_interface(gate_C; verbose = false)
    @test !check_inclusion_interface(gate_C; amount = :density, verbose = false)

    # An elasticity-only custom inclusion is not usable in conduction.
    elas_only = CustomInclusion(
        (1.0, 1.0, 1.0);
        strain_strain_loc = (P₁, P₀; kw...) -> strain_strain_loc(ell, P₁, P₀; kw...),
    )
    @test check_inclusion_interface(elas_only; verbose = false)
    @test !check_inclusion_interface(elas_only; physics = :conduction, verbose = false)

    @test_throws ArgumentError check_inclusion_interface(ell; physics = :magic)
    @test_throws ArgumentError check_inclusion_interface(ell; amount = :magic)
end

@testset "CustomInclusion — guard rails" begin
    ell = Ellipsoid(1.0)
    @test_throws ArgumentError CustomInclusion((1.0, 1.0, 1.0))
    @test_throws ArgumentError CustomInclusion(
        (1.0, 1.0, 1.0); hil_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...)
    )

    # Asking for a gate that was not supplied names the ones that were.
    only_A = CustomInclusion(
        (1.0, 1.0, 1.0); hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...)
    )
    @test_throws ArgumentError compliance_contribution(only_A, CI_C_M)

    # Level-0 interface.
    @test MeanFieldHomogenization.dimension(only_A) == 3
    @test MeanFieldHomogenization.shape_trait(only_A) === CustomShape
    @test get_array(MeanFieldHomogenization.shape_tensor(only_A)) ≈ Matrix(1.0I, 3, 3)
    @test MeanFieldHomogenization.element_type(only_A) === Float64

    # Scalar-varargs constructor.
    @test MeanFieldHomogenization.dimension(
        CustomInclusion(2.0, 1.0, 1.0; hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...))
    ) == 3
end
