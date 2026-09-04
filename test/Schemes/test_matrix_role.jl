using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

# =============================================================================
#  test_matrix_role.jl — the contract this refactor exists for.
#
#  Three notions used to travel together under the name "matrix": the phase
#  whose fraction is derived, the reference medium a scheme localizes in, and
#  the homogeneous solid of poromechanics. They are stated separately now, and
#  each block below pins one consequence of that separation.
# =============================================================================

const C_A = iso_stiffness(30.0, 12.0)
const C_B = iso_stiffness(60.0, 25.0)

_two_phase_strict() = begin
    rve = RVE(; closure = :strict)
    add_phase!(rve, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = 0.4)
    add_phase!(rve, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 0.6)
    rve
end

_two_phase_rest() = begin
    rve = RVE()
    add_phase!(rve, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = :rest)
    add_phase!(rve, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 0.6)
    rve
end

@testset "SelfConsistent needs no matrix at all" begin
    # The test the whole refactor exists for: before v0.8 `validate_rve` refused
    # an RVE that designated no matrix, so this could not be written down.
    poly = _two_phase_strict()
    C = homogenize(poly, SelfConsistent(), :C)
    k, mu = k_mu(C)
    @test isfinite(k) && isfinite(mu)

    # Declaring the same microstructure with a complement phase must not change
    # it: `:rest` is bookkeeping, not morphology.
    @test k_mu(homogenize(_two_phase_rest(), SelfConsistent(), :C))[1] ≈ k rtol = 1.0e-8

    # And it is bracketed by the bounds, which need no matrix either.
    @test k_mu(homogenize(poly, Reuss(), :C))[1] ≤ k ≤ k_mu(homogenize(poly, Voigt(), :C))[1]
end

@testset "the matrix is the scheme's choice, and it changes the answer" begin
    rve = _two_phase_strict()
    k_A = k_mu(homogenize(rve, MoriTanaka(:A), :C))[1]
    k_B = k_mu(homogenize(rve, MoriTanaka(:B), :C))[1]
    k_sc = k_mu(homogenize(rve, SelfConsistent(), :C))[1]

    # Two different composites out of one microstructure — which is exactly why
    # the choice cannot live on the RVE.
    @test !(k_A ≈ k_B)
    # The classical Hashin-Shtrikman ordering: taking the softer phase as the
    # matrix gives the lower estimate, and the self-consistent one sits between.
    @test k_A ≤ k_sc ≤ k_B

    # `matrix = ` is the same thing spelled as a keyword.
    @test k_mu(homogenize(rve, MoriTanaka(matrix = :A), :C))[1] ≈ k_A
    @test k_mu(homogenize(rve, Dilute(:A), :C))[1] ≈ k_mu(homogenize(rve, Dilute(matrix = :A), :C))[1]
end

@testset "an unnamed matrix resolves to the complement phase" begin
    rve = _two_phase_rest()
    @test remainder_phase_name(rve) === :A
    @test k_mu(homogenize(rve, MoriTanaka(), :C))[1] ≈
        k_mu(homogenize(rve, MoriTanaka(:A), :C))[1]
    @test k_mu(homogenize(rve, :mt, :C))[1] ≈ k_mu(homogenize(rve, MoriTanaka(:A), :C))[1]
end

@testset "an undecidable matrix is an error that names the candidates" begin
    rve = _two_phase_strict()
    err = try
        homogenize(rve, MoriTanaka(), :C)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    # The message IS the feature here: it must name the candidates and show the
    # form that fixes the call.
    @test occursin(":A", msg) && occursin(":B", msg)
    @test occursin("MoriTanaka(:", msg)
    @test occursin("fraction = :rest", msg)

    # Naming a phase that does not exist is a different, equally explicit error.
    @test_throws ArgumentError homogenize(rve, MoriTanaka(:nope), :C)

    # The schemes that distinguish no phase accept the very same RVE.
    for s in (Voigt(), Reuss(), SelfConsistent())
        @test MeanFieldHomogenization.validate_cell(rve, s) === rve
    end
    @test_throws ArgumentError MeanFieldHomogenization.validate_cell(rve, MoriTanaka())
end

@testset "fraction closures" begin
    # Strict: the declared fractions must already sum to one.
    bad = RVE(; closure = :strict)
    add_phase!(bad, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = 0.4)
    add_phase!(bad, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 0.5)
    @test_throws ArgumentError validate_rve(bad)
    @test validate_rve(_two_phase_strict()) isa RVE

    # Complement: one phase absorbs 1 - Σ f.
    comp = _two_phase_rest()
    @test volume_fraction(comp, :A) ≈ 0.4
    @test remainder_volume_fraction(comp) ≈ 0.4

    # ... and a negative complement warns by default, errors on demand.
    over = RVE()
    add_phase!(over, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = :rest)
    add_phase!(over, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 1.4)
    @test_logs (:warn,) validate_rve(over)
    strict_over = RVE(; closure = ComplementFraction(on_negative = :error))
    add_phase!(strict_over, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = :rest)
    add_phase!(strict_over, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 1.4)
    @test_throws ArgumentError validate_rve(strict_over)

    # Rescale: only the ratios matter.
    rel = RVE(; closure = :rescale)
    add_phase!(rel, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = 2.0)
    add_phase!(rel, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 3.0)
    @test volume_fraction(rel, :A) ≈ 0.4
    @test volume_fraction(rel, :B) ≈ 0.6
    @test k_mu(homogenize(rel, MoriTanaka(:A), :C))[1] ≈
        k_mu(homogenize(_two_phase_strict(), MoriTanaka(:A), :C))[1]

    # At most one complement phase, and none at all under Rescale.
    two = _two_phase_rest()
    @test_throws ArgumentError add_phase!(
        two, :C3, Ellipsoid(1.0), Dict(:C => C_A); fraction = :rest
    )
    @test_throws ArgumentError add_phase!(
        rel, :C3, Ellipsoid(1.0), Dict(:C => C_A); fraction = :rest
    )
end

@testset "crack densities take no part in any closure" begin
    rve = RVE(; closure = :rescale)
    add_phase!(rve, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = 2.0)
    add_phase!(rve, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 3.0)
    add_phase!(rve, :CR, PennyCrack(1.0), Dict(:C => C_A); density = 0.05)
    # The two volumes are renormalized between themselves; the density is not
    # touched, because a flat crack carries no volume to renormalize.
    @test volume_fraction(rve, :A) ≈ 0.4
    @test volume_fraction(rve, :B) ≈ 0.6
    @test crack_density(rve, :CR) ≈ 0.05
    @test volume_fraction(rve, :CR) == 0
end

@testset "sensitivities through the closures" begin
    # Complement: raising the inclusion fraction lowers the complement by as
    # much — the derivative that the pre-0.8 pipeline already had.
    dC = ForwardDiff.derivative(
        f -> begin
            rve = RVE()
            add_phase!(rve, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = :rest)
            add_phase!(rve, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = f)
            k_mu(homogenize(rve, MoriTanaka(:A), :C))[1]
        end, 0.3
    )
    @test isfinite(dC) && dC > 0        # the stiff phase stiffens the composite

    # Rescale: the resolved fractions are invariant under a global scaling of
    # the declared ones, so the derivatives along that direction must cancel.
    # This fails loudly if the quotient rule is not propagated.
    g(s) = begin
        rve = RVE(; closure = :rescale)
        add_phase!(rve, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = 2.0 * s)
        add_phase!(rve, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 3.0 * s)
        k_mu(homogenize(rve, MoriTanaka(:A), :C))[1]
    end
    @test abs(ForwardDiff.derivative(g, 1.0)) < 1.0e-10
end

@testset "the self-consistent seed" begin
    rve = _two_phase_strict()
    ref = k_mu(homogenize(rve, SelfConsistent(; init = :voigt, abstol = 1.0e-13), :C))[1]

    # Away from percolation the fixed point is the fixed point: the seed picks
    # the path, not the answer. This is the real specification of the iteration.
    for init in (:reuss, :A, :B)
        k = k_mu(homogenize(rve, SelfConsistent(; init = init, abstol = 1.0e-13), :C))[1]
        @test k ≈ ref rtol = 1.0e-8
    end
    k_tens = k_mu(
        homogenize(
            rve, SelfConsistent(; init = iso_stiffness(1.0, 0.4), abstol = 1.0e-13), :C
        )
    )[1]
    @test k_tens ≈ ref rtol = 1.0e-8

    # A phase named `:voigt` wins over the keyword — a phase name is the
    # caller's data, a keyword is ours.
    shadow = RVE(; closure = :strict)
    add_phase!(shadow, :voigt, Ellipsoid(1.0), Dict(:C => C_A); fraction = 0.4)
    add_phase!(shadow, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 0.6)
    @test MeanFieldHomogenization.Schemes._sc_initial(:voigt, shadow, :C) === C_A

    @test_throws ArgumentError homogenize(rve, SelfConsistent(; init = :nope), :C)
end

@testset "the error paths say what to do instead" begin
    # Each of these is a message a user can actually hit, so each is pinned.
    S = MeanFieldHomogenization.Schemes

    # A `Remainder` carries no value: reading one must fail by name rather than
    # quietly return something.
    @test_throws ArgumentError S.amount_value(Remainder())
    @test_throws ArgumentError S.scale_by_amount(Remainder(), C_A)

    # Closure coercion, including the spelling variants.
    @test S._to_closure(:strict) isa StrictFractions
    @test S._to_closure(:complement) isa ComplementFraction
    @test S._to_closure(:rest) isa ComplementFraction
    @test S._to_closure(:rescale) isa RescaledFractions
    @test S._to_closure(:normalize) isa RescaledFractions
    @test S._to_closure(nothing) === nothing
    @test S._to_closure(StrictFractions()) isa StrictFractions
    @test_throws ArgumentError S._to_closure(:bogus)

    # `on_negative` takes two values and says so.
    @test_throws ArgumentError ComplementFraction(:bogus)
    @test ComplementFraction(:error).on_negative === :error

    # A fraction Symbol other than `:rest` is a typo, not a feature.
    rve = RVE()
    @test_throws ArgumentError add_phase!(
        rve, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = :bogus
    )

    # Rescaling needs something to divide by.
    empty_rel = RVE(; closure = :rescale)
    add_phase!(empty_rel, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = 0.0)
    @test_throws ArgumentError validate_rve(empty_rel)

    # Setting the complement's amount is refused: it has none to set.
    comp = _two_phase_rest()
    @test_throws ArgumentError S.set_amount!(comp, :A, 0.5)
    @test_throws ArgumentError set_param(
        comp, MeanFieldHomogenization.AmountParameter(:A), 0.5
    )
end

@testset "set_amount! keeps the cached fraction sum in step" begin
    # The cache is what every scheme reads through `volume_fraction`; a mutator
    # that moved an amount without refreshing it would go unnoticed until a
    # homogenization silently used the old fraction.
    S = MeanFieldHomogenization.Schemes
    rve = _two_phase_rest()
    @test volume_fraction(rve, :A) ≈ 0.4

    S.set_amount!(rve, :B, 0.25)
    @test volume_fraction(rve, :B) ≈ 0.25
    @test volume_fraction(rve, :A) ≈ 0.75          # the complement followed
    @test remainder_volume_fraction(rve) ≈ 0.75

    # A crack density moves nothing: it is outside the unit sum.
    add_phase!(rve, :CR, PennyCrack(1.0), Dict(:C => C_A); density = 0.05)
    S.set_amount!(rve, :CR, 0.11)
    @test crack_density(rve, :CR) ≈ 0.11
    @test volume_fraction(rve, :A) ≈ 0.75
end

@testset "the removed API is gone" begin
    rve = _two_phase_rest()
    @test !hasfield(typeof(rve), :matrix_name)
    @test_throws MethodError RVE(:M)
    # Not even a method left to miss: the binding is gone from the module.
    @test !isdefined(MeanFieldHomogenization.Schemes, :matrix_volume_fraction)
    @test !isdefined(MeanFieldHomogenization.Schemes, :matrix_property)
    @test !isdefined(MeanFieldHomogenization.Schemes, :matrix_phase)
    # `add_matrix!` survives for an assembly, where the matrix is structural.
    @test hasmethod(add_matrix!, Tuple{ParticleAssembly, AbstractDict})
    @test !hasmethod(add_matrix!, Tuple{RVE, Ellipsoid, AbstractDict})
end

@testset "poromechanics names its solid phase" begin
    rve = RVE()
    add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => C_B); fraction = :rest)
    add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6)); fraction = 0.2)
    C_hom = homogenize(rve, MoriTanaka(:S), :C)
    b_default = biot_tensor(rve, C_hom)
    b_named = biot_tensor(rve, C_hom; solid = :S)
    @test get_array(b_default) ≈ get_array(b_named)
end
