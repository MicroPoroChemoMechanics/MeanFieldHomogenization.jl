# =============================================================================
#  test_cod_symbolic.jl — the COD chain on symbolic element types.
#
#  Number types are part of the contract here too.  Every closed form of
#  `cod_analytical.jl` is written generically over `T <: Number`, and
#  `Elliptic.ell_K` / `ell_E` are routed to `sympy.elliptic_{k,e}` by the
#  SymPy weak extension — so `cod_tensor` is expected to return a *symbolic*
#  COD tensor for a symbolic reference stiffness.  Two things this pins:
#
#   * the `η = 1` branch of `_elliptic_CS` must not be taken for `Sym`
#     (`T <: Real` is false), otherwise a symbolic `η` would silently collapse
#     to the penny values;
#   * `_ti_aligned` must not compare the symmetry axis with `isapprox`, which
#     is undefined on `Sym`.  That seam was the one place in the whole COD
#     chain without a `T <: Real` guard, and it made `cod_tensor` die in
#     dispatch on a `TensTI{4, Sym}` reference.
#
#  `scripts/09_cod_symbolic_green.jl` derives these same closed forms from the
#  Fourier Green operator; this file only guards the entry point.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using SymPy
using ForwardDiff
using Symbolics

# `Symbolics` must be a *declared* test dependency and loaded here, not merely
# reached as `TensND.Symbolics`: the weak extension
# `MeanFieldHomogenizationSymbolicsExt` is only registered when the trigger
# package is resolved in this environment's manifest.
const Sy = Symbolics

# `SymPy` and `Symbolics` both export `@syms`, so with both loaded the bare macro
# is ambiguous — SymPy's is qualified throughout this file, and Symbolics'
# variables are declared with `Sy.@variables`.

@testset "COD — is_hard_numeric is the predicate, not T <: Real" begin
    ell = MeanFieldHomogenization.Elliptic
    # The whole point: `Symbolics.Num <: Real` yet its comparisons are symbolic,
    # while `ForwardDiff.Dual <: Real` and compares to a `Bool`. Any guard that
    # keys on `<: Real` gets one of the two wrong; this predicate gets both.
    @test Sy.Num <: Real
    @test !ell.is_hard_numeric(Sy.Num)
    @test !ell.is_hard_numeric(Sym)
    @test ell.is_hard_numeric(Float64)
    @test ell.is_hard_numeric(Int)
    @test ell.is_hard_numeric(Rational{Int})
    @test ell.is_hard_numeric(typeof(ForwardDiff.Dual(1.0, 0.0)))
end

@testset "COD — ForwardDiff through the elastic closed forms" begin
    h = 1.0e-6
    iso(E, ν) = TensISO{3}(E / (1 - 2ν), E / (1 + ν))

    fE(E) = cod_tensor(EllipticCrack(1.0, 0.6), iso(E, 0.3))[3, 3]
    @test ForwardDiff.derivative(fE, 210.0) ≈ (fE(210.0 + h) - fE(210.0 - h)) / (2h) rtol = 1.0e-6

    fν(ν) = cod_tensor(EllipticCrack(1.0, 0.6), iso(210.0, ν))[3, 3]
    @test ForwardDiff.derivative(fν, 0.3) ≈ (fν(0.3 + h) - fν(0.3 - h)) / (2h) rtol = 1.0e-6

    fη(η) = cod_tensor(EllipticCrack(1.0, η), iso(210.0, 0.3))[3, 3]
    @test ForwardDiff.derivative(fη, 0.6) ≈ (fη(0.6 + h) - fη(0.6 - h)) / (2h) rtol = 1.0e-6

    fr(E) = cod_tensor(RibbonCrack(1.0), iso(E, 0.3))[3, 3]
    @test ForwardDiff.derivative(fr, 210.0) ≈ (fr(210.0 + h) - fr(210.0 - h)) / (2h) rtol = 1.0e-6

    fTI(E) = cod_tensor(PennyCrack(1.0), fromISO(iso(E, 0.3), [0.0, 0.0, 1.0]))[3, 3]
    @test ForwardDiff.derivative(fTI, 210.0) ≈ (fTI(210.0 + h) - fTI(210.0 - h)) / (2h) rtol = 1.0e-6
end

@testset "COD — Symbolics.Num through the elastic closed forms" begin
    # Needs the `MeanFieldHomogenizationSymbolicsExt` extension: without it
    # `ell_K`/`ell_E` fall through to the AGM recursion and unroll ~60 nested
    # `sqrt` into the expression tree.
    @test Base.get_extension(
        MeanFieldHomogenization, :MeanFieldHomogenizationSymbolicsExt
    ) !== nothing

    Sy.@variables Eν νν ην
    C = TensISO{3}(Eν / (1 - 2νν), Eν / (1 + νν))

    for crack in (EllipticCrack(1.0, 0.6), PennyCrack(1.0), RibbonCrack(1.0))
        B = cod_tensor(crack, C)
        @test eltype(get_array(B)) <: Sy.Num
    end

    # A *symbolic* aspect ratio: this is what `_classify_crack` and
    # `_sort_axes_and_basis` used to reject, both keying on `T <: Real`.
    B = cod_tensor(EllipticCrack(Sy.Num(1.0), ην), C)
    @test eltype(get_array(B)) <: Sy.Num

    # `ell_E` stays an unexpanded call rather than 60 AGM steps.
    ell = MeanFieldHomogenization.Elliptic
    @test occursin("ell_E", string(ell.ell_E(1 - ην^2)))
    @test !occursin("ell_E", string(ell.ell_E(0.36)))       # numeric still evaluates
end

@testset "COD — aligned TI on Symbolics.Num takes the closed form" begin
    # Two separate traps had to be cleared for this to work, and both are
    # regression-guarded here because each one silently *downgraded* the answer
    # instead of failing loudly:
    #
    #  1. `_is_unit_alignment` keyed on `::Real`, which `Num` satisfies, so
    #     `isapprox` threw;
    #  2. once that was fixed, `|axis·n̂|` came out as an unevaluated `abs(1.0)`
    #     — Symbolics does not fold `abs` of a literal, and not even `tsimplify`
    #     does (it returns `-1 + abs(1.0)`). The alignment test therefore said
    #     "not aligned" and `cod_tensor` fell through to a *numerical* back-end,
    #     which cannot run on a boxed scalar at all. Comparing the **square**
    #     uses only `*`, which does fold.
    Sy.@variables c11 c12 c33 c44
    MFHCr = MeanFieldHomogenization.Cracks
    C_al = tens_TI(c11, c12 / 2, c12, c11, c44, [0.0, 0.0, 1.0])
    C_rot = tens_TI(c11, c12 / 2, c12, c11, c44, [1.0, 0.0, 0.0])
    penny = PennyCrack(1.0)

    @test MFHCr._ti_aligned(C_al, crack_basis(penny))
    @test !MFHCr._ti_aligned(C_rot, crack_basis(penny))

    B = cod_tensor(penny, C_al)
    @test eltype(get_array(B)) <: Sy.Num
    @test !occursin("nan", lowercase(string(B[3, 3])))

    # A non-aligned symbolic TI reference has no closed form, so it must fail
    # with an *actionable* message rather than deep inside StaticArrays.
    err = try
        cod_tensor(penny, C_rot); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("isbits", err.msg)
end

@testset "COD symbolic — isotropic matrix, symbolic (E, ν, η)" begin
    SymPy.@syms Es::positive νs::positive ηs::positive
    C_sym = TensISO{3}(Es / (1 - 2 * νs), Es / (1 + νs))    # (3k, 2μ)

    crack = EllipticCrack(one(Sym), ηs)
    @test crack isa EllipticCrack{<:Sym, EllipticShape}

    B = cod_tensor(crack, C_sym)
    @test eltype(get_array(B)) <: Sym

    # The symbolic ell_K / ell_E extension must have been used: a symbolic η
    # cannot go through the `iszero(k²)` penny shortcut.
    @test occursin("elliptic_e", string(B[3, 3]))

    # Closed forms of `_cod_iso_ellipse`, re-derived here from the paper form.
    𝒦 = sympy.elliptic_k(1 - ηs^2)
    ℰ = sympy.elliptic_e(1 - ηs^2)
    𝒞 = (ℰ - ηs^2 * 𝒦) / (1 - ηs^2)
    𝒮 = (𝒦 - ℰ) / (1 - ηs^2)
    χ = 8 * (1 - νs^2) / (3 * Es)                            # no η in χ
    @test iszero(tsimplify(B[3, 3] - χ / ℰ))
    @test iszero(tsimplify(B[1, 1] - χ / ((1 - νs) * 𝒞 + ηs^2 * 𝒮)))
    @test iszero(tsimplify(B[2, 2] - χ / ((1 - νs) * ηs^2 * 𝒮 + 𝒞)))
    @test iszero(tsimplify(B[1, 2]))

    # Ribbon: fully rational, so the identity is exact and cheap.
    Br = cod_tensor(RibbonCrack(one(Sym)), C_sym)
    χr = PI * (1 - νs^2) / Es
    @test iszero(tsimplify(Br[1, 1] - χr / (1 - νs)))
    @test iszero(tsimplify(Br[2, 2] - χr))
    @test iszero(tsimplify(Br[3, 3] - χr))
end

@testset "COD symbolic — TI matrix aligned with the crack normal" begin
    # Symbolic stiffness components, TI axis = e₃ = crack normal.  Before the
    # `_ti_aligned` guard this call threw inside dispatch.
    SymPy.@syms C1111::positive C1122::real C1133::real C3333::positive C2323::positive
    C_ti = tens_TI(C1111, C1122, C1133, C3333, C2323, [Sym(0), Sym(0), Sym(1)])
    @test C_ti isa TensND.TensTI{4}
    @test eltype(get_array(C_ti)) <: Sym

    for crack in (PennyCrack(one(Sym)), RibbonCrack(one(Sym)))
        B = cod_tensor(crack, C_ti)
        @test eltype(get_array(B)) <: Sym
        # Diagonal in the crack frame, and every component non-trivial.
        @test iszero(tsimplify(B[1, 2]))
        @test iszero(tsimplify(B[1, 3]))
        @test iszero(tsimplify(B[2, 3]))
        @test all(!iszero(tsimplify(B[i, i])) for i in 1:3)
        # `!iszero` does NOT catch a NaN, and a symbolic penny used to produce
        # one: `_elliptic_CS` took its removable `η = 1` shortcut only for
        # `T <: Real`, so `𝒞 = (ℰ - η²𝒦)/k²` evaluated 0/0. Check explicitly.
        @test !any(occursin("nan", lowercase(string(B[i, i]))) for i in 1:3)
    end

    # A symbolic *aligned* TI reference must take the Analytical branch; a
    # symbolic non-aligned one must not (there is no closed form for it, and
    # the cubature back-ends cannot run on `Sym`).
    MFHC = MeanFieldHomogenization.Core
    MFHCr = MeanFieldHomogenization.Cracks
    @test MFHCr._ti_aligned(C_ti, crack_basis(PennyCrack(one(Sym))))
    C_ti_x = tens_TI(C1111, C1122, C1133, C3333, C2323, [Sym(1), Sym(0), Sym(0)])
    @test !MFHCr._ti_aligned(C_ti_x, crack_basis(PennyCrack(one(Sym))))
    @test MFHC._resolve_algo(Val(:auto), PennyCrack(one(Sym)), C_ti) isa MFHC.Analytical
end

@testset "COD symbolic — thermal (order 2) at full anisotropy" begin
    # The transport branch is symbolically transparent *end to end*, unlike the
    # elastic one: the order-2 acoustic form is a scalar, so the closed form
    # needs only a 2×2 eigenvalue problem on `adj(K₀)`. Before v0.4.0 the
    # anisotropic branch went through `eigen` + `svdvals` and could not run on
    # `Sym` at all.
    SymPy.@syms k₀s::positive ηs::positive

    K_iso = TensISO{3}(k₀s)
    for (crack, ref) in (
            (EllipticCrack(one(Sym), ηs), 4 / (3 * k₀s * sympy.elliptic_e(1 - ηs^2))),
            (PennyCrack(one(Sym)), 8 / (3 * PI * k₀s)),
            (RibbonCrack(one(Sym)), PI / (2 * k₀s)),
        )
        b = cod_tensor(crack, K_iso)
        @test b isa Sym
        @test iszero(tsimplify(b - ref))
    end

    # Fully anisotropic K₀, six free symbols: the adjugate branch must produce a
    # symbolic answer at all. That is the regression guard — before v0.4.0 this
    # call went through `eigen`/`svdvals` and threw on `Sym`.
    SymPy.@syms K11::positive K22::positive K33::positive K12::real K13::real K23::real
    K_aniso = TensND.Tens(
        Sym[K11 K12 K13; K12 K22 K23; K13 K23 K33], CanonicalBasis{3, Sym}()
    )
    b_aniso = cod_tensor(EllipticCrack(one(Sym), ηs), K_aniso)
    @test b_aniso isa Sym
    @test occursin("elliptic_e", string(b_aniso))
    @test !occursin("nan", lowercase(string(b_aniso)))

    # Value identities are checked at *concrete rational* aspect ratios, not at a
    # free `ηs`: the eigenvalue gap carries √(tr²−4det), which for the diagonal
    # cases is √((1−η²)²), and nothing in SymPy's assumption system says η ≤ 1,
    # so it stays unevaluated. With a concrete η the radical closes.
    SymPy.@syms k_ts::positive k_ns::positive
    K_ti = TensND.Tens(Sym[k_ts 0 0; 0 k_ts 0; 0 0 k_ns], CanonicalBasis{3, Sym}())

    for ηv in (Sym(1) / 2, Sym(7) / 10)
        ℰv = sympy.elliptic_e(1 - ηv^2)
        # The anisotropic branch reduces to the isotropic closed form …
        b_red = subs(
            cod_tensor(EllipticCrack(one(Sym), ηv), K_aniso),
            Dict(
                K11 => k₀s, K22 => k₀s, K33 => k₀s,
                K12 => Sym(0), K13 => Sym(0), K23 => Sym(0),
            ),
        )
        @test iszero(tsimplify(b_red - 4 / (3 * k₀s * ℰv)))
        # … and, for an aligned TI conductor, gives the geometric mean √(k_t k_n).
        @test iszero(
            tsimplify(
                cod_tensor(EllipticCrack(one(Sym), ηv), K_ti) -
                    4 / (3 * sqrt(k_ts * k_ns) * ℰv)
            )
        )
    end
end

@testset "COD symbolic — the TI formula reduces to the isotropic one" begin
    # The same matrix presented as a `TensISO` and as an *aligned* `TensTI` must
    # give the same 𝐁, for a **free symbolic aspect ratio** — which is what a
    # penny-only or Float64-only check cannot see (`test_cod_ti_aligned.jl`
    # covers the numeric side).
    #
    # The moduli are exact rationals rather than symbols on purpose: with
    # symbolic λ, μ the TI branch carries the nested radical σᵞ and the
    # difference becomes a `simplify` SymPy will not close. With rational
    # moduli σᵞ collapses to a number and the comparison is exact in η.
    SymPy.@syms ηs::positive
    λs, μs = Sym(1), Sym(1) / 2
    C_iso = TensISO{3}(3 * (λs + 2 * μs / 3), 2 * μs)
    C_ti = tens_TI(λs + 2μs, λs, λs, λs + 2μs, μs, [Sym(0), Sym(0), Sym(1)])
    @test C_ti isa TensND.TensTI{4}

    for crack in (EllipticCrack(one(Sym), ηs), PennyCrack(one(Sym)), RibbonCrack(one(Sym)))
        B_iso = cod_tensor(crack, C_iso)
        B_ti = cod_tensor(crack, C_ti)
        @test all(iszero(tsimplify(B_iso[i, i] - B_ti[i, i])) for i in 1:3)
    end
end
