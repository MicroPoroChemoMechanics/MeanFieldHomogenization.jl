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

@testset "COD symbolic — isotropic matrix, symbolic (E, ν, η)" begin
    @syms Es::positive νs::positive ηs::positive
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
    @syms C1111::positive C1122::real C1133::real C3333::positive C2323::positive
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
    @syms ηs::positive
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
