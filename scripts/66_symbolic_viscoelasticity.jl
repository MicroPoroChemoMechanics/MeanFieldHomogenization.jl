# =============================================================================
#  66_symbolic_viscoelasticity.jl
#
#  Viscoelasticity with symbolic parameters, and symbolic Laplace-Carson
#  inversion.
#
#  Two independent things are demonstrated:
#
#    1. Every model in the catalog accepts **symbolic parameters** — SymPy
#       `Sym` or Symbolics `Num` — and returns its transform as an exact
#       expression.  The constructors validate their arguments only when those
#       are hard numeric (`Elliptic.is_hard_numeric`), so a symbolic `τ` no
#       longer trips `0 < τ`.
#
#    2. With `SymPy` loaded, `inverse_laplace` and `inverse_carson` accept a
#       **symbolic time** and hand the inversion to SymPy, returning `R(t)` or
#       `J(t)` in closed form rather than a number.
#
#  The point of (2) is not to replace the four numerical quadratures — SymPy
#  cannot invert a fractional transform, which is precisely why they exist —
#  but to *derive* the closed forms the catalog would otherwise have to
#  hard-code, and to check the ones it does.
#
#  Usage  : julia --project=docs scripts/66_symbolic_viscoelasticity.jl
#  Output : printed expressions only; no figure.
#
#  This script is deliberately NOT published as a gallery tutorial: SymPy work
#  costs ~45 s of documentation build time per page.  The hand-written page
#  `docs/src/tutorials/symbolic_viscoelasticity.md` covers the same ground.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using SymPy
using Symbolics
using Printf

banner(s) = (println(); println("═"^72); println("  ", s); println("═"^72))

# ─── §1 Symbolic parameters, with Symbolics.jl ───────────────────────────────

banner("§1  Models with Symbolics.jl parameters")

Symbolics.@variables E∞ E₁ τ₁ kk p t

zm = zener_maxwell(E∞, E₁, τ₁)
println("zener_maxwell(E∞, E₁, τ₁)")
println("  R*(p) = ", Symbolics.simplify(carson_relaxation(zm, p)))
println("  J*(p) = ", Symbolics.simplify(carson_creep(zm, p)))
println("  R(t)  = ", relaxation(zm, t))

# The transform of a fractional model is just as symbolic.
Symbolics.@variables δ kexp hexp
hs = HuetSayegh(E∞, E₁, δ, τ₁, 0.2, 0.65)
println("\nHuetSayegh(E∞, E₁, δ, τ₁, 0.2, 0.65)")
println("  R*(p) = ", carson_relaxation(hs, p))

# …and lifting to a fourth-order tensor keeps everything symbolic.
iso = iso_rheology(Spring(kk), zm)
println("\niso_rheology(Spring(k), zener_maxwell(…))")
println("  C*(p) = ", carson_relaxation(iso, p))

# ─── §2 What a symbolic spectrum cannot do ───────────────────────────────────

banner("§2  The one thing a symbolic spectrum cannot do")

# `maxwell_to_kelvin` locates its roots by bisection between consecutive poles.
# That needs a real ordering and a real sign test, so it refuses — clearly,
# rather than failing later inside the bracketing.
try
    maxwell_to_kelvin(zm)
catch e
    println("maxwell_to_kelvin on symbolic times:")
    println("  ", first(split(sprint(showerror, e), " The transform")))
end
println("\n…but the transform itself stays exact and symbolic:")
println("  J*(p) R*(p) - 1 = ",
    Symbolics.simplify(carson_creep(zm, p) * carson_relaxation(zm, p) - 1))

# ─── §3 Symbolic Laplace-Carson inversion, with SymPy ────────────────────────

banner("§3  Symbolic inversion: the closed forms, derived")

SymPy.@syms ts::positive τ::positive Einf::positive E1::positive
SymPy.@syms ks::positive ηs::positive kp::positive ηp::positive

# The standard solid: both functions of time, from the transform alone.
z = zener_maxwell(Einf, E1, τ)
R_t = inverse_carson(p -> carson_relaxation(z, p), ts)
J_t = inverse_carson(p -> carson_creep(z, p), ts)
println("Zener, from R*(p) alone:")
println("  R(t) = ", R_t)
println("  J(t) = ", sympy.simplify(J_t))

# It agrees with the closed form the model already knows.
println("\n  R(t) - relaxation(model, t) = ",
    sympy.simplify(R_t - relaxation(z, ts)))

# Burgers: a fluid, whose creep function carries a term linear in t.
b = burgers(ks, ηs, kp, ηp)
Jb = inverse_carson(p -> carson_creep(b, p), ts)
println("\nBurgers J(t) = ", sympy.simplify(Jb))
println("  against the closed form : ",
    sympy.simplify(Jb - creep(b, ts)))

# The relaxation function of Burgers is the messy `cosh`/`sinh` expression that
# had to be derived in a computer-algebra system to serve as this package's
# oracle.  SymPy derives it here from the transform, in one line.
Rb = inverse_carson(p -> carson_relaxation(b, p), ts)
println("\nBurgers R(t), derived rather than transcribed:")
println("  ", Rb)

# ─── §4 Where SymPy stops, and why the quadratures exist ─────────────────────

banner("§4  How far the symbolic route goes")

SymPy.@syms Vsb::positive ps::positive αsym::positive E0::positive δs::positive

# (a) A springpot with a *rational* exponent inverts exactly, and agrees with
#     the closed form the model already carries.
sb = ScottBlair(Vsb, Sym(2) // 5)
println("(a) springpot, α = 2/5")
println("    SymPy   : ", inverse_carson(q -> carson_creep(sb, q), ts))
println("    model   : ", creep(sb, ts))
println("    equal   : ",
    sympy.simplify(inverse_carson(q -> carson_creep(sb, q), ts) - creep(sb, ts)) == 0)

# (b) The fractional Zener at α = 1/2 has a genuine closed form, and SymPy
#     finds it: E_{1/2}(-x) = e^{x²} erfc(x).
fz = FractionalZener(Einf, E0, τ, Sym(1) // 2)
println("\n(b) fractional Zener (Cole-Cole), α = 1/2")
println("    R(t) = ", inverse_carson(q -> carson_relaxation(fz, q), ts))

# (c) A *symbolic* exponent is where it stops — and it stops badly, returning
#     `nan` rather than an unevaluated transform. The extension warns.
println("\n(c) springpot with a symbolic exponent α")
println("    R(t) = ", inverse_carson(q -> carson_creep(ScottBlair(Vsb, αsym), q), ts))

# (d) A sum of several fractional powers — Huet-Sayegh, 2S2P1D — is left
#     unevaluated, which is at least honest.
hs2 = HuetSayegh(Einf, E0, δs, τ, Sym(1) // 5, Sym(2) // 3)
println("\n(d) Huet-Sayegh with rational exponents")
r_hs = inverse_carson(q -> carson_relaxation(hs2, q), ts)
println("    R(t) = ", first(string(r_hs), 150), "…")

println("""

So the symbolic route reaches: rational powers, and the α = 1/2 Cole-Cole
special case. It does **not** reach a symbolic exponent, nor a sum of several
fractional powers — those inverses are Mittag-Leffler functions or worse.

That is not a gap in this package: it is exactly why `inverse_carson` with a
numeric time and one of the four quadratures exists. `FixedTalbot(24)` inverts
every one of these transforms to about 1e-12.""")

# ─── §5 A cross-check the numerical route cannot give itself ─────────────────

banner("§5  Symbolic and numerical inversion, on the same model")

zn = zener_maxwell(2.0, 3.0, 1.5)
@printf("%8s  %18s  %18s  %10s\n", "t", "symbolic R(t)", "FixedTalbot(24)", "rel. diff")
for tv in (0.2, 1.0, 5.0)
    sym = Float64(R_t.subs(Dict(Einf => 2, E1 => 3, τ => Sym(3) / 2, ts => Sym(tv))))
    num = inverse_carson(p -> carson_relaxation(zn, p), tv)
    @printf("%8.2f  %18.12f  %18.12f  %10.1e\n", tv, sym, num, abs(sym - num) / sym)
end

println("\nDone.")
