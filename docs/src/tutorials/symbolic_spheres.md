# [Symbolic spheres: closed forms with SymPy and Symbolics.jl](@id tut-symbolic-spheres)

`TensND` is generic in its element type: the same tensor algebra (`⊡`, `inv`,
projectors, …) runs on `Float64`, on **SymPy.jl** `Sym` and on **Symbolics.jl**
`Num`. The classical homogenization formulas therefore come out **in closed
form**, with numbers substituted only at the end.

Running example: a single sphere in an isotropic matrix, in three regimes —
general inclusion, **porous** limit (``k_i,\mu_i \to 0``) and **rigid** limit
(``k_i,\mu_i \to \infty``). Terser script:
[`scripts/29_symbolic_schemes.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/29_symbolic_schemes.jl).

## Setup

```@example tutsymsph
using MeanFieldHomogenization
using TensND
using SymPy
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

@syms k0::positive μ0::positive ki::positive μi::positive f::positive ν0::real

C0 = iso_stiffness(k0, μ0)     # matrix stiffness   = TensISO{3}(3k₀, 2μ₀)
Ci = iso_stiffness(ki, μi)     # inclusion stiffness = TensISO{3}(3kᵢ, 2μᵢ)

# The Hill tensor of a sphere does not depend on its radius, so a plain
# Float64 unit sphere already gives a fully symbolic P once C0 is symbolic.
sphere = Ellipsoid(1.0)
nothing # hide
```

`@syms x::positive` both creates the SymPy symbol `x` and tells the solver
it is a positive real — this alone rules out a lot of spurious branches
later when `solve` and `simplify` have to pick between them.

## The Hill tensor and the Eshelby tensor, in closed form

[`hill_tensor`](@ref) and the double contraction `⊡` work unchanged on
symbolic stiffnesses:

```@example tutsymsph
P = hill_tensor(sphere, C0)
S = P ⊡ C0

αP, βP = get_data(P)
αS, βS = get_data(S)
simplify(αP), simplify(βP)
```

Every isotropic 4th-order tensor decomposes as ``\mathbb A = \alpha\,\mathbb
J + \beta\,\mathbb K`` on the spherical (``\mathbb J``) and deviatoric
(``\mathbb K``) projectors, and `TensND.get_data` returns exactly that pair
`(α, β)` — no further rescaling. For the sphere:

```math
\mathbb P = \frac{1}{3k_0+4\mu_0}\,\mathbb J
          + \frac{3(k_0+2\mu_0)}{5\mu_0(3k_0+4\mu_0)}\,\mathbb K,
\qquad
\mathbb S = \mathbb P:\mathbb C_0
          = \frac{3k_0}{3k_0+4\mu_0}\,\mathbb J
          + \frac{6(k_0+2\mu_0)}{5(3k_0+4\mu_0)}\,\mathbb K.
```

Substituting the isotropic identity ``k_0 = \dfrac{2\mu_0(1+\nu_0)}{3(1-2\nu_0)}``
recovers the two Eshelby (1957) eigenvalues everyone eventually memorizes:

```@example tutsymsph
k0_of_ν0 = 2 * μ0 * (1 + ν0) / (3 * (1 - 2 * ν0))
SJ = simplify(subs(αS, k0 => k0_of_ν0))
SK = simplify(subs(βS, k0 => k0_of_ν0))
SJ, SK
```

```math
S_{\mathbb J} = \frac{1+\nu_0}{3(1-\nu_0)},
\qquad
S_{\mathbb K} = \frac{2(4-5\nu_0)}{15(1-\nu_0)}.
```

## Dilute estimate — writing the master formula out by hand

The dilute strain concentration tensor and effective stiffness,

```math
\mathbb A_{\text{dil}} = \big(\mathbb I + \mathbb P:(\mathbb C_i-\mathbb C_0)\big)^{-1},
\qquad
\mathbb C_{\text{dil}} = \mathbb C_0 + f\,(\mathbb C_i-\mathbb C_0):\mathbb A_{\text{dil}},
```

translate directly into TensND tensor algebra — `inv` on a `TensISO` is just
the reciprocal of its two `(α, β)` scalars, so the whole computation stays
in closed form:

```@example tutsymsph
𝕀4 = tens_Id4(Val(3), Val(Sym))
Adil = inv(𝕀4 + P ⊡ (Ci - C0))
Cdil = C0 + f * ((Ci - C0) ⊡ Adil)

k_dil, μ_dil = k_mu(Cdil)
k_dil = simplify(k_dil)
μ_dil = simplify(μ_dil)
k_dil
```

To confirm this by-hand derivation against the package's own scheme API, the
ordinary `RVE(:M)` is enough — a symbolic fraction is stored as it comes,
without conversion:

```@example tutsymsph
rve = RVE(:M)
add_matrix!(rve, sphere, Dict(:C => C0))
add_phase!(rve, :I, sphere, Dict(:C => Ci); fraction = f)

kD, μD = k_mu(homogenize(rve, Dilute(), :C))
simplify(k_dil - kD), simplify(μ_dil - μD)
```

Both differences collapse to `0` — the hand-derived formula and
[`Dilute`](@ref)'s internal machinery agree exactly, symbolically.

## Mori–Tanaka estimate

[`MoriTanaka`](@ref) runs through the same symbolic-safe path (it only adds
a volume-weighted average of concentration tensors, still pure `TensISO`
algebra):

```@example tutsymsph
kMT, μMT = k_mu(homogenize(rve, MoriTanaka(), :C))
kMT = simplify(kMT)
μMT = simplify(μMT)
kMT
```

```math
k_{\text{MT}} = k_0 + \frac{f(k_i-k_0)A_k}{(1-f)+fA_k},
\qquad
A_k = \frac{k_0+4\mu_0/3}{k_i+4\mu_0/3}
```

(``A_k`` is the ``\mathbb J``-part of ``\mathbb A_{\text{dil}}`` above.)

## Two physical limits: porous and rigid

Substituting ``k_i=\mu_i=0`` — a `subs` call — gives the **porous** limit;
taking ``k_i,\mu_i \to \infty`` with SymPy's `limit` and `oo` gives the
**rigid** limit. Both keep the whole expression symbolic in `k₀, μ₀, f`:

```@example tutsymsph
k_dil_por = simplify(subs(k_dil, ki => 0, μi => 0))
kMT_por   = simplify(subs(kMT, ki => 0, μi => 0))
k_dil_por, kMT_por
```

```@example tutsymsph
k_dil_rig = simplify(limit(limit(k_dil, ki => oo), μi => oo))
kMT_rig   = simplify(limit(limit(kMT, ki => oo), μi => oo))
k_dil_rig, kMT_rig
```

`kMT_por` is exactly the Hashin–Shtrikman upper bound for a porous solid:

```math
k_{\text{MT}}^{\text{por}} = \frac{4\mu_0 k_0(1-f)}{4\mu_0+3k_0f},
\qquad
k_{\text{MT}}^{\text{rig}} = k_0 + \frac{f(3k_0+4\mu_0)}{3(1-f)}.
```

## Self-consistent: derived by hand, solved with `solve`

[`SelfConsistent`](@ref) and [`AsymmetricSelfConsistent`](@ref) are
intrinsically **numerical** in `MeanFieldHomogenization` — a damped Picard/Anderson
fixed-point iteration with convergence tests and positive-definiteness
guards, none of which can run on a `Sym` scalar (a comparison like `r <
abstol` simply has no meaning for a symbolic residual). So instead of
calling the API, the self-consistent condition

```math
\mathbb C_{\text{eff}} = \sum_i f_i\,\mathbb C_i:\mathbb A_i(\mathbb C_{\text{eff}})
```

is written out by hand for two isotropic spherical phases. It separates into
two **scalar** equations, coupled through the effective moduli themselves:

```@example tutsymsph
@syms ksc::positive μsc::positive

κstar = 4 * μsc / 3
μstar = μsc * (9 * ksc + 8 * μsc) / (6 * (ksc + 2 * μsc))

eqk = (1 - f) * (k0 - ksc) / (k0 + κstar) + f * (ki - ksc) / (ki + κstar)
eqμ = (1 - f) * (μ0 - μsc) / (μ0 + μstar) + f * (μi - μsc) / (μi + μstar)
nothing # hide
```

The general two-phase system couples `eqk` and `eqμ` through `μstar` — a
coupled polynomial with no compact closed form. Its **limits** do have one.
The porous limit (``k_i=\mu_i=0``), solved with SymPy's `solve`:

```@example tutsymsph
eqk_por = subs(eqk, ki => 0)
eqμ_por = subs(eqμ, μi => 0)
sol_por = solve([eqk_por, eqμ_por], [ksc, μsc])
ksc_por, μsc_por = sol_por[1]
length(sol_por)
```

The load-bearing branch must vanish **exactly** at the percolation threshold
``f=1/2`` for a random sphere assembly, whatever the matrix moduli — the same
branch whose collapse causes the numerical instability that [the
porous-materials tutorial](porous_materials.md) warns about. SymPy's `simplify`
does not collapse the nested `sqrt` left by `solve` at ``f=1/2``, so the check
is done numerically:

```@example tutsymsph
for (k0v, μ0v) in ((30.0, 15.0), (90.0, 30.0), (5.0, 50.0))
    kval = N(subs(ksc_por, k0 => k0v, μ0 => μ0v, f => Sym(1) // 2))
    μval = N(subs(μsc_por, k0 => k0v, μ0 => μ0v, f => Sym(1) // 2))
    println("k₀=$k0v  μ₀=$μ0v  ->  ksc(1/2)=$kval   μsc(1/2)=$μval")
end
```

Both vanish for every `(k₀,μ₀)` tried — the percolation threshold does not
depend on the matrix's own stiffness, only on the geometry (spheres).

For a genuinely numeric cross-check, [`AsymmetricSelfConsistent`](@ref) is
run on a plain `Float64` RVE and compared with the symbolic solution after
substituting numbers:

```@example tutsymsph
k0n, μ0n, fn = 30.0, 15.0, 0.2
rve_num = RVE(:M)
add_matrix!(rve_num, Ellipsoid(1.0), Dict(:C => iso_stiffness(k0n, μ0n)))
add_phase!(rve_num, :V, Ellipsoid(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6)); fraction = fn)
k_sc_num, μ_sc_num = k_mu(homogenize(rve_num, AsymmetricSelfConsistent(; abstol = 1.0e-12, maxiters = 200, select_best = true), :C))

k_sc_sym = N(subs(ksc_por, k0 => k0n, μ0 => μ0n, f => fn))
μ_sc_sym = N(subs(μsc_por, k0 => k0n, μ0 => μ0n, f => fn))

(k_sc_sym, k_sc_num), (μ_sc_sym, μ_sc_num)
```

The two agree to solver tolerance — the hand-derived closed form *is* what
the numerical scheme converges to, it just cannot be produced symbolically
by the iteration itself.

## Bonus: the same computation with Symbolics.jl

`TensND` does not care which symbolic engine produced its scalars. The same
`hill_tensor`/`⊡` code, run on `Symbolics.jl`'s `Num` type instead of
SymPy's `Sym`, gives the same `P` and `S`:

```@example tutsymsph
using Symbolics
Symbolics.@variables k0s μ0s

C0s = iso_stiffness(k0s, μ0s)
Ps = hill_tensor(sphere, C0s)
Ss = Ps ⊡ C0s

αPs, βPs = get_data(Ps)
Symbolics.simplify(αPs)
```

Same formula, different backend — the tensor algebra is agnostic to how the
scalars got their symbolic meaning.

## From formulas back to numbers

To close the loop, substitute a concrete matrix (``k_0=90,\ \mu_0=30``, a
typical cement-paste-like solid) into the three porous closed forms derived
above, and sweep the porosity up to the self-consistent percolation
threshold:

```@example tutsymsph
k_dil_por_num = subs(k_dil_por, k0 => 90.0, μ0 => 30.0)
kMT_por_num   = subs(kMT_por, k0 => 90.0, μ0 => 30.0)
ksc_por_num   = subs(ksc_por, k0 => 90.0, μ0 => 30.0)

fs = range(0.0, 0.48; length = 25)
k_dil_curve = [N(subs(k_dil_por_num, f => fv)) for fv in fs]
kMT_curve   = [N(subs(kMT_por_num, f => fv)) for fv in fs]
ksc_curve   = [N(subs(ksc_por_num, f => fv)) for fv in fs]

plt = plot(;
    xlabel = "porosity f", ylabel = "k_eff",
    legend = :topright, framestyle = :box, size = (760, 480),
    title = "Porous sphere — closed forms evaluated numerically",
    titlefontsize = 10,
)
plot!(plt, fs, k_dil_curve; label = "Dilute (closed form)", color = :blue, lw = 2)
plot!(plt, fs, kMT_curve; label = "Mori–Tanaka (closed form)", color = :green, lw = 2)
plot!(plt, fs, ksc_curve; label = "Self-consistent (closed form)", color = :purple, lw = 2)
plt
```

The self-consistent curve heads to zero as `f → 1/2`, as the percolation
check predicted; dilute and Mori–Tanaka never "see" the pores connecting and
stay positive throughout — the picture of [the porous-materials
tutorial](porous_materials.md), obtained from formulas rather than a sweep.
