# [From Echoes to MeanFieldHomogenization](@id tools-from-echoes)

`MeanFieldHomogenization` (MFH) is a Julia port of **Echoes** [echoes](@cite), the C++
mean-field homogenization library (with a Python interface) developed at
Cerema. If you already have Echoes scripts, this page is a direct
translation guide: the same RVE/scheme/homogenize concepts, a line-by-line
API table, and a worked example that calls **both** implementations —
Echoes live from Julia via [PyCall.jl](https://github.com/JuliaPy/PyCall.jl)
— to check that they agree.

The `@example` blocks run at build time; the Python and PyCall blocks are
shown for reference, and the Echoes numbers below were captured by running
them against Echoes 1.0.

## API translation table

Both libraries share the same mental model: an RVE holding a matrix phase
and inclusion/crack phases, each with a geometry and property dictionary,
homogenized by a named scheme. Only the surface syntax differs.

| Echoes (Python) | MeanFieldHomogenization (Julia) |
|---|---|
| `from echoes import *` | `using MeanFieldHomogenization, TensND` |
| `rve(matrix="SOLID")` | `RVE(:SOLID)` |
| `myrve["SOLID"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO], prop={"C": stiff_kmu(k, μ)})` | `add_matrix!(rve, Spheroid(1.0), Dict(:C => iso_stiffness(k, μ)); symmetrize = IsoSymmetrize())` |
| `myrve["PORE"] = ellipsoid(...)` then `myrve["PORE"].fraction = φ` | `add_phase!(rve, :PORE, Spheroid(1.0), Dict(:C => ...); fraction = φ, symmetrize = IsoSymmetrize())` |
| `crack(shape=spheroidal(ω), density=d, symmetrize=[ISO], prop={"C": tZ4})` | `add_phase!(rve, :CRACK, PennyCrack(1.0), Dict(:C => C0); density = d, symmetrize = IsoSymmetrize())` |
| `stiff_kmu(k, μ)` | `iso_stiffness(k, μ)` |
| `stiff_Enu(E, ν)` | `iso_stiffness_E_nu(E, ν)` |
| `tZ4` (zero 4th-order tensor, for voids/cracks) | `iso_stiffness(1e-6, 1e-6)` (near-zero, kept invertible) |
| `VOIGT`, `REUSS` | [`Voigt`](@ref), [`Reuss`](@ref) |
| `DIL`, `DILD` | [`Dilute`](@ref), [`DiluteDual`](@ref) |
| `MT` | [`MoriTanaka`](@ref) |
| `SC`, `ASC` | [`SelfConsistent`](@ref), [`AsymmetricSelfConsistent`](@ref) |
| `MAX`, `PCW` | [`Maxwell`](@ref), [`PonteCastanedaWillis`](@ref) |
| `DIFF` | [`DifferentialScheme`](@ref) |
| `homogenize(prop="C", rve=myrve, scheme=MT, epsrel=1e-6, maxnb=300, select_best=True)` | `homogenize(rve, MoriTanaka(), :C; abstol = 1e-6, maxiters = 300, select_best = true)` |
| `C.k`, `C.mu` | `k_mu(C)` |
| `np.trace(K.array) / 3.` | `tr(Array(homogenize(rve, scheme, :K))) / 3` |
| `.paramsym(sym=TI)` | `best_fit_ti` |

Two conventions worth flagging for a smooth transition:

- **Both `stiff_kmu`/`iso_stiffness` take *physical* `(k, μ)`.** Where they
  differ is in what the resulting tensor *stores*: MFH's raw
  `TensISO{3}(a, b)` constructor takes the pair `(3k, 2μ)`, not `(k, μ)` —
  see [the first tutorial](../tutorials/first_estimate.md#A-storage-convention-worth-knowing).
  Building with `iso_stiffness(k, μ)` (as in the table above) sidesteps
  this entirely.
- **Symbol-string vs. type-instance schemes.** Echoes selects a scheme via
  a module-level constant (`MT`, `SC`, …) passed to `homogenize`; MFH uses a
  scheme **type instance** (`MoriTanaka()`, `SelfConsistent(; kwargs...)`),
  which is also how solver options (`abstol`, `maxiters`, `select_best`)
  attach directly to the scheme rather than as loose `homogenize` keywords.
  A `Symbol` shortcut (`:mt`, `:sc`, …) is also accepted — see
  [the schemes manual](../manual/schemes.md) for the full alias table.

## Same problem, both sides

The classic porous benchmark — a solid matrix with spherical pores,
porosity ``\varphi \in [0, 1]`` — makes the translation concrete. In
Echoes (the classic porous benchmark from the Echoes book):

```python
from echoes import *

ks, mus = 72., 32.
kp, mup = 1.e-6, 1.e-6

myrve = rve(matrix="SOLID")
myrve["SOLID"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO],
                            prop={"C": stiff_kmu(ks, mus)})
myrve["PORE"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO],
                           prop={"C": stiff_kmu(kp, mup)})

def Chom_porous(phi, scheme):
    myrve["PORE"].fraction = phi
    myrve["SOLID"].fraction = 1. - phi
    C = homogenize(prop="C", rve=myrve, scheme=scheme,
                   epsrel=1.e-6, maxnb=300, select_best=True)
    return max(C.k, 0.), max(C.mu, 0.)
```

The same problem, live, in MFH:

```@example tut-echoes
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

const ks, mus = 72.0, 32.0
const kp, mup = 1.0e-6, 1.0e-6
C_s = iso_stiffness(ks, mus)
C_p = iso_stiffness(kp, mup)

function build_rve(φ)
    r = RVE(:SOLID)
    add_matrix!(r, Spheroid(1.0), Dict(:C => C_s); symmetrize = IsoSymmetrize())
    add_phase!(r, :PORE, Spheroid(1.0), Dict(:C => C_p); fraction = φ, symmetrize = IsoSymmetrize())
    return r
end

φs = collect(0.0:0.1:1.0)
k_mt = [k_mu(homogenize(build_rve(φ), MoriTanaka(), :C))[1] for φ in φs]
k_mt[1:4]
```

Line for line, the two snippets build the same RVE and call the same
scheme — this is the translation to internalize for every other Echoes
script.

## Calling Echoes from Julia with PyCall

To cross-check MFH against a *live* Echoes instead of hand-copied
numbers, [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) can call
Echoes directly from a Julia session. This requires Echoes installed in
some Python environment — any interpreter where `import echoes` succeeds
will do, conda or otherwise:

```julia
# One-time setup: point PyCall at the interpreter where `import echoes`
# succeeds, then rebuild the binding.
ENV["PYTHON"] = raw"/path/to/that/python"
using Pkg
Pkg.build("PyCall")
```

```julia
using PyCall

py"""
from echoes import *

def run_echoes(phi, scheme_name):
    _SCHEMES = {"MT": MT, "SC": SC, "DIFF": DIFF}
    scheme = _SCHEMES[scheme_name]
    myrve = rve(matrix="SOLID")
    myrve["SOLID"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO],
                                prop={"C": stiff_kmu(72., 32.)})
    myrve["PORE"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO],
                               prop={"C": stiff_kmu(1e-6, 1e-6)})
    myrve["PORE"].fraction = phi
    myrve["SOLID"].fraction = 1. - phi
    C = homogenize(prop="C", rve=myrve, scheme=scheme,
                   epsrel=1e-6, maxnb=300, select_best=True)
    return max(C.k, 0.), max(C.mu, 0.)
"""

k_e, μ_e = py"run_echoes"(0.3, "MT")
```

Running this snippet against Echoes 1.0 returns
`(33.46058186790858, 17.626741619570563)` at `φ = 0.3, scheme = "MT"`,
matching the reference table below to solver tolerance. This is the same
pattern used throughout
`scripts/bench_echoes/`
(see [Production cross-checks](@ref from-echoes-where-next)) — a
`py"""..."""` block defining a small helper,
called from Julia like an ordinary function.

## The benchmark result

The table below reproduces the captured Echoes 1.0 output for the porous
benchmark above (`ks, μs = 72, 32`; `φ = 0.0, 0.1, …, 1.0`), obtained
exactly as shown in the previous section. It is embedded here as literal
data — reading it back requires no Echoes installation — so this page
renders identically whether or not Echoes is available:

```@example tut-echoes
# Captured from a run of Echoes 1.0 (see the run_echoes snippet above),
# once per (φ, scheme) pair.
echoes_ref = Dict(
    "MT" => (
        k = [72.0, 55.443851, 43.065421, 33.460582, 25.791046, 19.525425, 14.31056, 9.90258, 6.127661, 2.858562, 1.0e-6],
        μ = [32.0, 26.415585, 21.685158, 17.626742, 14.106633, 11.024391, 8.303101, 5.882864, 3.716342, 1.765626, 1.0e-6],
    ),
    "DIFF" => (
        k = [72.0, 54.60602, 40.616795, 29.414749, 20.528772, 13.593911, 8.323876, 4.491934, 1.917566, 0.457355, 1.0e-6],
        μ = [32.0, 26.168105, 20.864777, 16.11255, 11.933231, 8.347851, 5.376596, 3.038716, 1.352384, 0.334464, 1.0e-6],
    ),
    "SC" => (
        k = [72.0, 53.608383, 37.156454, 22.689248, 10.27169, 0.076007, 6.0e-6, 3.0e-6, 2.0e-6, 1.0e-6, 1.0e-6],
        μ = [32.0, 25.866249, 19.629147, 13.264365, 6.737918, 0.056937, 5.0e-6, 2.0e-6, 2.0e-6, 1.0e-6, 1.0e-6],
    ),
)
nothing # hide
```

MFH, computed live, against those captured references:

```@example tut-echoes
mfh_schemes = Dict(
    "MT" => MoriTanaka(),
    "DIFF" => DifferentialScheme(; nsteps = 300),
    "SC" => SelfConsistent(; abstol = 1.0e-6, maxiters = 300, select_best = true),
)

_relerr(a, b) = abs(b) < 1.0e-9 ? abs(a - b) : abs(a - b) / abs(b)

plt = plot(;
    xlabel = "φ (porosity)", ylabel = "k_hom",
    legend = :topright, framestyle = :box, size = (760, 480),
)
for (name, color) in (("MT", :black), ("DIFF", :orange), ("SC", :red))
    k_jl = [max(k_mu(best_fit_iso(homogenize(build_rve(φ), mfh_schemes[name], :C)))[1], 0.0) for φ in φs]
    plot!(plt, φs, k_jl; label = "MFH $name", color = color, lw = 2)
    scatter!(plt, φs, echoes_ref[name].k; label = "Echoes $name", color = color, marker = :circle, markersize = 4)
end
plt
```

```@example tut-echoes
for name in ("MT", "DIFF", "SC")
    scheme = mfh_schemes[name]
    println("== $name ==")
    for (i, φ) in enumerate(φs)
        k_jl, _ = k_mu(best_fit_iso(homogenize(build_rve(φ), scheme, :C)))
        k_jl = max(k_jl, 0.0)
        k_e = echoes_ref[name].k[i]
        println("  φ=", round(φ, digits = 2), "  k_MFH=", round(k_jl, digits = 6),
                "  k_Echoes=", k_e, "  relerr=", round(_relerr(k_jl, k_e), sigdigits = 3))
    end
end
```

| Scheme | Agreement | Source of the residual |
| :--- | :--- | :--- |
| `Mori-Tanaka` | ``\sim 10^{-8}`` | floating-point floor; closed form on both sides |
| `Differential` | sub-percent to ``\varphi \approx 0.8``, a few percent beyond | independent step counts and quadrature on the same ODE |
| `SelfConsistent` | close to ``\varphi \approx 0.4`` | past percolation both sides sit at solver tolerance (``\sim 10^{-6}``), so the relative error stops being meaningful |

## [Production cross-checks](@id from-echoes-where-next)

`scripts/bench_echoes/` runs the same PyCall pattern systematically; tolerances
and measured agreement are in [Cross-validation](../developer/validation.md),
timings in [Performance vs Echoes](../developer/benchmarks.md).

Two tutorials carry a captured Echoes reference in the page itself, in the
literal-data style used above, and are worth reading next because each one turns
up a place where the two libraries do *not* line up trivially:

| Page | What the cross-check shows |
| :--- | :--- |
| [Crack distributions: isotropic or parallel](@ref tut-crack-distributions) | `MoriTanaka` and `SelfConsistent` reproduce Echoes to ``\sim 10^{-7}`` on both an isotropic and a single-orientation crack population — but `AsymmetricSelfConsistent` is a *different* fixed point, not the translation of Echoes' `SC`, and it percolates at about half the crack density |
| [Ageing creep: loading age against inclusion shape](@ref tut-ageing-ages-aspect) | the ageing-viscoelastic counterpart of the table above: `ViscoLaw(J, :creep)` against Echoes' `visco_prop={"C": (Js, CREEP)}`, swept over loading ages and aspect ratios |

Without Echoes or PyCall, each cross-check has a PyCall-free counterpart:
committed `*_python.json` dumps, and pure-Julia transcriptions such as
`scripts/55_ageing_creep_dirichlet_chains.jl`.
