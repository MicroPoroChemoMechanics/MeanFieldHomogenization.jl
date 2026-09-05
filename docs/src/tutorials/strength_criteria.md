# [From derivatives to a strength criterion](@id tut-strength-criteria)

This capstone tutorial combines the last three pages — porous
materials, scheme comparison, and automatic differentiation — into a
single applied result: a macroscopic **strength criterion** for a
porous solid, derived without ever writing a closed-form strength
formula by hand. It ports the porous strength-criterion benchmark of
the Echoes book [echoes](@cite).

## Setup

The solid and pore are both **oblate spheroids** (aspect ratio
``\omega = 0.1``) with a uniform spatial distribution of orientations
([`IsoSymmetrize`](@ref), as in the
[porous benchmark tutorial](porous_benchmark.md)), so the
homogenized stiffness is isotropic. The solid is much stiffer than the
pore, at a fixed porosity ``\varphi = 0.15``:

```@example tutstrength
using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

const k_s, μ_s0 = 1.0e6, 1.0
const TINY = 1.0e-12
const ω = 0.1
const φ = 0.15
nothing # hide
```

## The criterion

Under a macroscopic stress split into a mean part ``\Sigma_m`` and a
deviatoric magnitude ``\Sigma_d``, a porous solid governed by a von
Mises-type local strength ``\sigma_o`` in the solid phase admits, to
second order in the contrast, an **elliptical macroscopic strength
domain**:

```math
\frac{(\Sigma_m/\sigma_o)^2}{2A} + \frac{(\Sigma_d/\sigma_o)^2}{B} = \frac{1}{1-\varphi},
\qquad
A = \Big(\frac{\mu_s}{k_{\text{hom}}}\Big)^{\!2}\frac{\partial k_{\text{hom}}}{\partial\mu_s},
\qquad
B = \Big(\frac{\mu_s}{\mu_{\text{hom}}}\Big)^{\!2}\frac{\partial \mu_{\text{hom}}}{\partial\mu_s}.
```

``A`` and ``B`` are **not** independent material data — they are the
sensitivities of the two homogenized moduli to the *solid's own* shear
modulus, exactly the kind of derivative built up over the
[previous tutorial](sensitivities.md). Differentiating a closure
that homogenizes and reads back `(k_hom, μ_hom)` gives both at once:

```@example tutstrength
function k_mu_hom(μ_s::T, scheme) where {T}
    r = RVE(; T = T, distribution_shape = Ellipsoid(1.0))
    add_phase!(r, :SOLID, Spheroid(ω), Dict(:C => iso_stiffness(convert(T, k_s), μ_s)); fraction = :rest, symmetrize = IsoSymmetrize())
    add_phase!(r, :PORE, Spheroid(ω), Dict(:C => iso_stiffness(convert(T, TINY), convert(T, TINY)));
               fraction = convert(T, φ), symmetrize = IsoSymmetrize())
    C = homogenize(r, scheme, :C)
    return [k_mu(best_fit_iso(C))...]
end

function ellipse_radii(scheme)
    k_hom, μ_hom = k_mu_hom(μ_s0, scheme)
    dk_dμs, dμ_dμs = ForwardDiff.derivative(μ_s -> k_mu_hom(μ_s, scheme), μ_s0)
    A = (μ_s0 / k_hom)^2 * dk_dμs
    B = (μ_s0 / μ_hom)^2 * dμ_dμs
    a = sqrt((1 - φ) / (2A))   # semi-axis along Σ_m/σ_o
    b = sqrt((1 - φ) / B)      # semi-axis along Σ_d/σ_o
    return a, b
end
nothing # hide
```

`k_mu_hom` uses only public accessors — `iso_stiffness`,
[`homogenize`](@ref), `best_fit_iso`, `k_mu` — the same
building blocks as every earlier tutorial; there is no need to reach
into the package's internals to differentiate through a homogenization
call, whatever scheme is used.

## Strength ellipses for six schemes

```@example tutstrength
SCHEMES = [
    (MoriTanaka(), "MT"),
    (DiluteDual(), "DiluteDual"),
    (SelfConsistent(; abstol = 1.0e-8, maxiters = 300, select_best = true), "SelfConsistent"),
    (AsymmetricSelfConsistent(; abstol = 1.0e-8, maxiters = 300, select_best = true), "AsymmetricSelfConsistent"),
    (PonteCastanedaWillis(), "PCW"),
    (Maxwell(), "Maxwell"),
]

θ = range(0, π; length = 100)
plt = plot(;
    xlabel = "Σ_m / σ_o", ylabel = "Σ_d / σ_o",
    aspect_ratio = :equal, legend = :outerright, framestyle = :box, size = (760, 480),
)
for (scheme, label) in SCHEMES
    a, b = ellipse_radii(scheme)
    plot!(plt, a .* cos.(θ), b .* sin.(θ); lw = 2, label = label)
end
hline!(plt, [0.0]; color = :black, lw = 0.5, label = "")
vline!(plt, [0.0]; color = :black, lw = 0.5, label = "")
plt
```

The ellipses differ noticeably between schemes — `SelfConsistent` and
`AsymmetricSelfConsistent` predict a markedly smaller admissible domain
than `Mori-Tanaka` or `Maxwell` at this porosity, the same qualitative
ranking (percolating vs. isolated-pore topology) seen throughout the
[porous benchmark tutorial](porous_benchmark.md). Mean-field
homogenization gives the moduli; automatic differentiation gives their
sensitivities; together they yield a macroscopic strength criterion —
with no closed-form derivative ever written by hand, and no assumption
beyond the choice of scheme. For the same construction at three nested
scales, see
[Quasi-brittle strength of cement paste and mortar](../applications/strength.md).
