# [The strength of a porous plastic solid](@id mp-ex-strength)

This example obtains the macroscopic strength domain of a porous solid by
running the [secant homogenization](@ref mp-theory-strength) of a phase law. It
compares the result with the closed-form ellipse of Mori–Tanaka and with the
construction of the tutorial [From derivatives to a strength criterion](@ref tut-strength-criteria).

## The porous solid

The solid fills 85 % of the volume; the pores are spheres. The solid bounds the
norm of its stress deviator by ``\sigma_o``, its strength in units of which every
stress below is given:

```@example mp_ex_strength
using MeanFieldHomogenization, TensND, Plots

φ, σo = 0.15, 1.0
rve = RVE()
add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1e-12, 1e-12)); fraction = φ)
solid(m) = PowerLawElasticity(; k = 1e6, σ₀ = sqrt(3 / 2) * σo, ε₀ = 1.0, m)
nothing # hide
```

| code | theory |
|:--|:--|
| `:P` | the pores, a phase without a law: a linear stiffness close to zero |
| `solid(m)` | the solid: a power law of exponent `m`, rigid–perfectly plastic for `m = 0` |
| `σ₀ = sqrt(3 / 2) * σo` | the von Mises stress of the law, for a bound ``\sigma_o`` on ``\sigma_d`` |
| `k = 1e6` | a bulk modulus large against the secant shear modulus: plastic flow conserves the volume |

## From hardening to flow

Under a deviatoric macroscopic strain, a hardening solid (``m > 0``) makes the
composite harden, and the rigid–plastic solid (``m = 0``) makes it flow at a
constant stress:

```@example mp_ex_strength
dev = LoadingPath(range(0, 1; length = 21); ε11 = t -> 2.0 * t, ε22 = t -> -t, ε33 = t -> -t,
    ε23 = 0, ε13 = 0, ε12 = 0)
p = plot(; xlabel = "macroscopic strain E₁₁", ylabel = "Σ_d / σo", legend = :bottomright)
for m in (0.2, 0.05, 0.0)
    r = point_response(NonlinearHomogenized(rve, Dict(:S => solid(m))), dev; algorithm = LocalNewton(; reltol = 1e-8))
    Σd = [sqrt(2 / 3) * abs(r[:σ11][i] - r[:σ22][i]) for i in eachindex(r.t)]
    plot!(p, r[:ε11][2:end], Σd[2:end] / σo; lw = 2, label = "m = $m")
end
p
```

The strains are large because the solid is rigid until it flows: in this limit
only the direction of the strain matters, not its size. The tolerance of the
local Newton iteration is looser than the default: as the strain grows, the
secant shear modulus of the solid falls against its bulk modulus, and the
fixed point loses accuracy in proportion (see the
[numerical notes](@ref mp-theory-strength)).

## The strength domain

Each direction of the macroscopic strain, from pure expansion to pure
contraction through pure shear, gives a point of the boundary of the domain:

```@example mp_ex_strength
𝟏 = Mandel{Float64}(1, 1, 1, 0, 0, 0)
shear = Mandel{Float64}(1, -0.5, -0.5, 0, 0, 0)
mean_dev(σ) = (m = sum(σ[1:3]) / 3; (m, sqrt(sum(abs2, σ - m * 𝟏))))
function domain(h; algorithm = LocalNewton(; reltol = 1e-8))   # see the numerical notes
    pts = map(range(-π / 2 + 0.01, π / 2 - 0.01; length = 41)) do θ
        E = sin(θ) * 𝟏 / 3 + cos(θ) * shear
        mean_dev(point_update(h, E, initial_state(h), 0.0; algorithm)[1])
    end
    return first.(pts), last.(pts)
end
Σm, Σd = domain(NonlinearHomogenized(rve, Dict(:S => solid(0.0))))
θ = range(0, π; length = 200)
a, b = (1 - φ) * σo / sqrt(1.5φ), (1 - φ) * σo / sqrt(1 + 2φ / 3)      # semi-axes of the ellipse
plot(a * cos.(θ), b * sin.(θ); lw = 2, label = "closed-form ellipse (Mori–Tanaka)",
    xlabel = "Σ_m / σo", ylabel = "Σ_d / σo", aspect_ratio = :equal, legend = :outerright, size = (760, 360))
scatter!(Σm, Σd; ms = 3, label = "NonlinearHomogenized, m = 0")
```

The points of the phase law fall on the ellipse. The mean stress reaches far
larger values than the deviatoric one: a hydrostatic stress can only make the
porous solid flow by closing or opening its pores.

## Another scheme, another shape of pores

With flat spheroidal pores and solid grains of all orientations, and the
self-consistent scheme, the domain shrinks sharply. The construction of the
tutorial, from the derivatives of the linear moduli, gives the ellipse to
compare with:

```@example mp_ex_strength
ω = 0.1
scheme = SelfConsistent(; abstol = 1e-12, maxiters = 500)
function grains(μ_s::T) where {T}
    r = RVE(; T = T, distribution_shape = Ellipsoid(1.0))
    add_phase!(r, :S, Spheroid(ω), Dict(:C => iso_stiffness(convert(T, 1e6), μ_s)); fraction = :rest, symmetrize = IsoSymmetrize())
    add_phase!(r, :P, Spheroid(ω), Dict(:C => iso_stiffness(convert(T, 1e-12), convert(T, 1e-12))); fraction = convert(T, φ), symmetrize = IsoSymmetrize())
    return r
end
using ForwardDiff
kμ(μ) = [k_mu(best_fit_iso(homogenize(grains(μ), scheme, :C)))...]
k, μ = kμ(1.0)
dk, dμ = ForwardDiff.derivative(kμ, 1.0)
A, B = dk / k^2, dμ / μ^2
a_sc, b_sc = sqrt((1 - φ) / (2A)) * σo, sqrt((1 - φ) / B) * σo
Σm_sc, Σd_sc = domain(NonlinearHomogenized(grains(1.0), Dict(:S => solid(0.0)); scheme); algorithm = LocalNewton(; reltol = 1e-9))
plot(a_sc * cos.(θ), b_sc * sin.(θ); lw = 2, label = "derivative construction",
    xlabel = "Σ_m / σo", ylabel = "Σ_d / σo", aspect_ratio = :equal, legend = :outerright, size = (760, 360))
scatter!(Σm_sc, Σd_sc; ms = 3, label = "NonlinearHomogenized, m = 0")
```

The two routes agree here too. The tolerance of the local Newton iteration is
looser still, so as to stay above that of the self-consistent fixed point it
contains.

## Going further

- A hardening solid (``m > 0``), or any law of the strain alone, replaces the
  rigid–plastic one with no other change, and the composite runs at the
  quadrature points of a [finite-element computation](@ref fe-quickstart).
- The solid can carry a second phase, rigid or elastic, given as a phase
  without a law: the domain then accounts for it through the scheme.
