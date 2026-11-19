```@meta
EditURL = "../../../../scripts/75_crack_contact_islands.jl"
```

# [Cracks with contacts between their faces](@id app-crack-contacts)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/75_crack_contact_islands.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/75_crack_contact_islands.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/75_crack_contact_islands.jl`, runs it end to end.

!!! info "Before this page"
    The manual page [Planar cracks of any contour](@ref man-planar-cracks), and
    the theory page of the [same name](@ref th-gaussian-cracks), §5 and §6, for
    the compliance–resistance connection and for contacts.

The faces of a real crack often touch on small areas where they stay bonded.
Such contacts stiffen the crack far more than their area suggests, and a solid
whose cracks have contacts behaves as if its crack density were lower. This
page follows the study of [Markov, Kanaun and Kachanov](@cite markov2025) of a
circular crack of radius ``a`` with circular contacts of radius ``l``: one
contact, centered then shifted; many contacts on a regular pattern; contacts
gathered in a cluster; contacts placed at random. Each configuration is one
region, the disk without the contacts.

The curves of the paper are published as images, from which values cannot be
read back with precision. The comparisons below are with what the paper
states in words and numbers, and, for a centered contact, with the annular
crack of the package, computed by an independent method.

````@example crack_contacts
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Random
using Plots
gr()

# Interactive 3-D helpers; `pkgdir` rather than `@__DIR__` so the include also
# resolves inside a Documenter `@example` block.
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
````

The matrix is isotropic, of Young modulus ``E``, Poisson ratio ``\nu``, shear
modulus ``\mu`` and conductivity ``k_0``. The crack has the radius ``a = 1``.

````@example crack_contacts
const E, ν, k₀ = 1.0, 0.3, 1.0
const μ = E / (2 * (1 + ν))
const C₀ = iso_stiffness_E_nu(E, ν)
const K₀ = TensISO{3}(k₀)
nothing #hide
````

## The crack, and what measures the effect of its contacts

A crack with contacts is the disk without the disks of the contacts. `n` is
the number of nodes across the crack; a contact needs about three nodes on its
radius to be seen.

````@example crack_contacts
contacts(centers, l) = reduce(∪, [Disk(l; center = c) for c in centers])
crack_with(centers, l; n = 128) = PlanarCrack(setdiff(Disk(1.0), contacts(centers, l)); size = 1.0, n)
free = PlanarCrack(Disk(1.0); n = 128)
````

The effect of the contacts is measured by the *reduction factor* of the normal
compliance, ``\lambda = B_{nn}/B_{nn}^{0}``, the superscript 0 standing for the
crack without contact. Since the contribution of a crack to the compliance of
a solid is proportional to ``a^{3}B_{nn}``, a crack with contacts counts as a
free crack of radius ``\tilde a = \lambda^{1/3}a``.

In an isotropic matrix, the normal opening per unit traction and the jump of
potential per unit flux solve the same equation up to a constant
([theory](@ref th-gaussian-cracks), §5), so that
``B_{nn}/b = k_0(1-\nu)/\mu`` for any contour [kachanov2023](@cite): the
reduction factor of the resistance is ``\lambda`` too. The check on a crack
with three contacts:

````@example crack_contacts
three = crack_with([(0.4, 0.1), (-0.3, 0.35), (-0.1, -0.5)], 0.12)
Bnn = get_array(cod_tensor(three, C₀))[3, 3]
b = cod_tensor(three, K₀)
Bnn / b, k₀ * (1 - ν) / μ
````

``\lambda`` is therefore computed below from the conduction problem, which has
one unknown per node where elasticity has three.

````@example crack_contacts
const b₀ = cod_tensor(free, K₀)
λ(crack) = cod_tensor(crack, K₀) / b₀
nothing #hide
````

## One centered contact

The crack is then a ring, the [`AnnularCrack`](@ref) of the package, whose
opening is computed by a Galerkin method on Chebyshev polynomials
([Contributions from intensity factors](@ref th-rice-contributions)). The two
methods share nothing but the integral equation.

````@example crack_contacts
radii = 0.1:0.1:0.8
λ_gauss = [λ(crack_with([(0.0, 0.0)], l)) for l in radii]
λ_ring = [cod_tensor(AnnularCrack(1.0, l), K₀) / cod_tensor(PennyCrack(1.0), K₀) for l in radii]
maximum(abs.(λ_gauss .- λ_ring))
````

````@example crack_contacts
plt = plot(
    radii, λ_ring; lw = 2, label = "λ, annular crack (Galerkin)", xlabel = "contact radius l / a",
    ylabel = "reduction factor", ylims = (0, 1), legend = :topright
)
scatter!(plt, radii, λ_gauss; label = "λ, Gaussian approximation")
plot!(plt, radii, λ_gauss .^ (1 / 3); lw = 2, ls = :dash, label = "equivalent radius ã / a")
plt
````

A contact of a tenth of the radius, one percent of the area, removes an eighth
of the compliance; a contact of half the radius removes two thirds of it.

````@example crack_contacts
@printf("l/a = 0.1: λ = %.3f      l/a = 0.5: λ = %.3f\n", λ_gauss[1], λ_gauss[5])
````

## One shifted contact

A contact at the distance ``c`` of the center makes the crack anisotropic in
its plane: the three principal values of ``\boldsymbol{B}`` (sliding along the
shift, sliding across it, opening) are reduced by three factors
``\beta_i = B_i/B_i^{0}``, with ``\beta_3 = \lambda``.

````@example crack_contacts
B⁰ = diag(get_array(cod_tensor(free, C₀)))
shifted = [(0.2, 0.5), (0.4, 0.4), (0.6, 0.3), (0.8, 0.15)]
β = [diag(get_array(cod_tensor(crack_with([(c, 0.0)], l; n = 96), C₀))) ./ B⁰ for (l, c) in shifted]
for ((l, c), βᵢ) in zip(shifted, β)
    @printf(
        "l/a = %.1f, c/a = %.2f:  β = (%.3f, %.3f, %.3f)   largest gap %.1f %%\n",
        l, c, βᵢ..., 100 * (maximum(βᵢ) - minimum(βᵢ)) / βᵢ[3]
    )
end
````

The three factors stay close: the paper reports gaps below 6 % even for large
contacts, and below 3 % for ``l/a < 0.6``. The reduction of the compliance by a
contact is nearly the same in opening and in sliding, which is what allows a
single factor ``\lambda`` to be used for both.

## Many contacts on a regular pattern

``N`` contacts on a hexagonal lattice of step ``d``: ``N = 1, 7, 19, 37, 61``
for zero to four rings around the central one. The step is taken as
``d = 2a/(2k+1)`` for ``k`` rings, which spreads the contacts over the crack
(the paper does not give its value).

````@example crack_contacts
hexagon(k, d) = [(d * (i + j / 2), d * j * sqrt(3) / 2) for i in (-k):k for j in (-k):k if abs(i + j) ≤ k]
patterns = [(k, 1 + 3k * (k + 1), 2 / (2k + 1)) for k in 0:4]

panels = map(patterns) do (k, N, d)
    region = setdiff(Disk(1.0), contacts(hexagon(k, d), 0.3 * d))
    heatmap(
        coverage(region, GaussianGrid(region, 100))'; aspect_ratio = :equal, colorbar = false,
        axis = false, color = :grays, title = "N = $N", titlefontsize = 9
    )
end
plot(panels...; layout = (1, 5), size = (780, 180))
````

Two measures of the amount of contact are used: ``p = N l/a``, the sum of the
radii, and ``\tilde p = N l^{2}/a^{2}``, the fraction of the area in contact.
Each pattern is computed for the radii that the grid resolves
(``l \ge 0.05\,a``) and that leave the contacts apart (``2l < 0.9\,d``).

````@example crack_contacts
curves = map(patterns) do (k, N, d)
    ls = range(max(0.05, 0.3 / N), 0.45 * d; length = 6)
    (N, ls, [λ(crack_with(hexagon(k, d), l)) for l in ls])
end

left = plot(; xlabel = "p = N l / a", ylabel = "reduction factor λ", ylims = (0, 1), legend = :topright)
right = plot(; xlabel = "p̃ = N l² / a²", ylims = (0, 1), legend = false)
for (N, ls, λs) in curves
    plot!(left, N .* ls, λs; marker = :circle, label = "N = $N")
    plot!(right, N .* ls .^ 2, λs; marker = :circle)
end
plot(left, right; layout = (1, 2), size = (780, 330))
````

At a given fraction of the area in contact, many small contacts stiffen the
crack much more than a few large ones. The opening of the crack with 61
contacts at ``p = 4.7`` (``\tilde p = 0.36``), the case the paper illustrates,
under a unit normal traction (the jump of potential under a unit flux, times
``k_0(1-\nu)/\mu``):

````@example crack_contacts
k, N, d = patterns[5]
crack61 = crack_with(hexagon(k, d), 4.7 / N; n = 320)      ## 12 nodes on the radius of a contact
n̂ = tens_basis(crack61.basis, 3)
scale = k₀ * (1 - ν) / μ
_, jump = crack_opening(crack61, K₀, n̂)
x, y, Z = opening_surface(crack61, scale .* jump)
every = 1:2:length(x)                                      ## one node out of two is enough to draw
plotly_surface(
    x[every], y[every], Z[every, every]'; title = "Opening of a crack with 61 contacts, p = 4.7",
    zlabel = "normal opening", xlabel = "x / a", ylabel = "y / a", uid = "app-contacts-61",
    colorscale = "Viridis", reversescale = false, aspect = (1, 1, 0.45)
)
````

The faces open between the contacts and, more widely, in the ring that the
contacts leave free along the edge. The section of this surface by the
diameter ``y = 0``, which crosses nine contacts, with the opening of the crack
without contact (dashed):

````@example crack_contacts
row(y) = argmin(abs.(y))                                   ## the line of nodes next to y = 0
_, jump⁰ = crack_opening(free, K₀, n̂)
x⁰, y⁰, Z⁰ = opening_surface(free, scale .* jump⁰)
plt = plot(
    x⁰, Z⁰[:, row(y⁰)]; ls = :dash, color = :gray, label = "without contact",
    xlabel = "x / a", ylabel = "normal opening", legend = :right
)
plot!(plt, x, Z[:, row(y)]; lw = 1.5, label = "61 contacts")
plt
````

````@example crack_contacts
@printf("61 contacts, p = 4.7:  λ = %.4f, equivalent radius ã/a = %.3f\n", λ(crack61), λ(crack61)^(1 / 3))
````

## Contacts gathered in a cluster

Seven contacts, one at the center and six on a circle of radius ``r``, at the
density ``p = 0.84`` (``l = 0.12\,a``). ``r`` goes from contacts that nearly
touch to contacts next to the edge of the crack.

````@example crack_contacts
l₇ = 0.84 / 7
cluster(r) = [(0.0, 0.0); [(r * cos(θ), r * sin(θ)) for θ in (0:5) .* (π / 3)]]
rs = 0.26:0.06:0.86
λ_cluster = [λ(crack_with(cluster(r), l₇)) for r in rs]
plot(
    rs, λ_cluster; marker = :circle, xlabel = "cluster radius r / a", ylabel = "reduction factor λ",
    label = "7 contacts, p = 0.84", ylims = (0, 1)
)
````

The contacts are most efficient when they are spread at mid-radius; gathered
at the center or pushed against the edge, they stiffen the crack less.

````@example crack_contacts
@printf(
    "λ from %.3f (r/a = %.2f) to %.3f (r/a = %.2f): a difference of %.0f %%\n",
    minimum(λ_cluster), rs[argmin(λ_cluster)], maximum(λ_cluster), rs[argmax(λ_cluster)],
    100 * (1 - minimum(λ_cluster) / maximum(λ_cluster))
)
````

The paper reports differences of up to 30 % between spread and gathered
contacts.

**Holm's radius.** The radius of the single centered contact that stiffens the
crack as much as the cluster. It is found on the annular crack, by bisection.

````@example crack_contacts
ring_λ(c) = cod_tensor(AnnularCrack(1.0, c), K₀) / cod_tensor(PennyCrack(1.0), K₀)
function holm_radius(target)
    lo, hi = 1.0e-3, 0.99                       ## λ decreases from 1 to 0 with the radius
    for _ in 1:40
        mid = (lo + hi) / 2
        ring_λ(mid) > target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
holm = holm_radius(first(λ_cluster))
@printf(
    "tightest cluster (r/a = %.2f): Holm's radius %.3f a, radius of the disk that envelops it %.3f a\n",
    first(rs), holm, first(rs) + l₇
)
````

Even when the contacts nearly touch, the cluster stiffens the crack less than
the single contact that envelops it: the crack still opens between them.

## Contacts placed at random

``N`` contacts of equal radius, at the density ``p = 0.84``, drawn at random
without overlap; six draws for each ``N``. A draw that cannot be completed is
started again.

````@example crack_contacts
function random_contacts(rng, N, l; gap = 0.02)
    centers = Tuple{Float64, Float64}[]
    rejected = 0
    while length(centers) < N
        ρ, θ = (1 - l - gap) * sqrt(rand(rng)), 2π * rand(rng)
        c = (ρ * cos(θ), ρ * sin(θ))
        if all(q -> hypot(c[1] - q[1], c[2] - q[2]) > 2l + gap, centers)
            push!(centers, c)
        elseif (rejected += 1) > 10_000                 ## jammed: start again
            empty!(centers)
            rejected = 0
        end
    end
    return centers
end

rng = MersenneTwister(2025)
numbers = (5, 10, 20)
draws = [[λ(crack_with(random_contacts(rng, N, 0.84 / N), 0.84 / N; n = 160)) for _ in 1:6] for N in numbers]
for (N, λs) in zip(numbers, draws)
    @printf(
        "N = %2d:  λ between %.3f and %.3f (spread %.3f), mean %.3f\n",
        N, minimum(λs), maximum(λs), maximum(λs) - minimum(λs), sum(λs) / 6
    )
end
````

With five or ten contacts the reduction depends on where they fall; with
twenty the six draws agree within a few hundredths, and the reduction is
nearly a property of the density alone. The paper finds the same
stabilization at fifteen to twenty contacts. At a given ``p``, the mean
reduction factor also rises with ``N``: the same sum of radii shared between
more, smaller contacts stiffens the crack less.

## A solid with such cracks

In a scheme, the crack with contacts is a crack as any other. Randomly
oriented cracks of density ``\varepsilon = N a^{3} = 0.2``, each with the seven
contacts of the cluster at ``r = 0.5\,a``, against the same cracks without
contact, and against free cracks of the reduced density
``\lambda\,\varepsilon``:

````@example crack_contacts
function young_modulus(crack, ε)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :cracks, crack, Dict(:C => C₀); density = ε, symmetrize = IsoSymmetrize())
    return first(E_nu(best_fit_iso(homogenize(rve, MoriTanaka(), :C))))
end
with_contacts = crack_with(cluster(0.5), l₇; n = 96)
λ₇ = λ(with_contacts)
@printf(
    "free cracks: E/E₀ = %.4f\nwith contacts: E/E₀ = %.4f\nfree cracks of density λε: E/E₀ = %.4f\n",
    young_modulus(PennyCrack(1.0), 0.2), young_modulus(with_contacts, 0.2), young_modulus(PennyCrack(1.0), λ₇ * 0.2)
)
````

The last two values nearly coincide: since the contacts reduce the sliding
compliances as much as the normal one, a solid whose cracks have contacts
behaves as a solid with free cracks of the density ``\lambda\,\varepsilon``.

## Where to go next

- [Planar cracks of any contour](@ref man-planar-cracks): contours other than
  a disk.
- [Cracks from their intensity factors](@ref tut-rice-cracks): the annular
  crack and its intensity factors.

## References

```@bibliography
Pages = ["crack_contacts.md"]
Canonical = false
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

