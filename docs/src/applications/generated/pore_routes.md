```@meta
EditURL = "../../../../scripts/77_supersphere_pore_routes.jl"
```

# [A concave pore through every route](@id app-pore-routes)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/77_supersphere_pore_routes.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/77_supersphere_pore_routes.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/77_supersphere_pore_routes.jl`, runs it end to end.

!!! info "Before this page"
    [Choosing an inclusion](@ref man-choosing-inclusion), which compares the
    routes on paper, and
    [Concave pores: superspheres and superspheroids](@ref app-concave-pores),
    which details the finite-element cell used here.

A superspherical pore has no closed form. This page computes its compliance
contribution by the four routes the package offers for such a shape: a
published table entered as a custom inclusion, a finite-element cell, a
network trained on that cell, and the integral equation solved on a grid of
Gaussians. The published values are those of
[Chen, Sevostianov, Giraud and Grgic](@cite chenIJSS2018). The sphere, where
the answer is exact, measures the error of each route; the concave shapes
show where the routes part.

````@example pore_routes
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf
using Plots
gr()
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl")) # hide
nothing # hide
````

## The pore and the quantity compared

The supersphere of semi-axis ``a`` and exponent ``p`` is the body

```math
\left|\frac{x_1}{a}\right|^{2p} + \left|\frac{x_2}{a}\right|^{2p}
  + \left|\frac{x_3}{a}\right|^{2p} \le 1 .
```

It is the sphere for ``p = 1`` and the regular octahedron for ``p = 1/2``; it
is concave below ``1/2`` and tends to the cube as ``p \to \infty``.

````@example pore_routes
X, Y, Z = supersphere_surface(2 * 0.4, 1.0, 1.0, 1.0)      # hide
plotly_scene([surface_trace(X, Y, Z; color = "#f0ad4e", opacity = 0.9)]; uid = "routes-supersphere", height = 420, title = "Supersphere, a = 1, p = 0.4: concave, with conical points on the axes") # hide
````

A pore of volume ``V_p`` in a matrix of compliance ``\mathbb S_0``, under the
remote stress ``\boldsymbol\Sigma``, adds to the mean strain of a volume ``V``
the term

```math
\Delta\boldsymbol{\varepsilon} = \frac{V_p}{V}\,\mathbb H : \boldsymbol\Sigma ,
```

which defines its compliance contribution tensor ``\mathbb H``
([Concentration and contribution tensors](@ref th-concentration)). A
supersphere has the symmetry of the cube, so ``\mathbb H`` has three
independent components in the axes of the pore: ``H_{1111}``, ``H_{1122}``
and ``H_{1212}``. All the values of this page are ``E_0\,\mathbb H``, without
dimension, for a matrix of Young modulus ``E_0`` and Poisson ratio
``\nu_0 = 0.3``.

````@example pore_routes
const ν₀ = 0.3
const C₀ = iso_stiffness_E_nu(1.0, ν₀)                 ## E₀ = 1: the tensors below are E₀ ℍ
cubic(H) = (H[1, 1, 1, 1], H[1, 1, 2, 2], H[1, 2, 1, 2])
contribution(pore) = cubic(canonical_components(inverse_property_contribution(pore, 0 * C₀, C₀)))
nothing #hide
````

For the sphere the three components are known in closed form, through the
Hill tensor of the [`Ellipsoid`](@ref):

````@example pore_routes
exact_sphere = contribution(Ellipsoid(1.0))
````

## Route 1: published values, as a custom inclusion

The article tabulates the six components of ``E_0\,\mathbb H`` of the pore
``|x_1|^{2p} + |x_2|^{2p} + |x_3/\gamma|^{2p} \le 1`` for sixteen values of
``p`` and eight aspect ratios ``\gamma``, computed by finite elements with
``\nu_0 = 0.3``. The column ``\gamma = 1`` is the supersphere.

````@example pore_routes
literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
const chen = literature("chenIJSS2018")
published(name, p) = only(r for r in chen["tables"][name] if r["p"] == p)["gamma_1"]
published(p) = (published("H1111", p), published("H1122", p), published("H1212", p))
nothing #hide

Main.show_literature("chenIJSS2018") # hide
````

A table of contribution tensors is the level C of the
[custom-inclusion interface](@ref man-custom-inclusions): the two contribution
tensors are supplied, and nothing else. The stiffness contribution follows
from the compliance one by ``\mathbb N = -\mathbb C_0 : \mathbb H : \mathbb C_0``.
The published values are for ``\nu_0 = 0.3`` and scale as ``1/E_0``, which the
callbacks check.

````@example pore_routes
function published_pore(p)
    H₁ = cubic_from_constants(published(p)..., CanonicalBasis{3, Float64}())        ## E₀ ℍ
    E_of(P₀) = first(E_nu(P₀))
    function check(P₀)
        isapprox(last(E_nu(P₀)), ν₀; atol = 1.0e-12) ||
            throw(ArgumentError("the published values hold for a Poisson ratio of $ν₀"))
        return nothing
    end
    return CustomInclusion(;
        inverse_property_contribution = (P₁, P₀; kw...) -> (check(P₀); H₁ / E_of(P₀)),
        property_contribution = (P₁, P₀; kw...) -> (check(P₀); -(P₀ ⊡ (H₁ / E_of(P₀)) ⊡ P₀)),
    )
end
nothing #hide
````

Such a pore enters a cell like any other inclusion. Level C does not give the
concentration tensor, so the schemes that need it (`MoriTanaka`, the
self-consistent ones) refuse the phase; the dilute estimate, `Maxwell`, the
scheme of Ponte Castañeda and Willis and the differential scheme accept it.

````@example pore_routes
function porous(pore, f)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))        ## an isotropic distribution, read by Maxwell
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :pores, pore, Dict(:C => C₀); fraction = f, symmetrize = IsoSymmetrize())
    return rve
end
young(C) = first(E_nu(C))
[young(homogenize(porous(published_pore(p), 0.1), Maxwell(), :C)) for p in (0.4, 1.0)]
````

With randomly oriented pores at a porosity of 0.1, the Young modulus falls
further for the concave pore than for the spherical one.

## Route 2: a finite-element cell

[`FESupershapePore`](@ref MeanFieldHomogenization.FESupershapePore) solves one
pore in a finite cell with a corrected boundary condition
([Finite-element inclusions](@ref man-fe-inclusions)). The documentation build
does not mesh: the values below were computed once by
`scripts/fe/make_cell_figures.jl convergence`, on one octant of a cell of
radius ``6a``, for three levels of subdivision of the surface of the pore.

````@example pore_routes
const fe = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "docs", "src", "assets", "fe", "supersphere_convergence.json"))
fe_value(p, level) = only((r["H1111"], r["H1122"], r["H1212"]) for r in fe["rows"] if r["p"] == p && r["level"] == level)
fe_dofs(p, level) = only(r["dofs"] for r in fe["rows"] if r["p"] == p && r["level"] == level)
nothing #hide
````

The sphere is the control: the cell tends to the exact value as the surface
is refined.

````@example pore_routes
for level in (2, 3, 4)
    @printf "level %d  %6d unknowns   H1111 = %.4f   exact %.4f   error %+.2f %%\n" level fe_dofs(1.0, level) fe_value(1.0, level)[1] exact_sphere[1] 100 * (fe_value(1.0, level)[1] / exact_sphere[1] - 1)
end
````

A concave pore converges more slowly. Its surface has conical points on the
axes, where the field is singular, and each level still moves the result:

````@example pore_routes
plt = plot(; xlabel = "level of subdivision of the pore surface", ylabel = "E₀ H₁₁₁₁",
    title = "Superspherical pore, ν₀ = 0.3: finite-element cell and published value",
    titlefontsize = 9, legend = :outerright, size = (760, 360), xticks = 2:4)
for (i, p) in enumerate((0.4, 0.5, 0.7, 1.0))
    plot!(plt, 2:4, [fe_value(p, l)[1] for l in 2:4]; marker = :circle, color = i, label = "cell, p = $p")
    hline!(plt, [published(p)[1]]; color = i, linestyle = :dash, label = "published, p = $p")
end
plt
````

At ``p = 0.4`` the last refinement raises ``H_{1111}`` by 7 %, and the value
of the finest mesh is 2.4 % below the published one; at ``p = 0.7`` the two
differ by 0.4 %, and for a convex pore by 0.1 %.

## Route 3: a network trained on the cell

A shipped network returns the strain concentration of the pore as a function
of ``p`` and ``\nu_0`` ([The models shipped today](@ref man-neural-models)).
The same type receives it in place of a mesh; its evaluation needs no
extension and costs nothing.

````@example pore_routes
const network = load_surrogate(model_path("supershape_pore_elastic"))
network_pore(p) = FESupershapePore(Supersphere(1.0, p); elastic = network)
contribution(network_pore(1.0)), exact_sphere
````

The network was trained on cells of the fourth level, in a cell of radius
``4a``: it follows the cell, with its error in the concave range.

## Route 4: a grid of Gaussians

[`GaussianInclusion`](@ref) solves the integral equation of the heterogeneity
on a regular grid of `n` nodes per side, with no mesh
([Heterogeneities of any shape](@ref man-gaussian-inclusions)). A pore is
entered as a heterogeneity ``10^{4}`` times softer than the matrix.

````@example pore_routes
const solved = Dict{Tuple{Float64, Int}, NTuple{3, Float64}}()        ## each grid is solved once
gaussian(p; n) = get!(solved, (p, n)) do
    cubic(canonical_components(inverse_property_contribution(GaussianInclusion(Supersphere(1.0, p); n), 1.0e-4 * C₀, C₀)))
end
gaussian(1.0; n = 16), exact_sphere
````

## The four routes side by side

``E_0 H_{1111}`` along the family, from the concave pore to the cube-like
one. The network is drawn as a curve over its domain of training.

````@example pore_routes
ps_table = (0.4, 0.5, 0.7, 1.0, 1.5, 2.5)
ps_curve = range(0.36, 2.5; length = 60)
plt = plot(ps_curve, [contribution(network_pore(p))[1] for p in ps_curve]; label = "network",
    xlabel = "p", ylabel = "E₀ H₁₁₁₁", xscale = :log10, xticks = (collect(ps_table), string.(ps_table)),
    title = "Superspherical pore, ν₀ = 0.3: the routes compared", titlefontsize = 9, size = (760, 380))
scatter!(plt, collect(ps_table), [published(p)[1] for p in ps_table]; marker = :diamond, label = "published (Chen et al. 2018)")
scatter!(plt, collect(ps_table), [fe_value(p, 4)[1] for p in ps_table]; marker = :circle, label = "cell, level 4")
scatter!(plt, collect(ps_table), [gaussian(p; n = 16)[1] for p in ps_table]; marker = :utriangle, label = "Gaussians, n = 16")
scatter!(plt, [1.0], [exact_sphere[1]]; marker = :star5, markersize = 8, label = "exact (sphere)")
plt
````

The deviations from the published values, in percent, on the three
components:

````@example pore_routes
deviation(x, ref) = 100 * (x / ref - 1)
println("   p    route               H1111    H1122    H1212")
for p in ps_table
    ref = published(p)
    for (name, val) in (("cell, level 4", fe_value(p, 4)), ("network", contribution(network_pore(p))), ("Gaussians, n = 16", gaussian(p; n = 16)))
        @printf "%5.2f   %-18s %+7.2f  %+7.2f  %+7.2f\n" p name deviation.(val, ref)...
    end
end
````

Three observations.

- **For a convex pore the routes agree.** From the sphere to ``p = 2.5`` the
  cell is within 0.2 % of the published values and the network within 0.3 %.
  The grid of Gaussians is within 4 %.
- **A concave pore separates them.** At ``p = 0.4`` the finest cell is 2 %
  below the published value and still rising with the refinement, and the
  network trained on such cells is at the same distance; the grid of 16 nodes
  is 32 % below.
- **The published table is itself a computation.** Its authors report, for
  the spheroid, an error below 1 % except at the two flattest shapes. It is
  the reference of this page because it is the finest of the four meshes, not
  because it is exact.

The grid of Gaussians, refined on the most concave pore, comes closer
slowly: the arms of the pore are thinner than a few nodes, and the pore is
empty, the stiffest contrast for this method.

````@example pore_routes
for n in (16, 24)
    @printf "n = %2d   H1111 = %.4f   published %.4f   deviation %+.1f %%\n" n gaussian(0.4; n)[1] published(0.4)[1] deviation(gaussian(0.4; n)[1], published(0.4)[1])
end
````

## Which route, then

| Need | Route | What it costs |
| :--- | :--- | :--- |
| a shape and a matrix that a published table covers | the table, as a custom inclusion at level C | nothing; limited to the tabulated cases and to the schemes that level C serves |
| another exponent, another Poisson ratio, in a single computation | the finite-element cell | a mesh and a factorization; refine until the result settles |
| the same family evaluated many times, or differentiated with respect to ``p`` | the network | nothing at evaluation; its error is that of the cells it was trained on |
| a shape outside any family, without a mesh | the grid of Gaussians | seconds; a few percent for a convex shape, not suited to a strongly concave pore at these grids |

## Where to go next

- [Choosing an inclusion](@ref man-choosing-inclusion): the same comparison
  for every inclusion of the package.
- [Custom inclusions](@ref man-custom-inclusions): the three levels of entry.
- [Concave pores: superspheres and superspheroids](@ref app-concave-pores):
  the cell, its octant and the axisymmetric family.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

