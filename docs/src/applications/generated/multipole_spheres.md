```@meta
EditURL = "../../../../scripts/95_multipole_spheres.jl"
```

# [Arrays and clusters of spheres by the multipole expansion](@id app-mem-spheres)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/95_multipole_spheres.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/95_multipole_spheres.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/95_multipole_spheres.jl`, runs it end to end.

!!! info "Before this page"
    [The multipole expansion method](@ref th-mem) and its
    [elastic counterpart](@ref th-mem-elasticity), where the scheme is derived,
    and [Particle assemblies and N-body schemes](@ref man-assemblies) for the
    assemblies it acts on.

The multipole expansion method resolves the field around every sphere of an
assembly, up to a degree `tmax` of the harmonics. This page computes with
[`MultipoleExpansion`](@ref) the reference results of the method on spheres:
the conductivity of the three cubic arrays tabulated by
[kushch2013](@citet) and [kushch2014](@citet), and the conductivity and the
stiffness of the finite cluster of 27 spheres that serves as the test problem
of the sample codes of [kushch2013](@citet). The published values are read
from the two files below.

````@example multipole_spheres
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Markdown
using Printf
using Plots
gr()

# Interactive 3D helpers (Plotly through require.js).
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

literature(key) = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", key * ".json"))
book = literature("kushch2013")
paper = literature("kushch2014")
nothing #hide

Main.show_literature("kushch2013") # hide
````

````@example multipole_spheres
Main.show_literature("kushch2014") # hide
````

## 1. The simple cubic array

One sphere per cubic cell, of conductivity ``k_1`` in a matrix of
conductivity ``k_0 = 1``. [`cubic_lattice`](@ref) builds the periodic
assembly from the volume fraction, and the scheme returns the conductivity of
the periodic medium, which is isotropic for a cubic array.

````@example multipole_spheres
matrix = Dict(:K => TensISO{3}(1.0))
spheres(k₁) = Dict(:K => TensISO{3}(k₁))
conductivity(asm, tmax) = canonical_components(homogenize(asm, MultipoleExpansion(; tmax), :K))[3, 3]
nothing #hide
````

The table of [kushch2014](@citet) gives the conductivity to three decimals for
five contrasts, the last being perfectly conducting spheres, represented here
by ``k_1 = 10^8``. Each cell below shows the value computed at `tmax = 15`
and, in parentheses, the published one.

````@example multipole_spheres
contrasts = (0.0, 10.0, 100.0, 1000.0, 1.0e8)
columns = ("contrast_0", "contrast_10", "contrast_100", "contrast_1000", "contrast_infinity")
headers = ("0", "10", "100", "1000", "∞")

function comparison(kind, rows; tmax = 15)
    io = IOBuffer()
    println(io, "| ``c`` | ", join(("``k_1/k_0 = $(h)``" for h in headers), " | "), " |")
    println(io, "| :-- | ", join(fill(":--", length(headers)), " | "), " |")
    for row in rows
        cells = map(zip(contrasts, columns)) do (k₁, column)
            k = conductivity(cubic_lattice(kind, matrix, spheres(k₁); fraction = row["c"]), tmax)
            @sprintf("%.3f (%.3f)", k, row[column])
        end
        println(io, "| ", row["c"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end

comparison(:sc, paper["tables"]["conductivity_sc"])
````

The two agree to the last digit up to the fraction 0.45. At the fraction 0.5
and a high contrast the degree 15 is not enough: the spheres almost touch (the
fraction of contact is ``\pi/6 \approx 0.524``), and the field in the gap
between two neighbors needs harmonics of high degree.
The convergence is monotonic, and starts at `tmax = 1` from the estimate of
Maxwell and Mori–Tanaka, which the dipoles alone give exactly:

````@example multipole_spheres
degrees = 1:2:25
p_conv = plot(;
    xlabel = "tmax", ylabel = "k / k₀", legend = :right,
    title = "Simple cubic array, perfectly conducting spheres"
)
for c in (0.3, 0.4, 0.5)
    asm = cubic_lattice(:sc, matrix, spheres(1.0e8); fraction = c)
    plot!(p_conv, degrees, [conductivity(asm, t) for t in degrees]; marker = :circle, ms = 3, label = "c = $c")
    k_mt = canonical_components(homogenize(asm, MoriTanaka(), :K))[1, 1]
    scatter!(p_conv, [1], [k_mt]; marker = :star5, ms = 7, color = :black, label = c == 0.3 ? "Mori–Tanaka" : "")
end
published = last(paper["tables"]["conductivity_sc"])["contrast_infinity"]
hline!(p_conv, [published]; ls = :dash, color = :gray, label = "published, c = 0.5")
p_conv
````

The value printed by the sample code of [kushch2013](@citet) for the fraction
0.5, with five decimals, is reached at `tmax = 25`, the degree that code uses:

````@example multipole_spheres
asm = cubic_lattice(:sc, matrix, spheres(1.0e8); fraction = 0.5)
(computed = conductivity(asm, 25), published = book["quantities"]["periodic_conductivity_sc"]["value"])
````

## 2. Body-centered and face-centered arrays

The cell of the body-centered array holds two spheres and that of the
face-centered array four. Nothing changes in the call. At a given fraction
the three arrays have the same estimate of Mori–Tanaka; the multipole
expansion separates them, the more so as the spheres come close to contact,
which happens at the fractions 0.524, 0.680 and 0.740.

````@example multipole_spheres
fractions(kind) = range(0.05, max_packing_fraction(kind) - 0.02; length = 15)
p_arrays = plot(;
    xlabel = "volume fraction c", ylabel = "k / k₀", legend = :topleft, yscale = :log10,
    title = "Cubic arrays, k₁/k₀ = 100"
)
for (kind, name, color) in ((:sc, "conductivity_sc", 1), (:bcc, "conductivity_bcc", 2), (:fcc, "conductivity_fcc", 3))
    cs = fractions(kind)
    ks = [conductivity(cubic_lattice(kind, matrix, spheres(100.0); fraction = c), 11) for c in cs]
    plot!(p_arrays, cs, ks; color, label = "$(kind), tmax = 11")
    rows = book["tables"][name]
    scatter!(p_arrays, getindex.(rows, "c"), getindex.(rows, "contrast_100"); color, marker = :diamond, label = "$(kind), published")
end
mt(c) = (1 + 2c * 99 / 102) / (1 - c * 99 / 102)
plot!(p_arrays, 0.05:0.01:0.7, mt; color = :black, ls = :dash, label = "Mori–Tanaka")
p_arrays
````

The curves pass through the published points, except at the last fraction of
each array, where the degree 11 is short as in the simple cubic case.

## 3. A finite cluster of 27 spheres

The sample codes of [kushch2013](@citet) take a block of ``3\times3\times3``
equal spheres on a simple cubic lattice, in the infinite matrix. The block is
replaced by the homogeneous ball that holds the same fraction of spheres and
sends the same far field, which is the closure of [Maxwell](@ref th-scheme-maxwell);
[`MixedBC`](@ref) carries that ball.

````@example multipole_spheres
function block(n, c, properties, matrix_properties)
    R = cbrt(3c / (4π))                        # radius of the spheres, for a unit period
    asm = ParticleAssembly(; boundary = MixedBC(R * cbrt(n^3 / c)))
    add_matrix!(asm, matrix_properties)
    for i in 1:n, j in 1:n, k in 1:n
        add_particle!(asm, Symbol(:p, i, j, k), (i, j, k) .- (n + 1) / 2, Ellipsoid(R), properties)
    end
    return asm
end

q(name) = book["quantities"][name]["value"]
cluster = block(3, q("fcm_volume_fraction"), spheres(q("fcm_conductivity_contrast")), matrix)
nothing #hide
````

The spheres and the ball that closes the cluster:

````@example multipole_spheres
function cluster_scene(asm; uid)
    traces = String[]
    for name in particle_names(asm)
        R = particle_geometry(asm, name).semi_axes[1]
        X, Y, Z = ellipsoid_surface(R, R, R; center = particle_center(asm, name), nu = 17, nv = 9)
        push!(traces, surface_trace(X, Y, Z; color = "#4063D8"))
    end
    Rb = asm.boundary.shape.semi_axes[1]
    X, Y, Z = ellipsoid_surface(Rb, Rb, Rb; nu = 33, nv = 17)
    push!(traces, surface_trace(X, Y, Z; color = "#8c959f", opacity = 0.15))
    return plotly_scene(traces; uid, height = 480, title = "27 spheres at the fraction 0.2, in the ball of the same fraction")
end

cluster_scene(cluster; uid = "mem-cluster-27")
````

### Conductivity

The published value is that of the code, with five decimals:

````@example multipole_spheres
[(tmax = t, k = round(conductivity(cluster, t); digits = 5)) for t in (1, 3, 5, 7)]
````

````@example multipole_spheres
q("fcm_conductivity")
````

### Stiffness

The elasticity problem has the same geometry, with nearly rigid spheres and
the Poisson ratio 0.3 in both phases. The block has the cubic symmetry of its
lattice, hence three moduli: the bulk modulus and the two shear moduli
``\mu_1 = (C_{1111} - C_{1122})/2`` and ``\mu_2 = C_{1212}``, which a
one-site estimate does not tell apart.

````@example multipole_spheres
iso(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)        # (3k, 2μ)
ν = q("fcm_poisson_ratio")
C₀ = iso(1.0, ν)
k₀ = 2 * (1 + ν) / (3 * (1 - 2ν))
elastic = block(3, q("fcm_volume_fraction"), Dict(:C => iso(q("fcm_shear_contrast"), ν)), Dict(:C => C₀))

function moduli(scheme)
    M = mandel(homogenize(elastic, scheme, :C), CanonicalBasis{3, Float64}())
    return ((M[1, 1] + 2M[1, 2]) / 3k₀, (M[1, 1] - M[1, 2]) / 2, M[4, 4] / 2)
end

rows = [("Mori–Tanaka", moduli(MoriTanaka()))]
append!(rows, [("multipole expansion, tmax = $t", moduli(MultipoleExpansion(; tmax = t))) for t in (2, 4, 6)])
push!(rows, ("published", (q("fcm_bulk_modulus"), q("fcm_shear_modulus_1"), q("fcm_shear_modulus_2"))))
io = IOBuffer()
println(io, "| | ``k/k_0`` | ``\\mu_1/\\mu_0`` | ``\\mu_2/\\mu_0`` |\n| :-- | :-- | :-- | :-- |")
for (name, m) in rows
    @printf(io, "| %s | %.4f | %.4f | %.4f |\n", name, m...)
end
Markdown.parse(String(take!(io)))
````

The bulk modulus hardly leaves the one-site estimate. The two shear moduli do,
in opposite directions: the block resists a stretch along one of its axes
with a contraction along another (``\mu_1``) more than a shear between two of
its axes (``\mu_2``).

## 4. Field around two spheres

The coefficients of the series give the field at any point of the matrix.
Take two spheres ten times more conducting than the matrix, under a gradient
along the line of their centers; [`multipole_coefficients`](@ref) returns the
coefficients for the three unit gradients, and the temperature follows by
summing the decaying harmonics of each sphere.

````@example multipole_spheres
centers = Dict(:a => [0.0, 0.0, 0.55], :b => [0.0, 0.0, -0.55])
pair = ParticleAssembly(; boundary = MixedBC(3.0))
add_matrix!(pair, matrix)
for (name, X) in centers
    add_particle!(pair, name, X, Ellipsoid(0.5), spheres(10.0))
end
tmax = 12
A = multipole_coefficients(pair; scheme = MultipoleExpansion(; tmax))

function temperature(x; gradient = 3)
    T = x[gradient]
    for (name, X) in centers
        Y = irregular_harmonics(x - X, tmax)
        T += real(sum(A[name, t, s][gradient] * Y[t, s] for t in 1:tmax for s in (-t):t))
    end
    return T
end
nothing #hide
````

The spheres are a tenth of a radius apart. Each one is nearly isothermal, and
the temperature drop that the remote gradient would spread over the gap is
concentrated in it: the map below is the temperature in a plane through the
two centers, the matrix alone being colored.

````@example multipole_spheres
xs, zs = range(-1.3, 1.3; length = 131), range(-1.5, 1.5; length = 151)
inside(x, z) = any(X -> hypot(x, z - X[3]) < 0.5, values(centers))
field = [inside(x, z) ? NaN : temperature([x, 0.0, z]) for z in zs, x in xs]
p_field = heatmap(
    xs, zs, field; aspect_ratio = :equal, color = :balance, xlabel = "x₁", ylabel = "x₃", xlims = (-1.3, 1.3),
    colorbar_title = "T", title = "Temperature under a unit gradient along x₃"
)
contour!(p_field, xs, zs, field; levels = 21, color = :black, lw = 0.5, colorbar_entry = false)
p_field
````

Along the line of centers the gradient in the gap is several times the
remote one:

````@example multipole_spheres
gap_gradient = (temperature([0.0, 0.0, 0.04]) - temperature([0.0, 0.0, -0.04])) / 0.08
````

## Summary

| result | source | recovered |
| :-- | :-- | :-- |
| conductivity of the simple cubic array | [kushch2014](@cite), Table 2 | to the third decimal up to ``c = 0.45`` at `tmax = 15`; ``c = 0.5`` at `tmax = 25` |
| conductivity of the body-centered and face-centered arrays | [kushch2013](@cite), Tables 3.2 and 3.3 | on the published points, short of contact |
| conductivity of the cluster of 27 spheres | [kushch2013](@cite), appendix A.1 | five decimals at `tmax = 5` |
| bulk and shear moduli of the cluster of 27 rigid spheres | [kushch2013](@cite), appendix A.3 | four decimals at `tmax = 6` |

```@raw html
<details class="source-file"><summary>Show the code of <code>scripts/common/docviz.jl</code>, included above</summary>
```

```@eval
Main.show_source("scripts/common/docviz.jl")
```

```@raw html
</details>
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

