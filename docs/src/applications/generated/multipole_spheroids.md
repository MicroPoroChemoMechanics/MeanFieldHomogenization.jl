```@meta
EditURL = "../../../../scripts/99_multipole_spheroids.jl"
```

# [Spheroids and ellipsoids by the multipole expansion](@id app-mem-spheroids)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/99_multipole_spheroids.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/99_multipole_spheroids.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/99_multipole_spheroids.jl`, runs it end to end.

!!! info "Before this page"
    [The multipole expansion method for spheroids](@ref th-mem-spheroids)
    and [for ellipsoids](@ref th-mem-ellipsoids), where the scheme is derived,
    [with anisotropic constituents](@ref th-mem-anisotropic),
    [for spheroids in elasticity](@ref th-mem-spheroids-elasticity),
    [for ellipsoids in elasticity](@ref th-mem-ellipsoids-elasticity), and
    [Particle assemblies and N-body schemes](@ref man-assemblies) for the
    assemblies it acts on.

[kushch2015](@citet) give the conductivity of oblate spheroids on three
lattices, with a perfect interface, a low-conducting (Kapitza) one and a
highly conducting one. This page computes the same quantities with
[`MultipoleExpansion`](@ref) and sets them beside the published values, read
from the files below. It goes on with the stiffness of spheroids bonded by a
spring [kushch2018](@cite), with ellipsoids bonded by a Kapitza interface
[kushch2020](@cite), with spheroids in a transversely isotropic matrix
[kushch2014](@cite), with the stiffness of lattices of spheroids
[kushch2013](@cite), and with ellipsoids bonded by springs
[kushch2019a](@cite).

````@example multipole_spheroids
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

paper = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2015.json"))
quantity(name) = paper["quantities"][name]["value"]
nothing #hide

Main.show_literature("kushch2015") # hide
````

````@example multipole_spheroids
Main.show_literature("kushch2018") # hide
````

````@example multipole_spheroids
Main.show_literature("kushch2020") # hide
````

````@example multipole_spheroids
Main.show_literature("kushch2014") # hide
````

````@example multipole_spheroids
Main.show_literature("kushch2019a") # hide
````

## 1. The lattices

The spheroids are oblate, of semi-axes ``(l_1, l_1, e\,l_1)`` with
``e = 1/3``, aligned with ``\underline{e}_3``. The lattices are proportional:
the cell has the periods ``(1, 1, e)``, and holds one spheroid (SP), two
(BCP, the second at the center of the cell) or four (FCP, the others at the
centers of three faces). The fraction ``c`` sets ``l_1``. The interface is
carried by a one-layer [`LayeredSpheroid`](@ref): a
[`KapitzaInterface`](@ref) of resistance ``\rho`` for the low-conducting
interface, of parameter ``\tilde h_c = l_1/(\rho\,k_0)``, and a
[`SurfaceConductiveInterface`](@ref) of conductance ``k^{\mathrm s}`` for the
highly conducting one, of parameter ``\tilde h_s = k^{\mathrm s}/(l_1\,k_0)``.

````@example multipole_spheroids
e = quantity("aspect_ratio")
k₀ = TensISO{3}(1.0)
sites = Dict(
    "sp" => [(0.0, 0.0, 0.0)],
    "bcp" => [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    "fcp" => [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5)],
)
semi_axis(kind, c) = cbrt(3c / (4π * length(sites[kind])))

function lattice(kind, c; k₁, interface = l₁ -> PerfectInterface{Float64}())
    l₁ = semi_axis(kind, c)
    asm = ParticleAssembly(; boundary = PeriodicBox((1.0, 1.0, e)))
    add_matrix!(asm, Dict(:K => k₀))
    for (i, x) in enumerate(sites[kind])
        spheroid = LayeredSpheroid((e * l₁,), (l₁,), (TensISO{3}(k₁),); interfaces = (interface(l₁),))
        add_particle!(asm, Symbol(:p, i), (x[1], x[2], e * x[3]), spheroid, Dict(:K => TensISO{3}(k₁)))
    end
    return asm
end

kapitza(h̃c) = l₁ -> KapitzaInterface(l₁ / h̃c)
surface(h̃s) = l₁ -> SurfaceConductiveInterface(h̃s * l₁)
conductivity(asm, nmax) = (K = canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K)); (K[1, 1], K[3, 3]))
nothing #hide
````

The body-centered cell at ``c = 0.4``, with the spheroids of the neighboring
cells that overlap it:

````@example multipole_spheroids
function cell_scene(asm; uid)
    traces = String[]
    for name in particle_names(asm), shift in Iterators.product(-1:1, -1:1, -1:1)
        X₀ = particle_center(asm, name) .+ shift .* (1.0, 1.0, e)
        all(-0.6 .≤ X₀ ./ (1.0, 1.0, e) .≤ 0.6) || continue
        a, b = MeanFieldHomogenization.LayeredSpheroids.outer_semiaxes(particle_geometry(asm, name))
        X, Y, Z = ellipsoid_surface(b, b, a; center = X₀, nu = 25, nv = 13)
        push!(traces, surface_trace(X, Y, Z; color = "#4063D8"))
    end
    return plotly_scene(traces; uid, height = 480, title = "oblate spheroids of aspect ratio 1/3 on the BCP lattice, c = 0.4")
end

cell_scene(lattice("bcp", 0.4; k₁ = 1.0); uid = "mem-bcp-spheroids")
````

## 2. Convergence on the simple lattice

At ``c = 0.5`` the spheroids of the simple lattice nearly touch, by a gap of
``0.005`` along their axis. The table gives ``(k_{11}, k_{33})/k_0`` against
the truncation degree, computed and, in parentheses, published, for the
conductivity ratio ``k_1/k_0 = 1000``. Only the odd degrees enter: every
spheroid being a center of symmetry of the lattice, the field it receives
under a uniform gradient is odd about it.

````@example multipole_spheroids
k₁ = quantity("contrast_lc")
perfect = l₁ -> PerfectInterface{Float64}()
columns = (("perfect", "hc_infinity", perfect), ("``\\tilde h_c = 10``", "hc_10", kapitza(10.0)), ("``\\tilde h_c = 1``", "hc_1", kapitza(1.0)))
function convergence_table(rows)
    io = IOBuffer()
    println(io, "| ``n_{\\\\max}`` | ", join(first.(columns), " | "), " |")
    println(io, "| :-- | ", join(fill(":--", length(columns)), " | "), " |")
    for row in rows
        cells = map(columns) do (_, key, interface)
            k11, k33 = conductivity(lattice("sp", 0.5; k₁, interface), row["tmax"])
            @sprintf("%.3f, %.3f (%.3f, %.3f)", k11, k33, row["k11_$key"], row["k33_$key"])
        end
        println(io, "| ", row["tmax"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
convergence_table(r for r in paper["tables"]["convergence_sp"] if r["tmax"] ≤ 7)
````

The published values agree with the computed ones to the last printed digit
at every degree: the truncation is the same, and so is the solution. The
perfect interface converges slowly, as the article notes; a resistance at
the interface smooths the field and the degree 5 suffices.

## 3. Three lattices, three interfaces

Each cell gives ``(k_{11}, k_{33})/k_0`` computed at `nmax = 7` and, in
parentheses, published.

````@example multipole_spheroids
function lattices_table(table, k₁, interface; cs = (0.1, 0.2), nmax = 7)
    io = IOBuffer()
    println(io, "| ``c`` | SP | BCP | FCP |")
    println(io, "| :-- | :-- | :-- | :-- |")
    for row in paper["tables"][table]
        row["c"] in cs || continue
        cells = map(("sp", "bcp", "fcp")) do kind
            k11, k33 = conductivity(lattice(kind, row["c"]; k₁, interface), nmax)
            @sprintf("%.3f, %.3f (%.3f, %.3f)", k11, k33, row["k11_$kind"], row["k33_$kind"])
        end
        println(io, "| ", row["c"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
nothing #hide
````

**Perfect interface**, ``k_1/k_0 = 1000``:

````@example multipole_spheroids
lattices_table("lattices_perfect", k₁, perfect)
````

**Low-conducting interface**, ``\tilde h_c = 10``, ``k_1/k_0 = 1000``:

````@example multipole_spheroids
lattices_table("lattices_lc_10", k₁, kapitza(10.0); cs = (0.2,), nmax = 5)
````

**Highly conducting interface**, ``\tilde h_s = 10``, ``k_1/k_0 = 0.1``:

````@example multipole_spheroids
lattices_table("lattices_hc_10", quantity("contrast_hc"), surface(10.0); cs = (0.2,), nmax = 5)
````

The arrangement matters: with the perfect interface, the FCP lattice
conducts a quarter more than the simple one in the plane of the spheroids at
``c = 0.2``, and less along their axis. Denser lattices need higher degrees:
on the simple lattice at ``c = 0.3``, ``k_{33}`` reaches its published value
2.016 at `nmax = 13` (2.016 computed), against 1.971 at `nmax = 7`. The random
arrangements of the article, averaged over a hundred cells, are not
reproduced here.

## 4. The interface parameter

The conductivities of the simple lattice at ``c = 0.5`` against the
parameter ``\tilde h_c`` of the low-conducting interface, beside the
[Mori–Tanaka](@ref th-scheme-mori-tanaka) estimate built on the same layered
spheroid. The latter is the Maxwell estimate of the article, Table 1, which it
recovers:

````@example multipole_spheroids
function maxwell_table()
    io = IOBuffer()
    println(io, "| ``\\\\tilde h_c`` | ``(k_{11}, k_{33})/k_0`` at ``c = 0.5`` | published |")
    println(io, "| :-- | :-- | :-- |")
    row = only(r for r in paper["tables"]["maxwell_lc"] if r["c"] == 0.5)
    for (label, key, interface) in (("1", "hc_1", kapitza(1.0)), ("10", "hc_10", kapitza(10.0)), ("∞", "hc_infinity", perfect))
        K = canonical_components(homogenize(lattice("sp", 0.5; k₁, interface), MoriTanaka(), :K))
        println(io, @sprintf("| %s | %.3f, %.3f | %.3f, %.3f |", label, K[1, 1], K[3, 3], row["k11_$key"], row["k33_$key"]))
    end
    return Markdown.parse(String(take!(io)))
end
maxwell_table()
````

and the curves:

````@example multipole_spheroids
h̃ = 10 .^ range(-1, 2; length = 7)
mem = [conductivity(lattice("sp", 0.5; k₁, interface = kapitza(h)), 5) for h in h̃]
mt = map(h̃) do h
    K = canonical_components(homogenize(lattice("sp", 0.5; k₁, interface = kapitza(h)), MoriTanaka(), :K))
    (K[1, 1], K[3, 3])
end
p_interface = plot(;
    xscale = :log10, xlabel = "h̃c", ylabel = "k / k₀", legend = :topleft, size = (640, 400),
    title = "SP lattice, c = 0.5, k₁/k₀ = 1000",
)
plot!(p_interface, h̃, first.(mem); marker = :circle, label = "k₁₁, multipole expansion")
plot!(p_interface, h̃, last.(mem); marker = :square, label = "k₃₃, multipole expansion")
plot!(p_interface, h̃, first.(mt); linestyle = :dash, label = "k₁₁, Mori–Tanaka")
plot!(p_interface, h̃, last.(mt); linestyle = :dash, label = "k₃₃, Mori–Tanaka")
p_interface
````

The single-particle estimate misses the interactions over the whole range:
it overestimates ``k_{11}`` and underestimates ``k_{33}``, the two curves of
``k_{33}`` crossing only near ``\tilde h_c = 3``, and the gap grows with
``\tilde h_c``.

## 5. Elastic spheroids with a spring interface

[kushch2018](@citet) treats one spheroid with a spring interface of equal
normal and tangential stiffnesses ``D_s``, of parameter
``\tilde D = D_s \max(l_1, l_3)/\mu_0``, and gives the Maxwell estimate of aligned
spheroids at the fraction 0.3, ``\mu_1/\mu_0 = 100``, ``\nu_0 = 0.3``,
``\nu_1 = 0.2``. For aligned spheroids that estimate is Mori–Tanaka, computed
here on the [elastic layered spheroid](@ref th-spheroid-imperfect). Each cell
gives ``(C_{11}, C_{33}, C_{44})/\mu_0``, computed and, in parentheses, published.

````@example multipole_spheroids
elastic = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2018.json"))
iso(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)
eq(name) = elastic["quantities"][name]["value"]
C₀ = iso(1.0, eq("nu0"))
C₁ = iso(eq("mu_ratio"), eq("nu1"))
function spring_estimate(e, D̃)
    D = D̃ / max(1.0, e)
    spheroid = LayeredSpheroid((e,), (1.0,), (C₁,); interfaces = (SpringInterface(D, D),))
    s = cbrt(eq("fraction"))
    asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(1 / s, 1 / s, e / s)))
    add_matrix!(asm, Dict(:C => C₀))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), spheroid, Dict(:C => C₁))
    M = mandel(homogenize(asm, MoriTanaka(), :C), CanonicalBasis{3, Float64}())
    return (M[1, 1], M[3, 3], M[4, 4] / 2)
end
function spring_table()
    io = IOBuffer()
    println(io, "| ``\\\\tilde D`` | ``e = 1/3`` | ``e = 3`` |")
    println(io, "| :-- | :-- | :-- |")
    rows = elastic["tables"]["maxwell_aligned"]
    for D̃ in (1.0e-6, 1.0, 10.0, 1.0e6)
        cells = map((1 / 3, 3.0)) do e
            row = only(r for r in rows if r["e"] ≈ e && r["D"] == D̃)
            C = spring_estimate(e, D̃)
            @sprintf("%.4f, %.4f, %.4f (%.4f, %.4f, %.4f)", C..., row["C11"], row["C33"], row["C44"])
        end
        println(io, "| ", D̃, " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
spring_table()
````

Through a soft spring (``\tilde D = 10^{-6}``) spheroids a hundred times stiffer
than the matrix lower all its moduli (``C_{11} = C_{33} = 3.5\,\mu_0`` and
``C_{44} = \mu_0`` in the matrix alone); through a stiff one they act as
perfectly bonded.

## 6. Ellipsoids with a Kapitza interface

[kushch2020](@citet) gives the Maxwell estimate ``k_{11}/k_0`` of ellipsoids of
semi-axes ``(1.5, 1, 0.5)`` at the fraction 0.5, conductivity ratio 100,
against the conductance ``h_c`` of a Kapitza interface and the degree. One
[`InterfacedEllipsoid`](@ref) in a domain of its own shape is that estimate;
each cell gives the computed value and, in parentheses, the published one.

````@example multipole_spheroids
ellipsoids = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2020.json"))
semi = Tuple(ellipsoids["quantities"]["semi_axes"]["value"])
function ellipsoid_estimate(hc, nmax)
    c = ellipsoids["quantities"]["fraction"]["value"]
    asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(c))...)))
    add_matrix!(asm, Dict(:K => k₀))
    particle = InterfacedEllipsoid(semi..., KapitzaInterface(1 / hc))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), particle, Dict(:K => TensISO{3}(ellipsoids["quantities"]["contrast"]["value"])))
    return canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K))[1, 1]
end
function ellipsoid_table()
    io = IOBuffer()
    hcs = (("hc_1e-2", 1.0e-2), ("hc_1", 1.0), ("hc_1e2", 1.0e2))
    println(io, "| ``n_{\\\\max}`` | ", join(("``h_c = $(h)``" for (_, h) in hcs), " | "), " |")
    println(io, "| :-- | :-- | :-- | :-- |")
    for row in ellipsoids["tables"]["maxwell_kapitza"]
        row["nmax"] ≤ 5 || continue
        cells = [@sprintf("%.7f (%.7f)", ellipsoid_estimate(h, row["nmax"]), row[key]) for (key, h) in hcs]
        println(io, "| ", row["nmax"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
ellipsoid_table()
````

The seven printed digits are recovered at every degree: the response of the
ellipsoid couples its harmonics through the interface, and the truncation is
the same as in the article.

## 7. Spheroids in a transversely isotropic matrix

[kushch2014](@citet) consider spheroids of semi-axes ``(1, 1, e)`` and
conductivity ``10\,k_{11}`` in a matrix of conductivities ``k_{11} = 1`` across
and ``k_{33} = \kappa`` along ``\underline{e}_3``, on an orthorhombic lattice of
periods proportional to the semi-axes. They tabulate the anisotropy parameter
``\varpi = (k^{\mathrm{hom}}_{33}/\kappa)/k^{\mathrm{hom}}_{11}``, which measures
how much the particles change the anisotropy of the matrix. The scheme maps
the matrix to an isotropic one (see [the theory](@ref th-mem-anisotropic)),
where the spheroids become spheroids of semi-axis ratio ``e/\sqrt{\kappa}``,
or spheres when ``e = \sqrt{\kappa}``, of anisotropic conductivity.

````@example multipole_spheroids
anisotropy = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2014.json"))
function anisotropy_parameter(e, κ, c, nmax)
    semi = (1.0, 1.0, e)
    cell = ParticleAssembly(; boundary = PeriodicBox(cbrt(4π / 3 / c) .* semi))
    add_matrix!(cell, Dict(:K => TensTI{2}(1.0, κ, (0.0, 0.0, 1.0))))
    inclusion = anisotropy["quantities"]["inclusion_conductivity"]["value"]
    add_particle!(cell, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:K => TensISO{3}(inclusion)))
    k = diag(canonical_components(homogenize(cell, MultipoleExpansion(; nmax), :K)))
    return (k[3] / κ) / k[1]
end
function anisotropy_table(nmax)
    io = IOBuffer()
    fractions = ("0.15", "0.30", "0.40")
    println(io, "| ``e`` | ``\\\\kappa`` | ", join(("``c = $(c)``" for c in fractions), " | "), " |")
    println(io, "| :-- | :-- | :-- | :-- | :-- |")
    for row in anisotropy["tables"]["anisotropy_ti"]
        cells = map(fractions) do c
            computed, published = anisotropy_parameter(row["aspect_ratio"], row["matrix_anisotropy"], parse(Float64, c), nmax), row["c_$c"]
            cell = @sprintf("%.3f (%.3f)", computed, published)
            abs(computed - published) > 1.0e-3 ? "**$(cell)**" : cell
        end
        println(io, "| ", row["aspect_ratio"], " | ", row["matrix_anisotropy"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
anisotropy_table(8)
````

Each cell gives the computed value at `nmax = 8` and, in parentheses, the
published one; the cells in bold differ by more than one unit of the third
decimal. There are three, all at ``\kappa = 0.25``, where the map stretches
the particles most (to the ratio ``e/\sqrt{\kappa}`` = 2 and 4). The published
values ``\varpi = 1.282`` (``e = 1``, ``c = 0.3``) and ``1.793`` (``e = 2``,
``c = 0.4``) break the monotonic trend of their rows, which the computed
values follow; ``1.636`` (``e = 2``, ``c = 0.3``) differs by two units. The
computed values do not move when the degree is raised:

````@example multipole_spheroids
[round(anisotropy_parameter(e, 0.25, c, nmax); digits = 4) for (e, c) in ((1.0, 0.3), (2.0, 0.3), (2.0, 0.4)), nmax in (8, 10)]
````

The higher fractions of the table, 0.45 and 0.5, approach the closest packing
``\pi/6`` of the lattice and need degrees beyond those run here.

## 8. Stiffness of lattices of spheroids

[kushch2013](@citet), Table 7.6, gives ``C^{\mathrm{hom}}_{3333}/\mu_0`` of spheroids of
semi-axes ``(1, 1, \varepsilon)`` on an orthogonal lattice of periods
proportional to them, with ``\nu_0 = \nu_1 = 0.3``, against the shear modulus
ratio ``\mu_1/\mu_0`` (zero for cavities) and the volume fraction. The
spheres, ``\varepsilon = 1``, are those of the application
[Arrays and clusters of spheres by the multipole expansion](@ref app-mem-spheres);
the oblate spheroids follow, at the fraction 0.1 for both shapes and 0.3 for
``\varepsilon = 0.5``.

````@example multipole_spheroids
book = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2013.json"))
poisson_iso(μ) = TensISO{3}(2μ * 1.3 / 0.4, 2μ)          # (3κ, 2μ) at ν = 0.3
function lattice_stiffness(ε, μ₁, c, nmax)
    semi = (1.0, 1.0, ε)
    cell = ParticleAssembly(; boundary = PeriodicBox(cbrt(4π / 3 / c) .* semi))
    add_matrix!(cell, Dict(:C => poisson_iso(1.0)))
    add_particle!(cell, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:C => poisson_iso(μ₁)))
    M = TensND.mandel(homogenize(cell, MultipoleExpansion(; nmax), :C), TensND.CanonicalBasis{3, Float64}())
    return M[3, 3]
end
function lattice_table()
    io = IOBuffer()
    println(io, "| ``\\\\varepsilon`` | ``\\\\mu_1/\\\\mu_0`` | ``c = 0.1`` | ``c = 0.3`` |")
    println(io, "| :-- | :-- | :-- | :-- |")
    for row in book["tables"]["stiffness_spheroids"]
        ε, μ₁ = row["aspect_ratio"], row["contrast"]
        ε < 1 || continue
        # a flat cavity converges more slowly in the degree
        low = @sprintf("%.3f (%.3f)", lattice_stiffness(ε, μ₁, 0.1, ε == 0.2 && μ₁ == 0 ? 8 : 6), row["c_0.1"])
        high = ε == 0.5 ? @sprintf("%.3f (%.3f)", lattice_stiffness(ε, μ₁, 0.3, 8), row["c_0.3"]) : "–"
        println(io, "| ", ε, " | ", μ₁, " | ", low, " | ", high, " |")
    end
    return Markdown.parse(String(take!(io)))
end
lattice_table()
````

Each cell gives the computed value and, in parentheses, the published one.
They agree to within one unit of the third decimal but at ``\varepsilon = 0.5``,
``\mu_1/\mu_0 = 10``, ``c = 0.3``, where the book prints 5.675 and the
computed value is 5.765, the same digits in another order; the published
value is also below those of ``\varepsilon = 1`` (5.930) and
``\varepsilon = 0.2`` (5.737), where the computed ones fall in between.

## 9. Ellipsoids bonded by springs

[kushch2019a](@citet), Table 5, gives the Maxwell estimate of ellipsoids of
semi-axes ``(1.5, 1, 0.5)`` at the volume fraction 0.5, with
``\nu_0 = 0.3``, ``\nu_1 = 0.2``, ``\mu_1/\mu_0 = 100``, bonded by springs of
equal normal and tangential stiffness ``D``, from the perfect bond to the
cavity. One [`InterfacedEllipsoid`](@ref) in a domain of its own shape is that
estimate; the rows ``D = 10`` and ``D = 1`` are recovered with the stiffness
``D\,\mu_0`` (the text of the article states ``D\,\mu_0/a_1``).

````@example multipole_spheroids
springs = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2019a.json"))
spring_q(k) = springs["quantities"][k]["value"]
function ellipsoid_spring_estimate(D)
    semi = Tuple(spring_q("semi_axes"))
    ν_iso(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)
    cell = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(spring_q("fraction")))...)))
    add_matrix!(cell, Dict(:C => ν_iso(1.0, spring_q("nu0"))))
    C₁ = D == 0 ? TensISO{3}(0.0, 0.0) : ν_iso(spring_q("mu_ratio"), spring_q("nu1"))
    particle = D isa Number && D > 0 ? InterfacedEllipsoid(semi..., SpringInterface(D, D)) : Ellipsoid(semi...)
    add_particle!(cell, :p, (0.0, 0.0, 0.0), particle, Dict(:C => C₁))
    return TensND.mandel(homogenize(cell, MultipoleExpansion(; nmax = 8), :C), TensND.CanonicalBasis{3, Float64}())
end
function ellipsoid_spring_table()
    io = IOBuffer()
    entries = (("C1111", 1, 1), ("C3333", 3, 3), ("C2233", 2, 3), ("C1212", 6, 6))
    println(io, "| ``D`` | ", join(("``C_{$(k[2:end])}/\\\\mu_0``" for (k, _, _) in entries), " | "), " |")
    println(io, "| :-- | :-- | :-- | :-- | :-- |")
    for row in springs["tables"]["maxwell_spring"]
        D = row["D"] == "Infinity" ? Inf : row["D"]
        M = ellipsoid_spring_estimate(D)
        cells = [@sprintf("%.4f (%.4f)", M[i, j] / (i > 3 ? 2 : 1), row[k]) for (k, i, j) in entries]
        println(io, "| ", row["D"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
ellipsoid_spring_table()
````

Each cell gives the computed value at `nmax = 8` and, in parentheses, the
published one. The perfect bond and the cavity are Mori–Tanaka, which the
expansion gives exactly; there the published ``C_{2233}`` of the cavity,
0.2970, differs from that estimate, 0.2940.

## Summary

| result | source | recovered |
| :-- | :-- | :-- |
| convergence with the degree, simple lattice, ``c = 0.5`` | [kushch2015](@cite), Table 3 | to the last digit at every degree up to 7 |
| Maxwell estimate, low-conducting interface | [kushch2015](@cite), Table 1 | Mori–Tanaka on the layered spheroid, to the third decimal |
| three lattices, perfect interface | [kushch2015](@cite), Table 4 | to the third decimal at `nmax = 7` up to ``c = 0.2``; ``c = 0.3`` on the simple lattice at `nmax = 13` |
| low- and highly conducting interfaces | [kushch2015](@cite), Tables 6 and 8 | to the third decimal at `nmax = 5`, ``c = 0.2`` |
| elastic spheroids with a spring interface, Maxwell estimate | [kushch2018](@cite), Table 2 | to the fourth decimal |
| ellipsoids with a Kapitza interface, Maxwell estimate | [kushch2020](@cite), Table 2 | to the seventh decimal, degree by degree |
| stiffness of lattices of spheroids | [kushch2013](@cite), Table 7.6 | to within one unit of the third decimal at `nmax = 6` to 8, but for one entry printed with its digits permuted |
| ellipsoids bonded by springs, Maxwell estimate | [kushch2019a](@cite), Table 5 | to the fourth decimal with the stiffness ``D\,\mu_0``, but for one entry of the cavity, where the published value differs from Mori–Tanaka |
| spheroids in a transversely isotropic matrix, anisotropy parameter | [kushch2014](@cite), Table 5 | to the third decimal up to ``c = 0.4`` at `nmax = 8`, but for three entries at ``\kappa = 0.25`` |

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

