# [Particle assemblies and N-body schemes](@id man-assemblies)

Two schemes of the package resolve the interaction between individual inclusions
instead of averaging it: [`ClusterModel`](@ref) and [`EquivalentInclusion`](@ref). Both
need to know *where* the inclusions are, so both act on a
[`ParticleAssembly`](@ref) rather than on an `RVE`.

Theory: [interaction tensors](@ref th-interaction), [the cluster model](@ref
th-cluster), [the equivalent inclusion method](@ref th-eim).

## Building an assembly

```julia
using MeanFieldHomogenization, TensND

C_m = TensISO{3}(3 * 1.0, 2 * 0.4)      # k = 1, μ = 0.4
C_i = TensISO{3}(3 * 10.0, 2 * 6.0)

asm = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 3.0))
add_matrix!(asm, Dict(:C => C_m))
add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.25), Dict(:C => C_i))
add_particle!(asm, :p2, (0.5, 0.5, 0.5), Ellipsoid(0.25), Dict(:C => C_i))

homogenize(asm, ClusterModel(), :C)
homogenize(asm, EquivalentInclusion(), :C)
```

Volume fractions are **derived**, never stored — `f_a = |Ω_a| / |Ω|` — so the geometry
and the fractions cannot disagree:

```julia
particle_volume_fraction(asm, :p1)
inclusion_volume_fraction(asm)
matrix_volume_fraction(asm)
assembly_volume(asm)
```

[`validate_assembly`](@ref) checks that a matrix and at least one particle are present,
that the dimensions agree, that the cell is not overfilled, and that **no two particles
overlap** — the interaction kernel is undefined for overlapping regions, so this must
fail early rather than as a `DomainError` deep inside a lattice sum.

## Boundary treatments

The two schemes solve the same linear system and differ in how the far field is closed.
That difference lives in the boundary, not in the scheme.

| | measure of `Ω` | far-field term | source |
| :-- | :-- | :-- | :-- |
| [`PeriodicBox`](@ref)`(L; cutoff = R_c)` | `Lᵈ` | Hill tensor of the inclusion shape | Molinari & El Mouden |
| [`MixedBC`](@ref)`(shape)` | volume of `shape` | Hill tensor of the SVE, `ℙ_Ω` | Brisard et al. |

`PeriodicBox` tiles space and truncates the image sums to a **sphere** of radius `R_c`
around the receiver; the default `3L` sits inside the convergence plateau reported in
the paper, and `cutoff = 0` reduces every cluster to its own receiver. `MixedBC` needs
an *ellipsoidal* SVE (the derivation uses Eshelby's theorem on the domain itself) and
needs no periodization at all.

Each scheme keeps its own default, but either boundary works with either scheme — which
is what makes the cross-check below possible.

## Generators

```julia
# Molinari & El Mouden's cubic arrays. All sites of a Bravais lattice are
# equivalent, so the system collapses to one unknown.
asm = cubic_lattice(:sc, Dict(:C => C_m), Dict(:C => C_i); fraction = 0.3)
asm = cubic_lattice(:fcc, Dict(:C => C_m), Dict(:C => C_i); fraction = 0.4, cutoff = 4.0)
max_packing_fraction(:bcc)              # 0.6802 — where spheres touch

# A two-material motif: their BCC array of alternating voids and rigid spheres.
cubic_lattice(:bcc, Dict(:C => C_m), [Dict(:C => C_i), Dict(:C => C_void)]; fraction = 0.2)

# Brisard's hard-particle Metropolis microstructures, in a periodic box …
using Random
asm = random_assembly(32, Dict(:C => C_m), Dict(:C => C_i);
                      fraction = 0.3, rng = MersenneTwister(1), cycles = 20_000)

# … or inside an ellipsoidal SVE, which is the geometry of their Table 1.
asm = random_assembly(160, Dict(:C => C_m), Dict(:C => C_void);
                      radius = 1.0, dim = 2, rng = MersenneTwister(1),
                      boundary = MixedBC(Ellipsoid(20.0, 20.0)))
```

Both generators are deterministic given their inputs; always pass an explicit `rng`.

## Families

Particles sharing a **family** label are constrained to carry the same unknown. This is
how a periodic motif with symmetric particles is described without duplicating
equations, and it is what makes a lattice estimate cheap: `cubic_lattice` assigns one
family per distinct material, so a single-material SC, BCC or FCC array has exactly one
unknown whatever the number of sites.

```julia
add_particle!(asm, :a1, (0.0, 0.0, 0.0), Ellipsoid(0.1), Dict(:C => C_i); family = 1)
add_particle!(asm, :a2, (0.5, 0.0, 0.0), Ellipsoid(0.1), Dict(:C => C_i); family = 1)
family_labels(asm)          # [1]
```

## Conduction

Both kernels are written once and dispatch on the tensor order of the property, so
transport needs nothing new:

```julia
asm = cubic_lattice(:sc, Dict(:K => TensISO{3}(1.0)), Dict(:K => TensISO{3}(20.0));
                    fraction = 0.25)
homogenize(asm, ClusterModel(), :K)
```

## Interaction back-ends

`interaction_tensor` selects a back-end automatically; pass `method` to override, and
the option reaches the schemes too:

```julia
homogenize(asm, ClusterModel(; method = :multipole, order = 2), :C)
```

| `method` | applies to | notes |
| :-- | :-- | :-- |
| `:analytical` | ball and disk pairs | closed form, **exact at any separation** |
| `:multipole` | any ellipsoid pair | truncated expansion, default off the ball case |
| `:quadrature` | anything | product rule on the definition; the validation oracle, far too slow to assemble a system |

!!! warning "Isotropic reference only"
    The real-space Green operator is implemented for isotropic references. An
    anisotropic `C₀` raises an `ArgumentError` naming the limitation rather than
    silently using an isotropic kernel.

## Local fields and bounds

```julia
τ, names = eim_polarizations(asm)        # per-particle polarization operators
A, reps  = cluster_localizations(asm)    # per-family localization tensors
eim_bound_type(asm)                      # :upper, :lower or :none
```

[`eim_bound_type`](@ref) reports whether the equivalent-inclusion estimate is a rigorous
bound: `:upper` when the matrix is stiffer than every inhomogeneity, `:lower` when it is
softer, `:none` for mixed contrasts.

## Cross-checking the two schemes

On a periodic assembly with the same cutoff the two are the *same* linear system and
agree to machine precision — the identity Brisard et al. state in their §3.1:

```julia
asm = cubic_lattice(:sc, Dict(:C => C_m), Dict(:C => C_i); fraction = 0.25, cutoff = 3.0)
homogenize(asm, ClusterModel(), :C) ≈ homogenize(asm, EquivalentInclusion(), :C)   # true
```

Useful degeneracies to keep in mind when reading results:

* `cluster_radius = 0` ⟹ exactly [`MoriTanaka`](@ref);
* one spherical particle in a spherical `MixedBC` SVE ⟹ exactly `MoriTanaka`;
* a cubic array keeps the Mori-Tanaka **bulk** modulus exactly, at every fraction —
  only the shear response sees the arrangement.

## Sensitivities

An assembly answers the usual differentiation entry points, with two lenses no other
cell has:

```julia
derivative(asm, ClusterModel(), center_param(:p2, 1); indexer = C -> get_array(C)[1, 2, 1, 2])
derivative(asm, ClusterModel(), radius_param(:p1);    indexer = C -> get_array(C)[1, 2, 1, 2])
```

`PropertyParameter` works through the generic cell contract, so moduli need nothing
special. A `RadiusParameter` on a sphere sets every semi-axis together, so the geometry
stays spherical — and keeps its closed-form kernel — along the whole derivative.

## Multiscale

An assembly is an ordinary cell, so it plugs into the declarative multiscale seam from
both sides — store `Homogenized(cell, scheme)` as a property and it is resolved lazily
at `homogenize` time, memoized for the duration of the outer call.

```julia
# An assembly as the INNER cell: a composite whose matrix is itself particulate.
inner = cubic_lattice(:sc, Dict(:C => C_m), Dict(:C => C_i); fraction = 0.3)
outer = RVE(:M)
add_matrix!(outer, Ellipsoid(1.0), Dict(:C => Homogenized(inner, ClusterModel())))
add_phase!(outer, :F, Ellipsoid(1.0), Dict(:C => C_f); fraction = 0.2)
homogenize(outer, MoriTanaka(), :C)

# An assembly as the OUTER cell: a particle that is a composite in its own right.
asm = ParticleAssembly(; boundary = PeriodicBox(1.0))
add_matrix!(asm, Dict(:C => C_m))
add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.3),
              Dict(:C => Homogenized(sub_rve, MoriTanaka())))
homogenize(asm, ClusterModel(), :C)
```

Both give exactly the two-step result computed by hand — the seam adds nothing of its
own.

!!! note "Chaining N-body schemes needs the anisotropic Green operator"
    A cluster or equivalent-inclusion estimate on a cubic array is **cubic, not
    isotropic**: its two shear constants differ. Using one as the reference medium of
    another scale therefore requires the interaction tensor in an anisotropic reference
    — which is exactly what [`green_operator_aniso`](@ref) provides. This is why the
    Barnett line integral is not an optional extra but the enabler of multiscale
    chaining. `scripts/92` walks through a three-level example.

    The isotropic route stays the default and costs microseconds; the anisotropic one
    is a differentiated quadrature and costs milliseconds. Tune it with `green_nodes`
    (default 32) if a strongly anisotropic reference needs more.

### Sensitivities across scales

A `nested` lens addresses a scalar inside the inner cell:

```julia
lens = nested(:M, :C, property(:matrix, :C, :μ))     # an inner-scale modulus
derivative(outer, MoriTanaka(), lens; indexer = C -> get_array(C)[1, 2, 1, 2])
```

The lenses an assembly answers are its own — `radius_param`, `center_param` and
`property`. It has **no** `amount`: volume fractions are derived from the geometry and
the cell size, so there is nothing to vary independently, and asking for one raises a
message that names the lens to use instead.

## Nano-interfaces: the equivalent particle

[Dormieux, Lemarchand & Brisard 2016](@cite dormieux2016) is a different kind of result:
it needs no new scheme. A spheroidal nanoinclusion together with its Gurtin-Murdoch
interface behaves as a single particle of stiffness
``\mathbb C^{eq} = \mathbb C_I + \mathbb C^{int}``, after which the classical
concentration rule applies unchanged:

```julia
κs, μs = 64.0, 51.3                      # surface moduli, stiffness × length
sph = Ellipsoid(0.01)                    # 10 nm particles
C_eq = equivalent_particle(C_i, sph, κs, μs)

rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_m))
add_phase!(rve, :nano, sph, Dict(:C => C_eq); fraction = 0.15)
homogenize(rve, MoriTanaka(), :C)        # the paper's extended Mori-Tanaka
```

[`surface_stiffness`](@ref) returns ``\mathbb C^{int}`` for any spheroid, transversely
isotropic about the symmetry axis, with the platelet (``X \to 0``) and nanofiber
(``X \to \infty``) limits of the paper reproduced exactly. Since it scales as
`1/size`, the stiffening it produces is a genuine size effect and vanishes for large
particles.

## API

See [API — Interactions](@ref api-interactions) and
[API — Particle assemblies](@ref api-assemblies).
