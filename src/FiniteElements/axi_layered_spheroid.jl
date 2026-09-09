# =============================================================================
#  axi_layered_spheroid.jl — an `N`-layer spheroid solved by Fourier
#  axisymmetric finite elements.
#
#  The heterogeneous counterpart of `FEAxiSupershapePore`, and the `N`-layer
#  generalization of `FEExcenteredSphere`. Everything about the solve is shared
#  with both; what this file adds is the geometry, the per-layer material map,
#  and the layer averages the bounds need.
#
#  ## Free semi-axes, and why the confocal case is not privileged
#
#  Layers are given by their semi-axes, per layer, ascending — the same
#  `(axis_radii, disk_radii)` pair `LayeredSpheroid` takes, in the same order.
#  Handed `confocal_layer_radii(...)` the object is the body the analytic
#  solution solves, so a cross-check compares two solvers on one geometry rather
#  than on two transcriptions of it. Handed anything else it is a nest of
#  spheroids no closed form covers, which is the point of having it.
#
#  ## What is *not* here yet
#
#  Imperfect interfaces. The axisymmetric formulation has no interface term at
#  all — the only displacement jump in the package is the three-dimensional
#  crack's — so a spring or Kapitza layer is a whole new backend generic and a
#  separate piece of work. Until then the constructor accepts perfect interfaces
#  and refuses the others by name, rather than accepting them and ignoring them.
# =============================================================================

"""
    FEAxiLayeredSpheroid(axis_radii, disk_radii, props; kwargs...)

An `N`-layer coaxial, concentric spheroid in an isotropic matrix, solved by
Fourier axisymmetric finite elements on the meridian half-plane.

`axis_radii[ℓ]` and `disk_radii[ℓ]` are layer `ℓ`'s semi-axes along and across
the revolution axis, **ascending**, core first — the argument order of
[`LayeredSpheroid`](@ref MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid), so
[`confocal_layer_radii`](@ref MeanFieldHomogenization.LayeredSpheroids.confocal_layer_radii)
feeds both. `props` is one isotropic modulus tensor per layer: `Tens{4,3}` for
elasticity, `Tens{2,3}` for transport, and the order chosen fixes the object's
physics.

| keyword | default | meaning |
|---|---|---|
| `interfaces` | all perfect | per-layer interface; only `PerfectInterface` is implemented |
| `opts` | `FEAxiMeshOptions()` | mesh density and cell radius |
| `basis` | canonical | the revolution axis is the basis's third column |
| `backend` | `AutoBackend()` | finite-element backend |

Enters through **gate B**, with both localization tensors measured on the same
solve: the inclusion is heterogeneous, so the stress side is not `ℂ₁ : 𝔸_εε`
for any single `ℂ₁`.

# Examples

```julia
# The validation slice: confocal, described exactly as the analytic type is.
ar, dr = confocal_layer_radii(2.0, 1.0, (0.3, 0.7))
K = (TensISO{3}(5.0), TensISO{3}(2.0))
fe  = FEAxiLayeredSpheroid(ar, dr, K)
ana = LayeredSpheroid(ar, dr, K)

# Free radii: an oblate core in a prolate shell. No closed form covers it.
FEAxiLayeredSpheroid((0.4, 1.4), (0.9, 1.0), K)
```
"""
struct FEAxiLayeredSpheroid{T <: Number, P, B <: TensND.AbstractBasis} <:
    Core.AbstractCustomInclusion{T}
    axis_radii::Vector{T}
    disk_radii::Vector{T}
    props::P
    basis::B
    mesh::FEAxiMeshOptions
    cache::FECache
    backend::FEBackend
end

function FEAxiLayeredSpheroid(
        axis_radii, disk_radii, props::Tuple;
        interfaces = ntuple(_ -> LayeredSpheres.PerfectInterface{Float64}(), length(props)),
        opts::FEAxiMeshOptions = FEAxiMeshOptions(),
        basis::TensND.AbstractBasis = TensND.CanonicalBasis{3, Float64}(),
        backend::FEBackend = AutoBackend(),
    )
    ar = collect(Float64, axis_radii)
    dr = collect(Float64, disk_radii)
    n = length(ar)
    length(props) == n || throw(
        ArgumentError(
            "a layered spheroid needs one modulus tensor per layer: $n layers " *
                "against $(length(props)) properties"
        )
    )
    # Nesting first: a violation is a self-intersecting geometry that gmsh
    # rejects from deep inside its own pipeline, naming neither layer nor
    # semi-axis.
    check_nested_spheroids(ar, dr, opts.radius_ratio * max(maximum(ar), maximum(dr)))
    O = _tens_order(props[1])
    all(p -> _tens_order(p) == O, props) || throw(
        ArgumentError(
            "every layer must carry the same tensor order: order $O for layer 1, " *
                "and the object's physics is fixed by it"
        )
    )
    for (ℓ, i) in enumerate(interfaces)
        i isa LayeredSpheres.PerfectInterface || throw(
            ArgumentError(
                "layer $ℓ carries a $(nameof(typeof(i))), and the axisymmetric " *
                    "finite-element cell implements `PerfectInterface` only. An " *
                    "imperfect interface needs a displacement (or temperature) " *
                    "jump term, which this formulation does not have yet — it is " *
                    "refused here rather than accepted and ignored."
            )
        )
    end
    T = Float64
    return FEAxiLayeredSpheroid{T, typeof(props), typeof(basis)}(
        ar, dr, props, basis, opts, FECache(), backend
    )
end

"""
    LayeredSpheroidShape

[`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait) of
[`FEAxiLayeredSpheroid`](@ref). No kernel dispatches on it — the inclusion
supplies its own localization tensors.
"""
struct LayeredSpheroidShape end

tensor_order(s::FEAxiLayeredSpheroid) = _tens_order(s.props[1])

Core.dimension(::FEAxiLayeredSpheroid) = 3
Core.inclusion_basis(s::FEAxiLayeredSpheroid) = s.basis
Core.shape_trait(::FEAxiLayeredSpheroid) = LayeredSpheroidShape
Core.is_homogeneous_inclusion(::FEAxiLayeredSpheroid) = false
_fe_cache(s::FEAxiLayeredSpheroid) = s.cache

# `layer_count` belongs to `LayeredSpheres`, so this is a method on that
# function and not a homonym in this module — defining a second `layer_count`
# here would shadow nothing and resolve to neither, which is what a
# `MethodError` naming only the layered-sphere methods looks like.
"""
    layer_count(incl::FEAxiLayeredSpheroid) -> Int

How many layers the inclusion has, the matrix excluded.
"""
LayeredSpheres.layer_count(s::FEAxiLayeredSpheroid) = length(s.axis_radii)

# Local shorthand, so the rest of the file does not repeat the qualification.
_nlayers(s::FEAxiLayeredSpheroid) = length(s.axis_radii)

"""
    layer_volumes(incl) -> Vector

Volume of each layer, `4π/3 · a² c` of its own boundary minus the one inside
it. Closed form for *any* semi-axes, confocal or not — which is what makes it
the mesh check that is always available.
"""
function layer_volumes(s::FEAxiLayeredSpheroid)
    n = _nlayers(s)
    encl = [4π / 3 * s.disk_radii[ℓ]^2 * s.axis_radii[ℓ] for ℓ in 1:n]
    return [ℓ == 1 ? encl[1] : encl[ℓ] - encl[ℓ - 1] for ℓ in 1:n]
end

"""
    layer_fractions(incl) -> Vector

Each layer's share of the inclusion's volume, summing to one.
"""
function layer_fractions(s::FEAxiLayeredSpheroid)
    v = layer_volumes(s)
    return v ./ sum(v)
end

# The outer boundary's semi-axes, about the revolution axis: the third column of
# the basis is the axis, so the distinct semi-axis goes in slot 3.
function Core.shape_tensor(s::FEAxiLayeredSpheroid{T}) where {T}
    n = _nlayers(s)
    D = zeros(T, 3, 3)
    D[1, 1] = D[2, 2] = s.disk_radii[n]
    D[3, 3] = s.axis_radii[n]
    return TensND.Tens(D, s.basis)
end

# ─── Layer averages — what the bounds need ───────────────────────────────────
#
#  A heterogeneous inclusion has no single phase property, so `Voigt` and
#  `Reuss` cannot read one off the RVE. Here the internal fractions are a
#  closed form of the semi-axes, so the average is exact and free.

function _layer_average(
        s::FEAxiLayeredSpheroid, ::TensND.AbstractTens{O, 3}, f
    ) where {O}
    tensor_order(s) == O || throw(
        ArgumentError(
            "this `FEAxiLayeredSpheroid` carries order-$(tensor_order(s)) " *
                "constituents, so it cannot serve an order-$O bound. Build one " *
                "object per physics."
        )
    )
    fr = layer_fractions(s)
    return sum(fr[ℓ] * f(s.props[ℓ]) for ℓ in 1:_nlayers(s))
end

Schemes.has_layer_average(::FEAxiLayeredSpheroid) = true
Schemes._layer_voigt(s::FEAxiLayeredSpheroid, ref::TensND.AbstractTens) =
    _layer_average(s, ref, identity)
Schemes._layer_reuss(s::FEAxiLayeredSpheroid, ref::TensND.AbstractTens) =
    _layer_average(s, ref, inv)

fe_axi_localization(s::FEAxiLayeredSpheroid, P₀::TensND.AbstractTens; kw...) =
    _fe_axi_localization(s, P₀; kw...)

_fe_axi_localization(
    s::FEAxiLayeredSpheroid, P₀::TensND.AbstractTens; kw...
) = let res = get!(() -> _axi_run(s, P₀), s.cache.tensors, _fe_cache_key(P₀))
    (res.A, res.B)
end

"Bypass the cache and return every intermediate, including the uncorrected pair."
fe_axi_breakdown(s::FEAxiLayeredSpheroid, P₀::TensND.AbstractTens) = _axi_run(s, P₀)

# ─── Gate B — both localization tensors, from one solve ──────────────────────
#
#  The stress side is *not* `ℂ₁ : 𝔸_εε` for any single `ℂ₁`, so it is measured
#  on the finite-element solution beside the strain side. The three-argument
#  scheme signature carries a phase property this inclusion has no use for —
#  its constituents live in the geometry object — so it is accepted and ignored,
#  as for `LayeredSphere`.

for (gen, order, idx) in (
        (:(Core.strain_strain_loc), 4, 1),
        (:(Core.stress_strain_loc), 4, 2),
        (:(Core.gradient_gradient_loc), 2, 1),
        (:(Core.flux_gradient_loc), 2, 2),
    )
    @eval $gen(
        s::FEAxiLayeredSpheroid,
        ::TensND.AbstractTens{$order, 3},
        P₀::TensND.AbstractTens{$order, 3};
        kw...
    ) = _fe_axi_localization(s, P₀; kw...)[$idx]
end

"""
    fe_axi_mesh_report(incl::FEAxiLayeredSpheroid) -> NamedTuple

Mesh diagnostics: cell and node counts, and each layer's measured volume of
revolution against its closed form.

The closed form is `4πa²c/3` of the layer's own boundary minus the one inside
it, which exists for **any** semi-axes — confocal or not. So this is the check
that remains available on a geometry with no analytic solution at all, and it is
worth running before a long solve on a nest that has not been tried: the
`volume_error` column catches a mesh that failed to resolve a thin layer, which
otherwise looks perfectly reasonable.

The volumes are measured on the geometric interpolation with a quadrature of its
own, so they do not match the solver's `volume` to the last digit; both converge
to the same limit.
"""
function fe_axi_mesh_report(incl::FEAxiLayeredSpheroid)
    st = _axi_setup(incl)
    counts = fe_axi_grid_counts(st.backend, st.grid)
    vol(set) = fe_axi_region_volume(st.backend, st.grid, set)
    n = _nlayers(incl)
    exact = layer_volumes(incl)
    measured = [vol(axi_layer_set(ℓ)) for ℓ in 1:n]
    R = incl.mesh.radius_ratio *
        max(maximum(incl.axis_radii), maximum(incl.disk_radii))
    v_matrix = vol(AXI_SET_MATRIX)
    return (;
        backend = st.backend,
        ncells = counts.ncells,
        nnodes = counts.nnodes,
        ncells_by_layer = [counts.ncells_by_set[axi_layer_set(ℓ)] for ℓ in 1:n],
        ncells_matrix = counts.ncells_by_set[AXI_SET_MATRIX],
        volume_layers = measured,
        volume_layers_exact = exact,
        volume_error = maximum(
            abs(measured[ℓ] - exact[ℓ]) / exact[ℓ] for ℓ in 1:n
        ),
        volume_cell = sum(measured) + v_matrix,
        volume_cell_exact = 4π * R^3 / 3,
    )
end

# ─── The two physics ─────────────────────────────────────────────────────────

_axi_run(s::FEAxiLayeredSpheroid, C₀::TensND.AbstractTens{4, 3}) =
    _axi_layered_run_elastic(s, C₀)
_axi_run(s::FEAxiLayeredSpheroid, K₀::TensND.AbstractTens{2, 3}) =
    _axi_layered_run_cond(s, K₀)

"The region list the inclusion average runs over: every layer, matrix excluded."
_axi_layer_sets(s::FEAxiLayeredSpheroid) =
    ntuple(ℓ -> axi_layer_set(ℓ), _nlayers(s))

function _axi_layered_run_elastic(
        s::FEAxiLayeredSpheroid, C₀::TensND.AbstractTens{4, 3}
    )
    μ, ν = _fe_iso_moduli(C₀; what = "`FEAxiLayeredSpheroid`")
    R = _fe_frame(s)
    C0_66 = Core.mandel66_minor(_to_local4(C₀, R))
    Dmap = vcat(
        [
            axi_layer_set(ℓ) => _axi_check_ti(
                Core.mandel66_minor(_to_local4(s.props[ℓ], R)), "layer $ℓ"
            ) for ℓ in 1:_nlayers(s)
        ],
        [AXI_SET_MATRIX => C0_66],
    )
    backend = _axi_setup(s).backend
    sets = _axi_layer_sets(s)

    blocks = map((0, 1, 2)) do m
        mode = _axi_mode_setup(s, :elasticity, m)
        nload = m == 0 ? 2 : 1
        A_E, B_E, A_p, B_p, V = _axi_solve_mode(
            backend, mode, Dmap,
            (N, dNρ, dNz, ρ) -> _axi_B_elast(m, N, dNρ, dNz, ρ),
            v -> _axi_project(m, v), nload,
            j -> (x -> _axi_bc_affine(m, j, x[1], x[2])),
            (j, V) -> (x -> _axi_bc_dipole(m, j, x[1], x[2], μ, ν, V)),
            sets,
        )
        cols = m == 0 ? (1:2) : m == 1 ? (3:3) : (5:5)
        Qm = _AXI_Q[:, cols]
        A, B, X = _axi_correct(A_E, B_E, A_p, B_p, Qm' * C0_66 * Qm)
        (; m, A_E, B_E, A_p, B_p, A, B, X, V)
    end

    A66 = _axi_assemble_66(blocks[1].A, blocks[2].A[1, 1], blocks[3].A[1, 1])
    B66 = _axi_assemble_66(blocks[1].B, blocks[2].B[1, 1], blocks[3].B[1, 1])
    A66_E = _axi_assemble_66(blocks[1].A_E, blocks[2].A_E[1, 1], blocks[3].A_E[1, 1])
    B66_E = _axi_assemble_66(blocks[1].B_E, blocks[2].B_E[1, 1], blocks[3].B_E[1, 1])
    s.cache.assemblies += 1
    return (;
        A = _from_local66(A66, R), B = _from_local66(B66, R),
        A_uncorrected = _from_local66(A66_E, R),
        B_uncorrected = _from_local66(B66_E, R),
        blocks, volume = blocks[1].V,
    )
end

function _axi_layered_run_cond(
        s::FEAxiLayeredSpheroid, K₀::TensND.AbstractTens{2, 3}
    )
    k₀ = _fe_iso_scalar(K₀; what = "`FEAxiLayeredSpheroid`")
    R = _fe_frame(s)
    Dmap = vcat(
        [
            axi_layer_set(ℓ) => _axi_check_ti2(_to_local2(s.props[ℓ], R), "layer $ℓ")
                for ℓ in 1:_nlayers(s)
        ],
        [AXI_SET_MATRIX => _to_local2(K₀, R)],
    )
    backend = _axi_setup(s).backend
    sets = _axi_layer_sets(s)

    blocks = map((0, 1)) do m
        mode = _axi_mode_setup(s, :conduction, m)
        A_E, B_E, A_p, B_p, V = _axi_solve_mode(
            backend, mode, Dmap,
            (N, dNρ, dNz, ρ) -> _axi_B_cond(m, N, dNρ, dNz, ρ),
            g -> _axi_project_cond(m, g), 1,
            _j -> (x -> (_axi_bc_affine_cond(m, x[1], x[2]),)),
            (_j, V) -> (x -> (_axi_bc_dipole_cond(m, x[1], x[2], k₀, V),)),
            sets,
        )
        A, B, X = _axi_correct(A_E, B_E, A_p, B_p, fill(k₀, 1, 1))
        (; m, A_E, B_E, A_p, B_p, A, B, X, V)
    end

    # Mode 1 carries the transverse response and mode 0 the axial one, so the
    # diagonal is (t, t, a) about the revolution axis — the same reassembly the
    # core-shell driver does, and the same argument order.
    diag3(t, a) = [t 0.0 0.0; 0.0 t 0.0; 0.0 0.0 a]
    A33 = diag3(blocks[2].A[1, 1], blocks[1].A[1, 1])
    B33 = diag3(blocks[2].B[1, 1], blocks[1].B[1, 1])
    A33_E = diag3(blocks[2].A_E[1, 1], blocks[1].A_E[1, 1])
    B33_E = diag3(blocks[2].B_E[1, 1], blocks[1].B_E[1, 1])
    s.cache.assemblies += 1
    return (;
        A = _from_local33(A33, R), B = _from_local33(B33, R),
        A_uncorrected = _from_local33(A33_E, R),
        B_uncorrected = _from_local33(B33_E, R),
        blocks, volume = blocks[1].V,
    )
end
