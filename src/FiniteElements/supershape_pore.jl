# =============================================================================
#  supershape_pore.jl — a cavity of superspherical or superspheroidal shape.
#
#  The inclusion type that puts the cell of `cell_driver.jl` into a scheme.
#
#  ## Why it enters as a *heterogeneous* inclusion, and why that is not a lie
#
#  `is_homogeneous_inclusion` asks whether a single `ℂ₁` describes the interior,
#  because that is what `(ℂ₁ - ℂ₀):𝔸` and `inv(ℂ₁)` need.  For a cavity there is
#  none: `ℂ₁ = 0` and `inv(ℂ₁)` is meaningless.  So the answer is `false`, and
#  the package's own exact identities take over,
#
#      ℕ = 𝔸_σε - ℂ₀:𝔸_εε ,        ℍ = (𝔸_εε - 𝕊₀:𝔸_σε):𝕊₀ ,
#
#  in which the stress-side localization of a cavity is **exactly zero** — no
#  stress inside, by definition.  They collapse to `ℕ = -ℂ₀:𝔸` and `ℍ = 𝔸:𝕊₀`,
#  which is the right answer for a pore, reached with no `ℂ₁` anywhere in it.
#
#  That is worth spelling out because the obvious alternative is worse. Calling
#  a cavity homogeneous — it is, after all, uniformly empty — sends every scheme
#  down `(ℂ₁ - ℂ₀):𝔸` with whatever soft-but-not-zero stiffness the user gave
#  the phase, and the package's own porous tutorial uses `iso_stiffness(0.01,
#  0.005)` against a matrix near `0.83`.  The answer then carries a percent-level
#  error that has nothing to do with the mesh, and no diagnostic reveals it.
#  Declaring the truth about `inv(ℂ₁)` avoids the whole problem, and the phase
#  property becomes what it should be: ignored.
# =============================================================================

"""
    SupershapePoreShape

[`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait) of
[`FESupershapePore`](@ref). No kernel dispatches on it — the inclusion supplies
its own localization tensors.
"""
struct SupershapePoreShape end

"""
    FESupershapePore(shape; opts = FECellMeshOptions(), basis = nothing,
                     euler_angles = (), backend = AutoBackend())

A **cavity** of superspherical or superspheroidal shape, whose response is
computed on the truncated cell with a corrected boundary condition.

`shape` is a [`Supersphere`](@ref MeanFieldHomogenization.Superspheres.Supersphere)
or a [`Superspheroid`](@ref MeanFieldHomogenization.Superspheres.Superspheroid)
— see those for the ``2p`` convention. The phase property handed to
`add_phase!` is a **placeholder and is ignored**: the solve is the exact cavity
problem, and there is no ``\\mathbb C_1`` in the answer. Pass the matrix's own
stiffness and nothing is lost.

Both physics are served by the same object and the same mesh — a scalar solve
for conduction, a vector one for elasticity — so one type covers
`homogenize(rve, scheme, :C)` and `homogenize(rve, scheme, :K)`.

The reference medium must be **isotropic**: the corrected boundary condition
uses the closed-form Kelvin dipole field. Under an iterative scheme, whose
running estimate for a cubic inclusion is cubic and not isotropic, add
`symmetrize = IsoSymmetrize()` to the phase — the error message says so rather
than letting the scheme converge to something wrong.

Requires a finite-element backend: `import Ferrite, FerriteGmsh, Gmsh`.

# Symmetry

A supersphere in an isotropic matrix is **cubic**, so its compliance
contribution has three independent constants and not two. A superspheroid is
transversely isotropic about ``\\underline e_3``. Neither class is projected
onto by default: the raw tensors come out, and
[`cubic_residual`](@ref MeanFieldHomogenization.Elasticity.cubic_residual) is the free error bar on the cubic case — the distance
to a class the answer belongs to by group theory is discretization error and
nothing else.

# Example

```julia
using MeanFieldHomogenization
import Ferrite, FerriteGmsh, Gmsh

pore = FESupershapePore(Supersphere(1.0, 0.6);
                        opts = FECellMeshOptions(; level = 3, radius_ratio = 3.0))

C₀ = iso_stiffness(0.8333, 0.3846)          # E = 1, ν = 0.3
rve = RVE()
add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :pores, pore, Dict(:C => C₀); fraction = 0.05)
homogenize(rve, MoriTanaka(), :C)
```

See also [`FECellMeshOptions`](@ref), [`fe_cell_localization`](@ref),
[`fe_cell_mesh_report`](@ref).
"""
struct FESupershapePore{
        T <: Number, S <: AbstractSuperShape{T}, B <: TensND.AbstractBasis,
    } <: Core.AbstractCustomInclusion{T}
    shape::S
    basis::B
    mesh::FECellMeshOptions
    cache::FECache
    backend::FEBackend
end

function FESupershapePore(
        shape::AbstractSuperShape{T};
        opts::FECellMeshOptions = FECellMeshOptions(),
        basis::Union{Nothing, TensND.AbstractBasis} = nothing,
        euler_angles::Tuple{Vararg{Real}} = (),
        backend::FEBackend = AutoBackend(),
    ) where {T}
    bas = basis === nothing ? Core._default_basis(Float64, euler_angles) : basis
    return FESupershapePore{T, typeof(shape), typeof(bas)}(
        shape, bas, opts, FECache(), backend
    )
end

function Base.show(io::IO, s::FESupershapePore)
    return print(io, "FESupershapePore(", s.shape, ")")
end

# ─── Level 0 of the contract ─────────────────────────────────────────────────

Core.dimension(::FESupershapePore) = 3
Core.inclusion_basis(s::FESupershapePore) = s.basis
Core.shape_trait(::FESupershapePore) = SupershapePoreShape()
# See the header: `false` is the statement that `inv(ℂ₁)` is meaningless, which
# for a cavity it is — and it is what routes every contribution tensor through
# the exact identities instead of through the phase property.
Core.is_homogeneous_inclusion(::FESupershapePore) = false
_fe_cache(s::FESupershapePore) = s.cache

"""
    Core.shape_tensor(pore)

The semi-axes on the diagonal, in the inclusion's own basis: ``a`` three times
for a supersphere, ``(a, a, c)`` for a superspheroid. It describes the bounding
box of the family, not the body — a supershape has no shape tensor in the
ellipsoidal sense, which is the whole reason it is here.
"""
function Core.shape_tensor(s::FESupershapePore{T}) where {T}
    D = zeros(T, 3, 3)
    D[1, 1] = D[2, 2] = s.shape.a
    D[3, 3] = _pore_polar_semiaxis(s.shape)
    return TensND.Tens(D, s.basis)
end

_pore_polar_semiaxis(shape::Supersphere) = shape.a
_pore_polar_semiaxis(shape::Superspheroid) = shape.c

# ─── The memoized solve ──────────────────────────────────────────────────────

"""
    fe_cell_localization(pore, P₀) -> A

The localization tensor of the cavity — ``\\mathbb A_{\\varepsilon\\varepsilon}``
for a fourth-order reference, ``\\boldsymbol A_{\\nabla\\nabla}`` for a
second-order one — in the **global** frame.

Memoized on the reference medium expressed in the pore's own frame, which has a
useful consequence: a whole family of *orientations* of the same shape in the
same matrix shares one finite-element resolution. So a one-shot scheme costs a
single solve even under an orientation average.
"""
fe_cell_localization(s::FESupershapePore, P₀::TensND.AbstractTens) =
    _fe_cell_cached(s, P₀)

function _fe_cell_cached(s::FESupershapePore, P₀::TensND.AbstractTens)
    return get!(() -> _fe_cell_run(s, P₀), s.cache.tensors, _fe_cache_key(P₀))
end

"Build the cell once and keep it: the mesh does not depend on the reference."
function _fe_cell_setup(s::FESupershapePore, ncomp::Int)
    b = _resolve_backend(s.backend)
    if s.cache.setup === nothing
        g = fe_cell_grid(b, s.shape, s.mesh)
        s.cache.setup = (; backend = b, g.grid, g.built, spaces = Dict{Int, Any}())
    end
    st = s.cache.setup
    space = get!(st.spaces, ncomp) do
        fe_cell_space(st.backend, st.grid, s.mesh.order, ncomp)
    end
    ndofs, _, _ = fe_cell_dof_split(st.backend, space)
    _fe_check_budget(
        ndofs; max_dofs = s.mesh.max_dofs, min_free_gb = s.mesh.min_free_gb,
        what = ncomp == 1 ? "conduction solve" : "elasticity solve",
    )
    return (st.backend, space, st.built)
end

function _fe_cell_run(s::FESupershapePore, C₀::TensND.AbstractTens{4, 3})
    μ, ν = _fe_iso_moduli(C₀; what = "`FESupershapePore`")
    R = _fe_frame(s)
    backend, space, built = _fe_cell_setup(s, 3)
    L = _to_local4(C₀, R)
    Cloc = Tensors.SymmetricTensor{4, 3}((i, j, k, l) -> L[i, j, k, l])
    r = _cell_elastic_localization(
        backend, space, Cloc, μ, ν, built.cavity_volume
    )
    s.cache.assemblies += 1
    return _from_local66(r.A, R)
end

function _fe_cell_run(s::FESupershapePore, K₀::TensND.AbstractTens{2, 3})
    k₀ = _fe_iso_scalar(K₀; what = "`FESupershapePore`")
    R = _fe_frame(s)
    backend, space, built = _fe_cell_setup(s, 1)
    r = _cell_conduction_localization(backend, space, k₀, built.cavity_volume)
    s.cache.assemblies += 1
    return _from_local33(collect(r.A), R)
end

# ─── Gate B, with an exactly zero stress side ────────────────────────────────
#
#  The three-argument scheme signature carries a phase property this inclusion
#  has no use for, so the middle argument is accepted and ignored — as for
#  `LayeredSphere` and `FEExcenteredSphere`.

Core.strain_strain_loc(
    s::FESupershapePore, ::TensND.AbstractTens{4, 3}, C₀::TensND.AbstractTens{4, 3}; kw...
) = _fe_cell_cached(s, C₀)

Core.gradient_gradient_loc(
    s::FESupershapePore, ::TensND.AbstractTens{2, 3}, K₀::TensND.AbstractTens{2, 3}; kw...
) = _fe_cell_cached(s, K₀)

"""
    Core.stress_strain_loc(pore, C₁, C₀)
    Core.flux_gradient_loc(pore, K₁, K₀)

**Exactly zero.** A cavity carries no stress and no flux, so its stress-side
localization vanishes identically — not to the discretization error, but as a
matter of definition.

This is what makes the package's exact contribution identities collapse to
``\\mathbb N = -\\mathbb C_0:\\mathbb A`` and
``\\mathbb H = \\mathbb A:\\mathbb S_0``, and it is why no contribution tensor
is overridden here: the generic route is already the right one.
"""
Core.stress_strain_loc(
    ::FESupershapePore, ::TensND.AbstractTens{4, 3}, C₀::TensND.AbstractTens{4, 3}; kw...
) = TensND.TensCanonical(Tensors.Tensor{4, 3}((i, j, k, l) -> 0.0))

Core.flux_gradient_loc(
    ::FESupershapePore, ::TensND.AbstractTens{2, 3}, K₀::TensND.AbstractTens{2, 3}; kw...
) = TensND.TensCanonical(Tensors.Tensor{2, 3}((i, j) -> 0.0))

# ─── Diagnostics ─────────────────────────────────────────────────────────────

"""
    fe_cell_mesh_report(pore) -> NamedTuple

What the mesh actually is: cell and node counts, the two boundary areas, the
radius, the applied sizes, the snapping report, and the **three** volumes worth
comparing — the closed form, the flat triangulation it was built from, and the
curved boundary the solve sees.

Those three are the geometry error, made visible. The gap between the flat and
the curved volume is what snapping the mid-edge nodes bought; the gap between
the curved volume and the closed form is what is left.
"""
function fe_cell_mesh_report(s::FESupershapePore)
    backend, _, built = _fe_cell_setup(s, 1)
    st = s.cache.setup
    counts = fe_cell_counts(backend, st.grid)
    Vex = shape_volume(s.shape)
    return (;
        counts.ncells, counts.nnodes,
        counts.area_inclusion, counts.area_outer,
        area_outer_exact = 4π * built.R^2,
        R = built.R, h_in = built.h_in, h_out = built.h_out,
        snap = built.snap,
        volume_exact = Vex,
        volume_flat = mesh_volume(built.inner),
        volume_curved = built.cavity_volume,
        volume_error = (built.cavity_volume - Vex) / Vex,
        surface_triangles = triangle_count(built.inner),
        inclusion_quality = mesh_quality(built.inner),
    )
end
