# =============================================================================
#  backends.jl — which finite-element library performs the solve.
#
#  The scientific content of the axisymmetric solver — the Fourier operators,
#  the boundary data, the fixed point of the corrected boundary condition, the
#  memoization — lives in `src/`.  What a backend supplies is only the
#  discretization: a mesh, scalar Lagrange spaces, an assembly, a Dirichlet
#  lift and a quadrature.  That is the nine functions declared below.
#
#  A backend never sees a Fourier mode or a physics: the driver closes `Bop`
#  and `proj` over the mode before handing them over, so the backend is told
#  only "this many scalar fields, this operator, this projection".
# =============================================================================

"""
    FEBackend

Which finite-element library performs the solve. Concrete singletons:
[`FerriteBackend`](@ref), [`GridapBackend`](@ref) and the default
[`AutoBackend`](@ref).

The three are always defined, whether or not the corresponding package is
loaded — naming a backend costs nothing, only *solving* with it requires the
matching extension.
"""
abstract type FEBackend end

"""
    AutoBackend()

Default backend of a finite-element inclusion: pick whichever backend is
loaded, **at the first solve** rather than at construction.

Deferring the choice is what lets an inclusion be built, printed, stored in an
`RVE` and passed around in a session where no finite-element package has been
imported; the informative error arrives only when a scheme actually asks for a
localization tensor.

Priority is `FerriteBackend` then `GridapBackend`. Loading both is not an
error — it just means the first is chosen. To pick deliberately, pass
`backend = GridapBackend()` to the constructor.
"""
struct AutoBackend <: FEBackend end

"""
    FerriteBackend()

Solve with [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl); needs
`import Ferrite, FerriteGmsh, Gmsh`. The reference implementation, and the
faster of the two to run.
"""
struct FerriteBackend <: FEBackend end

"""
    GridapBackend()

Solve with [Gridap.jl](https://gridap.github.io/Gridap.jl); needs
`import Gridap, GridapGmsh` (GridapGmsh carries its own `gmsh`, so `Gmsh.jl`
is not required on this path).

Gridap states the weak form directly — `∫( ε(v) ⊙ (σ∘ε(u)) )dΩ` for the crack,
`∫( Bᵐ(v)' * D * Bᵐ(u) * ρ )dΩ` for the axisymmetric modes — which makes it the
easier of the two to read and to modify.
"""
struct GridapBackend <: FEBackend end

_backend_extension(::FerriteBackend) = :MeanFieldHomogenizationFerriteExt
_backend_extension(::GridapBackend) = :MeanFieldHomogenizationGridapExt

_backend_import(::FerriteBackend) = "import Ferrite, FerriteGmsh, Gmsh"
_backend_import(::GridapBackend) = "import Gridap, GridapGmsh"

_backend_loaded(b::FEBackend) =
    Base.get_extension(parentmodule(@__MODULE__), _backend_extension(b)) !== nothing

const _AXI_BACKENDS = (FerriteBackend(), GridapBackend())

"""
    _resolve_backend(b) -> FEBackend

Turn [`AutoBackend`](@ref) into a concrete backend, and check that a concrete
one is actually available. Called once per inclusion, at the first solve; the
result is pinned in the cache, so a single inclusion never mixes two backends'
grids.
"""
function _resolve_backend(b::FEBackend)
    _backend_loaded(b) || error(
        "the $(nameof(typeof(b))) finite-element backend is not loaded: " *
            "run `$(_backend_import(b))` first."
    )
    return b
end

function _resolve_backend(::AutoBackend)
    for b in _AXI_BACKENDS
        _backend_loaded(b) && return b
    end
    return error(
        "no finite-element backend is loaded: run `" *
            join(map(_backend_import, _AXI_BACKENDS), "` or `") * "` first."
    )
end

# ─── The backend contract ────────────────────────────────────────────────────
#
#  Nine functions, each with a fallback that names the offending backend.  A
#  new backend is exactly this list; nothing else in the package needs to know
#  that it exists.

_no_backend_method(f, b) = error(
    "`$f` is not implemented for $(nameof(typeof(b))). Either the extension " *
        "failed to load (`$(_backend_import(b))`), or this backend does not " *
        "support this inclusion."
)

"""
    fe_axi_grid(backend, incl)

Backend-native mesh of the meridian half-plane of `incl`, carrying the cell
sets `"core"`, `"shell"`, `"matrix"` and the boundary sets `"outer"`, `"axis"`
of [`_build_gmsh_axi_model`](@ref). The first node coordinate is the
cylindrical radius `ρ`, the second the axial coordinate `z`.
"""
fe_axi_grid(b::FEBackend, incl) = _no_backend_method("fe_axi_grid", b)

"""
    fe_axi_grid_counts(backend, grid) -> (; ncells, nnodes, ncells_by_set)

Cell and node counts, `ncells_by_set` being a `Dict{String,Int}` over the three
regions. Diagnostics only.
"""
fe_axi_grid_counts(b::FEBackend, grid) = _no_backend_method("fe_axi_grid_counts", b)

"""
    fe_axi_region_volume(backend, grid, set) -> Float64

Volume of revolution `2π ∫_set ρ dρ dz` of one region, on the geometric
interpolation of the mesh. Diagnostics only — the driver measures its own
volume with the mode's quadrature, and the two need not agree exactly.
"""
fe_axi_region_volume(b::FEBackend, grid, set) =
    _no_backend_method("fe_axi_region_volume", b)

"""
    fe_axi_mode(backend, grid, order, ncomp, axis_zeros) -> mode

Discretize one Fourier mode: `ncomp` scalar Lagrange fields of degree `order`
on `grid`, with a quadrature exact to degree `2 * order + 1`.

The dof numbering must span the **whole** space, with no Dirichlet elimination
— the driver does the free/prescribed split itself, so that one factorization
serves every right-hand side.

`axis_zeros` lists the component indices that the axis regularity of this mode
forces to vanish on the set `"axis"`.
"""
fe_axi_mode(b::FEBackend, grid, order, ncomp, axis_zeros) =
    _no_backend_method("fe_axi_mode", b)

"""
    fe_axi_dof_split(backend, mode) -> (ndofs, free, presc)

Total dof count and the two index vectors, in the numbering of
[`fe_axi_mode`](@ref). `presc` is the sorted union of the outer-boundary dofs
and the axis-pinned dofs; `free` is its complement.
"""
fe_axi_dof_split(b::FEBackend, mode) = _no_backend_method("fe_axi_dof_split", b)

"""
    fe_axi_set_dirichlet!(backend, mode, u, f) -> u

Write the Dirichlet data of one right-hand side into `u`: `u[d] = f(ρ_d, z_d)[k]`
for every dof `d` of the outer boundary, `k` being its component, **then**
`u[d] = 0` for every axis-pinned dof.

Two things make this the delicate function of the contract.

*The order matters.* The poles `(0, ±R)` belong to both the `"outer"` and the
`"axis"` sets. The axis must win, so the zeroing comes second.

*It must not touch the matrix.* The driver assembles and factorizes once, then
calls this once per load case. Re-assembling here would multiply the cost of a
solve by the number of loads.
"""
fe_axi_set_dirichlet!(b::FEBackend, mode, u, f) =
    _no_backend_method("fe_axi_set_dirichlet!", b)

"""
    fe_axi_stiffness(backend, mode, Dmap, Bop) -> AbstractMatrix

Stiffness of one mode over the whole dof numbering,

```
K = Σ_regions ∫_region Bop(v)' * D_region * Bop(u) * ρ dρ dz .
```

`Dmap` is a `Vector{Pair{String,Matrix{Float64}}}` — a vector, not a `Dict`, so
that the assembly order is reproducible — mapping a cell-set name to that
region's material matrix in the cylindrical `(ρ, θ, z)` basis.

`Bop(N, dNρ, dNz, ρ) -> Matrix{Float64}` of size `nrow × ncomp` is the
generalized-strain operator of one scalar shape function, already closed over
the Fourier mode. It is **R-linear in `(N, dNρ, dNz)`**, so a backend that
manipulates whole trial functions rather than shape functions may apply it to
`(u_c, ∂ρu_c, ∂zu_c)` directly instead of building an element `B` matrix.

The `ρ` in the measure is the single factor that turns a plane problem into a
solid of revolution.
"""
fe_axi_stiffness(b::FEBackend, mode, Dmap, Bop) =
    _no_backend_method("fe_axi_stiffness", b)

"""
    fe_axi_average(backend, mode, Dmap, u, Bop, proj, sets) -> (prim, dual, V)

Volume averages over `∪ sets` of the generalized strain and of the associated
generalized stress, both projected by `proj` onto the Kelvin basis of the mode,
plus the volume of revolution `V = 2π ∫ ρ dρ dz` of that union:

```
prim = ∫ proj(B u) ρ / ∫ ρ ,      dual = ∫ proj(D · B u) ρ / ∫ ρ .
```

The azimuthal integration has already been performed analytically inside
`proj`; what remains is the meridian quadrature.
"""
fe_axi_average(b::FEBackend, mode, Dmap, u, Bop, proj, sets) =
    _no_backend_method("fe_axi_average", b)

# ─── The crack contract ──────────────────────────────────────────────────────
#
#  Seven more methods, the flat-crack counterpart of the nine above.  The
#  problem is simpler — one vector field, one material, Dirichlet on the outer
#  sphere and nothing else — so the seam is narrower.

"""
    fe_crack_grid(backend, crack)

Backend-native mesh of the ball holding the crack, with the boundary sets
`"outer"` (the sphere) and `"crack"` (both lips, whose nodes the gmsh `Crack`
plugin has duplicated). The crack front must already be welded — see
[`_weld_msh_crack_front`](@ref).
"""
fe_crack_grid(b::FEBackend, crack) = _no_backend_method("fe_crack_grid", b)

"""
    fe_crack_counts(backend, grid) -> (; ncells, nnodes, nfacets_up, nfacets_dn,
                                         area_up, area_dn)

Mesh diagnostics: cell and node counts, and the facet count and area of each
lip. The two areas must both equal `πab` — that is what says the plugin split
the surface cleanly and the front weld did not glue the lips back together.
"""
fe_crack_counts(b::FEBackend, grid) = _no_backend_method("fe_crack_counts", b)

"""
    fe_crack_space(backend, grid, order) -> space

A vector-valued Lagrange space of degree `order` over the whole mesh, with a
quadrature exact to degree `2 * order` and **no** Dirichlet elimination — the
driver splits the dofs itself so that one factorization serves all six
right-hand sides.

The lips need no special treatment: they are traction-free naturally, because
their nodes are distinct. There is no interface term, no multiplier and no
contact condition anywhere in this problem.
"""
fe_crack_space(b::FEBackend, grid, order) = _no_backend_method("fe_crack_space", b)

"""
    fe_crack_dof_split(backend, space) -> (ndofs, free, presc)

Total dof count and the two index vectors, `presc` being the dofs of the outer
sphere.
"""
fe_crack_dof_split(b::FEBackend, space) = _no_backend_method("fe_crack_dof_split", b)

"""
    fe_crack_set_dirichlet!(backend, space, u, f) -> u

Write `u[d] = f(x_d)[k]` for every dof `d` of the outer sphere, `x_d` its node
and `k` its component. Called once per right-hand side; must not touch the
matrix.
"""
fe_crack_set_dirichlet!(b::FEBackend, space, u, f) =
    _no_backend_method("fe_crack_set_dirichlet!", b)

"""
    fe_crack_stiffness(backend, space, C) -> AbstractMatrix

Stiffness of linear elasticity, `∫ ε(v) : ℂ : ε(u) dΩ`, over the whole dof
numbering.

`C` is a `Tensors.SymmetricTensor{4,3}` and is **isotropic**: the corrected
boundary condition uses the closed-form Kelvin dipole field, so the driver
refuses anything else long before reaching here. A backend may therefore work
from `(λ, μ)` instead of from the full tensor.
"""
fe_crack_stiffness(b::FEBackend, space, C) = _no_backend_method("fe_crack_stiffness", b)

"""
    fe_crack_mean_jump(backend, space, u, S_f, b) -> Vector{3}

`⟨[[u]]⟩ / b`, the opening averaged over the crack surface and normalized by
the semi-minor axis — the convention of `cod_tensor`.

Measured as a surface integral of the trace of `u` on each lip, with no
assumption on the opening profile. The two lips are told apart by the sign of
`n ⋅ e₃`, `n` being the outward normal of the adjacent element: the lip whose
element sits above the crack carries `n = -e₃` and contributes `+u`.
"""
fe_crack_mean_jump(bk::FEBackend, space, u, S_f, b) =
    _no_backend_method("fe_crack_mean_jump", bk)

# ─── The cell contract ───────────────────────────────────────────────────────
#
#  Eight more methods, for the three-dimensional cell around a non-ellipsoidal
#  shape.  Between the two families above in width: one field over one region,
#  as the crack has, but two physics — a scalar temperature and a vector
#  displacement — sharing every function except the two that know what is being
#  integrated.
#
#  As in the other two families, the inclusion is **not meshed**. A pore's
#  boundary is naturally flux-free in conduction and traction-free in
#  elasticity, so the meshed region is the matrix shell alone and the inclusion
#  surface is simply a piece of boundary with nothing prescribed on it.

"""
    fe_cell_grid(backend, shape, opts)

Backend-native mesh of the cell around `shape`, carrying the cell set
`"matrix"` and the boundary sets `"inclusion"` and `"outer"` of
[`_build_gmsh_cell_model`](@ref).

The mesh is **second order with a curved boundary**: the mid-edge nodes of the
inclusion surface have been moved onto the exact shape, so the geometric
interpolation must be quadratic too. A backend that silently used a linear
geometry here would throw away the whole gain and report a plausible wrong
answer.
"""
fe_cell_grid(b::FEBackend, shape, opts) = _no_backend_method("fe_cell_grid", b)

"""
    fe_cell_counts(backend, grid) -> (; ncells, nnodes, area_inclusion, area_outer)

Mesh diagnostics: cell and node counts, and the area of each boundary set. The
outer area is compared against ``4\\pi R^2`` — it is what says the outer
surface was snapped onto the sphere rather than left a polyhedron.
"""
fe_cell_counts(b::FEBackend, grid) = _no_backend_method("fe_cell_counts", b)

"""
    fe_cell_space(backend, grid, order, ncomp) -> space

`ncomp` scalar Lagrange fields of degree `order` on `grid` — `ncomp = 1` for
conduction, `3` for elasticity — with a quadrature exact to degree
`2 * order`, a facet quadrature of the same degree, and a **quadratic geometric
interpolation** to follow the curved boundary.

The dof numbering must span the whole space, with no Dirichlet elimination: the
driver splits the dofs itself so that one factorization serves all twelve
right-hand sides.
"""
fe_cell_space(b::FEBackend, grid, order, ncomp) =
    _no_backend_method("fe_cell_space", b)

"""
    fe_cell_dof_split(backend, space) -> (ndofs, free, presc)

Total dof count and the two index vectors, `presc` being the dofs of the
`"outer"` boundary and nothing else. The inclusion surface carries no condition
at all — that is what makes it a pore.
"""
fe_cell_dof_split(b::FEBackend, space) = _no_backend_method("fe_cell_dof_split", b)

# Octant variants. `nothing` is the full cell and falls straight through, so the
# two paths share one call site in the driver; a parity class is the octant, and
# a backend that has not implemented it says so rather than quietly ignoring it.
fe_cell_dof_split(b::FEBackend, space, ::Nothing) = fe_cell_dof_split(b, space)
fe_cell_dof_split(b::FEBackend, space, χ) = _no_backend_method("fe_cell_dof_split", b)
fe_cell_set_dirichlet!(b::FEBackend, space, u, f, ::Nothing) =
    fe_cell_set_dirichlet!(b, space, u, f)
fe_cell_set_dirichlet!(b::FEBackend, space, u, f, χ) =
    _no_backend_method("fe_cell_set_dirichlet!", b)

"""
    fe_cell_set_dirichlet!(backend, space, u, f) -> u

Write `u[d] = f(x_d)[k]` for every dof `d` of the `"outer"` boundary, `x_d`
being its node and `k` its component; for a scalar field `f` returns a number.

Called once per right-hand side and **must not touch the matrix**: the driver
assembles and factorizes once, then calls this twelve times.
"""
fe_cell_set_dirichlet!(b::FEBackend, space, u, f) =
    _no_backend_method("fe_cell_set_dirichlet!", b)

"""
    fe_cell_stiffness(backend, space, material) -> AbstractMatrix

Stiffness over the whole dof numbering, over the matrix region:

```
conduction   K = ∫ ∇v ⋅ 𝐊 ⋅ ∇u dΩ         `material` a 3×3 matrix
elasticity   K = ∫ ε(v) : ℂ : ε(u) dΩ      `material` a SymmetricTensor{4,3}
```

Two methods on one name, told apart by the type of `material`. Both are
**isotropic**: the corrected boundary condition uses a closed-form dipole field,
and the driver refuses anything else long before reaching here, so a backend may
work from the scalar or from `(λ, μ)` instead of from the full object.
"""
fe_cell_stiffness(b::FEBackend, space, material) =
    _no_backend_method("fe_cell_stiffness", b)

"""
    fe_cell_mean_gradient(backend, space, u, V) -> NTuple{3}

``\\langle \\nabla T \\rangle`` over the **inclusion**, as the surface integral
``\\frac{1}{V}\\int_{\\partial\\mathcal I} T\\,\\underline n\\,\\mathrm dS``,
with `V` the volume of the inclusion.

The inclusion is not meshed, so its average is reached through the divergence
theorem on its boundary — which is also why it costs nothing beyond the solve.

**On the sign.** The normal a backend reports on a facet set points out of the
*meshed* region, hence **into** the pore, so it is the opposite of the
inclusion's own outward normal. The minus sign belongs here, and it is the same
one the crack and axisymmetric drivers carry in two different places for the
same reason. Getting it wrong is not invisible: the corrected answer then
carries exactly twice the truncation bias instead of none.
"""
fe_cell_mean_gradient(b::FEBackend, space, u, V) =
    _no_backend_method("fe_cell_mean_gradient", b)

"""
    fe_cell_mean_strain(backend, space, u, V) -> NTuple{6}

``\\langle \\varepsilon \\rangle`` over the inclusion in Kelvin-Mandel, as
``\\frac{1}{V}\\int_{\\partial\\mathcal I}
(\\underline u \\otimes \\underline n)^{\\mathrm s}\\,\\mathrm dS``.

Same normal convention, and the same minus sign, as
[`fe_cell_mean_gradient`](@ref).
"""
fe_cell_mean_strain(b::FEBackend, space, u, V) =
    _no_backend_method("fe_cell_mean_strain", b)
