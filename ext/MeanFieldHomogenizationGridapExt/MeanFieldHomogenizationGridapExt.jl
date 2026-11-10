"""
    MeanFieldHomogenizationGridapExt

Second finite-element backend of the axisymmetric Fourier solve, activated by
`import Gridap, GridapGmsh`.

Implements the nine methods of the [`MeanFieldHomogenization.FEBackend`](@ref) interface
for [`MeanFieldHomogenization.GridapBackend`](@ref), and nothing else: the Fourier
operators, the boundary data, the fixed point of the corrected boundary
condition and the memoization all live in `MeanFieldHomogenization.FiniteElements`, shared
with the Ferrite backend.

The point of having two is partly cross-validation — two unrelated
discretizations agreeing to a fraction of a percent is a much stronger
statement than one converging — and partly readability. Where Ferrite writes
an explicit element loop, Gridap states the weak form:

```julia
a(u, v) = ∫( Eᵐ(v) ⋅ (D ⋅ Eᵐ(u)) * ρ )dΩ
```

which is the equation of the axisymmetric problem, transcribed. Someone
adapting this to another morphology has one line of physics to change.

`GridapGmsh` carries its own copy of the gmsh API over the same `gmsh_jll`, so
`Gmsh.jl` is not needed on this path. The mesh itself is built by the shared
`MeanFieldHomogenization.FiniteElements._build_gmsh_axi_model`, written to a temporary
`.msh` file and read back by `GmshDiscreteModel`.
"""
module MeanFieldHomogenizationGridapExt

using MeanFieldHomogenization
using TensND
using Gridap
using Gridap.FESpaces: get_cell_dof_ids
using Gridap.Geometry: num_nodes
using Gridap.MultiField: MultiFieldFESpace
# `⊙` is exported by both Gridap and TensND; the explicit import settles it.
using Gridap.TensorValues: ⊙
using GridapGmsh
using GridapGmsh: gmsh
using LinearAlgebra
using Tensors

const FE = MeanFieldHomogenization.FiniteElements
const GB = FE.GridapBackend

# ─── Mesh ────────────────────────────────────────────────────────────────────

function FE.fe_axi_grid(::GB, incl::MeanFieldHomogenization.FEExcenteredSphere)
    a = Float64(incl.a)
    a_core = Float64(FE.core_radius(incl))
    d = Float64(FE.core_offset(incl))
    opts = incl.mesh
    R = opts.radius_ratio * a
    h_in = a / opts.nradial
    h_out = opts.coarsening * h_in

    path = joinpath(mktempdir(), "mfh_axi.msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        FE._build_gmsh_axi_model(gmsh, a, a_core, d, R, h_in, h_out)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    # `GmshDiscreteModel` runs its own gmsh session, hence the round trip
    # through a file rather than a handover of the live model; that session
    # also reports on stdout, which is why it is muted here.
    return redirect_stdout(devnull) do
        GmshDiscreteModel(path)
    end
end

FE.fe_axi_grid_counts(::GB, model) = (;
    ncells = num_cells(model),
    nnodes = num_nodes(model),
    ncells_by_set = Dict(
        s => num_cells(Triangulation(model, tags = s)) for
            s in (FE.AXI_SET_CORE, FE.AXI_SET_SHELL, FE.AXI_SET_MATRIX)
    ),
)

function FE.fe_axi_region_volume(::GB, model, set)
    Ω = Triangulation(model, tags = set)
    dΩ = Measure(Ω, 3)
    ρ = CellField(x -> x[1], Ω)
    return 2π * sum(∫(ρ)dΩ)
end

# ─── One mode ────────────────────────────────────────────────────────────────

"""
One Fourier mode, discretized with Gridap.

The scalar space carries **no** Dirichlet tags: every dof is free, and the
driver performs the free/prescribed split itself so that a single
factorization serves every right-hand side. The prescribed dofs are therefore
identified geometrically, from the cell-wise dof ids of the boundary
triangulations, and their coordinates read off two interpolations — exact on a
nodal Lagrange space, where interpolating `x -> x[1]` stores the abscissa of
each dof in that dof's slot.

Fields are stacked component-major by `MultiFieldFESpace`, so component `c`
occupies `(c-1) * nd1 .+ (1:nd1)` — the layout the generalized-strain operator
expects.
"""
struct GridapAxiMode{S, T, M, C}
    ncomp::Int
    nd1::Int
    ndofs::Int
    V::S
    U::T
    meas::M                                  # set name => (Measure, ρ CellField)
    coords::C                                # (ρ, z) of every scalar dof
    outer::Vector{Int}                       # scalar dofs on the outer boundary
    axis::Vector{Int}                        # global dofs pinned on the axis
    presc::Vector{Int}
    free::Vector{Int}
end

"""
    _boundary_dofs(V, model, reffe, tag) -> Vector{Int}

Indices, in the **unconstrained** numbering of `V`, of the scalar dofs lying on
the boundary set `tag`.

Two traps here. `get_cell_dof_ids(V, BoundaryTriangulation(...))` returns every
dof of each cell *adjacent* to the boundary, which would pin a whole layer of
elements to the analytical field instead of the boundary alone; the reliable
approach is a throwaway space tagged on that set — Gridap numbers its constrained
dofs negatively — compared cell by cell with the untagged space, whose local
dof ordering is identical because the reference element is.

And `tags` is plural because Gridap tags *entities*: the endpoints of a tagged
curve belong to point entities of their own and are not covered by the curve's
group, so the mesh declares them separately (`"outer_pts"`, `"axis_pts"`) and
both names must be passed.
"""
function _boundary_dofs(V, model, reffe, tags)
    Vd = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = tags)
    ids = Int[]
    for (free, tagged) in zip(get_cell_dof_ids(V), get_cell_dof_ids(Vd))
        for (i, t) in zip(free, tagged)
            t < 0 && push!(ids, i)
        end
    end
    return sort!(unique!(ids))
end

"Quadrature and the cylindrical radius `ρ` as a field, over one region."
function _region_measure(model, set, order)
    Ω = Triangulation(model, tags = set)
    return (Measure(Ω, 2 * order + 1), CellField(x -> x[1], Ω))
end

function FE.fe_axi_mode(::GB, model, order::Int, ncomp::Int, axis_zeros)
    reffe = ReferenceFE(lagrangian, Float64, order)
    V1 = TestFESpace(model, reffe; conformity = :H1)     # no Dirichlet tags
    nd1 = num_free_dofs(V1)

    V = MultiFieldFESpace([V1 for _ in 1:ncomp])
    U = MultiFieldFESpace([TrialFESpace(V1) for _ in 1:ncomp])

    meas = Dict(
        s => _region_measure(model, s, order) for
            s in (FE.AXI_SET_CORE, FE.AXI_SET_SHELL, FE.AXI_SET_MATRIX)
    )

    ρd = get_free_dof_values(interpolate(x -> x[1], V1))
    zd = get_free_dof_values(interpolate(x -> x[2], V1))

    outer = _boundary_dofs(V1, model, reffe, [FE.AXI_SET_OUTER, FE.AXI_SET_OUTER_PTS])
    axis_scalar = isempty(axis_zeros) ? Int[] :
        _boundary_dofs(V1, model, reffe, [FE.AXI_SET_AXIS, FE.AXI_SET_AXIS_PTS])
    axis = Int[i + (c - 1) * nd1 for c in axis_zeros for i in axis_scalar]

    presc = sort!(unique!(vcat([outer .+ (c - 1) * nd1 for c in 1:ncomp]..., axis)))
    free = setdiff(1:(ncomp * nd1), presc)
    return GridapAxiMode(
        ncomp, nd1, ncomp * nd1, V, U, meas,
        (collect(ρd), collect(zd)), outer, axis, presc, free
    )
end

FE.fe_axi_dof_split(::GB, ms::GridapAxiMode) = (ms.ndofs, ms.free, ms.presc)

function FE.fe_axi_set_dirichlet!(::GB, ms::GridapAxiMode, u, f)
    ρd, zd = ms.coords
    for i in ms.outer
        val = f((ρd[i], zd[i]))
        for c in 1:ms.ncomp
            u[i + (c - 1) * ms.nd1] = val[c]
        end
    end
    u[ms.axis] .= 0.0                        # the axis wins at the poles
    return u
end

# ─── The weak form ───────────────────────────────────────────────────────────
#
#  `Bop(N, dNρ, dNz, ρ)` returns an `nrow × ncomp` matrix whose column `c` is
#  the generalized strain produced by component `c`.  Because it is R-linear in
#  `(N, dNρ, dNz)`, applying it to a whole trial function and keeping column
#  `c` gives that component's contribution, and the total strain is their sum.
#  No element `B` matrix is ever formed.

# A `VectorValue` is a `Number` in Gridap, so `collect` on one yields a 0-d
# array, not a vector; `Tuple` is the way back to plain Julia values.
_vec(v) = collect(Tuple(v))

"Column `c` of the generalized-strain operator, applied to a scalar field."
function _bcol(Bop, nrow, c, N, ∇N, ρ)
    B = Bop(N, ∇N[1], ∇N[2], ρ)
    return VectorValue(ntuple(r -> B[r, c], nrow))
end

"""
    _bfield(w, Bop, nrow, c, ρ)

Contribution of the single scalar field `w`, taken as component `c`, to the
generalized strain — a `VectorValue{nrow}`-valued `CellField`.
"""
_bfield(w, Bop, nrow, c, ρ) =
    Operation((N, ∇N, r) -> _bcol(Bop, nrow, c, N, ∇N, r))(w, ∇(w), ρ)

"Generalized strain of a multi-field **function**, as a `VectorValue{nrow}` field."
_strain(ms::GridapAxiMode, w, Bop, nrow, ρ) =
    sum(_bfield(w[c], Bop, nrow, c, ρ) for c in 1:ms.ncomp)

_tensor(D) = TensorValue(D)

"""
    fe_axi_stiffness(GridapBackend(), mode, Dmap, Bop)

The weak form is written out as one term per (region, test component, trial
component) triple rather than as `E(v) ⋅ D ⋅ E(u)` over the summed strains.

That is deliberate. Summing the *bases* `v[1] + v[2] + …` before multiplying
mixes multi-field blocks that Gridap tracks separately, and the assembled
operator comes out subtly non-symmetric — a fraction of a percent, enough to
be mistaken for discretization error. Expanding the double sum keeps every
term a plain one-field-against-one-field product, which is the case Gridap's
block assembly is built for. `_strain` above may sum freely because it is only
ever applied to an `FEFunction`, where the sum is a genuine field sum.
"""
function FE.fe_axi_stiffness(::GB, ms::GridapAxiMode, Dmap, Bop)
    nrow = size(last(first(Dmap)), 1)
    terms = [
        (ms.meas[set]..., _tensor(D), ci, cj) for (set, D) in Dmap
            for ci in 1:ms.ncomp for cj in 1:ms.ncomp
    ]
    energy(u, v, ρ, Dt, ci, cj) =
        _bfield(v[ci], Bop, nrow, ci, ρ) ⋅ (Dt ⋅ _bfield(u[cj], Bop, nrow, cj, ρ)) * ρ
    a(u, v) = sum(∫(energy(u, v, ρ, Dt, ci, cj))dΩ for (dΩ, ρ, Dt, ci, cj) in terms)
    return assemble_matrix(a, ms.U, ms.V)
end

function FE.fe_axi_average(::GB, ms::GridapAxiMode, Dmap, u, Bop, proj, sets)
    nrow = size(last(first(Dmap)), 1)
    nout = length(proj(zeros(nrow)))
    P = Operation(e -> VectorValue(Tuple(proj(_vec(e)))))
    uh = FEFunction(ms.U, u)

    prim = zeros(nout)
    dual = zeros(nout)
    V = 0.0
    for (set, D) in Dmap
        set in sets || continue
        dΩ, ρ = ms.meas[set]
        Dt = _tensor(D)
        e = _strain(ms, uh, Bop, nrow, ρ)
        prim .+= _vec(sum(∫(P(e) * ρ)dΩ))
        dual .+= _vec(sum(∫(P(Dt ⋅ e) * ρ)dΩ))
        V += sum(∫(ρ)dΩ)
    end
    return prim ./ V, dual ./ V, 2π * V
end

# ═══ The flat crack ══════════════════════════════════════════════════════════
#
#  The crack is a zero-thickness discontinuity: the gmsh `Crack` plugin
#  duplicates the nodes of the lips, and `GmshDiscreteModel` carries them
#  through untouched — same node and cell counts as the Ferrite import — so
#  each lip face belongs to exactly one tetrahedron and is a genuine boundary
#  face. Nothing has to be said about the lips: they are traction-free because
#  the mesh says so.

"Vector-valued space of the crack problem, plus what the seam needs."
struct GridapCrackSpace{S, T, O, M, N}
    V::S
    U::T
    ndofs::Int
    dΩ::O                                    # the ball
    Γ::M                                     # both lips
    dΓ::N
    up::Vector{Float64}                      # -(n⋅e₃) per face: +1 on the upper lip
    coords::NTuple{3, Vector{Float64}}       # (x, y, z) of every dof
    comp::Vector{Int}                        # which component each dof carries
    outer::Vector{Int}
    presc::Vector{Int}
    free::Vector{Int}
end

const _E3 = VectorValue(0.0, 0.0, 1.0)

function FE.fe_crack_grid(::GB, crack::MeanFieldHomogenization.FEEllipticCrack)
    a, b = Float64(crack.a), Float64(crack.b)
    opts = crack.mesh
    path = joinpath(mktempdir(), "mfh_crack.msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        FE._build_gmsh_crack_model(gmsh, a, b, opts.radius_ratio * a, opts.htipdiv)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    nweld = FE._weld_msh_crack_front(path, a, b)
    nweld == 0 && @warn "no crack-front node pair was welded — the crack front " *
        "may be split, which overestimates the opening" a b
    return redirect_stdout(devnull) do
        GmshDiscreteModel(path)
    end
end

"Sign that tells the two lips apart: `-(n⋅e₃)`, one value per crack face."
function _lip_sign(Γ, dΓ)
    vals = (get_normal_vector(Γ) ⋅ _E3)(get_cell_points(dΓ))
    return [-first(v) for v in vals]
end

function FE.fe_crack_counts(::GB, model)
    Γ = BoundaryTriangulation(model, tags = FE.SET_CRACK)
    dΓ = Measure(Γ, 2)
    up = _lip_sign(Γ, dΓ)
    n = get_normal_vector(Γ)
    # `n⋅e₃` is exactly ∓1 on a flat crack, so these two integrands are the
    # indicator functions of the lips.
    area_up = sum(∫((1 - n ⋅ _E3) / 2)dΓ)
    area_dn = sum(∫((1 + n ⋅ _E3) / 2)dΓ)
    return (;
        ncells = num_cells(model),
        nnodes = num_nodes(model),
        nfacets_up = count(>(0), up),
        nfacets_dn = count(<(0), up),
        area_up, area_dn,
    )
end

function FE.fe_crack_space(::GB, model, order::Int)
    reffe = ReferenceFE(lagrangian, VectorValue{3, Float64}, order)
    V = TestFESpace(model, reffe; conformity = :H1)      # no Dirichlet tags
    U = TrialFESpace(V)
    ndofs = num_free_dofs(V)

    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2 * order)
    Γ = BoundaryTriangulation(model, tags = FE.SET_CRACK)
    dΓ = Measure(Γ, 2 * order)

    # Dof geometry, read off interpolations: on a nodal Lagrange space,
    # interpolating a function stores its value in each dof's own slot. Feeding
    # the same coordinate to all three components gives that dof's abscissa
    # whatever component it carries; feeding (1,2,3) gives the component index.
    coord(k) = get_free_dof_values(
        interpolate(x -> VectorValue(x[k], x[k], x[k]), V)
    )
    comp = round.(
        Int, get_free_dof_values(
            interpolate(_ -> VectorValue(1.0, 2.0, 3.0), V)
        )
    )

    outer = _boundary_dofs(
        V, model, reffe,
        [FE.SET_OUTER, FE.SET_OUTER_EDGES, FE.SET_OUTER_PTS]
    )
    presc = sort!(outer)
    free = setdiff(1:ndofs, presc)
    return GridapCrackSpace(
        V, U, ndofs, dΩ, Γ, dΓ, _lip_sign(Γ, dΓ),
        (collect(coord(1)), collect(coord(2)), collect(coord(3))),
        comp, outer, presc, free
    )
end

FE.fe_crack_dof_split(::GB, s::GridapCrackSpace) = (s.ndofs, s.free, s.presc)

function FE.fe_crack_set_dirichlet!(::GB, s::GridapCrackSpace, u, f)
    xs, ys, zs = s.coords
    for d in s.outer
        u[d] = f((xs[d], ys[d], zs[d]))[s.comp[d]]
    end
    return u
end

"""
    fe_crack_stiffness(GridapBackend(), space, C)

`∫ ε(v) : ℂ : ε(u) dΩ`, written in Lamé form. The driver's isotropy guard has
already refused any other reference medium — the corrected boundary condition
needs the closed-form Kelvin dipole — so reading `(λ, μ)` off `C` is exact, not
an approximation.
"""
function FE.fe_crack_stiffness(
        ::GB, s::GridapCrackSpace, C::Tensors.SymmetricTensor{4, 3, Float64}
    )
    λ, μ = C[1, 1, 2, 2], C[1, 2, 1, 2]
    σ(e) = λ * tr(e) * one(e) + 2μ * e
    a(u, v) = ∫(ε(v) ⊙ (σ ∘ ε(u)))s.dΩ
    return assemble_matrix(a, s.U, s.V)
end

function FE.fe_crack_mean_jump(
        ::GB, s::GridapCrackSpace, u::Vector{Float64}, S_f::Float64, b::Float64
    )
    uh = FEFunction(s.U, u)
    n = get_normal_vector(s.Γ)
    acc = sum(∫((-(n ⋅ _E3)) * uh)s.dΓ)
    return collect(Tuple(acc)) ./ (S_f * b)
end

end # module
