# =============================================================================
#  cell_backend.jl — the Ferrite implementation of the cell contract.
#
#  Eight methods, for the three-dimensional cell around a non-ellipsoidal
#  shape.  One field over one region and one Dirichlet boundary, so the seam is
#  narrow; what makes it different from the crack's is that the geometry is
#  **quadratic** — the mid-edge nodes of the inclusion surface sit on the exact
#  shape — and that the same space serves a scalar temperature and a vector
#  displacement.
# =============================================================================

"Grid, space and the pieces the seam needs, built once per cell."
struct FerriteCellSpace{D, C, F}
    dh::D
    cv::C
    fv::F
    grid::Any
    ch::Ferrite.ConstraintHandler
    bcref::Base.RefValue{Any}
    presc::Vector{Int}
    free::Vector{Int}
    ncomp::Int
end

# ─── Mesh ────────────────────────────────────────────────────────────────────

function FE.fe_cell_grid(::FE.FerriteBackend, shape, opts::MeanFieldHomogenization.FECellMeshOptions)
    opts.order == 2 || throw(
        ArgumentError(
            "the cell backend needs `order = 2`: the inclusion boundary is " *
                "curved, and a linear geometry would throw the whole gain away " *
                "while still returning a plausible number"
        )
    )
    gmsh.initialize()
    local grid, built
    try
        gmsh.option.setNumber("General.Terminal", opts.verbose ? 1 : 0)
        built = FE._build_gmsh_cell_model(gmsh, shape, opts)
        # Straight from the session rather than through a `.msh` on disk: unlike
        # the crack, nothing has to be repaired between meshing and import.
        grid = redirect_stdout(devnull) do
            FerriteGmsh.togrid()
        end
    finally
        gmsh.finalize()
    end
    return (; grid, built)
end

function FE.fe_cell_counts(::FE.FerriteBackend, grid)
    ip_geo = Ferrite.Lagrange{Ferrite.RefTetrahedron, 2}()
    qr = Ferrite.FacetQuadratureRule{Ferrite.RefTetrahedron}(4)
    return (;
        ncells = Ferrite.getncells(grid),
        nnodes = Ferrite.getnnodes(grid),
        area_inclusion = _cell_facetset_area(grid, FE.CELL_SET_INCLUSION, ip_geo, qr),
        area_outer = _cell_facetset_area(grid, FE.CELL_SET_OUTER, ip_geo, qr),
    )
end

"Area of a named facet set, measured on the (quadratic) geometric interpolation."
function _cell_facetset_area(grid, name, ip_geo, qr)
    fv = Ferrite.FacetValues(qr, ip_geo, ip_geo)
    A = 0.0
    for fi in Ferrite.getfacetset(grid, name)
        Ferrite.reinit!(fv, Ferrite.getcoordinates(grid, fi[1]), fi[2])
        for q in 1:Ferrite.getnquadpoints(fv)
            A += Ferrite.getdetJdV(fv, q)
        end
    end
    return A
end

# ─── Space ───────────────────────────────────────────────────────────────────

function FE.fe_cell_space(::FE.FerriteBackend, grid, order::Int, ncomp::Int)
    ncomp in (1, 3) || throw(ArgumentError("ncomp must be 1 or 3, got $ncomp"))
    # The geometric interpolation is quadratic, and that is the whole point: the
    # boundary was curved on purpose in `_snap_cell_surface_to_shape!`, and a
    # linear geometry would ignore it.
    ip_geo = Ferrite.Lagrange{Ferrite.RefTetrahedron, 2}()
    base = Ferrite.Lagrange{Ferrite.RefTetrahedron, order}()
    ip = ncomp == 1 ? base : base^3
    qr = Ferrite.QuadratureRule{Ferrite.RefTetrahedron}(2 * order)
    fqr = Ferrite.FacetQuadratureRule{Ferrite.RefTetrahedron}(2 * order)
    cv = Ferrite.CellValues(qr, ip, ip_geo)
    fv = Ferrite.FacetValues(fqr, ip, ip_geo)

    name = ncomp == 1 ? :T : :u
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, name, ip)
    Ferrite.close!(dh)

    bcref = Base.RefValue{Any}(ncomp == 1 ? (_ -> 0.0) : (_ -> (0.0, 0.0, 0.0)))
    ch = Ferrite.ConstraintHandler(dh)
    outer = Ferrite.getfacetset(grid, FE.CELL_SET_OUTER)
    if ncomp == 1
        Ferrite.add!(ch, Ferrite.Dirichlet(name, outer, (x, _t) -> bcref[](x)))
    else
        Ferrite.add!(
            ch, Ferrite.Dirichlet(name, outer, (x, _t) -> Ferrite.Vec{3}(Tuple(bcref[](x))))
        )
    end
    Ferrite.close!(ch)

    presc = sort!(collect(ch.prescribed_dofs))
    free = setdiff(1:Ferrite.ndofs(dh), presc)
    return FerriteCellSpace(dh, cv, fv, grid, ch, bcref, presc, free, ncomp)
end

FE.fe_cell_dof_split(::FE.FerriteBackend, s::FerriteCellSpace) =
    (Ferrite.ndofs(s.dh), s.free, s.presc)

function FE.fe_cell_set_dirichlet!(::FE.FerriteBackend, s::FerriteCellSpace, u, f)
    s.bcref[] = f
    Ferrite.update!(s.ch, 0.0)
    u[s.ch.prescribed_dofs] .= s.ch.inhomogeneities
    return u
end

# ─── Operators ───────────────────────────────────────────────────────────────

# Conduction: `material` is the 3×3 conductivity, and the driver has already
# refused anything but an isotropic one, so only its first entry is read.
function FE.fe_cell_stiffness(::FE.FerriteBackend, s::FerriteCellSpace, K0::AbstractMatrix)
    k0 = K0[1, 1]
    K = Ferrite.allocate_matrix(s.dh)
    asm = Ferrite.start_assemble(K)
    n = Ferrite.getnbasefunctions(s.cv)
    ke = zeros(n, n)
    for cell in Ferrite.CellIterator(s.dh)
        Ferrite.reinit!(s.cv, cell)
        fill!(ke, 0)
        for q in 1:Ferrite.getnquadpoints(s.cv)
            dΩ = Ferrite.getdetJdV(s.cv, q)
            for i in 1:n
                gi = Ferrite.shape_gradient(s.cv, q, i)
                for j in i:n
                    ke[i, j] += k0 * (gi ⋅ Ferrite.shape_gradient(s.cv, q, j)) * dΩ
                end
            end
        end
        for i in 1:n, j in 1:(i - 1)
            ke[i, j] = ke[j, i]
        end
        Ferrite.assemble!(asm, Ferrite.celldofs(cell), ke)
    end
    return K
end

function FE.fe_cell_stiffness(
        ::FE.FerriteBackend, s::FerriteCellSpace, C::Tensors.SymmetricTensor{4, 3, Float64}
    )
    K = Ferrite.allocate_matrix(s.dh)
    asm = Ferrite.start_assemble(K)
    n = Ferrite.getnbasefunctions(s.cv)
    ke = zeros(n, n)
    for cell in Ferrite.CellIterator(s.dh)
        Ferrite.reinit!(s.cv, cell)
        fill!(ke, 0)
        for q in 1:Ferrite.getnquadpoints(s.cv)
            dΩ = Ferrite.getdetJdV(s.cv, q)
            for i in 1:n
                σi = C ⊡ Ferrite.shape_symmetric_gradient(s.cv, q, i)
                for j in i:n
                    ke[i, j] += (σi ⊡ Ferrite.shape_symmetric_gradient(s.cv, q, j)) * dΩ
                end
            end
        end
        for i in 1:n, j in 1:(i - 1)
            ke[i, j] = ke[j, i]
        end
        Ferrite.assemble!(asm, Ferrite.celldofs(cell), ke)
    end
    return K
end

# The normal `FacetValues` reports points out of the *meshed* region, hence into
# the pore, so it is the opposite of the inclusion's own outward normal — which
# is where the minus sign in both averages comes from.
function FE.fe_cell_mean_gradient(
        ::FE.FerriteBackend, s::FerriteCellSpace, u::AbstractVector, V::Real
    )
    acc = zeros(3)
    for facet in Ferrite.FacetIterator(s.dh, Ferrite.getfacetset(s.grid, FE.CELL_SET_INCLUSION))
        Ferrite.reinit!(s.fv, facet)
        ue = @view u[Ferrite.celldofs(facet)]
        for q in 1:Ferrite.getnquadpoints(s.fv)
            dΓ = Ferrite.getdetJdV(s.fv, q)
            nrm = Ferrite.getnormal(s.fv, q)
            T = Ferrite.function_value(s.fv, q, ue)
            @inbounds for i in 1:3
                acc[i] += T * nrm[i] * dΓ
            end
        end
    end
    return (-acc[1] / V, -acc[2] / V, -acc[3] / V)
end

function FE.fe_cell_mean_strain(
        ::FE.FerriteBackend, s::FerriteCellSpace, u::AbstractVector, V::Real
    )
    acc = zeros(3, 3)
    for facet in Ferrite.FacetIterator(s.dh, Ferrite.getfacetset(s.grid, FE.CELL_SET_INCLUSION))
        Ferrite.reinit!(s.fv, facet)
        ue = @view u[Ferrite.celldofs(facet)]
        for q in 1:Ferrite.getnquadpoints(s.fv)
            dΓ = Ferrite.getdetJdV(s.fv, q)
            nrm = Ferrite.getnormal(s.fv, q)
            uq = Ferrite.function_value(s.fv, q, ue)
            @inbounds for i in 1:3, j in 1:3
                acc[i, j] += 0.5 * (uq[i] * nrm[j] + uq[j] * nrm[i]) * dΓ
            end
        end
    end
    acc ./= -V
    r2 = sqrt(2.0)
    return (
        acc[1, 1], acc[2, 2], acc[3, 3],
        r2 * acc[2, 3], r2 * acc[1, 3], r2 * acc[1, 2],
    )
end
