# =============================================================================
#  axi_backend.jl — the Ferrite implementation of the axisymmetric contract.
#
#  Nine methods, and nothing else.  Every Fourier-, physics- and
#  tensor-algebra question has already been answered by the driver in
#  `src/FiniteElements/axi_driver.jl`; what is left here is discretization:
#  scalar Lagrange fields, a dof numbering, an assembly loop, a Dirichlet lift
#  and a quadrature.
# =============================================================================

const _AXI_FIELDS = (:c1, :c2, :c3)

"""
One Fourier mode, discretized with Ferrite. Built once per inclusion and mode
and reused for every reference medium.

`perm` converts Ferrite's interleaved per-cell dof layout into the
component-major layout `(c-1) * nb + i` that the generalized-strain operator
expects. It is purely element-local — the global vectors the driver handles
are never permuted — which is why it never crosses the backend seam.

`bcref` holds the boundary-data closure of the load case currently being
solved. Ferrite bakes the constrained dof list into the `ConstraintHandler` at
`close!` time; re-pointing `bcref` and calling `update!` refills the
inhomogeneities in place, with no reassembly and no refactorization.
"""
struct AxiModeSetup{D, C, F}
    ncomp::Int
    dh::D
    cv::C
    # Only a cavity uses these: `⟨ε⟩_D` of a pore is a boundary integral.
    fv::F
    perm::Vector{Int}
    ch::Ferrite.ConstraintHandler
    bcref::Base.RefValue{Any}
    axis::Vector{Int}
    presc::Vector{Int}
    free::Vector{Int}
end

# ─── Mesh ────────────────────────────────────────────────────────────────────

function FE.fe_axi_grid(::FE.FerriteBackend, incl::MeanFieldHomogenization.FEExcenteredSphere)
    a = Float64(incl.a)
    a_core = Float64(FE.core_radius(incl))
    d = Float64(FE.core_offset(incl))
    opts = incl.mesh
    R = opts.radius_ratio * a
    h_in = a / opts.nradial
    h_out = opts.coarsening * h_in

    gmsh.initialize()
    local grid
    try
        gmsh.option.setNumber("General.Terminal", 0)
        FE._build_gmsh_axi_model(gmsh, a, a_core, d, R, h_in, h_out)
        grid = FerriteGmsh.togrid()
    finally
        gmsh.finalize()
    end
    return grid
end

function FE.fe_axi_grid(
        ::FE.FerriteBackend, incl::MeanFieldHomogenization.FEAxiSupershapePore
    )
    opts = incl.mesh
    R = opts.radius_ratio * Float64(MeanFieldHomogenization.bounding_radius(incl.shape))
    # `h_in` is set from the transverse semi-axis, as for the core-shell model,
    # so `nradial` keeps the same meaning across the two morphologies.
    h_in = Float64(incl.shape.a) / opts.nradial
    h_out = opts.coarsening * h_in
    gmsh.initialize()
    local grid
    try
        gmsh.option.setNumber("General.Terminal", 0)
        FE._build_gmsh_axi_pore_model(
            gmsh, incl.shape, R, h_in, h_out, opts.nprofile, opts.tip_refine
        )
        grid = FerriteGmsh.togrid()
    finally
        gmsh.finalize()
    end
    return grid
end

# `haskey` rather than the three names unconditionally: a cavity has one region,
# and asking for a cellset that does not exist is an error rather than an empty
# answer.
function FE.fe_axi_grid(
        ::FE.FerriteBackend, incl::MeanFieldHomogenization.FEAxiLayeredSpheroid
    )
    opts = incl.mesh
    n = length(incl.axis_radii)
    # `h_in` from the *outer* transverse semi-axis, so `nradial` keeps the same
    # meaning as for the other axisymmetric morphologies; the mesher then caps
    # it per layer against the layer's own thickness.
    a_out = incl.disk_radii[n]
    R = opts.radius_ratio * max(a_out, incl.axis_radii[n])
    h_in = a_out / opts.nradial
    h_out = opts.coarsening * h_in
    gmsh.initialize()
    local grid
    try
        gmsh.option.setNumber("General.Terminal", 0)
        FE._build_gmsh_axi_layered_model(
            gmsh, incl.axis_radii, incl.disk_radii, R, h_in, h_out, opts.nprofile
        )
        grid = FerriteGmsh.togrid()
    finally
        gmsh.finalize()
    end
    return grid
end

# Every region the grid actually has, read off the grid rather than guessed: the
# three fixed names of the core-shell model, plus every set whose name matches
# the layer pattern. Enumerating candidate indices up to a bound would work
# until someone asked for one layer more than the bound, and would then drop
# that layer from the counts silently.
function _axi_region_names(grid)
    have = keys(Ferrite.getcellsets(grid))
    fixed = [
        s for s in (FE.AXI_SET_CORE, FE.AXI_SET_SHELL, FE.AXI_SET_MATRIX) if s in have
    ]
    lay = [s for s in have if !isnothing(match(FE.AXI_LAYER_SET_PATTERN, s))]
    sort!(lay; by = s -> parse(Int, match(FE.AXI_LAYER_SET_PATTERN, s)[1]))
    return vcat(fixed, lay)
end

FE.fe_axi_grid_counts(::FE.FerriteBackend, grid) = (;
    ncells = Ferrite.getncells(grid),
    nnodes = Ferrite.getnnodes(grid),
    ncells_by_set = Dict(
        s => length(Ferrite.getcellset(grid, s)) for s in _axi_region_names(grid)
    ),
)

function FE.fe_axi_region_volume(::FE.FerriteBackend, grid, set)
    ip_geo = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{Ferrite.RefTriangle}(3), ip_geo, ip_geo)
    V = 0.0
    for ci in Ferrite.getcellset(grid, set)
        coords = Ferrite.getcoordinates(grid, ci)
        Ferrite.reinit!(cv, coords)
        for q in 1:Ferrite.getnquadpoints(cv)
            V += Ferrite.getdetJdV(cv, q) * Ferrite.spatial_coordinate(cv, q, coords)[1]
        end
    end
    return 2π * V
end

# ─── One mode ────────────────────────────────────────────────────────────────

function FE.fe_axi_mode(::FE.FerriteBackend, grid, order::Int, ncomp::Int, axis_zeros)
    fields = _AXI_FIELDS[1:ncomp]

    ip = Ferrite.Lagrange{Ferrite.RefTriangle, order}()
    ip_geo = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()
    qr = Ferrite.QuadratureRule{Ferrite.RefTriangle}(2 * order + 1)
    cv = Ferrite.CellValues(qr, ip, ip_geo)
    fqr = Ferrite.FacetQuadratureRule{Ferrite.RefTriangle}(2 * order + 1)
    fv = Ferrite.FacetValues(fqr, ip, ip_geo)

    dh = Ferrite.DofHandler(grid)
    for f in fields
        Ferrite.add!(dh, f, ip)
    end
    Ferrite.close!(dh)
    perm = reduce(vcat, [collect(Ferrite.dof_range(dh, f)) for f in fields])

    # Outer boundary: values supplied per solve through `bcref`.
    bcref = Base.RefValue{Any}(_ -> ntuple(_ -> 0.0, ncomp))
    ch = Ferrite.ConstraintHandler(dh)
    for (k, f) in pairs(fields)
        Ferrite.add!(
            ch, Ferrite.Dirichlet(
                f, Ferrite.getfacetset(grid, FE.AXI_SET_OUTER), (x, _t) -> bcref[](x)[k]
            )
        )
    end
    Ferrite.close!(ch)

    # Axis: the components regularity forces to vanish.
    axis = Int[]
    if !isempty(axis_zeros)
        cha = Ferrite.ConstraintHandler(dh)
        for k in axis_zeros
            Ferrite.add!(
                cha, Ferrite.Dirichlet(
                    fields[k], Ferrite.getfacetset(grid, FE.AXI_SET_AXIS), (_x, _t) -> 0.0
                )
            )
        end
        Ferrite.close!(cha)
        axis = collect(cha.prescribed_dofs)
    end

    presc = sort!(union(collect(ch.prescribed_dofs), axis))
    free = setdiff(1:Ferrite.ndofs(dh), presc)
    return AxiModeSetup(ncomp, dh, cv, fv, perm, ch, bcref, axis, presc, free)
end

FE.fe_axi_dof_split(::FE.FerriteBackend, ms::AxiModeSetup) =
    (Ferrite.ndofs(ms.dh), ms.free, ms.presc)

function FE.fe_axi_set_dirichlet!(::FE.FerriteBackend, ms::AxiModeSetup, u, f)
    ms.bcref[] = f
    Ferrite.update!(ms.ch, 0.0)
    u[ms.ch.prescribed_dofs] .= ms.ch.inhomogeneities
    u[ms.axis] .= 0.0                      # the axis wins at the poles
    return u
end

# ─── Assembly and averages ───────────────────────────────────────────────────
#
#  Both walk the same quadrature and build the same element operator, so the
#  inner loop is shared.  The measure is `ρ dρ dz`: that single factor is what
#  turns a plane problem into a solid of revolution.

"Fill `Bq` with the generalized-strain operator of one quadrature point."
function _axi_fill_B!(Bq, ms::AxiModeSetup, Bop, q, ρ, nb)
    fill!(Bq, 0)
    for i in 1:nb
        N = Ferrite.shape_value(ms.cv, q, i)
        ∇N = Ferrite.shape_gradient(ms.cv, q, i)
        Bi = Bop(N, ∇N[1], ∇N[2], ρ)
        for c in 1:ms.ncomp, r in axes(Bi, 1)
            Bq[r, (c - 1) * nb + i] = Bi[r, c]
        end
    end
    return Bq
end

function FE.fe_axi_stiffness(::FE.FerriteBackend, ms::AxiModeSetup, Dmap, Bop)
    K = Ferrite.allocate_matrix(ms.dh)
    asm = Ferrite.start_assemble(K)
    nb = Ferrite.getnbasefunctions(ms.cv)
    nl = nb * ms.ncomp
    ke = zeros(nl, nl)
    Bq = zeros(size(last(first(Dmap)), 1), nl)
    grid = Ferrite.get_grid(ms.dh)

    for (setname, D) in Dmap
        for cell in Ferrite.CellIterator(ms.dh, Ferrite.getcellset(grid, setname))
            Ferrite.reinit!(ms.cv, cell)
            coords = Ferrite.getcoordinates(cell)
            fill!(ke, 0)
            for q in 1:Ferrite.getnquadpoints(ms.cv)
                ρ = Ferrite.spatial_coordinate(ms.cv, q, coords)[1]
                dΩ = Ferrite.getdetJdV(ms.cv, q) * ρ
                _axi_fill_B!(Bq, ms, Bop, q, ρ, nb)
                ke .+= (Bq' * D * Bq) .* dΩ
            end
            Ferrite.assemble!(asm, Ferrite.celldofs(cell)[ms.perm], ke)
        end
    end
    return K
end

function FE.fe_axi_average(::FE.FerriteBackend, ms::AxiModeSetup, Dmap, u, Bop, proj, sets)
    nb = Ferrite.getnbasefunctions(ms.cv)
    nl = nb * ms.ncomp
    nrow = size(last(first(Dmap)), 1)
    Bq = zeros(nrow, nl)
    prim = zeros(length(proj(zeros(nrow))))
    dual = zeros(length(prim))
    V = 0.0
    grid = Ferrite.get_grid(ms.dh)

    for (setname, D) in Dmap
        setname in sets || continue
        for cell in Ferrite.CellIterator(ms.dh, Ferrite.getcellset(grid, setname))
            Ferrite.reinit!(ms.cv, cell)
            coords = Ferrite.getcoordinates(cell)
            ue = u[Ferrite.celldofs(cell)[ms.perm]]
            for q in 1:Ferrite.getnquadpoints(ms.cv)
                ρ = Ferrite.spatial_coordinate(ms.cv, q, coords)[1]
                dΩ = Ferrite.getdetJdV(ms.cv, q) * ρ
                _axi_fill_B!(Bq, ms, Bop, q, ρ, nb)
                e = Bq * ue
                prim .+= proj(e) .* dΩ
                dual .+= proj(D * e) .* dΩ
                V += dΩ
            end
        end
    end
    return prim ./ V, dual ./ V, 2π * V
end

"""
Line integral of `(ū ⊗ n)ˢ` on the meridian trace of a cavity wall, with the
measure `ρ dl` and no normalization. See the generic for why this exists and why
the divergence identity is not used instead.

The facet interpolation is the same as the cell's, so `ū` is exact on the trace
to the order of the discretization; the spline that carries the profile is what
sets the geometry error, and the mesh report is what measures it.
"""
function FE.fe_axi_pore_boundary(
        ::FE.FerriteBackend, ms::AxiModeSetup, u::AbstractVector, dofmap, proj, set
    )
    nb = Ferrite.getnbasefunctions(ms.fv)
    ncyl = zeros(6)
    acc = zeros(length(proj(zeros(6))))
    grid = Ferrite.get_grid(ms.dh)
    r2 = sqrt(2.0)
    # The volume the *meshed* wall actually encloses, by the same quadrature:
    # `V = (1/3)∮ x·n_D dS` with `n_D` outward from the cavity, which in the
    # meridian plane is `-(ρ n_ρ + z n_z)` against the facet normal. It has to
    # be this and not the closed form, or the average is normalized by one
    # volume and integrated over the boundary of another — a systematic error
    # of the geometry's own size that no refinement removes, because refining
    # shrinks both together.
    vol = 0.0
    for facet in Ferrite.FacetIterator(ms.dh, Ferrite.getfacetset(grid, set))
        Ferrite.reinit!(ms.fv, facet)
        ue = u[Ferrite.celldofs(facet)[ms.perm]]
        coords = Ferrite.getcoordinates(facet)
        for q in 1:Ferrite.getnquadpoints(ms.fv)
            ρ = Ferrite.spatial_coordinate(ms.fv, q, coords)[1]
            nrm = Ferrite.getnormal(ms.fv, q)
            dΓ = Ferrite.getdetJdV(ms.fv, q) * ρ
            # The solved unknowns are the *mapped* ones; `dofmap` takes them
            # back to `(ū_ρ, ū_θ, ū_z)`, which is what a tensor product needs.
            usol = ntuple(ms.ncomp) do c
                v = 0.0
                for i in 1:nb
                    v += Ferrite.shape_value(ms.fv, q, i) * ue[(c - 1) * nb + i]
                end
                v
            end
            ūρ = ūθ = ūz = 0.0
            for c in 1:ms.ncomp
                ūρ += dofmap[1, c] * usol[c]
                ūθ += dofmap[2, c] * usol[c]
                ūz += dofmap[3, c] * usol[c]
            end
            nρ, nz = nrm[1], nrm[2]
            ncyl[1] = ūρ * nρ
            ncyl[2] = 0.0                       # the facet normal has no azimuthal part
            ncyl[3] = ūz * nz
            ncyl[4] = r2 * (ūθ * nz) / 2
            ncyl[5] = r2 * (ūρ * nz + ūz * nρ) / 2
            ncyl[6] = r2 * (ūθ * nρ) / 2
            acc .+= proj(ncyl) .* dΓ
            vol -= (ρ * nρ + coords_z(ms.fv, q, coords) * nz) * dΓ
        end
    end
    return acc, 2π * vol / 3
end

"The `z` of a facet quadrature point — spelled out so the line above stays readable."
coords_z(fv, q, coords) = Ferrite.spatial_coordinate(fv, q, coords)[2]
