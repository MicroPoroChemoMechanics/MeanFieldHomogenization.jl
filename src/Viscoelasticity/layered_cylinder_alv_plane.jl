# =============================================================================
#  layered_cylinder_alv_plane.jl — a `LayeredCylinder` in a two-dimensional
#  cell in aging viscoelasticity: concentric rings.
#
#  In a cell of the plane the cylinder is its section, and only the problems of
#  the plane are solved (LayeredCylinders/plane.jl): the in-plane dilatation and
#  the transverse shear, whose recurrences are those the three-dimensional
#  cylinder solves too. As in space (layered_cylinder_alv.jl), they are solved
#  once with n × n Volterra matrices as moduli, k = α/2 and m = β/2 read on the
#  isotropic blocks (α, β) of the operators of the plane of the reference and
#  of the layers. Every operator of the rings is isotropic in the plane, stored
#  by its two blocks (me, T) = (α, β).
# =============================================================================

# The dimension of the space of a reference or of a layer property, read on its
# class: a law at its first instant; a plain matrix is one of space.
_reference_space(law::AbstractViscoLaw, times) = Val(_space_dim(_class_of(visco_eval(law, first(times), first(times)))))
_reference_space(R::VolterraOperator, times) = Val(_space_dim(R.class))
_reference_space(::TensND.AbstractTens{O, D}, times) where {O, D} = Val(D)
_reference_space(::AbstractMatrix, times) = Val(3)

# The moduli of the plane of an isotropic operator of the plane, as the radial
# problems read them (`l`, the coupling to the axial strain, is zero in the plane).
function _plane_moduli_alv(P, grid::TimeGrid, who::AbstractString)
    # An operator off the isotropic class is refused by the conversion.
    α, β = _as_iso_estimate(_relaxation_operator(P, grid)).blocks
    L = (; k = α ./ 2, m = β ./ 2, l = zero(α))
    _check_invertible(L.k + L.m, "plane-strain modulus k + m", who)
    _check_invertible(L.m, "transverse shear modulus m", who)
    return L
end

# The solution of the problems of the plane, wrapped so that the operators of the
# cylinder dispatch on it.
struct _PlaneRingsALV{S, G <: TimeGrid}
    sol::S
    grid::G
end

function _cylinder_alv_solution(::Val{4}, ::Val{2}, cyl::LayeredCylinder{T, N}, ref, times) where {T, N}
    grid = TimeGrid(times)
    L₀ = _plane_moduli_alv(ref, grid, "the reference medium")
    Ls = ntuple(k -> _plane_moduli_alv(layer_modulus(cyl, k), grid, "layer $k"), Val(N))
    E = promote_type(T, eltype(L₀.k), map(L -> eltype(L.k), Ls)..., interfaces_eltype(cyl.interfaces))
    return _PlaneRingsALV(_LC._plane_elastic_solution(E, cyl, L₀, Ls), grid)
end
_cylinder_alv_solution(::Val{2}, ::Val{2}, cyl::LayeredCylinder, ref, times) =
    _alv_plane_error("a LayeredCylinder in conduction")

_strain_entries_alv(::Val{4}, s::_PlaneRingsALV, cyl, ::Nothing, external, internal) = _LC._plane_whole_strain(s.sol, external)
function _strain_entries_alv(::Val{4}, s::_PlaneRingsALV, cyl, layer::Int, external, internal)
    _LC._check_layer(cyl, layer)
    return _LC._plane_layer_strain(s.sol, layer, external, internal)
end
_flux_entries_alv(::Val{4}, s::_PlaneRingsALV, cyl, ::Nothing, external) = _LC._plane_whole_stress(s.sol, external)
_flux_entries_alv(::Val{4}, s::_PlaneRingsALV, cyl, layer::Int, external) =
    (_LC._check_layer(cyl, layer); _LC._plane_layer_stress(s.sol, layer))
_contribution_entries_alv(::Val{4}, s::_PlaneRingsALV, external) = _LC._plane_contribution(s.sol, external)

# An operator of the rings: isotropic in the plane, its blocks the coefficients on
# 𝕁 and 𝕂.
_canonical_blocks(::Val{4}, cyl::LayeredCylinder, c::NamedTuple{(:me, :T)}, times) =
    VolterraOperator(_ISO4_PLANE, (c.me, c.T), TimeGrid(times))

# The stiffness a bound assigns to the rings: the average of the operators of the
# layers (Voigt), or the inverse of the average of their inverses (Reuss).
function _layer_bound_alv(scheme, ::Val{2}, cyl::LayeredCylinder{T, N}, times) where {T, N}
    grid = TimeGrid(times)
    Cs = [_relaxation_operator(layer_modulus(cyl, k), grid) for k in 1:N]
    fs = [layer_volume_fraction(cyl, k) for k in 1:N]
    return _in_class_algebra((), (Cs,)) do a, Cs
        _alg_layer_bound(scheme, a, Cs, fs)
    end
end
_alg_layer_bound(::Reuss, a, Cs, fs) = _alg_reuss(a, Cs, fs)
_alg_layer_bound(::HomogenizationScheme, a, Cs, fs) = _alg_voigt(a, Cs, fs)
