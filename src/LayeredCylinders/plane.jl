# =============================================================================
#  plane.jl — a `LayeredCylinder` in a two-dimensional cell: concentric rings.
#
#  The properties of the layers and of the reference are tensors of the plane
#  (`TensISO{4,2}`, `TensISO{2,2}`), isotropic. Only the problems of the plane
#  are solved: the in-plane dilatation and the transverse shear in elasticity,
#  the transverse potential problem in conduction. The axial and the antiplane
#  problems of the three-dimensional cylinder are not. The axis of the cylinder
#  is not read: it is the normal to the plane.
# =============================================================================

const _Plane4 = TensND.AbstractTens{4, 2}
const _Plane2 = TensND.AbstractTens{2, 2}

# Moduli of the plane of the stiffness 2k 𝕁 + 2m 𝕂. `l` multiplies the axial
# strain in the kernel, which is zero in the plane.
function _plane_moduli(C)
    P = _plane_isotropic(C)
    k, m = TensND.bulk_shear(P)
    return (; k, m, l = zero(eltype(P)))
end
_plane_conductivity(K) = first(TensND.get_data(_plane_isotropic(K)))

_check_plane_stiffness(L, where_) = _check_plane_stiffness(Val(is_hard_numeric(_entries_eltype(L))), L, where_)
_check_plane_stiffness(::Val{false}, L, where_) = nothing
function _check_plane_stiffness(::Val{true}, L, where_)
    isfinite(L.k) && isfinite(L.m) && L.k > 0 && L.m > 0 || throw(
        ArgumentError(
            "$where_ of the LayeredCylinder has the moduli of the plane (k, μ) = $((L.k, L.m)): " *
                "they must be finite and positive (an incompressible or a void layer is not supported)"
        )
    )
    return nothing
end

"""
    _plane_elastic_solution(cyl, c₀) -> NamedTuple

The two problems of the plane of the rings in the reference `c₀`, each under
its unit remote loading: `dilatation` and `shear`.
"""
function _plane_elastic_solution(cyl::LayeredCylinder{T, N}, c₀::_Plane4) where {T, N}
    _has_radial_layer(cyl) && _no_radial("the two-dimensional cell")
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    L₀ = _plane_moduli(c₀)
    Ls = map(_plane_moduli, cyl.moduli)
    _check_plane_stiffness(L₀, "the reference medium")
    foreach(k -> _check_plane_stiffness(Ls[k], "layer $k"), 1:N)
    S = promote_type(
        T, _entries_eltype(L₀), map(_entries_eltype, Ls)..., interfaces_eltype(cyl.interfaces)
    )
    return _plane_elastic_solution(S, cyl, L₀, Ls)
end
function _plane_elastic_solution(::Type{S}, cyl::LayeredCylinder{T, N}, L₀, Ls) where {S, T, N}
    radii = map(x -> S(x), cyl.radii)
    L₀ = _in_type(S, L₀)
    Ls = map(L -> _in_type(S, L), Ls)
    itfs = cyl.interfaces
    dilatation = first(_solve_dilatation(radii, Ls, itfs, L₀))
    shear = _solve_shear(radii, Ls, itfs, L₀)
    fractions = ntuple(k -> S(layer_volume_fraction(cyl, k)), Val(N))
    return (; radii, Ls, L₀, fractions, interfaces = itfs, dilatation, shear)
end

function _plane_conduction_solution(cyl::LayeredCylinder{T, N}, κ₀::_Plane2) where {T, N}
    _has_radial_layer(cyl) && _no_radial("the two-dimensional cell")
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    k₀ = (; t = _plane_conductivity(κ₀))
    ks = map(K -> (; t = _plane_conductivity(K)), cyl.moduli)
    _check_conductivity(k₀, "the reference medium"; positive = true)
    foreach(k -> _check_conductivity(ks[k], "layer $k"), 1:N)
    S = promote_type(T, typeof(k₀.t), map(k -> typeof(k.t), ks)..., interfaces_eltype(cyl.interfaces))
    return _plane_conduction_solution(S, cyl, S(k₀.t), map(k -> S(k.t), ks))
end
function _plane_conduction_solution(::Type{S}, cyl::LayeredCylinder, k₀, ks) where {S}
    radii = map(x -> S(x), cyl.radii)
    transverse = _solve_potential(TransverseConduction(), radii, ks, cyl.interfaces, k₀)
    return (; radii, ks, k₀, transverse)
end

# Packing: the coefficients on 𝕁 and 𝕂 of the plane, or on its identity.
_plane_tensor(c::NamedTuple{(:me, :T)}) = TensND.TensISO{2}(c.me, c.T)
_plane_stiffness_entries(L) = (; me = 2 * L.k, T = 2 * L.m)
_plane_compose(C, A) = (; me = C.me * A.me, T = C.T * A.T)
_plane_layer_stress(sol, k::Int) =
    _plane_compose(_plane_stiffness_entries(sol.Ls[k]), _plane_layer_strain(sol, k, false, false))
_plane_contribution(sol, external::Bool) = _minus(
    _plane_whole_stress(sol, external),
    _plane_compose(_plane_stiffness_entries(sol.L₀), _plane_whole_strain(sol, external)),
)

_plane_whole_gradient(sol, external::Bool) =
    _side(sol.transverse, length(sol.radii), external)[1] / sol.radii[end]
_plane_whole_flux(sol, external::Bool) = _side(sol.transverse, length(sol.radii), external)[2]
_plane_layer_gradient(sol, k::Int, external::Bool, internal::Bool) =
    _annulus_gradient(TransverseConduction(), sol.transverse, sol.radii, k, external, internal)

# ── Elasticity ───────────────────────────────────────────────────────────────

"""
    gradient_gradient_concentration(cyl::LayeredCylinder, c₁, c₀::AbstractTens{4,2}; external = true) -> TensISO{4,2}
    gradient_gradient_concentration(cyl::LayeredCylinder, κ₁, κ₀::AbstractTens{2,2}; external = true) -> TensISO{2,2}

The same tensors in a two-dimensional cell, where the cylinder is its section,
concentric rings: the layers and the reference `c₀` are isotropic tensors of
the plane, and the result is one. Only the problems of the plane are solved.
The other functions of the cylinder ([`flux_gradient_concentration`](@ref),
`property_contribution`, the tensors of a layer) have the same methods.
"""
function gradient_gradient_concentration(
        cyl::LayeredCylinder, ::_Plane4, c₀::_Plane4; external::Bool = true, kw...
    )
    return _plane_tensor(_plane_whole_strain(_plane_elastic_solution(cyl, c₀), external))
end
function flux_gradient_concentration(
        cyl::LayeredCylinder, ::_Plane4, c₀::_Plane4; external::Bool = true, kw...
    )
    return _plane_tensor(_plane_whole_stress(_plane_elastic_solution(cyl, c₀), external))
end
function Core.property_contribution(
        cyl::LayeredCylinder, ::_Plane4, c₀::_Plane4; external::Bool = true, kw...
    )
    return _plane_tensor(_plane_contribution(_plane_elastic_solution(cyl, c₀), external))
end
Core.property_contribution(cyl::LayeredCylinder, c₀::_Plane4; kw...) =
    Core.property_contribution(cyl, c₀, c₀; kw...)
function Core.concentration_and_contribution(
        cyl::LayeredCylinder, ::_Plane4, c₀::_Plane4; external::Bool = true, kw...
    )
    sol = _plane_elastic_solution(cyl, c₀)
    return (_plane_tensor(_plane_whole_strain(sol, external)), _plane_tensor(_plane_contribution(sol, external)))
end
function Core.concentration_and_flux_average(
        cyl::LayeredCylinder, ::_Plane4, c₀::_Plane4; external::Bool = true, kw...
    )
    sol = _plane_elastic_solution(cyl, c₀)
    return (_plane_tensor(_plane_whole_strain(sol, external)), _plane_tensor(_plane_whole_stress(sol, external)))
end
function gradient_gradient_concentration(
        cyl::LayeredCylinder, c₀::_Plane4;
        layer::Int, external::Bool = false, internal::Bool = false, kw...
    )
    _check_layer(cyl, layer)
    return _plane_tensor(_plane_layer_strain(_plane_elastic_solution(cyl, c₀), layer, external, internal))
end
function flux_gradient_concentration(cyl::LayeredCylinder, c₀::_Plane4; layer::Int, kw...)
    _check_layer(cyl, layer)
    return _plane_tensor(_plane_layer_stress(_plane_elastic_solution(cyl, c₀), layer))
end

# ── Conduction ───────────────────────────────────────────────────────────────

function gradient_gradient_concentration(
        cyl::LayeredCylinder, ::_Plane2, κ₀::_Plane2; external::Bool = true, kw...
    )
    return TensND.TensISO{2}(_plane_whole_gradient(_plane_conduction_solution(cyl, κ₀), external))
end
function flux_gradient_concentration(
        cyl::LayeredCylinder, ::_Plane2, κ₀::_Plane2; external::Bool = true, kw...
    )
    return TensND.TensISO{2}(_plane_whole_flux(_plane_conduction_solution(cyl, κ₀), external))
end
function Core.property_contribution(
        cyl::LayeredCylinder, ::_Plane2, κ₀::_Plane2; external::Bool = true, kw...
    )
    sol = _plane_conduction_solution(cyl, κ₀)
    return TensND.TensISO{2}(_plane_whole_flux(sol, external) - sol.k₀ * _plane_whole_gradient(sol, external))
end
Core.property_contribution(cyl::LayeredCylinder, κ₀::_Plane2; kw...) =
    Core.property_contribution(cyl, κ₀, κ₀; kw...)
function Core.concentration_and_contribution(
        cyl::LayeredCylinder, ::_Plane2, κ₀::_Plane2; external::Bool = true, kw...
    )
    sol = _plane_conduction_solution(cyl, κ₀)
    G, Q = _plane_whole_gradient(sol, external), _plane_whole_flux(sol, external)
    return (TensND.TensISO{2}(G), TensND.TensISO{2}(Q - sol.k₀ * G))
end
function Core.concentration_and_flux_average(
        cyl::LayeredCylinder, ::_Plane2, κ₀::_Plane2; external::Bool = true, kw...
    )
    sol = _plane_conduction_solution(cyl, κ₀)
    return (TensND.TensISO{2}(_plane_whole_gradient(sol, external)), TensND.TensISO{2}(_plane_whole_flux(sol, external)))
end
function gradient_gradient_concentration(
        cyl::LayeredCylinder, κ₀::_Plane2;
        layer::Int, external::Bool = false, internal::Bool = false, kw...
    )
    _check_layer(cyl, layer)
    return TensND.TensISO{2}(_plane_layer_gradient(_plane_conduction_solution(cyl, κ₀), layer, external, internal))
end
function flux_gradient_concentration(cyl::LayeredCylinder, κ₀::_Plane2; layer::Int, kw...)
    _check_layer(cyl, layer)
    sol = _plane_conduction_solution(cyl, κ₀)
    return TensND.TensISO{2}(sol.ks[layer] * _plane_layer_gradient(sol, layer, false, false))
end

# ── Layer averages of the properties (Voigt and Reuss bounds) ────────────────

# Every layer is an isotropic tensor of the plane: the average is that of the
# coefficients.
function _plane_average(cyl::LayeredCylinder{T, N}, tensors::Tuple) where {T, N}
    f = _layer_fractions(cyl)
    data = reduce((x, y) -> x .+ y, ntuple(k -> f[k] .* TensND.get_data(tensors[k]), Val(N)))
    return TensND.rebuild(first(tensors), data)
end
layer_property_average(cyl::LayeredCylinder, ::TensND.AbstractTens{O, 2}) where {O} =
    _plane_average(cyl, map(_plane_isotropic, cyl.moduli))
layer_inverse_property_average(cyl::LayeredCylinder, ::TensND.AbstractTens{O, 2}) where {O} =
    _plane_average(cyl, map(inv ∘ _plane_isotropic, cyl.moduli))
