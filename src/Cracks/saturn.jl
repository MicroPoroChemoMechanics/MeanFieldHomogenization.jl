# =============================================================================
#  saturn.jl — the "Saturn ring": a spherical pore with an annular crack in its
#  equatorial plane. Only the opening mode of the ring is published (Fett 1994),
#  so only the normal component of the compliance contribution follows.
# =============================================================================

"""
    _fett_F(λ, ν) -> T

Geometric factor of the stress intensity factor ``K_I = \\sigma F(\\lambda)
\\sqrt{\\pi c}`` of an annular crack of length ``c`` grown from a spherical pore of
radius ``a`` in an isotropic matrix of Poisson ratio ``\\nu``, under the remote
stress ``\\sigma`` normal to the crack, ``\\lambda = c/a``
[fett1994](@cite), as written in [kachanov2018; eqs. (4.4.16) and (4.4.17)](@cite):

```math
F(\\lambda) = \\frac{2}{\\pi}\\sqrt{\\frac{\\lambda+2}{\\lambda+1}}
\\left[\\frac{1}{2(1+\\lambda)^2} + \\frac{3}{(7-5\\nu)(1+\\lambda)^4} + 1\\right]
+ \\frac{\\Delta F}{(1+2\\lambda)^2},
\\quad
\\Delta F = 1.122\\,\\frac{39-5\\nu}{27-5\\nu}
- \\frac{6\\sqrt2}{\\pi}\\left(\\frac12 + \\frac{1}{7-5\\nu}\\right).
```
"""
function _fett_F(λ, ν)
    T = promote_type(typeof(λ), typeof(ν))
    ΔF = T(1122) / 1000 * (39 - 5ν) / (27 - 5ν) - 6 * sqrt(T(2)) / T(π) * (one(T) / 2 + 1 / (7 - 5ν))
    bracket = 1 / (2 * (1 + λ)^2) + 3 / ((7 - 5ν) * (1 + λ)^4) + 1
    return 2 / T(π) * sqrt((λ + 2) / (λ + 1)) * bracket + ΔF / (1 + 2λ)^2
end

"""
    saturn_ring_compliance(a, c, C₀; reltol=1e-8) -> (; pore, ring, total)

Normal component ``V H_{nnnn}`` of the compliance contribution of a spherical pore of
radius ``a`` surrounded by an annular crack of length ``c`` in its equatorial plane of
normal ``\\underline{n}`` (outer radius ``a + c``), in an isotropic matrix `C₀`
— the "Saturn ring" [kachanov2012](@cite).

- `pore`: the pore alone, ``\\tfrac{4\\pi}{3}a^3\\,H_{nnnn}`` of a spherical void;
- `ring`: the crack, grown from the surface of the pore at ``\\rho = a`` to
  ``\\rho = a + c`` along circular fronts, by Rice's formula with the factor
  ``F`` of Fett (see `_fett_F`) — ``K_I = F(\\rho/a - 1)\\sqrt{\\pi(\\rho - a)}`` per
  unit normal stress;
- `total = pore + ring`.

The values are those of one object in a volume ``V``: ``\\Delta S_{nnnn} = N\\,
\\mathrm{total}`` for ``N`` objects per unit volume. Only the opening mode of the
ring is published, so the shear components, and therefore the full tensor a scheme
needs, are not available; for ``c \\gg a`` the ring tends to the penny-shaped crack
of radius ``a + c``.
"""
function saturn_ring_compliance(a, c, C₀::TensND.AbstractTens{4, 3}; reltol = 1.0e-8)
    is_hard_numeric(typeof(a)) && !(a > 0 && c > 0) &&
        throw(ArgumentError("a Saturn ring needs a pore radius a > 0 and a crack length c > 0, got a = $a, c = $c"))
    _, ν = TensND.E_nu(_iso_matrix(C₀, "a Saturn ring"))
    front(ρ) = _circle_front(
        2 * oftype(float(ρ), π) * ρ / 4, true,
        _fett_F(ρ / a - 1, ν) * sqrt(oftype(float(ρ), π) * (ρ - a)), zero(ρ), zero(ρ)
    )
    B = rice_cod_tensor(front, a, a + c, C₀, CircularFronts; size = a + c, reltol)
    T = eltype(TensND.canonical_components(B))
    # V H_nnnn = (4π/3)(3/4) ℓ³ B_nn for the penny normalization of `rice_cod_tensor`.
    ring = T(π) * (a + c)^3 * TensND.canonical_components(B)[3, 3]
    H_void = inverse_property_contribution(
        Elasticity.Ellipsoid(a), zero(C₀), C₀
    )
    pore = 4 * T(π) / 3 * a^3 * TensND.canonical_components(H_void)[3, 3, 3, 3]
    return (; pore, ring, total = pore + ring)
end
