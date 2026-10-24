# =============================================================================
#  pair_quadrature.jl — brute-force evaluation of the two-inclusion
#  interaction integral by a product quadrature rule.
#
#      𝕋^{ab} = (1/|Ω_a|) ∫_{Ω_a} ∫_{Ω_b} 𝔾⁰(x - y) dV_y dV_x
#
#  This is the reference implementation: slow, geometry-agnostic, and used to
#  validate the closed forms and the multipole truncation.  It is never the
#  `:auto` back-end.
#
#  Each ellipsoid is mapped from the unit ball, x = x_c + Q·diag(a)·u with
#  ‖u‖ ≤ 1, so the domain becomes a product and the integrand stays smooth
#  (the two regions do not overlap).  On the unit ball we use
#
#    * Gauss-Legendre in the radial variable — after absorbing the ρ^{d-1}
#      Jacobian the integrand is a smooth function of ρ,
#    * Gauss-Legendre in cos θ (3D),
#    * the trapezoidal rule in the azimuth φ, which is spectrally accurate
#      on a periodic interval.
#
#  Cost is n_rad · n_pol · n_azi per region, squared for the pair — hence the
#  modest defaults and the "validation only" status.
# =============================================================================

"""
    _pair_quadrature(incl_a, incl_b, r, P₀; nodes=(6, 6, 12), kw...) -> AbstractTens

Direct product-quadrature evaluation of the two-inclusion interaction
integral. `nodes` gives the number of radial, polar and azimuthal points per
region (the polar entry is ignored in 2D).

Geometry-agnostic and used as the cross-validation oracle for the closed-form
and multipole back-ends; it is far too slow to assemble an N-body system.
"""
function _pair_quadrature(
        incl_a, incl_b, r::AbstractVector, P₀::TensND.AbstractTens;
        nodes::NTuple{3, Int} = (6, 6, 12), kw...
    )
    pa, wa = _region_nodes(incl_a, nodes)
    pb, wb = _region_nodes(incl_b, nodes)
    # Weights are normalized to sum to 1 over Ω_a (the 1/|Ω_a| prefactor) and
    # to |Ω_b| over Ω_b (the plain volume integral).
    Wa = sum(wa)
    acc = zero(MFH_Core._green_operator(P₀, r; kw...))
    for (xa, ua) in zip(pa, wa), (xb, ub) in zip(pb, wb)
        acc = acc + (ua * ub) * MFH_Core._green_operator(P₀, r .+ xb .- xa; kw...)
    end
    return TensND.Tens(acc / Wa)
end

# ─── Node generation on an ellipsoid, centered at the origin ────────────────

# The nodes are listed azimuth fastest, then polar angle, then radius. The
# element type is the one of the frame and the semi-axes, so that a dual radius
# carries through.
function _region_nodes(ell::Ellipsoid{3}, nodes::NTuple{3, Int})
    nr, np, na = nodes
    Q = SMatrix{3, 3}(TensND.vecbasis(ell.basis, :cov))
    a = SVector(ell.semi_axes)
    jac = prod(a)                           # |det| of the unit-ball mapping
    ρ, wρ = MFH_Core.gauss_legendre_nodes(nr, 0.0, 1.0)
    c, wc = MFH_Core.gauss_legendre_nodes(np, -1.0, 1.0)  # c = cos θ
    φ = [2π * (k - 1) / na for k in 1:na]
    unit(i, j, k) = ρ[i] * SVector(sqrt(1 - c[j]^2) * cos(φ[k]), sqrt(1 - c[j]^2) * sin(φ[k]), c[j])
    pts = vec([Q * (a .* unit(i, j, k)) for k in 1:na, j in 1:np, i in 1:nr])
    wts = vec([wρ[i] * ρ[i]^2 * wc[j] * (2π / na) * jac for k in 1:na, j in 1:np, i in 1:nr])
    return pts, wts
end

function _region_nodes(ell::Ellipsoid{2}, nodes::NTuple{3, Int})
    nr, _, na = nodes
    Q = SMatrix{2, 2}(TensND.vecbasis(ell.basis, :cov))
    a = SVector(ell.semi_axes)
    jac = prod(a)
    ρ, wρ = MFH_Core.gauss_legendre_nodes(nr, 0.0, 1.0)
    φ = [2π * (k - 1) / na for k in 1:na]
    pts = vec([Q * (a .* (ρ[i] * SVector(cos(φ[k]), sin(φ[k])))) for k in 1:na, i in 1:nr])
    wts = vec([wρ[i] * ρ[i] * (2π / na) * jac for k in 1:na, i in 1:nr])
    return pts, wts
end
