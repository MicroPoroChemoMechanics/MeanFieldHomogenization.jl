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

function _region_nodes(ell::Ellipsoid{3}, nodes::NTuple{3, Int})
    nr, np, na = nodes
    Q = MFH_Core._basis_matrix(ell.basis)
    a1, a2, a3 = ell.semi_axes
    jac = a1 * a2 * a3                      # |det| of the unit-ball mapping
    ρ, wρ = MFH_Core.gauss_legendre_nodes(nr, 0.0, 1.0)
    c, wc = MFH_Core.gauss_legendre_nodes(np, -1.0, 1.0)  # c = cos θ
    φ = [2π * (k - 1) / na for k in 1:na]
    wφ = fill(2π / na, na)
    pts = Vector{SVector{3, Float64}}()
    wts = Float64[]
    for i in 1:nr, j in 1:np, k in 1:na
        s = sqrt(1 - c[j]^2)
        u = SVector{3, Float64}(ρ[i] * s * cos(φ[k]), ρ[i] * s * sin(φ[k]), ρ[i] * c[j])
        x = Q * SVector{3, Float64}(a1 * u[1], a2 * u[2], a3 * u[3])
        push!(pts, x)
        push!(wts, wρ[i] * ρ[i]^2 * wc[j] * wφ[k] * jac)
    end
    return pts, wts
end

function _region_nodes(ell::Ellipsoid{2}, nodes::NTuple{3, Int})
    nr, _, na = nodes
    Q = MFH_Core._basis_matrix(ell.basis)
    a1, a2 = ell.semi_axes
    jac = a1 * a2
    ρ, wρ = MFH_Core.gauss_legendre_nodes(nr, 0.0, 1.0)
    φ = [2π * (k - 1) / na for k in 1:na]
    wφ = fill(2π / na, na)
    pts = Vector{SVector{2, Float64}}()
    wts = Float64[]
    for i in 1:nr, k in 1:na
        u = SVector{2, Float64}(ρ[i] * cos(φ[k]), ρ[i] * sin(φ[k]))
        x = Q * SVector{2, Float64}(a1 * u[1], a2 * u[2])
        push!(pts, x)
        push!(wts, wρ[i] * ρ[i] * wφ[k] * jac)
    end
    return pts, wts
end
