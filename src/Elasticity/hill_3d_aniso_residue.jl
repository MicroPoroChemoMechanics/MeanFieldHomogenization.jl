# =============================================================================
#  hill_3d_aniso_residue.jl
#
#  Thin wrapper around the factored polynomial machinery of
#  `Core.green_residue.jl`.  The Hill residue algorithm specific pieces
#  — per-φ ζ(z) parametrization, 21 Voigt-indexed numerators, Masson
#  log-factor post-processing — live here.  The generic polynomial
#  bookkeeping (acoustic tensor, adjugate, determinant, root finding) is
#  delegated to `Core._build_poly_system`.
#
#  Implementation of Masson (2008) Eq. (20). Float64 only.
# =============================================================================

"""
    _hill_3d_aniso_residue(ell, C₀; abstol, reltol, maxiters) -> AbstractTens{4,3}

Hill polarization tensor of a 3-D ellipsoid in an arbitrarily
anisotropic matrix, evaluated by reducing the 2-D surface integral of
[willis1977](@cite) to a 1-D line quadrature via the
Cauchy residue theorem of [masson2008](@cite). The inner
``\\varphi`` integral is collapsed into a finite sum over the roots of
the determinant of the acoustic tensor lying inside the unit circle.

!!! note "Float64 only, by design"
    The residue algorithm finds polynomial roots (`PolynomialRoots`) and
    therefore only accepts `Float64` coefficients — it does NOT propagate
    `ForwardDiff.Dual`, `Complex`, or symbolic element types.  For those,
    use the `NestedQuadGK` backend (`method = :nestedquadgk`), which is
    type-generic and AD-compatible.  The `:auto` dispatch already picks
    `NestedQuadGK` automatically whenever `eltype(C₀) !== Float64`
    (see `Core/dispatch.jl`); only an explicit `method = :residues` under a
    non-`Float64` matrix will error.
"""
function _hill_3d_aniso_residue(
        ell::Ellipsoid{3}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 100_000
    )
    η = Float64(ell.semi_axes[2] / ell.semi_axes[1])
    ω = Float64(ell.semi_axes[3] / ell.semi_axes[1])

    C₀_princ = TensND.change_tens(C₀, ell.basis)
    C = zeros(Float64, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C[i, j, k, l] = Float64(C₀_princ[i, j, k, l])
    end

    voigt_ij = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

    const_I = ComplexF64(0.0, 1.0)

    function hill_at_phi(φ::Float64)
        m1 = cos(φ)
        m2 = sin(φ) / η
        # ζ(z) = (m1, m2, z/ω) = α₀ + α₁·z with α₀=(m1,m2,0), α₁=(0,0,1/ω).
        α₀ζ = [m1, m2, 0.0]
        α₁ζ = [0.0, 0.0, 1.0 / ω]

        sys = MFH_Core._build_poly_system(C, α₀ζ, α₁ζ)
        adj_poly = sys.adj_poly
        Q = sys.Q
        roots_uhp = sys.roots_uhp

        # Multiplicity-aware processing: promote near-multiple Bairstow /
        # Durand-Kerner clusters to exact multiplicity counts via the
        # polynomial-derivative criterion. Reference point z=i is added
        # explicitly so the log_I term can read off its multiplicity.
        z_list, mults, idxI = MFH_Core._gather_almost_multiple_roots(
            Q, roots_uhp; ref = const_I
        )

        # ζ as a vector of Polynomial{ComplexF64,:z}
        ζ_poly = [Polynomial(ComplexF64[α₀ζ[i], α₁ζ[i]], :z) for i in 1:3]

        # 21 numerator polynomials (upper-Voigt) of degree ≤ 5
        P_num = Vector{Polynomial{ComplexF64, :z}}(undef, 21)
        idx = 0
        for I in 1:6
            vi, vj = voigt_ij[I]
            for J in I:6
                vk, vl = voigt_ij[J]
                idx += 1
                p = (
                    ζ_poly[vi] * (adj_poly[vj, vk] * ζ_poly[vl] + adj_poly[vj, vl] * ζ_poly[vk]) +
                        ζ_poly[vj] * (adj_poly[vi, vk] * ζ_poly[vl] + adj_poly[vi, vl] * ζ_poly[vk])
                )
                P_num[idx] = p * 0.25
            end
        end

        res = zeros(Float64, 21)

        # Term 1: z = i  (multiplicity-aware log_I residue)
        mI = idxI > 0 ? mults[idxI] : 0
        for α in 1:21
            r = MFH_Core._residue_logI(P_num[α], Q, mI)
            isnan(real(r)) && return fill(NaN, 21)   # signal fallback
            res[α] += real(r)
        end

        # Term 2: other UHP roots of Q (multiplicity-aware log_z residue)
        for j in eachindex(z_list)
            j == idxI && continue
            mj = mults[j]
            mj > 0 || continue
            zr = z_list[j]
            for α in 1:21
                r = MFH_Core._residue_logz(P_num[α], Q, zr, mj)
                isnan(real(r)) && return fill(NaN, 21)
                res[α] += real(r)
            end
        end

        return res
    end

    MFH_Core._bump!(MFH_Core.QUADGK_OUTER)
    P_vals, _ = MFH_Core._counted_quadgk(
        hill_at_phi, 0.0, π;
        atol = abstol, rtol = reltol, maxevals = maxiters
    )

    # Detect residue-path failure (NaN propagated from an unsupported
    # multiplicity, typically TI/ORTHO matrix aligned with the ellipsoid axes
    # producing mult ≥ 4 in log_I or mult ≥ 3 in log_z) and silently fall
    # back to the nested-QuadGK backend, which handles those cases via
    # ordinary adaptive quadrature (ForwardDiff-compatible, no extra
    # dependency). Previously this fell back to the DECUHR backend, which is
    # now an optional extension; NestedQuadGK targets the same integral.
    if any(isnan, P_vals)
        return _hill_3d_aniso_nestedquadgk(
            ell, C₀;
            abstol = abstol, reltol = reltol,
            maxiters = maxiters
        )
    end

    P_vals ./= π

    P_arr = zeros(Float64, 3, 3, 3, 3)
    idx = 0
    for I in 1:6, J in I:6
        i, j = voigt_ij[I]
        k, l = voigt_ij[J]
        idx += 1
        v = P_vals[idx]
        P_arr[i, j, k, l] = v;  P_arr[j, i, k, l] = v
        P_arr[i, j, l, k] = v;  P_arr[j, i, l, k] = v
        P_arr[k, l, i, j] = v;  P_arr[l, k, i, j] = v
        P_arr[k, l, j, i] = v;  P_arr[l, k, j, i] = v
    end

    return TensND.Tens(P_arr, ell.basis)
end
