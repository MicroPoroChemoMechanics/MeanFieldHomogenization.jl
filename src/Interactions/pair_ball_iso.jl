# =============================================================================
#  pair_ball_iso.jl — closed-form interaction tensors between two balls
#  (3D) or two disks (2D) embedded in an isotropic reference medium.
#
#  Why these are exact, not asymptotic.  Write Γ⁰ for the real-space Green
#  operator (see `Core/green_operator.jl`).  The average of a smooth field
#  over a ball of radius a is given by the solid mean-value expansion
#
#       ⟨f⟩_{B_a} = f + a²/(2(d+2)) Δf + O(a⁴ Δ²f) ,
#
#  and applying it once for the source region and once for the receiver gives
#
#       Γ^{ab} = V_b [ Γ⁰(r) + (a²+b²)/(2(d+2)) ΔΓ⁰(r) + O(Δ²Γ⁰) ] .
#
#  The elastic Green function is *biharmonic*, so Δ²Γ⁰ ≡ 0 away from the
#  source and the expansion TERMINATES: the formula above is exact for two
#  non-overlapping balls, at any separation.  In conduction Γ⁰ is harmonic,
#  ΔΓ⁰ ≡ 0, and the interaction is exactly the point-dipole field.
#
#  This is the structural reason behind the closed form of Molinari &
#  El Mouden (1996), App. A, whose ρ² = (a²+b²)/R² correction is precisely
#  the ΔΓ⁰ term above — the two were cross-checked component by component
#  when this file was written.
#
#  Elasticity, 3D.  Molinari & El Mouden (1996) App. A, after Berveiller,
#  Fassi-Fehri & Hihi (1987), give the components in the frame whose third
#  axis carries the line of centers.  With
#
#       κ = -b³ / (12 R³ μ (1-ν)) ,     ρ² = (a² + b²)/R² ,
#
#  they read Γ₁₁₁₁ = Γ₂₂₂₂ = κ(1-4ν+9ρ²/5), Γ₁₁₂₂ = κ(-1+3ρ²/5),
#  Γ₁₁₃₃ = κ(2-12ρ²/5), Γ₁₂₁₂ = κ(1-2ν+3ρ²/5), Γ₁₃₁₃ = Γ₂₃₂₃ = κ(1+ν-12ρ²/5)
#  and Γ₃₃₃₃ = κ(-8+8ν+24ρ²/5).
#
#  That set is transversely isotropic about the line of centers and satisfies
#  Γ₁₂₁₂ = (Γ₁₁₁₁ - Γ₁₁₂₂)/2 identically, so it is carried exactly by a
#  five-parameter `TensTI{4}` — five scalars instead of an 81-component array,
#  with the axis handled by TensND rather than by a hand-rolled rotation.
#  The Walpole parameters below follow from the component table by
#
#       p₁ = Γ₃₃₃₃, p₂ = Γ₁₁₁₁+Γ₁₁₂₂, p₃ = √2 Γ₁₁₃₃,
#       p₅ = Γ₁₁₁₁-Γ₁₁₂₂, p₆ = 2 Γ₁₃₁₃ .
# =============================================================================

"""
    _pair_ball_iso(a, b, r, C₀::TensISO{4,3}) -> TensTI{4}

Closed-form interaction tensor between two non-overlapping balls of radii `a`
(receiver) and `b` (source) whose centers are separated by the vector `r`, in
an isotropic elastic reference `C₀`
([Molinari & El Mouden 1996](@cite molinari1996), App. A;
[Berveiller et al. 1987](@cite berveiller1987)).

Returned in the sign convention of [`interaction_tensor`](@ref): contracting
with a uniform polarization of the source ball gives the *average strain
induced in the receiver*. The result is transversely isotropic about the line
of centers and has a strictly vanishing isotropic part.
"""
function _pair_ball_iso(a, b, r::AbstractVector, C₀::TensND.TensISO{4, 3})
    E, ν = MFH_Core.extract_iso_moduli(C₀)
    μ = E / (2 * (1 + ν))
    R = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
    _check_separation(R, a, b)
    T = promote_type(typeof(a), typeof(b), typeof(μ), typeof(R))
    ρ² = (a^2 + b^2) / R^2
    κ = -T(b)^3 / (12 * R^3 * μ * (1 - ν))
    p1 = κ * (-8 + 8ν + 24ρ² / 5)
    p2 = κ * (-4ν + 12ρ² / 5)
    p3 = sqrt(T(2)) * κ * (2 - 12ρ² / 5)
    p5 = κ * (2 - 4ν + 6ρ² / 5)
    p6 = 2κ * (1 + ν - 12ρ² / 5)
    n̂ = [r[1] / R, r[2] / R, r[3] / R]
    return TensND.TensTI{4}(p1, p2, p3, p5, p6, n̂)
end

"""
    _pair_ball_iso(a, b, r, C₀::TensISO{4,2}) -> Tens{4,2}

Plane-strain elastic interaction between two non-overlapping disks. Built
from the exact mean-value identity recalled in the file header,

```math
\\mathbb{T}^{ab} = \\pi b^{2}\\Big[\\Gamma^{0}(r)
  + \\frac{a^{2}+b^{2}}{8}\\,\\Delta\\Gamma^{0}(r)\\Big],
```

with `2(d+2) = 8` in two dimensions. The Laplacian of the plane-strain Green
operator is itself closed-form and, remarkably, independent of the reference
Poisson ratio:

```math
\\Delta\\Gamma^{0}(r) = \\frac{24}{8\\pi\\mu(1-\\nu)\\,r^{4}}
   \\Big[\\mathbb{K}_2 - (2\\underline{n}\\otimes\\underline{n} - \\boldsymbol{1})
                     \\otimes(2\\underline{n}\\otimes\\underline{n} - \\boldsymbol{1})\\Big],
```

``\\mathbb{K}_2`` being the two-dimensional deviatoric projector.
"""
function _pair_ball_iso(a, b, r::AbstractVector, C₀::TensND.TensISO{4, 2})
    E, ν = MFH_Core.extract_iso_moduli(C₀)
    μ = E / (2 * (1 + ν))
    R = sqrt(r[1]^2 + r[2]^2)
    _check_separation(R, a, b)
    V_b = π * b^2
    Γ = MFH_Core.green_operator_iso(C₀, r)
    ΔΓ = _laplacian_green_operator_iso_2d(μ, ν, r)
    return TensND.Tens(V_b * (Γ + (a^2 + b^2) / 8 * ΔΓ))
end

"""
    _pair_ball_iso(a, b, r, K₀::TensISO{2,3}) -> Tens{2,3}

Conduction counterpart in three dimensions. The exterior field of a uniformly
polarized ball is exactly a dipole field whose components are *harmonic* away
from the source, so `ΔΓ⁰ ≡ 0` and the mean-value expansion collapses to its
first term:

```math
\\mathbb{T}^{ab} = \\frac{4\\pi b^{3}}{3}\\,\\Gamma^{0}(r)
 = \\frac{b^{3}}{3\\sigma_0 R^{3}}\\big(3\\,\\underline{n}\\otimes\\underline{n} - \\boldsymbol{1}\\big).
```

It is traceless, and the receiver radius `a` does not appear at all.
"""
function _pair_ball_iso(a, b, r::AbstractVector, K₀::TensND.TensISO{2, 3})
    R = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
    _check_separation(R, a, b)
    V_b = 4 * π * b^3 / 3
    return TensND.Tens(V_b * MFH_Core.green_operator_iso(K₀, r))
end

"""
    _pair_ball_iso(a, b, r, K₀::TensISO{2,2}) -> Tens{2,2}

Two-dimensional conduction counterpart, for two disks of radii `a` and `b`.
Same exactness argument as in 3D — `V_b = π b²` times the plane Green-operator
Hessian. Up to the opposite sign convention this is Eq. (26) of
[Brisard et al. 2023](@cite brisard2023).
"""
function _pair_ball_iso(a, b, r::AbstractVector, K₀::TensND.TensISO{2, 2})
    R = sqrt(r[1]^2 + r[2]^2)
    _check_separation(R, a, b)
    V_b = π * b^2
    return TensND.Tens(V_b * MFH_Core.green_operator_iso(K₀, r))
end

# Laplacian of the plane-strain Green operator, needed for the finite-size
# correction of a disk pair.  Derived in closed form from the kernel of
# `green_operator_iso`; independent of ν, and annihilated by a second
# Laplacian (which is what makes the disk-pair formula exact).
function _laplacian_green_operator_iso_2d(μ, ν, x::AbstractVector)
    r = sqrt(x[1]^2 + x[2]^2)
    n = SVector{2}(x[1] / r, x[2] / r)
    A = 24 * one(r) / (8 * π * μ * (1 - ν) * r^4)
    T = typeof(A)
    δ = (i, j) -> MFH_Core._δ(i, j, T)
    return SArray{Tuple{2, 2, 2, 2}}(
        @inbounds [
            A * (
                    (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)) / 2 - δ(i, j) * δ(k, l) / 2
                    - (2 * n[i] * n[j] - δ(i, j)) * (2 * n[k] * n[l] - δ(k, l))
                )
                for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        ]
    )
end

function _check_separation(R, a, b)
    R > a + b || throw(
        ArgumentError(
            "interaction_tensor: the two regions overlap (separation $(R) ≤ " *
                "$(a) + $(b)). The interaction kernel is only defined for " *
                "non-overlapping inclusions."
        )
    )
    return nothing
end
