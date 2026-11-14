# Shared by the LayeredCylinder test files (each includes it once).
using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
using ForwardDiff
using JSON

const LCyl = MeanFieldHomogenization.LayeredCylinders

# Transversely isotropic stiffness of Hill moduli (k, l, m, n, p) about `axis`.
lc_ti(k, l, m, n, p; axis = (0.0, 0.0, 1.0)) =
    TensND.ti_from_constants(k + m, k - m, l, n, p, axis)
lc_K(t, a; axis = (0.0, 0.0, 1.0)) = TensND.TensTI{2}(t, a, axis)
lc_cc(t) = TensND.canonical_components(t)
lc_data(t) = collect(TensND.get_data(t))
lc_iso(λ, μ) = TensISO{3}(3 * (λ + 2μ / 3), 2μ)

# Fiber, coating and reference used throughout (Hill moduli written out).
const LC_F = (k = 30.0, l = 12.0, m = 9.0, n = 150.0, p = 14.0)
const LC_M = (k = 4.0, l = 2.0, m = 1.5, n = 7.0, p = 1.5)
const LC_0 = (k = 5.0, l = 1.0, m = 2.0, n = 9.0, p = 3.0)
lc_tensor(L; kw...) = lc_ti(L.k, L.l, L.m, L.n, L.p; kw...)

# The thin layer that stands for an interface, of thickness h around r:
# a stiff isotropic one for a membrane (κˢ, μˢ), a soft one for a spring (sₙ, sₜ).
function lc_thin_membrane(κs, μs, h)
    μ = μs / h
    λs = (κs - μs) / h                  # plane-stress Lamé modulus 2λμ/(λ + 2μ)
    return lc_iso(2μ * λs / (2μ - λs), μ)
end
lc_thin_spring(sn, st, h) = lc_ti(h / sn - h / st, 0.3h, h / st, h, h / st)
