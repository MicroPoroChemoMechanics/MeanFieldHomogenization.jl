"""
derive_shear_transition.py — derive the closed-form 4×4 shear (Y₂)
amplitude transition `T = M_b(R)^{-1} · M_a(R)` at a perfect interface
between two isotropic layers in (κ, μ) parametrization, then dump
Julia code that builds each entry of T as a polynomial in
`(M_κ_a, M_μ_a, M_κ_b, M_μ_b, R)` divided by a common denominator.

The denominator is `det(M_b(R))` (a closed-form polynomial in
`κ_b, μ_b, R`), and the numerator of each entry is a polynomial in
the moduli of both layers and `R`.

For ALV, each scalar becomes an `n × n` Volterra matrix and the
divides are stable Volterra divides (no explicit M^{-1}).
"""

import sympy as sp
import re

# Symbols
ka, mua, kb, mub, R = sp.symbols('ka mua kb mub R', positive=True)
xa = ka / mua
xb = kb / mub

def M_matrix(r, k, mu):
    x = k / mu
    a2 = 6 * (3*x - 2)
    g2 = 15*x + 11
    a4 = 3 * (x + 1)
    M = sp.Matrix([
        [2*r,                a2 * r**3,                 3 / r**4,             a4 / r**2          ],
        [r,                  g2 * r**3,                -1 / r**4,             1 / r**2           ],
        [4 * mu,              6 * (2 - 3*x) * mu * r**2, -24 * mu / r**5,    -2 * (9*x + 4) * mu / r**3],
        [2 * mu,              2 * (24*x + 5) * mu * r**2,  8 * mu / r**5,     3 * x * mu / r**3],
    ])
    return sp.simplify(M)

print("Building M_a(R), M_b(R)...")
Ma = M_matrix(R, ka, mua)
Mb = M_matrix(R, kb, mub)

print("Computing det(M_b)...")
detMb = sp.factor(sp.det(Mb))
print("det(M_b) =", detMb)

print("Computing M_b^{-1} (will be M_b_adj / det)...")
Mb_inv = Mb.inv()
print("Computing T = M_b^{-1} · M_a (this may take a minute)...")
T = sp.simplify(Mb_inv * Ma)

# Now analyze: each T[i, j] is a rational fn of (ka, mua, kb, mub, R).
# We want common denominator and numerator.
print("\n=== Closed-form shear (Y2) transition entries (perfect interface) ===")
print(f"\ndenominator (common): {sp.simplify(detMb)}")

for i in range(4):
    for j in range(4):
        entry = sp.simplify(T[i, j])
        num, den = sp.fraction(sp.together(entry))
        num = sp.expand(num)
        den = sp.factor(den)
        print(f"\nT[{i+1},{j+1}] = ({num}) / ({den})")
