"""
    MeanFieldHomogenization.Elliptic

Type-generic complete and incomplete elliptic integrals.

Provides the Legendre integrals ``K(m)``, ``E(m)``, ``F(\\varphi, m)``,
``E(\\varphi, m)`` and Carlson's symmetric integrals ``R_F(x, y, z)``,
``R_D(x, y, z)`` in a form that works for every `Number` subtype —
`Float64`, `ForwardDiff.Dual`, `BigFloat`, `SymPy.Sym`, `Symbolics.Num`,
and arbitrary user-defined scalar types.

Dispatch table
--------------

| Scalar type           | Backend                         |
| :-------------------- | :------------------------------ |
| `Float64`             | `Elliptic.jl` (GSL C binding)   |
| `ForwardDiff.Dual`    | AGM / Carlson (pure arithmetic) |
| `SymPy.Sym`           | `sympy.elliptic_{k,e,f}` via the optional SymPy extension |
| `Symbolics.Num`       | AGM / Carlson (arithmetic-only — verbose but correct; user can `simplify()`) |
| any other `Number`    | AGM / Carlson                   |

Users can add methods to [`ell_K`](@ref), [`ell_E`](@ref), [`ell_F`](@ref),
[`ell_RF`](@ref), [`ell_RD`](@ref) for their own number types — downstream
callers will automatically pick them up.

# References

* AGM for ``K, E``: Abramowitz & Stegun 17.6, NIST DLMF 19.8.
* Carlson's symmetric integrals: B.C. Carlson,
  *Numerical computation of real or complex elliptic integrals*,
  Numerical Algorithms 10 (1995) 13-26.
* Iterative duplication scheme: public-domain SLATEC routines
  `DRF` / `DRD` (B.C. Carlson and E.M. Notis, Ames Laboratory, 1981).
"""
module Elliptic

import Elliptic as _Elliptic
import ForwardDiff

include("api.jl")
include("agm.jl")
include("carlson.jl")

export ell_K, ell_E, ell_F, ell_RF, ell_RD
export is_hard_numeric

end # module
