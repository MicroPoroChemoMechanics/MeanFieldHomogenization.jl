# ═════════════════════════════════════════════════════════════════════════════
#  Public API — dispatching wrappers
# ═════════════════════════════════════════════════════════════════════════════

"""
    ell_K(m) -> T

Complete elliptic integral of the first kind
``K(m) = \\int_0^{\\pi/2} \\mathrm{d}\\theta/\\sqrt{1-m\\sin^2\\theta}``.

`m` is the *parameter* (not the modulus): ``m = k^2``. Type-generic: works
with any `Number` subtype.
"""
@inline ell_K(m::Float64) = _Elliptic.K(m)
@inline ell_K(m::T) where {T <: Number} = _ell_K_agm(m)

"""
    ell_E(m) -> T

Complete elliptic integral of the second kind
``E(m) = \\int_0^{\\pi/2} \\sqrt{1-m\\sin^2\\theta}\\,\\mathrm{d}\\theta``.
"""
@inline ell_E(m::Float64) = _Elliptic.E(m)
@inline ell_E(m::T) where {T <: Number} = _ell_E_agm(m)

"""
    ell_F(φ, m) -> T

Incomplete elliptic integral of the first kind
``F(\\varphi, m) = \\int_0^\\varphi \\mathrm{d}\\theta/\\sqrt{1-m\\sin^2\\theta}``.
"""
@inline ell_F(φ::Float64, m::Float64) = _Elliptic.F(φ, m)
@inline ell_F(φ::Number, m::Number) = _ell_F_inc(φ, m)

"""
    ell_E(φ, m) -> T

Incomplete elliptic integral of the second kind
``E(\\varphi, m) = \\int_0^\\varphi \\sqrt{1-m\\sin^2\\theta}\\,\\mathrm{d}\\theta``.

The 1-argument `ell_E(m)` (complete integral) and the 2-argument
`ell_E(φ, m)` (incomplete integral) coexist via arity dispatch —
identical to the convention of `Elliptic.jl`.
"""
@inline ell_E(φ::Float64, m::Float64) = _Elliptic.E(φ, m)
@inline ell_E(φ::Number, m::Number) = _ell_E_inc(φ, m)
