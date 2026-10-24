"""Echoes side of `benchmark_porous.jl`: the effective (k, mu) of a porous solid.

Imported from Julia with PythonCall (`pyimport("porous_echoes")`), the directory
of this file being put on `sys.path` first.
"""
import echoes
from echoes import (rve, ellipsoid, spheroidal, stiff_kmu, homogenize, ISO,
                    VOIGT, REUSS, MT, SC, ASC, DIFF, DIL, DILD, MAX, PCW)

_SCH = {'VOIGT':VOIGT, 'REUSS':REUSS, 'MT':MT, 'SC':SC, 'ASC':ASC,
        'DIFF':DIFF, 'DIL':DIL, 'DILD':DILD, 'MAX':MAX, 'PCW':PCW}

def py_chom_porous(phi, sch_name, omegas=1.0, omegap=1.0, iso_sym=False,
                   ks=72., mus=32., kp=1.e-6, mup=1.e-6,
                   epsrel=1.e-10, maxnb=300):
    Cs = stiff_kmu(ks, mus); Cp = stiff_kmu(kp, mup)
    ver = rve(matrix='SOLID')
    if iso_sym:
        ver['SOLID'] = ellipsoid(shape=spheroidal(omegas), symmetrize=[ISO], prop={'C':Cs})
        ver['PORE']  = ellipsoid(shape=spheroidal(omegap), symmetrize=[ISO], prop={'C':Cp})
    else:
        ver['SOLID'] = ellipsoid(shape=spheroidal(omegas), prop={'C':Cs})
        ver['PORE']  = ellipsoid(shape=spheroidal(omegap), prop={'C':Cp})
    ver['PORE'].fraction  = phi
    ver['SOLID'].fraction = 1.0 - phi
    ver.set_prop('C', Cs)
    def run(nb):
        C = homogenize(prop='C', rve=ver, scheme=_SCH[sch_name],
                       verbose=False, maxnb=nb, epsrel=epsrel,
                       select_best=True)
        return float(C.k), float(C.mu)
    try:
        if sch_name == 'DIFF':
            # echoes integrates the differential scheme in `maxnb` explicit
            # steps, first order: at 300 steps it is 3 % off at φ = 0.9. The
            # reference is the Richardson extrapolation of 3000 and 30000 steps
            # (it agrees with an independent RK4 integration to 1e-7).
            (k1, m1), (k2, m2) = run(3000), run(30000)
            k, mu = k2 + (k2 - k1) / 9, m2 + (m2 - m1) / 9
        else:
            k, mu = run(maxnb)
        return max(k, 0.0), max(mu, 0.0)
    except Exception as e:
        return float('nan'), float('nan')
