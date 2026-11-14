# Aligned elastic fibers in an aging matrix: Echoes' ALV homogenization (Mori-Tanaka
# and dilute), the reference of `test/Viscoelasticity/test_layered_cylinder_alv.jl`
# for MFH's LayeredCylinder. Echoes has no infinite cylinder in ALV: a spheroid of
# aspect ratio 1e6 stands for it (an O(ln ω/ω²) difference, below 1e-10).
# Run from the repository root:
#     ~/miniforge3/envs/fenicsx-env/bin/python3 scripts/bench_echoes/cylinder_alv_echoes.py
import json, numpy as np, echoes
from echoes import J4, K4
T = np.array([0.0, 0.3, 0.7, 1.2, 2.0])
k0, mu0, tau = 4.0, 1.5, 0.8
g = lambda tp: 1.0 + 0.5 * tp
Rm = lambda t, tp: (g(tp) * np.exp(-(t - tp) / tau)) * (3 * k0 * J4 + 2 * mu0 * K4) if t >= tp else 0.0 * J4
kf, muf = 30.0, 12.0
Rf = lambda t, tp: (3 * kf * J4 + 2 * muf * K4) if t >= tp else 0.0 * J4
out = {}
for name, sch in (("MT", echoes.MT), ("DIL", echoes.DIL)):
    v = echoes.rve(matrix="M")
    v["M"] = echoes.ellipsoid(shape=echoes.spherical, fraction=0.6, visco_prop={"C": (Rm, echoes.RELAXATION)})
    v["F"] = echoes.ellipsoid(shape=echoes.ellipsoidal(np.array([1.0, 1.0, 1.0e6, 0.0, 0.0, 0.0])), fraction=0.4,
                              visco_prop={"C": (Rf, echoes.RELAXATION)})
    M = np.array(echoes.homogenize_visco(rve=v, scheme=sch, prop="C", time_series=T, unitsize=6,
                                         maxnb=200, epsrel=1e-10, verbose=False), dtype=float)
    out[name] = M.tolist()
    print(name, M.shape, M[2, 2], M[0, 0])
json.dump({"times": T.tolist(), **out}, open("test/Viscoelasticity/echoes_cylinder_alv.json", "w"))
