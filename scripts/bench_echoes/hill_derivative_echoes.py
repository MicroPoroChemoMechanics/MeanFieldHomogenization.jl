"""Echoes side of `benchmark_hill_derivative.jl`: the derivative of the Hill tensor.

Imported from Julia with PythonCall (`pyimport("hill_derivative_echoes")`), the
directory of this file being put on `sys.path` first.
"""
import echoes
import numpy as np


def hill_deriv_iso(ell, k, mu, index):
    C = echoes.stiff_kmu(float(k), float(mu))
    dP = echoes.hill_derivative(ell, C, index, echoes.ISO,
                                epsrel=1e-8, epsabs=1e-8, maxnb=400000, epsroots=1e-8)
    return np.asarray(dP)


def hill_deriv_ortho(ell, params9_echoes_order, index):
    C = echoes.tensor(list(params9_echoes_order), [0.0, 0.0, 0.0])
    dP = echoes.hill_derivative(ell, C, index, echoes.ORTHO, algo=echoes.NUMINT3D,
                                epsrel=1e-8, epsabs=1e-8, maxnb=400000, epsroots=1e-8)
    return np.asarray(dP)
