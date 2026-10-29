"""Materials of MeanFieldHomogenization.jl at the quadrature points of a finite-element code.

A finite-element code evaluates its strains at all the quadrature points of a
mesh in one array; :class:`Material` hands that array to Julia once per
assembly, where ``MeanFieldHomogenization.update_points!`` computes every stress
and consistent tangent, on as many threads as Julia was started with, and
writes them into arrays that NumPy reads without a copy.

Julia is embedded through ``juliacall``. :func:`start` must run before any
other call; it uses the Julia and the Julia project the environment names
(``PYTHON_JULIAPKG_EXE``, ``PYTHON_JULIAPKG_PROJECT``) and installs nothing
(``PYTHON_JULIAPKG_OFFLINE=yes``): the project must hold
MeanFieldHomogenization and PythonCall.

Tensors cross the boundary in Kelvin–Mandel components, MeanFieldHomogenization's
convention, ordered (11, 22, 33, √2·23, √2·13, √2·12); :func:`mandel_from_tensor`,
:func:`tensor_from_mandel` and :func:`tensor4_from_mandel` convert from and to
the full tensors a finite-element code works with.
"""

import atexit
import ctypes
import os
import signal
import sys
import time

import numpy as np

__all__ = [
    "start",
    "julia",
    "restore_signals",
    "Material",
    "mandel_from_tensor",
    "tensor_from_mandel",
    "tensor4_from_mandel",
]

_SQRT2 = np.sqrt(2.0)
# The Mandel slot of each index pair (i, j) of a symmetric tensor of order 2.
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_WEIGHTS = np.array([1.0, 1.0, 1.0, _SQRT2, _SQRT2, _SQRT2])
_SLOT = np.empty((3, 3), dtype=int)
for _a, (_i, _j) in enumerate(_PAIRS):
    _SLOT[_i, _j] = _SLOT[_j, _i] = _a

_JL = None


# Julia handles SIGSEGV (its threads stop at safepoints through it), SIGBUS,
# SIGILL, SIGFPE and SIGINT. A library initialized after Julia may reset them:
# gmsh.initialize() puts them back to their defaults, and the next garbage
# collection on several threads then kills the process. Julia's handlers are
# saved when it starts and put back before every call into Julia and at exit.
_SIGNALS = tuple(getattr(signal, name) for name in ("SIGSEGV", "SIGBUS", "SIGILL", "SIGFPE", "SIGINT"))


class _SigAction(ctypes.Structure):
    # struct sigaction of the C library: glibc (Linux), then Darwin.
    if sys.platform.startswith("linux"):
        _fields_ = [("handler", ctypes.c_void_p), ("mask", ctypes.c_ulong * 16), ("flags", ctypes.c_int), ("restorer", ctypes.c_void_p)]
    else:
        _fields_ = [("handler", ctypes.c_void_p), ("mask", ctypes.c_uint32), ("flags", ctypes.c_int)]


_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform != "win32" else None
_JULIA_SIGNALS = {}


def _save_signals():
    if _LIBC is None:
        return
    for number in _SIGNALS:
        action = _SigAction()
        _LIBC.sigaction(number, None, ctypes.byref(action))
        _JULIA_SIGNALS[number] = action


def restore_signals():
    """Put back the signal handlers Julia installed when it started.

    mfh_fenicsx does it before each of its calls into Julia; call it before
    calling Julia directly after a library that resets the handlers (gmsh).
    """
    for number, action in _JULIA_SIGNALS.items():
        _LIBC.sigaction(number, ctypes.byref(action), None)


def start(project=None, threads=None, julia_exe=None):
    """Start Julia and load MeanFieldHomogenization; return Julia's ``Main``.

    ``project`` is the Julia project (it must hold MeanFieldHomogenization and
    PythonCall), ``threads`` the number of Julia threads, ``julia_exe`` the Julia
    executable; each defaults to the environment, which an explicit argument
    overrides. Nothing is installed. A second call returns the running Julia;
    its settings can no longer change.
    """
    global _JL
    if _JL is not None:
        return _JL
    if project is not None:
        os.environ["PYTHON_JULIAPKG_PROJECT"] = str(project)
    if julia_exe is not None:
        os.environ["PYTHON_JULIAPKG_EXE"] = str(julia_exe)
    if threads is not None:
        os.environ["PYTHON_JULIACALL_THREADS"] = str(threads)
    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    from juliacall import Main as jl

    _save_signals()
    # Registered after juliacall's own exit hook, so that it runs before it.
    atexit.register(restore_signals)
    jl.seval("using MeanFieldHomogenization")
    # One compiled closure, so that a call costs no `seval`.
    jl.seval(
        "_mfh_fenicsx_update!(σ, 𝕂, ε, states, Δt, material; options...) = "
        "(update_points!(σ, 𝕂, ε, states, Δt, material; options...); nothing)"
    )
    # The committed values of an internal variable of a point law at every point.
    jl.seval(
        """begin
            # A scalar variable is one row, as a tensor is six.
            _mfh_fenicsx_rows(r::Integer) = r:r
            _mfh_fenicsx_rows(r) = r
            function _mfh_fenicsx_internal(states, law, name::Symbol)
                points = reduce(vcat, states.old)
                return [st.y[i] for i in _mfh_fenicsx_rows(law.layout[name]), st in points]
            end
        end"""
    )
    _JL = jl
    return jl


def julia():
    """Julia's ``Main``, once :func:`start` has run, its signal handlers in place."""
    if _JL is None:
        raise RuntimeError("mfh_fenicsx: call mfh_fenicsx.start() first")
    restore_signals()
    return _JL


def mandel_from_tensor(t):
    """Mandel components ``(6, n)``, Fortran-ordered, of ``n`` symmetric tensors ``(n, 3, 3)``."""
    t = np.asarray(t, dtype=float).reshape(-1, 3, 3)
    v = np.empty((6, t.shape[0]), order="F")
    for a, (i, j) in enumerate(_PAIRS):
        v[a] = _WEIGHTS[a] * t[:, i, j]
    return v


def tensor_from_mandel(v):
    """The ``(n, 3, 3)`` symmetric tensors of Mandel components ``(6, n)``."""
    v = np.asarray(v)
    return (v[_SLOT] / (_WEIGHTS[_SLOT])[..., None]).transpose(2, 0, 1)


def tensor4_from_mandel(k):
    """The ``(n, 3, 3, 3, 3)`` tensors, with the minor symmetries, of Mandel matrices ``(6, 6, n)``."""
    k = np.asarray(k)
    w = _WEIGHTS[_SLOT]
    full = k[_SLOT[:, :, None, None], _SLOT[None, None, :, :]]
    full = full / (w[:, :, None, None] * w[None, None, :, :])[..., None]
    return full.transpose(4, 0, 1, 2, 3)


class Material:
    """A mechanical material of MeanFieldHomogenization at ``ncells × nqp`` points.

    ``material`` is a Julia material (a law of ``MaterialLaws``, a homogenized
    composite, a polycrystal), built with :func:`julia` or given as Julia code.
    The points are ordered cell by cell, ``nqp`` per cell, as a quadrature space
    of dolfinx orders them. :meth:`update` computes the stresses and tangents of
    a step without committing it; :meth:`commit` makes its states the current
    ones once the step has converged, :meth:`rollback` discards them.
    :attr:`julia_time` sums the seconds spent in Julia by :meth:`update`, over
    :attr:`calls` calls.
    """

    def __init__(self, material, ncells, nqp):
        jl = julia()
        self.material = jl.seval(material) if isinstance(material, str) else material
        self.ncells, self.nqp = int(ncells), int(nqp)
        self.n = self.ncells * self.nqp
        self.states = jl.QuadratureStates(self.material, self.ncells, self.nqp)
        # Buffers Julia writes into, read by NumPy without a copy.
        self.stress = np.zeros((6, self.n), order="F")
        self.tangent = np.zeros((6, 6, self.n), order="F")
        self._strain = np.zeros((6, self.n), order="F")
        self.julia_time, self.calls = 0.0, 0

    def update(self, strain, dt, **options):
        """Stresses and tangents at the Mandel strains ``(6, n)`` over a step ``dt``.

        ``options`` are passed to ``update_points!`` (``algorithm``, ``ntasks``,
        ``external``). Returns the buffers ``(stress (6, n), tangent (6, 6, n))``, overwritten by
        the next call. A point that fails to integrate raises Julia's
        ``LocalIntegrationError`` (a ``juliacall.JuliaError``).
        """
        strain = np.asarray(strain, dtype=float)
        if strain.shape != (6, self.n):
            raise ValueError(f"strain of shape {strain.shape}, expected (6, {self.n})")
        self._strain[...] = strain
        jl = julia()
        start = time.perf_counter()
        jl._mfh_fenicsx_update_b(self.stress, self.tangent, self._strain, self.states, float(dt), self.material, **options)
        self.julia_time += time.perf_counter() - start
        self.calls += 1
        return self.stress, self.tangent

    def internal(self, name):
        """The committed values ``(width, n)`` of the internal variable ``name`` of a point law.

        ``width`` is 1 for a scalar (a crack density, a cumulated plastic strain),
        6 for a tensor in Mandel components.
        """
        jl = julia()
        return np.asarray(jl._mfh_fenicsx_internal(self.states, self.material, jl.Symbol(name)))

    def commit(self):
        """Make the states of the last update the current ones."""
        julia().commit_b(self.states)

    def rollback(self):
        """Discard the states of the last update."""
        julia().rollback_b(self.states)
