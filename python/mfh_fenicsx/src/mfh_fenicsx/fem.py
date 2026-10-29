"""A MeanFieldHomogenization material as the stress of a dolfinx problem.

:class:`Stress` builds the stress as a ``FEMExternalOperator`` of
dolfinx-external-operator acting on the strain of a displacement field. The
residual is written with it in UFL as with any stress; its derivative, the
consistent tangent, is the operator's derivative. At each Newton iteration
:meth:`Stress.update` evaluates the strains at every quadrature point and makes
one Julia call that computes the stresses and the tangents together; the
tangent the Jacobian asks for is the one of that call, not a second integration.

In two dimensions the problem is plane strain: the material is integrated in
three dimensions with ε₁₃ = ε₂₃ = ε₃₃ = 0, and the in-plane components of the
stress and of the tangent enter the problem. The out-of-plane stress σ₃₃ is
kept in :attr:`Material.stress` (Mandel slot 2).
"""

import basix.ufl
import numpy as np
import ufl
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

from . import Material, _SLOT, _WEIGHTS

__all__ = ["Newton", "Stress", "strain"]


def strain(u):
    """The small strain ε(u) = sym ∇u."""
    return ufl.sym(ufl.grad(u))


def _mandel_maps(d):
    """Linear maps between the ``d × d`` components and the Mandel components.

    ``to`` (6, d²) takes the row-major components of a symmetric tensor to its
    Mandel vector (the slots outside the plane stay zero); ``back`` (d², 6)
    takes a Mandel vector to the in-plane components.
    """
    to = np.zeros((6, d * d))
    back = np.zeros((d * d, 6))
    for i in range(d):
        for j in range(d):
            a = _SLOT[i, j]
            # Each off-diagonal component is met twice: (i, j) and (j, i).
            to[a, i * d + j] = _WEIGHTS[a] if i == j else _WEIGHTS[a] / 2
            back[i * d + j, a] = 1.0 / _WEIGHTS[a]
    return to, back


class Stress:
    """The stress of ``material`` on the strain of the displacement ``u``.

    ``material`` is a Julia material or Julia code that builds one;
    ``quadrature_degree`` sets the quadrature of the stress space, which the
    measure of the forms must match (``ufl.dx(metadata={"quadrature_degree": q})``,
    or :attr:`dx`). :attr:`sigma` is the stress to write the residual with;
    :attr:`dt` the time step the next :meth:`update` integrates over.
    """

    def __init__(self, material, u, quadrature_degree=2, dt=1.0, **options):
        mesh = u.function_space.mesh
        self.dim = d = mesh.geometry.dim
        if d not in (2, 3):
            raise ValueError(f"Stress: a mesh of dimension 2 or 3 is expected; got {d}")
        tdim = mesh.topology.dim
        imap = mesh.topology.index_map(tdim)
        ncells = imap.size_local + imap.num_ghosts
        element = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(d, d), degree=quadrature_degree)
        from dolfinx import fem

        self.space = fem.functionspace(mesh, element)
        nqp = self.space.element.interpolation_points.shape[0]
        self.material = Material(material, ncells, nqp)
        self.dt = dt
        self.options = options
        self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
        self._to, self._back = _mandel_maps(d)
        self._strain_seen = None
        self.sigma = FEMExternalOperator(strain(u), function_space=self.space, external_function=self._external)

    def forms(self, residual, u):
        """The residual form, written with :attr:`sigma`, and its Jacobian, with the operator replaced.

        Returns ``(F, J, operators)``: the forms to compile with ``fem.form``
        and the external operators :meth:`update` evaluates.
        """
        J = ufl.derivative(residual, u, ufl.TrialFunction(u.function_space))
        J_expanded = ufl.algorithms.expand_derivatives(J)
        F_replaced, F_ops = replace_external_operators(residual)
        J_replaced, J_ops = replace_external_operators(J_expanded)
        return F_replaced, J_replaced, F_ops + J_ops

    def update(self, operators):
        """Evaluate the strains and the operators of ``operators`` (one Julia call)."""
        evaluate_external_operators(operators, evaluate_operands(operators))

    def commit(self):
        """Make the states of the last update the current ones."""
        self.material.commit()
        self._strain_seen = None

    def rollback(self):
        """Discard the states of the last update."""
        self.material.rollback()
        self._strain_seen = None

    # The external function: a derivative multi-index to a callable on the
    # evaluated operand (cells × points·d²).
    def _external(self, derivatives):
        if derivatives == (0,):
            return self._stress_values
        if derivatives == (1,):
            return self._tangent_values
        raise NotImplementedError(f"Stress: derivative {derivatives} of the stress is not available")

    def _integrate(self, eps):
        # The stress and the tangent come from one call; the Jacobian evaluated
        # at the same strains, step and states reads the tangent of that call.
        seen = self._strain_seen
        if seen is None or seen[0] != self.dt or not np.array_equal(seen[1], eps):
            d2 = self.dim * self.dim
            mandel = self._to @ eps.reshape(-1, d2).T
            self.material.update(mandel, self.dt, **self.options)
            self._strain_seen = (self.dt, eps.copy())
        return self.material.stress, self.material.tangent

    def _stress_values(self, eps):
        sigma, _ = self._integrate(eps)
        return (self._back @ sigma).T.ravel()

    def _tangent_values(self, eps):
        _, tangent = self._integrate(eps)
        return np.einsum("Ia,Jb,abn->nIJ", self._back, self._back, tangent, optimize=True).ravel()


class Newton:
    """Newton's method on the residual ``residual(u; v)`` written with ``stress.sigma``.

    The forms are compiled once; :meth:`solve` runs one load step from the
    current ``u``, the Dirichlet conditions ``bcs`` taking their current values,
    and returns the number of iterations. Convergence is read on the residual,
    the rows of the Dirichlet conditions removed: below ``atol``, or below
    ``rtol`` times the first one. The step is not committed: call
    ``stress.commit()`` once it is accepted, ``stress.rollback()`` otherwise.
    :attr:`history` holds the residual norms of the last step.
    """

    def __init__(self, stress, residual, u, bcs, rtol=1.0e-10, atol=1.0e-10, maxiter=25, petsc_options=None):
        from dolfinx import fem
        from petsc4py import PETSc

        F, J, self.operators = stress.forms(residual, u)
        self.F, self.J = fem.form(F), fem.form(J)
        self.stress, self.u, self.bcs = stress, u, bcs
        self.rtol, self.atol, self.maxiter = rtol, atol, maxiter
        self.du = fem.Function(u.function_space)
        self.ksp = PETSc.KSP().create(u.function_space.mesh.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        for key, value in (petsc_options or {}).items():
            PETSc.Options()[key] = value
        self.ksp.setFromOptions()

    def residual_norm(self):
        """The norm of the residual at the current ``u``, operators evaluated."""
        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc
        from petsc4py import PETSc

        self.stress.update(self.operators)
        b = assemble_vector(self.F)
        apply_lifting(b, [self.J], [self.bcs], x0=[self.u.x.petsc_vec], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, self.u.x.petsc_vec, -1.0)
        return b, b.norm()

    def solve(self):
        """One step: Newton iterations from the current ``u``; the number of iterations."""
        from dolfinx.fem.petsc import assemble_matrix
        from petsc4py import PETSc

        self.history = []
        for iteration in range(self.maxiter + 1):
            b, norm = self.residual_norm()
            self.history.append(norm)
            first = self.history[0]
            if norm <= self.atol or norm <= self.rtol * first and iteration > 0:
                b.destroy()
                return iteration
            if iteration == self.maxiter:
                break
            A = assemble_matrix(self.J, self.bcs)
            A.assemble()
            self.ksp.setOperators(A)
            self.ksp.solve(b, self.du.x.petsc_vec)
            self.du.x.scatter_forward()
            self.u.x.array[:] -= self.du.x.array
            A.destroy()
            b.destroy()
        b.destroy()
        raise RuntimeError(f"Newton: no convergence in {self.maxiter} iterations (residual {norm:.3e}, first {first:.3e})")
