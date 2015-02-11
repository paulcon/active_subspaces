from .qp_solver import QPSolver

class CvxoptSolver(QPSolver):
    """
    Implementation of QPSolver that uses CVXOpt to solve linear and quadratic programs
    """

    def linear_program_eq(self, c, A, b, lb, ub):
        """See QPSolver#linear_program_eq"""

        raise NotImplementedError()

    def quadratic_program_bnd(self, c, Q, lb, ub):
        """See QPSolver#quadratic_program_bnd"""

        raise NotImplementedError()

    def quadratic_program_ineq(self, c, Q, A, b):
        """See QPSolver#quadratic_program_ineq"""

        raise NotImplementedError()
