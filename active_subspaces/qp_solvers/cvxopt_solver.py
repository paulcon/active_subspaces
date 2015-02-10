from active_subspaces.qp_solvers.qp_solver import QPSolver

class CvxoptSolver(QPSolver):
    def linear_program_eq(self, c, A, b, lb, ub):
        raise NotImplementedError()

    def quadratic_program_bnd(self, c, Q, lb, ub):
        raise NotImplementedError()

    def quadratic_program_ineq(self, c, Q, A, b):
        raise NotImplementedError()
