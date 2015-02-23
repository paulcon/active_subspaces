import abc

class QPSolver(object):
    """
    Abstract Base Class for QP Solvers

    To use custom QP Solver, subclass QPSolver:

    class CustomSolver(QPSolver):
        ...

    and call:

    QPSolver.set_qp_solver(CustomSolver)
    """
    __metaclass__  = abc.ABCMeta

    qp_solver_class = None

    @abc.abstractmethod
    def linear_program_eq(self, c, A, b, lb, ub):
        """
        linear program eq description

        Arguments:
            c:
            A:
            b:
            lb:
            ub:
        Output:

        """

        raise NotImplementedError()

    @abc.abstractmethod
    def quadratic_program_bnd(self, c, Q, lb, ub):
        """
        quadratic program bnd description

        Arguments:
            c:
            Q:
            lb:
            ub:
        Output:

        """

        raise NotImplementedError()

    @abc.abstractmethod
    def quadratic_program_ineq(self, c, Q, A, b):
        """
        quadratic program ineq description

        Arguments:
            c:
            Q:
            A:
            b:
        Output:

        """

        raise NotImplementedError()

    @classmethod
    def set_qp_solver(cls, qp_solver_cls):
        # if !instance(qp_solver_cls(), QPSolver):
            # raise TypeError('qp_solver_cls must be an a sublcass of QPSolver')

        QPSolver.qp_solver_class = qp_solver_cls

    @classmethod
    def get_qp_solver(cls):
        if QPSolver.qp_solver_class != None:
            return QPSolver.qp_solver_class()

        try:
            __import__('imp').find_module('gurobipy')
            from active_subspaces.qp_solvers.gurobi_solver import GurobiSolver
            QPSolver.set_qp_solver(GurobiSolver)
            return GurobiSolver()
        except ImportError as gurobi_error:
            try:
                __import__('imp').find_module('cvxopt')
                from active_subspaces.qp_solvers.cvxopt_solver import CvxoptSolver
                QPSolver.set_qp_solver(CvxoptSolver)
                return CvxoptSolver()
            except ImportError as cvxopt_error:
                raise Exception('no solver present')
