"""Solvers for the linear and quadratic programs in active subspaces."""
import numpy as np
from scipy.optimize import linprog, minimize

# checking to see if system has gurobi
try:
    HAS_GUROBI = True
    import gurobipy as gpy
except ImportError, e:
    HAS_GUROBI = False
    pass

# string constants for QP solver names
solver_SCIPY = 'SCIPY'
solver_GUROBI = 'GUROBI'

class QPSolver():
    """A class for solving linear and quadratic programs.

    Attributes
    ----------
    solver : str 
        identifies which linear program software to use

    Notes
    -----
    The class checks to see if Gurobi is present. If it is, it uses Gurobi to
    solve the linear and quadratic programs. Otherwise, it uses scipy
    implementations to solve the linear and quadratic programs.
    """
    solver = None

    def __init__(self, solver='GUROBI'):
        """Initialize a QPSolver.

        Parameters
        ----------
        solver : str, optional 
            identifies which linear program software to use. Options are 
            'GUROBI' and 'SCIPY'. (default 'GUROBI')
        """

        if solver==solver_GUROBI and HAS_GUROBI:
            self.solver = solver_GUROBI
        elif solver=='SCIPY':
            self.solver = solver_SCIPY
        else:
            self.solver = solver_SCIPY


    def linear_program_eq(self, c, A, b, lb, ub):
        """Solves an equality constrained linear program with variable bounds.

        This method returns the minimizer of the following linear program.

        minimize  c^T x
        subject to  A x = b
        lb <= x <= ub

        Parameters
        ----------
        c : ndarray 
            m-by-1 matrix for the linear objective function
        A : ndarray 
            M-by-m matrix that contains the coefficients of the linear equality 
            constraints
        b : ndarray
            M-by-1 matrix that is the right hand side of the equality 
            constraints
        lb : ndarray
            m-by-1 matrix that contains the lower bounds on the variables
        ub : ndarray
            m-by-1 matrix that contains the upper bounds on the variables

        Returns
        -------
        x : ndarray
            m-by-1 matrix that is the minimizer of the linear program
        """
        if self.solver == solver_SCIPY:
            return _scipy_linear_program_eq(c, A, b, lb, ub)
        elif self.solver == solver_GUROBI:
            return _gurobi_linear_program_eq(c, A, b, lb, ub)
        else:
            raise ValueError('QP solver {} not available'.format(self.solver))

    def linear_program_ineq(self, c, A, b):
        """Solves an inequality constrained linear program.

        This method returns the minimizer of the following linear program.

        minimize  c^T x
        subject to  A x >= b

        Parameters
        ----------
        c : ndarray
            m-by-1 matrix for the linear objective function
        A : ndarray
            M-by-m matrix that contains the coefficients of the linear equality 
            constraints
        b : ndarray 
            size M-by-1 matrix that is the right hand side of the equality 
            constraints

        Returns
        -------
        x : ndarray
            m-by-1 matrix that is the minimizer of the linear program
        
        """
        if self.solver == solver_SCIPY:
            return _scipy_linear_program_ineq(c, A, b)
        elif self.solver == solver_GUROBI:
            return _gurobi_linear_program_ineq(c, A, b)
        else:
            raise ValueError('QP solver {} not available'.format(self.solver))

    def quadratic_program_bnd(self, c, Q, lb, ub):
        """Solves a quadratic program with variable bounds.

	This method returns the minimizer of the following linear program.

        minimize  c^T x + x^T Q x
        subject to  lb <= x <= ub

        Parameters
        ----------
        c : ndarray
            m-by-1 matrix that contains the coefficients of the linear term in 
            the objective function
        Q : ndarray
            m-by-m matrix that contains the coefficients of the quadratic term 
            in the objective function
        lb : ndarray
            m-by-1 matrix that contains the lower bounds on the variables
        ub : ndarray
            m-by-1 matrix that contains the upper bounds on the variables

        Returns
        -------
        x : ndarray 
            m-by-1 matrix that is the minimizer of the quadratic program
        
        """
        if self.solver == solver_SCIPY:
            return _scipy_quadratic_program_bnd(c, Q, lb, ub)
        elif self.solver == solver_GUROBI:
            return _gurobi_quadratic_program_bnd(c, Q, lb, ub)
        else:
            raise ValueError('QP solver {} not available'.format(self.solver))

    def quadratic_program_ineq(self, c, Q, A, b):
        """Solves an inequality constrained quadratic program.

        
        This method returns the minimizer of the following quadratic program.

        minimize  c^T x + x^T Q x
        subject to  A x >= b

        Parameters
        ----------
        c : ndarray
            m-by-1 matrix that contains the coefficients of the linear term in 
            the objective function
        Q : ndarray
            m-by-m matrix that contains the coefficients of the quadratic term 
            in the objective function
        A : ndarray 
            M-by-m matrix that contains the coefficients of the linear equality 
            constraints
        b : ndarray
            M-by-1 matrix that is the right hand side of the equality 
            constraints

        Returns
        -------
        x : ndarray
            m-by-1 matrix that is the minimizer of the quadratic program.

        """
        if self.solver == solver_SCIPY:
            return _scipy_quadratic_program_ineq(c, Q, A, b)
        elif self.solver == solver_GUROBI:
            return _gurobi_quadratic_program_ineq(c, Q, A, b)
        else:
            raise ValueError('QP solver {} not available'.format(self.solver))

def _scipy_linear_program_eq(c, A, b, lb, ub):

    c = c.reshape((c.size,))
    b = b.reshape((b.size,))

    # make bounds
    bounds = []
    for i in range(lb.size):
        bounds.append((lb[i,0], ub[i,0]))

    res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, options={"disp": False})
    if res.success:
        return res.x.reshape((c.size,1))
    else:
        np.savez('bad_scipy_lp_eq_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, A=A, b=b, lb=lb, ub=ub, res=res)
        raise Exception('Scipy did not solve the LP. Blame Scipy.')
        return None

def _scipy_linear_program_ineq(c, A, b):

    c = c.reshape((c.size,))
    b = b.reshape((b.size,))

    # make unbounded bounds
    bounds = []
    for i in range(c.size):
        bounds.append((None, None))

    A_ub, b_ub = -A, -b
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options={"disp": False})
    if res.success:
        return res.x.reshape((c.size,1))
    else:
        np.savez('bad_scipy_lp_ineq_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, A=A, b=b, res=res)
        raise Exception('Scipy did not solve the LP. Blame Scipy.')
        return None

def _scipy_quadratic_program_bnd(c, Q, lb, ub):

    # define the objective and gradient
    def fun(x):
        f = np.dot(x, c) + np.dot(x, np.dot(Q, x.T))
        return f[0]

    def jac(x):
        j = c.T + 2.0*np.dot(x, Q)
        return j[0]

    # make bounds
    bounds = []
    for i in range(lb.size):
        bounds.append((lb[i,0],ub[i,0]))

    x0 = np.zeros((c.size,))
    res = minimize(fun, x0, method='L-BFGS-B', jac=jac,
                    bounds=bounds, options={"disp": False})

    if res.success:
        xstar = res.x
        if isinstance(xstar, float):
            xstar = np.array([[xstar]])
        return xstar.reshape((c.size,1))
    else:
        np.savez('bad_scipy_qp_bnd_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, Q=Q, lb=lb, ub=ub, res=res)
        raise Exception('Scipy did not solve the LP. Blame Scipy.')
        return None

def _scipy_quadratic_program_ineq(c, Q, A, b):

    b = b.reshape((b.size,))

    # define the objective and gradient
    def fun(x):
        f = np.dot(x, c) + np.dot(x, np.dot(Q, x.T))
        return f[0]

    def jac(x):
        j = c.T + 2.0*np.dot(x, Q)
        return j[0]

    # inequality constraints
    cons = ({'type':'ineq',
            'fun' : lambda x: np.dot(A, x) - b,
            'jac' : lambda x: A})

    x0 = np.zeros((c.size,))
    res = minimize(fun, x0, method='SLSQP', jac=jac,
                    constraints=cons, options={"disp": False})

    if res.success:
        xstar = res.x
        if isinstance(xstar, float):
            xstar = np.array([[xstar]])
        return xstar.reshape((c.size,1))
    else:
        np.savez('bad_scipy_qp_ineq_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, Q=Q, A=A, b=b, res=res)
        raise Exception('Scipy did not solve the LP. Blame Scipy.')
        return None

def _gurobi_linear_program_eq(c, A, b, lb, ub):

    m,n = A.shape
    model = gpy.Model()
    model.setParam('OutputFlag', 0)

    # Add variables to model
    vars = []
    for j in range(n):
        vars.append(model.addVar(lb=lb[j,0], ub=ub[j,0], vtype=gpy.GRB.CONTINUOUS))
    model.update()

    # Populate linear constraints
    for i in range(m):
        expr = gpy.LinExpr()
        for j in range(n):
            expr += A[i,j]*vars[j]
        model.addConstr(expr, gpy.GRB.EQUAL, b[i,0])

    # Populate objective
    obj = gpy.LinExpr()
    for j in range(n):
        obj += c[j,0]*vars[j]
    model.setObjective(obj)
    model.update()

    # Solve
    model.optimize()

    if model.status == gpy.GRB.OPTIMAL:
        return np.array(model.getAttr('x', vars)).reshape((n,1))
    else:
        np.savez('bad_gurobi_lp_eq_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, A=A, b=b, lb=lb, ub=ub, model=model)
        raise Exception('Gurobi did not solve the LP. Blame Gurobi.')
        return None


def _gurobi_linear_program_ineq(c, A, b):

    m,n = A.shape
    model = gpy.Model()
    model.setParam('OutputFlag', 0)

    # Add variables to model
    vars = []
    for j in range(n):
        vars.append(model.addVar(lb=-gpy.GRB.INFINITY,
                    ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS))
    model.update()

    # Populate linear constraints
    for i in range(m):
        expr = gpy.LinExpr()
        for j in range(n):
            expr += A[i,j]*vars[j]
        model.addConstr(expr, gpy.GRB.GREATER_EQUAL, b[i,0])

    # Populate objective
    obj = gpy.LinExpr()
    for j in range(n):
        obj += c[j,0]*vars[j]
    model.setObjective(obj)
    model.update()

    # Solve
    model.optimize()

    if model.status == gpy.GRB.OPTIMAL:
        return np.array(model.getAttr('x', vars)).reshape((n,1))
    else:
        np.savez('bad_gurobi_lp_ineq_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, A=A, b=b, model=model)
        raise Exception('Gurobi did not solve the LP. Blame Gurobi.')
        return None

def _gurobi_quadratic_program_bnd(c, Q, lb, ub):

    n = Q.shape[0]
    model = gpy.Model()
    model.setParam('OutputFlag', 0)

    # Add variables to model
    vars = []
    for j in range(n):
        vars.append(model.addVar(lb=lb[j,0], ub=ub[j,0], vtype=gpy.GRB.CONTINUOUS))
    model.update()

    # Populate objective
    obj = gpy.QuadExpr()
    for i in range(n):
        for j in range(n):
            obj += Q[i,j]*vars[i]*vars[j]

    for j in range(n):
        obj += c[j,0]*vars[j]
    model.setObjective(obj)
    model.update()

    # Solve
    model.optimize()

    if model.status == gpy.GRB.OPTIMAL:
        return np.array(model.getAttr('x', vars)).reshape((n,1))
    else:
        np.savez('bad_gurobi_qp_bnd_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, Q=Q, lb=lb, ub=ub, model=model)
        raise Exception('Gurobi did not solve the QP. Blame Gurobi.')
        return None

def _gurobi_quadratic_program_ineq(c, Q, A, b):

    m,n = A.shape
    model = gpy.Model()
    model.setParam('OutputFlag', 0)

    # Add variables to model
    vars = []
    for j in range(n):
        vars.append(model.addVar(lb=-gpy.GRB.INFINITY,
                    ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS))
    model.update()

    # Populate linear constraints
    for i in range(m):
        expr = gpy.LinExpr()
        for j in range(n):
            expr += A[i,j]*vars[j]
        model.addConstr(expr, gpy.GRB.GREATER_EQUAL, b[i,0])

    # Populate objective
    obj = gpy.QuadExpr()
    for i in range(n):
        for j in range(n):
            obj += Q[i,j]*vars[i]*vars[j]

    for j in range(n):
        obj += c[j,0]*vars[j]
    model.setObjective(obj)
    model.update()

    # Solve
    model.optimize()

    if model.status == gpy.GRB.OPTIMAL:
        return np.array(model.getAttr('x', vars)).reshape((n,1))
    else:
        np.savez('bad_gurobi_qp_ineq_{:010d}'.format(np.random.randint(int(1e9))),
                c=c, Q=Q, A=A, b=b, model=model)
        raise Exception('Gurobi did not solve the QP. Blame Gurobi.')
        return None
