import numpy as np
from scipy.optimize import linprog, minimize
try:
    import gurobipy as gpy
except ImportError, e:
    pass

# string constants for QP solver names
solver_SCIPY = 'SCIPY'
solver_GUROBI = 'GUROBI'


class QPSolver():
    def __init__(self, solver='SCIPY'):
                
        if not (solver == solver_SCIPY or solver == solver_GUROBI):
            raise ValueError('QP solver %s not available' % solver)
        self.solver = solver

    def linear_program_eq(self, c, A, b, lb, ub):
        """
        Solves the equality constrained linear program
        minimize_x  c^T x
        subject to  A x = b
                    lb <= x <= ub

        Arguments:
            c:
            A:
            b:
            lb:
            ub:
        Output:

        """
        if self.solver == solver_SCIPY:
            c = c.reshape((c.size,))
            b = b.reshape((b.size,))
            return scipy_linear_program_eq(c, A, b, lb, ub)
        elif self.solver == solver_GUROBI:
            return gurobi_linear_program_eq(c, A, b, lb, ub)
        else:
            raise ValueError('QP solver %s not available' % self.solver)

    def quadratic_program_bnd(self, c, Q, lb, ub):
        """
        Solves the bound constrained quadratic program
        minimize_x  c^T x + x^T Q x
        subject to  lb <= x <= ub


        Arguments:
            c:
            Q:
            lb:
            ub:
        Output:

        """
        if self.solver == solver_SCIPY:
            return scipy_quadratic_program_bnd(c, Q, lb, ub)
        elif self.solver == solver_GUROBI:
            return gurobi_quadratic_program_bnd(c, Q, lb, ub)
        else:
            raise ValueError('QP solver %s not available' % self.solver)

    def quadratic_program_ineq(self, c, Q, A, b):
        """
        Solves the linear inequality constrained quadratic program
        minimize_x  c^T x + x^T Q x
        subject to  A x >= b

        Arguments:
            c:
            Q:
            A:
            b:
        Output:

        """
        if self.solver == solver_SCIPY:
            b = b.reshape((b.size,))
            return scipy_quadratic_program_ineq(c, Q, A, b)
        elif self.solver == solver_GUROBI:
            return gurobi_quadratic_program_ineq(c, Q, A, b)
        else:
            raise ValueError('QP solver %s not available' % self.solver)

def scipy_linear_program_eq(c, A, b, lb, ub):
    
    # make bounds
    bounds = []
    for i in range(lb.size):
        bounds.append((lb[i,0],ub[i,0]))

    res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, options={"disp": False})
    return res.x.reshape((c.size,1))

def scipy_quadratic_program_bnd(c, Q, lb, ub):
    
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
        return res.x.reshape(c.shape)
    else:
        raise Exception('Scipy did not solve the QP. Blame Scipy.')
        return None
        
def scipy_quadratic_program_ineq(c, Q, A, b):
    
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
        return res.x.reshape(c.shape)
    else:
        raise Exception('Scipy did not solve the QP. Blame Scipy.')
        return None

def gurobi_linear_program_eq(c, A, b, lb, ub):

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
        raise Exception('Gurobi did not solve the LP. Blame Gurobi.')
        return None

def gurobi_quadratic_program_bnd(c, Q, lb, ub):

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
        raise Exception('Gurobi did not solve the QP. Blame Gurobi.')
        return None

def gurobi_quadratic_program_ineq(c, Q, A, b):

    m,n = A.shape
    model = gpy.Model()
    model.setParam('OutputFlag', 0)

    # Add variables to model
    vars = []
    for j in range(n):
        vars.append(model.addVar(lb=-gpy.GRB.INFINITY, ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS))
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
        raise Exception('Gurobi did not solve the QP. Blame Gurobi.')
        return None
