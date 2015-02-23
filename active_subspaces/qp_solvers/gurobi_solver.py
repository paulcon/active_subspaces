import numpy as np
from .qp_solver import QPSolver
from gurobipy import *

class GurobiSolver(QPSolver):
    """
    Implementation of QPSolver that uses Gurobi to solve linear and quadratic programs
    """
    def linear_program_eq(self, c, A, b, lb, ub):
        """See QPSolver#linear_program_eq"""

        m,n = A.shape
        model = Model()
        model.setParam('OutputFlag', 0)

        # Add variables to model
        vars = []
        for j in range(n):
            vars.append(model.addVar(lb=lb[j],ub=ub[j],vtype=GRB.CONTINUOUS))
        model.update()

        # Populate linear constraints
        for i in range(m):
            expr = LinExpr()
            for j in range(n):
                expr += A[i,j]*vars[j]
            model.addConstr(expr,GRB.EQUAL,b[i])

        # Populate objective
        obj = LinExpr()
        for j in range(n):
            obj += c[j,0]*vars[j]
        model.setObjective(obj)
        model.update()

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return model.getAttr('x', vars)
        else:
            raise Exception('Gurobi did not solve the LP. Blame Gurobi.')
            return None

    def quadratic_program_bnd(self, c, Q, lb, ub):
        """See QPSolver#quadratic_program_bnd"""

        n = Q.shape[0]
        model = Model()
        model.setParam('OutputFlag', 0)

        # Add variables to model
        vars = []
        for j in range(n):
            vars.append(model.addVar(lb=lb[j],ub=ub[j],vtype=GRB.CONTINUOUS))
        model.update()

        # Populate objective
        obj = QuadExpr()
        for i in range(n):
            for j in range(n):
                obj += Q[i,j]*vars[i]*vars[j]

        for j in range(n):
            obj += c[j]*vars[j]
        model.setObjective(obj)
        model.update()

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return model.getAttr('x', vars)
        else:
            raise Exception('Gurobi did not solve the QP. Blame Gurobi.')
            return None

    def quadratic_program_ineq(self, c, Q, A, b):
        """See QPSolver#quadratic_program_ineq"""

        m,n = A.shape
        model = Model()
        model.setParam('OutputFlag', 0)

        # Add variables to model
        vars = []
        for j in range(n):
            vars.append(model.addVar(vtype=GRB.CONTINUOUS))
        model.update()

        # Populate linear constraints
        for i in range(m):
            expr = LinExpr()
            for j in range(n):
                expr += A[i,j]*vars[j]
            model.addConstr(expr,GRB.GREATER_EQUAL,b[i])

        # Populate objective
        obj = QuadExpr()
        for i in range(n):
            for j in range(n):
                obj += Q[i,j]*vars[i]*vars[j]

        for j in range(n):
            obj += c[j,0]*vars[j]
        model.setObjective(obj)
        model.update()

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return model.getAttr('x', vars)
        else:
            raise Exception('Gurobi did not solve the QP. Blame Gurobi.')
            return None
