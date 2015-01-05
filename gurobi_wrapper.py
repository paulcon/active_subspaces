import numpy as np
from gurobipy import *

def linear_program_eq(c,A,b,lb,ub):
    
    m,n = A.shape
    model = Model()

    # Add variables to model
    for j in range(n):
        model.addVar(lb=lb[j],ub=ub[j],vtype=GRB.CONTINUOUS)
    model.update()
    vars = model.getVars()

    # Populate linear constraints
    for i in range(m):
        expr = LinExpr()
        for j in range(n):
            expr += A[i,j]*vars[j]
        model.addConstr(expr,GRB.EQUAL,b[i])

    # Populate objective
    obj = LinExpr()
    for j in range(n):
        obj += c[j]*vars[j]
    model.setObjective(obj)
    model.update()

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.getAttr('x', vars)
    else:
        return None

def quadratic_program_bnd(c,Q,lb,ub):
    
    n = Q.shape[0]
    model = Model()

    # Add variables to model
    for j in range(n):
        model.addVar(lb=lb[j],ub=ub[j],vtype=GRB.CONTINUOUS)
    model.update()
    vars = model.getVars()

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
        return None

if __name__ == '__main__':
    
    m,n = 3,4
    c = np.zeros(n)
    A = np.eye(3,4)
    b = np.ones(m)
    lb = -np.ones(n)
    ub = np.ones(n)
    x = linear_program_eq(c,A,b,lb,ub)
    print x
    
    c = np.zeros(n)
    Q = np.eye(n)
    lb = -np.ones(n)
    ub = np.ones(n)
    x = quadratic_program_bnd(c,Q,lb,ub)
    print x
    