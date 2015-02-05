import numpy as np
from gurobipy import *

def linear_program_eq(c,A,b,lb,ub):
    
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

def quadratic_program_bnd(c,Q,lb,ub):
    
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
        
def quadratic_program_ineq(c,Q,A,b):
    
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
