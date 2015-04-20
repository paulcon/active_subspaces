import numpy as np
from utils.utils import process_inputs_outputs
from utils.simrunners import SimulationRunner, SimulationGradientRunner
from utils.plotters import eigenvalues, subspace_errors, eigenvectors, sufficient_summary
from as_response_surfaces import ActiveSubspaceResponseSurface
from as_integrals import integrate, av_integrate
from as_optimizers import minimize
from subspaces import Subspaces
from gradients import local_linear_gradients, finite_difference_gradients
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                    UnboundedActiveVariableMap, BoundedActiveVariableMap

class ActiveSubspaceReducedModel():
    bndflag = None # indicates if domain is bounded
    X = None # sample points
    f = None # function evaluations at sample points
    m = None # dimension of inputs
    n = None # dimension of reduced space
    funr = None
    dfunr = None
    av_respsurf = None # response surface

    def __init__(self, bounded_inputs=False):
        self.bndflag = bounded_inputs

    def build_from_data(self, X, f, df=None, avdim=None):
        X, f, M, m = process_inputs_outputs(X, f)
        self.X, self.f, self.m = X, f, m
        
        # if gradients aren't available, estimate them from data
        if df is None:
            df = local_linear_gradients(X, f)
        
        # compute the active subspace
        ss = Subspaces()
        ss.compute(df)
        if avdim is not None:
            ss.partition(avdim)
        self.n = ss.W1.shape[1]
        print 'Dimension of subspace is {:d}'.format(self.n)
        
        # set up the active variable domain and map
        if self.bndflag:
            avdom = UnboundedActiveVariableDomain(ss)
            avmap = UnboundedActiveVariableMap(avdom)
        else:
            avdom = BoundedActiveVariableDomain(ss)
            avmap = BoundedActiveVariableMap(avdom)
            
        # build the response surface
        avrs = ActiveSubspaceResponseSurface(avmap)
        avrs.train_with_data(X, f)
        self.av_respsurf = avrs
        
    def build_from_interface(self, m, fun, dfun=None, avdim=None):
        self.m = m

        # number of gradient samples
        M = 6*(m+1)*np.log(m)

        # sample points for gradients
        if self.bflag:
            X = np.random.uniform(-1.0, 1.0, size=(M, m))
        else:
            X = np.random.normal(size=(M, m))
        funr = SimulationRunner(fun) 
        f = funr.run(X)
        self.X, self.f, self.funr = X, f, funr
        
        # sample the simulation's gradients
        if dfun == None:
            df = finite_difference_gradients(X, fun)
        else:
            dfunr = SimulationGradientRunner(dfun)
            df = dfunr.run(X)
            self.dfunr = dfunr

        # compute the active subspace
        ss = Subspaces()
        ss.compute(df)
        if avdim is not None:
            ss.partition(avdim)
        self.n = ss.W1.shape[1]
        print 'Dimension of subspace is {:d}'.format(self.n)
        
        # set up the active variable domain and map
        if self.bndflag:
            avdom = UnboundedActiveVariableDomain(ss)
            avmap = UnboundedActiveVariableMap(avdom)
        else:
            avdom = BoundedActiveVariableDomain(ss)
            avmap = BoundedActiveVariableMap(avdom)
            
        # build the response surface
        avrs = ActiveSubspaceResponseSurface(avmap)
        avrs.train_with_interface(fun, np.power(10,self.n))
        self.av_respsurf = avrs

    def diagnostics(self):
        ss = self.av_respsurf.avmap.domain.subspaces
        eigenvalues(ss.eigenvalues[:10,0], e_br=ss.e_br[:10,:])
        subspace_errors(ss.sub_br[:10,:])
        eigenvectors(ss.eigenvectors[:,:4])
        Y = np.dot(self.X, ss.eigenvectors[:,:2])
        sufficient_summary(Y, self.f)

    def predict(self, X, compvar=False, compgrad=False):
        return self.avrs.predict(X, compgrad=compgrad, compvar=compvar)

    def average(self, N):
        if self.funr is not None:
            mu, lb, ub = integrate(self.funr, self.av_respsurf.avmap, N)
        else:
            mu = av_integrate(self.av_respsurf, self.av_respsurf.avmap, N)
            lb, ub = None, None
        return mu, lb, ub

    def probability(self, lb, ub):

        M = 10000
        if self.bndflag:
            X = np.random.uniform(-1.0,1.0,size=(M,self.m))
        else:
            X = np.random.normal(size=(M,self.m))
        f = self.av_respsurf(X)[0]
        c = np.logical_and((f>lb), (f<ub))
        p = np.sum(c.astype(int)) / float(M)
        lb, ub = p+2.58*np.sqrt(p*(1-p)/M), p-2.58*np.sqrt(p*(1-p)/M)
        return p, lb, ub

    def minimum(self):
        xstar, fstar = minimize(self.av_respsurf, self.X, self.f)
        return fstar, xstar
        
