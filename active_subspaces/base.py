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
    bounded_inputs = None # indicates if domain is bounded
    X = None # sample points
    f = None # function evaluations at sample points
    m = None # dimension of inputs
    n = None # dimension of reduced space
    fun_run = None
    dfun_run = None
    av_respsurf = None # response surface

    def build_from_data(self, X, f, df=None, avdim=None, bounded_inputs=False):
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
        print 'The dimension of the active subspace is {:d}.'.format(self.n)
        
        # set up the active variable domain and map
        if bounded_inputs:
            avdom = BoundedActiveVariableDomain(ss)
            avmap = BoundedActiveVariableMap(avdom)
        else:
            avdom = UnboundedActiveVariableDomain(ss)
            avmap = UnboundedActiveVariableMap(avdom)
        self.bounded_inputs = bounded_inputs
            
        # build the response surface
        avrs = ActiveSubspaceResponseSurface(avmap)
        avrs.train_with_data(X, f)
        self.av_respsurf = avrs
        
    def build_from_interface(self, m, fun, dfun=None, avdim=None, bounded_inputs=False):
        self.m = m

        # number of gradient samples
        M = int(np.floor(6*(m+1)*np.log(m)))

        # sample points for gradients
        if bounded_inputs:
            X = np.random.uniform(-1.0, 1.0, size=(M, m))
        else:
            X = np.random.normal(size=(M, m))
        self.bounded_inputs = bounded_inputs
            
        fun_run = SimulationRunner(fun) 
        f = fun_run.run(X)
        self.X, self.f, self.fun_run = X, f, fun_run
        
        # sample the simulation's gradients
        if dfun == None:
            df = finite_difference_gradients(X, fun)
        else:
            dfun_run = SimulationGradientRunner(dfun)
            df = dfun_run.run(X)
            self.dfun_run = dfun_run

        # compute the active subspace
        ss = Subspaces()
        ss.compute(df)
        if avdim is not None:
            ss.partition(avdim)
        self.n = ss.W1.shape[1]
        print 'The dimension of the active subspace is {:d}.'.format(self.n)
        
        # set up the active variable domain and map
        if bounded_inputs:
            avdom = BoundedActiveVariableDomain(ss)
            avmap = BoundedActiveVariableMap(avdom)
        else:
            avdom = UnboundedActiveVariableDomain(ss)
            avmap = UnboundedActiveVariableMap(avdom)
        self.bounded_inputs = bounded_inputs
            
        # build the response surface
        avrs = ActiveSubspaceResponseSurface(avmap)
        avrs.train_with_interface(fun, int(np.power(5,self.n)))
        self.av_respsurf = avrs

    def diagnostics(self):
        ss = self.av_respsurf.avmap.domain.subspaces
        eigenvalues(ss.eigenvalues[:10,0], e_br=ss.e_br[:10,:])
        subspace_errors(ss.sub_br[:10,:])
        eigenvectors(ss.eigenvectors[:,:4])
        Y = np.dot(self.X, ss.eigenvectors[:,:2])
        sufficient_summary(Y, self.f)

    def predict(self, X, compvar=False, compgrad=False):
        if X.shape[1] != self.m:
            raise Exception('The dimension of the points is {:d} but should be {:d}.'.format(X.shape[1], self.m))
        return self.av_respsurf.predict(X, compgrad=compgrad, compvar=compvar)

    def average(self, N):
        if self.fun_run is not None:
            mu, lb, ub = integrate(self.fun_run, self.av_respsurf.avmap, N)
        else:
            mu = av_integrate(self.av_respsurf, self.av_respsurf.avmap, N)
            lb, ub = None, None
        return mu, lb, ub

    def probability(self, lb, ub):

        M = 10000
        if self.bounded_inputs:
            X = np.random.uniform(-1.0,1.0,size=(M,self.m))
        else:
            X = np.random.normal(size=(M,self.m))
        f = self.av_respsurf(X)
        c = np.all(np.hstack(( f>lb, f<ub )), axis=1)
        p = np.sum(c) / float(M)
        plb, pub = p+2.58*np.sqrt(p*(1-p)/M), p-2.58*np.sqrt(p*(1-p)/M)
        return p, plb, pub

    def minimum(self):
        xstar, fstar = minimize(self.av_respsurf, self.X, self.f)
        return fstar, xstar
        
