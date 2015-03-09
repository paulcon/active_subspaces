import numpy as np
from utils.utils import process_inputs
from utils.simrunners import SimulationRunner, SimulationGradientRunner
from utils.plotters import eigenvalues, subspace_errors, eigenvectors, sufficient_summary
from utils.response_surfaces import GaussianProcess
from subspaces import Subspaces
from gradients import local_linear_gradients, finite_difference_gradients
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain
from as_integrals import as_integrate
from as_optimizers import as_minimize, UnboundedMinVariableMap, BoundedMinVariableMap

class ActiveSubspaceModel():
    bflag = None # indicates if domain is bounded
    X = None # sample points
    m = None # dimension of inputs
    f = None # user code evaluations
    df = None # gradients or approximations
    f_runner = None # simulation runner
    df_runner = None # simulation gradient runner
    subspaces = None # subspaces
    domain = None # description of domain
    rs = None # response surface

    def __init__(self, bflag=False):
        self.bflag = bflag

    def build_from_data(self, X, f, df=None):
        X, M, m = process_inputs(X)
        self.X, self.m, self.f = X, m, f

        # if gradients aren't available, estimate them from data
        if df is None:
            df = local_linear_gradients(X, f)
        self.df = df
        
        # compute the active subspaces
        self.subspaces = build_subspaces(df)
        
    def build_from_interface(self, m, fun, dfun=None):
        self.m = m

        # number of gradient samples
        M = 6*(m+1)*np.log(m)

        # sample points
        if self.bflag:
            X = np.random.uniform(-1.0, 1.0, size=(M, m))
        else:
            X = np.random.normal(size=(M, m))
        self.X = X

        # sample the simulation's outputs
        sr = SimulationRunner(fun)
        self.f_runner = sr
        
        # sample the simulation's gradients
        if dfun == None:
            df = finite_difference_gradients(X, sr)
        else:
            sgr = SimulationGradientRunner(dfun)
            df = sgr.run(X)
            self.df_runner = sgr
        self.df = df

        # compute the active subspaces
        self.subspaces = build_subspaces(df)

    def diagnostics(self):
        ss = self.subspaces
        eigenvalues(ss.eigenvalues, e_br=ss.e_br)
        subspace_errors(ss.sub_br)
        eigenvectors(ss.eigenvectors)
        Y = np.dot(self.X, ss.eigenvectors[:,:2])
        sufficient_summary(Y, self.f)

    def set_domain(self):
        ss = self.subspaces
        if self.bflag:
            self.domain = BoundedActiveVariableDomain(ss)
        else:
            self.domain = UnboundedActiveVariableDomain(ss)

    def set_response_surface(self):
        ss = self.subspaces
        Y = np.dot(self.X, ss.W1)
        rs = GaussianProcess(e=ss.eigenvalues)
        rs.train(Y, self.f)
        self.rs = rs

    def predict(self, X, compvar=False, compgrad=False):
        if self.rs is None:
            self.set_response_surface()

        W1 = self.subspaces.W1
        Y = np.dot(X, W1)
        f, df, v = self.rs.predict(Y, compgrad=compgrad, compvar=compvar)
        if compgrad:
            df = np.dot(df, W1.T)
        return f, df, v

    def gradient(self, X):
        f, df, v = self.predict(X, compgrad=True)
        return df

    def mean(self, N):
        if self.domain is None:
            self.set_domain()
        if self.rs is None:
            self.set_response_surface()
        return as_integrate(self, self.domain, self.subspaces, N)[0]
        
    def variance(self, N):
        if self.domain is None:
            self.set_domain()
        if self.rs is None:
            self.set_response_surface()
        
        mu = self.mean(N)
        def varfun(x):
            return (self.predict(x)[0] - mu)**2
        return as_integrate(varfun, self.domain, self.subspaces, N)[0]

    def probability(self, lb, ub):
        if self.domain is None:
            self.set_domain()
        if self.rs is None:
            self.set_response_surface()

        M = 10000
        if self.bflag:
            X = np.random.uniform(-1.0,1.0,size=(M,self.m))
        else:
            X = np.random.normal(size=(M,self.m))
        f = self.predict(X)[0]
        c = np.logical_and((f>lb), (f<ub))
        prob = np.sum(c.astype(int)) / float(M)
        return prob

    def minimum(self):
        if self.domain is None:
            self.set_domain()
        if self.rs is None:
            self.set_response_surface()

        def av_fun(y):
            n = y.size
            return self.rs.predict(y.reshape((1, n)))[0]
            
        ystar, fval = as_minimize(av_fun, self.domain)
        if self.bflag:
            mvm = BoundedMinVariableMap()
        else:
            mvm = UnboundedMinVariableMap()
        mvm.train(self.X, self.f)
        return mvm.inverse(ystar), fval
        
    def __call__(self,x):
        return self.predict(x)[0]

def build_subspaces(df):
    ss = Subspaces()
    ss.compute_spectral_decomposition(df)
    ss.compute_bootstrap_ranges()
    ss.partition()
    return ss
