import numpy as np
from utils import process_inputs
from datagen import SimulationRunner,SimulationGradientRunner, \
    local_linear_gradients,finite_difference_gradients
from discover import Subspaces,ActiveSubspacePlotter
from domain import UnboundedActiveVariableDomain,BoundedActiveVariableDomain
from respsurf import GaussianProcess
from scipy.optimize import minimize,fminbound
from maps import MinVariableMap

class ActiveSubspaceModel():
    bflag = None # indicates if domain is bounded
    X = None # sample points
    m = None # dimension of inputs
    f = None # user code evaluations
    df = None # gradients or approximations
    fun = None # interface to user code
    dfun = None # interface to user gradient code
    subspace = None # subspaces
    n = None # dimension of active subspace
    domain = None # description of domain
    plotter = None # make diagnostic plots
    rs = None # response surface

    def __init__(self,bflag=False):
        self.bflag = bflag
        self.subspace = Subspaces()

    def compute_subspaces(self,df):
        # set up the active and inactive subspaces
        self.subspace.compute(df)
        n = self.subspace.compute_paritition
        print '%d active variables' % n
        self.subspace.partition(n)

    def build_from_data(self,X,f,df=None):
        X,M,m = process_inputs(X)
        self.X,self.m,self.f = X,m,f

        # if gradients aren't available, estimate them from data
        if df is None:
            df = local_linear_gradients(X,f)
        self.df = df

        self.compute_subspaces(df)

    def build_from_interface(self,m,fun,dfun=None):
        self.m = m

        # number of gradient samples
        M = 6*(m+1)*np.log(m)

        # sample points
        if self.bflag:
            X = np.random.uniform(-1.0,1.0,size=(M,m))
        else:
            X = np.random.normal(size=(M,m))
        self.X = X

        # sample the simulation's outputs
        sr = SimulationRunner(fun)
        self.f = sr.run(X)

        # sample the simulation's gradients
        if dfun == None:
            df = finite_difference_gradients(X,fun)
        else:
            sgr = SimulationGradientRunner(dfun)
            df = sgr.run(X)
        self.df = df

        self.compute_subspaces(df)

    def diagnostics(self):
        ss = self.subspace
        ss.bootstrap()
        asp = ActiveSubspacePlotter()
        self.plotter = asp

        asp.eigenvalues(ss.eigenvalues,e_br=ss.e_br)
        asp.subspace_errors(ss.sub_br)
        asp.eigenvectors(ss.eigenvectors)
        Y = np.dot(self.X,ss.eigenvectors[:,:2])
        asp.sufficient_summary(Y,self.f)

    def set_domain(self):
        ss = self.subspace
        if self.bflag:
            self.domain = BoundedActiveVariableDomain(ss.W1)
        else:
            self.domain = UnboundedActiveVariableDomain(ss.W1)

    def set_response_surface(self):
        ss = self.subspace
        Y = np.dot(self.X,ss.W1)
        rs = GaussianProcess(e=ss.eigenvalues)
        rs.train(Y,self.f)
        self.rs = rs

    def predict(self,X,compvar=False,compgrad=False):
        if self.rs is None:
            self.set_response_surface()

        W1 = self.subspace.W1
        Y = np.dot(X,W1)
        f,df,v = self.rs.predict(Y,compgrad=compgrad,compvar=compvar)
        if compgrad:
            df = np.dot(df,W1.T)
        return f,df,v

    def gradient(self,X):
        f,df,v = self.predict(X,compgrad=True)
        return df

    def mean(self,N=None):
        # adding the N=None until I implement the integration
        return np.mean(self.f)

    def variance(self,N=None):
        # adding the N=None until I implement the integration
        return np.var(self.f)

    def probability(self,lb,ub):
        M = 10000
        if self.bflag:
            X = np.random.uniform(-1.0,1.0,size=(M,self.m))
        else:
            X = np.random.normal(size=(M,self.m))
        f = self.predict(X)[0]
        c = np.logical_and((f>lb), (f<ub))
        prob = np.sum(c.astype(int))/float(M)
        return prob

    def as_minimize(self,fun):
        n = self.subspace.W1.shape[1]
        if self.domain is None:
            self.set_domain()

        opts = {'disp':True,'maxiter':1e4,'ftol':1e-9}
        if self.bflag:
            if n==1:
                yl,yu = self.domain.vertY[0],self.domain.vertY[1]
                ystar = fminbound(fun,yl,yu,xtol=1e-9,maxfun=1e4,full_output=1)[0]
            else:
                y0 = np.random.normal(size=(1,n))
                cons = self.domain.constraints()
                result = minimize(fun,y0,constraints=cons,method='SLSQP',options=opts)
                ystar = result.x
        else:
            y0 = np.random.normal(size=(1,n))
            result = minimize(fun,y0,method='SLSQP',options=opts)
            ystar = result.x

        ss = self.subspace
        mvm = MinVariableMap(ss.W1,ss.W2)
        mvm.train(self.X,self.f,bflag=self.bflag)
        xstar = mvm.inverse(ystar.reshape((1,n)))
        return xstar

    def minimum(self):
        def fun(y):
            n = y.size
            return self.rs.predict(y.reshape((1,n)))[0]
        xmin = self.as_minimize(fun)
        fmin = fun(np.dot(xmin,self.subspace.W1))
        return fmin,xmin

    def __call__(self,x):
        return self.predict(x)[0]


