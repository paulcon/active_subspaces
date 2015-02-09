# #!/usr/bin/python

# import numpy as np
# import pandas as pn
# import active_subspaces as ac
# import zonotopes as zn
# import asutils as au
# import scipy.spatial as sp
# import gaussian_quadrature as gq
# from active_subspaces import VariableMap,OptVariableMap
# import matplotlib.pyplot as plt

# def quadtest(X):
#     M,m = X.shape
#     B = np.random.normal(size=(m,m))
#     Q = np.linalg.qr(B)[0]
#     e = np.array([10**(-i) for i in range(1,m+1)])
#     A = np.dot(Q,np.dot(np.diagflat(e),Q.T))
#     f = np.zeros((M,1))
#     for i in range(M):
#         z = X[i,:]
#         f[i] = 0.5*np.dot(z,np.dot(A,z.T))

#     return f,e,Q,A

# def quad_fun(x):
#     A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
#        [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
#        [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
#     f = 0.5*np.dot(x.T,np.dot(A,x))
#     df = np.dot(A,x)
#     return f,df

# def quad_fun_nograd(x):
#     A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
#        [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
#        [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
#     f = 0.5*np.dot(x.T,np.dot(A,x))
#     return f

# def test_interface(fun):
#     M,m = 300,3
#     X = np.random.uniform(-1.0,1.0,(M,m))
#     F,dF = ac.sample_function(X,fun,dflag=True)

#     e,W,e_br,sub_br = ac.compute_active_subspace(dF,2)

#     ac.plot_eigenvalues(e,e_br=e_br)
#     ac.plot_subspace_errors(sub_br)
#     ac.plot_eigenvectors(W[:,:1])

#     # make 1d sufficient summary plots
#     y = np.dot(X,W[:,0])
#     ac.sufficient_summary_plot(y,F)

#     # make 2d sufficient summary plots
#     y = np.dot(X,W[:,:2])
#     ac.sufficient_summary_plot(y,F)

#     return 0

# def test_interface_nograd(fun):
#     MM,M,m = 30,300,3
#     X = np.random.uniform(-1.0,1.0,(M,m))
#     F = ac.sample_function(X,fun,dflag=False)

#     # test finite difference gradients
#     ddF = ac.finite_difference_gradients(X,fun)

#     # test local linear model gradients
#     XX = np.random.uniform(-1.0,1.0,(MM,m))
#     ddF = ac.local_linear_gradients(X,F,XX)

#     e,W,e_br,sub_br = ac.compute_active_subspace(ddF,2)

#     ac.plot_eigenvalues(e,e_br=e_br)
#     ac.plot_subspace_errors(sub_br)
#     ac.plot_eigenvectors(W[:,:1])

#     # make 1d sufficient summary plots
#     y = np.dot(X,W[:,0])
#     ac.sufficient_summary_plot(y,F)

#     # make 2d sufficient summary plots
#     y = np.dot(X,W[:,:2])
#     ac.sufficient_summary_plot(y,F)

#     return 0

# def test_load_data():

#     # load data
#     df = pn.DataFrame.from_csv('data/quad_nograd.txt')
#     data = df.values
#     X = data[:,0:3]
#     F = data[:,3]
#     M,m = X.shape

#     # test local linear model gradients
#     MM = 100
#     XX = np.random.uniform(-1.0,1.0,(MM,m))
#     ddF = ac.local_linear_gradients(X,F,XX)

#     e,W,e_br,sub_br = ac.compute_active_subspace(ddF,2)

#     ac.plot_eigenvalues(e,e_br=e_br)
#     ac.plot_subspace_errors(sub_br)
#     ac.plot_eigenvectors(W[:,:1])

#     # make 1d sufficient summary plots
#     y = np.dot(X,W[:,0])
#     ac.sufficient_summary_plot(y,F)

#     # make 2d sufficient summary plots
#     y = np.dot(X,W[:,:2])
#     ac.sufficient_summary_plot(y,F)

#     # load data
#     df = pn.DataFrame.from_csv('data/quad.txt')
#     data = df.values
#     X = data[:,:3]
#     F = data[:,3]
#     dF = data[:,4:]
#     M,m = X.shape

#     e,W,e_br,sub_br = ac.compute_active_subspace(dF,2)

#     ac.plot_eigenvalues(e,e_br=e_br)
#     ac.plot_subspace_errors(sub_br)
#     ac.plot_eigenvectors(W[:,:1])

#     # make 1d sufficient summary plots
#     y = np.dot(X,W[:,0])
#     ac.sufficient_summary_plot(y,F)

#     # make 2d sufficient summary plots
#     y = np.dot(X,W[:,:2])
#     ac.sufficient_summary_plot(y,F)

#     return 0

# def test_quick_check():
#     # load data
#     df = pn.DataFrame.from_csv('data/quad_nograd.txt')
#     data = df.values
#     X = data[:,0:3]
#     F = data[:,3]
#     w = ac.quick_check(X,F)
#     return 0

# def write_quad_csv():
#     X = np.random.uniform(-1.0,1.0,(100,3))
#     D = np.zeros((100,4))
#     for i in range(100):
#         x = X[i,:]
#         f = quad_fun_nograd(x.T)
#         D[i,:3] = x.T
#         D[i,3] = f

#     labels = ['p1','p2','p3','output']
#     df = pn.DataFrame(data=D,columns=labels)
#     df.to_csv('data/quad_nograd.txt')

#     X = np.random.uniform(-1.0,1.0,(100,3))
#     D = np.zeros((100,7))
#     for i in range(100):
#         x = X[i,:]
#         f,df = quad_fun(x.T)
#         D[i,:3] = x.T
#         D[i,3] = f
#         D[i,4:] = df.T

#     labels = ['p1','p2','p3','output','d1','d2','d3']
#     df = pn.DataFrame(data=D,columns=labels)
#     df.to_csv('data/quad.txt')

# def test_gauss_design(fun):
#     M,m = 300,3
#     X = np.random.uniform(-1.0,1.0,(M,m))
#     F,dF = ac.sample_function(X,fun,dflag=True)

#     e,W,e_br,sub_br = ac.compute_active_subspace(dF,2)

#     X,ind,y = ac.response_surface_design(W,2,[3,3],5)

#     return 0

# def test_fun(X):
#     return np.sum(-np.cos(np.pi*X)+X,axis=1)

# def test_variable_map():
#     m,n = 5,2
#     M,NMC = 20,3
#     W = np.linalg.qr(np.random.normal(size=(m,m)))[0]
#     bflag = 0

#     vm = VariableMap(W,n,bflag)
#     if bflag:
#         X = np.random.uniform(-1.0,1.0,size=(M,m))
#         Y,Z = vm.forward(X)
#         X0,ind = vm.inverse(Y,NMC)
#     else:
#         X = np.random.normal(size=(M,m))
#         Y,Z = vm.forward(X)
#         X0,ind = vm.inverse(Y,NMC)

#     X = np.random.uniform(-1.0,1.0,size=(M,m))
#     f = test_fun(X)
#     ovm = OptVariableMap(W,n,X,f,bflag)
#     if bflag:
#         X = np.random.uniform(-1.0,1.0,size=(M,m))
#         Y,Z = ovm.forward(X)
#         X0 = ovm.inverse(Y)
#     else:
#         X = np.random.normal(size=(M,m))
#         Y,Z = ovm.forward(X)
#         X0 = ovm.inverse(Y)

# def test_zonotopes():
#     m,n = 10,2
#     W1 = np.linalg.qr(np.random.normal(size=(m,n)))[0]
#     yzv = zn.zonotope_vertices(W1)
#     N = 20
#     Y,res = zn.maximin_design(yzv,N)

#     plt.close('all')
#     plt.figure()
#     plt.plot(Y[:,0],Y[:,1],'k.')
#     plt.axes().set_aspect('equal')

#     V = sp.Voronoi(Y)
#     sp.voronoi_plot_2d(V)
#     plt.axes().set_aspect('equal')

#     T = sp.Delaunay(Y)
#     C = []
#     for t in T.simplices:
#         C.append(np.mean(T.points[t],axis=0))
#     centroids = np.array(C)
#     sp.delaunay_plot_2d(T)
#     plt.plot(centroids[:,0],centroids[:,1],'ro')
#     plt.axes().set_aspect('equal')

#     plt.show()

# def test_gq():
#     x,w = gq.gauss_hermite([4,3,2])
#     print x
#     print w

# def test_normalizers():
#     lb = np.array([-2.0,2.0])
#     ub = np.array([-1.0,3.0])
#     bn = au.BoundedNormalizer(lb,ub)
#     x = np.random.uniform(-1.0,1.0,size=(10,2))
#     y = bn.unnormalize(x.copy())
#     z = bn.normalize(y)
#     print np.linalg.norm(x-z)

#     C = np.array([[1.0,0.1],[0.1,1.0]])
#     mu = np.array([2.0,5.0])
#     un = au.UnboundedNormalizer(mu,C)
#     x = np.random.normal(size=(10,2))
#     y = un.unnormalize(x.copy())
#     z = un.normalize(y)
#     print np.linalg.norm(x-z)

# def test_variable_maps():
#     M,m,n,NMC = 10,4,2,5
#     W = np.linalg.qr(np.random.normal(size=(m,m)))[0]
#     vm = VariableMap(W,n,bflag=1)
#     X = np.random.uniform(-1.0,1.0,size=(M,m))
#     Y = vm.forward(X)[0]
#     XX = vm.inverse(Y,NMC)
#     '''
#     print 'X'
#     print X
#     print 'Y'
#     print Y
#     print 'XX'
#     print XX
#     '''

# def ressurf_test():
#     '''
#     X1,X2 = np.meshgrid(np.linspace(-1.0,1.0,21),np.linspace(-1.0,1.0,21))
#     X = np.hstack((X1.reshape((X1.size,1)),X2.reshape((X2.size,1))))
#     f = np.sin(np.pi*np.sum(X,axis=1))
#     e = np.array([1.0,0.5,0.1])
#     Xstar = np.random.uniform(-1.0,1.0,size=(100,2))
#     fstar,vstar = gaussian_process_regression(X,f,Xstar,e=e,gl=0.0,gu=100.0,N=5)
#     '''
#     '''
#     X = np.linspace(-1.0,1.0,21).reshape((21,1))
#     f = np.sin(np.pi*X)
#     e = np.array([1.0,0.5,0.1])
#     D = np.load('true.npz')
#     Xstar = D['Xstar']
#     gp = GaussianProcess(5)
#     gp.train(X,f,e,gl=0.0,gu=100.0)
#     fstar,dfstar,vstar = gp.predict(Xstar,compvar=True)
#     plt.plot(X,f,'k-',Xstar,fstar,'bx')
#     plt.show()
#     print 'Error: %6.4e,%6.4e' % (np.linalg.norm(fstar-D['fstar']),np.linalg.norm(vstar-D['vstar']))
#     print 'Error: %6.4e' % np.linalg.norm(gp(Xstar)[0]-D['fstar'])
#     '''
#     '''
#     X = np.linspace(-1.0,1.0,21).reshape((21,1))
#     f = np.sin(np.pi*X)
#     e = np.array([1.0,0.5,0.1])
#     D = np.load('true.npz')
#     Xstar = D['Xstar']
#     gp = GaussianProcess(5)
#     gp.train(X,f,gl=0.0,gu=100.0)
#     fstar,dfstar,vstar = gp.predict(Xstar,compvar=True,compgrad=True)
#     h = 1e-9
#     fdfstar = (gp(Xstar+h)[0] - gp(Xstar)[0])/h
#     print 'Grad err: %6.4e' % np.linalg.norm(dfstar-fdfstar)
#     '''
#     X = np.linspace(-1.0,1.0,51).reshape((51,1))
#     f = np.sin(np.pi*X)
#     pr = PolynomialRegression(5)
#     pr.train(X,f)
#     D = np.load('true.npz')
#     Xstar = D['Xstar']
#     fstar,dfstar,vstar = pr.predict(Xstar,compgrad=True,compvar=True)
#     plt.close('all')
#     plt.figure()
#     plt.plot(X,f,'k-',Xstar,fstar,'bx')
#     plt.title('Prediction')
#     plt.figure()
#     plt.plot(Xstar,np.pi*np.cos(np.pi*Xstar),'ro',Xstar,dfstar,'bx')
#     plt.title('Derivative')
#     plt.figure()
#     plt.plot(Xstar,vstar,'bx')
#     plt.title('Variance')
#     plt.show()

# def gurobi_test():
#     m,n = 3,4
#     c = np.zeros(n)
#     A = np.eye(3,4)
#     b = np.ones(m)
#     lb = -np.ones(n)
#     ub = np.ones(n)
#     x = linear_program_eq(c,A,b,lb,ub)
#     print x

#     c = np.zeros(n)
#     Q = np.eye(n)
#     lb = -np.ones(n)
#     ub = np.ones(n)
#     x = quadratic_program_bnd(c,Q,lb,ub)
#     print x

# if __name__ == "__main__":

#     #if not test_load_data():
#     #    print 'Success!'

#     #if not test_interface(quad_fun):
#     #    print 'Success!'

#     #if not test_interface_nograd(quad_fun_nograd):
#     #    print 'Success!'

#     #if not test_quick_check():
#     #    print 'Success!'

#     #if not test_gauss_design(quad_fun):
#     #    print 'Success!'

#     if not test_variable_maps():
#         print 'Success!'


#
