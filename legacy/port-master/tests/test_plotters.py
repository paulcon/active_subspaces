from unittest import TestCase
import unittest
import active_subspaces.utils.plotters as plt
import numpy as np
import matplotlib.pyplot as mplt

class TestPlotters(TestCase):

    def tearDown(self):
        mplt.close("all")

    def test_eigenvalues(self):
        e = np.power(10*np.ones(6),-np.arange(1,7)).reshape((6,1))
        plt.eigenvalues(e)

    def test_eigenvalues_br(self):
        e = np.power(10*np.ones(6),-np.arange(1,7)).reshape((6,1))
        e_br = np.hstack((0.5*e,1.3*e))
        plt.eigenvalues(e, e_br=e_br)

    def test_eigenvalues_br_label(self):
        e = np.power(10*np.ones(6),-np.arange(1,7)).reshape((6,1))
        e_br = np.hstack((0.5*e,1.3*e))
        plt.eigenvalues(e, e_br=e_br, out_label='testing')

    def test_subspace_errors(self):
        sub_br = np.array([[0.01,0.05,0.1],[0.1,0.25,0.5],[0.2,0.4,0.8]])
        plt.subspace_errors(sub_br, out_label='testing')

    def test_eigenvectors_0(self):
        W = np.eye(4)
        plt.eigenvectors(W[:,0].reshape((4,1)))

    def test_eigenvectors_1(self):
        W = np.eye(4)
        plt.eigenvectors(W[:,:2].reshape((4,2)))

    def test_eigenvectors_0_labels(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        plt.eigenvectors(W[:,0].reshape((4,1)), in_labels=in_labels, out_label='data')

    def test_eigenvectors_1_labels(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        plt.eigenvectors(W[:,:2].reshape((4,2)), in_labels=in_labels, out_label='data')

    def test_eigenvectors_2(self):
        W = np.eye(4)
        plt.eigenvectors(W[:,:3].reshape((4,3)))

    def test_eigenvectors_3(self):
        W = np.eye(4)
        plt.eigenvectors(W)

    def test_eigenvectors_2_labels(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        plt.eigenvectors(W[:,:3].reshape((4,3)), in_labels=in_labels, out_label='data')

    def test_eigenvectors_3_labels(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        plt.eigenvectors(W, in_labels=in_labels, out_label='data')

    def test_eigenvectors_0_br(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        W_br = np.array([[0.9,1.0],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1]])
        plt.eigenvectors(W[:,0].reshape((4,1)), W_br=W_br, in_labels=in_labels, out_label='data')

    def test_eigenvectors_1_br(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        W_br = np.array([[0.9,1.0,-0.1,0.1],[-0.1,0.1,0.9,1.0],[-0.1,0.1,-0.1,0.1],[-0.1,0.1,-0.1,0.1]])
        plt.eigenvectors(W[:,:2].reshape((4,2)), W_br=W_br, in_labels=in_labels, out_label='data')

    def test_eigenvectors_2_br(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        W_br = np.array([[0.9,1.0,-0.1,0.1,-0.1,0.1],
                        [-0.1,0.1,0.9,1.0,-0.1,0.1],
                        [-0.1,0.1,-0.1,0.1,0.9,1.0],
                        [-0.1,0.1,-0.1,0.1,-0.1,0.1]])
        plt.eigenvectors(W[:,:3].reshape((4,3)), W_br=W_br, in_labels=in_labels, out_label='data')

    def test_eigenvectors_3_br(self):
        W = np.eye(4)
        in_labels = ['a','b','c','d']
        W_br = np.array([[0.9,1.0,-0.1,0.1,-0.1,0.1,-0.1,0.1],
                        [-0.1,0.1,0.9,1.0,-0.1,0.1,-0.1,0.1],
                        [-0.1,0.1,-0.1,0.1,0.9,1.0,-0.1,0.1],
                        [-0.1,0.1,-0.1,0.1,-0.1,0.1,0.9,1.0]])
        plt.eigenvectors(W, W_br=W_br, in_labels=in_labels, out_label='data')

    def test_ssp1(self):
        y = np.random.uniform(-1.0,1.0,size=(10,1))
        f = np.sin(y)
        plt.sufficient_summary(y, f)

    def test_ssp2(self):
        y = np.random.uniform(-1.0,1.0,size=(20,2))
        f = np.sin(y[:,0])*np.sin(y[:,1])
        plt.sufficient_summary(y, f)
"""
    def test_zonotope_0(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0 = data['df']

        sub = ss.Subspaces()
        sub.compute(df0)
        sub.partition(2)

        np.random.seed(42)
        bavd = dom.BoundedActiveVariableDomain(sub)
        vertices = bavd.vertY
        plt.zonotope_2d_plot(vertices)

    def test_zonotope_1(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0 = data['df']

        sub = ss.Subspaces()
        sub.compute(df0)
        sub.partition(2)

        np.random.seed(42)
        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)
        Y = av_design(bavm, 8, NMC=1)[0]

        vertices = bavd.vertY
        plt.zonotope_2d_plot(vertices, design=Y)

    def test_zonotope_2(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0 = data['df']

        sub = ss.Subspaces()
        sub.compute(df0)
        sub.partition(2)

        np.random.seed(42)
        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)
        Y = av_design(bavm, 8, NMC=1)[0]

        vertices = bavd.vertY

        Xp = np.random.uniform(-1.0,1.0,size=(20, sub.W1.shape[0]))
        Yp = np.dot(Xp, sub.W1)
        fp = np.sum(Yp, axis=1)

        plt.zonotope_2d_plot(vertices, design=Y, y=Yp, f=fp)

    def test_zonotope_3(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0 = data['df']

        sub = ss.Subspaces()
        sub.compute(df0)
        sub.partition(2)

        np.random.seed(42)
        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)
        Y = av_design(bavm, 8, NMC=1)[0]

        vertices = bavd.vertY

        Yp, Yw = av_quadrature_rule(bavm, 8)

        plt.zonotope_2d_plot(vertices, design=Y, y=Yp, f=Yw, out_label='quadrature rule')
"""

if __name__ == '__main__':
    mplt.close('all')
    unittest.main()
