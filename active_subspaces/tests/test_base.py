from unittest import TestCase
import unittest
import active_subspaces.response_surfaces as asm
import active_subspaces.optimizers as aso
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import active_subspaces.base as base
import helper
import numpy as np

class TestBase(TestCase):
    writeData = False
    
    def quad_fun(self, x):
        A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
                    [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
                    [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
        x = x.reshape((3,1))
        return 0.5*np.dot(x.T,np.dot(A,x))
        
    def quad_dfun(self, x):
        A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
                    [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
                    [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
        return np.dot(A,x.reshape((3,1)))
        
    def test_rs_ubnd_int(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_data(X, f, df=df)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()
        if self.writeData:
            np.savez('data/test_base_0_0',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_0_0.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
    def test_rs_bnd_int(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_data(X, f, df=df)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

        if self.writeData:
            np.savez('data/test_base_0_1',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_0_1.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
    def test_rs_ubnd_2d_int(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_data(X, f, df=df, avdim=2)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

        if self.writeData:
            np.savez('data/test_base_0_2',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_0_2.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
    def test_rs_bnd_2d_int(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_data(X, f, df=df, avdim=2)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

        if self.writeData:
            np.savez('data/test_base_0_3',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_0_3.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
    
    def test_rs_diag(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_data(X, f, df=df)
        
        model.diagnostics()
        
    def test_rs_predict(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        modelN = base.ActiveSubspaceReducedModel(3, False)
        modelN.build_from_data(X, f, df=df)
        
        XN = np.random.normal(size=X.shape)
        modelN.predict(XN)
        
        modelU = base.ActiveSubspaceReducedModel(3, True)
        modelU.build_from_data(X, f, df=df)
        
        XU = np.random.uniform(-1.0, 1.0, size=X.shape)
        modelU.predict(XU)
        
    def test_fun_rs_ubnd_int(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_interface(self.quad_fun, avdim=1)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

        if self.writeData:
            np.savez('data/test_base_1_0',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_1_0.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
    def test_fun_rs_bnd_int(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_interface(self.quad_fun, avdim=1)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

        if self.writeData:
            np.savez('data/test_base_1_1',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_1_1.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
    def test_fun_rs_ubnd_2d_int(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_interface(self.quad_fun, avdim=2)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

        if self.writeData:
            np.savez('data/test_base_1_2',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_1_2.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
    def test_fun_rs_bnd_2d_int(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        np.random.seed(43)
        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_interface(self.quad_fun, avdim=2)
        
        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

        if self.writeData:
            np.savez('data/test_base_1_3',avg=avg, prob=prob, pl=pl, pu=pu, xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_base_1_3.npz')
        np.testing.assert_equal(avg, data_test['avg'])
        np.testing.assert_equal(prob, data_test['prob'])
        np.testing.assert_equal(pl, data_test['pl'])
        np.testing.assert_equal(pu, data_test['pu'])
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd avg: {:6.4f}'.format(avg)
        print 'ubnd prob: {:6.4f}, {:6.4f}, {:6.4f}'.format(pl,prob,pu)
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
if __name__ == '__main__':
    unittest.main()