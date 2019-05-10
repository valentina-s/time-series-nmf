import numpy.testing as nt
from scipy.io import loadmat
import ssnmf
import numpy.linalg as LA


import os
# output = loadmat(os.path.join(os.path.dirname(__file__),'output.mat'))
import sys
sys.path.append('..')
import os.path as op
data_path = op.join(ssnmf.__path__[0],'data')

def test_all_zero_one_iteration():
    # testing no sparsity, no smoothnes, 1 iteration
    output = loadmat(op.join(data_path,'output.mat'))
    model = ssnmf.smoothNMF(r=5, max_iter=1, betaW=0, betaH=0)
    model.fit(output['V'], W=output['init_W'], H=output['init_H'])

    print(LA.norm(output['V'] - (model.W@model.H)))

    # testing the initial distance
    nt.assert_almost_equal(LA.norm(output['V'] - (output['init_W']@output['init_H'])),213.432630275)

    # compare cost after 1 iteration
    nt.assert_almost_equal(model.cost, 9516.581524438)




def test_all_zero_200_iterations():
    # testing no sparsity, no smoothness, 200 iterations
    output = loadmat(op.join(data_path,'output.mat'))
    model = ssnmf.smoothNMF(r=5, max_iter=200, betaW=0, betaH=0)
    model.fit(output['V'], W=output['init_W'], H=output['init_H'])
    nt.assert_almost_equal(model.cost[-1],3636.162716116)


def sparse_one_iteration():
    # testing sparsity, 1 iteration
    output = loadmat(op.join(data_path,'output.mat'))
    model = ssnmf.smoothNMF(r=5, max_iter=1, sparsity=1, smoothness=0, betaW=0, betaH=0)
    model.fit(output['V'], W=output['init_W'], H=output['init_H'])
    nt.assert_almost_equal(model.cost[-1],4750.738752595)


def test_smooth_one_iteration():
    # testing smoothness, 1 iteration
    output = loadmat(op.join(data_path,'output.mat'))
    model = ssnmf.smoothNMF(r=5, max_iter=1, sparsity=0, smoothness=1, betaW=0.0, betaH=0.0)
    model.fit(output['V'], W=output['init_W'], H=output['init_H'])

    import numpy.linalg as LA
    print(LA.norm(model.W))
    print(LA.norm(model.H))

    nt.assert_almost_equal(LA.norm(model.W),4.7809,decimal=4)
    nt.assert_almost_equal(LA.norm(model.H),39.6015,decimal=4)
    nt.assert_almost_equal(model.cost[-1],6667.921143908)
    
def test_smooth_and_parse_one_iterations():
    # testing sparsity and smoothness, 1 iteration
    output = loadmat(op.join(data_path,'output.mat'))
    model = ssnmf.smoothNMF(r=5, max_iter=1, sparsity=1, smoothness=1, betaW=0, betaH=0)
    model.fit(output['V'], W=output['init_W'], H=output['init_H'])
    nt.assert_almost_equal(model.cost[-1],6715.167611171)


def test_smooth_and_parse_200_iterations():
    # testing sparsity and smoothness, 200 iterations
    output = loadmat(op.join(data_path,'output.mat'))
    model = ssnmf.smoothNMF(r=5, max_iter=200, sparsity=1, smoothness=1, betaW=0.0, betaH=0.0)
    model.fit(output['V'], W=output['init_W'], H=output['init_H'])
    nt.assert_almost_equal(model.cost[-1],3909.6946, decimal=4)

def test_smooth_and_sparse_200_iterations_betas():
    # testing sparsity and smoothness, 200 iterations
    output = loadmat(op.join(data_path,'output.mat'))
    model = ssnmf.smoothNMF(r=5, max_iter=200, sparsity=1, smoothness=1, betaW=0.1, betaH=0.1)
    model.fit(output['V'], W=output['init_W'], H=output['init_H'])
    nt.assert_almost_equal(model.cost[-1],3893.69665, decimal=4)

