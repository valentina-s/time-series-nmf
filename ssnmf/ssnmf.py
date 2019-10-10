import numpy as np
from numpy import linalg as LA


class smoothNMF():

    """
    Algorithm for NMF with eucidian norm as objective function and
    L1 constraint on W for sparse paterns and Tikhonov regularization
    for smooth activation coefficients.

    [W, H, objective] = sparse_nmf(v, params)

    Objective function:
    || W * H - V ||² + lambda * || W ||_1 + eta || H T ||² + betaW ||W||²
                                                        + betaH ||H||²


     Inputs:
     ------

     V:  matrix to be factorized
     params: optional parameters
         sparsity: weight for the L1 sparsity penalty (default: 0)

         smoothness: weight for the smoothness constraint.

         max_iter: maximum number of iterations (default: 100)

         conv_eps: threshold for early stopping (default: 0,
                                                 i.e., no early stopping)
         random_seed: set the random seed to the given value
                       (default: 1; if equal to 0, seed is not set)
         init_W:   initial setting for W (default: random;
                                          either init_w or r have to be set)
         r: (K)       # basis functions (default: based on init_w's size;
                                      either init_w or r have to be set)
         init_H:   initial setting for H (default: random)
         init_W:   initial setting for W (default: random)

         gamma1:   constant > 1 for the gradient descend step of W.

         gamma2:   constant > 1 for the gradient descend step of W.

         betaH:   constant. L-2 constraint for H.
         betaW:   constant. L-2 constraint for W.

     Outputs:
     -------

         W: matrix of basis functions
         H: matrix of activations
         objective: objective function values throughout the iterations
         iter_times: time passed until iteration ith

    Note: when smoothness is zero: the betaH and betaW are not used (sort of set to zero)?

    """

# this is the case with no sparsity and smoothness

    def __init__(self, sparsity=0, smoothness=0, early_stopping=0,
    gamma1=1.001, gamma2=1.001, betaH=0, betaW=0, r=5, max_iter=200):

        self.max_iter = max_iter
        self.sparsity = sparsity
        self.smoothness = smoothness
        self.early_stopping = early_stopping
        if sparsity == 0 and smoothness == 0:
            self.gamma1 = gamma1/2.
            self.gamma2 = gamma2/2.
        else:
            self.gamma1 = gamma1
            self.gamma2 = gamma2
        self.betaH = betaH
        self.betaW = betaW
        self.r = r


    def fit(self, V, W=None, H=None):

        [W, H, obj] = smooth_nmf(V, sparsity=self.sparsity, smoothness=self.smoothness,
            early_stopping=self.early_stopping, gamma1=self.gamma1, gamma2=self.gamma2,
            betaH=self.betaH, betaW=self.betaW, r=self.r, max_iter=self.max_iter, W=W, H=H)

        self.W = W
        self.H = H
        self.cost = obj


def _update_H(V, W, H, gamma2, sparsity, smoothness, betaH, TTp, TTp_norm):

    if smoothness == 0 and sparsity == 0:
        # standard NMF update
        d = gamma2 * 2 * (LA.norm(W @ W.T, 'fro'))
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - V))
        # proximity operator
        H_new = np.maximum(H_new, 0)

    elif sparsity > 0 and smoothness == 0:
        # sparse NMF update

        d = gamma2 * 2 * (LA.norm(W@W.T) + betaH)
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - V) + betaH * H)
        # proximity operator
        H_new[H_new < 0] = 0

    elif sparsity == 0 and smoothness > 0:
        # smooth NMF update
        d = gamma2 * 2 * (LA.norm(W@W.T) + smoothness * TTp_norm)
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - V) + smoothness * (H @ TTp))
        # proximity operator
        H_new = np.maximum(H_new, 0)

    elif sparsity > 0 and smoothness > 0:
        # sparse and smooth NMF update

        d = gamma2 * 2 * (LA.norm(W@W.T) + smoothness * TTp_norm + betaH)
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - V) + smoothness * (H @ TTp) + betaH * H)
        # proximity operator
        H_new = np.maximum(H_new, 0)

    return(H_new)


def _update_W(V, W, H, gamma1, sparsity, smoothness, betaW):

    if sparsity == 0 and smoothness == 0:
        # standard NMF update

        c = gamma1 * 2 * LA.norm(H @ H.T, 'fro')
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - V) @ H.T)

        # proximity operator
        W_new = np.maximum(W_new, 0)

    elif sparsity > 0 and smoothness == 0:
        # sparse NMF update

        c = gamma1 * 2 * LA.norm(H @ H.T)
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - V) @ H.T)
        # proximity operator
        W_new = np.maximum(W_new - 2 * sparsity / c, 0)

    elif sparsity == 0 and smoothness > 0:
        # smooth NMF update

        c = gamma1 * 2 * (LA.norm(H@H.T, 'fro') + betaW)
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - V) @ H.T + betaW * W)
        # proximity operator
        W_new = np.maximum(W_new, 0)

    elif sparsity > 0 and smoothness > 0:
        # smooth and sparse NMF update

        c = gamma1 * 2 * (LA.norm(H @ H.T) + betaW)
        # gradient descend
        z1 = W - (1 / c) * 2 * ((W @ H - V) @ H.T + betaW * W)
        # proximity operator
        W_new = np.maximum(z1 - 2 * sparsity / c, 0)

    return(W_new)


def _initialize(V, W, H, r):
    if W is None:
        W = np.random.uniform(0, 1, size=(V.shape[0], r))
    if H is None:
        H = np.random.uniform(0, 1, size=(r, V.shape[1]))
    return(W, H)


def _objective_function(V, W, H, sparsity, smoothness, betaW, betaH, T=None):
    objective = LA.norm(W @ H - V, 'fro')**2 + \
        smoothness * LA.norm(H @ T, 'fro')**2 +\
        sparsity * np.sum(np.sum(np.abs(W)))\
        + betaW * LA.norm(W, 'fro') + betaH * LA.norm(H, 'fro')

    return(objective)


def smooth_nmf(V, W, H, sparsity=0, smoothness=0, early_stopping=0,
    gamma1=1.001, gamma2=1.001, betaH=0.1, betaW=0.1, r=5, max_iter=100, TTp=None, TTp_norm=None):

    W, H = _initialize(V, W, H, r)

    obj = []  # list storing the objective function value

    if smoothness > 0:
        # Tikhonov regularization matrix
        T = (np.eye(V.shape[1]) - np.diag(np.ones((V.shape[1]-1,)), -1))[:, :-1]
        TTp = T@T.T
        TTp_norm = LA.norm(TTp)
    else:
        # setting to zero: will not use it since smoothness will be zero anyway
        T = np.zeros((V.shape[1], V.shape[1]))

    for it in range(max_iter):

        W = _update_W(V, W, H, gamma1=gamma1, sparsity=sparsity, smoothness=smoothness, betaW=betaW)
        H = _update_H(V, W, H, gamma2=gamma2, sparsity=sparsity, smoothness=smoothness, betaH=betaH, TTp=TTp, TTp_norm=TTp_norm)

        obj.append(_objective_function(V, W, H, sparsity=sparsity, smoothness=smoothness, betaW=betaW, betaH=betaH, T=T))

    return(W, H, obj)
