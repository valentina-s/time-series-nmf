import numpy as np
from numpy import linalg as LA
import numbers



class smoothNMF():

    """
    Algorithm for NMF with eucidian norm as objective function and
    L1 constraint on W for sparse paterns and Tikhonov regularization
    for smooth activation coefficients.

    [W, H, objective] = sparse_nmf(v, params)

    Objective function:
    || W * H - X ||² + lambda * || W ||_1 + eta || H T ||² + betaW ||W||²
                                                        + betaH ||H||²


     Inputs:
     ------

     X:  matrix to be factorized
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


    def fit(self, X, W=None, H=None, init=None):

        [W, H, obj] = smooth_nmf(X, sparsity=self.sparsity, smoothness=self.smoothness,
            early_stopping=self.early_stopping, gamma1=self.gamma1, gamma2=self.gamma2,
            betaH=self.betaH, betaW=self.betaW, r=self.r, max_iter=self.max_iter, W=W, H=H, init=init)

        self.W = W
        self.H = H
        self.cost = obj


def _update_H(X, W, H, gamma2, sparsity, smoothness, betaH, TTp, TTp_norm):

    if smoothness == 0 and sparsity == 0:
        # standard NMF update
        # d = gamma2 * 2 * (LA.norm(W @ W.T, 'fro'))
        d = gamma2 * 2 * np.sqrt(np.trace((W.T @ W) @ (W.T @ W)))
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - X))
        # proximity operator
        H_new = np.maximum(H_new, 0)

    elif sparsity > 0 and smoothness == 0:
        # sparse NMF update

        # d = gamma2 * 2 * (LA.norm(W@W.T) + betaH)
        d = gamma2 * 2 * (np.sqrt(np.trace((W.T @ W)@(W.T @ W))) + betaH)
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - X) + betaH * H)
        # proximity operator
        H_new[H_new < 0] = 0

    elif sparsity == 0 and smoothness > 0:
        # smooth NMF update
        # d = gamma2 * 2 * (LA.norm(W@W.T) + smoothness * TTp_norm)
        d = gamma2 * 2 * (np.sqrt(np.trace((W.T @ W)@(W.T @ W))) + smoothness * TTp_norm)
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - X) + smoothness * (H @ TTp))
        # proximity operator
        H_new = np.maximum(H_new, 0)

    elif sparsity > 0 and smoothness > 0:
        # sparse and smooth NMF update

        #d = gamma2 * 2 * (LA.norm(W@W.T) + smoothness * TTp_norm + betaH)
        d = gamma2 * 2 * (np.sqrt(np.trace((W.T @ W)@ (W.T @ W))) + smoothness * TTp_norm + betaH)
        # gradient descend
        H_new = H - (1 / d) * 2 * (W.T @ (W @ H - X) + smoothness * (H @ TTp) + betaH * H)
        # proximity operator
        H_new = np.maximum(H_new, 0)

    return(H_new)


def _update_W(X, W, H, gamma1, sparsity, smoothness, betaW):

    if sparsity == 0 and smoothness == 0:
        # standard NMF update

        c = gamma1 * 2 * LA.norm(H @ H.T, 'fro')
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - X) @ H.T)

        # proximity operator
        W_new = np.maximum(W_new, 0)

    elif sparsity > 0 and smoothness == 0:
        # sparse NMF update

        c = gamma1 * 2 * LA.norm(H @ H.T)
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - X) @ H.T)
        # proximity operator
        W_new = np.maximum(W_new - 2 * sparsity / c, 0)

    elif sparsity == 0 and smoothness > 0:
        # smooth NMF update

        c = gamma1 * 2 * (LA.norm(H@H.T, 'fro') + betaW)
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - X) @ H.T + betaW * W)
        # proximity operator
        W_new = np.maximum(W_new, 0)

    elif sparsity > 0 and smoothness > 0:
        # smooth and sparse NMF update

        c = gamma1 * 2 * (LA.norm(H @ H.T) + betaW)
        # gradient descend
        z1 = W - (1 / c) * 2 * ((W @ H - X) @ H.T + betaW * W)
        # proximity operator
        W_new = np.maximum(z1 - 2 * sparsity / c, 0)

    return(W_new)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _initialize(X, W, H, r, init=None, eps=1e-6, random_state=None):
    #if W is None:
    #    W = np.random.uniform(0, 1, size=(X.shape[0], r))
    #if H is None:
    #    H = np.random.uniform(0, 1, size=(r, X.shape[1]))
    #return(W, H)

    if init == 'random':
        W = np.random.uniform(0, 1, size=(X.shape[0], r))
        H = np.random.uniform(0, 1, size=(r, X.shape[1]))

    # NNDSVD initialization
    # code from
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/_nmf.py
    U, S, V = LA.svd(X, full_matrices=False)
    U = U[:, :r]
    V = V[:r, :]
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, r):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = LA.norm(x_p), LA.norm(y_p)
        x_n_nrm, y_n_nrm = LA.norm(x_n), LA.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H


def _objective_function(X, W, H, sparsity, smoothness, betaW, betaH, T=None):
    objective = LA.norm(W @ H - X, 'fro')**2 + \
        smoothness * LA.norm(H @ T, 'fro')**2 +\
        sparsity * np.sum(np.sum(np.abs(W)))\
        + betaW * LA.norm(W, 'fro') + betaH * LA.norm(H, 'fro')

    return(objective)


def smooth_nmf(X, W, H, r=None, init=None, sparsity=0, smoothness=0, early_stopping=0,
    gamma1=1.001, gamma2=1.001, betaH=0.1, betaW=0.1, max_iter=100,
    TTp=None, TTp_norm=None):

    if r == None:
        r = min(X.shape[0], X.shape[1])

    if init == 'custom':
        pass # need to add checks for None, or option to set one
    else:
        W, H = _initialize(X, W, H, r, init=init, eps=1e-6, random_state=None)

    obj = []  # list storing the objective function value

    if smoothness > 0:
        # Tikhonov regularization matrix
        T = (np.eye(X.shape[1]) - np.diag(np.ones((X.shape[1]-1,)), -1))[:, :-1]
        TTp = T@T.T
        TTp_norm = LA.norm(TTp)
    else:
        # setting to zero: will not use it since smoothness will be zero anyway
        T = np.zeros((X.shape[1], X.shape[1]))

    for it in range(max_iter):

        W = _update_W(X, W, H, gamma1=gamma1, sparsity=sparsity, smoothness=smoothness, betaW=betaW)
        H = _update_H(X, W, H, gamma2=gamma2, sparsity=sparsity, smoothness=smoothness, betaH=betaH, TTp=TTp, TTp_norm=TTp_norm)

        obj.append(_objective_function(X, W, H, sparsity=sparsity, smoothness=smoothness, betaW=betaW, betaH=betaH, T=T))

    return(W, H, obj)
