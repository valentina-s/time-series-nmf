import numpy as np
from numpy import linalg as LA
import numbers
from operator import eq
import os
import math



class smoothNMF():

    """
    Model for NMF with Frobeneous norm as objective function and
    L1 constraint on W for sparse paterns and Tikhonov regularization
    for smooth activation coefficients.

    Objective function:
    || W * H - X ||² + lambda * || W ||_1 + eta || H T ||² + betaW ||W||²
                                                        + betaH ||H||²

    Attributes
    ----------

    sparsity: weight for the L1 sparsity penalty (default: 0)

    smoothness: weight for the smoothness constraint.

    n_components: (K)       # basis functions (default: based on init_w's size;
                                          either init_w or r have to be set)

    gamma1:   constant > 1 for the gradient descend step of W.

    gamma2:   constant > 1 for the gradient descend step of W.

    betaH:   constant. L-2 constraint for H.

    betaW:   constant. L-2 constraint for W.

    W: array of basis vectors, n_features x n_components

    H: array of activations, n_components x n_observations

    cost: objective function values throughout the iterations

    Note: when smoothness is zero: the betaH and betaW are not used (sort of set to zero)?

    """

# this is the case with no sparsity and smoothness

    def __init__(self, sparsity=0, smoothness=0, early_stopping=0,
    gamma1=1.001, gamma2=1.001, betaH=0, betaW=0, n_components=None, max_iter=200):

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
        self.n_components = n_components


    def fit(self, X, W=None, H=None, init=None,
            checkpoint_idx=None, checkpoint_dir=None, checkpoint_file=None, random_state=None):
        """
            Perform the decomposition with the PALM method.

            Inputs
            ------

            X:    the array to be decomposed in a format n_features x n_observations.
            W:    initial estimate for W
            H:    initial estimate for H
            init: select the type of ininitialization
                  'custom' - use the provided W and H
                  'random' - uniform distribution on [0,1]
                  'nndsvd' - nonnegative double svd (default)
                  'nndsvda'- nonnegative double svd where zeros filled with average
                  'nndsvdar' - nonnegative double svd where zeros fillded with small random values
            checkpoint_idx - list of iteration indeces for which to save the decompositions
                  the format is a dictionary where the decompositions can be accessed by the iteration index,
                  each decomposition is a dictionary with H and W keys
                  (note that 0 in the checkpoint_idx corresponds to the initial condition,
                   and checkpoint_idx=range(max_iter+1) will store the decomposition of all updates
                   including the final result)
            checkpoint_dir - the dir where to store the checkpoint file,
                              by default it gets stored in the execution directory
            checkpoint_file - filename of the checkpoint file,
                              by default it is 'chkpt-DATE-TIMESTAMP.db'

        """

        [W, H, obj] = smooth_nmf(X, sparsity=self.sparsity,
                                    smoothness=self.smoothness,
                                    early_stopping=self.early_stopping,
                                    gamma1=self.gamma1,
                                    gamma2=self.gamma2,
                                    betaH=self.betaH,
                                    betaW=self.betaW,
                                    n_components=self.n_components,
                                    max_iter=self.max_iter,
                                    W=W,
                                    H=H,
                                    init=init,
                                    checkpoint_idx=checkpoint_idx,
                                    checkpoint_dir=checkpoint_dir,
                                    checkpoint_file=checkpoint_file,
                                    random_state=random_state)

        self.W = W
        self.H = H
        self.cost = obj

    def connectivity(self, H=None):
        """
        Compute the connectivity matrix for the samples based on their mixture coefficients.

        The connectivity matrix C is a symmetric matrix which shows the shared membership of the samples: entry C_ij is 1 iff sample i and
        sample j belong to the same cluster, 0 otherwise. Sample assignment is determined by its largest metagene expression value.

        Return connectivity matrix.

        :param idx: Used in the multiple NMF model. In factorizations following
            standard NMF model or nonsmooth NMF model ``idx`` is always None.
        :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively)
        """
        if H is None:
            H = self.H
        idx = np.argmax(H, axis=0)
        mat1 = np.tile(idx, (H.shape[1], 1))
        mat2 = np.tile(np.reshape(idx.T,(len(idx),1)), (1, H.shape[1]))
        conn = eq(np.mat(mat1), np.mat(mat2))
        return np.mat(conn, dtype='d')



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

        #c = gamma1 * 2 * LA.norm(H @ H.T, 'fro')
        c = gamma1*2*np.sqrt(np.trace((H.T @ H)@(H.T @ H)))
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - X) @ H.T)

        # proximity operator
        W_new = np.maximum(W_new, 0)

    elif sparsity > 0 and smoothness == 0:
        # sparse NMF update

        # c = gamma1 * 2 * LA.norm(H @ H.T)
        c = gamma1*2*np.sqrt(np.trace((H.T @ H)@(H.T @ H)))
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - X) @ H.T)
        # proximity operator
        W_new = np.maximum(W_new - 2 * sparsity / c, 0)

    elif sparsity == 0 and smoothness > 0:
        # smooth NMF update

        #c = gamma1 * 2 * (LA.norm(H@H.T, 'fro') + betaW)
        c = gamma1*2*(np.sqrt(np.trace((H.T @ H)@(H.T @ H)))+betaW)
        # gradient descend
        W_new = W - (1 / c) * 2 * ((W @ H - X) @ H.T + betaW * W)
        # proximity operator
        W_new = np.maximum(W_new, 0)

    elif sparsity > 0 and smoothness > 0:
        # smooth and sparse NMF update

        # c = gamma1 * 2 * (LA.norm(H @ H.T) + betaW)
        c = gamma1*2*(np.sqrt(np.trace((H.T @ H)@(H.T @ H)))+betaW)
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


def _initialize(X, W, H, n_components, init=None, eps=1e-6, random_state=None):
    #if W is None:
    #    W = np.random.uniform(0, 1, size=(X.shape[0], r))
    #if H is None:
    #    H = np.random.uniform(0, 1, size=(r, X.shape[1]))
    #return(W, H)

    if init == 'random':
        rng = check_random_state(random_state)
        W = rng.uniform(0, 1, size=(X.shape[0], n_components))
        H = rng.uniform(0, 1, size=(n_components, X.shape[1]))

    elif init == 'random_vcol':
        """
        Return initialized basis and mixture matrix. Initialized matrices are of
        the same type as passed target matrix.

        :param V: Target matrix, the matrix for MF method to estimate.
        :type V: One of the :class:`scipy.sparse` sparse matrices types or or
                :class:`numpy.matrix`
        :param rank: Factorization rank.
        :type rank: `int`
        :param options: Specify the algorithm and model specific options (e.g. initialization of
                extra matrix factor, seeding parameters).

                Option ``p_c`` represents the number of columns of target matrix used
                to average the column of basis matrix. Default value for ``p_c`` is
                1/5 * (target.shape[1]).
                Option ``p_r`` represent the number of rows of target matrix used to
                average the row of basis matrix. Default value for ``p_r`` is 1/5 * (target.shape[0]).
        :type options: `dict`
        """

        # p_c = options.get('p_c', int(ceil(1. / 5 * X.shape[1])))
        # p_r = options.get('p_r', int(ceil(1. / 5 * X.shape[0])))

        p_c = int(math.ceil(1. / 5 * X.shape[1]))
        p_r = int(math.ceil(1. / 5 * X.shape[0]))
        prng = np.random.RandomState()

        W = np.zeros((X.shape[0], n_components))
        H = np.zeros((n_components, X.shape[1]))
        for i in range(n_components):
            W[:, i] = X[:, prng.randint(
                low=0, high=X.shape[1], size=p_c)].mean(axis=1)
            H[i, :] = X[
                prng.randint(low=0, high=X.shape[0], size=p_r), :].mean(axis=0)
    else:
        # NNDSVD initialization
        # code from
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/_nmf.py
        U, S, V = LA.svd(X, full_matrices=False)
        U = U[:, :n_components]
        V = V[:n_components, :]
        W, H = np.zeros(U.shape), np.zeros(V.shape)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        for j in range(1, n_components):
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


def smooth_nmf(X, W, H, n_components=None, init=None, sparsity=0, smoothness=0, early_stopping=0,
    gamma1=1.001, gamma2=1.001, betaH=0.1, betaW=0.1, max_iter=100,
    TTp=None, TTp_norm=None, checkpoint_idx=None, checkpoint_dir=None, checkpoint_file=None, random_state=None):

    """

        Palm algorithm for smooth and sparse NMF

        Inputs
        ------

        X:  array to be factorized in n_features x n_observations format

        H:   initial setting for H (default: random)

        W:   initial setting for W (default: random)

        n_components: the rank of the decomposition

        init: initionalization type (default is nndsvd)

        sparsity: sparsity regularization parameter

        smoothness: smoothness regularization parameter

        gamma1:   constant > 1 for the gradient descend step of W.

        gamma2:   constant > 1 for the gradient descend step of W.

        betaH:   constant. L-2 constraint for H.

        betaW:   constant. L-2 constraint for W.

        max_iter: maximum number of iterations (default: 100)

        conv_eps: threshold for early stopping (default: 0,
                                                     i.e., no early stopping)
        random_seed: set the random seed to the given value
                           (default: 1; if equal to 0, seed is not set)

        checkpoint_idx:   iteration to save

        checkpoint_dir:   directory to save checkpoint files

        checkpoint_file:   filename for saving checkpoints


        Return
        ------
        W - final W
        H - final H
        obj - list of the cost at each iteration

    """


    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    if init == 'custom':
        pass # need to add checks for None, or option to set one
    else:
        W, H = _initialize(X, W, H, n_components, init=init, eps=1e-6, random_state=random_state)

    obj = []  # list storing the objective function value

    if smoothness > 0:
        # Tikhonov regularization matrix
        T = (np.eye(X.shape[1]) - np.diag(np.ones((X.shape[1]-1,)), -1))[:, :-1]
        TTp = T@T.T
        TTp_norm = LA.norm(TTp)
    else:
        # setting to zero: will not use it since smoothness will be zero anyway
        T = np.zeros((X.shape[1], X.shape[1]))

    # open checkpoint file
    from datetime import datetime
    if checkpoint_idx is not None:
        import shelve
        if checkpoint_dir is not None:
            if os.path.isdir(checkpoint_dir):
                if checkpoint_file is not None:
                    fpath = os.path.join(checkpoint_dir, checkpoint_file)
                else:
                    fpath = os.path.join(checkpoint_dir, 'chkpt-'+('-'.join(str(datetime.now()).split(' '))))

                if os.path.exists(fpath+'.db'):
                    os.remove(fpath+'.db')
                chkpt_file = shelve.open(fpath)

                if 0 in checkpoint_idx:
                    # storing initial conditions
                    chkpt_data = {'H':H,'W':W}
                    chkpt_file[str(0)] = chkpt_data
            else:
                raise ValueError('Please provide an existing directory.')
        else:
            chkpt_file = shelve.open('chkpt-'+'-'.join(str(datetime.now()).split(' ')))
        #chkpt_file = shelve.open('chkpt')

    for it in range(max_iter):

        W = _update_W(X, W, H, gamma1=gamma1, sparsity=sparsity, smoothness=smoothness, betaW=betaW)
        H = _update_H(X, W, H, gamma2=gamma2, sparsity=sparsity, smoothness=smoothness, betaH=betaH, TTp=TTp, TTp_norm=TTp_norm)

        obj.append(_objective_function(X, W, H, sparsity=sparsity, smoothness=smoothness, betaW=betaW, betaH=betaH, T=T))

        #saving the model checkpoints
        if checkpoint_idx is not None:
            if (it+1) in checkpoint_idx:
                chkpt_data = {'H':H,'W':W}
                chkpt_file[str(it+1)] = chkpt_data
                #pickle.dump(chkpt_data, chkpt_file)

    if checkpoint_idx is not None:
        chkpt_file.close()



    return(W, H, obj)
