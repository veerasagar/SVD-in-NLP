import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def svd_eig(A, k=None):
    m, n = A.shape
    if k is None or k > n:
        k = n
    
    if sp.issparse(A):
        AtA = (A.T).dot(A)
    else:
        AtA = np.dot(A.T, A)

    if k < n:
        eigvals, V = eigsh(AtA, k=k, which = 'LM')
    else:
        eigvals, V = np.linalg.eigh(AtA)

    if V.ndim == 1:
        V = V[:, np.newaxis]

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    sigma = np.sqrt(np.maximum(eigvals, 0))

    U = np.zeros((m, k))

    for i in range(k):
        if sigma[i] > 1e-10:
            U[:, i] = A @ V[:, i] / sigma[i]
        else:
            U[:, i] = A @ V[:, i]

    return U, sigma, V