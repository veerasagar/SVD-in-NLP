import numpy as np

def compute_svd(A):
    """
    Compute Singular Value Decomposition of matrix A.
    :param A: Input matrix
    :return: U, Sigma, V^T matrices
    """
    U, S, Vt = np.linalg.svd(A)
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    np.fill_diagonal(Sigma, S)
    
    return U, Sigma, Vt

# Example usage
A = np.array([[3, 2, 2], [2, 3, -2]])
U, Sigma, Vt = compute_svd(A)

print("Original Matrix:")
print(A)
print("\nU Matrix:")
print(U)
print("\nSigma Matrix:")
print(Sigma)
print("\nV^T Matrix:")
print(Vt)

# Verification
reconstructed_A = U @ Sigma @ Vt
print("\nReconstructed A:")
print(reconstructed_A)
