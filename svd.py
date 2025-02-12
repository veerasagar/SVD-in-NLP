import numpy as np

def manual_svd(A):
    """
    Compute the reduced SVD of matrix A manually.
    A = U * Sigma * V^T
    Returns:
        U: Matrix with left singular vectors as its columns.
        sigma: Array of singular values.
        V: Matrix with right singular vectors as its columns.
    """
    # Step 1: Compute A^T A (which is symmetric and positive semi-definite)
    ATA = np.dot(A.T, A)
    
    # Step 2: Compute eigenvalues and eigenvectors of A^T A.
    # Using 'eigh' because ATA is symmetric.
    eigenvalues, V = np.linalg.eigh(ATA)
    
    # Step 3: Sort the eigenvalues (and corresponding eigenvectors) in descending order.
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]
    
    # Step 4: The singular values are the square roots of the eigenvalues.
    # (We use np.maximum to avoid small negative numbers due to numerical error.)
    sigma = np.sqrt(np.maximum(eigenvalues, 0))
    
    # Step 5: Compute the left singular vectors.
    # For each nonzero singular value, u_i = A * v_i / sigma_i.
    m, n = A.shape
    U = np.zeros((m, len(sigma)))
    for i in range(len(sigma)):
        if sigma[i] > 1e-10:
            U[:, i] = np.dot(A, V[:, i]) / sigma[i]
        else:
            # If sigma is (almost) zero, assign a zero vector.
            # (In a full SVD one would complete U to an orthonormal basis.)
            U[:, i] = np.zeros(m)
    
    return eigenvalues, U, sigma, V

# Example usage with a sample matrix A:
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

eu, U, sigma, V = manual_svd(A)

print("Matrix A:")
print(A)

print("Eigen values")
print(eu)

print("\nLeft singular vectors (U):")
print(U)

print("\nSingular values (sigma):")
print(sigma)

print("\nRight singular vectors (V):")
print(V)