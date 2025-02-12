# SVD-in-NLP

## Core Concept

SVD factorizes a matrix \( X \) into three components:

\[ X = U \Sigma V^T \]

- **\( U \)**: Orthogonal matrix (term-topic relationships).
- **\( \Sigma \)**: Diagonal matrix of singular values (strength of topics).
- **\( V^T \)**: Orthogonal matrix (document-topic relationships).

## Key Applications in NLP

### Latent Semantic Analysis (LSA/LSI)

**Purpose**: Capture hidden semantic relationships between terms and documents.

**Process**: Apply SVD to a term-document matrix (e.g., TF-IDF weighted). Truncate to \( k \) singular values to obtain a low-rank approximation.

**Benefits**:

- Reduces dimensionality, addressing sparsity and noise.
- Mitigates synonymy (different words with similar meanings) and polysemy (same word with multiple meanings) by grouping related terms/documents in latent space.
- Enhances tasks like document clustering, classification, and retrieval.

### Word Embeddings

**Co-occurrence Matrix Factorization**: SVD decomposes word-context matrices (e.g., word-word co-occurrence counts) to produce dense vector representations.

**Example**: Early methods like GloVe (Global Vectors) use matrix factorization, though modern approaches (e.g., Word2Vec) often employ neural networks.

### Information Retrieval

Project queries and documents into a shared latent space for relevance matching, improving search results even with vocabulary mismatch.

### Topic Modeling

Identifies latent topics by interpreting singular vectors as topics. Each document is a combination of these topics, offering a simpler alternative to probabilistic models like LDA.

## Practical Considerations

### Dimensionality Reduction

- Choose \( k \) (number of retained singular values) via explained variance or cross-validation.
- Balances computational efficiency and information retention.

### Preprocessing

- Use TF-IDF instead of raw counts to emphasize discriminative terms.

### Scalability

- Randomized SVD or incremental methods handle large matrices efficiently.

## Limitations

- **Linearity Assumption**: Struggles with non-linear semantic relationships.
- **Orthogonality Constraint**: Topics are forced to be orthogonal, which may not reflect real-world overlaps.
- **Context Ignorance**: Fails to capture word order or context, unlike transformers or RNNs.

## Comparison to Modern Methods

**Strengths**: Computationally efficient, mathematically interpretable.

**Weaknesses**: Outperformed by neural models (e.g., BERT) in contextual tasks but remains useful for baseline analysis or resource-constrained settings.

## Example Workflow

1. Construct a TF-IDF term-document matrix \( X \).
2. Apply SVD: \( X \approx U_k \Sigma_k V_k^T \).
3. Represent documents as \( \Sigma_k V_k^T \) (low-dimensional vectors).
4. Use these vectors for tasks like clustering or retrieval.

## Conclusion

SVD is a versatile tool in NLP for uncovering latent semantics, reducing dimensionality, and improving computational efficiency. While newer deep learning methods have surpassed it in some areas, SVD remains relevant for its simplicity and effectiveness in foundational tasks.




## Working

The singular value decomposition (SVD) of a matrix is a factorization that expresses any \( m \times n \) matrix \( A \) as
\[
A = U \Sigma V^T,
\]
where:

- **\( U \)** is an \( m \times m \) orthogonal matrix (its columns are orthonormal vectors),
- **\( \Sigma \)** is an \( m \times n \) diagonal matrix with nonnegative entries (the singular values) on the diagonal, and
- **\( V \)** is an \( n \times n \) orthogonal matrix.

Below is a step-by-step procedure to compute the SVD of a matrix \( A \):

---

### **Step 1. Compute \( A^T A \)**
- Form the matrix \( A^T A \). This is an \( n \times n \) symmetric and positive semi-definite matrix.
- Because \( A^T A \) is symmetric, it has a full set of real eigenvalues and orthonormal eigenvectors.

---

### **Step 2. Find the Eigenvalues and Eigenvectors of \( A^T A \)**
- **Eigenvalues:** Solve the characteristic equation
  \[
  \det(A^T A - \lambda I) = 0.
  \]
  Let the eigenvalues be \(\lambda_1, \lambda_2, \dots, \lambda_n\). These will be nonnegative (i.e., \(\lambda_i \ge 0\)).

- **Singular Values:** Define the singular values \(\sigma_i\) of \( A \) as the nonnegative square roots of these eigenvalues:
  \[
  \sigma_i = \sqrt{\lambda_i}.
  \]
  It is common practice to order them in descending order:
  \[
  \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r \ge 0,
  \]
  where \( r \) is the rank of \( A \).

- **Eigenvectors:** Let \( v_1, v_2, \dots, v_n \) be the corresponding unit-norm eigenvectors of \( A^T A \). These will form the columns of the matrix \( V \).

---

### **Step 3. Form the Matrix \( V \)**
- Assemble the eigenvectors \( v_i \) into the orthogonal matrix \( V \):
  \[
  V = \begin{bmatrix} v_1 & v_2 & \cdots & v_n \end{bmatrix}.
  \]
- Since the eigenvectors are orthonormal, \( V \) satisfies \( V^T V = I_n \).

---

### **Step 4. Compute the Columns of \( U \)**
For each nonzero singular value \(\sigma_i\) (i.e., for \( i \) such that \(\sigma_i > 0\)):

1. **Calculate:** 
   \[
   u_i = \frac{1}{\sigma_i} A v_i.
   \]
   This gives a unit vector in \( \mathbb{R}^m \).

2. **Assemble:** These vectors \( u_i \) will be the first \( r \) columns of the matrix \( U \).

- **Handling Zero Singular Values:** If \( A \) has zero singular values (i.e., if \( r < m \)), then you need to choose additional orthonormal vectors \( u_{r+1}, \dots, u_m \) to complete the basis for \( \mathbb{R}^m \). These additional vectors can be chosen arbitrarily, as long as they are orthogonal to the computed \( u_i \) and to each other.

---

### **Step 5. Form the Matrix \( \Sigma \)**
- Construct the \( m \times n \) diagonal matrix \( \Sigma \) whose diagonal entries are the singular values:
  \[
  \Sigma = \begin{bmatrix}
  \sigma_1 & 0 & \cdots & 0 \\
  0 & \sigma_2 & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & \cdots & \sigma_p \\
  \end{bmatrix},
  \]
  where \( p = \min(m, n) \). The singular values beyond the rank of \( A \) (if any) will be zero.

---

### **Step 6. Write the Final SVD**
- You now have:
  \[
  A = U \Sigma V^T.
  \]
- Verify that:
  - \( U \) is orthogonal (\( U^T U = I_m \)),
  - \( V \) is orthogonal (\( V^T V = I_n \)), and
  - \( \Sigma \) is diagonal (with possibly additional zero rows or columns, as needed).

---

### **Alternative Approach**
Another common method is to compute the eigen-decomposition of \( AA^T \) instead:
- Find the eigenvalues and eigenvectors of \( AA^T \) to get \( U \).
- Then, use the relation \( v_i = \frac{1}{\sigma_i} A^T u_i \) to compute \( V \).
  
Both approaches will yield the same singular values.

---

### **Example**

Suppose
\[
A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}.
\]

1. **Compute \( A^T A \):**
   \[
   A^T A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}^T \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 10 & 6 \\ 6 & 10 \end{bmatrix}.
   \]

2. **Find Eigenvalues:**
   \[
   \det \left( \begin{bmatrix} 10-\lambda & 6 \\ 6 & 10-\lambda \end{bmatrix} \right) = (10-\lambda)^2 - 36 = 0.
   \]
   Solving:
   \[
   (10-\lambda)^2 = 36 \quad \Rightarrow \quad 10-\lambda = \pm 6,
   \]
   which gives:
   \[
   \lambda_1 = 4 \quad \text{and} \quad \lambda_2 = 16.
   \]
   Thus, the singular values are:
   \[
   \sigma_1 = \sqrt{16} = 4, \quad \sigma_2 = \sqrt{4} = 2.
   \]

3. **Find Eigenvectors:** (Omit the detailed calculation here, but suppose you find normalized eigenvectors \( v_1 \) and \( v_2 \).)

4. **Compute \( U \):** For each \( i \),
   \[
   u_i = \frac{1}{\sigma_i} A v_i.
   \]

5. **Assemble \( U \), \( \Sigma \), and \( V \).**

You can check that \( A = U \Sigma V^T \).

---

### **Summary**

To find the SVD of a matrix \( A \):

1. **Compute \( A^T A \) and find its eigenvalues \(\lambda_i\) and eigenvectors \(v_i\).**
2. **The singular values are \(\sigma_i = \sqrt{\lambda_i}\) (arranged in descending order).**
3. **Form \( V \) from the eigenvectors \( v_i \).**
4. **For each nonzero \(\sigma_i\), compute \( u_i = \frac{1}{\sigma_i} A v_i \) to form the columns of \( U \).**
5. **Complete \( U \) if necessary, and form the diagonal matrix \( \Sigma \) with the singular values.**
6. **Write the decomposition as \( A = U \Sigma V^T \).**

This is the standard method to compute the SVD of a matrix. For larger or more complex matrices, numerical algorithms (like the Golub–Kahan bidiagonalization) are used in practice.
