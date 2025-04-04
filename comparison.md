# Matrix Decomposition Algorithms: Advantages

This document outlines the key advantages of different matrix decomposition algorithms for solving linear systems.

## LU Decomposition

- **Efficiency**: Slightly faster than QR decomposition for large square matrices (n×n)
- **Repeated Solves**: Efficient for solving multiple systems with the same coefficient matrix but different right-hand sides
- **Memory Usage**: Requires less memory than QR for square matrices
- **Computational Cost**: O(n³) operations for an n×n matrix

## QR Decomposition

- **Numerical Stability**: More stable than LU decomposition, especially for ill-conditioned matrices
- **Least Squares**: Natural solution for overdetermined systems (tall matrices where m > n)
- **Repeated Solves**: Efficient for solving multiple systems with the same coefficient matrix
- **Orthogonality**: The Q matrix is orthogonal, which preserves geometric properties

## Cholesky Decomposition

- **Applicability**: Only applicable for positive semi-definite (PSD) matrices
- **Efficiency**: Faster than both LU and QR decompositions because it only needs to compute a single triangular matrix (takes advantage of symmetry)
- **Computational Cost**: Approximately half the cost of LU decomposition, roughly O(n³/3) operations
- **Numerical Stability**: Very stable for PSD matrices
- **Memory Usage**: Requires the least memory of all three methods

## Singular Value Decomposition (SVD)

- **Numerical Stability**: Most stable of all decomposition methods, especially for ill-conditioned and rank-deficient matrices
- **Computational Cost**: Slowest of all methods, typically O(mn²) operations for an m×n matrix where m ≥ n
- **Versatility**: Can be applied to any matrix (rectangular or square, full rank or rank-deficient)
- **Pseudoinverse**: Provides a direct way to compute the Moore-Penrose pseudoinverse
- **Applications**: Excellent for dimensionality reduction (PCA), data compression, noise reduction, and solving least squares problems
- **Rank Determination**: Reveals the effective rank of a matrix through its singular values

## When to Use Each Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Square, well-conditioned matrices | LU (faster) |
| Ill-conditioned matrices | QR (more stable) or SVD (most stable) |
| Overdetermined systems (least squares) | QR or SVD |
| Symmetric positive (semi-)definite matrices | Cholesky (fastest) |
| Rank-deficient matrices | SVD |
| Computing pseudoinverse | SVD |
| Dimensionality reduction (PCA) | SVD |
| Multiple right-hand sides | Any (all are efficient) |
