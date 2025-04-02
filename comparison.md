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

## When to Use Each Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Square, well-conditioned matrices | LU (faster) |
| Ill-conditioned matrices | QR (more stable) |
| Overdetermined systems (least squares) | QR |
| Multiple right-hand sides | Either (both are efficient) |
