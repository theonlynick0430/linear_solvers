import torch 
from torch.linalg import eigh


def svd(A):
    """
    SVD decomposition of a matrix A.

    Runtime: O(n^3)

    Args:
        A (torch.Tensor): The input matrix of shape (m, n).

    Returns:
        U (torch.Tensor): The left singular vectors of shape (m, r).
        S (torch.Tensor): The singular values of shape (r,).
        V (torch.Tensor): The right singular vectors of shape (n, r).
        r (int): The rank of the matrix A.
    """
    m, n = A.shape
    ATA = torch.matmul(A.T, A)
    eigenvalues, eigenvectors = eigh(ATA) # runtime: O(n^3)
    r = torch.sum(eigenvalues > 1e-10)
    U = torch.zeros((m, r), device=A.device, dtype=A.dtype)
    S = torch.zeros(r, device=A.device, dtype=A.dtype)
    V = torch.zeros((n, r), device=A.device, dtype=A.dtype)
    for i in range(r):
        S[i] = torch.sqrt(eigenvalues[i])
        V[:, i] = eigenvectors[:, i]
        if S[i] != 0:
            U[:, i] = torch.matmul(A, V[:, i]) / S[i]
    return U, S, V, r


def solve(A, b):
    """
    Solve the linear system Ax = b using SVD decomposition.

    Runtime: O(n^3)

    Args:
        A (torch.Tensor): The coefficient matrix of shape (m, n).
        b (torch.Tensor): The right-hand side vector of shape (m,).

    Returns:
        x (torch.Tensor): The solution vector of shape (n,).
    """
    _, n = A.shape
    U, S, V, r = svd(A)
    if r < n:
        raise ValueError("The matrix A is singular.")
    S_inv = torch.diag(1.0 / S)
    x = V @ S_inv @ U.T @ b
    return x


# Test implementation
if __name__ == '__main__':
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=torch.float32)
    b = torch.tensor([1, 2, 3], dtype=torch.float32)
    x = solve(A, b)
    # compare with real solution
    torch.allclose(torch.matmul(A, x), b, rtol=1e-3, atol=1e-3)
    