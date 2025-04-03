import torch


def cholesky(A):
    """
    Cholesky decomposition of a positive definite matrix A.

    Runtime: O(n^3)

    Args:
        A (torch.Tensor): The input matrix of shape (n, n).

    Returns:
        L (torch.Tensor): The lower triangular matrix of shape (n, n).
        invertible (bool): Whether the matrix A is invertible.
    """
    n = A.shape[0]
    L = torch.zeros((n, n), device=A.device, dtype=A.dtype)
    for i in range(n):
        for j in range(i+1):
            if i == j:
                L[i, j] = torch.sqrt(A[i, i] - torch.sum(L[i, :i]**2))
                if L[i, j] == 0:
                    return L, False
            else:
                L[i, j] = (A[i, j] - torch.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L, True


def solve(A, b):
    """
    Solve the linear system Ax = b using Cholesky decomposition.

    Runtime: O(n^3)

    Args:
        A (torch.Tensor): The coefficient matrix of shape (n, n).
        b (torch.Tensor): The right-hand side vector of shape (n,).

    Returns:
        x (torch.Tensor): The solution vector of shape (n,).
    """
    n = A.shape[0]
    L, invertible = cholesky(A)
    if not invertible:
        raise ValueError("The matrix A is singular.")
    y = torch.zeros(n, device=A.device, dtype=A.dtype)
    # forward substitution (O(n^2))
    for i in range(n):
        y[i] = (b[i] - torch.dot(L[i, :i], y[:i])) / L[i, i]
    # backward substitution (O(n^2))
    x = torch.zeros(n, device=A.device, dtype=A.dtype)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - torch.dot(L[i, i+1:], x[i+1:])) / L[i, i]
    return x


# Test implementation
if __name__ == '__main__':
    A = torch.tensor([[1, 2, 3], [2, 5, 6], [3, 6, 10]], dtype=torch.float32)
    b = torch.tensor([1, 2, 3], dtype=torch.float32)
    x = solve(A, b)
    # compare with real solution
    torch.allclose(torch.matmul(A, x), b, rtol=1e-3, atol=1e-3)