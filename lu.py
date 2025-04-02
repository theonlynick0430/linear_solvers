import torch


def lu(A):
    """
    LU decomposition of a matrix A.

    Runtime: O(n^3)

    Args:
        A (torch.Tensor): The input matrix of shape (n, n).

    Returns:
        P (torch.Tensor): The permutation matrix of shape (n, n).
        L (torch.Tensor): The lower triangular matrix of shape (n, n).
        U (torch.Tensor): The upper triangular matrix of shape (n, n).
        invertible (bool): Whether the matrix A is invertible.
    """
    n = A.shape[0]
    P = torch.eye(n, device=A.device, dtype=A.dtype)
    L = torch.eye(n, device=A.device, dtype=A.dtype)
    U = A.clone()
    for i in range(n):
        # find the row with the largest absolute value in column i
        k = torch.argmax(torch.abs(U[i:, i])) + i
        if U[k, i] == 0:
            return P, L, U, False
        # swap rows i and k
        U[[i, k], :] = U[[k, i], :]
        P[[i, k], :] = P[[k, i], :]
        # same as swapping rows and columns in L
        L[[i, k], :i] = L[[k, i], :i]
        # eliminate the entries below the diagonal in column i
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
    return P, L, U, True


def solve(A, b):
    """
    Solve the linear system Ax = b using LU decomposition.

    Runtime: O(n^3)

    Args:
        A (torch.Tensor): The coefficient matrix of shape (n, n).
        b (torch.Tensor): The right-hand side vector of shape (n,).

    Returns:
        x (torch.Tensor): The solution vector of shape (n,).
    """
    n = A.shape[0]
    P, L, U, invertible = lu(A)
    if not invertible:
        raise ValueError("The matrix A is singular.")
    # P_inv = P^T = P
    y = torch.matmul(P, b)
    # forward substitution (O(n^2))
    for i in range(n):
        y[i] -= torch.dot(L[i, :i], y[:i])
    # backward substitution (O(n^2))
    x = torch.zeros(n, device=A.device, dtype=A.dtype)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - torch.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


# Test implementation
if __name__ == '__main__':
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=torch.float32)
    b = torch.tensor([1, 2, 3], dtype=torch.float32)
    x = solve(A, b)
    # compare with real solution
    torch.allclose(torch.matmul(A, x), b, rtol=1e-3, atol=1e-3)