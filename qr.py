import torch


def qr(A):
    """
    QR decomposition of a matrix A.
    
    Runtime: O(mn^2)

    Args:
        A (torch.Tensor): The input matrix of shape (m, n) where m >= n.

    Returns:
        Q (torch.Tensor): The orthogonal matrix of shape (m, n).
        R (torch.Tensor): The upper triangular matrix of shape (n, n).
        invertible (bool): Whether the matrix A is invertible.
    """
    m, n = A.shape
    Q = torch.zeros((m, n), device=A.device, dtype=A.dtype)
    R = torch.zeros((n, n), device=A.device, dtype=A.dtype)
    for i in range(n):
        ui = A[:, i]
        vi = ui.clone()
        for j in range(i):
            vj = Q[:, j]
            R[j, i] = torch.dot(ui, vj)
            vi -= R[j, i] * vj
        R[i, i] = torch.norm(vi)
        if R[i, i] == 0:
            return Q, R, False
        vi /= R[i, i]
        Q[:, i] = vi
    return Q, R, True


def solve(A, b):
    """
    Solve the linear system Ax = b using QR decomposition.
    If m > n, the least squares solution is implicitly returned.

    Runtime: O(mn^2)

    Args:
        A (torch.Tensor): The coefficient matrix of shape (m, n).
        b (torch.Tensor): The right-hand side vector of shape (m,).

    Returns:
        x (torch.Tensor): The solution vector of shape (n,).
    """
    _, n = A.shape
    Q, R, invertible = qr(A)
    if not invertible:
        raise ValueError("The matrix A is singular.")
    y = torch.matmul(Q.T, b)
    x = torch.zeros(n, device=A.device, dtype=A.dtype)
    # backward substitution (O(n^2))
    x[n-1] = y[n-1] / R[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - torch.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x


# Test implementation
if __name__ == '__main__':
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=torch.float32)
    b = torch.tensor([1, 2, 3], dtype=torch.float32)
    x = solve(A, b)
    # compare with real solution
    torch.allclose(torch.matmul(A, x), b, rtol=1e-3, atol=1e-3)