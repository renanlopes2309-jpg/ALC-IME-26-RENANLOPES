import numpy as np

def resolve_lu(A, b):
    """Resolve Ax = b por decomposição LU sem pivoteamento.

    Retorna (L, U, x), nesta ordem.
    Usa numpy apenas para instanciar arrays (array, eye, zeros).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A deve ser uma matriz quadrada.')
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError('b deve ser um vetor com n elementos.')

    # ---------- Decomposição A = LU (eliminação ) ----------
    L = np.eye(n)              # diagonal de L com 1's
    U = np.array(A, dtype=float)  # U começa como cópia de A e vira triangular superior

    for k in range(n):
        if abs(U[k, k]) < 1e-12:
            raise Exception('Pivô nulo encontrado. Utilize uma função alternativa '
                            '(por exemplo, LU com pivoteamento parcial) para resolver o sistema.')
        for i in range(k + 1, n):
            # multiplicador usado para zerar U[i, k]
            multiplicador = U[i, k] / U[k, k]
            # o multiplicador é guardado diretamente em L, abaixo da diagonal
            L[i, k] = multiplicador
            # linha_i <- linha_i - multiplicador * linha_k
            for j in range(k, n):
                U[i, j] = U[i, j] - multiplicador * U[k, j]
            U[i, k] = 0.0

    # ---------- Substituição progressiva: Ly = b ----------
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = b[i] - soma  # L[i, i] = 1

    # ---------- Substituição regressiva: Ux = y ----------
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]

    return L, U, x


if __name__ == '__main__':
    A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
    b = [4, 10, 24]
    L, U, x = resolve_lu(A, b)
    print('L =\n', L)
    print('U =\n', U)
    print('x =', x)
