#!/usr/bin/python3
# -*- encoding: utf-8 -*-


import math
import sys

import numpy as np


def cholesky(A):
    assert A.shape[0] == A.shape[1]
    assert np.array_equal(A, A.T)

    L = np.copy(A)
    n = A.shape[0]

    for k in range(n-1):
        try:
            L[k, k] = math.sqrt(L[k, k])
        except ValueError:
            print('The matrix is not positive definite.')
            sys.exit(1)
        for i in range(k+1, n):
            try:
                L[i, k] /= L[k, k]
            except ZeroDivisionError:
                print('The matrix is not positive definite.')
                sys.exit(1)
        for j in range(k+1, n):
            for i in range(j, n):
                L[i, j] -= L[i, k] * L[j, k]
    try:
        L[n-1, n-1] = math.sqrt(L[n-1, n-1])
    except ValueError:
        print('The matrix is not positive definite.')
        sys.exit(1)
    for i in range(n-1):
        for j in range(i+1, n):
            L[i, j] = 0.0
    return L


def qr_givens(A_orig):
    def _rot_matrix(a, b):
        norm = math.sqrt(a**2 + b**2)
        assert norm != 0
        c = a / norm
        s = -b / norm
        return np.array([[c, -s], [s, c]], dtype=float)

    def _join_bigger_rotation(total_size, rot, i, j):
        ret = np.identity(total_size, dtype=float)
        ret[i, i] = rot[0, 0]
        ret[j, j] = rot[1, 1]
        ret[i, j] = rot[0, 1]
        ret[j, i] = rot[1, 0]
        return ret

    A = np.copy(A_orig)
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    Q = np.identity(n, dtype=float)
    for column_index in range(n):
        for row_index in range(column_index + 1, n):
            if A[row_index, column_index] == 0.0:
                continue
            a = A[column_index, column_index]
            b = A[row_index, column_index]
            curr_q_smaller = _rot_matrix(a, b)
            curr_q = _join_bigger_rotation(n, curr_q_smaller, column_index, row_index)
            Q = np.dot(curr_q, Q)
            A = np.dot(curr_q, A)
    Q = Q.T
    return Q, A


def qr_householder(A_orig):
    def _householder_matrix(x):
        aux = np.zeros(x.shape[0], dtype=float)
        aux[0] = np.linalg.norm(x)
        assert aux[0] != 0.0
        v = x - aux
        return np.identity(x.shape[0]) - 2 * np.outer(v, v) / np.dot(v, v)

    def _join_identity_q(total_size, q):
        assert q.shape[0] == q.shape[1]
        assert total_size >= q.shape[0]
        ret = np.identity(total_size, dtype=float)
        ret[-q.shape[0]:, -q.shape[0]:] = q
        return ret

    A = np.copy(A_orig)
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    Q = np.identity(n, dtype=float)
    for column_index in range(n):
        x = A[column_index:, column_index]
        if np.array_equal(x, np.eye(1, x.shape[0], 0, dtype=float)[0]):
            continue
        curr_q_smaller = _householder_matrix(x)
        curr_q = _join_identity_q(n, curr_q_smaller)
        Q = np.dot(curr_q, Q)
        A = np.dot(curr_q, A)
    Q = Q.T
    return Q, A




if __name__ == '__main__':
    print('CHOLESKY')
    print()
    A = np.array([[1, 0, 0], [0, 2, -2], [0, -2, 3]], dtype=float)
    print('A')
    print(A)
    L = cholesky(A)
    print('L')
    print(L)
    print('np.dot(L, L.T)')
    print(np.dot(L, L.T))

    print()
    print('-'*50)
    print()

    print('QR')
    print()
    A = np.array([[1, 1, 1], [0, 1, -1], [0, 1, 1]], dtype=float)
    print('A')
    print(A)

    Q_numpy, R_numpy = np.linalg.qr(A)
    print('Q_numpy')
    print(Q_numpy)
    print('R_numpy')
    print(R_numpy)
    print('np.dot(Q_numpy, R_numpy)')
    print(np.dot(Q_numpy, R_numpy))
    print()

    Q_givens, R_givens = qr_givens(A)
    print('Q_givens')
    print(Q_givens)
    print('R_givens')
    print(R_givens)
    print('np.dot(Q_givens, R_givens)')
    print(np.dot(Q_givens, R_givens))
    print()

    Q_householder, R_householder = qr_householder(A)
    print('Q_householder')
    print(Q_householder)
    print('R_householder')
    print(R_householder)
    print('np.dot(Q_householder, R_householder)')
    print(np.dot(Q_householder, R_householder))
