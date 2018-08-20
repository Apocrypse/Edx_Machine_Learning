from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
mu = 0
sigma2 = 0.1
d = 5
iteration = 50

# print(np.isnan(train_data).sum())
# print(np.min(train_data), np.max(train_data))


# Implement function here
def PMF(train_data):

    # initialize matrix
    N1 = int(np.amax(train_data[:, 0]))
    N2 = int(np.amax(train_data[:, 1]))
    L = np.zeros((iteration, 1))
    U_matrices = np.zeros((iteration, N1, d))
    V_matrices = np.zeros((iteration, d, N2))

    # generate v randomly
    V = np.random.normal(mu, 1/lam, (d, N2))

    # build M matrix
    M = np.zeros((N1, N2))
    M_missing = np.ones((N1, N2), dtype=np.int32)
    for rating in train_data:
        r = int(rating[0])
        c = int(rating[1])
        M[r-1, c-1] = rating[2]
        M_missing[r-1, c-1] = 0

    for i in range(iteration):

        V_matrices[i] = V

        # update U
        U = updateU(lam, sigma2, d, M, M_missing, V, N1, N2)
        U_matrices[i] = U

        # calculate L
        L[i] = calL(lam, sigma2, M, M_missing, U, V, N1, N2)

        # update V
        V = updateV(lam, sigma2, d, M, M_missing, U, N1, N2)

    return L, U_matrices, V_matrices


def updateU(lam, sigma2, d, M, M_missing, V, N1, N2):

    part1 = lam * sigma2 * np.eye(d)
    U = np.zeros((N1, d))

    for i in range(N1):
        part2 = np.zeros((d, d))
        part3 = np.zeros((d, 1))

        for j in range(N2):
            if not M_missing[i, j]:
                Vj = V[:,j].reshape(d, 1)
                part2 += np.dot(Vj, Vj.T)
                part3 += M[i, j] * Vj

        U[i,:] = np.dot(np.linalg.inv(part1 + part2), part3).reshape(-1)

    return U


def updateV(lam, sigma2, d, M, M_missing, U, N1, N2):

    part1 = lam * sigma2 * np.eye(d)
    V = np.zeros((d, N2))

    for j in range(N2):
        part2 = np.zeros((d, d))
        part3 = np.zeros((d, 1))

        for i in range(N1):
            if not M_missing[i, j]:
                Ui = U[i,:].reshape(d, 1)
                part2 += np.dot(Ui, Ui.T)
                part3 += M[i, j] * Ui

        V[:,j] = np.dot(np.linalg.inv(part1 + part2), part3).reshape(-1)

    return V


def calL(lam, sigma2, M, M_missing, U, V, N1, N2):

    part1 = 0
    for i in range(N1):
        for j in range(N2):
            if not M_missing[i, j]:
                part1 += (M[i, j] - np.dot(U[i,:], V[:,j].T)) ** 2
    part1 /= 2 * sigma2

    part2 = lam / 2 * (((np.linalg.norm(U, axis=1)) ** 2).sum())

    part3 = lam / 2 * (((np.linalg.norm(V, axis=0)) ** 2).sum())

    return - part1 - part2 - part3


# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9].T, delimiter=",")
np.savetxt("V-25.csv", V_matrices[24].T, delimiter=",")
np.savetxt("V-50.csv", V_matrices[49].T, delimiter=",")
