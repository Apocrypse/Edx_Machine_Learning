from __future__ import division
import numpy as np
import sys


X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

def priorMLE(y_train):

    n = y_train.shape[0]
    classes = np.unique(y_train)

    priors = {}
    for c in classes:
        priors[c] = np.sum(y_train == c) / n

    return priors


def likelihoodsMLE(X_train, y_train):

    n, d = X_train.shape
    classes = np.unique(y_train)

    mus = {}
    sigmas = {}
    sigma_dets = {}
    for c in classes:
        c_id = (y_train == c)
        X = X_train[c_id, :]
        mu = np.matrix(np.mean(X, axis=0)).T
        sigma = gaussianCovMLE(X, mu)
        sigma_det = np.linalg.det(sigma)
        mus[c], sigmas[c], sigma_dets[c] = mu, sigma, sigma_det

    return mus, sigmas, sigma_dets


def gaussianCovMLE(X, mu):

    n, d = X.shape

    sigma = 0
    for i in range(n):
        x = np.matrix(X[i, :]).T
        sigma += (x - mu) * (x - mu).T / n

    return sigma


def multiGaussianProb(x, mu, sigma, sigma_det):

    coef = np.sqrt(sigma_det) ** (-1)
    exp = np.exp(- 0.5 * (x - mu).T * sigma.I * (x - mu))

    return coef * exp


def pluginClassifier(X_train, y_train, X_test):

    # calculate prior by MLE
    pis = priorMLE(y_train)
    # calculate parameters of likelihood by MLE
    mus, sigmas, sigma_dets = likelihoodsMLE(X_train, y_train)

    classes = np.unique(y_train)
    c = classes.shape[0]

    n, d = X_test.shape
    prob_pred = np.zeros((n, c))

    for i in range(n):
        x = np.matrix(X_test[i, :]).T

        i_prob = np.zeros(c)
        for j in range(c):
            # calculate Gaussian likelihood
            mu = mus[classes[j]]
            sigma = sigmas[classes[j]]
            sigma_det = sigma_dets[classes[j]]
            likelihood = multiGaussianProb(x, mu, sigma, sigma_det)
            # calculate proportion of posterior
            pi = pis[classes[j]]
            poster = pi * likelihood
            i_prob[j] = poster

        prob_pred[i, :] = i_prob / np.sum(i_prob)

    return prob_pred


# assuming final_outputs is returned from function
final_outputs = pluginClassifier(X_train, y_train, X_test)

# write output to file
np.savetxt("probs_test.csv", final_outputs, delimiter=",")
