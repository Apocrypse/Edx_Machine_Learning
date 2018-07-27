import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import sys


X = np.genfromtxt(sys.argv[1], delimiter = ",")


def KMeans(data):

	cluster_num = 5

	# calculate data size
	n = len(data)
	d = len(data[0])

	# initialize centroids randomly
	centerslist = np.empty((cluster_num, d))
	ind = range(n)
	np.random.shuffle(ind)
	for k in range(cluster_num):
		centerslist[k] = data[ind[k]]

	# iterate
	for i in range(10):

		# calculate distance between centroids and data
		distances = np.array(ssd.cdist(centerslist, data, 'euclidean'))

		# update classification by assigning data to closest centroids
		classes = np.argmin(distances, axis=0)

		# update centroids by calculating average
		centerslist = np.empty((cluster_num, d))
		for k in range(cluster_num):
			data_k = data[classes==k, :]
			centroids_k = np.mean(data_k, axis=0)
			centerslist[k] =  centroids_k

		filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
		np.savetxt(filename, centerslist, delimiter=",")


def EMGMM(data):

	cluster_num = 5

	# calculate data size
	n = len(data)
	d = len(data[0])

	# initialize pis as uniform distribution
	pis = np.ones(cluster_num) / cluster_num

	# initialize mus by randomly selecting data points
	mus = np.empty((cluster_num, d))
	ind = range(n)
	np.random.shuffle(ind)
	for k in range(cluster_num):
		mus[k] = data[ind[k]]

	# initialize sigamas as identity matrix
	sigmas = np.empty((cluster_num, d, d))
	for k in range(cluster_num):
		sigmas[k] = np.eye(d)

    # iterate
	for i in range(10):

		# E-step
		phis = np.empty((n, cluster_num))
		for j in range(n):
			x = np.matrix(data[j]).T
			phi = expectation(x, cluster_num, pis, mus, sigmas)
			phis[j] = phi

		# M-step
		pis, mus, sigmas = maximumLikelihood(data, cluster_num, n, d, phis)
		print(pis)

		filename = "pi-" + str(i+1) + ".csv"
		np.savetxt(filename, pis, delimiter=",")
		filename = "mu-" + str(i+1) + ".csv"
		np.savetxt(filename, mus, delimiter=",")  #this must be done at every iteration

		for k in range(cluster_num):
			filename = "Sigma-" + str(k+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
			np.savetxt(filename, sigmas[k], delimiter=",")


def expectation(x, cluster_num, pis, mus, sigmas):

	phi = np.empty(cluster_num)

	for k in range(cluster_num):
		pi = pis[k]
		mu = np.matrix(mus[k]).T
		sigma = np.matrix(sigmas[k])
		phi[k] = pi * multiGaussian(x, mu, sigma)
	phi = phi / np.sum(phi)

	return phi


def multiGaussian(x, mu, sigma):

    coef = np.sqrt(np.linalg.det(sigma)) ** (-1)
    exp = np.exp(- 0.5 * (x - mu).T * sigma.I * (x - mu))

    return coef * exp


def maximumLikelihood(data, cluster_num, n, d, phis):

	# update pis
	print(phis)
	n_k = np.sum(phis, axis=0)
	pis = n_k / n

	# update mus
	mus = np.empty((cluster_num, d))
	sigmas = np.empty((cluster_num, d, d))
	for k in range(cluster_num):
		mu = np.dot(phis[:, k], data) / n_k[k]
		mus[k] = mu
		mu = np.matrix(mu).T

		# update sigmas
		sigma = np.zeros((d, d))
		for i in range(n):
			x = np.matrix(data[i]).T
			sigma += phis[i, k] * (x - mu) * (x - mu).T
		sigmas[k] = sigma / n_k[k]

	return pis, mus, sigmas


# output
centerslist = KMeans(X)
pi_mu_sigma = EMGMM(X)
