import numpy as np
import scipy.sparse as sparse

def generate_asset_data(n, m, T, mean = 0, var = 1, same_or_different = 'same'):
	'''
	Inputs:
	n - dimension of problem (i.e., number of assets)
	m - number of factors
	T - number of time periods
	mean - mean of distribution
	var - variance of distribution
	same_or_different - 'same' gives Fs and Ds that are the same. 'different' does it differently. 

	Outputs:
	mus - list of mean vectors
	Fs - list of factor loading matrices
	Rs - list of diagonal matrices in factor model Sigma = FF' + R
	Ds - Matrices of transaction costs
	'''
	np.random.seed(0)

	mus = dict()
	Fs = dict()
	Rs = dict()
	Ds = dict()

	s = np.random.rand(n)

	#last element of mu should be 0
	mu = np.sqrt(var)*np.random.randn(n,1) + mean
	mu[-1] = 0
	F = np.sqrt(var)*np.random.randn(n,m) + mean
	R = np.sqrt(var) #R = stddev * I, but we really just need stddev to do computations
	
	#Ds must have zeros at last row/col
	D = np.sqrt(var)*np.random.rand(n) + mean
	D[-1] = 0


	for t in range(T):
		if same_or_different == 'same':
			mus[t] = mu
			Fs[t] = F
			Rs[t] = R
			Ds[t] = D
		else:
			mus[t] = np.sqrt(var)*np.random.randn(n, 1) + mean
			Fs[t] = np.sqrt(var)*np.random.randn(n,m) + mean
			Rs[t] = np.sqrt(var)
			Ds[t] = np.sqrt(var)*np.random.rand(n) + mean

	return mus, Fs, Rs, Ds, s



