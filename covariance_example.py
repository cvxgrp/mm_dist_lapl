import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.sparse as sparse
from scipy import io
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

def make_elementwise_laplacian_matrix(rows, cols):
	'''
	This makes the laplacian matrix for a grid of points.
	Input: 
		Number of rows (rows), number of columns (cols)
	Output: 
		Laplacian matrix corresponding to grid graph with rows rows and cols columns.
	'''
	n = rows*cols
	A = np.zeros((n,n))
	for r in range(rows):
	    for c in range(cols):
	        idx = r*cols + c
	        # Two inner diagonals
	        if c > 0: A[idx-1,idx] = A[idx,idx-1] = 1
	        # Two outer diagonals
	        if r > 0: A[idx-cols,idx] = A[idx,idx-cols] = 1

	D = np.diag(sum(A))
	return D-A

'''
We load the actual covariances into the script, and grab the dimensions from this.
'''
SS = np.loadtxt(open("Data/largeCov_225_n=30.csv", "rb"), delimiter=",")
n = 30
create_L_matrix = False #Flag this as true if you need to make a new L matrix; flag as false if you already have saved the L matrix.
p1 = int(np.shape(SS)[0]/n) #get size of axis 1 of the grid
p2 = int(np.shape(SS)[1]/n) #get size of axis 2 of the grid
p = p1*p2
num_data_points = int(n//1.5) #number of data points per node is only 2/3 of the number of dimensions for each covariance matrix.


kappa = 8e-2 #local regularization parameter. kept constant among all lambdas in the regularization path.

'''
We will now load the actual covariance matrices into a dictionary S_actual.
'''
S_actual = dict()
count = 0
for i in range(p1):
	for j in range(p2):
		index = p1*count + j
		S_actual[index] = SS[i*n:(i+1)*n, j*n:(j+1)*n]
	count += 1

'''
We will generate the empirical covariances for each i by taking num_data_points 
samples and pulling data from a multivariate normal distribution with mean zero 
and covariance S_actual[i]. ALL_DATA_POINTS stores all of the samples, which will
be used in computing the covariance of all the samples.
'''
np.random.seed(1)
ALL_DATA_POINTS = np.zeros((n,1))

S = dict()
for i in range(p):
	rand_data = np.random.multivariate_normal(np.zeros(n), S_actual[i], size=num_data_points).T
	ALL_DATA_POINTS = np.concatenate([ALL_DATA_POINTS, rand_data], axis=1)
	S[i] = np.cov(rand_data)

ALL_DATA_POINTS = ALL_DATA_POINTS[:, 1:]

'''Calculate MSE for lambda = 0'''
MSE_0 = 0
for i in range(p):
	MSE_0 += np.linalg.norm(np.linalg.inv(S_actual[i]) - np.linalg.inv(S[i] + kappa*np.eye(n)), 'fro')**2
MSE_0 /= (n**2 * p)
print('RMSE for LAMBDA = 0 is %s.' % np.sqrt(MSE_0))

'''Calculate MSE for lambda -> infinity'''
MSE_infty = 0
for i in range(p):
	MSE_infty += np.linalg.norm(np.linalg.inv(S_actual[i]) - np.linalg.inv(np.cov(ALL_DATA_POINTS)), 'fro')**2
MSE_infty /= (n**2 * p)
print('RMSE for LAMBDA -> Infinity is %s.' % np.sqrt(MSE_infty))

'''
In this section, for the particular dimensions of the problem, we either compute 
the Laplacian for lambda=1 (and scale in the regpath); or, we load the Laplacian
if it is already available.
'''
filename_L = 'Data/L_Matrix_p_%s_n_%s' % (p, n)
if create_L_matrix:
	print('Creating L')
	L_orig = sparse.lil_matrix((n*p, n*p))
	L_elems = make_elementwise_laplacian_matrix(p1, p2)

	for i in range(p):
		for j in range(p):
			L_orig[i*n:(i+1)*n, j*n:(j+1)*n] = L_elems[i,j] * sparse.eye(n)

	np.save(filename_L, L_orig)
else:
	print('Loading L')
	L_orig = np.load('%s.npy' % filename_L)
	L_orig = L_orig.item()


def Ltheta_covariance(L_elems_scalar, theta):
	'''
	Method for computing L*theta fast.
	Inputs:
		L_elems_scalar: L, but instead of being blocked, its just the scalar value.
		theta: input to MM iteration.
	Outputs:
		H: L * theta.

	'''
	(_, n) = theta.shape
	H = np.zeros(theta.shape)

	for i in range(L_elems_scalar.shape[0]):
		for q in np.nonzero(L_elems_scalar[i,:])[0]:
			H[i*n:(i+1)*n, :] += L_elems_scalar[i,q]*theta[q*n:(q+1)*n, :]

	return H


def update(parameters):
	'''
	Compute closed form update.
	Inputs:
		parameters - List of 6 parameters needed.
			0: index corresponding to which covariance the update is on
			1: kappa, local regularization parameter
			2: HH_k, the indices of H_k that are needed for this update.
			3: TT_prev, the previous value of T that is needed for this update.
			4: S_i, the empirical covariance
			5: alpha_i, the diagonal element of the majorizing quadratic matrix Lhat for this index.

	Outputs:
		idx: index corresponding to which covariance the update is on.
		T_update: the updated value of Theta.
		t_finish_cov: the amount of time it took to compute this update.
	'''

	t_start_cov = time.time()

	idx = parameters[0]
	KAPPA = parameters[1]
	HH_k = parameters[2]
	TT_prev = parameters[3]
	S_i = parameters[4]
	alpha_i = parameters[5]

	D, Q = np.linalg.eigh(S_i + HH_k + KAPPA*np.eye(n) - alpha_i * TT_prev)

	T_update = ( 1/(2*alpha_i) ) * Q.dot( np.diag(-D + np.sqrt(np.square(D) + (4*alpha_i)*np.ones(D.shape))) ).dot( Q.T )

	t_finish_cov = time.time() - t_start_cov

	return (idx, T_update, t_finish_cov)

'''Solve using MM.'''
max_iter = 500 #max number of iters per value of lambda.
abs_tol = 1e-5 #absolute tolerance
rel_tol = 1e-3 #relative tolerance
tot_iters = 0 #counter to count total number of iterations
RMSE_VEC = [] #array that will hold the regpath MSEs
t_reg_path = 0
numProcesses = cpu_count() #number of processes on your computer.
T = np.zeros((n*p, n)) #Theta
T_prev = np.zeros((n*p, n)) #previous value of theta
count_lambd = 0
L_elements = make_elementwise_laplacian_matrix(p1,p2)
L_orig_diag = L_orig.diagonal() #To be multiplied by the appropriate value of lambda.
T_prev_init = np.vstack([np.linalg.inv(S[i] + kappa*np.eye(n)) for i in range(p)])
exponents = np.logspace(-5,4,num=100,base=10)

print('Begin regularization path.')

t_whole_regpath = 0
for lambd in exponents:
	L = lambd * L_orig

	LHAT_diag = 2.5*lambd*L_orig_diag #we define LHAT_ii to be > 2*lambd*L_ii

	LHAT_L = sparse.diags(LHAT_diag) - L #Cache LHAT - L.
	NORM_LHAT_L = sparse.linalg.norm(LHAT_L, 'fro')

	residual = []

	if lambd == exponents[0]:
		T_prev = np.vstack([np.linalg.inv(S[i] + kappa*np.eye(n)) for i in range(p)])

	t_whole_regpath_start = time.time()
	for iters in range(max_iter):
		tot_iters += 1
		'''Compute linear term'''
		H_k = Ltheta_covariance(lambd*L_elements, T_prev)

		'''Zip parameters for use in multiprocessing'''
		zip_parameters = []
		for ii in range(p):
			zip_parameters.append((ii, kappa, H_k[ii*n:(ii+1)*n, :], T_prev[ii*n:(ii+1)*n, :], S[ii], LHAT_diag[ii*n]))

		'''Update in parallel, and make sure each node's optimal estimate is in its correct position'''
		with Pool(processes=numProcesses) as pool:
			OUTPUT = pool.map(update, zip_parameters)
		OUTPUT.sort()

		'''Accumulate results of inverse covariances'''
		T = np.vstack([t[1] for t in OUTPUT])

		'''Test stopping criterion (only if past first iteration)'''
		t_stopping_criterion = time.time()
		if (iters > 0):
			r_k_norm = np.linalg.norm(LHAT_L.dot(T-T_prev), 'fro')
			rel_norm = NORM_LHAT_L + np.linalg.norm(T)
			T_prev = np.copy(T)
			if iters%20 == 0:
				print('Iteration %s: MM residual norm is %s.' % (iters, r_k_norm))
			residual.append(r_k_norm)
			if r_k_norm <= abs_tol + rel_tol*rel_norm:
				print('Lambda = %s. MM algorithm converged at iteration %s.' % (lambd, iters))
				break
			
	t_lambda = time.time() - t_whole_regpath_start
	t_whole_regpath += t_lambda

	'''Calculate MSE for this optimal estimate'''
	MSE_MM = 0
	for i in range(p):
		MSE_MM += np.linalg.norm(np.linalg.inv(S_actual[i]) - T[i*n:(i+1)*n,:], 'fro')**2
	MSE_MM /= (n**2 * p)
	RMSE_VEC.append(np.sqrt(MSE_MM))

	print('Total problem solve time for this value of lambda was %s seconds.' % t_lambda)
	print('MSE for (lambda, kappa) = %s is %s.' % ((lambd, kappa), RMSE_VEC[-1]))

print('Entire regularization path took %s seconds.' % t_whole_regpath)
print('Entire regularization path took %s iterations.' % tot_iters)

if len(exponents) > 1:
	rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	rc('text', usetex=True)

	plt.ioff()
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure()
	reg_path, = plt.semilogx(exponents, RMSE_VEC, label='MSE')
	plt.xlabel(r'$\lambda$')
	plt.ylabel(r'Root-mean-square error')
	plt.savefig('Results/reg_path_%s_%s_%s_%s.png' % (p, n, str(time.strftime('%Y%m%d_%H%M%S',time.gmtime())), kappa))
	plt.close(fig)
