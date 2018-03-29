import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.sparse as sparse
import cvxpy as cvx
import scipy.sparse as sparse
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count

def make_L_finance(Ds, n, T):
	#Make L matrix
	L = sparse.lil_matrix((n*T, n*T))
	for i in range(T):
		for j in range(T):
			#form L
			if i == j:
				if i == 0:
					L[i*n:(i+1)*n, j*n:(j+1)*n] = sparse.diags(Ds[1])
				if i == T-1:
					L[i*n:(i+1)*n, j*n:(j+1)*n] = sparse.diags(Ds[T-1])
				else:
					L[i*n:(i+1)*n, j*n:(j+1)*n] = sparse.diags(Ds[i]+Ds[i+1])
			elif j - i == 1:
				L[i*n:(i+1)*n, j*n:(j+1)*n] = sparse.diags(-Ds[j])
			elif i - j == 1:
				L[i*n:(i+1)*n, j*n:(j+1)*n] = sparse.diags(-Ds[i])
	return L

def risk_adj_return(x, s, Fs, Rs, Ds, mus, n, T, L, gamma=1):
	'''
	Compute objective function to portfolio problem
	'''
	obj = 0
	CASH = np.eye(1, n, n-1).T
	
	for ii in range(T):
		if ii == 0:
			obj += s.T.dot(cvx.neg(x[ii*n:(ii+1)*n]).value) + gamma*(cvx.sum_squares(Fs[ii].T.dot(x[ii*n:(ii+1)*n])).value +\
			 Rs[ii]*cvx.sum_squares(x[ii*n:(ii+1)*n]).value) - np.dot(mus[ii].T,x[ii*n:(ii+1)*n])
			+ Ds[0]*cvx.sum_squares(x[:n]).value - (Ds[0]*CASH).T.dot(x[:n])
		else:
			obj += s.T.dot(cvx.neg(x[ii*n:(ii+1)*n]).value) + gamma*(cvx.sum_squares(Fs[ii].T.dot(x[ii*n:(ii+1)*n])).value +\
			 Rs[ii]*cvx.sum_squares(x[ii*n:(ii+1)*n]).value) - np.dot(mus[ii].T,x[ii*n:(ii+1)*n])
	
	return float(obj + 0.5*cvx.quad_form(x, L).value)

def make_LHAT_matrix(L, scheme='L_diags'): #\hat{L} in our manuscript
	if scheme == 'eig':
		LHAT = max(sparse.linalg.eigs(L)[0]) * sparse.eye(np.shape(L)[0])
	elif scheme == 'L_diags':
		LHAT = 3 * np.max(L.diagonal()) * sparse.eye(np.shape(L)[0])
	return LHAT

def solve_complete(x, obj_no_lap_reg, constraints, L, Ds, SOLVER='OSQP'):
	'''
	Solves normal multi-period portfolio problem
	'''
	(n, p) = x.shape
	x = cvx.reshape(x, (n*p,1))
	lap_reg = cvx.quad_form(x,L)/2

	optval = cvx.Problem(cvx.Minimize(sum(obj_no_lap_reg)+lap_reg), constraints).solve(
		solver=SOLVER,parallel=False,eps_abs=1e-3, eps_rel=1e-3, max_iter=1000
		)
	#if objectives are slightly off, decrease eps_abs and eps_rel. improves accuracy but decreases speed of convergence.

	return x.value, optval

def Lx_finance(Ds, x, p):
	'''
	Fast way of computing h = L*x for the finance case.
	Inputs:
	Ds - Dictionary of transaction cost matrix
	x - Input vector
	p - Number of time periods.

	Output:
	h - Output h = Lx
	'''
	n = np.shape(Ds[0])[0]
	h = np.zeros((n * p, 1))
	for ii in range(p):
		if ii == 0:
			h[ii*n:(ii+1)*n] = Ds[ii+1].dot(x[0:n] - x[n:2*n])
		elif ii == p-1:
			h[ii*n:(ii+1)*n] = Ds[ii].dot(-x[(p-2)*n:(p-1)*n] + x[(p-1)*n:p*n])
		else:
			h[ii*n:(ii+1)*n] = -Ds[ii].dot(x[(ii-1)*n:(ii)*n]) + (Ds[ii]+Ds[ii+1]).dot(x[(ii)*n:(ii+1)*n]) - Ds[ii+1].dot(x[(ii+1)*n:(ii+2)*n])
	return h


def finance_example_regular_solve(problem_data):
	'''
	Solves multi-period portfolio optimization problem by forming one large problem,
	and invoking CVXPY's solve() method.
	'''
	L = problem_data['L']
	n = problem_data['n']
	m = problem_data['m']
	T = problem_data['p']
	gamma = problem_data['gamma']
	tolerance = problem_data['tol']
	mus = problem_data['mus']
	Fs = problem_data['Fs']
	Rs = problem_data['Rs']
	Ds = problem_data['Ds']
	s = problem_data['s']
	SOLVER = problem_data['SOLVER']

	print('Solving problem via CVXPY, no MM.')
	CASH = np.eye(1, n, n-1).T.reshape(n,) #cash vector
	x = cvx.Variable((n,T))
	f = cvx.Variable((m,T))
	objs_normal = 0#Ds[0] * cvx.sum_squares(x[:,0]) - (np.dot(Ds[0],CASH).T * x[:,0])
	constraints_normal = [x[:,0] == CASH, x[:,T-1] == CASH]
	for ii in range(T):
		constraints_normal += [sum(x[:,ii]) == 1, f[:, ii] == Fs[ii].T * x[:,ii]]
		objs_normal += s.T*cvx.neg(x[:,ii]) + gamma*(cvx.sum_squares(f[:,ii]) + Rs[ii]*cvx.sum_squares(x[:,ii])) - (mus[ii].T*x[:,ii])

	#Solve problem with CVXPY
	t_start_reg = time.time()
	(x_opt, optval) = solve_complete(x, objs_normal, constraints_normal, L, Ds, SOLVER)
	t_reg =	time.time() - t_start_reg

	return x_opt, optval, t_reg

def solve_cvx_prob(cvx_prob):
	'''
	cvx_prob is of form (problem #, cvx problem instance, associated opt. variable, solver)
	'''
	#print('SOLVING')
	t_start_subprob = time.time()
	result = cvx_prob[1].solve(solver=cvx_prob[3], parallel=True, warm_start=True, max_iter=2000, eps_abs=1e-9, eps_rel=1e-9)
	time_solve_prob = time.time() - t_start_subprob
	x_i_tilde = cvx_prob[2].value
	return (cvx_prob[0], x_i_tilde, time_solve_prob)

def solve_MPO_MM(problem_data):
	'''
	Solves the multi-period portfolio optimization problem via the MM algorithmic framework
	outlined in the paper.

	Inputs:
	problem_data = dictionary of all of the problem data.

	Output(s):
	x_mm = Optimal solution to multi-period portfolio optimization problem.
	time_solve = Time it takes to solve the overall problem, minus overhead.
	t_iter = Average time to solve each iteration.
	residual = Residuals for each iteration of the algorithm.
	'''

	'''Unwrap all of the data, and instantiate other parameters.'''
	L = problem_data['L']
	n = problem_data['n']
	m = problem_data['m']
	p = problem_data['p']
	gamma = problem_data['gamma']
	tolerance = problem_data['tol']
	max_iter = problem_data['max_iter']
	mus = problem_data['mus']
	Fs = problem_data['Fs']
	Rs = problem_data['Rs']
	Ds = problem_data['Ds']
	s = problem_data['s']
	LHAT = problem_data['LHAT']
	LHAT_max_eigval = problem_data['LHAT_max_eigval']
	SOLVER = problem_data['SOLVER']
	x_prev = problem_data['x_prev']

	CASH = np.eye(1, n, n-1).T.reshape(n,) #cash vector	
	numProcesses = cpu_count()
	LHAT_L = LHAT_max_eigval*sparse.eye(n*p) - L #LHAT - L
	print('Tolerance is %s.' % tolerance)
	time_res = 0
	t_solver = 0
	time_solve = 0
	residual = []

	h_k = cvx.Parameter((n*p,))
	cvxpy_solve=False #When this == False, we use multiprocessing pooling. When this == true, we use cvxpy's .solve().
	Xs = [cvx.Variable(n) for ii in range(p)]
	f = [cvx.Variable(m) for ii in range(p)]
	objectives = []

	def objective(ii):
		full_obj = cvx.Minimize(
	       s.T * cvx.neg(Xs[ii])
	       + gamma * (cvx.sum_squares(f[ii]) + Rs[ii] * cvx.sum_squares(Xs[ii]))
	       - mus[ii].T * Xs[ii]
	       + .5 * LHAT_max_eigval * cvx.sum_squares( Xs[ii] - cvx.reshape(x_prev[ii*n:(ii+1)*n], (n,)) )
	       + h_k[ii*n:(ii+1)*n].T * Xs[ii]
	       )
		return full_obj

	def constraints(ii):
		constr = [sum(Xs[ii]) == 1, f[ii] == Fs[ii].T * Xs[ii]]
		if (ii == 0) or (ii == p - 1):
			constr += [Xs[ii] == CASH]
		return constr

	def subproblem(ii):
		return cvx.Problem(objective(ii), constraints(ii))

	for iters in range(max_iter):
		'''Compute linear term h_k = L*x_prev'''
		t_linear_term_start = time.time()
		h_k = Lx_finance(Ds, x_prev.value, p).reshape(n*p,)
		time_solve += (time.time() - t_linear_term_start)

		'''Update in parallel'''
		if cvxpy_solve:
			Ps = []
			for ii in range(p):
				Ps.append(subproblem(ii))
			print('solving.')
			t_start = time.time()
			sum(Ps).solve(solver=SOLVER, parallel=True, warm_start=True, eps_abs=1e-9, eps_rel=1e-9)	
			t_solver += time.time() - t_start
			print('solved.')
			X = np.hstack((Xii.value for Xii in Xs))

		else:
			probs = []
			for ii in range(p):
				probs.append((ii, subproblem(ii), Xs[ii], SOLVER))
			OUTPUT = []
			t_start = time.time()

			with Pool(processes=numProcesses) as pool:
				OUTPUT = pool.map(solve_cvx_prob, probs)
			OUTPUT.sort()

			X = np.vstack([x[1] for x in OUTPUT])

			#We do the following loop to make sure that past vals of X is being passed in the opt prob. for warm start
			for i in range(p):
				Xs[i].value = OUTPUT[i][1]

			X = X.reshape(n*p,1)
			time_node = max(np.hstack(t[2] for t in OUTPUT))
			time_solve += time_node


		'''Test stopping criterion'''
		if iters > 0:
			r_k = LHAT_L.dot(X - x_prev.value)
			residual.append(np.linalg.norm(r_k))
			print('Iteration #%s. MM residual is %s.' % (iters+1, str(residual[-1])) )
			if residual[-1] <= tolerance:
				print('MM algorithm converged at iteration %s.' % (iters+1))
				break
		x_prev.value = X


	t_iter = time_solve / (iters+1) #divide by (iters+1) because of iteration zero.
	print('Average computation time per iteration is %s seconds.' % t_iter)
	x_mm = X

	return x_mm, time_solve, t_iter, residual