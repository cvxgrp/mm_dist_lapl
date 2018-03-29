import time
import sys
import cvxpy as cvx
import numpy as np
import scipy as sc
from helper_functions import *
from asset_data import *

if __name__ == '__main__':
	recursionLimit = 1000000
	sys.setrecursionlimit(recursionLimit)
	assert(sys.getrecursionlimit() == recursionLimit)

	n = 1000 #Number of assets
	m = 50 #Number of factors
	T = 30 #Number of time periods
	print('%s ASSETS, %s FACTORS, %s TIME PERIODS.' % (n,m,T))
	tol = 1e-6
	max_iter = 100
	mus, Fs, Rs, Ds, s = generate_asset_data(n, m, T, 0, 1,'different') #code in asset_data
	CASH = np.eye(1, n, n-1).T #cash vector
	gamma = 10

	problem_data = dict()
	problem_data['L'] = make_L_finance(Ds, n, T)
	problem_data['LHAT'] = make_LHAT_matrix(problem_data['L'], 'L_diags')
	problem_data['LHAT_max_eigval'] = np.real(max(sparse.linalg.eigs(problem_data['L'])[0]))
	problem_data['n'] = n
	problem_data['m'] = m
	problem_data['p'] = T
	problem_data['gamma'] = gamma
	problem_data['tol'] = tol
	problem_data['max_iter'] = max_iter
	problem_data['mus'] = mus
	problem_data['Fs'] = Fs
	problem_data['Rs'] = Rs
	problem_data['Ds'] = Ds
	problem_data['s'] = s
	problem_data['SOLVER'] = 'OSQP'
	print('Using %s as solver.' % problem_data['SOLVER'])

	'''Solve static problem to get initial points for MM.'''
	print('Solving static problem.')
	t_start_static = time.time()
	x_static = cvx.Variable((n,1))
	obj_static = s.T*cvx.neg(x_static) + gamma*(cvx.sum_squares(Fs[0].T * x_static) + Rs[0]*cvx.sum_squares(x_static)) - mus[0].T*x_static
	constraints_static = [sum(x_static) == 1]
	cvx.Problem(cvx.Minimize(obj_static), constraints_static).solve()
	t_static = time.time() - t_start_static
	print('Computation time was %s seconds for the static problem.' % str(t_static))

	print('Solve MPO problem using problem.solve().')
	(x_opt, obj_opt, t_reg) = finance_example_regular_solve(problem_data)
	print('Time it took for regular solve = %s' % t_reg)
	
	'''instantiate x^0 with static problem solution.'''
	x_prev = cvx.Parameter((n*T,1))
	x_prev.value = np.zeros((n*T, 1))
	for ii in range(T):
		if (ii == 0) or (ii == T-1):
			x_prev.value[ii*n:(ii+1)*n] = CASH
		else:
			x_prev.value[ii*n:(ii+1)*n] = x_static.value

	problem_data['x_prev'] = x_prev

	'''Now that L and optimization problem has been formulated, solve via MM.'''
	(x_mm, t_MM, t_iter, residual) = solve_MPO_MM(problem_data)

	'''Print out metrics'''
	print('Solving with MM algorithm took %s seconds to compute.' % str(round(t_MM, 2)) )
	print('Solving normally took %s seconds to compute.' %  str(round(t_reg, 2)) )

	Fx_mm = risk_adj_return(x_mm.reshape(n*T, 1), s, Fs, Rs, Ds, mus, n, T, problem_data['L'], gamma)
	Fx_regular = risk_adj_return(x_opt.reshape(n*T,1), s, Fs, Rs, Ds, mus, n, T, problem_data['L'], gamma)
	
	print('F(x_mm) = %s' % str(Fx_mm))
	print('F(x_regular) = %s' % str(Fx_regular))
	print('F(x_mm) was %f percent away from F(x_regular).' % (100*np.abs(Fx_mm - Fx_regular)/Fx_regular))