# IMPORTING THE MODULES

import subprocess
import sys
import os
#os.environ["OMP_NUM_THREADS"] = str(16)

import numpy as np
from numpy import abs, sqrt, log10, sin, cos, exp
np.set_printoptions(precision=16)
from math import pi, factorial
from numpy.random import rand
from numpy.linalg import norm
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg, csc_matrix
import time
from scipy.optimize import minimize
import pickle

from qonn_cobyla import *

# =========================================================================================================

# DEFINITION OF FUNCTIONS

def save(output, simulation_parameters):

    N_p = simulation_parameters[2]
    seed = simulation_parameters[4]

    store_dir = '/mnt/netapp1/Store_CSIC/home/csic/qia/amd/scaling_parameters/'
    import pickle 
    name = 'seed_kerr_res_asaf_d_uptoN'
    with open(store_dir + '/params_p_'+name+'_Np={}_seed={}.p'.format(int(N_p), seed), 'wb') as fp:
        pickle.dump(output[0], fp)

    with open(store_dir + '/cost_p_'+name+'_Np={}_seed={}.p'.format(int(N_p), seed), 'wb') as fp:
        pickle.dump(output[1], fp)

    with open(store_dir + '/params_m_'+name+'_Np={}_seed={}.p'.format(int(N_p), seed), 'wb') as fp:
        pickle.dump(output[2], fp)

    with open(store_dir + '/cost_m_'+name+'_Np={}_seed={}.p'.format(int(N_p), seed), 'wb') as fp:
        pickle.dump(output[3], fp)

    return

# We also prepare the execution used, with the corresponding ansatz:

def optimization(simulation_parameters, phi, delta, phi_delta, max_iter, conv_tol, options):

    N_e = simulation_parameters[0]
    N_c = simulation_parameters[1]
    N_p = simulation_parameters[2]
    k = simulation_parameters[3]
    seed = simulation_parameters[4]

    # Storing lists
    params_p_list = []
    params_m_list = []
    cost_p_list = []
    cost_m_list = []

    # Bounds for Kerr nonlinearity
    limits = [1.5693036971270247, 1.0357537688442213, 0.3927045226130653]

    # Initial parameters preparation
    np.random.seed(57*seed)
    parameters_p = np.array(0.1*(rand(2)-0.5), dtype=np.float64)

    # Initial parameters measurement
    np.random.seed(19*seed)
    parameters_m = np.array(0.1*(rand(2)-0.5), dtype=np.float64)

    # Kerr circuit
    print('======> Kerr circuit')
    start = time.time()

    d_list = np.arange(1, 11, 1)

    for d in d_list:
        print('=======> d = {:}'.format(d))

        layers_p = d
        layers_m = d

        setup = Setup(N_e, N_c, N_p, k)
        oc = KerrMeasCircuit(setup, layers_p, layers_m, delta)

        limit = limits[int(k-2)]/(2*layers_p)
        bounds = layers_m*[[0, 2*pi], [-limit, limit]]
        cons = []
        for factor in range(len(bounds)):
            lower, upper = bounds[factor]
            l = {'type': 'ineq',
                'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            u = {'type': 'ineq',
                'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(l)
            cons.append(u)

        # Initial state: coherent state (up to an N_p/2 threshold in each mode)
        alpha = sqrt(N_p/4)
        psi_coh = 0.0
        for n in range(int(N_p + 1)):
            cavity = np.zeros(N_p + 1, dtype=np.complex128)
            cavity[n] = 1.0
            psi_coh += alpha**n/sqrt(float(factorial(n)))*cavity
        psi_coh = exp(-abs(alpha)**2/2) * psi_coh
        psi_coh = psi_coh / sqrt(np.real(psi_coh[np.newaxis, :].conj() @ psi_coh[:, np.newaxis]))[0][0]
        psi_0 = np.kron(psi_coh, psi_coh)

        res = minimize(oc.preparation_qfi, parameters_p, args=(phi, phi_delta, psi_0), method='COBYLA',
            tol=conv_tol, options=options, constraints=cons)
        
        params_p_list.append(res['x'])
        cost_p_list.append(oc.preparation_qfi(res['x'], phi, phi_delta, psi_0))

        parameters_p = np.concatenate((res['x'], np.zeros(2, dtype=np.float64)))
        
        print('Preparation + encoding done!')

        # Preparation circuit evaluation
        psi = oc.eval_rho(res['x'], psi_0)
        psi, grad_psi = oc.eval_rho_interf_grad(phi, psi)

        print('Circuit evaluation done!')

        # Measurement VQC
        oc = KerrMeasCircuit(setup, layers_p, layers_m, delta)

        res = minimize(oc.measurement_cfi, parameters_m, args=(psi, grad_psi), method='COBYLA',
            tol=conv_tol, options=options, constraints=cons)

        params_m_list.append(res['x'])
        cost_m_list.append(oc.measurement_cfi(res['x'], psi, grad_psi))   

        parameters_m = np.concatenate((res['x'], np.zeros(2, dtype=np.float64)))

        print('Measurement done!')

        output = [params_p_list, cost_p_list, params_m_list, cost_m_list]

        save(output, simulation_parameters)

    print('t = {:} s'.format(time.time() - start))

    return

def execution(simulation_parameters):

    phi = [0, pi/3, 0]
    delta = 1e-2
    phi_delta = [0, pi/3 + delta, 0]

    # Optimization parameters
    max_iter = 1000 # Maximum number of iterations
    conv_tol = 1e-10 # Convergence tolerance
    options = {'maxiter': max_iter}

    optimization(simulation_parameters, phi, delta, phi_delta, max_iter, conv_tol, options)

    return

def run():

    # Get job id from SLURM.
    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    print('jobid = ', jobid)
    print(sys.argv)
    N_e = 2
    N_c = 2
    N_p = int(sys.argv[1]) # Number of emitters
    seed = int(jobid)
    k = 2 # Exponent of the H=U*n^k interaction

    simulation_parameters = [N_e, N_c, N_p, k, seed]

    execution(simulation_parameters)
    
    print('end!')
    return

if __name__ == '__main__':
    start_time = time.time()
    run()
    print('--- %s seconds ---' % (time.time()-start_time))
