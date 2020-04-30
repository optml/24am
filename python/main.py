'''
 *
 * This is a sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine
 * and is based on the paper
 *    P. Richtarik, M.Jahani, S. Damla Ahipasaoglu and M. Takac
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *    https://arxiv.org/abs/1212.4137
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 *
 *
 *  Created on: Apr 12, 2020 
 *      Author: majid jahani (majidjahani89@gmail.com, maj316@lehigh.edu)
 *
 *
 *
'''
# required libs
import os
os.environ["OMP_NUM_THREADS"] = "1" # specify number of threads
import numpy as np
from numpy import linalg as LA
import heapq
import argparse

from sklearn.preprocessing import scale
from scipy.sparse import random
from scipy import stats
from scipy.stats import rv_continuous
from scipy import sparse
from scipy.sparse import *
from scipy.sparse.linalg import norm
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns
import pickle

from utils import *


def run_formulation(args, A_centered, X0, p):
    x = X0/LA.norm(X0.A)
    varCase = args['formulation'].split('_')[0]
    prevF = 1e-6
    gamma = 1
    matric_vec_prdct = MatVecMult(A_centered)
    for i in range(args['maxIter']):
        
        # u = Ax
        u = matric_vec_prdct.Ax(x.A)
        
        # If L2 variance:
        if args['formulation'].split('_')[0] == 'L2var':
            # y = u/ ||u||_2
            y = u / LA.norm(u, 2)    
            
        # If L1 variance    
        if args['formulation'].split('_')[0] == 'L1var':
            # y = sgn(u)
            y = np.sign(u)
            
        # v = A^Ty
        v = matric_vec_prdct.ATy(y)
        
        # If L0 penalty
        if args['formulation'].split('_')[1] == 'L0cons':
            # T_sv : denote the vector obtained from a by 
            # retaining only the 's' largest components of 'v'
            T_sv = T_s_v(args['sparsity'],v)
    
            # x = T_sv/ ||T_sv||_2
            x = T_sv/ LA.norm(T_sv.A)
            
            newF = F_x_y_sparse (A_centered, x, y, 1, 1, 0)
        
        # If L1 penalty
        if args['formulation'].split('_')[1] == 'L1cons':
            x, gamma = step_L1cons(args , v, gamma, p, i)
            
            newF = F_x_y_sparse(A_centered, x, y, 1, 1, 0)
            pass
        
        # If L0 constraint
        if args['formulation'].split('_')[1] == 'L0pen':
            x, gamma = step_L0pen( args, v, gamma, p, i)
            newF = F_x_y_sparse (A_centered, x, y, 2, 0, gamma)
            pass
        
        # If L1 constraint
        if args['formulation'].split('_')[1] == 'L1pen':
            x, gamma = step_L1pen(args , v, gamma, p, i)
            newF = F_x_y_sparse (A_centered, x, y, 1, 1, gamma)
            pass
        
        # stopping criteria
        if i >= args['stabilityIter']:
            if newF/prevF <= 1 + args['tol'] and (newF/prevF >= 1):
                break
        
        prevF = newF        
    
    # calculation of expected variance
    if args['formulation'].split('_')[0] == 'L2var':
        # var = ||u||_2^2 where u = Ax
        expVar = LA.norm(u,2)**2
        
    if args['formulation'].split('_')[0] == 'L1var':
        # var = ||u||_1 where u = Ax
        expVar = LA.norm(u,1)
        
        
    return x, expVar



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='24am')
    parser.add_argument('--formulation', default='L2var_L0cons',\
                        help='L2var_L0cons, L1var_L0cons,\
                              L2var_L1cons, L1var_L1cons,\
                              L2var_L0pen,  L1var_L0pen,\
                              L2var_L1pen,  L1var_L1pen')
    parser.add_argument('--dataname', default = 'ATandT_Database_of_Faces', type=str,\
                        help="AT&T Database of Faces")
    parser.add_argument('--dataDir', default = './data/', type=str, help="data directory")
    parser.add_argument('--resultsDir', default = './results/', type=str, help="data directory")
    parser.add_argument('--seed', default = 1, type=int, help="random seed")
    parser.add_argument('--density_of_SP', default = 1, type=float, help="density of starting Point")
    parser.add_argument('--sparsity', default = 16, type=int, help="sparsity target")
    parser.add_argument('--tol', default = 1e-6, type=float, help="tolerance")
    parser.add_argument('--maxIter', default = 200, type=int, help="max num of iterations")
    parser.add_argument('--numOfTrials', default = 10, type=int, help="num Of trials")
    parser.add_argument('--stabilityIter', default = 30, type=int, help="stability of gamma")
    parser.add_argument('--incDelta', default = 1e-3, type=float, help="rate of delta increase")
    
    
    
    

    args, unknown = parser.parse_known_args()
    args = vars(args)
    
    # getting the data
    A = pickle.load(open(args['dataDir']+args['dataname']+'.pickle','rb'))

    # normalizing each row of A
    A = normalize(A, axis=0, norm='l2')
    
    # n:= num of rows, p:= num of columns
    n = A.shape[0]
    p = A.shape[1]
    
    # centralizing A (column based) 
    A_centered = A - A.mean(0)
    
    R = CustomDistribution(seed=args['seed'])
    R_obj = R()  # get a frozen version of the distribution

    bestVar = -np.inf
    explainedVarSet = []
    for seed in range(args['numOfTrials']):
        
        args['seed'] = seed
        
        # initial point w.r.t. the mentioned seed
        X0 = random(p, 1, density = args['density_of_SP'], 
                    random_state = args['seed'], 
                    data_rvs = R_obj.rvs)
        
        x, expVar = run_formulation(args, A_centered, X0, p)
        
        if expVar > bestVar:
            bestVar = expVar
            
        explainedVarSet.append([seed, args['sparsity'], expVar])

        
    seedSet = []
    sparseDegreeSet = []
    ratioExplainedVarSet = []
    explainedvarianceSet = []

    for h in explainedVarSet:
        seedSet.append(h[0])
        sparseDegreeSet.append(h[1])
        explainedvarianceSet.append(h[2])
        ratioExplainedVarSet.append(h[2]/bestVar)
        
    output = pd.DataFrame({'Seed':seedSet,
                        'Explained variance/ Best explained variance':ratioExplainedVarSet,\
                        'Target sparsity level s':sparseDegreeSet ,
                        'Explained variance':explainedvarianceSet})
    
    output.to_pickle(args['resultsDir']+args['formulation']+'_'+'expectedVar_numTraial_'+\
              str(args['numOfTrials'])+'_sparsity_'+str(args['sparsity'])+'.pkl')
    
    
