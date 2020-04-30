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
 *  required functions and classes
 *
'''
# required libs
import os
os.environ["OMP_NUM_THREADS"] = "1" # specify number of threads
import numpy as np
from numpy import linalg as LA
import heapq

from sklearn.preprocessing import scale
from scipy.sparse import random
from scipy import stats
from scipy.stats import rv_continuous
from scipy import sparse
from scipy.sparse import *
from scipy.sparse.linalg import norm
from scipy.sparse import coo_matrix

import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns
import pickle




# ----------------------------------------
# generate random sparse vector
class CustomDistribution(rv_continuous):
    def _rvs(self, *args, **kwargs):
        return self._random_state.randn(*self._size)

    
# ----------------------------------------
# calculation of F(x,y)
def F_x_y_sparse(A, x, y, power, penNorm, gamma):
    Ax = A.dot(x.A)
    yTAx = y.T.dot(Ax)
    if penNorm == 0:
        return pow(yTAx, power) - gamma * np.count_nonzero(x.A)
    return pow(yTAx, power) - gamma * LA.norm(x.A, penNorm)
# ----------------------------------------
# calculation of T_s_v(s,v)
# T_sv : denote the vector obtained from a by 
# retaining only the 's' largest components of 'v'
def T_s_v(s,v):
    vAbs = np.abs(v)
    idx = np.array(heapq.nlargest(s, range(len(vAbs)), vAbs.take))
    return coo_matrix((v[idx].reshape(s,), \
                        (np.array(idx), np.zeros(s))),\
                        shape=(len(v),1))

# ----------------------------------------
# calculation of U_gamma_a(gamma,a)
# (U_gamma(a) )[i] := a[i][sgn(a[i]**2 - gamma)]_+
def U_gamma_a(gamma,a):
    tmp = a**2 - gamma
    sgnTmp = np.sign(tmp)
    tmp2 = sgnTmp.clip(0)
    return a * tmp2
# ----------------------------------------
# calculation of V_gamma_a(gamma,a)
# (V_gamma(a) )[i] := sgn(a[i])[|a[i]| - gamma)]_+
def V_gamma_a(gamma,a):
    tmp = np.abs(a) - gamma
    sgnTmp = np.sign(tmp)
    tmp2 = sgnTmp.clip(0)
    return a * tmp2
# ----------------------------------------
# class for multiplication of Ax and A^Ty
class MatVecMult:
    def __init__(self,
                A,
                AT = None
                ):
        self.A = A
        self.AT = A.T
        
    def Ax(self, x):
        return  self.A.dot(x)
        
    def ATy(self, y):
        return self.AT.dot(y)

# ----------------------------------------
# calculation of lambda_s_v(s,v)
# lambda_s(v) := argmin_{lambda >= 0} ( lambda \sqrt(s) + ||V_lambda(v)||_2 )
def lambda_s_v(s,v):
    sq_constr = np.sqrt(s + 0.0)
    myvector = sorted(v, key=abs) # sorted based on abs
    lambda_Low = 0;
    lambda_High = abs(myvector[-1])# lambda = \|x\|_\infty
    linfty=abs(myvector[-1])
    sum_abs_x = 0
    sum_abs_x2 = 0
    epsilon = 0.000001
    tmp = abs(myvector[-1])
    subgrad = 0
    total_elements = 0
    subgradOld = 0
    w = 0
    subgradRightOld = sq_constr
    for i in range(len(v)):
        if (i > 0):
            subgradOld = subgrad
       
        tmp = abs(myvector[-i-1])
        sum_abs_x += tmp
        sum_abs_x2 += tmp * tmp
        total_elements += 1
        #compute subgradient close to lambda=x_{(i)}
        lambda_High = tmp;
        if (i < len(v) - 1):
            lambda_Low = abs(myvector[- i - 2])
        else:
            lambda_Low = 0
        subgradLeft = (sum_abs_x - total_elements * tmp)/np.sqrt(sum_abs_x2 - 2 * tmp * sum_abs_x/
                + total_elements * tmp * tmp)
        if (subgradLeft > sq_constr and subgradRightOld < sq_constr): 
            w = tmp
            
            break
        
        subgradRight = (sum_abs_x - total_elements * lambda_Low)/np.sqrt(sum_abs_x2 - 2 * lambda_Low * sum_abs_x/
                + total_elements * lambda_Low * lambda_Low)
        a = total_elements * (total_elements - sq_constr * sq_constr);
        b = 2 * sum_abs_x * sq_constr * sq_constr - 2 * total_elements * sum_abs_x
        c = sum_abs_x * sum_abs_x - sq_constr * sq_constr * sum_abs_x2;
        w = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a);
        if (w > lambda_Low and  w < lambda_High ): 
            break
        
        w = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        if (w > lambda_Low and  w < lambda_High and w < linfty- epsilon ):
            
            break
        
        subgradRightOld = subgradRight
    return w[0]

# ----------------------------------------
# finding 'x' for the case with L0 penalty
def step_L0pen(args,v,gamma, p, i):
    delta = 1e-8
    s = args['sparsity']
    idx = min(s+1, p)
    tmp = sorted(v**2, key=abs)
    if i <= args['stabilityIter']:
        gamma = np.abs( tmp[-idx])

    U_gamma_v = U_gamma_a(gamma,v)
    U_gamma_v = sparse.coo_matrix(U_gamma_v)
    
    # gaurantee that sparsity level of U_gamma_v is at most 's'
    while csr_matrix.count_nonzero(U_gamma_v) > s:
        gamma =  tmp[-idx] + delta 
        U_gamma_v = U_gamma_a(gamma,v)
        U_gamma_v = sparse.coo_matrix(U_gamma_v)
        delta *= args['incDelta']
        
    # x = U_gamma_v / ||U_gamma_v||_2
    x = U_gamma_v/ LA.norm(U_gamma_v.A)
    x = sparse.coo_matrix(x)
    return x, gamma
# ----------------------------------------
# finding 'x' for the case with L1 penalty
def step_L1pen(args,v,gamma, p, i):
    delta = 1e-8
    s = args['sparsity']
    idx = min(s+1, p)
    tmp = sorted(v, key=abs)
    if i <= args['stabilityIter']:
        gamma = np.abs( tmp[-idx])

    V_gamma_v = V_gamma_a(gamma,v)
    V_gamma_v = sparse.coo_matrix(V_gamma_v)
    
    # gaurantee that sparsity level of V_gamma_v is at most 's'
    while csr_matrix.count_nonzero(V_gamma_v) > s:
        gamma =  np.abs(tmp[-idx]) + delta 
        V_gamma_v = V_gamma_a(gamma,v)
        V_gamma_v = sparse.coo_matrix(V_gamma_v)
        delta *= args['incDelta']
        
    # x = V_gamma_v / ||V_gamma_v||_2
    x = V_gamma_v/ LA.norm(V_gamma_v.A)
    x = sparse.coo_matrix(x)
    return x, gamma

# ----------------------------------------
# finding 'x' for the case with L1 constraint
def step_L1cons(args,v,gamma, p, i):
    delta = 1e-8
    s = args['sparsity']
    idx = min(s+1, p)
    tmp = sorted(v, key=abs)
    if i <= args['stabilityIter']:
        _lambda_s_v_Val = lambda_s_v(s,v)
        tmp = sorted(v, key=abs)
        s_th_el_of_V = np.abs( tmp[-int( s )] ) - delta

        if s != p:
            idxLamba = np.where(np.abs(tmp) == list(filter(lambda k: k > _lambda_s_v_Val, np.abs(tmp) ))[0])[0][0]
            idxSortedV = np.where(np.abs(tmp) == list(filter(lambda k: k > s_th_el_of_V, np.abs(tmp) ))[0])[0][0]

            if np.abs(s - p + idxLamba) < np.abs(s - p + idxSortedV):
                gamma = _lambda_s_v_Val
            else:
                gamma = s_th_el_of_V
        else:
            gamma = 0 # bcs V_gamma_v will be V itself

    V_gamma_v = V_gamma_a(gamma,v)

    V_gamma_v = sparse.coo_matrix(V_gamma_v)
    gammatmp = gamma
    
    # gaurantee that sparsity level of V_gamma_v is at most 's'
    while csr_matrix.count_nonzero(V_gamma_v) > s:
        gamma = gammatmp + delta 
        V_gamma_v = V_gamma_a(gamma,v)

        V_gamma_v = sparse.coo_matrix(V_gamma_v)

        delta *= args['incDelta']

    # x = V_gamma_v / ||V_gamma_v||_2
    x = V_gamma_v/ LA.norm(V_gamma_v.A)
    x = sparse.coo_matrix(x)
    
    return x, gamma
# ----------------------------------------
