# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2016-11-24 18:25:48
# @Last Modified by:   twankim
# @Last Modified time: 2016-11-28 14:41:35
# -*- coding: utf-8 -*-

import disPCA_serial
import numpy as np
from numpy import random
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import time

# ----------- Parameters for test -------------
t1s = [5,10,15,20,30,50]
t2 = 10 # target dimension of global PCA
eps_t1_bam = np.zeros(len(t1s))
eps_t1_ran = np.zeros(len(t1s))
eps_min_bam = np.zeros(len(t1s))
eps_min_ran = np.zeros(len(t1s))
eps_max_bam = np.zeros(len(t1s))
eps_max_ran = np.zeros(len(t1s))
iterMax = 10

# ----------- Parameters for disPCA and BAM --------------
n = 5000 # dimension of column space
m = 200 # dimension of row space
d = 10 # number of distributed system
n = 10000
mode_exact = 1
mode_sample = 0
mode_norm = 0
gen_mode = 1

# Verbose option
verbose = False

if verbose:
    def vprint( obj ):
        print obj
else:   
    def vprint( obj ):
        pass

# Generate random matrix A
def genRanMat(n,m,gen_mode=0,t1=0):
    if gen_mode == 0: # Random matrix with Fixed singular values (decaying 1/k)
        A = random.rand(n,m)
        U, S, Vh = np.linalg.svd(A)
        for i in range(2*t1):
            S[i] = 1+1/float(i+1)
        S[2*t1:] = 0.1
        A = U[:,:len(S)].dot(np.diag(S)).dot(Vh)
    else: # Standard normal random marix
        A = random.randn(n,m)
    return A


for idxt1, t1 in enumerate(t1s):
    # t1: target dimension of local PCA
    # Note that t1, t2 < m
    eps_bam = np.zeros(iterMax)
    eps_ran = np.zeros(iterMax)
    
    for iterN in range(iterMax):
        print "-------------- {}: Trial # {}/{} --------------".format(idxt1,iterN,iterMax-1)
        
        # Generate random matrix A
        A = genRanMat(n,m,gen_mode,t1)

        # Random (random distribution)
        vprint(" Distributing rows of matrix (random)...")
        pca_ran = disPCA_serial.disPCA(A,d)
        
        vprint(" Applying disPCA algorithm (random)...\n")
        # apply disPCA    
        pca_ran.fit(t1=t1,t2=t2)

        err_disPCA_ran = pca_ran.errLowrank()
        
        # Balanced
        vprint(" Distributing rows of matrix (balanced)...")
        pca_bam = disPCA_serial.disPCA(A,d)
        Ais,idxEmpty,numEmpty = pca_bam.disBAM(mode_exact=mode_exact,
                                               mode_sample=mode_sample,
                                               mode_norm=mode_norm
                                               )
        vprint("     -> Number of empty bins: {}".format(numEmpty))
        vprint("     Number of rows {}".format(", ".join(map(str,[Ais[j].shape[0] for j in range(d)]))))
        
        vprint(" Applying disPCA algorithm (balanced)...\n")
        # apply disPCA
        pca_bam.fit(t1=t1,t2=t2)

        err_disPCA_bam = pca_bam.errLowrank()
        
        vprint(" Applying SVD for approximation...\n")
        U, S, Vh = svds(A, k=t2)
        Aopt = U.dot(np.diag(S)).dot(Vh)
        err_opt = (np.linalg.norm(A-Aopt,'fro'))**2
    
        eps_bam[iterN] = err_disPCA_bam/err_opt-1
        eps_ran[iterN] = err_disPCA_ran/err_opt-1
    
    eps_t1_bam[idxt1] = np.mean(np.log10(eps_bam))
    eps_t1_ran[idxt1] = np.mean(np.log10(eps_ran))
    if np.mean(eps_bam) < 1e-9:
        eps_t1_bam[idxt1] = -9
    if np.mean(eps_ran) < 1e-9:
        eps_t1_ran[idxt1] = -9
        
    eps_min_bam[idxt1] = min(np.log10(eps_bam))
    eps_min_ran[idxt1] = min(np.log10(eps_ran))
    eps_max_bam[idxt1] = max(np.log10(eps_bam))
    eps_max_ran[idxt1] = max(np.log10(eps_ran))
        
    print "<disPCA, balanced> Error (epsilon): {}".format(eps_t1_bam[idxt1])
    print "<disPCA, random(uniformly)> Error (epsilon): {}\n".format(eps_t1_ran[idxt1])
    
plt.figure()
plt.plot(np.array(t1s)/float(t2),eps_t1_bam,'rx-',np.array(t1s)/float(t2),eps_t1_ran,'b^-')
plt.xlabel('t1/t2')
plt.ylabel('log10(epsilon)')
plt.legend(['BAM+disPCA','disPCA'])
plt.title('<Mean epsilon>\nA random (n={}, m={}, d={})'.format(n,m,d))

plt.figure()
plt.plot(np.array(t1s)/float(t2),eps_max_bam,'rx-',np.array(t1s)/float(t2),eps_max_ran,'b^-')
plt.xlabel('t1/t2')
plt.ylabel('max(log10(epsilon))')
plt.legend(['BAM+disPCA','disPCA'])
plt.title('<Max epsilon>\nA random (n={}, m={}, d={})'.format(n,m,d))

plt.show()

# print "<disPCA, balanced>"
# print "- Error (epsilon) / Balance score: " + str(err_disPCA/err_opt-1) + " / " + str(calcUnbal(A,Ais))
# print "<disPCA, unbalanced>"
# print "- Error (epsilon) / Balance score: " + str(err_disPCA2/err_opt-1) + " / " + str(calcUnbal(A,Ais2))
# print "- Error in Frobenius norm (optimal): " + str(err_opt)