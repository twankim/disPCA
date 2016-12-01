# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2016-11-24 18:25:48
# @Last Modified by:   twankim
# @Last Modified time: 2016-11-30 18:16:49
# -*- coding: utf-8 -*-

import disPCA_serial
import numpy as np
from numpy import random
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import time

# ----------- Parameters for disPCA and BAM --------------
n = 10000 # dimension of column space
m = 200 # dimension of row space
d = 20 # number of distributed system
mode_exact = 0
mode_sample = 0
mode_norm = 0
gen_mode = 0
normtype = 'fro'

# ----------- Parameters for test -------------
t1s = [2,5,10,15,20,25]
t2 = 10 # target dimension of global PCA
rs = range(2,d+1,2)
eps_t1_ran = np.zeros(len(t1s))
eps_min_ran = np.zeros(len(t1s))
eps_max_ran = np.zeros(len(t1s))
eps_t1_bam = np.zeros((len(t1s),len(rs)))
eps_min_bam = np.zeros((len(t1s),len(rs)))
eps_max_bam = np.zeros((len(t1s),len(rs)))
iterMax = 10
verbose = False # Verbose option

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Temporary
n=600
m=50
d=6
t1s = [2,5,7,10,12,15,20]
t2 = 10 # target dimension of global PCA
rs = range(2,d+1,2)
eps_t1_ran = np.zeros(len(t1s))
eps_min_ran = np.zeros(len(t1s))
eps_max_ran = np.zeros(len(t1s))
eps_t1_bam = np.zeros((len(t1s),len(rs)))
eps_min_bam = np.zeros((len(t1s),len(rs)))
eps_max_bam = np.zeros((len(t1s),len(rs)))
mode_exact = 0
gen_mode = 0
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if verbose:
    def vprint( obj ):
        print obj
else:   
    def vprint( obj ):
        pass

# Generate random matrix A
def genRanMat(n,m,gen_mode=0,k=0,d=d):
    assert n>m, "!!!! n must be larger than m"
    assert k<m, "!!!! k must be smaller than m"
    assert d<n, "!!!! d must be smaller than n"

    if gen_mode == 1: # Random matrix with Fixed singular values (decaying 1/k)
        A = random.rand(n,m)
        U, S, Vh = np.linalg.svd(A)
        for i in range(k):
            S[i] = 1+1/float(i+1)
        S[k:] = S[k-1]*0.01
        A = U[:,:len(S)].dot(np.diag(S)).dot(Vh)
    elif gen_mode == 2: # Random matrix with several same rows
        n_sub = int(n/d)
        Asub = random.randn(n_sub,m)
        A = np.tile(Asub,(d,1))
        A = np.concatenate((A,A[:n-n_sub*d,:]),axis=0)
    else: # Standard normal random marix
        A = random.randn(n,m)
    return A


for idxt1, t1 in enumerate(t1s):
    # t1: target dimension of local PCA
    # Note that t1, t2 < m
    eps_ran = np.zeros(iterMax)
    eps_bam = np.zeros((iterMax,len(rs)))
    
    print "\n<t1={}, t2={}>".format(t1,t2)
    for iterN in range(iterMax):
        print "-------------- {}: Trial # {}/{} --------------".format(idxt1,iterN,iterMax-1)
        
        # Generate random matrix A
        A = genRanMat(n,m,gen_mode,t2)

        # Random (random distribution)
        vprint(" Distributing rows of matrix (random)...")
        pca_ran = disPCA_serial.disPCA(A,d)
        pca_ran.disRand()
        
        vprint(" Applying disPCA algorithm (random)...\n")
        # apply disPCA    
        pca_ran.fit(t1=t1,t2=t2)

        err_disPCA_ran = pca_ran.score(normtype)

        # Balanced
        err_disPCA_bam = np.zeros(len(rs))
        for idx_r, r in enumerate(rs):
            vprint(" Distributing rows of matrix (balanced)...")
            pca_bam = disPCA_serial.disPCA(A,d,r)
            pca_bam.disBAM(mode_exact=mode_exact, mode_sample=mode_sample, mode_norm=mode_norm)

            dRows = [Ais_j.shape[0] for Ais_j in pca_bam.Ais]
            idxEmpty = np.where(dRows == 0)
            numEmpty = d - np.count_nonzero(dRows) # Check whether there is an empty bin
            vprint("     -> Number of empty bins: {}".format(numEmpty))
            vprint("     Number of rows {}".format(", ".join(map(str,dRows))))
        
            vprint(" Applying disPCA algorithm (balanced)...\n")
            # apply disPCA
            pca_bam.fit(t1=t1,t2=t2)

            err_disPCA_bam[idx_r] = pca_bam.score(normtype)
        
        # Evaluation
        vprint(" Applying SVD for approximation...\n")
        U, S, Vh = svds(A, k=t2)
        Aopt = U.dot(np.diag(S)).dot(Vh)
        err_opt = pca_bam.errLowrank(A,Aopt,normtype)
    
        eps_ran[iterN] = err_disPCA_ran/err_opt-1
        eps_bam[iterN,:] = err_disPCA_bam/err_opt-1
    
    eps_t1_ran[idxt1] = np.mean(np.log10(eps_ran))
    eps_t1_bam[idxt1,:] = np.mean(np.log10(eps_bam),axis=0)

    eps_t1_ran[np.where(np.mean(eps_ran)<1e-9)] = -9
    eps_t1_bam[np.where(np.mean(eps_bam,axis=0)<1e-9)] = -9
        
    eps_min_ran[idxt1] = np.log10(eps_ran).min()
    eps_max_ran[idxt1] = np.log10(eps_ran).max()
    
    eps_min_bam[idxt1,:] = np.log10(eps_bam).min(axis=0)
    eps_max_bam[idxt1,:] = np.log10(eps_bam).max(axis=0)
    
        
    print "<disPCA, random (uniformly)> Error (epsilon): {}".format(eps_t1_ran[idxt1])
    for i_r, r in enumerate(rs):
        print "<disPCA, balanced (r={})> Error (epsilon): {}".format(r,eps_t1_bam[idxt1,i_r])
    
plt.figure()
plt.plot(np.array(t1s)/float(t2),eps_t1_ran,'^-')
legends = ['disPCA']
for i_r, r in enumerate(rs):
    plt.plot(np.array(t1s)/float(t2),eps_t1_bam[:,i_r],'x-')
    legends.append('BAM+disPCA (r={})'.format(r))
plt.xlabel('t1/t2')
plt.ylabel('log10(epsilon)')
plt.legend(legends)
plt.title('<Mean log-epsilon>\nA random (n={}, m={}, d={}) gen_mode={}'.format(n,m,d,gen_mode))

plt.figure()
plt.plot(np.array(t1s)/float(t2),eps_max_ran,'^-')
for i_r, r in enumerate(rs):
    plt.plot(np.array(t1s)/float(t2),eps_max_bam[:,i_r],'x-')
plt.xlabel('t1/t2')
plt.ylabel('max(log10(epsilon))')
plt.legend(legends)
plt.title('<Max log-epsilon>\nA random (n={}, m={}, d={}) gen_mode={}'.format(n,m,d,gen_mode))

plt.show()

# print "<disPCA, balanced>"
# print "- Error (epsilon) / Balance score: " + str(err_disPCA/err_opt-1) + " / " + str(calcUnbal(A,Ais))
# print "<disPCA, unbalanced>"
# print "- Error (epsilon) / Balance score: " + str(err_disPCA2/err_opt-1) + " / " + str(calcUnbal(A,Ais2))
# print "- Error in Frobenius norm (optimal): " + str(err_opt)