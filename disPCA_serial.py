# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2016-11-22 22:35:15
# @Last Modified by:   twankim
# @Last Modified time: 2016-11-28 20:01:48
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import (svd, eig, pinv)
from numpy import random
from scipy.sparse.linalg import (svds, eigs)
from scipy import spatial
import math

class disPCA:
    def __init__(self,A,d):
        self.A = A
        self.d = d
        self.Ais = self.disRand() # initialize as random distribution
        self.C = None
        self.Bis = None
        self.Atis = None
        self.mode_exact = None
        self.mode_sample = None
        self.mode_norm = None
        self.err_lr = None
        self.avgDiff = None

    # Random distribution of matrix A (row-wise separation)
    def disRand(self):
        n = np.shape(self.A)[0]
        idx_rand = random.permutation(n)
        return [self.A[idx_rand[range(gid,n,self.d)],:] for gid in range(self.d)]

    def bamCompare(self,Ais,i_row,i_ran,mode_exact):
        if mode_exact == 0: # Approximate distance
            val0 = spatial.distance.cosine(self.A[i_row,:][None,:],
                                           Ais[i_ran[0]].T.dot(Ais[i_ran[0]]).dot(self.A[i_row,:][:,None])
                                           )
            val1 = spatial.distance.cosine(self.A[i_row,:][None,:],
                                           Ais[i_ran[1]].T.dot(Ais[i_ran[1]]).dot(self.A[i_row,:][:,None])
                                           )
        else: # Exact distance with projected vector
            val0 = spatial.distance.cosine(self.A[i_row,:][None,:],
                                           pinv(Ais[i_ran[0]]).dot(Ais[i_ran[0]]).dot(self.A[i_row,:][:,None])
                                           )
            val1 = spatial.distance.cosine(self.A[i_row,:][None,:],
                                           pinv(Ais[i_ran[1]]).dot(Ais[i_ran[1]]).dot(self.A[i_row,:][:,None])
                                           )
        if val0 > val1:
            # First sampled bin has larger distance 
            return i_ran[1]
        else:
            # Second sampled bin has smaller distance
            return i_ran[0]

    # Balanced Allocation for Matrix algorithm
    # A: input matrix (n by m)
    # d: number of distributed system
    # mode_exact: 0 (approximate cosine disstance without projection)
    #             1 (exact cosine distance with projection)
    # mode_sample: 0 (uniform sampling distribution)
    #              1 (update sampling distribution method 1)
    #              2 (update sampling distribution method 2)
    # mode_norm: 0 (consider only distance)
    #            1 (consider both distance and number of rows)
    def disBAM(self,mode_exact=0,mode_sample=0,mode_norm=0):
        self.mode_exact = mode_exact
        self.mode_sample = mode_sample
        self.mode_norm = mode_norm

        d = self.d
        Ais = [None] * d # allocation result
        n = np.shape(self.A)[0] # number of rows in A
        pis = [1/float(d)] * d # initialize sampling distribution

        dRows = np.zeros(d) # number of rows stored in Ais    
        dNorms = np.zeros(d) # squared 2 norms for distributed matrices

        for i in range(n):
            # 1) Sample two bins based on distribution
            idx_ran = np.random.choice(d, 2, replace=False, p=pis)
            idx_sel = 0

            if mode_norm == 0: # Consider only frobenius norm
                # Select bin to store ith row of A with balancing
                if dRows[idx_ran[0]] == 0:
                    idx_sel = idx_ran[0]
                elif dRows[idx_ran[1]] == 0:
                    idx_sel = idx_ran[1]
                else:
                    idx_sel = self.bamCompare(Ais,i,idx_ran,mode_exact)
            else: # consider both frobenius norm and number of rows 
                # select bin to store ith row of A with balancing
                if dRows[idx_ran[0]] < dRows[idx_ran[1]]:
                    # First sampled bin has smaller number of rows
                    idx_sel = idx_ran[0]
                elif dRows[idx_ran[0]] > dRows[idx_ran[1]]:
                    # Second sampled bin has smaller number of rows
                    idx_sel = idx_ran[1]
                else:
                    # Two sampled bin has same number of rows
                    if dRows[idx_ran[0]] == 0:
                        idx_sel = idx_ran[0]
                    elif dRows[idx_ran[1]] == 0:
                        idx_sel = idx_ran[1]
                    else:
                        idx_sel = self.bamCompare(Ais,i,idx_ran,mode_exact)

            # 2) Store ith row in selected bin
            if dRows[idx_sel] == 0:
                Ais[idx_sel] = self.A[i,:][None,:]
            else:
                Ais[idx_sel] = np.concatenate((Ais[idx_sel],self.A[i,:][None,:]),axis=0)
        
            # Update stored number of rows
            dRows[idx_sel] += 1

            # 3) Updating sampling distribution
            if mode_sample != 0:
                # Update 2 norm
                dNorms[idx_sel] = np.linalg.norm(Ais[idx_sel],2)**2
                if mode_sample == 1:
                    pis = self.sampDist(dRows)
                elif mode_sample == 2:
                    pis = self.sampDist2(dRows)
                else:
                    pis = self.sampDist(dRows)

        idxEmpty = np.where(dRows == 0)
        numEmpty = d - np.count_nonzero(dRows) # Check whether there is an empty bin

        self.Ais = Ais
        return Ais, idxEmpty, numEmpty

    # function for performing distributed PCA
    # Ais: list of distributed matrices [A1, A2, ... Ad]
    #       Ai has size ni by m
    # t1: target dimension for local PCA
    # t2: target dimension for glboal PCA
    # C: matrix with size m by t2 where columns are t2 principal components of A
    def fit(self, t1, t2):
        d = self.d # number of distributed matrice
        m = np.shape(self.A)[1] # dimension of row space
        Bis = [None] * d # outputs of local PCA
        Atis = [None] * d # rank t1 approximation of each Ai

        # local PCA
        for i in range(d):
            # U, S, Vh = svd(Ais[i])
            # Bis[i] = np.diag(S[:t1]).dot(Vh[:t1,:])
            # Atis[i] = U[:,:t1].dot(Bis[i])
            ni = self.Ais[i].shape[0]
            if t1 < ni:
                U, S, Vt = svds(self.Ais[i], k=t1)
                Bis[i] = np.diag(S).dot(Vt)
                Atis[i] = U.dot(Bis[i])
            else: # Number of Rows are less than t1
                U, S, Vt = svd(self.Ais[i])
                Bis[i] = np.diag(S).dot(Vt[:ni,:])
                Atis[i] = self.Ais[i]
    
        # global PCA
        K = np.zeros((m,m))
        for i in range(d):
            K += Bis[i].T.dot(Bis[i])
        
        # L,Q = eig(K)
        # C = Q[:,:t2]
        # C = C.real
        L,Q = eigs(K, k=t2)
        self.C = Q.real
        self.Bis = Bis
        self.Atis = Atis
    
    # Project A onto C using distributed calculation to get low rank approximation
    # Return error of low rank approximation in Frobenius norm
    def errLowrank(self):
        assert self.Bis != None, "Run fit function first to run distributed PCA"
    
        # Compute Lowrank approximation of A
        # by projecting parition onto CC' and stack those matrices
        Aapprox = np.concatenate([self.Atis[i].dot(self.C).dot(self.C.T) for i in range(self.d)],axis=0)
        
        self.err_lr = (np.linalg.norm(self.A-Aapprox,'fro'))**2
        return self.err_lr

    # Calculate unbalancedness of distribution in frobenius norm    
    def calcUnbal(self):
        avgNorm = math.sqrt((np.linalg.norm(self.A,'fro')**2)/float(self.d))
        self.avgDiff = np.sum([abs(avgNorm - np.linalg.norm(Ais[i],'fro'))/float(self.d)\
                               for i in range(self.d)])
        return self.avgDiff

    @staticmethod
    def sampDist(dVals):
        dVals = dVals/sum(dVals)
        pis = np.exp(-dVals)
        pis = pis/np.sum(pis)
        return pis.tolist()

    @staticmethod
    def sampDist2(dVals):
        dVals = dVals/sum(dVals)
        pis = 1/(dVals+1)
        pis = pis/np.sum(pis)
        return pis.tolist()