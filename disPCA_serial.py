# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2016-11-22 22:35:15
# @Last Modified by:   twankim
# @Last Modified time: 2016-12-07 16:12:36
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import (svd, eig, pinv)
from numpy import random
from scipy.sparse.linalg import (svds, eigs)
from scipy import spatial
import math

class disPCA:
    def __init__(self,A,d,r=1):
        self.A = A
        self.d = d
        self.r = r # Don't need for random distribution
        self.Ais = None
        self.idx_dist = None
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
        self.idx_dist = [idx_rand[range(gid,n,self.d)] for gid in range(self.d)]
        self.Ais = [self.A[idx_Ai,:] for idx_Ai in self.idx_dist]

    def bamCompare(self,Ais,i_row,i_ran,mode_exact):
        if mode_exact == 0: # Approximate distance
            vals = [spatial.distance.cosine(self.A[i_row,:][None,:],
                                           Ais[i_r].T.dot(Ais[i_r]).dot(self.A[i_row,:][:,None])
                                           ) for i_r in i_ran]
        else: # Exact distance with projected vector
            vals = [spatial.distance.cosine(self.A[i_row,:][None,:],
                                            pinv(Ais[i_r]).dot(Ais[i_r]).dot(self.A[i_row,:][:,None])
                                            ) for i_r in i_ran]
        # Return sampled bin with the smallest distance
        return i_ran[np.argmin(vals)]

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
        r = self.r
        Ais = [None] * d # allocation result
        idx_dist = [None] * d# indices of A for each Ai
        n = np.shape(self.A)[0] # number of rows in A
        pis = [1/float(d)] * d # initialize sampling distribution

        dRows = np.zeros(d) # number of rows stored in Ais    
        dNorms = np.zeros(d) # squared 2 norms for distributed matrices

        for i in range(n):
            # 1) Sample two bins based on distribution
            idx_ran = np.random.choice(d, r, replace=False, p=pis)
            idx_sel = 0

            if mode_norm == 0: # Consider only frobenius norm
                # Select bin to store ith row of A with balancing
                if any(dRows[idx_ran] == 0):
                    # A bin with 0 row exists
                    idx_sel = idx_ran[dRows[idx_ran].argmin()]
                else:
                    idx_sel = self.bamCompare(Ais,i,idx_ran,mode_exact)
            else: # consider both frobenius norm and number of rows 
                # select bin to store ith row of A with balancing
                if len(set(dRows))>1:
                    # Number of rows are not even among r sampled bins
                    idx_sel = idx_ran[dRows[idx_ran].argmin()]
                else:
                    # Number of rows are all equal
                    if dRows[idx_ran[0]] == 0:
                        # Number of rows are all 0
                        idx_sel = idx_ran[0]
                    else:
                        idx_sel = self.bamCompare(Ais,i,idx_ran,mode_exact)

            # 2) Store ith row in selected bin
            if dRows[idx_sel] == 0:
                Ais[idx_sel] = self.A[i,:][None,:]
                idx_dist[idx_sel] = np.array([i])
            else:
                Ais[idx_sel] = np.concatenate((Ais[idx_sel],self.A[i,:][None,:]),axis=0)
                idx_dist[idx_sel] = np.append(idx_dist[idx_sel],i)
        
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
        self.Ais = Ais
        self.idx_dist = idx_dist

    # function for performing distributed PCA
    # Ais: list of distributed matrices [A1, A2, ... Ad]
    #       Ai has size ni by m
    # t1: target dimension for local PCA
    # t2: target dimension for glboal PCA
    # C: matrix with size m by t2 where columns are t2 principal components of A
    def fit(self, t1, t2):
        assert self.Ais != None, "!!!! First, distribute rows of A using disRand or disBAM"
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
            if t1 < ni: # Target rank t1 is less than Number of Rows
                U, S, Vt = svds(self.Ais[i], k=t1)
                Bis[i] = np.diag(S).dot(Vt)
                Atis[i] = U.dot(Bis[i])
            else: # Number of Rows is less than t1
                U, S, Vt = svd(self.Ais[i])
                Bis[i] = np.diag(S[:ni]).dot(Vt[:ni,:])
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
    def score(self,normtype):
        assert self.Bis != None, "!!!! Run fit function first to run distributed PCA"
        # Compute Lowrank approximation of A
        # by projecting parition onto CC' and stack those matrices
        Aapprox = np.zeros(self.A.shape)

        for Ati,idx_Ai in zip(self.Atis,self.idx_dist):
            Aapprox[idx_Ai,:] = Ati.dot(self.C).dot(self.C.T)

        self.err_lr = self.errLowrank(self.A,Aapprox,normtype)
        return self.err_lr

    # !!!!!!!!!! To be fixed !!!!!!!!!!!!!!!!
    # Calculate unbalancedness of distribution in frobenius norm
    def calcUnbal(self):
        avgNorm = math.sqrt((np.linalg.norm(self.A,'fro')**2)/float(self.d))
        self.avgDiff = np.sum([abs(avgNorm - np.linalg.norm(Ai,'fro'))/float(self.d)\
                               for Ai in self.Ais])
        return self.avgDiff

    @staticmethod
    def errLowrank(A,Aapprox,normtype='fro'):
        assert A.shape == Aapprox.shape, "!!!! Shape of two input matrices must match"
        return (np.linalg.norm(A-Aapprox,normtype))**2

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