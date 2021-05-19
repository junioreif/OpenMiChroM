# Copyright (c) 2020-2021 The Center for Theoretical Biological Physics (CTBP) - Rice University
# This file is from the Open-MiChroM project, released under the MIT License.

R"""  
The :class:`~.Optimization` classes perform the energy function parameters training of the chromosomes based on experimental Hi-C data.
"""

from simtk.openmm.app import *
import simtk.openmm as openmm
import simtk.unit as units
from sys import stdout, argv
import numpy as np
from six import string_types
import os
import time
import random
import h5py
from scipy.spatial import distance
import scipy as sp
import itertools
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import normalize


class FullTraining:
    R"""
    The :class:`~.FullTraining` class performs the parameters training for each selected loci pair interaction. 
    
    Details about the methodology are decribed in "Zhang, Bin, and Peter G. Wolynes. "Topology, structures, and energy landscapes of human chromosomes." Proceedings of the National Academy of Sciences 112.19 (2015): 6062-6067."
    
    
    The :class:`~.FullTraining` class receive a Hi-C matrix (text file) as input. The parameters :math:`\mu` (mi) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
    
    Args:
        mi (float, required):
            Parameter in the probability of crosslink function.
        rc (float, required):
            Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`.
        cutoff (float, optional):
            Cutoff value for reducing the noise in the original data. Values lower than the **cutoff** are considered :math:`0.0`.
        reduce (bool, optional):
            Whether to reduce the number of interactions to be considered in the inversion. If False, it will consider every possible interaction :math:`(N*(N-1)/2)`. If True, it is necessary to give values for the lower and higher cutoffs. (Default value: :code:`True`). 
        pair_h (int, required if **reduce** = :code:`True`):
            Loci selection to apply the high-resolution cutoff. If **pair_h** = 2, the interaction in the high-resolution index grid :math:`2 : 2 : N × 2:2:N`  are subject to a cutoff value **c_h**, where `N` is the total number of monomers interactions  (Default value = 2).
        c_h (float, required if **reduce** = :code:`True`)):
            The the high-resolution cutoff. (Default value = 0.1).
        pair_l (int, required if **reduce** = :code:`True`)):
            Loci selection to apply the high-resolution cutoff. If **pair_l** = 4, the interaction in the low-resolution index grid :math:`1:4:N×1:4:N`  are subject to a cutoff value **c_l**, where `N` is the total number of monomers interactions  (Default value = 4).
        c_l (float, required if **reduce** = :code:`True`)):
            The the low-resolution cutoff. (Default value = 0.02).
    """
    def __init__(self, state, expHiC, mi=1.0, rc=2.5, 
                 cutoff=0.0, reduce=True,
                 pair_h=2, c_h=0.1, pair_l=4, c_l=0.02, gpu=False 
                ):
            
        self.mi = mi
        self.rc = rc
        self.cutoff = cutoff
        self.gpu = gpu
        
        self.getHiCexp(expHiC, centerRemove=False, centrange=[0,0])
        self.hic_sparse = sp.sparse.csr_matrix(np.triu(self.expHiC, k=2))
        if (reduce):
            self.appCutoff(pair_h, c_h, pair_l, c_l)
       

        self.ind = self.get_indices(self.hic_sparse)
        if (self.gpu):
            import pycuda.driver as drv
            import pycuda.gpuarray as gpuarray
            import pycuda.autoinit
            import skcuda.linalg as linalg

        self.size = len(self.ind)     
        self.Pi = np.zeros(self.size)
        self.Prod_dist = np.zeros(self.hic_sparse.shape)
        self.PiPj = np.zeros((self.size,self.size))
        self.NFrames = 0

    def appCutoff(self, pair_h, c_h, pair_l, c_l):
        R"""
        Applies the cutoff for low- and high-resolution values.
        """
        N = self.hic_sparse.shape[0]
        print('Non-zero interactions before the cutoff: ', self.hic_sparse.nnz)

        hic_full = self.hic_sparse.todense()
        hic_final = np.zeros(self.hic_sparse.shape)


        values = [n for n in range(0,N,pair_h)]
        index =  [x for x in itertools.combinations_with_replacement(values, r=2)]
        for i in index:
            if (hic_full[i] > c_h):
                hic_final[i] = hic_full[i]

        print('Non-zero interactions after high-resolution cutoff ({}): {}'.format( c_h, sp.sparse.csr_matrix(np.triu(hic_final, k=2)).nnz ))

        values = [n for n in range(1,N,pair_l)]
        index =  [x for x in itertools.combinations_with_replacement(values, r=2)]
        for i in index:
            if (hic_full[i] > c_l):
                hic_final[i] = hic_full[i]
        self.hic_sparse = sp.sparse.csr_matrix(np.triu(hic_final, k=2))
        print('Non-zero interactions after low-resolution cutoff ({}): {}'.format(c_l,  self.hic_sparse.nnz ))

    
    def get_indices(self, hic):
        R"""
        Receives non-zero interaction indices, *i.e.*, the loci pair *i* and *j* which interaction will be optimized.
        """
        index = sp.sparse.find(hic)
        self.rows = index[0]
        self.cols = index[1]
        self.values = index[2]
        ind = []
        for i in range(len(index[0])):
            ind.append((self.rows[i], self.cols[i]))
        return(ind)
        
    def getHiCexp(self, filename, centerRemove=False, centrange=[0,0]):
        R"""
        Receives the experimental Hi-C map (Full dense matrix) in a text format and performs the data normalization from Hi-C frequency/counts/reads to probability.
        
        Args:
            centerRemove (bool, optional):
                Whether to set the contact probability of the centromeric region to zero. (Default value: :code:`False`).
            centrange (list, required if **centerRemove** = :code:`True`)):
                Range of the centromeric region, *i.e.*, :code:`centrange=[i,j]`, where *i* and *j*  are the initial and final beads in the centromere. (Default value = :code:`[0,0]`).
        """
        allmap = np.loadtxt(filename)

        r=np.triu(allmap, k=1)
        r[np.isinf(r)]= 0.0
        r[np.isnan(r)]= 0.0
        r = normalize(r, axis=1, norm='max') 
        rd = np.transpose(r) 
        self.expHiC = r+rd + np.diag(np.ones(len(r)))
        
        if (centerRemove):
            centrome = range(centrange[0],centrange[1])
            self.expHiC[centrome,:] = 0.0
            self.expHiC[:,centrome] = 0.0
        self.expHiC[self.expHiC<self.cutoff] = 0.0
        
    def probCalc(self, state):
        R"""
        Calculates the contact probability matrix and the cross term of the Hessian.
        """

        Prob = 0.5*(1.0 + np.tanh(self.mi*(self.rc - distance.cdist(state,state, 'euclidean'))))
    
        Pi = []
        for i in self.ind:
            Pi.append(Prob[i[0], i[1]])
        Pi = np.array(Pi)
    

        #if (self.gpu):
        #    linalg.init()
        #    Pi_gpu = gpuarray.to_gpu(Pi)
        #    PiPj_gpu = linalg.dot(Pi_gpu, Pi_gpu)
        #    self.PiPj += PiPj_gpu
        #else:
        PiPj = np.outer(Pi,Pi)
        self.PiPj += PiPj                                       

     
        self.Prod_dist += Prob
        self.Pi += Pi
        self.NFrames += 1
        
    def getPearson(self):
        R"""
        Calculates the Pearson's Correlation between the experimental Hi-C used as a reference for the training and the *in silico* Hi-C obtained from the optimization step.
        """
        r1 = sp.sparse.csr_matrix((self.Pi/self.NFrames,(self.rows,self.cols)), shape=(self.expHiC.shape[0],self.expHiC.shape[0])).todense()
        r2 = self.hic_sparse.todense()

        r1[np.isinf(r1)]= 0.0
        r1[np.isnan(r1)]= 0.0
        r1[r1 <= 0.001]= 0.0
        r2[np.isinf(r2)]= 0.0
        r2[np.isnan(r2)]= 0.0
        r2[r2<=0.001] = 0.0

        np.fill_diagonal(r1,0.0)
        np.fill_diagonal(r2,0.0)


        SEED = 100
        random.seed(SEED)
        print(len(list(r1[np.triu_indices(np.shape(r1)[0])])))
        print("\n")
        print(np.int(0.01*np.shape(r1)[0]*np.shape(r1)[0]))                              
        
        a1 = np.asarray(random.sample(list(r1[np.triu_indices(np.shape(r1)[0])]),np.int(0.01*np.shape(r1)[0]*np.shape(r1)[0])))
        a1 = r1[np.triu_indices(np.shape(r1)[0])]
        random.seed(SEED)
        a2 = np.asarray(random.sample(list(r2[np.triu_indices(np.shape(r2)[0])]),np.int(0.01*np.shape(r2)[0]*np.shape(r2)[0])))
        a2 = r2[np.triu_indices(np.shape(r1)[0])]


        return(pearsonr(a1,a2)[0])
        
    def getLambdas(self):
        R"""
        Calculates the Lagrange multipliers of each selected interaction and returns the matrix containing the energy values for the optimization step.
        """
        self.phi_exp = self.values
        self.phi_sim = self.Pi/self.NFrames
        gij = self.phi_exp - self.phi_sim
        

        Pi2_mean = np.outer(self.phi_sim,self.phi_sim)

        PiPj_mean = self.PiPj/self.NFrames

        Bij = PiPj_mean - Pi2_mean
        
        #if (self.gpu):
        #    Bij_gpu = gpuarray.to_gpu(Bij)
        #    invBij_gpu = linalg.pinv(Bij_gpu)
        #else:
        invBij = sp.linalg.pinvh(Bij)

        #calculate lambdas
        #if (self.gpu):
        #    gij_gpu = gpuarray.to_gpu(gij)
        #    lambdas_gpu = linalg.dot(invBij_gpu, gij_gpu)
        #    lambdas = lambdas_gpu.get()
        #else:
        lambdas = np.matmul(invBij, gij)

        
        lamb_matrix = sp.sparse.csr_matrix((lambdas,(self.rows,self.cols)), shape=(self.expHiC.shape[0],self.expHiC.shape[0]))
        
        self.error = (np.sum(np.absolute(gij)))/(np.sum(self.phi_exp))
        
        return(lamb_matrix)


class CustomMiChroMTraining:
    R"""
    The :class:`~.CustomMiChroMTraining` class performs the parameters training employing MiChroM (Minimal Chromatin Model) energy function. 
    
    Details about the methodology are decribed in "Di Pierro, Michele, et al. "Transferable model for chromosome architecture." Proceedings of the National Academy of Sciences 113.43 (2016): 12168-12173."
    
    
    The :class:`~.CustomMiChroMTraining` class receive a Hi-C matrix (text file) as input. The parameters :math:`\mu` (mi) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
    
    :class:`~.CustomMiChroMTraining` optimizes the type-to-type (Types) and the Ideal Chromosome (IC) potential parameters separately. 
    
    Args:
        ChromSeq (file, required):
           Chromatin sequence of types file. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`_.
        mi (float, required):
            Parameter in the probability of crosslink function (Default value = 3.22, for human chromosomes in interphase).
        rc (float, required):
            Parameter in the probability of crosslink function, :math:`f(rc) = 0.5` (Default value = 1.78, for human chromosomes in interphase).
        cutoff (float, optional):
            Cutoff value for reducing the noise in the original data. Values lower than the **cutoff** are considered :math:`0.0`.
        dinit (int, required):
            The first neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential for training. (Default value = 3).
        dend (int, required):
            The last neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential for training. (Default value = 200).
    """
   
    #def __init__(self, state, ChromSeq="chr_beads.txt", mi=3.22, rc=1.78, cutoff=0.0, dinit=3, dend=200): 
 
    def __init__(self, state, TypeList=None, name='distMatrix', nHood=3, cutoff=0.0, mi=5.33, rc=1.61,lamb_size=200): 
        self.name = name

        self.size = len(state)
        self.P=np.zeros((self.size,self.size))
        self.Pold=np.zeros((self.size,self.size))
        self.r_cut = rc 
        self.mu  = mi 
        self.Bij = np.zeros((dend,dend))
        self.diff_types = set(ChromSeq)
        self.n_types = len(self.diff_types)
        self.n_inter = int(self.n_types*(self.n_types-1)/2 + self.n_types)
        self.polds_type = np.zeros((self.n_types, self.n_types))
        self.Bij_type = np.zeros((self.n_inter,self.n_inter))
        self.Nframes = 0 
        self.dinit = dinit
        self.cutoff = cutoff
        
##########################################################################################
#### IDEAL CHROMOSOME OPTIMIZATION
##########################################################################################
    
    def probCalculation(self, state, dmax=200):
        R"""
        Calculates the contact probability matrix and the cross term of the Hessian for the Ideal Chromosome optimization.
        """
        PiPj = np.zeros((dmax,dmax))
        self.Pold += self.P
        self.P = 0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state,state, 'euclidean'))))
        self.P[self.P<self.cutoff] = 0.0
        dmaxl = range(dmax)
        for i, j in itertools.product(dmaxl,dmaxl):
            PiPj[i,j] = np.mean(np.diagonal(self.P, offset=(i+self.dinit)))*np.mean(np.diagonal(self.P, offset=(j+self.dinit)))
      
        self.Bij += PiPj
        self.Nframes += 1 
        
    
    def calc_sim_phi(self, init=3, dmax=200):
        R"""
        Calculates the contact probability as a function of the genomic distance from simulations for the Ideal Chromosome optimization.
        """
        phi = np.zeros(dmax)
        pmean = self.Pold/self.Nframes
        for i in range(dmax):
             phi[i] =  np.mean(np.diagonal(pmean, offset=(i+init)))
        return phi
    
    def getBijsim(self):
        R"""
        Normalizes the cross term of the Hessian by the number of frames in the simulation for the Ideal Chromosome optimization.
        """
        return self.Bij/self.Nframes
    
    def getHiCexp(self, filename):
        R"""
        Receives the experimental Hi-C map (Full dense matrix) in a text format and performs the data normalization from Hi-C frequency/counts/reads to probability.
        """
        allmap = np.loadtxt(filename)

        r=np.triu(allmap, k=1)
        r[np.isinf(r)]= 0.0
        r[np.isnan(r)]= 0.0
        r = normalize(r, axis=1, norm='max')
        rd = np.transpose(r)
        self.expHiC = r+rd + np.diag(np.ones(len(r)))
        self.expHiC[self.expHiC<self.cutoff] = 0.0

    def calc_exp_phi(self, init=3, dmax=200):
        R"""
        Calculates the contact probability as a function of the genomic distance from the experimental Hi-C for the Ideal Chromosome optimization.
        """
        phi = np.zeros(dmax)
        for i in range(dmax):
             phi[i] =  np.mean(np.diagonal(self.expHiC, offset=(i+init)))
        return phi
    
    def getlambfromfile(self, filename):
        R"""
        Receives the Lagrange multipliers of the Ideal Chromosome optimization from a text file.
        """
        aFile = open(filename,'r')
        pos = aFile.read().splitlines()
        for t in range(len(pos)):
            pos[t] = float(pos[t])
        return np.array(pos)
    
    def getLamb(self, dmax=200, exp_map='file.dense'):
        R"""
        Calculates the Lagrange multipliers for the Ideal Chromosome optimization and returns a array containing the energy values for the IC optimization step.
        """    
        self.getHiCexp(exp_map)

        
        phi_exp = self.calc_exp_phi(init=self.dinit, dmax=dmax)
        
        phi_sim = self.calc_sim_phi(init=self.dinit, dmax=dmax)
        
        gij = -phi_sim + phi_exp   # *1/beta = 1     
    
        Res = np.zeros((dmax,dmax))
        Bijmean = self.getBijsim()

        for i, j in itertools.product(range(dmax),range(dmax)):
            Res[i,j] = Bijmean[i,j] - (phi_sim[i]*phi_sim[j])
         
        invRes = sp.linalg.pinv(Res)

        erro = np.sum(np.absolute(gij))/np.sum(phi_exp)
        pear = self.getPearson()
        
                             
        with open('error_and_pearsonC_IC','a') as tf:
            tf.write("Error: %f  Pearson's Correlation: %f\n" % (erro, pear))
        
        return(np.dot(invRes,gij))
    
##########################################################################################
#### TYPES OPTIMIZATION
##########################################################################################


    def probCalculation_types(self, state, typeList):
        R"""
        Calculates the contact probability matrix and the cross term of the Hessian for the type-to-type interactions optimization.
        """    
        PiPj = np.zeros((self.n_types,self.n_types))
        n = int(self.n_types)
        p_instant = np.zeros((n,n))
        
        n_inter = self.n_inter
        
        just = {}
        ind = np.triu_indices(n)
        
        for tt in self.diff_types:
            just[tt] = ([i for i, e in enumerate(typeList) if e == tt])
        self.Pold += self.P       
        
        self.P = 0.5*(1.0 + np.tanh(self.mu*(self.r_cut - distance.cdist(state,state, 'euclidean'))))
        self.P[self.P<self.cutoff] = 0.0
        
        vec = []
        for pcount,q in enumerate(itertools.combinations_with_replacement(just.keys(), r=2)):
            nt=0
            for i, j in itertools.product(just[q[0]],just[q[1]]):

                p_instant[ind[0][pcount], ind[1][pcount]] += self.P[i,j]

                nt += 1
            p_instant[ind[0][pcount], ind[1][pcount]] = p_instant[ind[0][pcount], ind[1][pcount]]/nt #pi
            vec.append(p_instant[ind[0][pcount], ind[1][pcount]])
        vec = np.array(vec)

        PiPj = np.outer(vec,vec)
        
        self.polds_type += p_instant 
        self.Bij_type += PiPj
        self.Nframes += 1
        
    
    def calc_exp_phi_types(self, typeList):
        R"""
        Calculates the average of the contact probability for each chromatin type (compartment annotation) from the experimental Hi-C for the Types optimization.
        """
        n = int(self.n_types)
        phi = np.zeros((n,n))
        just = {}
        ind = np.triu_indices(n)
        
        for tt in self.diff_types:
            just[tt] = ([i for i, e in enumerate(typeList) if e == tt])


        for pcount,q in enumerate(itertools.combinations_with_replacement(just.keys(), r=2)):
            nt=0
            for i, j in itertools.product(just[q[0]],just[q[1]]):
                phi[ind[0][pcount], ind[1][pcount]] += self.expHiC[i,j]
                nt += 1
            phi[ind[0][pcount], ind[1][pcount]] = phi[ind[0][pcount], ind[1][pcount]]/nt

        return phi
    
    
    def calc_sim_phi_types(self):
        R"""
        Calculates the average of the contact probability for each chromatin type (compartment annotation) from simulation for the Types optimization.
        """
        return self.polds_type/self.Nframes
    
    def getPiPjsim_types(self):
        R"""
        Normalizes the cross term of the Hessian by the number of frames in the simulation for the Types optimization.
        """
        return self.Bij_type/self.Nframes
    
    def getHiCSim(self):
        R"""
        Calculates the *in silico* Hi-C map (Full dense matrix).
        """
        return self.Pold/self.Nframes
    
    def getPearson(self):
        R"""
        Calculates the Pearson's Correlation between the experimental Hi-C used as a reference for the training and the *in silico* Hi-C obtained from the optimization step.
        """
        r1 = self.getHiCSim()
        r2 = self.expHiC

        r1[np.isinf(r1)]= 0.0
        r1[np.isnan(r1)]= 0.0
        r1[r1 <= 0.001]= 0.0
        r2[np.isinf(r2)]= 0.0
        r2[np.isnan(r2)]= 0.0
        r2[r2<=0.001] = 0.0

        np.fill_diagonal(r1,0.0)
        np.fill_diagonal(r2,0.0)


        SEED = 100
        random.seed(SEED)
        a1 = np.asarray(random.sample(list(r1[np.triu_indices(np.shape(r1)[0])]),np.int(0.1*np.shape(r1)[0]*np.shape(r1)[0])))
        a1 = r1[np.triu_indices(np.shape(r1)[0])]
        random.seed(SEED)
        a2 = np.asarray(random.sample(list(r2[np.triu_indices(np.shape(r2)[0])]),np.int(0.1*np.shape(r2)[0]*np.shape(r2)[0])))
        a2 = r2[np.triu_indices(np.shape(r1)[0])]


        return(pearsonr(a1,a2)[0])
        
    def getLamb_types(self,typeList, exp_map):
        R"""
        Calculates the Lagrange multipliers of each type-to-type interaction and returns the matrix containing the energy values for the optimization step.
        """
        self.getHiCexp(exp_map)
        
        phi_exp = self.calc_exp_phi_types(typeList)
        
        phi_sim = self.calc_sim_phi_types()
        
        gij = -phi_sim + phi_exp

        PiPj_mean = self.getPiPjsim_types()
        

        ind = np.triu_indices(self.n_types)
        phi_sim_linear = []

        for pcount,q in enumerate(itertools.combinations_with_replacement(range(self.n_types), r=2)):
            phi_sim_linear.append(phi_sim[ind[0][pcount], ind[1][pcount]])
           
        phi_sim_linear = np.array(phi_sim_linear)

        Pi2_mean = np.outer(phi_sim_linear,phi_sim_linear)

        Bij_mean = PiPj_mean - Pi2_mean
    
        invBij_mean = sp.linalg.pinv(Bij_mean)

        erro = np.sum(np.absolute(gij))/np.sum(phi_exp)
        pear = self.getPearson()
           
        with open('error_and_pearsonC_types','a') as tf:
            tf.write("Error: %f  Pearson's Correlation: %f\n" % (erro, pear))
        
        ind = np.triu_indices(self.n_types)
        gij_vec = []
        for pcount,q in enumerate(itertools.combinations_with_replacement(range(self.n_types), r=2)):
            gij_vec.append(gij[ind[0][pcount], ind[1][pcount]])
        gij_vec = np.array(gij_vec)
        
        lambdas = np.matmul(invBij_mean, gij_vec)
        
        new = np.zeros((self.n_types,self.n_types))
        
        inds = np.triu_indices_from(new)
        new[inds] = lambdas
        new[(inds[1], inds[0])] = lambdas 
        
        return(new)