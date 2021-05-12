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
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import pycuda.driver as drv
#import pycuda.gpuarray as gpuarray
#import pycuda.autoinit
#import skcuda.linalg as linalg



class Agnostic:
    def __init__(self, state, expHiC, mi=1.0, rc=2.5, 
                 cutoff=0.0, reduce=False,
                 pair_h=2, c_h=0.08, pair_l=3, c_l=0.02, gpu=False 
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
        #print(self.ind)
    
            
        self.size = len(self.ind)     
        
        
        self.Pi = np.zeros(self.size)
   
        self.Prod_dist = np.zeros(self.hic_sparse.shape)
        self.PiPj = np.zeros((self.size,self.size))
        self.NFrames = 0
        #print(self.polds_type.shape, self.Bij_type.shape)
        #self.polds_type = np.zeros((self.size, self.size), dtype=np.float16)
        #self.Bij_type = np.zeros((self.n_inter,self.n_inter),dtype=np.float16)

    def appCutoff(self, pair_h, c_h, pair_l, c_l):
        N = self.hic_sparse.shape[0]
        print('initial non-zero interactions: ', self.hic_sparse.nnz)

        hic_full = self.hic_sparse.todense()
        hic_final = np.zeros(self.hic_sparse.shape)

        #high cutoff 
        values = [n for n in range(0,N,pair_h)]
        index =  [x for x in itertools.combinations_with_replacement(values, r=2)]
        for i in index:
            if (hic_full[i] > c_h):
                hic_final[i] = hic_full[i]

        print('non-zero interactions after high cutoff ({}): {}'.format( c_h, sp.sparse.csr_matrix(np.triu(hic_final, k=2)).nnz ))


        #low cutoff 
        values = [n for n in range(1,N,pair_l)]
        index =  [x for x in itertools.combinations_with_replacement(values, r=2)]
        for i in index:
            if (hic_full[i] > c_l):
                hic_final[i] = hic_full[i]
        self.hic_sparse = sp.sparse.csr_matrix(np.triu(hic_final, k=2))
        print('non-zero interactions after low cutoff ({}): {}'.format(c_l,  self.hic_sparse.nnz ))

    
    def get_indices(self, hic):
        index = sp.sparse.find(hic)
        self.rows = index[0]
        self.cols = index[1]
        self.values = index[2]
        ind = []
        for i in range(len(index[0])):
            ind.append((self.rows[i], self.cols[i]))
        return(ind)
        
    def getHiCexp(self, filename, centerRemove=False, centrange=[0,0]):
        allmap = np.loadtxt(filename)



        r=np.triu(allmap, k=1) #tirando a diagonal principal e pegando só a matriz superior
        r[np.isinf(r)]= 0.0
        r[np.isnan(r)]= 0.0
        r = normalize(r, axis=1, norm='max') #normalizando em função do maior valor
        rd = np.transpose(r) #criando a matriz triagular inferior
        self.expHiC = r+rd + np.diag(np.ones(len(r))) #somando tudo e adicionado 1 na diagonal princial
        
        if (centerRemove):
            centrome = range(centrange[0],centrange[1])
            self.expHiC[centrome,:] = 0.0
            self.expHiC[:,centrome] = 0.0
        self.expHiC[self.expHiC<self.cutoff] = 0.0
        
    def probCalc(self, state):
        
        Prob = 0.5*(1.0 + np.tanh(self.mi*(self.rc - distance.cdist(state,state, 'euclidean'))))
        #self.P[self.P<self.cutoff] = 0.0

        #calculo do pi
        Pi = []
        for i in self.ind:
            #print(i[0],i[1], Prob[i[0], i[1]])
            Pi.append(Prob[i[0], i[1]])
        Pi = np.array(Pi)
    
        if (self.gpu):
            linalg.init()
            Pi_gpu = gpuarray.to_gpu(Pi)
            PiPj_gpu = linalg.dot(Pi_gpu, Pi_gpu)
            self.PiPj += PiPj_gpu
        else:
            PiPj = np.outer(Pi,Pi)
            self.PiPj += PiPj #<PiPj>                                              
     
        #print(PiPj.shape, self.Bij_type.shape)
        #self.PiPj += PiPj #<PiPj>
        self.Prod_dist += Prob
        self.Pi += Pi
        self.NFrames += 1
        
    def getPearson(self):
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
        print(np.int(0.01*np.shape(r1)[0]*np.shape(r1)[0]))                              #0,01*137*137
        
        a1 = np.asarray(random.sample(list(r1[np.triu_indices(np.shape(r1)[0])]),np.int(0.01*np.shape(r1)[0]*np.shape(r1)[0])))
        a1 = r1[np.triu_indices(np.shape(r1)[0])]
        random.seed(SEED)
        a2 = np.asarray(random.sample(list(r2[np.triu_indices(np.shape(r2)[0])]),np.int(0.01*np.shape(r2)[0]*np.shape(r2)[0])))
        a2 = r2[np.triu_indices(np.shape(r1)[0])]


        return(pearsonr(a1,a2)[0])
        
    def getLambdas(self):
        #calculate gij = phi_exp - phi_sim
        self.phi_exp = self.values
        self.phi_sim = self.Pi/self.NFrames
        gij = self.phi_exp - self.phi_sim
        

        #calculate <Pi^2>
        Pi2_mean = np.outer(self.phi_sim,self.phi_sim)

        #calculate <PiPj>
        PiPj_mean = self.PiPj/self.NFrames


        #calculate Bij
        Bij = PiPj_mean - Pi2_mean
        
        if (self.gpu):
            Bij_gpu = gpuarray.to_gpu(Bij)
            invBij_gpu = linalg.pinv(Bij_gpu)
        else:
            invBij = sp.linalg.pinvh(Bij)

        
        #calculate lambdas
        if (self.gpu):
            gij_gpu = gpuarray.to_gpu(gij)
            lambdas_gpu = linalg.dot(invBij_gpu, gij_gpu)
            lambdas = lambdas_gpu.get()
        else:
            lambdas = np.matmul(invBij, gij)

        
        lamb_matrix = sp.sparse.csr_matrix((lambdas,(self.rows,self.cols)), shape=(self.expHiC.shape[0],self.expHiC.shape[0]))
        
        
        #calculate analyses
        self.error = (np.sum(np.absolute(gij)))/(np.sum(self.phi_exp))
        #self.Pearson = self.getPearson()
        
        return(lamb_matrix)


