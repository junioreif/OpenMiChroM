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


class Trainer:

    ##initialize the Phage set
    '''
    This classe is used to optimize constants values for Open-Michron Potentials
    '''
 
    def __init__(self, state, TypeList=None, name='distMatrix', nHood=3, cutoff=0.0, mi=5.33, rc=1.61,lamb_size=200): 
        self.name = name
        self.size = len(state)
        self.P=np.zeros((self.size,self.size))
        self.Pold=np.zeros((self.size,self.size))
        self.r_cut = rc #1.78 #1.985907 #1.558225 #1.78       #parameter for contact function f
        self.mu  = mi  #3.22 #2.12096 #3.77805 #3.22   #parameter for contact function f
        self.Bij = np.zeros((lamb_size,lamb_size))
        self.diff_types = set(TypeList)
        self.n_types = len(self.diff_types)
        self.n_inter = int(self.n_types*(self.n_types-1)/2 + self.n_types)
        self.polds_type = np.zeros((self.n_types, self.n_types))
        self.Bij_type = np.zeros((self.n_inter,self.n_inter))
        self.Nframes = 0 
        self.dinit = nHood
        self.cutoff = cutoff
        
        
##########################################################################################
#### IDEAL CROMOSSOME OPTIMIZATION
##########################################################################################
    
    def probCalculation(self, state, dmax=200):
        #remember dinit = 3, i.e, Bij[0,1] = 3,4
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
        phi = np.zeros(dmax)
        pmean = self.Pold/self.Nframes
        for i in range(dmax):
             phi[i] =  np.mean(np.diagonal(pmean, offset=(i+init)))
        return phi
    
    def getBijsim(self):
        return self.Bij/self.Nframes
    
    
    def getHiCexp(self, filename):
        allmap = np.loadtxt(filename)



        r=np.triu(allmap, k=1) #tirando a diagonal principal e pegando só a matriz superior
        r[np.isinf(r)]= 0.0
        r[np.isnan(r)]= 0.0
        r = normalize(r, axis=1, norm='max') #normalizando em função do maior valor
        rd = np.transpose(r) #criando a matriz triagular inferior
        self.expHiC = r+rd + np.diag(np.ones(len(r))) #somando tudo e adicionado 1 na diagonal princial
        self.expHiC[self.expHiC<self.cutoff] = 0.0

    def calc_exp_phi(self, init=3, dmax=200):
        phi = np.zeros(dmax)
        
        for i in range(dmax):
             phi[i] =  np.mean(np.diagonal(self.expHiC, offset=(i+init)))
        return phi
    
    
    def getlambfromfile(self, filename):
        aFile = open(filename,'r')
        pos = aFile.read().splitlines()
        for t in range(len(pos)):
            pos[t] = float(pos[t])
        return np.array(pos)
    
    def getLamb(self, dmax=200, exp_map='file.dense'):
        
   
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
            tf.write("Error: %f  Pearson Correlation: %f\n" % (erro, pear))
        
        
        
        return(np.dot(invRes,gij))
    
##########################################################################################
#### TYPES OPTIMIZATION
##########################################################################################


    def probCalculation_types(self, state, typeList):
        
        PiPj = np.zeros((self.n_types,self.n_types))
        n = int(self.n_types)
        p_instant = np.zeros((n,n))
        
        n_inter = self.n_inter
        
        just = {}
        ind = np.triu_indices(n)
        
        for tt in self.diff_types:
            just[tt] = ([i for i, e in enumerate(typeList) if e == tt])
        #print("just\n",just)
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
        self.Bij_type += PiPj #<PiPj>
        self.Nframes += 1
    
    def calc_exp_phi_types(self, typeList):

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
        
        #phi = phi + np.triu(phi, k=1).T

        return phi
    
    
    def calc_sim_phi_types(self):
        return self.polds_type/self.Nframes
    
    
    def getPiPjsim_types(self):
        return self.Bij_type/self.Nframes
    
    def getHiCSim(self):
        return self.Pold/self.Nframes
    
    def getPearson(self):
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
        
        self.getHiCexp(exp_map) #ok

        
        phi_exp = self.calc_exp_phi_types(typeList)  #<pi_exp>

        
        phi_sim = self.calc_sim_phi_types() #ok <pi_sim>

        
        gij = -phi_sim + phi_exp   # *1/beta = 1    #<pi_sim> - <pi_exp>

        PiPj_mean = self.getPiPjsim_types() #<pipj> for n = 3, matrix 6x6
        

        ind = np.triu_indices(self.n_types)
        phi_sim_linear = []

        for pcount,q in enumerate(itertools.combinations_with_replacement(range(self.n_types), r=2)):
            phi_sim_linear.append(phi_sim[ind[0][pcount], ind[1][pcount]])
            #print("Pi2_mean\n",pcount, phi_sim[ind[0][pcount], ind[1][pcount]])
        phi_sim_linear = np.array(phi_sim_linear)


        
        
        Pi2_mean = np.outer(phi_sim_linear,phi_sim_linear) # for n = 3, matrix 6x6


        
        Bij_mean = PiPj_mean - Pi2_mean
        

        invBij_mean = sp.linalg.pinv(Bij_mean)

        
        erro = np.sum(np.absolute(gij))/np.sum(phi_exp)
        pear = self.getPearson()
        

                             
        with open('error_and_pearsonC_types','a') as tf:
            tf.write("Error: %f  Pearson Correlation: %f\n" % (erro, pear))
        
        
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
