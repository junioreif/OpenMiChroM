from OpenMiChroM.Optimization import CustomMiChroMTraining #optimization michrom parameters module


#modules to load and plot .dense file 
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import h5py
import sys


iteraction = sys.argv[1]
inputFolder = sys.argv[2]
seq = sys.argv[3]
dense = sys.argv[4]
replicas = sys.argv[5:]



opt2 = CustomMiChroMTraining(ChromSeq=seq,
                            mu=3.22, rc = 1.78)

for replica in replicas:
    with h5py.File(replica + "/polds_type_" + str(iteraction)+".h5", 'r') as hf:
        opt2.polds_type += hf['polds_type'][:]

    with h5py.File(replica + "/Bij_type_" + str(iteraction)+".h5", 'r') as hf:
        opt2.Bij_type += hf['Bij_type'][:]

    with h5py.File(replica + "/Nframes_" + str(iteraction)+".h5", 'r') as hf:
        opt2.Nframes +=hf['Nframes'][()]

    with h5py.File(replica + "/Pold_" + str(iteraction)+".h5", 'r') as hf:
        opt2.Pold += hf['Pold'][:]
    
lambdas = opt2.getLamb_types(exp_map=dense)
print(lambdas)  

old = pd.read_csv(inputFolder + "/lambda_" + str(iteraction), sep=None, engine='python')
lambda_old = old.values
seq = old.columns
print(lambda_old)

damp = 10**-6
lambda_new = lambda_old - damp*lambdas

print(lambda_new)

#prob of A/B in sim and exp
phi_sim = opt2.calc_sim_phi_types().ravel()
phi_exp = opt2.calc_exp_phi_types().ravel()
np.savetxt('phi_sim_' + str(iteraction), phi_sim)
np.savetxt('phi_exp', phi_exp)

#plt.plot(phi_sim, label="sim")
#plt.plot(phi_exp, label="exp")
#plt.legend()

#HiC_simulate
dense_sim = opt2.getHiCSim()
np.savetxt('hic_sim_' + str(iteraction)+'.dense', dense_sim)
#plt.matshow(dense_sim, norm=mpl.colors.LogNorm(vmin=0.0001, vmax=dense_sim.max()),cmap="Reds")

#save the new lambda file
lamb = pd.DataFrame(lambda_new,columns=seq)
lamb.to_csv(inputFolder + "/lambda_" + str(int(iteraction)+1), index=False)