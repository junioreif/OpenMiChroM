from OpenMiChroM.Optimization import CustomMiChroMTraining

import numpy as np
import pandas as pd
import h5py
import hdf5plugin  
import sys
import os

# Parameters from the submission scritps
iteration    = sys.argv[1]
lambdaFolder = sys.argv[2]
seq          = sys.argv[3]
dense        = sys.argv[4]
replicas     = sys.argv[5:]
# lambdaFile_types  = "input/lambda_types"

lambda_old_file = lambdaFolder + "/lambda_" + str(iteration)
lambdaFile_types = lambda_old_file

# Parameters for the crosslinking function
cons_mu = 3.22 
cons_rc = 1.78

# Parameters for the IC 
dinit = 3
dend = 200

# Parameter for lambda equation
damp = 3e-7

opt2 = CustomMiChroMTraining(ChromSeq=seq, TypesTable=lambdaFile_types, 
                             mu=cons_mu, rc=cons_rc,
                            #  IClist=lambda_old_file, 
                            #  dinit=dinit, dend=dend
                             )

for replica in replicas:

    print("Reading replica ", replica)

    with h5py.File(replica + "/Nframes_" + str(iteration) + "_" + str(replica.split('_')[-1])+".h5", 'r') as hf:
        opt2.Nframes += hf['Nframes'][()]

    with h5py.File(replica + "/Pold_" + str(iteration) + "_" + str(replica.split('_')[-1])+".h5", 'r') as hf:
        opt2.Pold += hf['Pold'][:]

    # For Types
    with h5py.File(replica + "/Pold_type_" + str(iteration) + "_" + str(replica.split('_')[-1])+".h5", 'r') as hf:
        opt2.Pold_type += hf['Pold_type'][:]

    with h5py.File(replica + "/PiPj_type_" + str(iteration) + "_" + str(replica.split('_')[-1])+".h5", 'r') as hf:
        opt2.PiPj_type += hf['PiPj_type'][:]
    
    # For Ideal Chromosome
    # with h5py.File(replica + "/PiPj_IC_" + str(iteration) + "_" + str(replica.split('_')[-1])+".h5", 'r') as hf:
    #     opt2.PiPj_IC += hf['PiPj_IC'][:]

# For types    
lambdas = opt2.get_lambdas_types(exp_map=dense, damp=damp, write_error=True)
# For IC
# lambdas = opt2.get_lambdas_IC(exp_map=dense, damp=damp, write_error=True)

# Save the new lambda file
# For Types
lambdas.to_csv(lambdaFolder + "/lambda_" + str(int(iteration)+1), index=False)
# For IC
# np.savetxt(lambdaFolder + "/lambda_" + str(int(iteration)+1), lambdas)

print(lambdas)

#prob of A/B in sim and exp
# For Types
# phi_sim = opt2.calc_phi_sim_types().ravel()
# phi_exp = opt2.calc_phi_exp_types().ravel()
# For IC
phi_sim = opt2.calc_phi_sim_types().ravel()
phi_exp = opt2.calc_phi_exp_types().ravel()

# Create directories for saving
for f in ["phi", "hic"]:
    if not os.path.exists(f):
        os.mkdir(f)

np.savetxt('phi/phi_sim_' + str(iteration), phi_sim)
np.savetxt('phi/phi_exp', phi_exp)

#HiC_simulate
dense_sim = opt2.get_HiC_sim()
np.savetxt('hic/hic_sim_' + str(iteration)+'.dense', dense_sim)