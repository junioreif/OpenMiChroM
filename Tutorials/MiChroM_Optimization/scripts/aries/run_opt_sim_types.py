from OpenMiChroM.ChromDynamics import MiChroM 
from OpenMiChroM.Optimization import CustomMiChroMTraining 

import sys
import numpy as np
import pandas as pd
import h5py

# Parameters from the submission scritps
rep              = sys.argv[1]
sequence         = sys.argv[2]
# lambdaFile_IC = sys.argv[3] 
lambdaFile_types = sys.argv[3] 
folder           = sys.argv[4]
iteration        = sys.argv[5]
# lambdaFile_types  = "input/lambda_types"

print("folder = ", folder)

# Simulation name and platform
sim_name     = "opt"
gpu_platform = "OpenCL"

# Parameters for the crosslinking function
cons_mu = 3.22 
cons_rc = 1.78

# Parameters for the IC 
dinit=3
dend=200

# Simulation time for Collapse
block_collapse    = 1000
n_blocks_collapse = 1000

# Simulation time to feed Optimization
block_opt    = 1000
n_blocks_opt = 5000

#
# Setup MiChroM object
#
sim = MiChroM(name=sim_name, temperature=1.0, time_step=0.01)
sim.setup(platform=gpu_platform)
sim.saveFolder(folder)
mychro = sim.createSpringSpiral(ChromSeq=sequence)
sim.loadStructure(mychro, center=True)

# Adding Potentials subsection

# **Homopolymer Potentials**  
sim.addFENEBonds(kfb=30.0)
sim.addAngles(ka=2.0)
sim.addRepulsiveSoftCore(Ecut=4.0)

# **Chromosome Potentials**
sim.addCustomTypes(TypesTable=lambdaFile_types, mu=cons_mu, rc=cons_rc)
# sim.addCustomIC(mu=cons_mu, rc=cons_rc, IClist=lambdaFile_IC,
#                 dinit=dinit, dend=dend) 

#
# Collapse simulation
#
sim.addFlatBottomHarmonic(kr=5*10**-3, n_rad=8.0)

block    = block_collapse
n_blocks = n_blocks_collapse

# save initial structure and open trajectory file
sim.saveStructure(mode='ndb')
# sim.initStorage(filename=sim_name)

for _ in range(n_blocks):
    sim.runSimBlock(block, increment=True)
    # sim.saveStructure()

sim.saveStructure(mode='ndb')

#
# Optimization simulation
#

# Remove Flat initialized in Collapse
sim.removeFlatBottomHarmonic()

# Add a confinement potential with density=0.1 (volume fraction)
sim.addSphericalConfinementLJ()

# Initialize optimization object
opt = CustomMiChroMTraining(ChromSeq=sequence, TypesTable=lambdaFile_types, 
                            mu=cons_mu, rc=cons_rc, 
                            # IClist=lambdaFile_IC, dinit=dinit, dend=dend
                            )

block    = block_opt
n_blocks = n_blocks_opt

for _ in range(n_blocks):
    # perform 1 block of simulation
    sim.runSimBlock(block, increment=True)
    # sim.saveStructure()

    # feed optimization with the last chromosome configuration 
    # For types
    opt.prob_calculation_types(sim.getPositions()) 
    # For IC
    # opt.prob_calculation_IC(sim.getPositions())

# save final structure and close traj file
sim.saveStructure(mode='ndb')
# sim.storage[0].close()

# Save incremented step files
with h5py.File(sim.folder + "/Nframes_" + str(iteration) + "_" + str(rep) + ".h5", 'w') as hf:
    hf.create_dataset("Nframes",  data=opt.Nframes)

with h5py.File(sim.folder + "/Pold_" + str(iteration) + "_" + str(rep) + ".h5", 'w') as hf:
    hf.create_dataset("Pold",  data=opt.Pold)

# For Types
with h5py.File(sim.folder + "/Pold_type_" + str(iteration) + "_" + str(rep) + ".h5", 'w') as hf:
    hf.create_dataset("Pold_type",  data=opt.Pold_type)

with h5py.File(sim.folder + "/PiPj_type_" + str(iteration) + "_" + str(rep) + ".h5", 'w') as hf:
    hf.create_dataset("PiPj_type",  data=opt.PiPj_type)

# For Ideal Chromosome    
# with h5py.File(sim.folder + "/PiPj_IC_" + str(iteration) + "_" + str(rep) + ".h5", 'w') as hf:
#     hf.create_dataset("PiPj_IC",  data=opt.PiPj_IC)
