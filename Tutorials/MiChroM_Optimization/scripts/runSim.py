from OpenMiChroM.ChromDynamics import MiChroM #Open-MiChrom simualtion module
from OpenMiChroM.Optimization import CustomMiChroMTraining #optimization michrom parameters module

import sys
import numpy as np
import pandas as pd
import h5py

rep = sys.argv[1]
seqFile = sys.argv[2]
lambdaFile = sys.argv[3] 
folder = sys.argv[4]

sim = MiChroM(name='opt_chr10_100K',temperature=1.0, time_step=0.01)
sim.setup(platform="CUDA")
sim.saveFolder(folder)

mychro = sim.create_springSpiral(ChromSeq=seqFile)

sim.loadStructure(mychro, center=True)

# **Homopolymer Potentials**  
sim.addFENEBonds(kfb=30.0)
sim.addAngles(ka=2.0)
sim.addRepulsiveSoftCore(Ecut=4.0)

# **Chromosome Potentials**
sim.addCustomTypes(mu=3.22, rc = 1.78, TypesTable=lambdaFile)

sim.addFlatBottomHarmonic( kr=5*10**-3, n_rad=8.0)

block = 5*10**2 
n_blocks = 10**3

for _ in range(n_blocks):
    sim.runSimBlock(block, increment=False)


opt = CustomMiChroMTraining(ChromSeq=seqFile,
                            mu=3.22, rc = 1.78)

block = 1000
n_blocks = 5000

for _ in range(n_blocks):
    sim.runSimBlock(block, increment=True) #perform 1 block of simulation
    opt.probCalculation_types(sim.getPositions()) #feed the optimization with the last position 



with h5py.File(sim.folder + "/polds_type_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("polds_type",  data=opt.polds_type)

with h5py.File(sim.folder + "/Bij_type_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("Bij_type",  data=opt.Bij_type)

with h5py.File(sim.folder + "/Nframes_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("Nframes",  data=opt.Nframes)

with h5py.File(sim.folder + "/Pold_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("Pold",  data=opt.Pold)
