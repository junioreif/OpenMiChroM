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

sim = MiChroM(name='opt_chr10_100K', temperature=1.0, timeStep=0.01)
sim.setup(platform="CUDA")
sim.saveFolder(folder)

mychro = sim.createSpringSpiral(ChromSeq=seqFile)

sim.loadStructure(mychro, center=True)

# **Homopolymer Potentials**  
sim.addFENEBonds(kFb=30.0)
sim.addAngles(kA=2.0)
sim.addRepulsiveSoftCore(eCut=4.0)

# **Chromosome Potentials**
sim.addCustomTypes(mu=3.22, rc = 1.78, TypesTable=lambdaFile)

sim.addFlatBottomHarmonic(kR=5*10**-3, nRad=8.0)
sim.createSimulation()

block = 5*10**2 
n_blocks = 10**3

for _ in range(n_blocks):
    sim.run(nsteps=block, report=False, blockSize=block)


opt = CustomMiChroMTraining(ChromSeq=seqFile,
                            mu=3.22, rc = 1.78)

block = 1000
n_blocks = 5000

for _ in range(n_blocks):
    sim.run(nsteps=block, report=False, blockSize=block)
    opt.prob_calculation_types(sim.getPositions())



with h5py.File(sim.folder + "/Pold_type_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("Pold_type", data=opt.Pold_type)

with h5py.File(sim.folder + "/PiPj_type_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("PiPj_type", data=opt.PiPj_type)

with h5py.File(sim.folder + "/Nframes_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("Nframes",  data=opt.Nframes)

with h5py.File(sim.folder + "/Pold_" + str(rep)+".h5", 'w') as hf:
    hf.create_dataset("Pold",  data=opt.Pold)
