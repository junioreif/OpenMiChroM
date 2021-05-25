import sys

sys.path.append('../OpenMiChroM')

from ChromDynamics import MiChroM
from Optimization import FullTraining, CustomMiChroMTraining



class testMichrom():
    def runDefault(self):
        a = MiChroM(name="test", temperature=1.0, time_step=0.01)
        a.setup(platform="cuda", integrator="Langevin", precision='single')
        a.saveFolder('output')
        myChrom = a.create_springSpiral(ChromSeq=sys.path[0]+'/chr10/chr10_beads.txt')
        a.loadStructure(myChrom, center=True)
        a.addFENEBonds(kfb=30.0) 
        a.addAngles(ka=2.0)
        a.addRepulsiveSoftCore(Ecut=4.0)
        a.addTypetoType(mu=3.22, rc=1.78)
        a.addIdealChromosome(mu=3.22, rc=1.78)
        a.addFlatBottomHarmonic(kr=5*10**-3, n_rad=10.0)

        for _ in range(10):
            a.doBlock(2000, increment=False)
    
        a.printForces()
        a.saveStructure(mode = 'ndb')
        a.saveStructure(mode = 'pdb')
        a.saveStructure(mode = 'gro')
        a.saveStructure(mode = 'xyz')

    def testCustomMiChroM(self):
        b = CustomMiChroMTraining(ChromSeq=sys.path[0] + '/training/seq_c18_10')
        assert(len(b.Bij_type) == 6)
        
        import h5py
        import numpy as np
        filename = sys.path[0] + '/training/test_0.cndb'
        mode = 'r'
        myfile = h5py.File(filename, mode)
        print("Calculating probabilities for 10 frames...")
        for i in range(1,10):
            tl = np.array(myfile[str(i)])
            b.probCalculation(state=tl)
            b.probCalculation_types(state=tl)
        print('Get Lambdas values for Types and IC...')
        ty = b.getLamb(exp_map=sys.path[0] + '/training/c18_10.dense')
        ic = b.getLamb_types(exp_map=sys.path[0] + '/training/c18_10.dense')
        print('Finished')


run = testMichrom()

#run.runDefault()
run.testCustomMiChroM()