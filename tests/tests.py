import sys

sys.path.append('../OpenMiChroM')

from ChromDynamics import MiChroM
from Optimization import FullTraining, CustomMiChroMTraining
from CndbTools import cndbTools



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
            a.runSimBlock(2000, increment=False)
    
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
            b.probCalculation_IC(state=tl)
            b.probCalculation_types(state=tl)
        print('Getting parameters for Types and IC...')
        ty = b.getLamb(exp_map=sys.path[0] + '/training/c18_10.dense')
        ic = b.getLamb_types(exp_map=sys.path[0] + '/training/c18_10.dense')
        print('Finished')

    def testCndbTools(self):
        traj1 = cndbTools.load(filename=sys.path[0] + '/training/test_0.cndb')
        print(traj1)
        sampleA1 = cndbTools.xyz(frames=[1,100,1], beadSelection=traj1.dictChromSeq['A1'])
        sampleB1 = cndbTools.xyz(frames=[1,100,1], beadSelection=traj1.dictChromSeq['B1'])

        print("compute RG...")
        rg1 = cndbTools.compute_RG(sampleA1)

        print("compute RDF...")
        xa1, rdf_a1 = cndbTools.compute_RDF(sampleA1, radius=20, bins=200)
        xb1, rdf_b1 = cndbTools.compute_RDF(sampleB1, radius=20, bins=200)

        print("create a contact probability matrix...")
        alldata = cndbTools.xyz()
        dense = cndbTools.traj2HiC(alldata)
        print('Finished')


run = testMichrom()

run.runDefault()
run.testCustomMiChroM()
run.testCndbTools()