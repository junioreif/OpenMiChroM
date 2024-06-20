import sys

sys.path.append('../OpenMiChroM')


from ChromDynamics import MiChroM
from Optimization import FullTraining, CustomMiChroMTraining, AdamTraining
from CndbTools import cndbTools
import h5py
import numpy as np
import pandas as pd




class testMichrom():
   
    def runDefault(self):
        a = MiChroM(name="test", temperature=1.0, time_step=0.01)
        a.setup(platform="opencl", integrator="Langevin", precision='single')
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
        cndbt = cndbTools()
        traj = cndbt.load(filename=sys.path[0] + '/training/test_0.cndb')
        print(traj)
        print(traj.uniqueChromSeq)
        sampleA1 = cndbt.xyz(frames=[1,100,1], beadSelection=traj.dictChromSeq['A1'])
        sampleB1 = cndbt.xyz(frames=[1,100,1], beadSelection=traj.dictChromSeq['B1'])

        print("Computing RG...")
        rg = cndbt.compute_RG(sampleA1)

        print("Computing RDP...")
        xa1, rdp_a1 = cndbt.compute_RDP(sampleA1, radius=20.0, bins=200)
        xb1, rdp_b1 = cndbt.compute_RDP(sampleB1, radius=20.0, bins=200)

        print("Computing Chirality...")
        psi = cndbt.compute_Chirality(cndbt.xyz(frames=[1,100,1]),4)

        print("Generating the contact probability matrix...")
        alldata = cndbt.xyz()
        dense = cndbt.traj2HiC(alldata)
        
        print('Finished')
        
    def testAdamTraining(self):
        opt = AdamTraining(mu=3.22, rc = 1.78, eta=0.01, it=1)
        opt.getPars(HiC="AdamTraining/input/chr19_50k.dense")
        with h5py.File("AdamTraining/input/Pi_0.h5", 'r') as hf:
            opt.Pi += hf['Pi'][:]
            opt.NFrames += int(np.loadtxt("AdamTraining/input/Nframes_0"))
        lamb_new = opt.getLamb(Lambdas="AdamTraining/input/lambda_0")

        lamb_new.to_csv("AdamTraining/output/lambda_1", index=False)

        ff_new = pd.read_csv("AdamTraining/output/lambda_1")



run = testMichrom()

# run.runDefault()
# run.testCustomMiChroM()
# run.testCndbTools()
run.testAdamTraining()
