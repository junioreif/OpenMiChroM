import sys

sys.path.append('../OpenMiChroM')

from ChromDynamics import MiChroM
from Optimization import FullTraining, CustomMiChroMTraining



class testMichrom():
    def runDefault(self):
        a = MiChroM(name="test", temperature=120, timestep=0.01)
        a.setup(platform="cuda", integrator="Langevin", precision='single')
        a.saveFolder('output')
        mypol = a.create_springSpiral(type_list=sys.path[0]+'/chr10/chr10_beads.txt')
        a.load(mypol, center=True)
        a.addFENEBonds(k=30.0) 
        a.addAngles(k=2.0)
        a.addRepulsiveSoftCore(Ecut=4.0)
        a.addTypetoType(mi=3.22, rc=1.78)
        a.addIdealChromosome(mi=3.22, rc=1.78)
        a.addSphericalConfinement(density=0.1, k=10)

        for _ in range(10):
            a.doBlock(100, increment=False)

run = testMichrom()

run.runDefault()
