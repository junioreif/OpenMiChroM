import numpy as np
from ActiveMonomerModule import ActiveMonomer


class test():
    def __init__(self):
        
        sim=ActiveMonomer(
            time_step=0.01, 
            collision_rate=10.0, 
            temperature=120.0,
            name="ActiveMonomer", 
            active_corr_time=10.0, 
            act_seq=[0.,3,4],
            platform="opencl")
        
    
        sim.saveFolder('out/')
        struct=sim.create_springSpiral(ChromSeq='in/random.txt')

        sim.loadStructure(struct, center=True)

        sim.addHarmonicBonds(kfb=70.0)
        #asim.addFENEBonds(kfb=30.0)

        sim.addAngles(ka=2.0)

        #asim.addRepulsiveSoftCore(Ecut=4.0)
        sim.addSelfAvoidance(Ecut=4.0)

        # sim.addCustomTypes(mu=3.22, rc = 1.78, TypesTable='types.csv')
        # sim.addTypetoType(mu=3.22, rc = 1.78)
        #asim.addIdealChromosome(mu=3.22, rc = 1.78, dinit=3, dend=500)
        sim.addFlatBottomHarmonic( kr=30.0, n_rad=10.0)

        print([(xx,sim.forceDict[xx].getForceGroup()) for xx in sim.forceDict.keys()])

        block = 100
        n_blocks = 10

        sim.initStorage(filename="test")
        # positions=[]
        sim.runSimBlock(10, increment=False)
        sim.initVelocities(mult=0.0) #set velocities (active force) to zero
        for _ in range(n_blocks):
            sim.runSimBlock(block, increment=False)
            sim.state = sim.context.getState(getPositions=True,getEnergy=True, getForces=True, getVelocities=True)
            # positions.append(asim.state.getPositions(asNumpy=True))
            # sim.saveStructure()
            # print(sim.state.getVelocities(asNumpy=True))
        print([(xx,sim.forceDict[xx].getForceGroup()) for xx in sim.forceDict.keys()])


        # print(sim.)
        sim.storage[0].close()

        
        
run=test()


        