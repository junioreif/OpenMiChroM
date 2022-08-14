import numpy as np
import simtk.unit as unit
from openmmtools.constants import kB
from openmmtools.integrators import ThermostatedIntegrator
import sys 
sys.path.append('/home/sb95/ActiveOpenMiChroM/OpenMiChroM/')
from ChromDynamics import MiChroM

class PersistentBrownianIntegrator(ThermostatedIntegrator):
    def __init__(self,
                timestep=0.001, 
                temperature=120.0,
                collision_rate=0.1,
                persistent_time=10.0,
                constraint_tolerance=1e-8,
                 ):

        # Create a new CustomIntegrator
        super(PersistentBrownianIntegrator, self).__init__(temperature, timestep)
        #parameters
        kbT = kB * temperature
        
        #add globall variables
        self.addGlobalVariable("kbT", kbT)
        self.addGlobalVariable("g", collision_rate)
        self.addGlobalVariable("Ta", persistent_time)
        self.setConstraintTolerance(constraint_tolerance)

        self.addPerDofVariable("x1", 0) # for constraints

        self.addUpdateContextState()
        #update velocities. note velocities are active and not derived from positions.
        self.addComputePerDof("v", "(exp(- dt / Ta ) * v) + ((sqrt(1 - exp( - 2 * dt / Ta)) * f0 / g) * gaussian)")
        self.addConstrainVelocities()

        self.addComputePerDof("x", "x + (v * dt) + (dt * f / g) + (sqrt(2 * (kbT / g) * dt) * gaussian)")
        
        #remove the contribution from force group 0: the persistent force, which is already taken into account in the v*dt term
        self.addComputePerDof("x", "x - (dt  * f0 / g)")

        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained

class ActiveMonomer(MiChroM):
    
    def __init__(
        self, 
        time_step=0.001, 
        collision_rate=0.1, 
        temperature=120.0,
        name="ActiveMonomer", 
        active_corr_time=10.0, 
        act_seq=None,
        platform="opencl"):

        super(ActiveMonomer, self).__init__(name=name, 
            velocity_reinitialize=False,
            temperature=temperature,
            collision_rate=collision_rate,)
    
        self.timestep=time_step
        self.name=name
        self.collisionRate=collision_rate
        self.temperature=temperature
        self.activeCorrTime=active_corr_time

        #set up integrator        
        integrator=PersistentBrownianIntegrator(
            timestep=self.timestep, 
            collision_rate=self.collisionRate,
            temperature=self.temperature,
            persistent_time=self.activeCorrTime,
            )
        """
        integrator=CustomBrownianIntegrator(temperature=self.temperature * unit.kelvin,
                                            timestep=self.timestep * unit.picoseconds,
                                            collision_rate=self.collisionRate / unit.picoseconds,
                                            noise_corr=self.activeCorrTime * unit.picoseconds)
        """
        self.setup(platform=platform,integrator=integrator,)  

        #define active force group
        act_force=self.mm.CustomExternalForce(" - f_act * (x + y + z)")
        act_force.addPerParticleParameter('f_act')
        self.forceDict["ActiveForce"]=act_force
        self.forceDict["ActiveForce"].setForceGroup(0)
        
        try:
            act_seq=np.asfarray(act_seq)
            for bead_id, Fval in enumerate(act_seq):
                if Fval>0:
                    self.forceDict["ActiveForce"].addParticle(int(bead_id),[Fval])

            print('\n\
            ==================================\n\
            ActiveMonomer now set up.\n\
            Active correlation time: {}\n\
            Total number of active particles: {}\n\
            ==================================\n'.format(active_corr_time, self.forceDict["ActiveForce"].getNumParticles()))
        
        except (ValueError,):
            print('Critical Error! Active force not added.')



    def addHarmonicBonds(self, kfb=30.0):

        for start, end, isRing in self.chains:
            for j in range(start, end):
                self.addHarmonicBond_ij(j, j + 1, kfb=kfb)
                self.bondsForException.append((j, j + 1))

            if isRing:
                self.addHarmonicBond_ij(start, end, distance=1, kfb=kfb)
                self.bondsForException.append((start, end ))

        self.metadata["HarmonicBond"] = repr({"kfb": kfb})
        
    def _initHarmonicBond(self, kfb=30,r0=1.0):
        
        if "HarmonicBond" not in list(self.forceDict.keys()):
            force = ("0.5 * kfb * (r-r0)*(r-r0)")
            bondforceGr = self.mm.CustomBondForce(force)
            bondforceGr.addGlobalParameter("kfb", kfb)
            bondforceGr.addGlobalParameter("r0", r0) 
                
            self.forceDict["HarmonicBond"] = bondforceGr
        
    def addHarmonicBond_ij(self, i, j, distance=None, kfb=30):
        
        if (i >= self.N) or (j >= self.N):
            raise ValueError("\n Cannot add a bond between beads  %d,%d that are beyond the chromosome length %d" % (i, j, self.N))
        if distance is None:
            distance = self.length_scale
        else:
            distance = self.length_scale * distance
        distance = float(distance)

        self._initHarmonicBond(kfb=kfb)
        self.forceDict["HarmonicBond"].addBond(int(i), int(j), [])


    def addSelfAvoidance(self, Ecut=4.0, k_rep=20.0, r0=0.9):

        Ecut = Ecut*self.Epsilon
        
        repul_energy = ("0.5 * Ecut * (1.0 + tanh(1.0 - (k_rep * (r - r0))))")
        
        self.forceDict["SelfAvoidance"] = self.mm.CustomNonbondedForce(repul_energy)
        repulforceGr = self.forceDict["SelfAvoidance"]
        repulforceGr.addGlobalParameter('Ecut', Ecut)
        repulforceGr.addGlobalParameter('r0', r0)
        repulforceGr.addGlobalParameter('k_rep', k_rep)
        repulforceGr.setCutoffDistance(3.0)

        for _ in range(self.N):
            repulforceGr.addParticle(())
