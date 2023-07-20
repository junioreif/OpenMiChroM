# Copyright (c) 2020-2023 The Center for Theoretical Biological Physics (CTBP) - Rice University
# This file is from the Open-MiChroM project, released under the MIT License. 

R"""  
The :class:`~.ChromDynamics` classes perform chromatin dynamics based on the compartment annotations sequence of chromosomes. The simulations can be performed either using the default parameters of MiChroM (Minimal Chromatin Model) or using custom values for the type-to-type and Ideal Chromosome parameters..
"""

# with OpenMM 7.7.0, the import calls have changed. So, try both, if needed
try:
    try:
        # >=7.7.0
        from openmm.app import *
        import openmm as openmm
        import openmm.unit as units
    except:
        # earlier
        print('Unable to load OpenMM as \'openmm\'. Will try the older way \'simtk.openmm\'')
        from simtk.openmm.app import *
        import simtk.openmm as openmm
        import simtk.unit as units  
except:
    print('Failed to load OpenMM. Check your configuration.')


from sys import stdout
import numpy as np
from six import string_types
import os
import time
import random
import h5py
import pandas as pd
from pathlib import Path
import types


class MiChroM:
    R"""
    The :class:`~.MiChroM` class performs chromatin dynamics employing the default MiChroM energy function parameters for the type-to-type and Ideal Chromosome interactions.
    
    Details about the MiChroM (Minimal Chromatin Model) energy function and the default parameters are decribed in "Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173."
    
    
    The :class:`~.MiChroM` sets the environment to start the chromatin dynamics simulations.
    
    Args:
        time_step (float, required):
            Simulation time step in units of :math:`\tau`. (Default value = 0.01).
        collision_rate (float, required):
            Friction/Damping constant in units of reciprocal time (:math:`1/\tau`). (Default value = 0.1).
        temperature (float, required):
            Temperature in reduced units. (Default value = 1.0).
        verbose (bool, optional):
            Whether to output the information in the screen during the simulation. (Default value: :code:`False`). 
        velocity_reinitialize (bool, optional):
            Reset/Reinitialize velocities if :math:`E_{kin}` is greater than 5.0. (Default value: :code:`True`). 
        name (str):
            Name used in the output files. (Default value: *Chromosome*). 
        length_scale (float, required):
            Length scale used in the distances of the system in units of reduced length :math:`\sigma`. (Default value = 1.0).
        mass_scale (float, required):
            Mass scale used in units of :math:`\mu`. (Default value = 1.0).
    """
    def __init__(
        self, time_step=0.01, collision_rate=0.1, temperature=1.0,
        verbose=False,
        velocity_reinitialize=True,
        name="Chromosome",
        length_scale=1.0,
        mass_scale=1.0):
            self.name = name
            self.timestep = time_step
            self.collisionRate = collision_rate
            self.temperature = temperature / 0.008314
            self.verbose = verbose
            self.velocityReinitialize = velocity_reinitialize
            self.loaded = False
            self.forcesApplied = False
            self.folder = "."
            self.metadata = {}
            self.length_scale = length_scale
            self.mass_scale = mass_scale
            self.eKcritical = 50000000
            self.nm = units.meter * 1e-9
            self.Sigma = 1.0
            self.Epsilon = 1.0
            #####################        A1         A2        B1        B2        B3        B4       NA   
            self.inter_Chrom_types =[-0.268028,-0.274604,-0.262513,-0.258880,-0.266760,-0.266760,-0.225646, #A1
                                     -0.274604,-0.299261,-0.286952,-0.281154,-0.301320,-0.301320,-0.245080, #A2
                                     -0.262513,-0.286952,-0.342020,-0.321726,-0.336630,-0.336630,-0.209919, #B1
                                     -0.258880,-0.281154,-0.321726,-0.330443,-0.329350,-0.329350,-0.282536, #B2
                                     -0.266760,-0.301320,-0.336630,-0.329350,-0.341230,-0.341230,-0.349490, #B3
                                     -0.266760,-0.301320,-0.336630,-0.329350,-0.341230,-0.341230,-0.349490, #B4
                                     -0.225646,-0.245080,-0.209919,-0.282536,-0.349490,-0.349490,-0.255994] #NA
            self.printHeader()
            

    def setup(self, platform="CUDA", PBC=False, PBCbox=None, GPU="default",
              integrator="langevin", errorTol=None, precision="mixed",deviceIndex="0"):
        
        R"""Sets up the parameters of the simulation OpenMM platform.

        Args:

            platform (str, optional):
                Platform to use in the simulations. Opitions are *CUDA*, *OpenCL*, *HIP*, *CPU*, *Reference*. (Default value: *CUDA*). 

            PBC (bool, optional)
                Whether to use periodic boundary conditions. (Default value: :code:`False`). 

            PBCbox ([float,float,float], optional):
                Define size of the bounding box for PBC. (Default value: :code:`None`).

            GPU ( :math:`0` or :math:`1`, optional):
                Switch to another GPU. Machines with one GPU automatically select the right GPU. Machines with two or more GPUs select GPU that is less used.

            integrator (str):
                Integrator to use in the simulations. Options are *langevin*,  *variableLangevin*, *verlet*, *variableVerlet* and, *brownian*. (Default value: *langevin*).
            verbose (bool, optional):
                Whether to output the information in the screen during the simulation. (Default value: :code:`False`).
            deviceIndex (str, optional):
                Set of Platform device index IDs. Ex: 0,1,2 for the system to use the devices 0, 1 and 2. (Use only when GPU != default)

            errorTol (float, required if **integrator** = *variableLangevin*):
                Error tolerance parameter for *variableLangevin* integrator.
        """

        self.step = 0
        if PBC == True:
            self.metadata["PBC"] = True

        precision = precision.lower()
        if precision not in ["mixed", "single", "double"]:
            raise ValueError("Precision must be mixed, single or double")

        self.kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA  
        self.kT = self.kB * self.temperature  
        self.mass = 10.0 * units.amu * self.mass_scale
        self.bondsForException = []
        self.mm = openmm
        self.system = self.mm.System()
        self.PBC = PBC

        if self.PBC == True:  
            if PBCbox is None:  
                data = self.getPositions()
                data -= np.min(data, axis=0)

                datasize = 1.1 * (2 + (np.max(self.getPositions(), axis=0) - np.min(self.getPositions(), axis=0)))

                self.SolventGridSize = (datasize / 1.1) - 2
                print("density is ", self.N / (datasize[0]
                    * datasize[1] * datasize[2]))
            else:
                PBCbox = np.array(PBCbox)
                datasize = PBCbox

            self.metadata["PBCbox"] = PBCbox
            self.system.setDefaultPeriodicBoxVectors([datasize[0], 0.,
                0.], [0., datasize[1], 0.], [0., 0., datasize[2]])
            self.BoxSizeReal = datasize

        self.GPU = str(GPU)
        properties = {}
        if self.GPU.lower() != "default":
            properties["DeviceIndex"] = deviceIndex
            properties["Precision"] = precision
        self.properties = properties

        if platform.lower() == "opencl":
            platformObject = self.mm.Platform.getPlatformByName('OpenCL')

        elif platform.lower() == "reference":
            platformObject = self.mm.Platform.getPlatformByName('Reference')

        elif platform.lower() == "cuda":
            platformObject = self.mm.Platform.getPlatformByName('CUDA')

        elif platform.lower() == "cpu":
            platformObject = self.mm.Platform.getPlatformByName('CPU')

        elif platform.lower() == "hip":
            platformObject = self.mm.Platform.getPlatformByName('HIP')

        else:
            self.exit("\n!!!! Unknown platform !!!!\n")
        self.platform = platformObject

        self.forceDict = {}

        self.integrator_type = integrator
        if isinstance(integrator, string_types):
            integrator = str(integrator)
            if integrator.lower() == "langevin":
                self.integrator = self.mm.LangevinIntegrator(self.temperature,
                    self.collisionRate, self.timestep)
            elif integrator.lower() == "variablelangevin":
                self.integrator = self.mm.VariableLangevinIntegrator(self.temperature,
                    self.collisionRate, errorTol)
            elif integrator.lower() == "verlet":
                self.integrator = self.mm.VariableVerletIntegrator(self.timestep)
            elif integrator.lower() == "variableverlet":
                self.integrator = self.mm.VariableVerletIntegrator(errorTol)

            elif integrator.lower() == 'brownian':
                self.integrator = self.mm.BrownianIntegrator(self.temperature,
                    self.collisionRate, self.timestep)
            else:
                print ('please select from "langevin", "variablelangevin", '
                       '"verlet", "variableVerlet", '
                       '"brownian" or provide an integrator object')
        else:
            self.integrator = integrator
            self.integrator_type = "UserDefined"
            
    def saveFolder(self, folder):
        
        R"""Sets the folder path to save data.

        Args:

            folder (str, optional):
                Folder path to save the simulation data. If the folder path does not exist, the function will create the directory.
        """
    
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        self.folder = folder
        
    def loadStructure(self, filename,center=True,masses=None):
 
        R"""Loads the 3D position of each bead of the chromosome polymer in the OpenMM system platform.

        Args:

            center (bool, optional):
                Whether to move the center of mass of the chromosome to the 3D position ``[0, 0, 0]`` before starting the simulation. (Default value: :code:`True`).
            masses (array, optional):
                Masses of each chromosome bead measured in units of :math:`\mu`. (Default value: :code:`None`).
        """
    
        data = filename

        data = np.asarray(data, float)

        if len(data) == 3:
            data = np.transpose(data)
        if len(data[0]) != 3:
            self._exitProgram("Wrong file format")
        if np.isnan(data).any():
            self._exitProgram("\n!!!! The file contains NAN's !!!!\n")

        if center is True:
            av = np.mean(data, 0)
            data -= av

        if center == "zero":
            minvalue = np.min(data, 0)
            data -= minvalue

        self.setPositions(data)
       
        if masses == None:
            self.masses = [1. for _ in range(self.N)]
        else:
            self.masses = masses
            
        if not hasattr(self, "chains"):
            self.setChains()
            
    def setChains(self, chains=[(0, None, 0)]):
        
        R"""Sets configuration of the chains in the system. This information is later used for adding Bonds and Angles of the Homopolymer potential.

        Args:

            chains (list of tuples, optional):
                The list of chains in the format [(start, end, isRing)]. isRing is a boolean whether the chromosome chain is circular or not (Used to simulate bacteria genome, for example). The particle range should be semi-open, i.e., a chain  :math:`(0,3,0)` links the particles :math:`0`, :math:`1`, and :math:`2`. If :code:`bool(isRing)` is :code:`True` , the first and last particles of the chain are linked, forming a ring. The default value links all particles of the system into one chain. (Default value: :code:`[(0, None, 0)]`).
        """

        self.chains = [i for i in chains]  
        for i in range(len(self.chains)):
            start, end, isRing = self.chains[i]
            self.chains[i] = (start, end, isRing)
            
    def setPositions(self, beadsPos , random_offset = 1e-5):
        
        R"""Sets the 3D position of each bead of the chromosome polymer in the OpenMM system platform.

        Args:

            beadsPos (:math:`(N, 3)` :class:`numpy.ndarray`):
                Array of XYZ positions for each bead (locus) in the polymer model.
            random_offset (float, optional):
                A small increment in the positions to avoid numeral instability and guarantee that a *float* parameter will be used. (Default value = 1e-5).       
        """
        
        data = np.asarray(beadsPos, dtype="float")
        if random_offset:
            data = data + (np.random.random(data.shape) * 2 - 1) * random_offset
        
        self.data = units.Quantity(data, self.nm)
        self.N = len(self.data)
        if hasattr(self, "context"):
            self.initPositions()
            
    def getPositions(self):
        R"""
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
        """
        
        return np.asarray(self.data / self.nm, dtype=np.float32)

    def getVelocities(self):
        R"""
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of velocities.
        """

        state = self.context.getState(getVelocities=True)
        vel = state.getVelocities()

        return np.asarray(vel / (self.nm / units.picosecond ), dtype=np.float32)

        
    def randomizePositions(self):
        R"""
        Runs automatically to offset the positions if it is an integer (int) variable.
        """
        data = self.getPositions()
        data = data + np.random.randn(*data.shape) * 0.0001
        self.setPositions(data)
        

    def getLoops(self, looplists):
        R"""
        Get the loop position (CTFC anchor points) for each chromosome.
        
        .. note:: For Multi-chain simulations, the ordering of the loop list files is important! The order of the files should be the same as used in the other functions.

        Args:

            looplists (list[str]): 
                List with the names of the files containing loop information. Each file should be a two-column text file containing the index *i* and *j* of the loci pairs that forms the loop anchors.
        """
        self.loopPosition = []
        for file, chain in zip(looplists,self.chains):
            aFile = open(file,'r')
            pos = aFile.read().splitlines()
            m = int(chain[0])
            for t in range(len(pos)):
                pos[t] = pos[t].split()
                pos[t][0] = int(pos[t][0]) +m
                pos[t][1] = int(pos[t][1]) +m
                self.loopPosition.append(pos[t])
                
    
    ##============================
    ##      FORCES          
    ##============================

    def addCylindricalConfinement(self, r_conf=5.0, z_conf=10.0, kr=30.0):
        cyl_conf_energy="step(r_xy-r_cyn) * 0.5 * k_cyn * (r_xy-r_cyn)^2 + step(z^2-zconf^2) * 0.5 * k_cyn * (z-zconf)^2; r_xy=sqrt(x*x+y*y)"
        
        cyl_conf_fg = self.mm.CustomExternalForce(cyl_conf_energy)
        cyl_conf_fg.addGlobalParameter('r_cyn', r_conf)
        cyl_conf_fg.addGlobalParameter('k_cyn', kr)
        cyl_conf_fg.addGlobalParameter('zconf', z_conf)
        
        self.forceDict["CylindricalConfinement"]=cyl_conf_fg

        for i in range(self.N):
            self.forceDict["CylindricalConfinement"].addParticle(i, [])
    
    def addFlatBottomHarmonic(self, kr=5*10**-3, n_rad=10.0):
        
        R"""
        Sets a Flat-Bottom Harmonic potential to collapse the chromosome chain inside the nucleus wall. The potential is defined as: :math:`step(r-r0) * (kr/2)*(r-r0)^2`.

        Args:

            kr (float, required):
                Spring constant. (Default value = 5e-3). 
            n_rad (float, required):
                Nucleus wall radius in units of :math:`\sigma`. (Default value = 10.0).  
        """

        restraintForce = self.mm.CustomExternalForce("step(r-r_res) * 0.5 * kr * (r-r_res)^2; r=sqrt(x*x+y*y+z*z)")
        restraintForce.addGlobalParameter('r_res', n_rad)
        restraintForce.addGlobalParameter('kr', kr)
        
        for i in range(self.N):
            restraintForce.addParticle(i, [])
            
        self.forceDict["FlatBottomHarmonic"] = restraintForce
    
    def addSphericalConfinementLJ(self, r="density", density=0.1):
                                
        R"""
        Sets the nucleus wall potential according to MiChroM Energy function. The confinement potential describes the interaction between the chromosome and a spherical wall.

        Args:

            r (float or str="density", optional):
                Radius of the nucleus wall. If **r="density"** requires a **density** value.
            density (float, required if **r="density"**):
                Density of the chromosome beads inside the nucleus. (Default value = 0.1).  
        """        

        spherForce = self.mm.CustomExternalForce("(4 * GROSe * ((GROSs/r)^12 - (GROSs/r)^6) + GROSe) * step(GROScut - r);"
                                                 "r= R - sqrt(x^2 + y^2 + z^2) ")
            
        self.forceDict["SphericalConfinementLJ"] = spherForce

        for i in range(self.N):
            spherForce.addParticle(i, [])
        if r == "density":
            r = (3 * self.N / (4 * 3.141592 * density)) ** (1 / 3.)

        self.sphericalConfinementRadius = r

        spherForce.addGlobalParameter('R', r)
        spherForce.addGlobalParameter('GROSe', 1.0)
        spherForce.addGlobalParameter('GROSs', 1.0)
        spherForce.addGlobalParameter("GROScut", 2.**(1./6.))
        
        return r

        
    def addFENEBonds(self, kfb=30.0):
        
        R"""
        Adds FENE (Finite Extensible Nonlinear Elastic) bonds between neighbor loci :math:`i` and :math:`i+1` according to "Halverson, J.D., Lee, W.B., Grest, G.S., Grosberg, A.Y. and Kremer, K., 2011. Molecular dynamics simulation study of nonconcatenated ring polymers in a melt. I. Statics. The Journal of chemical physics, 134(20), p.204904".

        Args:

            kfb (float, required):
                Bond coefficient. (Default value = 30.0).
          """

        for start, end, isRing in self.chains:
            for j in range(start, end):
                self.addBond(j, j + 1, kfb=kfb)
                self.bondsForException.append((j, j + 1))

            if isRing:
                self.addBond(start, end, distance=1, kfb=kfb)
                self.bondsForException.append((start, end ))

        self.metadata["FENEBond"] = repr({"kfb": kfb})
        

    def _initFENEBond(self, kfb=30):
        R"""
        Internal function that inits FENE bond force.
        """
        if "FENEBond" not in list(self.forceDict.keys()):
            force = ("- 0.5 * kfb * r0 * r0 * log(1-(r/r0)*(r/r0)) + (4 * e * ((s/r)^12 - (s/r)^6) + e) * step(cut - r)")
            bondforceGr = self.mm.CustomBondForce(force)
            bondforceGr.addGlobalParameter("kfb", kfb)
            bondforceGr.addGlobalParameter("r0", 1.5) 
            bondforceGr.addGlobalParameter('e', 1.0)
            bondforceGr.addGlobalParameter('s', 1.0)
            bondforceGr.addGlobalParameter("cut", 2.**(1./6.))
                
            self.forceDict["FENEBond"] = bondforceGr
        

    def addBond(self, i, j, distance=None, kfb=30):
        
        R"""
        Adds bonds between loci :math:`i` and :math:`j` 

        Args:

            kfb (float, required):
                Bond coefficient. (Default value = 30.0).
            i (int, required):
                Locus index **i**.
            j (int, required):
                Locus index **j**
          """

        if (i >= self.N) or (j >= self.N):
            raise ValueError("\n Cannot add a bond between beads  %d,%d that are beyond the chromosome length %d" % (i, j, self.N))
        if distance is None:
            distance = self.length_scale
        else:
            distance = self.length_scale * distance
        distance = float(distance)

        self._initFENEBond(kfb=kfb)
        self.forceDict["FENEBond"].addBond(int(i), int(j), [])

            
    def addAngles(self, ka=2.0):
        
        R"""
        Adds an angular potential between bonds connecting beads :math:`i − 1, i` and :math:`i, i + 1` according to "Halverson, J.D., Lee, W.B., Grest, G.S., Grosberg, A.Y. and Kremer, K., 2011. Molecular dynamics simulation study of nonconcatenated ring polymers in a melt. I. Statics. The Journal of chemical physics, 134(20), p.204904".
        
        Args:

            ka (float, required):
                Angle potential coefficient. (Default value = 2.0).
        """
        
        try:
            ka[0]
        except:
            ka = np.zeros(self.N, float) + ka
        angles = self.mm.CustomAngleForce(
            "ka *  (1 - cos(theta - 3.141592))")
        
        angles.addPerAngleParameter("ka")
        for start, end, isRing in self.chains:
            for j in range(start + 1, end):
                angles.addAngle(j - 1, j, j + 1, [ka[j]])
                
                
            if isRing:
                angles.addAngle(end - 1, end , start, [ka[end]])
                angles.addAngle(end , start, start + 1, [ka[start]])

        self.metadata["AngleForce"] = repr({"stiffness": ka})
        self.forceDict["AngleForce"] = angles
        
        
    def addRepulsiveSoftCore(self, Ecut=4.0,CutoffDistance=3.0):
        
        R"""
        Adds a soft-core repulsive interaction that allows chain crossing, which represents the activity of topoisomerase II. Details can be found in the following publications: 
        
            - Oliveira Jr., A.B., Contessoto, V.G., Mello, M.F. and Onuchic, J.N., 2021. A scalable computational approach for simulating complexes of multiple chromosomes. Journal of Molecular Biology, 433(6), p.166700.
            - Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173.
            - Naumova, N., Imakaev, M., Fudenberg, G., Zhan, Y., Lajoie, B.R., Mirny, L.A. and Dekker, J., 2013. Organization of the mitotic chromosome. Science, 342(6161), pp.948-953.

        Args:

            Ecut (float, required):
                Energy cost for the chain passing in units of :math:`k_{b}T`. (Default value = 4.0).
          """
        
        nbCutOffDist = self.Sigma * 2. ** (1. / 6.) #1.112
        
        Ecut = Ecut*self.Epsilon
        
        r_0 = self.Sigma*(((0.5*Ecut)/(4.0*self.Epsilon) - 0.25 +((0.5)**(2.0)))**(1.0/2.0) +0.5)**(-1.0/6.0)
        
        repul_energy = ("LJ * step(r - r_0) * step(CutOff - r)"
                       " + step(r_0 - r)* 0.5 * Ecut * (1.0 + tanh( (2.0 * LJ/Ecut) - 1.0 ));"
                       "LJ = 4.0 * Epsi * ((Sig/r)^12 - (Sig/r)^6) + Epsi")
        
        self.forceDict["RepulsiveSoftCore"] = self.mm.CustomNonbondedForce(
            repul_energy)
        repulforceGr = self.forceDict["RepulsiveSoftCore"]
        repulforceGr.addGlobalParameter('Epsi', self.Epsilon)
        repulforceGr.addGlobalParameter('Sig', self.Sigma)
        repulforceGr.addGlobalParameter('Ecut', Ecut)
        repulforceGr.addGlobalParameter('r_0', r_0)
        repulforceGr.addGlobalParameter('CutOff', nbCutOffDist)
        repulforceGr.setCutoffDistance(CutoffDistance)

        for _ in range(self.N):
            repulforceGr.addParticle(())
        
    def addTypetoType(self, mu=3.22, rc = 1.78 ):
        R"""
        Adds the type-to-type interactions according to the MiChroM energy function parameters reported in "Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173". 
        
        The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        Args:

            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
        """

        self.metadata["TypetoType"] = repr({"mu": mu})

        path = "share/MiChroM.ff"
        pt = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(pt,path)

        self.addCustomTypes(name="TypetoType", mu=mu, rc=rc, TypesTable=filepath)
        

    def addCustomTypes(self, name="CustomTypes", mu=3.22, rc = 1.78, TypesTable=None,CutoffDistance=3.0):
        R"""
        Adds the type-to-type potential using custom values for interactions between the chromatin types. The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        The function receives a txt/TSV/CSV file containing the upper triangular matrix of the type-to-type interactions. A file example can be found `here <https://github.com/junioreif/OpenMiChroM/blob/main/OpenMiChroM/share/MiChroM.ff>`__.
        
        +---+------+-------+-------+
        |  |   A  |   B   |   C   |
        +---+------+-------+-------+
        |  | -0.2 | -0.25 | -0.15 |
        +---+------+-------+-------+
        |  |      |  -0.3 | -0.15 |
        +---+------+-------+-------+
        |  |      |       | -0.35 |
        +---+------+-------+-------+
        
        Args:

            name (string, required):
                Name to customType Potential. (Default value = "CustomTypes") 
            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
            TypesTable (file, required):
                A txt/TSV/CSV file containing the upper triangular matrix of the type-to-type interactions. (Default value: :code:`None`).


        """

        self.metadata["CrossLink"] = repr({"mu": mu})
        if not hasattr(self, "type_list_letter"):
            raise ValueError("Chromatin sequence not defined!")

        energy = "mapType(t1,t2)*0.5*(1. + tanh(mu*(rc - r)))*step(r-lim)"
        
        crossLP = self.mm.CustomNonbondedForce(energy)
    
        crossLP.addGlobalParameter('mu', mu)
        crossLP.addGlobalParameter('rc', rc)
        crossLP.addGlobalParameter('lim', 1.0)
        crossLP.setCutoffDistance(CutoffDistance)

        tab = pd.read_csv(TypesTable, sep=None, engine='python')

        header_types = list(tab.columns.values)

        if not set(self.diff_types).issubset(set(header_types)):
            errorlist = []
            for i in self.diff_types:
                if not (i in set(header_types)):
                    errorlist.append(i)
            raise ValueError("Types: {} are not present in TypesTables: {}\n".format(errorlist, header_types))

        diff_types_size = len(header_types)
        lambdas = np.triu(tab.values) + np.triu(tab.values, k=1).T
        lambdas = list(np.ravel(lambdas))
          
        fTypes = self.mm.Discrete2DFunction(diff_types_size,diff_types_size,lambdas)
        crossLP.addTabulatedFunction('mapType', fTypes) 
            
        self._createTypeList(header_types)
        crossLP.addPerParticleParameter("t")

        for i in range(self.N):
            value = [float(self.type_list[i])]
            crossLP.addParticle(value)
                
        self.forceDict[name] = crossLP
    
    def _createTypeList(self, header_types):
        R"""
        Internal function for indexing unique chromatin types.
        """
        typesDict = {}
        for index,type in enumerate(header_types):
            typesDict[type] = index

        self.type_list = [typesDict[letter] for letter in self.type_list_letter]
        
        
    def addLoops(self, mu=3.22, rc = 1.78, X=-1.612990, looplists=None):
        R"""
        Adds the Loops interactions according to the MiChroM energy function parameters reported in "Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173". 
        
        The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        .. note:: For Multi-chain simulations, the ordering of the loop list files is important! The order of the files should be the same as used in the other functions.
        
        Args:

            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
            X (float, required):
                Loop interaction parameter. (Default value = -1.612990).
            looplists (list[str], required): 
                List with the names of the files containing loop information. Each file should be a two-column text file containing the index *i* and *j* of the loci pairs that forms the loop anchors. (Default value: :code:`None`).
        """
        
        if isinstance(looplists, str):
            looplists = [looplists]

        ELoop = "qsi*0.5*(1. + tanh(mu*(rc - r)))"
                
        Loop = self.mm.CustomBondForce(ELoop)
        
        Loop.addGlobalParameter('mu', mu)  
        Loop.addGlobalParameter('rc', rc) 
        Loop.addGlobalParameter('qsi', X) 
        
        self.getLoops(looplists)
        
        for p in self.loopPosition:
            Loop.addBond(p[0]-1,p[1]-1)
  
        self.forceDict["Loops"] = Loop  
        
    def addCustomIC(self, mu=3.22, rc = 1.78, dinit=3, dend=200, IClist=None,CutoffDistance=3.0):
        R"""
        Adds the Ideal Chromosome potential using custom values for interactions between beads separated by a genomic distance :math:`d`. The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        Args:
        
            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
            dinit (int, required):
                The first neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential. (Default value = 3).
            dend (int, required):
                The last neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential. (Default value = 200).
            IClist (file, optional):
                A one-column text file containing the energy interaction values for loci *i* and *j* separated by a genomic distance :math:`d`. (Default value: :code:`None`).
        
        """

        energyIC = ("step(d-dinit)*IClist(d)*step(dend -d)*f*step(r-lim);"
                    "f=0.5*(1. + tanh(mu*(rc - r)));"
                    "d=abs(idx2-idx1)")

               
        IC = self.mm.CustomNonbondedForce(energyIC)

        IClist_listfromfile = np.loadtxt(IClist)
        IClist = np.append(np.zeros(dinit),IClist_listfromfile)[:-dinit]
        
        tabIClist = self.mm.Discrete1DFunction(IClist)
        IC.addTabulatedFunction('IClist', tabIClist) 

        IC.addGlobalParameter('dinit', dinit) 
        IC.addGlobalParameter('dend', dend)
        
        IC.addGlobalParameter('mu', mu)  
        IC.addGlobalParameter('rc', rc) 
        IC.addGlobalParameter('lim', 1.0)
        
        IC.setCutoffDistance(CutoffDistance)


        IC.addPerParticleParameter("idx")

        for i in range(self.N):
                IC.addParticle([i])
        
        self.forceDict["CustomIC"] = IC
        
    def addIdealChromosome(self, mu=3.22, rc = 1.78, Gamma1=-0.030,Gamma2=-0.351,
                           Gamma3=-3.727, dinit=3, dend=500,CutoffDistance=3.0):
        
        R"""
        Adds the Ideal Chromosome potential for interactions between beads separated by a genomic distance :math:`d` according to the MiChroM energy function parameters reported in "Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173". 
        
        The set of parameters :math:`\{\gamma_d\}` of the Ideal Chromosome potential is fitted in a function: :math:`\gamma(d) = \frac{\gamma_1}{\log{(d)}} +\frac{\gamma_2}{d} +\frac{\gamma_3}{d^2}`. 
        
        The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        Args:
        
            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
            Gamma1 (float, required):
                Ideal Chromosome parameter. (Default value = -0.030).
            Gamma2 (float, required):
                Ideal Chromosome parameter. (Default value = -0.351).
            Gamma3 (float, required):
                Ideal Chromosome parameter. (Default value = -3.727).
            dinit (int, required):
                The first neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential. (Default value = 3).
            dend (int, required):
                The last neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential. (Default value = 500).
        """


        energyIC = ("step(d-dinit)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dend -d)*f;"
                   "f=0.5*(1. + tanh(mu*(rc - r)));"
                   "d=abs(idx1-idx2)")

        IC = self.mm.CustomNonbondedForce(energyIC)

        IC.addGlobalParameter('gamma1', Gamma1) 
        IC.addGlobalParameter('gamma2', Gamma2)
        IC.addGlobalParameter('gamma3', Gamma3)
        IC.addGlobalParameter('dinit', dinit) 
        IC.addGlobalParameter('dend', dend) 
        
        IC.addGlobalParameter('mu', mu)  
        IC.addGlobalParameter('rc', rc) 
        
        IC.setCutoffDistance(CutoffDistance)


        IC.addPerParticleParameter("idx")

        for i in range(self.N):
                IC.addParticle([i])
        
        self.forceDict["IdealChromosome"] = IC
        
        
    def addMultiChainIC(self, mu=3.22, rc = 1.78, Gamma1=-0.030,Gamma2=-0.351,
                           Gamma3=-3.727, dinit=3, dend=500, chainIndex=0,CutoffDistance=3.0):
        
        R"""
        Adds the Ideal Chromosome potential for multiple chromosome simulations. The interactions between beads separated by a genomic distance :math:`d` is applied according to the MiChroM energy function parameters reported in "Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173". 
        
        The set of parameters :math:`\{\gamma_d\}` of the Ideal Chromosome potential is fitted in a function: :math:`\gamma(d) = \frac{\gamma_1}{\log{(d)}} +\frac{\gamma_2}{d} +\frac{\gamma_3}{d^2}`. 
        
        The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        Args:
        
            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
            Gamma1 (float, required):
                Ideal Chromosome parameter. (Default value = -0.030).
            Gamma2 (float, required):
                Ideal Chromosome parameter. (Default value = -0.351).
            Gamma3 (float, required):
                Ideal Chromosome parameter. (Default value = -3.727).
            dinit (int, required):
                The first neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential. (Default value = 3).
            dend (int, required):
                The last neighbor in sequence separation (Genomic Distance) to be considered in the Ideal Chromosome potential. (Default value = 500).
            chainIndex (integer, required):
                The index of the chain to add the Ideal Chromosome potential. All chains are stored in :code:`self.chains`. (Default value: :code:`0`).
        """

        energyIC = ("step(d-dinit)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dend-d)*f;"
                   "f=0.5*(1. + tanh(mu*(rc - r)));"
                   "d=abs(idx1-idx2)")
        
        
        IC = self.mm.CustomNonbondedForce(energyIC)

        IC.addGlobalParameter('gamma1', Gamma1) 
        IC.addGlobalParameter('gamma2', Gamma2)
        IC.addGlobalParameter('gamma3', Gamma3)
        IC.addGlobalParameter('dinit', dinit)
        IC.addGlobalParameter('dend', dend)
      
        IC.addGlobalParameter('mu', mu)  
        IC.addGlobalParameter('rc', rc) 
        
        IC.setCutoffDistance(CutoffDistance)
        
        chain = self.chains[chainIndex]

        groupList = list(range(chain[0],chain[1]+1))
        
        IC.addInteractionGroup(groupList,groupList)
        
        IC.addPerParticleParameter("idx")

        for i in range(self.N):
            IC.addParticle([i])
        
        self.forceDict["IdealChromosomeChain{0}".format(chainIndex)] = IC


    def addCorrelatedNoise(self, act_seq='none', atol=1e-5):
        R"""
        This function assigns active force to monomers, as defined by the act_seq array.
        Args:
            
            act_seq (list, required): 
                This list should contain the same number of elements as the number of monomers, where the value at index i corresponds to the active force F for the bead i

            atol (float, optional):
                tolerance for force values. If force is less than atol then it is ignored.

        Returns Error if act_seq is not the same length as the sequence file.
        """
        
        try:
            act_seq=np.asfarray(act_seq)

            #ensure all the monomers are accounted for
            assert len(act_seq)==self.N

            #do not redistribute velocities based on Maxwell Boltzman since this is a non-equilbrium sim
            self.velocityReinitialize = False

            #define active force group
            act_force = self.mm.CustomExternalForce(" - F_act * (x + y + z)")
            act_force.addPerParticleParameter('F_act')
            self.forceDict["ActiveForce"] = act_force

            # Important for the active force group to be set to "0". 
            # This assignment is however overrun by _applyForces() inside runSimBlock(). 
            # Hence it is necessary that active force is the *first* force field added to the simulation.
            self.forceDict["ActiveForce"].setForceGroup(0)
            
            #assign correlated noise to the monomers with non-zero active force
            for bead_id, Fval in enumerate(act_seq):
                if Fval>atol:    
                    self.forceDict["ActiveForce"].addParticle(int(bead_id),[Fval])

            print('\n\
            ==================================\n\
            Active Monomers (correlated noise) added.\n\
            Active correlation time: {}\n\
            Total number of active monomers: {}\n\
            Total number of monomers: {}\n\
            ==================================\n'.format(self.integrator.corrTime, self.forceDict["ActiveForce"].getNumParticles(), self.N))
        
        except (AttributeError):
            print("Structure not loaded! Load structure before adding activity.\nCorrelated noise has NOT been added!")
            
        except (ValueError, AssertionError):
            print('Active sequence (act_seq) either not defined or all the monomers are not accounted for!\nCorrelated noise has NOT been added!')


    def addHarmonicBonds(self, kfb=30.0, r0=1.0):
        R"""
        This function adds harmonic bonds to all the monomers within a chain
        Args:
            kfb (float, required): 
                bond stiffness
            
            r0 (float, required):
                equilibrium distance for the bond
        """

        for start, end, isRing in self.chains:
            for j in range(start, end):
                self.addHarmonicBond_ij(j, j + 1, kfb=kfb, r0=r0)

            if isRing:
                self.addHarmonicBond_ij(start, end, kfb=kfb, r0=r0)

        self.metadata["HarmonicBond"] = repr({"kfb": kfb})
        

    def _initHarmonicBond(self, kfb=30,r0=1.0):
        R"""
        Internal function used to initiate Harmonic Bond force group
        Args:
            kfb (float, required): 
                bond stiffness
            
            r0 (float, required):
                equilibrium distance for the bond
        """
        
        if "HarmonicBond" not in list(self.forceDict.keys()):
            force = ("0.5 * kfb * (r-r0)*(r-r0)")
            bondforceGr = self.mm.CustomBondForce(force)
            bondforceGr.addGlobalParameter("kfb", kfb)
            bondforceGr.addGlobalParameter("r0", r0) 
                
            self.forceDict["HarmonicBond"] = bondforceGr
        

    def addHarmonicBond_ij(self, i, j, r0=1.0, kfb=30):
        R"""
        Internal function used to add bonds between i and j monomers
        Args:
            i,j (int, required):
                monomers to be bonded

            kfb (float, required): 
                bond stiffness
            
            r0 (float, required):
                equilibrium distance for the bond
        """
        
        if (i >= self.N) or (j >= self.N):
            raise ValueError("\n Cannot add a bond between beads  %d,%d that are beyond the chromosome length %d" % (i, j, self.N))
        
        self._initHarmonicBond(kfb=kfb, r0=r0)
        self.forceDict["HarmonicBond"].addBond(int(i), int(j), [])
        self.bondsForException.append((int(i), int(j)))


    def addSelfAvoidance(self, Ecut=4.0, k_rep=20.0, r0=1.0):
        R"""
        This adds Soft-core self avoidance between all non-bonded monomers.
        This force is well behaved across all distances (no diverging parts)
        Args:
            Ecut (float, required): 
                energy associated with full overlap between monomers
            k_rep (float, required):
                steepness of the sigmoid repulsive potential

            r0 (float, required):
                distance from the center at which the sigmoid is half as strong
        """

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


    def _getForceIndex(self, forceName):
        R""""
        Get the index of one of the forces in the force dictionary.
        """   

        forceObject = self.forceDict[forceName]

        index = [i for i,systemForce in enumerate(self.system.getForces()) if systemForce.this == forceObject.this]

        if len(index) == 1:
            return index[0]
        else:
            raise Exception("Found more than one force with input name!")


    def _isForceDictEqualSystemForces(self):
        R""""
        Internal function that returns True when forces in self.forceDict and in self.system are equal.
        """

        forcesInDict = [ x.this for x in self.forceDict.values() ]
        forcesInSystem = [ x.this for x in self.system.getForces() ]

        if not len(forcesInDict) == len(forcesInSystem):
            return False
        else:
            isEqual = []
            for i in forcesInDict:
                isEqual.append((i in forcesInSystem))
            return all(isEqual)


    def removeForce(self, forceName):
        R""""
        Remove force from the system.
        """

        if forceName in self.forceDict:
            self.system.removeForce(self._getForceIndex(forceName))
            del self.forceDict[forceName]

            self.context.reinitialize(preserveState=True) 
            print(f"Removed {forceName} from the system!")
            assert self._isForceDictEqualSystemForces(), 'Forces in forceDict should be the same as in the system!'

        else:
            raise ValueError("The system does not have force {0}.\nThe forces applied in the system are: {}\n".format(forceName, self.forceDict.keys()))


    def removeFlatBottomHarmonic(self):
        R""""
        Remove FlatBottomHarmonic force from the system.
        """

        forceName = "FlatBottomHarmonic"

        self.removeForce(forceName)


    def addAdditionalForce(self, forceFunction, **args):
        R"""
        Add an additional force after the system has already been initialized.

        Args:

            forceFunciton (function, required):
                Force function to be added. Example: addSphericalConfinementLJ
            **args (collection of arguments, required):
                Arguments of the function to add the force. Consult respective documentation.
        """
        
        # warning for user defined force function
        if not isinstance(forceFunction, types.MethodType):
            print("WARNING! Using user defined force function. Make sure to include the new force object in the MiChroM.forceDict dictionary.")

        #store old forcedict keys
        oldForceDictKeys = self.forceDict.keys()
        
        # call the function --  
        # the force name is added to the forceDict but not yet added to the system
        forceFunction(**args)

        # find the new forceDict name
        newForceDictKey = [x for x in self.forceDict.keys() if x not in oldForceDictKeys]

        if len(newForceDictKey) == 0:
            raise Exception("No new force in MiChroM.forceDict! The new force is either already present or was not added properly to the force dictionary.")
        elif len(newForceDictKey) > 1:
            raise Exception("Force function added multiple new forces in MiChroM! Please break down the function so each force is added separately.")
        
        force = self.forceDict[newForceDictKey[0]]
        
        # exclusion list
        exc = self.bondsForException

        # set all the attributes of the force (see _applyForces)
        if hasattr(force, "addException"):
            print('Add exceptions for {0} force'.format(newForceDictKey))
            for pair in exc:
                force.addException(int(pair[0]),
                    int(pair[1]), 0, 0, 0, True)
                
        elif hasattr(force, "addExclusion"):
            print('Add exclusions for {0} force'.format(newForceDictKey))
            for pair in exc:
                force.addExclusion(int(pair[0]), int(pair[1]))

        if hasattr(force, "CutoffNonPeriodic") and hasattr(
                                                force, "CutoffPeriodic"):
            if self.PBC:
                force.setNonbondedMethod(force.CutoffPeriodic)
                print("Using periodic boundary conditions!!!!")
            else:
                force.setNonbondedMethod(force.CutoffNonPeriodic)
        
        # add the force
        print("adding force ", newForceDictKey, self.system.addForce(self.forceDict[newForceDictKey]))

        #assign force groups
        for name in self.forceDict.keys():
            force_group = self.forceDict[name].getForceGroup()
            if force_group>31: 
                force_group=31
                print("Attention, force was added to Force Group 31 because no other was available.")
            
            self.forceDict[name].setForceGroup(force_group)

        # reinitialize the system with the new force after force group assignments
        self.context.reinitialize(preserveState=True) 
        
        assert self._isForceDictEqualSystemForces(), 'Forces in forceDict should be the same as in the system!'


    def _loadParticles(self):
        R"""
        Internal function that loads the chromosome beads into the simulations system.
        """
        if not hasattr(self, "system"):
            return
        if not self.loaded:
            for mass in self.masses:
                self.system.addParticle(self.mass * mass)
            if self.verbose == True:
                print("%d particles loaded" % self.N)
            self.loaded = True
            

    def _applyForces(self):
        R"""Internal function that adds all loci to the system and applies all the forces present in the forcedict."""

        if self.forcesApplied == True:
            return
        self._loadParticles()

        exc = self.bondsForException
        print("Number of exceptions:", len(exc))

        if len(exc) > 0:
            exc = np.array(exc)
            exc = np.sort(exc, axis=1)
            exc = [tuple(i) for i in exc]
            exc = list(set(exc)) 

        for i in list(self.forceDict.keys()): 
            force = self.forceDict[i]
            if hasattr(force, "addException"):
                print('Add exceptions for {0} force'.format(i))
                for pair in exc:
                    force.addException(int(pair[0]),
                        int(pair[1]), 0, 0, 0, True)
            elif hasattr(force, "addExclusion"):
                print('Add exclusions for {0} force'.format(i))
                for pair in exc:
                    force.addExclusion(int(pair[0]), int(pair[1]))

            if hasattr(force, "CutoffNonPeriodic") and hasattr(
                                                    force, "CutoffPeriodic"):
                if self.PBC:
                    force.setNonbondedMethod(force.CutoffPeriodic)
                    print("Using periodic boundary conditions!!!!")
                else:
                    force.setNonbondedMethod(force.CutoffNonPeriodic)
            print("adding force ", i, self.system.addForce(self.forceDict[i]))
        
        ForceGroupIndex = 0
        for i,name in enumerate(self.forceDict):
            if "IdealChromosomeChain" in name:
                self.forceDict[name].setForceGroup(31) 
            else:
                self.forceDict[name].setForceGroup(ForceGroupIndex)
                ForceGroupIndex+= 1
            
        self.context = self.mm.Context(self.system, self.integrator, self.platform, self.properties)
        self.initPositions()
        self.initVelocities()
        self.forcesApplied = True
     
        with open(str(Path(self.folder))+'/platform_info.dat', 'w') as f:
                print('Name: ', self.platform.getName(), file=f)
                print('Speed: ',self.platform.getSpeed(), file=f)
                print('Property names: ',self.platform.getPropertyNames(), file=f)
                for name in self.platform.getPropertyNames():
                     print(name,' value: ',self.platform.getPropertyValue(self.context, name), file=f) 

    
    def loadNDB(self, NDBfiles=None):
        R"""
        Loads a single or multiple *.ndb* files and gets position and types of the chromosome beads.
        Details about the NDB file format can be found at the `Nucleome Data Bank <https://ndb.rice.edu/ndb-format>`__.
        
            - Contessoto, V.G., Cheng, R.R., Hajitaheri, A., Dodero-Rojas, E., Mello, M.F., Lieberman-Aiden, E., Wolynes, P.G., Di Pierro, M. and Onuchic, J.N., 2021. The Nucleome Data Bank: web-based resources to simulate and analyze the three-dimensional genome. Nucleic Acids Research, 49(D1), pp.D172-D182.
        
        Args:

            NDBfiles (file, required):
                Single or multiple files in *.ndb* file format.  (Default value: :code:`None`).
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
   
        """

        x = []
        y = []
        z = []
        start = 0
        typesLetter = []
        chains = []
        sizeChain = 0

        for ndb in NDBfiles:
            aFile = open(ndb,'r')

            lines = aFile.read().splitlines()

            for line in lines:
                line = line.split()

                if line[0] == 'CHROM':
                    x.append(float(line[5]))
                    y.append(float(line[6]))
                    z.append(float(line[7]))

                    typesLetter.append(line[2])

                    sizeChain += 1
                elif line[0] == "TER" or line[0] == "END":
                    break

            chains.append((start, sizeChain-1, 0))
            start = sizeChain 

        print("Chains: ", chains)

        if set(typesLetter) == {'SQ'}:
            typesLetter = ['bead{:}'.format(x) for x in range(1,len(typesLetter)+1)]

        self.diff_types = set(typesLetter)
        
        self.type_list_letter = typesLetter

        self.setChains(chains)

        return np.vstack([x,y,z]).T
    
    
    def loadGRO(self, GROfiles=None, ChromSeq=None):
        R"""
        Loads a single or multiple *.gro* files and gets position and types of the chromosome beads.
        Initially, the MiChroM energy function was implemented in GROMACS. Details on how to run and use these files can be found at the `Nucleome Data Bank <https://ndb.rice.edu/GromacsInput-Documentation>`__.
        
            - Contessoto, V.G., Cheng, R.R., Hajitaheri, A., Dodero-Rojas, E., Mello, M.F., Lieberman-Aiden, E., Wolynes, P.G., Di Pierro, M. and Onuchic, J.N., 2021. The Nucleome Data Bank: web-based resources to simulate and analyze the three-dimensional genome. Nucleic Acids Research, 49(D1), pp.D172-D182.
        
        Args:

            GROfiles (list of files, required):
                List with a single or multiple files in *.gro* file format.  (Default value: :code:`None`).
            ChromSeq (list of files, optional):
                List of files with sequence information for each chromosomal chain. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`__.
                If the chromatin types considered are different from the ones used in the original MiChroM (A1, A2, B1, B2, B3, B4, and NA), the sequence file must be provided, otherwise all the chains will be defined with 'NA' type.
                
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
   
        """
        
        x = []
        y = []
        z = []
        start = 0
        chains = []
        sizeChain = 0
        typesLetter = []
        
        for gro in GROfiles:
            aFile = open(gro,'r')
            pos = aFile.read().splitlines()
            size = int(pos[1])
            
            for t in range(2, len(pos)-1):

                try:
                    float(pos[t].split()[3]); float(pos[t].split()[4]); float(pos[t].split()[5])
                    pos[t] = pos[t].split()
                except:
                    pos[t] = [str(pos[t][0:10]).split()[0], str(pos[t][10:15]), str(pos[t][15:20]), str(pos[t][20:28]), str(pos[t][28:36]), str(pos[t][36:44])]
                
                x.append(float(pos[t][3]))
                y.append(float(pos[t][4]))
                z.append(float(pos[t][5]))
                
                typesLetter.append(self._aa2types(pos[t][0][-3:]))
                sizeChain += 1

            chains.append((start, sizeChain-1, 0))
            start = sizeChain 
            
        if not ChromSeq is None:

            if len(ChromSeq) != len(GROfiles):
                raise ValueError("Number of sequence files provided must agree with number of coordinate files!")

            typesLetter = []
            for seqFile in ChromSeq:
                print('Reading sequence in {}...'.format(seqFile))
                with open(seqFile, 'r') as sequence:
                    for type in sequence:
                        typesLetter.append(type.split()[1])

        if len(typesLetter) != len(x):
            raise ValueError("Sequence length is different from coordinates length!")

        print("Chains: ", chains)

        self.diff_types = set(typesLetter)

        self.type_list_letter = typesLetter

        self.setChains(chains)

        return np.vstack([x,y,z]).T

    def _aa2types (self, amino_acid):
        
        Type_conversion = {'ASP':"A1", 'GLU':"A2", 'HIS':"B1", 'LYS':"B2", 'ARG':"B3", 'ARG':"B3", 'ASN':"NA"}

        if amino_acid in Type_conversion.keys():
            return Type_conversion[amino_acid]
        else:
            return 'NA'
                    
    def loadPDB(self, PDBfiles=None, ChromSeq=None):
        
        R"""
        Loads a single or multiple *.pdb* files and gets position and types of the chromosome beads.
        Here we consider the chromosome beads as the carbon-alpha to mimic a protein. This trick helps to use the standard macromolecules visualization software. 
        The type-to-residue conversion follows: {'ALA':0, 'ARG':1, 'ASP':2, 'GLU':3,'GLY':4, 'LEU' :5, 'ASN' :6}.
        
        Args:

            PDBfiles (list of files, required):
                List with a single or multiple files in *.pdb* file format.  (Default value: :code:`None`).
            ChromSeq (list of files, optional):
                List of files with sequence information for each chromosomal chain. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`__.
                If the chromatin types considered are different from the ones used in the original MiChroM (A1, A2, B1, B2, B3, B4, and NA), the sequence file must be provided, otherwise all the chains will be defined with 'NA' type.

        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
   
        """
        
        x = []
        y = []
        z = []
        start = 0
        chains = []
        sizeChain = 0
        typesLetter = []
        
        for pdb in PDBfiles:
            aFile = open(pdb,'r')
            pos = aFile.read().splitlines()

            for t in range(len(pos)):

                try:
                    float(pos[t].split()[5]); float(pos[t].split()[6]); float(pos[t].split()[7])
                    pos[t] = pos[t].split()
                except:
                    pos[t] = [str(pos[t][0:6]).split()[0], pos[t][6:11], pos[t][12:6], pos[t][17:20], pos[t][22:26], pos[t][30:38], pos[t][38:46], pos[t][46:54]]

                if pos[t][0] == 'ATOM':
                    x.append(float(pos[t][5]))
                    y.append(float(pos[t][6]))
                    z.append(float(pos[t][7]))
                    typesLetter.append(self._aa2types(pos[t][3]))
                    sizeChain += 1

            chains.append((start, sizeChain-1, 0))
            start = sizeChain 

        if not ChromSeq is None:

            if len(ChromSeq) != len(PDBfiles):
                raise ValueError("Number of sequence files provided must agree with number of coordinate files!")

            typesLetter = []
            for seqFile in ChromSeq:
                print('Reading sequence in {}...'.format(seqFile))
                with open(seqFile, 'r') as sequence:
                    for type in sequence:
                        typesLetter.append(type.split()[1])

        if len(typesLetter) != len(x):
            raise ValueError("Sequence length is different from coordinates length!")
            
        print("Chains: ", chains)

        self.diff_types = set(typesLetter)

        self.type_list_letter = typesLetter

        self.setChains(chains)

        return np.vstack([x,y,z]).T


    def createSpringSpiral(self, ChromSeq=None, isRing=False):
        
        R"""
        Creates a spring-spiral-like shape for the initial configuration of the chromosome polymer.
        
        Args:

            ChromSeq (file, required):
                Chromatin sequence of types file. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`__.
            isRing (bool, optional):
                Whether the chromosome chain is circular or not (Used to simulate bacteria genome, for example). f :code:`bool(isRing)` is :code:`True` , the first and last particles of the chain are linked, forming a ring. (Default value = :code:`False`).
                
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
   
        """

        x = []
        y = []
        z = []
        self._translate_type(ChromSeq)
        beads = len(self.type_list_letter)
        
        for i in range(beads):
            if (isRing):
                a = 2.0*((beads-1)/beads)*np.pi*(i-1)/(beads-1)
                a1 = 2.0*((beads-1)/beads)*np.pi*(2-1)/(beads-1)
            else:
                a = 1.7*np.pi*(i-1)/(beads-1)
                a1 = 1.7*np.pi*(2-1)/(beads-1)
            b=1/np.sqrt((4-3.0*np.cos(a1)-np.cos(10*a1)*np.cos(a1))**2 +
                (0-3.0*np.sin(a1)-np.cos(10*a1)*np.sin(a1))**2+(np.sin(10*a1))**2)

            x.append(1.5*np.pi*b+3*b*np.cos(a)+b*np.cos(10*a)*np.cos(a))
            y.append(1.5*np.pi*b+3.0*b*np.sin(a)+b*np.cos(10*a)*np.sin(a))
            z.append(1.5*np.pi*b+b*np.sin(10*a))
        
        chain = []
        if (isRing):
            chain.append((0,beads-1,1))
        else:
            chain.append((0,beads-1,0))

        self.setChains(chain)

        return np.vstack([x,y,z]).T
    
    def random_ChromSeq(self, Nbeads):
        
        R"""
        Creates a random sequence of chromatin types for the chromosome beads.
        
        Args:

            Nbeads (int, required):
                Number of beads of the chromosome polymer chain. (Default value = 1000).
        Returns:
            :math:`(N, 1)` :class:`numpy.ndarray`:
                Returns an 1D array of a randomized chromatin type annotation sequence.
   
        """
        
        return random.choices(population=[0,1,2,3,4,5], k=Nbeads)
    
    def _translate_type(self, filename):
        
        R"""Internal function that converts the letters of the types numbers following the rule: 'A1':0, 'A2':1, 'B1':2, 'B2':3,'B3':4,'B4':5, 'NA' :6.
        
         Args:

            filename (file, required):
                Chromatin sequence of types file. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`_.

        """        
        
        self.diff_types = []
        self.type_list_letter = []

        af = open(filename,'r')
        pos = af.read().splitlines()
        
        for t in range(len(pos)):
            pos[t] = pos[t].split()
            if pos[t][1] in self.diff_types:
                self.type_list_letter.append(pos[t][1])
            else:
                self.diff_types.append(pos[t][1]) 
                self.type_list_letter.append(pos[t][1])

    def createLine(self, ChromSeq):
        
        R"""
        Creates a straight line for the initial configuration of the chromosome polymer.
        
        Args:

            ChromSeq (file, required):
                Chromatin sequence of types file. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`__.
            length_scale (float, required):
                Length scale used in the distances of the system in units of reduced length :math:`\sigma`. (Default value = 1.0).    
                
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
   
        """

        self._translate_type(ChromSeq)
        beads = len(self.type_list_letter)

        length_scale = 1.0
        x = []
        y = []
        z = []
        for i in range(beads):
            x.append(0.15*length_scale*beads+(i-1)*0.6)
            y.append(0.15*length_scale*beads+(i-1)*0.6)
            z.append(0.15*length_scale*beads+(i-1)*0.6)
        
        chain = []
        chain.append((0,beads-1,0))
        self.setChains(chain)

        return np.vstack([x,y,z]).T
    
    def createRandomWalk(self, ChromSeq=None):    
        R"""
        Creates a chromosome polymer chain with beads position based on a random walk.
        
        Args:

        ChromSeq (file, required):
            Chromatin sequence of types file. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`__.
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
   
        """
        
        self._translate_type(ChromSeq)
        Nbeads = len(self.type_list_letter)

        segment_length = 1
        step_size = 1

        theta = np.repeat(np.random.uniform(0., 1., Nbeads // segment_length + 1), segment_length)
        theta = 2.0 * np.pi * theta[:Nbeads]
        u = np.repeat(np.random.uniform(0., 1., Nbeads // segment_length + 1), segment_length)
        u = 2.0 * u[:Nbeads] - 1.0
        x = step_size * np.sqrt(1. - u * u) * np.cos(theta)
        y = step_size * np.sqrt(1. - u * u) * np.sin(theta)
        z = step_size * u
        x, y, z = np.cumsum(x), np.cumsum(y), np.cumsum(z)

        chain = []
        chain.append((0,Nbeads-1,0))
        self.setChains(chain)

        return np.vstack([x, y, z]).T

    def initStructure(self, mode='auto', CoordFiles=None, ChromSeq=None, isRing=False):

        R"""
        Creates the coordinates for the initial configuration of the chromosomal chains and sets their sequence information.
 
        Args:

        mode (str, required):
            - 'auto' - Creates a spring-spiral-like shape when a CoordFiles is not provided. If CoordFiles is provided, it loads the respective type of coordinate files (.ndb, .gro, or .pdb). (Default value = 'auto').
            - 'line' - Creates a straight line for the initial configuration of the chromosome polymer. Can only be used to create single chains.
            - 'spring' - Creates a spring-spiral-like shape for the initial configuration of the chromosome polymer. Can only be used to create single chains.
            - 'random' - Creates a chromosome polymeric chain with beads positions based on a random walk. Can only be used to create single chains.
            - 'ndb' - Loads a single or multiple *.ndb* files and gets the position and types of the chromosome beads.
            - 'pdb' - Loads a single or multiple *.pdb* files and gets the position and types of the chromosome beads.
            - 'gro' - Loads a single or multiple *.gro* files and gets the position and types of the chromosome beads.

        CoordFiles (list of files, optional):
            List of files with xyz information for each chromosomal chain. Accepts .ndb, .pdb, and .gro files. All files provided in the list must be in the same file format.

        ChromSeq (list of files, optional):
            List of files with sequence information for each chromosomal chain. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`__.
            If the chromatin types considered are different from the ones used in the original MiChroM (A1, A2, B1, B2, B3, B4, and NA), the sequence file must be provided when loading .pdb or .gro files, otherwise, all the chains will be defined with 'NA' type. For the .ndb files, the sequence used is the one provided in the file.
        
        isRing (bool, optional):
            Whether the chromosome chain is circular or not (used to simulate bacteria genome, for example). To be used with the option :code:`'random'`. If :code:`bool(isRing)` is :code:`True` , the first and last particles of the chain are linked, forming a ring. (Default value = :code:`False`).
 
        Returns:
            :math:`(N, 3)` :class:`numpy.ndarray`:
                Returns an array of positions.
   
        """

        if mode == 'auto':
            if CoordFiles is None:
                mode = 'spring'
            else:
                _, extension = os.path.splitext(CoordFiles[0])
                if extension == '.pdb':
                    mode = 'pdb'
                elif extension == '.gro':
                    mode = 'gro'
                elif extension == '.ndb':
                    mode = 'ndb'
                else:
                    raise ValueError("Unrecognizable coordinate file.")

        if isinstance(CoordFiles, str):
            CoordFiles = [CoordFiles]

        if isinstance(ChromSeq, str):
            ChromSeq = [ChromSeq]

        if mode in ['spring', 'line', 'random']:
            if isinstance(ChromSeq, list):
                
                if len(ChromSeq) > 1:
                    raise ValueError("'{}' mode can only be used to create single chains.".format(mode))

            if CoordFiles != None:
                raise ValueError("Providing coordinates' file not compatible with mode '{0}'.".format(mode))

        if mode == 'line':

            return self.createLine(ChromSeq=ChromSeq[0])

        elif mode == 'spring':

            return self.createSpringSpiral(ChromSeq=ChromSeq[0], isRing=isRing)

        elif mode == 'random':

            return self.createRandomWalk(ChromSeq=ChromSeq[0])

        elif mode == 'ndb':

            if not ChromSeq is None:
                print("Attention! Sequence files are not considered for 'ndb' mode.")

            if CoordFiles is None:
                raise ValueError("Coordinate files required for mode '{:}'!".format(mode))

            return self.loadNDB(NDBfiles=CoordFiles)
            
        elif mode == 'pdb':
            if CoordFiles is None:
                raise ValueError("Coordinate files required for mode '{:}'!".format(mode))

            return self.loadPDB(PDBfiles=CoordFiles,ChromSeq=ChromSeq)

        elif mode == 'gro':

            if CoordFiles is None:
                raise ValueError("Coordinate files required for mode '{:}'!".format(mode))

            return self.loadGRO(GROfiles=CoordFiles,ChromSeq=ChromSeq)

        else:
            if mode != 'auto':
                raise ValueError("Mode '{:}' not supported!".format(mode))


    def initStorage(self, filename, mode="w"):
        
        R"""
        Initializes the *.cndb* files to store the chromosome structures. 
        
        Args:

            filename (str, required):
                 Filename of the cndb/h5dict storage file.
            mode (str, required):
                - 'w' - Create file, truncate if exists. (Default value = w).
                - 'w-' - Create file, fail if exists. 
                - 'r+' - Continue saving the structures in the same file that must exist.   
        """
        
        self.storage = []

        if mode not in ['w', 'w-', 'r+']:
            raise ValueError("Wrong mode to open file."
                             " Only 'w','w-' and 'r+' are supported")
        if (mode == "w-") and os.path.exists(filename):
            raise IOError("Cannot create file... file already exists."                          " Use mode ='w' to override")
        for k, chain in zip(range(len(self.chains)),self.chains):
            fname = os.path.join(self.folder, filename + '_' +str(k) + '.cndb')
            self.storage.append(h5py.File(fname, mode))    
            self.storage[k]['types'] = self.type_list_letter[chain[0]:chain[1]+1]

        if mode == "r+":
            myKeys = []
            for i in list(self.storage.keys()):
                try:
                    myKeys.append(int(i))
                except:
                    pass
            maxkey = max(myKeys) if myKeys else 1
            self.step = maxkey - 1
            self.setPositions(self.storage[str(maxkey - 1)])

                    
    def saveStructure(self, filename=None, mode="auto", h5dictKey="1", pdbGroups=None):
        R"""
        Save the 3D position of each bead of the chromosome polymer over the chromatin dynamics simulations.
        
        Args:

            filename (str, required):
                 Filename of the storage file.
            mode (str, required):
                - 'ndb' - The Nucleome Data Bank file format to save 3D structures of chromosomes. Please see the `NDB - Nucleome Data Bank <https://ndb.rice.edu/ndb-format>`__. for details.
                - 'cndb' - The compact ndb file format to save 3D structures of chromosomes. The binary format used the `hdf5 - Hierarchical Data Format <https://www.hdfgroup.org/solutions/hdf5/>`__ to store the data. Please see the NDB server for details. (Default value = cndb).
                - 'pdb' - The Protein Data Bank file format. Here, the chromosome is considered to be a protein where the locus is set at the carbon alpha position. This trick helps to use the standard macromolecules visualization software.  
                - 'gro' - The GROMACS file format. Initially, the MiChroM energy function was implemented in GROMACS. Details on how to run and use these files can be found at the `Nucleome Data Bank <https://ndb.rice.edu/GromacsInput-Documentation>`__.
                - 'xyz' - A XYZ file format.
                
        """
        
        
        data = self.getPositions()
        
        if filename is None:
            filename = self.name +"_block%d." % self.step + mode

        filename = os.path.join(self.folder, filename)
        
        if not hasattr(self, "type_list_letter"):
            raise ValueError("Chromatin sequence not defined!")
        
        if mode == "auto":
            if hasattr(self, "storage"):
                mode = "h5dict"
            else:
                mode = 'ndb'

        if mode == "h5dict":
            if not hasattr(self, "storage"):
                raise Exception("Cannot save to h5dict!"                                    " Initialize storage first!")
            for k, chain in zip(range(len(self.chains)),self.chains):
                self.storage[k][str(self.step)] = data[chain[0]:chain[1]+1]
            return
        
        elif mode == "xyz":
            lines = []
            lines.append(str(len(data)) + "\n")

            for particle in data:
                lines.append("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
            if filename == None:
                return lines
            elif isinstance(filename, string_types):
                with open(filename, 'w') as myfile:
                    myfile.writelines(lines)
            else:
                return lines

        elif mode == 'pdb':
            atom = "ATOM  {0:5d} {1:^4s}{2:1s}{3:3s} {4:1s}{5:4d}{6:1s}   {7:8.3f}{8:8.3f}{9:8.3f}{10:6.2f}{11:6.2f}          {12:>2s}{13:2s}"
            ter = "TER   {0:5d}      {1:3s} {2:1s}{3:4d}{4:1s}"
            model = "MODEL     {0:4d}"
            title = "TITLE     {0:70s}"

            Res_conversion = {'A1':'ASP', 'A2':'GLU', 'B1':'HIS', 'B2':'LYS', 'B3':'ARG', 'B4':'ARG', 'NA':'ASN'}
            Type_conversion = {0:'CA',1:'CA',2:'CA',3:'CA',4:'CA',5:'CA',6:'CA'}
            
            for chainNum, chain in zip(range(len(self.chains)),self.chains):
                pdb_string = []
                filename = self.name +"_" + str(chainNum) + "_block%d." % self.step + mode

                filename = os.path.join(self.folder, filename)
                data_chain = data[chain[0]:chain[1]+1]
                types_chain = self.type_list_letter[chain[0]:chain[1]+1] 

                pdb_string.append(title.format(self.name + " - chain " + str(chainNum)))
                pdb_string.append(model.format(0))

                totalAtom = 1
                for i, line in zip(types_chain, data_chain):
                    if not i in Res_conversion:
                        Res_conversion[i] = 'GLY'

                    pdb_string.append(atom.format(totalAtom,"CA","",Res_conversion[i],"",totalAtom,"",line[0], line[1], line[2], 1.00, 0.00, 'C', ''))
                    totalAtom += 1
                
                pdb_string.append(ter.format(totalAtom,Res_conversion[i],"",totalAtom,""))
                pdb_string.append("ENDMDL")
                np.savetxt(filename,pdb_string,fmt="%s")

                    
        elif mode == 'gro':
            
            gro_style = "{0:5d}{1:5s}{2:5s}{3:5d}{4:8.3f}{5:8.3f}{6:8.3f}"
            gro_box_string = "{0:10.5f}{1:10.5f}{2:10.5f}"

            Res_conversion = {'A1':'ASP', 'A2':'GLU', 'B1':'HIS', 'B2':'LYS', 'B3':'ARG', 'B4':'ARG', 'NA':'ASN'}
            Type_conversion = {'A1':'CA', 'A2':'CA', 'B1':'CA', 'B2':'CA', 'B3':'CA', 'B4':'CA', 'NA':'CA'}
            
            for chainNum, chain in zip(range(len(self.chains)),self.chains):
                
                gro_string = []
                filename = self.name +"_" + str(chainNum) + "_block%d." % self.step + mode
                filename = os.path.join(self.folder, filename)
                
                data_chain = data[chain[0]:chain[1]+1] 
                types_chain = self.type_list_letter[chain[0]:chain[1]+1] 

                gro_string.append(self.name +"_" + str(chainNum))
                gro_string.append(len(data_chain))
                
                totalAtom = 1
                for i, line in zip(types_chain, data_chain):
                    if not i in Res_conversion:
                        Res_conversion[i] = 'GLY'
                        Type_conversion[i] = 'CA'
                    
                    gro_string.append(str(gro_style.format(totalAtom, Res_conversion[i],Type_conversion[i],totalAtom,
                                    float(line[0]), float(line[1]), float(line[2]))))
                    
                    totalAtom += 1
                        
                gro_string.append(str(gro_box_string.format(0.000,0.000,0.000)))
                np.savetxt(filename,gro_string,fmt="%s")
        
        elif mode == 'ndb':
            ndb_string     = "{0:6s} {1:8d} {2:2s} {3:6s} {4:4s} {5:8d} {6:8.3f} {7:8.3f} {8:8.3f} {9:10d} {10:10d} {11:8.3f}"
            header_string  = "{0:6s}    {1:40s}{2:9s}   {3:4s}"
            title_string   = "{0:6s}  {1:2s}{2:80s}"
            author_string  = "{0:6s}  {1:2s}{2:79s}"
            expdata_string = "{0:6s}  {1:2s}{2:79s}"
            model_string   = "{0:6s}     {1:4d}"
            seqchr_string  = "{0:6s} {1:3d} {2:2s} {3:5d}  {4:69s}" 
            ter_string     = "{0:6s} {1:8d} {2:2s}        {3:2s}" 
            loops_string   = "{0:6s}{1:6d} {2:6d}"
            master_string  = "{0:6s} {1:8d} {2:6d} {3:6d} {4:10d}" 
            Type_conversion = {0:'A1',1:'A2',2:'B1',3:'B2',4:'B3',5:'B4',6:'NA'}
            
            def chunks(l, n):
                n = max(1, n)
                return ([l[i:i+n] for i in range(0, len(l), n)])
            
            for chainNum, chain in zip(range(len(self.chains)),self.chains):
                filename = self.name +"_" + str(chainNum) + "_block%d." % self.step + mode
                ndbf = []
                
                filename = os.path.join(self.folder, filename)
                data_chain = data[chain[0]:chain[1]+1]
                
                ndbf.append(header_string.format('HEADER','NDB File genereted by Open-MiChroM'," ", " "))
                ndbf.append(title_string.format('TITLE ','  ','A Scalable Computational Approach for '))
                ndbf.append(title_string.format('TITLE ','2 ','Simulating Complexes of Multiple Chromosomes'))
                ndbf.append(expdata_string.format('EXPDTA','  ','Cell Line  @50k bp resolution'))
                ndbf.append(expdata_string.format('EXPDTA','  ','Simulation - Open-MiChroM'))
                ndbf.append(author_string.format('AUTHOR','  ','Antonio B. Oliveira Junior - 2020'))
                
                seqList = self.type_list_letter[chain[0]:chain[1]+1]

                if len(self.diff_types) == len(self.data):
                    seqList = ['SQ' for x in range(len(self.type_list_letter))]

                seqChunk = chunks(seqList,23)
                
                for num, line in enumerate(seqChunk):
                    ndbf.append(seqchr_string.format("SEQCHR", num+1, "C1", len(seqList)," ".join(line)))
                ndbf.append("MODEL 1")
                
                for i, line in zip(list(range(len(data_chain))), data_chain):
                    ndbf.append(ndb_string.format("CHROM", i+1, seqList[i]," ","C1",i+1,
                                        line[0], line[1], line[2],
                                        int((i) * 50000)+1, int(i * 50000+50000), 0))
                ndbf.append("END")
                
                if hasattr(self, "loopPosition"):
                    loops = self.loopPosition[chain[0]:chain[1]+1]
                    loops.sort()
                    for p in loops:
                        ndbf.append(loops_string.format("LOOPS",p[0],p[1]))
                    
                np.savetxt(filename,ndbf,fmt="%s")
        
    def runSimBlock(self, steps=None, increment=True, num=None):
        R"""
        Performs a block of simulation steps.
        
        Args:

            steps (int, required):
                 Number of steps to perform in the block.
            increment (bool, optional):
                 Whether to increment the steps counter. Typically it is set :code:`False` during the collapse or equilibration simulations. (Default value: :code:`True`).
            num (int or None, required):
                 The number of subblocks to split the steps of the primary block. (Default value: :code:`None`).                
        """

        if self.forcesApplied == False:
            if self.verbose:
                print("applying forces")
                stdout.flush()
            self._applyForces()
            self.forcesApplied = True
        if increment == True:
            self.step += 1
        if steps is None:
            steps = self.steps_per_block
        if (increment == True) and ((self.step % 50) == 0):
            self.printStats()

        for attempt in range(6):
            print("bl=%d" % (self.step), end=' ')
            stdout.flush()
            if self.verbose:
                print()
                stdout.flush()

            if num is None:
                num = steps // 5 + 1
            a = time.time()
            for _ in range(steps // num):
                if self.verbose:
                    print("performing integration")
                self.integrator.step(num)  # integrate!
                stdout.flush()
            if (steps % num) > 0:
                self.integrator.step(steps % num)

            self.state = self.context.getState(getPositions=True,
                                               getEnergy=True)

            b = time.time()
            coords = self.state.getPositions(asNumpy=True)
            newcoords = coords / self.nm

            eK = (self.state.getKineticEnergy() / self.N / units.kilojoule_per_mole)
            eP = self.state.getPotentialEnergy() / self.N / units.kilojoule_per_mole


            if self.velocityReinitialize:
                if eK > 5.0:
                    print("(i)", end=' ')
                    self.initVelocities()
            print("pos[1]=[%.1lf %.1lf %.1lf]" % tuple(newcoords[0]), end=' ')


            if ((np.isnan(newcoords).any()) or (eK > self.eKcritical) or
                (np.isnan(eK)) or (np.isnan(eP))):

                self.context.setPositions(self.data)
                self.initVelocities()
                print("eK={0}, eP={1}, trying one more time at step {2} ".format(eK, eP, self.step))
            else:
                dif = np.sqrt(np.mean(np.sum((newcoords -
                    self.getPositions()) ** 2, axis=1)))
                print("dr=%.2lf" % (dif,), end=' ')
                self.data = coords
                print("t=%2.1lfps" % (self.state.getTime() / (units.second * 1e-12)), end=' ')
                print("kin=%.2lf pot=%.2lf" % (eK,
                    eP), "Rg=%.3lf" % self.chromRG(), end=' ')
                print("SPS=%.0lf" % (steps / (float(b - a))), end=' ')

                if (self.integrator_type.lower() == 'variablelangevin'
                    or self.integrator_type.lower() == 'variableverlet'):
                    dt = self.integrator.getStepSize()
                    mass = self.system.getParticleMass(1)
                    dx = (units.sqrt(2.0 * eK * self.kT / mass) * dt)
                    print('dx=%.2lfpm' % (dx / self.nm * 1000.0), end=' ')

                print("")
                break

        return {"Ep":eP, "Ek":eK}
        
        
    def initPositions(self):
        
        R"""
        Internal function that sends the locus coordinates to OpenMM system. 
        """

        print("Positions... ")
        try:
            self.context
        except:
            raise ValueError("No context, cannot set velocs."                             " Initialize context before that")

        self.context.setPositions(self.data)
        print(" loaded!")
        state = self.context.getState(getPositions=True, getEnergy=True)
        
        eP = state.getPotentialEnergy() / self.N / units.kilojoule_per_mole
        print("potential energy is %lf" % eP)
        
    def initVelocities(self, mult=1.0):
        R"""
        Internal function that set the locus velocity to OpenMM system. 
        
        Args:

            mult (float, optional):
                 Rescale initial velocities. (Default value = 1.0). 
        """
        try:
            self.context
        except:
            raise ValueError("No context, cannot set velocs."                             "Initialize context before that")

        sigma = units.sqrt(self.Epsilon*units.kilojoule_per_mole / self.system.getParticleMass(
            1)) 
        velocs = units.Quantity(mult * np.random.normal(
            size=(self.N, 3)), units.meter) * (sigma / units.meter)

        self.context.setVelocities(velocs) 
        
    def setFibPosition(self, positions, returnCM=False, factor=1.0):
        R"""
        Distributes the center of mass of chromosomes on the surface of a sphere according to the Fibonacci Sphere algorithm.
        
        Args:

            positions (:math:`(Nbeads, 3)` :class:`numpy.ndarray`, required):
                The array of positions of the chromosome chains to be distributed in the sphere surface.
            returnCM (bool, optional):
                Whether to return an array with the center of mass of the chromosomes. (Default value: :code:`False`).
            factor (float, optional):
                Scale coefficient to be multiplied to the radius of the nucleus, determining the radius of the sphere
                in which the center of mass of chromosomes will be distributed. The radius of the nucleus is calculated 
                based on the number of beads to generate a volume density of 0.1. 

                :math:`R_{sphere} = factor * R_{nucleus}`
                
        Returns:

            :math:`(Nbeads, 3)` :class:`numpy.ndarray`:
                Returns an array of positions to be loaded into OpenMM using the function :class:`loadStructure`.

            :math:`(Nchains, 3)` :class:`numpy.ndarray`:
                Returns an array with the new coordinates of the center of mass of each chain.

        """
        
        def fibonacciSphere(samples=1, randomize=True):
            R"""
            Internal function for running the Fibonacci Sphere algorithm.
            """
            rnd = 1.
            if randomize:
                rnd = random.random() * samples

            points = []
            offset = 2./samples
            increment = np.pi * (3. - np.sqrt(5.))

            for i in range(samples):
                y = ((i * offset) - 1) + (offset / 2)
                r = np.sqrt(1 - y**2)
                phi = ((i + rnd) % samples) * increment
                x = np.cos(phi) * r
                z = np.sin(phi) * r
                points.append([x,y,z])

            np.random.shuffle(points)
            
            return points
    
        points = fibonacciSphere(len(self.chains))
        
        R_nucleus = ( (self.chains[-1][1]+1) * (1.0/2.0)**3 / 0.1 )**(1.0/3.0)
        
        for i in range(len(self.chains)):
            points[i] = [ x * factor * R_nucleus for x in points[i]]
            positions[self.chains[i][0]:self.chains[i][1]+1] += np.array(points[i])
            
        if returnCM:
            return positions,points
        else:
            return positions
        
    def chromRG(self):
        R"""
        Calculates the Radius of Gyration of a chromosome chain.
        
        Returns:
                Returns the Radius of Gyration in units of :math:`\sigma`
        """
        data = self.getScaledData()
        data = data - np.mean(data, axis=0)[None,:]
        return np.sqrt(np.sum(np.var(np.array(data), 0)))
    
    def getScaledData(self):
        R"""
        Internal function for keeping the system in the simulation box if PBC is employed.
        """
        if self.PBC != True:
            return self.getPositions()
        alldata = self.getPositions()
        boxsize = np.array(self.BoxSizeReal)
        mults = np.floor(alldata / boxsize[None, :])
        toRet = alldata - mults * boxsize[None, :]
        assert toRet.min() >= 0
        return toRet
        
    def printStats(self):
        R"""
        Prints some statistical information of a system.
        """
        state = self.context.getState(getPositions=True,
            getVelocities=True, getEnergy=True)

        eP = state.getPotentialEnergy()
        pos = np.array(state.getPositions() / (units.meter * 1e-9))
        bonds = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
        
        # delete "bonds" from different chains
        indexInDifferentChains = [x[1] for x in self.chains]
        bonds = np.delete(bonds, indexInDifferentChains[:-1])

        sbonds = np.sort(bonds)
        vel = state.getVelocities()
        mass = self.system.getParticleMass(1)
        vkT = np.array(vel / units.sqrt(self.Epsilon*units.kilojoule_per_mole / mass), dtype=float)
        self.velocs = vkT
        EkPerParticle = 0.5 * np.sum(vkT ** 2, axis=1)

        cm = np.mean(pos, axis=0)
        centredPos = pos - cm[None, :]
        dists = np.sqrt(np.sum(centredPos ** 2, axis=1))
        per95 = np.percentile(dists, 95)
        den = (0.95 * self.N) / ((4. * np.pi * per95 ** 3) / 3)
        per5 = np.percentile(dists, 5)
        den5 = (0.05 * self.N) / ((4. * np.pi * per5 ** 3) / 3)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        minmedmax = lambda x: (x.min(), np.median(x), x.mean(), x.max())

        print()
        print("Statistics for the simulation %s, number of particles: %d, "        " number of chains: %d" % (
            self.name, self.N, len(self.chains)))
        print()
        print("Statistics for particle position")
        print("     mean position is: ", np.mean(
            pos, axis=0), "  Rg = ", self.chromRG())
        print("     median bond size is ", np.median(bonds))
        print("     three shortest/longest (<10)/ bonds are ", sbonds[
            :3], "  ", sbonds[sbonds < 10][-3:])
        if (sbonds > 10).sum() > 0:
            print("longest 10 bonds are", sbonds[-10:])

        print("     95 percentile of distance to center is:   ", per95)
        print("     density of closest 95% monomers is:   ", den)
        print("     density of the core monomers is:   ", den5)
        print("     min/median/mean/max coordinates are: ")
        print("     x: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(x))
        print("     y: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(y))
        print("     z: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(z))
        print()
        print("Statistics for velocities:")
        print("     mean kinetic energy is: ", np.mean(
            EkPerParticle), "should be:", 1.5)
        print("     fastest particles are (in kT): ", np.sort(
            EkPerParticle)[-5:])

        print()
        print("Statistics for the system:")
        print("     Forces are: ", list(self.forceDict.keys()))
        print("     Number of exceptions:  ", len(self.bondsForException))
        print()
        print("Potential Energy Ep = ", eP / self.N / units.kilojoule_per_mole)
        
    def printForces(self):
        R"""
        Prints the energy values for each force applied in the system.
        """
        forceNames = []
        forceValues = []
        
        for n in (self.forceDict):
            forceNames.append(n)
            forceValues.append(self.context.getState(getEnergy=True, groups={self.forceDict[n].getForceGroup()}).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole))
        forceNames.append('Potential Energy (total)')
        forceValues.append(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole))
        forceNames.append('Potential Energy (per loci)')
        forceValues.append(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(units.kilojoules_per_mole)/self.N)
        df = pd.DataFrame(forceValues,forceNames)
        df.columns = ['Values']
        print(df)

    def printHeader(self):
        print('{:^96s}'.format("***************************************************************************************"))
        print('{:^96s}'.format("**** **** *** *** *** *** *** *** OpenMiChroM-1.0.7 *** *** *** *** *** *** **** ****"))
        print('')
        print('{:^96s}'.format("OpenMiChroM is a Python library for performing chromatin dynamics simulations."))
        print('{:^96s}'.format("OpenMiChroM uses the OpenMM Python API,"))
        print('{:^96s}'.format("employing the MiChroM (Minimal Chromatin Model) energy function."))
        print('{:^96s}'.format("The chromatin dynamics simulations generate an ensemble of 3D chromosomal structures"))
        print('{:^96s}'.format("that are consistent with experimental Hi-C maps, also allows simulations of a single"))
        print('{:^96s}'.format("or multiple chromosome chain using High-Performance Computing "))
        print('{:^96s}'.format("in different platforms (GPUs and CPUs)."))
        print('{:^96s}'.format("OpenMiChroM documentation is available at https://open-michrom.readthedocs.io"))
        print('')
        print('{:^96s}'.format("OpenMiChroM is described in: Oliveira Junior, A. B & Contessoto, V, G et. al."))
        print('{:^96s}'.format("A Scalable Computational Approach for Simulating Complexes of Multiple Chromosomes."))
        print('{:^96s}'.format("Journal of Molecular Biology. doi:10.1016/j.jmb.2020.10.034."))
        print('{:^96s}'.format("and"))
        print('{:^96s}'.format("Oliveira Junior, A. B. et al."))
        print('{:^96s}'.format("Chromosome Modeling on Downsampled Hi-C Maps Enhances the Compartmentalization Signal."))
        print('{:^96s}'.format("J. Phys. Chem. B, doi:10.1021/acs.jpcb.1c04174."))
        print('')
        print('{:^96s}'.format("Copyright (c) 2023, The OpenMiChroM development team at"))
        print('{:^96s}'.format("Rice University"))
        print('{:^96s}'.format("***************************************************************************************"))
        stdout.flush()
