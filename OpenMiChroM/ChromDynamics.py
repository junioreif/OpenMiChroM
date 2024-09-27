# Copyright (c) 2020-2024 The Center for Theoretical Biological Physics (CTBP)
# Rice University
# This file is from the Open-MiChroM project, released under the MIT License.

R"""
The `ChromDynamics` classes perform chromatin dynamics based on the compartment
annotation sequences of chromosomes. Simulations can be performed either using
the default parameters of MiChroM (Minimal Chromatin Model) or using custom
values for the type-to-type and Ideal Chromosome parameters.

Details about the MiChroM energy function and default parameters are described in:
Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G., & Onuchic, J.N. (2016).
Transferable model for chromosome architecture. *Proceedings of the National Academy of Sciences*, 113(43), 12168-12173.
"""

# Import OpenMM, handling compatibility with different versions
try:
    # For OpenMM versions >= 7.7.0
    from openmm.app import *
    import openmm
    import openmm.unit as units
except ImportError:
    # Fallback for earlier versions
    print("Unable to load OpenMM as 'openmm'. Trying 'simtk.openmm'...")
    try:
        from simtk.openmm.app import *
        import simtk.openmm as openmm
        import simtk.unit as units
    except ImportError:
        raise ImportError("Failed to load OpenMM. Please check your installation and configuration.")

from sys import stdout
import warnings
import numpy as np
import os
import random
import pandas as pd
from pathlib import Path
import types
from textwrap import dedent
from . import __version__
from .CustomReporter import *

class MiChroM:
    R"""
    The `MiChroM` class performs chromatin dynamics simulations using the default MiChroM energy function parameters for type-to-type and Ideal Chromosome interactions.

    Details about the MiChroM (Minimal Chromatin Model) energy function and the default parameters are described in:
    Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G., & Onuchic, J.N. (2016). Transferable model for chromosome architecture. *Proceedings of the National Academy of Sciences*, 113(43), 12168-12173.

    The `MiChroM` class sets up the environment to start chromatin dynamics simulations.

    Args:
        name (str, optional): 
            Name used in the output files. Defaults to "OpenMichrom".
        timeStep (float, optional): 
            Simulation time step in units of τ. Defaults to 0.01.
        collisionRate (float, optional): 
            Friction/damping constant in units of reciprocal time (1/τ). Defaults to 0.1.
        temperature (float, optional): 
            Temperature in reduced units. Defaults to 1.0.
    """
    def __init__(self, name="OpenMiChroM", timeStep=0.01, collisionRate=0.1, temperature=1.0):
        self.name = name
        self.timeStep = timeStep
        self.collisionRate = collisionRate
        self.temperature = temperature / 0.008314
        self.loaded = False
        self.contexted = False
        self.folder = "."
        self.nm = units.meter * 1e-9
        self.sigma = 1.0
        self.epsilon = 1.0
        self.printHeader()
        print(f"TEST VERSION {__version__}")

            
    def setup(self, platform="CUDA", gpu="default",
            integrator="langevin", precision="mixed", deviceIndex="0"):
        R"""Sets up the simulation environment.

        Tries to select the computational platform in the following priority order:
        the specified platform, then 'CUDA', 'HIP', 'OpenCL', and 'CPU'. If the
        preferred platform is not available, it will attempt to use the next
        available platform in the list and print an informational message.

        Args:
            platform (str, optional): The preferred computation platform to use.
                Defaults to 'CUDA'.
            gpu (str, optional): GPU device index or 'default'. Defaults to 'default'.
            integrator (str or OpenMM Integrator, optional): The integrator to use for the simulation.
                Can be a string (e.g., 'langevin') or an OpenMM Integrator object.
                Defaults to 'langevin'.
            precision (str, optional): The floating point precision to use ('mixed', 'single', or 'double').
                Defaults to 'mixed'.
            deviceIndex (str, optional): The device index to use if specifying a GPU device.
                Defaults to '0'.

        Raises:
            ValueError: If an unknown integrator or precision is specified.
            Exception: If no suitable computational platform is available.
        """
        precision = precision.lower()
        if precision not in ["mixed", "single", "double"]:
            raise ValueError("Precision must be 'mixed', 'single', or 'double'.")

        self.kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
        self.kT = self.kB * self.temperature
        self.mass = 10.0 * units.amu
        self.bondsForException = []
        self.mm = openmm
        self.system = self.mm.System()
        self.step = 0
        self.gpu = str(gpu)

        # Define the platform priority order
        default_platforms = ['CUDA', 'HIP', 'OpenCL', 'CPU']

        # Rearrange the platform priority so that the specified platform is first
        preferred_platform = platform.upper()
        platform_priority = [preferred_platform] + [p for p in default_platforms if p != preferred_platform]

        # Dictionary to map platform names to their specific property names
        property_names = {
            'CUDA': {'DeviceIndex': 'CudaDeviceIndex', 'Precision': 'CudaPrecision'},
            'HIP': {'DeviceIndex': 'HipDeviceIndex', 'Precision': 'HipPrecision'},
            'OPENCL': {'DeviceIndex': 'OpenCLDeviceIndex', 'Precision': 'OpenCLPrecision'},
            'CPU': {}
        }

        # Attempt to set up the platform
        for plat_name in platform_priority:
            try:
                self.platform = self.mm.Platform.getPlatformByName(plat_name)
                print(f"Using platform: {plat_name}")

                # Set platform-specific properties
                properties = {}
                if plat_name != 'CPU':
                    if self.gpu.lower() != "default":
                        properties[property_names[plat_name]['DeviceIndex']] = deviceIndex
                    properties[property_names[plat_name]['Precision']] = precision
                self.properties = properties
                break
            except Exception as e:
                print(f"Platform '{plat_name}' is not available: {e}")
        else:
            raise Exception("No suitable computational platform is available.")

        self.forceDict = {}

        if isinstance(integrator, str):
            integrator_name = integrator.lower()
            if integrator_name == "langevin":
                self.integrator = self.mm.LangevinIntegrator(
                    self.temperature, self.collisionRate, self.timeStep)
                self.integrator_type = "Langevin"
            else:
                raise ValueError(f"Unknown integrator '{integrator}'.")
        else:
            self.integrator = integrator
            self.integrator_type = "UserDefined"
            

    def saveFolder(self, folderPath):
        R"""Sets the folder path to save data.

        Args:
            folderPath (str): The folder path where simulation data will be saved.
                If the folder does not exist, it will be created.
        """
        os.makedirs(folderPath, exist_ok=True)
        self.folder = folderPath

        
    def loadStructure(self, data, center=True, massList=None):
        R"""Loads the 3D positions of each bead of the chromosome polymer into the OpenMM system.

        Args:
            data (array-like): The initial positions of the beads. Should be an array of shape (N, 3) or (3, N).
            center (bool, optional): If True, centers the chromosome's center of mass at [0, 0, 0]. Defaults to True.
            massList (array-like, optional): Masses of each chromosome bead in units of μ. If None, all masses are set to 1.0. Defaults to None.

        Raises:
            ValueError: If the input data is not in the correct format or contains NaN values.
        """
        data = np.asarray(data, dtype=float)

        if data.shape[1] != 3:
            raise ValueError("Input data must have shape (N, 3).")

        if np.isnan(data).any():
            raise ValueError("Input data contains NaN values.")

        if center:
            data -= np.mean(data, axis=0)

        self.setPositions(data)

        if massList is None:
            self.masses = np.ones(self.N)
        else:
            massList = np.asarray(massList, dtype=float)

            if len(massList) != self.N:
                raise ValueError(f"Mass list length {len(massList)} does not match number of beads {self.N}.")
            self.masses = massList

        if not hasattr(self, "chains"):
            self.setChains()


    def setChains(self, chains=None):
        R"""Sets the configuration of the chains in the system.

        This information is used later for adding bonds and angles in the homopolymer potential.

        Args:
            chains (list of tuples, optional): A list of chains in the format `[(start, end, isRing)]`.
                `isRing` is a boolean indicating whether the chromosome chain is circular or not (used to
                simulate bacterial genomes, for example). The particle range should be semi-open; for
                example, a chain `(0, 3, False)` links the particles `0`, `1`, and `2`. If `isRing` is
                `True`, the first and last particles of the chain are linked, forming a ring. Defaults to
                `[(0, None, False)]`, which links all particles of the system into one chain.
        """
        if chains is None:
            chains = [(0, None, False)]
        else:
            # Validate and process the chains
            validated_chains = []
            for chain in chains:
                start, end, isRing = chain

                isRing = bool(isRing)
                validated_chains.append((start, end, isRing))

            chains = validated_chains

        self.chains = chains.copy()

            
    def setPositions(self, beadsPos, randomize=False, randomOffset=1e-5):
        R"""Sets the 3D positions of each bead of the chromosome polymer in the OpenMM system.

        Args:
            beadsPos (numpy.ndarray of shape (N, 3)):
                Array of XYZ positions for each bead (locus) in the polymer model.
            randomize (bool, optional):
                If True, adds a small random offset to the positions to avoid numerical instability.
                Defaults to False.
            randomOffset (float, optional):
                The magnitude of the random offset to be added if randomize is True.
                Defaults to 1e-5.
        """
        data = np.asarray(beadsPos, dtype=float)
        if randomize:
            data += (np.random.random(data.shape) * 2 - 1) * randomOffset

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
    def addCylindricalConfinement(self, rConf=5.0, zConf=10.0, kConf=30.0):
        R"""Adds a cylindrical confinement potential to the system.

        This potential confines particles within a cylinder of radius `rConf` and height `2 * zConf`.
        Particles outside this cylinder experience a harmonic restoring force pushing them back inside.

        Args:
            rConf (float, optional): Radius of the cylindrical confinement in nanometers. Defaults to 5.0.
            zConf (float, optional): Half-height of the cylindrical confinement along the z-axis in nanometers. Defaults to 10.0.
            kConf (float, optional): Force constant (stiffness) of the confinement potential. Defaults to 30.0.
        """
        cylConfEnergy = (
            "step(r_xy - r_cyn) * 0.5 * k_cyn * (r_xy - r_cyn)^2 + "
            "step(abs(z) - zconf) * 0.5 * k_cyn * (abs(z) - zconf)^2;"
            "r_xy = sqrt(x^2 + y^2)"
        )
        cylConfForce = self.mm.CustomExternalForce(cylConfEnergy)
        cylConfForce.addGlobalParameter('r_cyn', rConf)
        cylConfForce.addGlobalParameter('k_cyn', kConf)
        cylConfForce.addGlobalParameter('zconf', zConf)

        self.forceDict["CylindricalConfinement"] = cylConfForce

        for i in range(self.N):
            self.forceDict["CylindricalConfinement"].addParticle(i, [])

    
    def addFlatBottomHarmonic(self, kR=5e-3, nRad=10.0):
        R"""
        Adds a flat-bottom harmonic potential to confine the chromosome chain inside the nucleus wall.

        The potential is defined as:
            V(r) = step(r - r0) * (kR / 2) * (r - r0)^2

        where:
            - `r` is the distance from the origin (center of the nucleus)
            - `r0` (nRad) is the nucleus radius
            - `kR` is the spring constant of the potential

        This potential applies no force when particles are inside the nucleus (r ≤ r0) and applies a harmonic restoring force when particles are outside the nucleus (r > r0).

        Args:
            kR (float, optional):
                Spring constant of the harmonic potential. Defaults to 5e-3.
            nRad (float, optional):
                Nucleus radius in units of σ. Defaults to 10.0.
        """
        # Define the energy expression for the flat-bottom harmonic potential
        energyExpression = (
            "step(r - rRes) * 0.5 * kR * (r - rRes)^2;"
            "r = sqrt(x^2 + y^2 + z^2)"
        )

        # Create the custom external force using the energy expression
        restraintForce = self.mm.CustomExternalForce(energyExpression)
        restraintForce.addGlobalParameter('rRes', nRad)
        restraintForce.addGlobalParameter('kR', kR)

        # Apply the force to all particles in the system
        for i in range(self.N):
            restraintForce.addParticle(i, [])

        # Add the force to the force dictionary
        self.forceDict["FlatBottomHarmonic"] = restraintForce


    def addSphericalConfinementLJ(self, radius="density", density=0.1):
        R"""
        Adds a spherical confinement potential to the system according to the MiChroM energy function.

        This potential describes the interaction between the chromosome and a spherical wall,
        effectively confining the particles within a sphere of specified radius.

        Args:
            radius (float or str, optional):
                Radius of the spherical confinement. If set to "density", the radius is calculated
                based on the specified density. Defaults to "density".
            density (float, optional):
                Density of the chromosome beads inside the nucleus. Required if `radius` is "density".
                Defaults to 0.1.

        Notes:
            - If `radius` is "density", the radius is calculated using the formula:

            radius = (3 * N / (4 * π * density)) ** (1/3)

            where N is the number of particles in the system.
            - The confinement potential is modeled using a shifted Lennard-Jones potential.
        """
        # Define the energy expression for the spherical confinement using a shifted Lennard-Jones potential
        energyExpression = (
            "(4 * epsilon * ((sigma/deltaR)^12 - (sigma/deltaR)^6) + epsilon) * step(cutoff - deltaR);"
            "deltaR = R - sqrt(x^2 + y^2 + z^2)"
        )

        # Create the custom external force using the energy expression
        sphericalForce = self.mm.CustomExternalForce(energyExpression)

        # Calculate radius if set to "density"
        if radius == "density":
            radius = (3 * self.N / (4 * 3.141592653589793 * density)) ** (1 / 3.)

        self.sphericalConfinementRadius = radius

        # Add global parameters to the force
        sphericalForce.addGlobalParameter('R', radius)
        sphericalForce.addGlobalParameter('epsilon', 1.0)
        sphericalForce.addGlobalParameter('sigma', 1.0)
        sphericalForce.addGlobalParameter('cutoff', 2.0 ** (1.0 / 6.0))

        # Apply the force to all particles in the system
        for i in range(self.N):
            sphericalForce.addParticle(i, [])

        # Add the force to the force dictionary
        self.forceDict["SphericalConfinementLJ"] = sphericalForce

        
    def addFENEBonds(self, kFb=30.0, bonds=None):
        R"""
        Adds FENE (Finite Extensible Nonlinear Elastic) bonds to the system.

        This function initializes the FENE bond force if it has not been added yet, and adds bonds between specified pairs of loci.
        By default, if no bonds are specified, it will add bonds between neighboring loci in each chain according to the chain definitions.
        If a chain is specified as a ring, it will also add a bond between the first and last loci of that chain.

        The FENE bonds are defined according to the method described in:
        Halverson, J.D., Lee, W.B., Grest, G.S., Grosberg, A.Y., & Kremer, K. (2011).
        Molecular dynamics simulation study of nonconcatenated ring polymers in a melt. I. Statics.
        The Journal of Chemical Physics, 134(20), 204904.

        Args:
            kFb (float, optional):
                Bond coefficient (force constant) for the FENE bonds. Defaults to 30.0.
            bonds (list of tuples, optional):
                A list of tuples specifying the pairs of loci to bond. Each tuple is (i, j).
                If None, bonds between neighboring loci in the chains will be added. Defaults to None.

        Raises:
            ValueError: If any of the loci indices are out of bounds.
        """
        # Initialize the FENE bond force if not already done
        if "FENEBond" not in self.forceDict:
            # Define the FENE bond potential
            feneEnergy = (
                "-0.5 * kFb * fr0^2 * log(1 - (r / fr0)^2) + "
                "(4 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon) * step(cutoff - r)"
            )
            feneBondForce = self.mm.CustomBondForce(feneEnergy)
            feneBondForce.addGlobalParameter("kFb", kFb)
            feneBondForce.addGlobalParameter("fr0", 1.5)
            feneBondForce.addGlobalParameter("epsilon", 1.0)
            feneBondForce.addGlobalParameter("sigma", 1.0)
            feneBondForce.addGlobalParameter("cutoff", 2.0 ** (1.0 / 6.0))
            self.forceDict["FENEBond"] = feneBondForce

        if bonds is None:
            # Add bonds between neighboring loci in the chains
            for start, end, isRing in self.chains:
                for j in range(start, end):
                    i1, i2 = j, j + 1
                    if i1 >= self.N or i2 >= self.N:
                        raise ValueError(
                            f"Cannot add a bond between beads {i1} and {i2}; indices are out of bounds for chromosome length {self.N}."
                        )
                    self.forceDict["FENEBond"].addBond(int(i1), int(i2), [])
                    self.bondsForException.append((i1, i2))
                if isRing:
                    i1, i2 = start, end
                    if i1 >= self.N or i2 >= self.N:
                        raise ValueError(
                            f"Cannot add a bond between beads {i1} and {i2}; indices are out of bounds for chromosome length {self.N}."
                        )
                    self.forceDict["FENEBond"].addBond(int(i1), int(i2), [])
                    self.bondsForException.append((i1, i2))
        else:
            # Add specified bonds
            for (i, j) in bonds:
                if i >= self.N or j >= self.N:
                    raise ValueError(
                        f"Cannot add a bond between beads {i} and {j}; indices are out of bounds for chromosome length {self.N}."
                    )
                self.forceDict["FENEBond"].addBond(int(i), int(j), [])
                self.bondsForException.append((i, j))


    def addAngles(self, kA=2.0):
        R"""
        Adds an angular potential between bonds connecting beads i-1, i, and i+1.

        This function adds angle forces to the system to enforce stiffness between sequential beads,
        according to the method described in:
        Halverson, J.D., Lee, W.B., Grest, G.S., Grosberg, A.Y., & Kremer, K. (2011).
        Molecular dynamics simulation study of nonconcatenated ring polymers in a melt. I. Statics.
        *The Journal of Chemical Physics*, 134(20), 204904.

        Args:
            kA (float or array-like, optional):
                Angle potential coefficient(s). If a single float is provided, the same coefficient is used
                for all angles. If an array is provided, it must have a length of N-2 (number of angles),
                and each angle will use the corresponding coefficient. Defaults to 2.0.

        Raises:
            ValueError: If kA is an array and its length does not match the expected number of angles.
        """
        # Determine the number of angles based on chains
        num_angles = 0
        for start, end, isRing in self.chains:
            num_angles += (end - start - 1)
            if isRing:
                num_angles += 2  # For the two additional angles in a ring

        # Ensure kA is an array of coefficients
        if np.isscalar(kA):
            kA_array = np.full(num_angles, kA, dtype=float)
        else:
            kA_array = np.asarray(kA, dtype=float)
            if len(kA_array) != num_angles:
                raise ValueError(
                    f"The length of kA ({len(kA_array)}) must match the number of angles ({num_angles})."
                )

        # Define the angle force expression
        angleForceExpression = "kA * (1 - cos(theta - pi))"

        # Create the custom angle force
        angleForce = self.mm.CustomAngleForce(angleForceExpression)
        angleForce.addPerAngleParameter("kA")
        angleForce.addGlobalParameter("pi", np.pi)

        angle_index = 0  # Index to track position in kA_array

        # Add angles for each chain
        for start, end, isRing in self.chains:
            # For linear chains
            for j in range(start + 1, end):
                i1, i2, i3 = j - 1, j, j + 1
                if i3 >= self.N:
                    break  # Avoid index out of range
                angleForce.addAngle(i1, i2, i3, [kA_array[angle_index]])
                angle_index += 1
            # For ring chains, add angles that wrap around
            if isRing:
                # Angle between last three beads
                i1, i2, i3 = end - 1, end, start
                angleForce.addAngle(i1, i2, i3, [kA_array[angle_index]])
                angle_index += 1
                # Angle wrapping from end to start + 1
                i1, i2, i3 = end, start, start + 1
                angleForce.addAngle(i1, i2, i3, [kA_array[angle_index]])
                angle_index += 1

        # Add the angle force to the system
        self.forceDict["AngleForce"] = angleForce


    def addRepulsiveSoftCore(self, eCut=4.0, cutoffDistance=3.0):
        R"""
            Adds a soft-core repulsive interaction that allows chain crossing, representing the activity of topoisomerase II.

            Details can be found in the following publications:

            - Oliveira Jr., A.B., Contessoto, V.G., Mello, M.F., & Onuchic, J.N. (2021). A scalable computational approach for simulating complexes of multiple chromosomes. *Journal of Molecular Biology*, 433(6), 166700.
            - Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G., & Onuchic, J.N. (2016). Transferable model for chromosome architecture. *Proceedings of the National Academy of Sciences*, 113(43), 12168-12173.
            - Naumova, N., Imakaev, M., Fudenberg, G., Zhan, Y., Lajoie, B.R., Mirny, L.A., & Dekker, J. (2013). Organization of the mitotic chromosome. *Science*, 342(6161), 948-953.

            Args:
                eCut (float, optional):
                    Energy cost for chain crossing in units of \(k_B T\). Defaults to 4.0.
                cutoffDistance (float, optional):
                    Cutoff distance for the nonbonded interactions. Defaults to 3.0.
        """
        # Calculate the nonbonded cutoff distance
        nbCutoffDist = self.sigma * 2.0 ** (1.0 / 6.0)

        # Scale eCut by epsilon
        eCut *= self.epsilon

        # Calculate r0 based on eCut and epsilon
        r0 = self.sigma * (((0.5 * eCut) / (4.0 * self.epsilon) - 0.25 + (0.5) ** 2.0) ** 0.5 + 0.5) ** (-1.0 / 6.0)

        # Define the energy expression for the repulsive soft-core potential
        repulEnergy = (
            "LJ * step(r - r0) * step(cutoff - r)"
            " + step(r0 - r) * 0.5 * eCut * (1.0 + tanh((2.0 * LJ / eCut) - 1.0));"
            "LJ = 4.0 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon"
        )

        # Create the custom nonbonded force
        repulForce = self.mm.CustomNonbondedForce(repulEnergy)
        repulForce.addGlobalParameter('epsilon', self.epsilon)
        repulForce.addGlobalParameter('sigma', self.sigma)
        repulForce.addGlobalParameter('eCut', eCut)
        repulForce.addGlobalParameter('r0', r0)
        repulForce.addGlobalParameter('cutoff', nbCutoffDist)
        repulForce.setCutoffDistance(cutoffDistance)

        # Add particles to the force
        for _ in range(self.N):
            repulForce.addParticle(())

        # Add the force to the force dictionary
        self.forceDict["RepulsiveSoftCore"] = repulForce

        
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


    def addCustomMultiChainIC(self, mu=3.22, rc = 1.78, dinit=3, dend=1000, chainIndex=None, IClist=None, CutoffDistance=3.0):

        energyIC = ("step(d-dinit)*ic(d)*step(dend-d)*f;"
                   "f=0.5*(1. + tanh(mu*(rc - r)));"
                   "d=abs(idx1-idx2)")
        
        IC = self.mm.CustomNonbondedForce(energyIC)

        IClist_listfromfile = np.loadtxt(IClist)
        IClist = np.zeros(self.N)
        for d, value in enumerate(IClist_listfromfile, start=dinit):
            IClist[d] = value
                
        tabIClist = self.mm.Discrete1DFunction(IClist)
        IC.addTabulatedFunction('ic', tabIClist)

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


        self.forceDict["CustomIC_chain_"+str(chainIndex)] = IC
        

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


    def addHarmonicBonds(self, bondStiffness=30.0, equilibriumDistance=1.0):
        R"""
        Adds harmonic bonds to all monomers within each chain. For each chain, bonds are created 
        between consecutive monomers. If the chain forms a ring, an additional bond is created 
        between the first and last monomer to complete the ring.

        Args:
            bondStiffness (float, optional): The stiffness of the bond. Default is 30.0.
            equilibriumDistance (float, optional): The equilibrium distance for the bond. Default is 1.0.
        """
        # Initialize the Harmonic Bond force group if it hasn't been initialized yet
        if "HarmonicBond" not in self.forceDict:
            bondForceExpression = "0.5 * kfb * (r - r0)^2"
            harmonicBondForce = self.mm.CustomBondForce(bondForceExpression)
            harmonicBondForce.addGlobalParameter("kfb", bondStiffness)
            harmonicBondForce.addGlobalParameter("r0", equilibriumDistance)
            self.forceDict["HarmonicBond"] = harmonicBondForce

        harmonicBondForce = self.forceDict["HarmonicBond"]

        for chain in self.chains:
            start, end, isRing = chain
            # Iterate through monomers in the chain to add consecutive bonds
            for j in range(start, end):
                i = j
                k = j + 1
                if i >= self.N or k >= self.N:
                    raise ValueError(
                        f"\nCannot add a bond between beads {i}, {k} as they exceed the chromosome length {self.N}."
                    )
                harmonicBondForce.addBond(int(i), int(k), [])
                self.bondsForException.append((int(i), int(k)))

            # If the chain forms a ring, add a bond between the first and last monomer
            if isRing:
                if start >= self.N or end >= self.N:
                    raise ValueError(
                        f"\nCannot add a bond between beads {start}, {end} as they exceed the chromosome length {self.N}."
                    )
                harmonicBondForce.addBond(int(start), int(end), [])
                self.bondsForException.append((int(start), int(end)))


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
        R"""
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
        R"""
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
        R"""
        Remove FlatBottomHarmonic force from the system.
        """

        forceName = "FlatBottomHarmonic"

        self.removeForce(forceName)


    def addAdditionalForce(self, forceFunction, *args, **kwargs):
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
            warnings.warn("Using user defined force function. Make sure to include the new force object in the MiChroM.forceDict dictionary.")

        #store old forcedict keys
        oldForceDictKeys = list(self.forceDict.keys())
        
        # call the function --  
        # the force name is added to the forceDict but not yet added to the system
        forceFunction(*args, **kwargs)

        # find the new forceDict name
        newKeys = list(set(oldForceDictKeys)^set(self.forceDict.keys()))
        
        try:
            newForceDictKey = newKeys.pop()
        except:
            newForceDictKey = None
        finally:
            if newForceDictKey is None:
                raise ValueError("No new force in MiChroM.forceDict! The new force is either already present or was not added properly to the force dictionary.")
            if newKeys:
                raise ValueError("Force function added multiple new forces in MiChroM! Please break down the function so each force is added separately.")
        
        force = self.forceDict[newForceDictKey]
       
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
        Internal function that loads the chromosome beads into the simulation system.
        """
        if not hasattr(self, "system"):
            return
        if not self.loaded:
            for beadMass in self.masses:
                self.system.addParticle(self.mass * beadMass)
            self.loaded = True


    def createSimulation(self):
        R"""
        Initializes the simulation context and adds forces to the system.

        This function checks if the simulation context has already been created. If not, it loads the particles,
        processes any exceptions (bonds that should not be included in nonbonded interactions), adds
        forces to the system, and sets up the simulation context.
        """
        if getattr(self, 'contexted', False):
            return

        self._loadParticles()

        exceptions = self.bondsForException
        if exceptions:
            # Ensure each bond is a tuple with sorted indices to avoid duplicates like (i, j) and (j, i)
            exceptions = [tuple(sorted(bond)) for bond in exceptions]
            # Remove duplicate bonds by converting the list to a set, then back to a list
            exceptions = list(set(exceptions))

        for forceName, force in self.forceDict.items():
            if hasattr(force, "addException"):
                #print(f"Adding exceptions for '{forceName}' force")
                for pair in exceptions:
                    force.addException(int(pair[0]), int(pair[1]), 0.0, 0.0, 0.0, True)
            elif hasattr(force, "addExclusion"):
                #print(f"Adding exclusions for '{forceName}' force")
                for pair in exceptions:
                    force.addExclusion(int(pair[0]), int(pair[1]))

            if hasattr(force, "CutoffNonPeriodic") and hasattr(force, "CutoffPeriodic"):
                force.setNonbondedMethod(force.CutoffNonPeriodic)

            self.system.addForce(force)
            print(f"{forceName} was added")

        forceGroupIndex = 0
        for forceName, force in self.forceDict.items():
            if "IdealChromosomeChain" in forceName:
                force.setForceGroup(31)
            else:
                force.setForceGroup(forceGroupIndex)
                forceGroupIndex += 1

        #self.context = self.mm.Context(
            #self.system, self.integrator, self.platform, self.properties)
        self.simulation = Simulation(None, self.system, self.integrator, self.platform, self.properties)
        self.context = self.simulation.context
        self.initPositions()
        self.initVelocities()
        self.contexted = True
        print('Context created!')

        simulationInfo = (
                f"\nSimulation name: {self.name}\n"
                f"Number of beads: {self.N}, Number of chains: {len(self.chains)}"
            )
            
        # Get the state of the simulation
        state = self.context.getState(getPositions=True, getEnergy=True, getVelocities=True)
        
        # Calculate energies per bead
        eKin = state.getKineticEnergy() / self.N / units.kilojoule_per_mole
        ePot = state.getPotentialEnergy() / self.N / units.kilojoule_per_mole
        
        # Prepare energy information
        energyInfo = (
            f"Potential energy: {ePot:.5f}, "
            f"Kinetic Energy: {eKin:.5f} at temperature: {self.temperature * 0.008314}"
        )
        
        # Gather platform information
        platformName = self.platform.getName()
        platformSpeed = self.platform.getSpeed()
        propertyNames = self.platform.getPropertyNames()
        
        platformInfo = [
            f"Platform Name: {platformName}",
            f"Platform Speed: {platformSpeed}",
            f"Platform Property Names: {propertyNames}",
        ]
        
        for name in propertyNames:
            value = self.platform.getPropertyValue(self.context, name)
            platformInfo.append(f"{name} Value: {value}")
        
        # Print information to console
        print(simulationInfo)
        print(energyInfo)
        print(f'\nPotential energy per forceGroup:\n {self.printForces()}')
        
        filePath = Path(self.folder) / 'initialStats.txt'
        with open(filePath, 'w') as f:
            for info in platformInfo:
                print(info, file=f)
            print(simulationInfo, file=f)
            print(energyInfo, file=f)
            print(f'\nPotential energy per forceGroup:\n {self.printForces()}', file=f)


    def createReporters(self, statistics=True, traj=False, trajFormat="cndb", energyComponents=False,
                         interval=1000):
        R"""
        Configures and attaches reporters to the simulation for data collection during simulation runs.
        This method sets up custom reporters for the OpenMM `Simulation` object to collect simulation statistics and/or save trajectory data at specified intervals. It supports saving energies per force group and various trajectory file formats.

        Args:
            statistics (bool, optional):
                If `True`, attaches a reporter to collect simulation statistics such as step number, radius of gyration (RG), total energy, potential energy, kinetic energy, and temperature.
                (Default: `True`)
            traj (bool, optional):
                If `True`, attaches a reporter to save trajectory data during the simulation.
                (Default: `False`)
            trajFormat (str, optional):
                The file format to save the trajectory data. Options are `'cndb'`, `'ndb'`, `'pdb'`, `'gro'`, `'xyz'`.
                (Default: `'cndb'`)
            energyComponents (bool, optional):
                If `True`, saves energy components per force group to a separate file named `'energyComponents.txt'` in the simulation folder.
                Requires that `statistics` is `True`.
                (Default: `False`)
            interval (int, optional):
                The interval (in time steps) at which to report data.
                (Default: `1000`)
        """
        
        if traj:
            save_structure_reporter = SaveStructure(
                filePrefix=f'{self.name}',
                reportInterval=interval,
                mode=trajFormat, 
                folder=self.folder,
                chains=self.chains,
                typeListLetter=self.type_list_letter
            )
            self.simulation.reporters.append(save_structure_reporter)
        if statistics:
            if energyComponents:
                simulation_reporter = SimulationReporter(
                    file=f'{self.folder}/statistics.txt',
                    reportInterval=interval,
                    N=self.N,
                    reportPerForceGroup=True, 
                    forceGroupFile=f'{self.folder}/energyComponents.txt',
                    forceDict=self.forceDict
                )
                self.simulation.reporters.append(simulation_reporter)

            else:
                simulation_reporter = SimulationReporter(
                    file=f'{self.folder}/statistics.txt',
                    reportInterval=interval,
                    N=self.N,
                    reportPerForceGroup=False, 
                )
                self.simulation.reporters.append(simulation_reporter)


    def run(self, nsteps, report=True, interval=10**4):
        R"""
        Executes the simulation for a specified number of steps, with optional progress reporting.

        This function runs the simulation by advancing it through the defined number of steps (`nsteps`).
        If `report` is set to `True`, it adds a `StateDataReporter` to the simulation's reporters
        to output progress information to the standard output (`stdout`) at every `interval` steps.

        Args:
            nsteps (int):
                The total number of steps to execute in the simulation. Must be a positive integer.
            
            report (bool, optional):
                Determines whether to enable progress reporting during the simulation run.
                If `True`, a `StateDataReporter` is added to provide real-time updates.
                Default is `True`.
            
            interval (int, optional):
                The number of simulation steps between each progress report.
                This defines how frequently the `StateDataReporter` outputs status information.
                Must be a positive integer.
                Default is `10**4` (10,000 steps).

        Example:
            ```python
            # Run the simulation for 500,000 steps with progress reporting every 5,000 steps
            simulation.run(nsteps=500000, report=True, interval=5000)
            ```
        """
        if report:
            self.simulation.reporters.append(StateDataReporter(stdout, interval, step=True, remainingTime=True,
                                                  progress=True, speed=True, totalSteps=nsteps, separator="\t"))
        
        self.simulation.step(nsteps)


    def addHomoPolymerForces(self, harmonic=False, kFb=30, kA=2.0, eCut= 4.0):
        R"""
        Adds homopolymer forces to the system based on the specified parameters.

        Depending on the `harmonic` flag, this function will add either harmonic bonds
        or FENE bonds to the polymer chains. Additionally, it sets up angle forces and
        repulsive soft-core interactions.

        Args:
            harmonic (bool, optional):
                If True, adds harmonic bonds to the monomers. If False, adds FENE bonds.
                Default is False.
            
            KFb (float, optional):
                The stiffness of the bonds. Relevant when `harmonic` is True.
                Default is 30.0.
            
            kA (float, optional):
                The stiffness of the angles between bonded monomers.
                Default is 2.0.
            
            eCut (float, optional):
                The cutoff distance for the repulsive soft-core potential.
                Default is 4.0.
        """
        if harmonic:
            self.addHarmonicBonds(bondStiffness=kFb)
        else:
            self.addFENEBonds(kFb=kFb)

        self.addAngles(kA=kA)
        self.addRepulsiveSoftCore(eCut= eCut)


    def buildClassicMichrom(self, ChromSeq=None, CoordFiles=None, mode='auto'):
        R"""
        Builds a Classic Michrom simulation setup by initializing the structure, loading it,
        adding polymer forces, defining interactions between types, setting up the ideal chromosome,
        adding flat-bottom harmonic potentials, and creating the simulation.

        Args:
            ChromSeq (str, optional):
                Path to the chromosome sequence file. If `None`, a default path is used.
                Default is `None`.
            
            CoordFiles (str, optional):
                Path to the coordinate files. If `None`, a default path is used.
                Default is `None`.
            
            mode (str, optional):
                Mode of initialization for the structure. Supported modes include 'spring', 'line', etc.
                Default is `'auto'`.
        
        Example:
            ```python
            simulation = MichromSimulation()
            simulation.buildClassicMichrom(
                ChromSeq="/path/to/chromosome_sequence.txt",
                CoordFiles="/path/to/coordinate_files/",
                mode='spring'
            )
            ```
        """

        initialPos = self.initStructure(mode=mode, CoordFiles=CoordFiles, ChromSeq=ChromSeq, isRing=False)
        self.loadStructure(initialPos, center=True)
        self.addHomoPolymerForces()

        self.addTypetoType(mu=3.22, rc=1.78)
        self.addIdealChromosome(mu=3.22, rc=1.78, dinit=3, dend=500)

        self.addFlatBottomHarmonic(kR=5*10**-3, nRad=15.0)
        self.createSimulation()


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


    def saveStructure(self, filename=None, mode="gro"):
        R"""
        Saves the 3D positions of beads during the simulation in various file formats.

        This method exports the simulation's bead positions to files in formats such as XYZ, PDB, GRO, and NDB. It supports multiple chains and assigns residue names based on bead types. The method ensures that the output directory exists and handles different file formats appropriately.

        Args:
            filename (str, optional):
                The name of the file to save the structure. If `None`, the filename is automatically generated using the simulation's name and current step number with the specified mode as the file extension.
                (Default: `None`)
            mode (str, optional):
                The file format to save the structure. Supported formats are:
                - `'xyz'`: XYZ format.
                - `'pdb'`: Protein Data Bank format.
                - `'gro'`: GROMACS GRO format.
                - `'ndb'`: Nucleome Data Bank format.
                (Default: `'gro'`)
        """
        
        data = self.getPositions()
        
        if filename is None:
            filename = self.name +"_step%d." % self.step + mode

        filename = os.path.join(self.folder, filename)
        
        if not hasattr(self, "type_list_letter"):
            raise ValueError("Chromatin sequence not defined!")
        
        if mode == "xyz":
            lines = []
            lines.append(str(len(data)) + "\n")

            for particle in data:
                lines.append("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
            if filename == None:
                return lines
            elif isinstance(filename, str):
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
   
        
    def initPositions(self):
        R"""
        Internal function that sets the locus coordinates in the OpenMM system.
        
        Raises:
            ValueError: If the simulation context has not been initialized.
        """
        if not hasattr(self, 'context'):
            raise ValueError("No context; cannot set positions. Initialize the context before calling initPositions.")

        print("Setting positions...", end='', flush=True)
        self.context.setPositions(self.data)
        print(" loaded!")


    def initVelocities(self):
        R"""
        Internal function that sets the initial velocities of the loci in the OpenMM system.
        
        Raises:
            ValueError: If the simulation context has not been initialized.
        """
        if not hasattr(self, 'context'):
            raise ValueError("No context; cannot set velocities. Initialize the context before calling initVelocities.")
        
        print("Setting velocities...", end='', flush=True)
        # Set velocities using OpenMM's built-in method
        temperature = self.temperature * units.kelvin
        self.context.setVelocitiesToTemperature(temperature)
        print(" loaded!")

        
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
        df = pd.DataFrame(forceValues,forceNames)
        df.columns = ['Values']
        return(df)


    def printHeader(self):
        print('{:^96s}'.format("***************************************************************************************"))
        print('{:^96s}'.format("**** **** *** *** *** *** *** *** OpenMiChroM-"+__version__+" *** *** *** *** *** *** **** ****"))
        print('')
        print('{:^96s}'.format("OpenMiChroM is a Python library for performing chromatin dynamics simulations."))
        print('{:^96s}'.format("OpenMiChroM uses the OpenMM Python API,"))
        print('{:^96s}'.format("employing the MiChroM (Minimal Chromatin Model) energy function."))
        print('{:^96s}'.format("The chromatin dynamics simulations generate an ensemble of 3D chromosomal structures"))
        print('{:^96s}'.format("that are consistent with experimental Hi-C maps, also allows simulations of a single"))
        print('{:^96s}'.format("or multiple chromosome chain using High-Performance Computing "))
        print('{:^96s}'.format("in different platforms (GPUs and CPUs)."))
        print('')
        print('{:^96s}'.format("OpenMiChroM documentation is available at https://open-michrom.readthedocs.io"))
        print('')
        print('{:^96s}'.format("OpenMiChroM is described in: Oliveira Junior, A. B & Contessoto, V, G et. al."))
        print('{:^96s}'.format("A Scalable Computational Approach for Simulating Complexes of Multiple Chromosomes."))
        print('{:^96s}'.format("Journal of Molecular Biology. doi:10.1016/j.jmb.2020.10.034."))
        print('{:^96s}'.format("We also thank the polychrom <https://github.com/open2c/polychrom>"))
        print('{:^96s}'.format("where part of this code was inspired - 10.5281/zenodo.3579472."))
        print('')
        print('{:^96s}'.format("Copyright (c) 2024, The OpenMiChroM development team at"))
        print('{:^96s}'.format("Rice University"))
        print('{:^96s}'.format("***************************************************************************************"))
        stdout.flush()