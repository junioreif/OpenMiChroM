import os
import h5py
import numpy as np
import csv
from openmm import unit
from openmm.app import Reporter
from pathlib import Path

class SaveStructure(Reporter):
    """
    Custom OpenMM reporter to save the 3D positions of beads during simulation.

    This reporter supports multiple file formats: cndb, ndb, pdb, gro, xyz.

    Args:
        filePrefix (str): Prefix for the output files.
        reportInterval (int): The interval (in time steps) at which to report the data.
        mode (str): The file format to save the data. Options are 'cndb', 'ndb', 'pdb', 'gro', 'xyz'.
        folder (str, optional): Directory where the files will be saved. Defaults to current directory.
        chains (list of tuples): List of chains, where each chain is a tuple (start, end, isRing).
        typeListLetter (list): List of bead types corresponding to their letters.
        diffTypes (list, optional): List indicating different types for each bead. Defaults to None.
    """

    def __init__(self, filePrefix, reportInterval, mode='cndb', folder='.',
                 chains=None, typeListLetter=None, diffTypes=None):
        super().__init__(reportInterval)
        self.filePrefix = filePrefix
        self.reportInterval = reportInterval
        self.mode = mode.lower()
        self.folder = folder
        self.chains = chains or []
        self.typeListLetter = typeListLetter
        self.diffTypes = diffTypes
        self.step = 0

        # Ensure the output folder exists
        os.makedirs(self.folder, exist_ok=True)

        # Initialize storage if using 'cndb' mode
        if self.mode == 'cndb':
            self.storage = []
            for k, chain in enumerate(self.chains):
                fname = os.path.join(self.folder, f"{self.filePrefix}_{k}.cndb")
                storageFile = h5py.File(fname, "w")
                storageFile['types'] = self.typeListLetter[chain[0]:chain[1]+1]
                self.storage.append(storageFile)

    def __del__(self):
        # Close any open storage files
        if self.mode == 'cndb' and hasattr(self, 'storage'):
            for storageFile in self.storage:
                storageFile.close()

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Returns:
            A tuple containing the number of steps until the next report, and whether
            the positions, velocities, forces, energies, and parameters are needed.
        """
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        return (steps, True, False, False, False, False)

    def report(self, simulation, state):
        """Generate a report.

        Args:
            simulation: The Simulation to generate a report for.
            state: The current State of the simulation.
        """
        # Get positions as a NumPy array in nanometers
        data = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Save the structure based on the specified mode
        if self.mode == 'cndb':
            for k, chain in enumerate(self.chains):
                self.storage[k][str(self.step)] = data[chain[0]:chain[1]+1]
        elif self.mode == 'xyz':
            filename = os.path.join(self.folder, f"{self.filePrefix}_block{self.step}.xyz")
            with open(filename, 'w') as myfile:
                myfile.write(f"{len(data)}\n")
                myfile.write("Atoms\n")
                for particle in data:
                    myfile.write("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
        elif self.mode == 'pdb':
            for chainNum, chain in enumerate(self.chains):
                filename = f"{self.filePrefix}_{chainNum}_block{self.step}.pdb"
                filename = os.path.join(self.folder, filename)
                data_chain = data[chain[0]:chain[1]+1]
                types_chain = self.typeListLetter[chain[0]:chain[1]+1]

                with open(filename, 'w') as pdb_file:
                    pdb_file.write(f"TITLE     {self.filePrefix} - chain {chainNum}\n")
                    pdb_file.write(f"MODEL     {self.step+1}\n")

                    totalAtom = 1
                    Res_conversion = {'A1':'ASP', 'A2':'GLU', 'B1':'HIS', 'B2':'LYS', 'B3':'ARG', 'B4':'ARG', 'NA':'ASN'}
                    for i, line in enumerate(data_chain):
                        resName = Res_conversion.get(types_chain[i], 'GLY')
                        pdb_line = (
                            f"ATOM  {totalAtom:5d}  CA  {resName} A{totalAtom:4d}    "
                            f"{line[0]:8.3f}{line[1]:8.3f}{line[2]:8.3f}  1.00  0.00           C\n"
                        )
                        pdb_file.write(pdb_line)
                        totalAtom += 1
                    pdb_file.write(f"ENDMDL\n")
        elif self.mode == 'gro':
            for chainNum, chain in enumerate(self.chains):
                filename = f"{self.filePrefix}_{chainNum}_block{self.step}.gro"
                filename = os.path.join(self.folder, filename)
                data_chain = data[chain[0]:chain[1]+1]
                types_chain = self.typeListLetter[chain[0]:chain[1]+1]

                with open(filename, 'w') as gro_file:
                    gro_file.write(f"{self.filePrefix}_{chainNum}\n")
                    gro_file.write(f"{len(data_chain)}\n")

                    totalAtom = 1
                    Res_conversion = {'A1':'ASP', 'A2':'GLU', 'B1':'HIS', 'B2':'LYS', 'B3':'ARG', 'B4':'ARG', 'NA':'ASN'}
                    for i, line in enumerate(data_chain):
                        resName = Res_conversion.get(types_chain[i], 'GLY')
                        gro_line = (
                            f"{totalAtom:5d}{resName:<5}{'CA':>5}{totalAtom:5d}"
                            f"{line[0]:8.3f}{line[1]:8.3f}{line[2]:8.3f}\n"
                        )
                        gro_file.write(gro_line)
                        totalAtom += 1
                    gro_file.write(f"   0.00000   0.00000   0.00000\n")
        elif self.mode == 'ndb':
            for chainNum, chain in enumerate(self.chains):
                filename = f"{self.filePrefix}_{chainNum}_block{self.step}.ndb"
                filename = os.path.join(self.folder, filename)
                data_chain = data[chain[0]:chain[1]+1]
                types_chain = self.typeListLetter[chain[0]:chain[1]+1]

                with open(filename, 'w') as ndb_file:
                    ndb_file.write(f"HEADER    NDB File generated by Open-MiChroM\n")
                    ndb_file.write(f"TITLE     {self.filePrefix} - chain {chainNum}\n")
                    ndb_file.write(f"MODEL        {self.step+1}\n")
                    for i, line in enumerate(data_chain):
                        ndb_line = (
                            f"CHROM  {i+1:8d} {types_chain[i]:>2}      A{chainNum+1:4d} {i+1:8d}"
                            f"{line[0]:8.3f}{line[1]:8.3f}{line[2]:8.3f}"
                            f"{(i)*50000+1:10d}{(i+1)*50000:10d}    0.000\n"
                        )
                        ndb_file.write(ndb_line)
                    ndb_file.write("END\n")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        self.step += 1

class SimulationReporter(Reporter):
    """
    Custom OpenMM reporter to record simulation data such as step, RG, Etotal, Epot, Ekin, temperature.

    Optionally saves energies per force group in a separate CSV file.

    Args:
        file (str): Path to the output CSV file where the data will be saved.
        reportInterval (int): The interval (in time steps) at which to report the data.
        N (int): Number of particles in the system.
        reportPerForceGroup (bool, optional): If True, saves energies per force group. Defaults to False.
        forceGroupFile (str, optional): Path to the output CSV file for force group energies. Required if reportPerForceGroup is True.
    """

    def __init__(self, file, reportInterval, N, reportPerForceGroup=False, forceGroupFile=None):
        super().__init__(reportInterval)
        self.file = file
        self.reportInterval = reportInterval
        self.N = N
        self.reportPerForceGroup = reportPerForceGroup
        self.forceGroupFile = forceGroupFile
        self._initialized = False
        self._file_handle = None
        self._force_group_file_handle = None

    def __del__(self):
        # Close any open files
        if self._file_handle is not None:
            self._file_handle.close()
        if self._force_group_file_handle is not None:
            self._force_group_file_handle.close()

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Returns:
            A tuple containing the number of steps until the next report, and whether
            the positions, velocities, forces, energies, and parameters are needed.
        """
        steps = self.reportInterval - simulation.currentStep % self.reportInterval
        # We need positions and energies to calculate RG and energies
        return (steps, True, False, False, True, False)

    def report(self, simulation, state):
        """Generate a report.

        Args:
            simulation: The Simulation to generate a report for.
            state: The current State of the simulation.
        """
        if not self._initialized:
            # Initialize files and write headers
            # Open the main file
            self._file_handle = open(self.file, 'w', newline='')
            self._csv_writer = csv.writer(self._file_handle)
            # Write header
            self._csv_writer.writerow(['Step', 'RG', 'Etotal', 'Epot', 'Ekin', 'Temperature'])

            if self.reportPerForceGroup:
                if self.forceGroupFile is None:
                    raise ValueError("forceGroupFile must be specified when reportPerForceGroup is True.")
                self._force_group_file_handle = open(self.forceGroupFile, 'w', newline='')
                self._force_group_csv_writer = csv.writer(self._force_group_file_handle)
                # Get force group names and indices
                system = simulation.context.getSystem()
                self.force_groups = {}
                for i in range(system.getNumForces()):
                    force = system.getForce(i)
                    group = force.getForceGroup()
                    name = type(force).__name__
                    if group not in self.force_groups:
                        self.force_groups[group] = name
                    else:
                        # Handle forces with the same group
                        self.force_groups[group] += f", {name}"
                # Sort force groups by group index
                sorted_force_groups = sorted(self.force_groups.items())
                # Write header
                header = ['Step'] + [f'Group {group}: {name}' for group, name in sorted_force_groups]
                self._force_group_csv_writer.writerow(header)
            self._initialized = True

        # Get necessary data from the state
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        kineticEnergy = state.getKineticEnergy()
        potentialEnergy = state.getPotentialEnergy()
        totalEnergy = kineticEnergy + potentialEnergy

        # Calculate radius of gyration (RG)
        data = positions
        data -= np.mean(data, axis=0)
        squared_distances = np.sum(data**2, axis=1)
        rg_squared = np.mean(squared_distances)
        rg = np.sqrt(rg_squared)

        # Calculate temperature
        # The formula in the original code seems incorrect. The temperature is calculated using:
        # T = (2 * KE) / (N * kB)
        # But in OpenMM, the temperature is stored in the integrator, so we can get it directly.
        temperature = kineticEnergy * 2 / (simulation.context.getSystem().getDegreesOfFreedom() * unit.MOLAR_GAS_CONSTANT_R)

        # Convert energies to per particle and in kJ/mol units
        eKin = kineticEnergy / self.N / unit.kilojoule_per_mole
        ePot = potentialEnergy / self.N / unit.kilojoule_per_mole
        eTotal = totalEnergy / self.N / unit.kilojoule_per_mole

        # Write data to main CSV file
        self._csv_writer.writerow([simulation.currentStep, f"{rg:.5f}", f"{eTotal:.5f}", f"{ePot:.5f}", f"{eKin:.5f}", f"{temperature.value_in_unit(unit.kelvin):.5f}"])

        if self.reportPerForceGroup:
            force_group_energies = []
            sorted_force_groups = sorted(self.force_groups.items())
            for group, name in sorted_force_groups:
                group_bitmask = 1 << group
                group_state = simulation.context.getState(getEnergy=True, groups=group_bitmask)
                group_energy = group_state.getPotentialEnergy() / self.N / unit.kilojoule_per_mole
                force_group_energies.append(f"{group_energy:.5f}")
            # Write to the force group CSV file
            self._force_group_csv_writer.writerow([simulation.currentStep] + force_group_energies)
