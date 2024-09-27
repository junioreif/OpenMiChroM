import os
import h5py
import numpy as np
from openmm import unit
from openmm.app import StateDataReporter
from datetime import datetime


class SaveStructure(StateDataReporter):
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
        #super().__init__(reportInterval)
        super(SaveStructure, self).__init__(filePrefix, reportInterval)
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
        if self.mode == 'swb':
            info = {
            'version' : '1.0.0',
            'format' : 'swb',
            'genome' : 'hg38',
            'pointtype' : 'single_point',
            'title': 'Trajectory from OpenmiChroM apy',
            'author' : 'Antonio B Oliveira Junior',
            'date' : str(datetime.now())
            }
            self.storage = []
            for k, chain in enumerate(self.chains):
                fname = os.path.join(self.folder, f"{self.filePrefix}_{k}.swb")
                storageFile = h5py.File(fname, "w")
                H = storageFile.create_group('Header')
                H.attrs.update(info)
                C = storageFile.create_group(self.filePrefix)
                C.create_group('spatial_position')
                l = chain[1]+1 - chain[0]
                range_list = [[int(n*50000+1), int(n*50000+50000)] for n in range(l)]
                C.create_dataset('genomic_position',data=np.array(range_list))
                self.storage.append(storageFile)

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
        # #its not work in notebooks
        if self.mode == 'cndb' and hasattr(self, 'storage'):
            for storageFile in self.storage:
                storageFile.close()

        if self.mode == 'swb' and hasattr(self, 'storage'):
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
        if self.mode == 'swb':
            for k, chain in enumerate(self.chains):
                pos = self.storage[k][f'/{self.filePrefix}/spatial_position']
                pos.create_dataset(f't_{self.step}',data=data[chain[0]:chain[1]+1])

        # Save the structure based on the specified mode
        elif self.mode == 'cndb':
            for k, chain in enumerate(self.chains):
                self.storage[k][str(self.step)] = data[chain[0]:chain[1]+1]

        elif self.mode == 'xyz':
            filename = os.path.join(self.folder, f"{self.filePrefix}_state{self.step}.xyz")
            with open(filename, 'w') as myfile:
                myfile.write(f"{len(data)}\n")
                myfile.write("Atoms\n")
                for particle in data:
                    myfile.write("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
        elif self.mode == 'pdb':
            for chainNum, chain in enumerate(self.chains):
                filename = f"{self.filePrefix}_{chainNum}_state{self.step}.pdb"
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
                filename = f"{self.filePrefix}_{chainNum}_state{self.step}.gro"
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
                filename = f"{self.filePrefix}_{chainNum}_state{self.step}.ndb"
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

class SimulationReporter(StateDataReporter):
    """
    Custom OpenMM reporter to record simulation data such as step, RG, Etotal, Epot, Ekin, temperature.

    Optionally saves energies per force group in a separate text file.

    Args:
        file (str): Path to the output text file where the data will be saved.
        reportInterval (int): The interval (in time steps) at which to report the data.
        N (int): Number of particles in the system.
        reportPerForceGroup (bool, optional): If True, saves energies per force group. Defaults to False.
        forceGroupFile (str, optional): Path to the output text file for force group energies. Required if reportPerForceGroup is True.
    """

    def __init__(self, file, reportInterval, N, reportPerForceGroup=False, forceGroupFile=None, forceDict=None, **kwargs):
        super(SimulationReporter, self).__init__(file, reportInterval, **kwargs)

        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
                
        if self._openedFile:
            self._out = open(file, 'w')
            if reportPerForceGroup:
                if forceGroupFile is None:
                    forceGroupFile = 'energies.txt'
                self._forceout = open(forceGroupFile, 'w')
        else:
            self._out = file
            if reportPerForceGroup:
                self._forceout = forceGroupFile

        self.file = file
        self.reportInterval = reportInterval
        self.N = N
        self.reportPerForceGroup = reportPerForceGroup
        self.forceGroupFile = forceGroupFile
        self._initialized = False
        self._file_handle = None
        self._force_group_file_handle = None
        self.forceDict = forceDict

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
        return (steps, True, False, False, True, True)

    def report(self, simulation, state):
        """Generate a report.

        Args:
            simulation: The Simulation to generate a report for.
            state: The current State of the simulation.
        """

        if not self._hasInitialized:
            self._initializeConstants(simulation)
            
            print('#Step,RG,Etotal,Epot,Ekin,Temperature\n', file=self._out)
            try:
                self._out.flush()
            except AttributeError:
                pass

            if self.reportPerForceGroup:
                # Get force group names and indices
                forceNames = []
                for n in (self.forceDict):
                    forceNames.append(n)
                # Write header
                header = ['Step'] + [f'{name}' for  name in forceNames]
                print(header, file=self._forceout)
                try:
                    self._forceout.flush()
                except AttributeError:
                    pass

            self._initialSteps = simulation.currentStep
            self._hasInitialized = True

        self._checkForErrors(simulation, state)

        values = self._constructReportValues(simulation, state)
        
        # Write the values.
        print(self._separator.join([str(values[0])] + [f"{v:.3f}" for v in values[1:]]), file=self._out)
        #print(self._separator.join(f"{v:.3f}" for v in values), file=self._out)
        try:
            self._out.flush()
        except AttributeError:
            pass

        if self.reportPerForceGroup:
            force_group_energies = []
            force_group_energies.append(str(values[0]))
            for n in (self.forceDict):
                group_energy = simulation.context.getState(getEnergy=True, groups={self.forceDict[n].getForceGroup()}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                force_group_energies.append(f"{group_energy:.3f}")
            print(self._separator.join(force_group_energies), file=self._forceout)
            try:
                self._forceout.flush()
            except AttributeError:
                pass
    
        
    def _constructReportValues(self, simulation, state):
        """Query the simulation for the current state of our observables of interest.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation

        Returns
        -------
        A list of values summarizing the current state of
        the simulation, to be printed or saved. Each element in the list
        corresponds to one of the columns in the resulting CSV file.
        """
        values = []
        # Get necessary data from the state

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        kineticEnergy = state.getKineticEnergy()
        potentialEnergy = state.getPotentialEnergy()
        totalEnergy = potentialEnergy + kineticEnergy

        # Convert energies to per particle and in kJ/mol units
        eKin = kineticEnergy / self.N / unit.kilojoule_per_mole
        ePot = potentialEnergy / self.N / unit.kilojoule_per_mole
        eTotal = totalEnergy / self.N / unit.kilojoule_per_mole

        # Calculate radius of gyration (RG)
        data = positions
        data -= np.mean(data, axis=0)
        squared_distances = np.sum(data**2, axis=1)
        rg_squared = np.mean(squared_distances)
        rg = np.sqrt(rg_squared)
       

        values.append(simulation.currentStep)
        values.append(rg)
        values.append(eTotal)
        values.append(ePot)
        values.append(eKin)
        
        integrator = simulation.context.getIntegrator()
        if hasattr(integrator, 'computeSystemTemperature'):
            values.append(integrator.computeSystemTemperature().value_in_unit(unit.kelvin))
        else:
            values.append((2*(kineticEnergy / self.N) /(3.0*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)*0.008314)
        return values