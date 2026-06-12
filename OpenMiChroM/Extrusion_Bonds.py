import numpy as np
from numpy.random import randint, rand

class Loop_Extrusion_Manager():
    """ This class genenerates a loop extruder trajectory of shape (num_steps, extruder_count, 2). 
    See the SV paper for details. 

    A great deal of this code is attributed to A. Sanborn; see Sanborn et al., 2015.

    Args:
        fprobs_fix (array, required):
            1-D array of length = number of beads on your polymer. Value at each index is the probability that 
            the *left* foot of an extruder will become fixed if it steps here
        rprobs_fix (array, required):
            same for right feet
        num_steps (int, required):
            length of trajectory in steps
        extruder_count (int, required):
            number of extruders
        off_rate (float, optional):
            probability (between 0 and 1) that an extruder will 'fall off' the chromatin at any timestep. Should be set very low 
            (see Sanborn paper, SV paper; can be ~ 1/500).
        unfix_conversion_factor (float, optional):
            probability of unfixing = 1/probability of fixing * conversion_factor
            In terms of residence time, E[residence time on motif] = probability of fixing / conversion factor 

            
    Methods:
        get_extrusion_bonds():
            returns loop extruder trajectory of shape (num_steps, extruder_count, 2) when called. 
    """
    def __init__(self,fprobs_fix,rprobs_fix,num_steps,extruder_count,off_rate = 1/500,unfix_conversion_factor = 1/4_000):
        self.chrom_length = len(fprobs_fix)
        self.fprobs_fix = fprobs_fix
        self.rprobs_fix = rprobs_fix
        self.num_steps = num_steps
        self.extruder_count = extruder_count
        self.off_rate = off_rate
        num = 1.0/4000.0
        self.fprobs_unfix = np.divide(
            num,
            self.fprobs_fix,
            out=np.zeros_like(self.fprobs_fix, dtype=float),
            where=self.fprobs_fix != 0
        )
        self.rprobs_unfix = np.divide(
            num,
            self.rprobs_fix,
            out=np.zeros_like(self.rprobs_fix, dtype=float),
            where=self.rprobs_fix != 0
        )
        
        

    def get_extrusion_bonds(self):
        fprobs_fix = self.fprobs_fix
        rprobs_fix = self.rprobs_fix
        fprobs_unfix = self.fprobs_unfix
        rprobs_unfix = self.rprobs_unfix
        num_steps = self.num_steps
        extruder_count = self.extruder_count
        off_rate = self.off_rate
        chrom_length = self.chrom_length

        # print(f"forwards probs of fixing = {fprobs_fix}")
        # print(f"reverse probs of fixing = {rprobs_fix}")
        # print(f"forwards probs of unfixing = {fprobs_unfix}")
        # print(f"reverse probs of unfixing = {rprobs_unfix}")

        # list of fixed anchors, bool for each anchor
        fixed = np.zeros((extruder_count, 2)).astype("bool")

        # list of extruder positions: timesteps x num_bonds x 2
        bonds = np.zeros((num_steps + 1, extruder_count, 2)).astype("int")

        #### Initialize the bonds! ####
        # Initialize length=2 bonds at random positions, independent of boundaries.
        # Bond positions run from 1 to polylen
        start_pos = randint(3, chrom_length-2,extruder_count)
        for i in range(extruder_count):
            while np.sum(np.abs(start_pos - start_pos[i]) <= 2) >= 2:
                start_pos[i] = randint(3, chrom_length-2)
            bonds[0, i, 0] = start_pos[i] - 1
            bonds[0, i, 1] = start_pos[i] + 1


        #### slide the bonds! ####
        for step in range(num_steps):
            
            # shift all the extruding bonds
            for ndx in range(extruder_count):
                bond = bonds[step, ndx]

                # reset bond?
                reset_bond = False
                # fix bonds?
                #NOTE: We never unfix probabilistically, except via off_rate
                if rand() < fprobs_fix[bond[0]-1]:
                    fixed[ndx, 0] = True
                if rand() < rprobs_fix[bond[1]-1]:
                    fixed[ndx, 1] = True


                #NOTE: Here, I add unfixing
                if rand() < fprobs_unfix[bond[0]-1]:
                    fixed[ndx, 0] = False
                if rand() < rprobs_unfix[bond[1]-1]:
                    fixed[ndx, 1] = False

                # slide bonds
                nextbond = [max(bond[0] - 1, 2), \
                            min(bond[1] + 1, chrom_length-2)]

                # check for fixed loop anchors
                if fixed[ndx, 0]:
                    nextbond[0] = bond[0]
                if fixed[ndx, 1]:
                    nextbond[1] = bond[1]

                
                
                # ----- RESET BOND? -----


                # anchors from other bonds
                other_anchors = np.hstack([
                    bonds[step+1, :ndx].flatten(),
                    bonds[step,   ndx+1:].flatten()
                ]).astype(int)

                # which of those feet are fixed?
                fixed_flat = np.hstack([
                    fixed[:ndx].flatten(),
                    fixed[ndx+1:].flatten()
                ])

                fixed_posns = other_anchors.copy() 
                 # mask non-fixed feet
                fixed_posns[~fixed_flat.astype(bool)] = -1 

                # reset if we collide with a *fixed* anchor
                # specifically, check if our nextbond positions (future moves) are in the fixed anchor list
                if (fixed_posns == nextbond[0]).any() or (fixed_posns == nextbond[1]).any():
                    reset_bond = True




                # check endpoints
                if nextbond[0] <= 2 or nextbond[1] >= chrom_length-2:
                    reset_bond = True
                
                # reset bond randomly?
                if rand() < off_rate:
                    reset_bond = True

                if reset_bond:
                    # unfix anchors
                    fixed[ndx, :] = False
                    retry = True
                    while retry:
                        start_pos = randint(3, chrom_length-2)
                        retry = np.any(np.abs(other_anchors - start_pos) <= 1)
                    nextbond[0] = start_pos - 1
                    nextbond[1] = start_pos + 1

                # record shifted bond
                bonds[step + 1, ndx, 0] = nextbond[0]
                bonds[step + 1, ndx, 1] = nextbond[1]
        print ("Generated extrusion bonds list. ")

        return bonds
