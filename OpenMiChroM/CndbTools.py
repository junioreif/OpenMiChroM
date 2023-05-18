# Copyright (c) 2020-2023 The Center for Theoretical Biological Physics (CTBP) - Rice University
# This file is from the Open-MiChroM project, released under the MIT License.

R"""
The :class:`~.cndbTools` class perform analysis from **cndb** or **ndb** - (Nucleome Data Bank) file format for storing an ensemble of chromosomal 3D structures.
Details about the NDB/CNDB file format can be found at the `Nucleome Data Bank <https://ndb.rice.edu/ndb-format>`__.
"""

import h5py
import numpy as np
import os
from scipy.spatial import distance

class cndbTools:

    def __init__(self):
        self.Type_conversion = {'A1':0, 'A2':1, 'B1':2, 'B2':3, 'B3':4, 'B4':5, 'NA':6}
        self.Type_conversionInv = {y:x for x,y in self.Type_conversion.items()}
    
    def load(self, filename):
        R"""
        Receives the path to **cndb** or **ndb** file to perform analysis.
        
        Args:
            filename (file, required):
                Path to cndb or ndb file. If an ndb file is given, it is converted to a cndb file and saved in the same directory.
        """
        f_name, file_extension = os.path.splitext(filename)
        
        if file_extension == ".ndb":
            filename = self.ndb2cndb(f_name)   

        self.cndb = h5py.File(filename, 'r')
        
        self.ChromSeq = list(self.cndb['types'].asstr())
        self.uniqueChromSeq = set(self.ChromSeq)
        
        self.dictChromSeq = {}
        
        for tt in self.uniqueChromSeq:
            self.dictChromSeq[tt] = ([i for i, e in enumerate(self.ChromSeq) if e == tt])
        
        self.Nbeads = len(self.ChromSeq)
        self.Nframes = len(self.cndb.keys()) -1
        
        return(self)
    
    
    def xyz(self, frames=[1,None,1], beadSelection=None, XYZ=[0,1,2]):
        R"""
        Get the selected beads' 3D position from a **cndb** or **ndb** for multiple frames.
        
        Args:
            frames (list, required):
                Define the range of frames that the position of the bead will get extracted. The range list is defined by :code:`frames=[initial, final, step]`. (Default value: :code: `[1,None,1]`, all frames)
            beadSelection (list of ints, required):
                List of beads to extract the 3D position for each frame. The list is defined by :code: `beadSelection=[0,1,2,...,N-1]`. (Default value: :code: `None`, all beads) 
            XYZ (list, required):
                List of the axis in the Cartesian coordinate system that the position of the bead will get extracted for each frame. The list is defined by :code: `XYZ=[0,1,2]`. where 0, 1 and 2 are the axis X, Y and Z, respectively. (Default value: :code: `XYZ=[0,1,2]`) 
    
        Returns:
            (:math:`N_{frames}`, :math:`N_{beads}`, 3) :class:`numpy.ndarray`: Returns an array of the 3D position of the selected beads for different frames.
        """
        frame_list = []
        
        if beadSelection == None:
            selection = np.arange(self.Nbeads)
        else:
            selection = np.array(beadSelection)
            
        if frames[1] == None:
            frames[1] = self.Nframes
        
        for i in range(frames[0],frames[1],frames[2]):
            frame_list.append(np.take(np.take(np.array(self.cndb[str(i)]), selection, axis=0), XYZ, axis=1))
        return(np.array(frame_list))
    
    def ndb2cndb(self, filename):
        R"""
        Converts an **ndb** file format to **cndb**.
        
        Args:
            filename (path, required):
                 Path to the ndb file to be converted to cndb.
        """
        Main_chrom      = ['ChrA','ChrB','ChrU'] # Type A B and Unknow
        Chrom_types     = ['ZA','OA','FB','SB','TB','LB','UN']
        Chrom_types_NDB = ['A1','A2','B1','B2','B3','B4','UN']
        Res_types_PDB   = ['ASP', 'GLU', 'ARG', 'LYS', 'HIS', 'HIS', 'GLY']
        Type_conversion = {'A1': 0,'A2' : 1,'B1' : 2,'B2' : 3,'B3' : 4,'B4' : 5,'UN' : 6}
        title_options = ['HEADER','OBSLTE','TITLE ','SPLT  ','CAVEAT','COMPND','SOURCE','KEYWDS','EXPDTA','NUMMDL','MDLTYP','AUTHOR','REVDAT','SPRSDE','JRNL  ','REMARK']
        model          = "MODEL     {0:4d}"
        atom           = "ATOM  {0:5d} {1:^4s}{2:1s}{3:3s} {4:1s}{5:4d}{6:1s}   {7:8.3f}{8:8.3f}{9:8.3f}{10:6.2f}{11:6.2f}          {12:>2s}{13:2s}"
        ter            = "TER   {0:5d}      {1:3s} {2:1s}{3:4d}{4:1s}"

        file_ndb = filename + str(".ndb")
        name     = filename + str(".cndb")

        cndbf = h5py.File(name, 'w')
        
        ndbfile = open(file_ndb, "r")
        
        loop = 0
        types = []
        types_bool = True
        loop_list = []
        x = []
        y = [] 
        z = []

        frame = 0

        for line in ndbfile:
    
            entry = line[0:6]

            info = line.split()


            if 'MODEL' in entry:
                frame += 1

                inModel = True

            elif 'CHROM' in entry:

                subtype = line[16:18]

                types.append(subtype)
                x.append(float(line[40:48]))
                y.append(float(line[49:57]))
                z.append(float(line[58:66]))

            elif 'ENDMDL' in entry:
                if types_bool:
                    typelist = [Type_conversion[x] for x in types]
                    cndbf['types'] = typelist
                    types_bool = False

                positions = np.vstack([x,y,z]).T
                cndbf[str(frame)] = positions
                x = []
                y = []
                z = []

            elif 'LOOPS' in entry:
                loop_list.append([int(info[1]), int(info[2])])
                loop += 1
        
        if loop > 0:
            cndbf['loops'] = loop_list

        cndbf.close()
        return(name)

    
#########################################################################################
#### Analysis start here!
#########################################################################################

    def compute_Orientation_OP(self,xyz,chrom_start=0,chrom_end=1000,vec_length=4):
        from collections import OrderedDict
        import itertools
        R"""
        Calculates the Orientation Order Parameter OP. Details are decribed in "Zhang, Bin, and Peter G. Wolynes. "Topology, structures, and energy landscapes of human chromosomes." Proceedings of the National Academy of Sciences 112.19 (2015): 6062-6067."
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.
            chrom_start (int, required):
                First bead to consider in the calculations (Default value = 0).
            chrom_end (int, required):
                Last bead to consider in the calculations (Default value = 1000).  
            vec_length (int, required):
                Number of neighbor beads to build the vector separation :math:`i` and :math:`i+4` if vec_length is set to 4. (Default value = 4).  
           
                       
        Returns:
            Oijx:class:`numpy.ndarray`:
                Returns the genomic separation employed in the calculations.
            Oijy:class:`numpy.ndarray`:
                Returns the Orientation Order Parameter OP as a function of the genomic separation.
        """

        vec_rij=[] 
        for i in range(chrom_start,chrom_end-vec_length):
            vec_rij.append(xyz[i+vec_length]-xyz[i]) # from a trajectory, gets the vector ri,i+vec_length

        dot_ri_rj=[]
        ij=[]
        for i in itertools.combinations_with_replacement(range(1,chrom_end-vec_length),2):  
            dot_ri_rj.append(np.dot(vec_rij[i[0]]/np.linalg.norm(vec_rij[i[0]]),vec_rij[i[1]]/np.linalg.norm(vec_rij[i[1]]))) # dot product between all vector r,r+_vec_length
            ij.append(i[1]-i[0]) # genomic separation

        d = OrderedDict()
        for k, v in zip(ij, dot_ri_rj):
            d[k] = d.get(k, 0) + v # sum the values v for each genomic separation k
        
        Oijx=[]
        Oijy=[]
        for i in range(1,np.size(list(d.keys()))+1):
            Oijx.append(list(d.keys())[i-1]+1)
            Oijy.append(list(d.values())[i-1]/(chrom_end-list(d.keys())[i-1])) # gets Oij normalized by the number of elements. For example, in a chromosome of length 100, genomic distance 1 has more elements (99) considered than genomic distance 100 (1).

        return np.asarray(Oijx),np.asarray(Oijy)


    def compute_FFT_from_Oij(Oijy,lowcut=1, highcut=500, order=5):
        from scipy.signal import butter, lfilter
        from scipy.fftpack import fft
        R"""
        Calculates the Fourier transform of the Orientation Order Parameter OP. Details are decribed in "Zhang, Bin, and Peter G. Wolynes. "Topology, structures, and energy landscapes of human chromosomes." Proceedings of the National Academy of Sciences 112.19 (2015): 6062-6067."
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.
            lowcut (int, required):
                Filter to cut low frequencies (Default value = 1).
            highcut (int, required):
                Filter to cut high frequencies (Default value = 500).  
            order (int, required):
                Order of the Butterworth filter obtained from `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`__. (Default value = 5).  
           
                       
        Returns:
            xf:class:`numpy.ndarray`:
                Return frequencies.
            yf:class:`numpy.ndarray`:
                Returns the Fourier transform of the Orientation Order Parameter OP in space of 1/Chrom_Length.
        """
        N=np.shape(Oijy)[0]
        y = _butter_bandpass_filter(Oijy, lowcut, highcut, N, order=5)
        xf = np.linspace(1, N//2 , N//2)
        yf=fft(y)/len(y)
        return (xf[0:N//2]-1)/N,np.abs(yf[0:N//2])

    def _butter_bandpass(lowcut, highcut, fs, order=5):
        R"""
        Internal function for selecting frequencies.
        """
        nyq = fs//2
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        R"""
        Internal function for filtering bands.
        """
        b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def compute_Chirality(self,xyz,neig_beads=4):
        R"""
        Calculates the Chirality parameter :math:`\Psi`. Details are decribed in "Zhang, B. and Wolynes, P.G., 2016. Shape transitions and chiral symmetry breaking in the energy landscape of the mitotic chromosome. Physical review letters, 116(24), p.248101."
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.
            neig_beads (int, required):
                Number of neighbor beads to consider in the calculation (Default value = 4).  
                       
        Returns:
            :class:`numpy.ndarray`:
                Returns the Chirality parameter :math:`\Psi` for each bead.
        """
        Psi=[]
        for frame in range(len(xyz)):
            XYZ = xyz[frame]
            Psi_per_bead=[]
            for i in range(0,np.shape(xyz)[0] - np.ceil(1.25*neig_beads).astype('int')):
                a=i
                b=int(np.round(i+0.5*neig_beads))
                c=int(np.round(i+0.75*neig_beads))
                d=int(np.round(i+1.25*neig_beads))

                AB = XYZ[b]-XYZ[a]
                CD = XYZ[d]-XYZ[c]
                E = (XYZ[b]-XYZ[a])/2.0 + XYZ[a]
                F = (XYZ[d]-XYZ[c])/2.0 + XYZ[c]
                Psi_per_bead.append(np.dot((F-E),np.cross(CD,AB))/(np.linalg.norm(F-E)*np.linalg.norm(AB)*np.linalg.norm(CD)))
            Psi.append(Psi_per_bead)
            
        return np.asarray(Psi)

    def compute_RG(self, xyz):
        R"""
        Calculates the Radius of Gyration. 
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.  
                       
        Returns:
            :class:`numpy.ndarray` (dim: Tx1):
                Returns the Radius of Gyration in units of :math:`\sigma`.
        """
        rcm=np.mean(xyz, axis=1,keepdims=True)
        xyz_rel_to_cm= xyz - np.tile(rcm,(xyz.shape[1],1))
        rg=np.sqrt(np.mean(np.linalg.norm(xyz_rel_to_cm,axis=2)**2,axis=1))
        return rg

    def compute_GyrTensorEigs(self, xyz):
        R"""
        Calculates the eigenvalues of the Gyration tensor:
        For a cloud of N points with positions: {[xi,yi,zi]},gyr tensor is a symmetric matrix defined as,
        
        gyr= (1/N) * [[sum_i(xi-xcm)(xi-xcm)  sum_i(xi-xcm)(yi-ycm) sum_i(xi-xcm)(zi-zcm)],
                      [sum_i(yi-ycm)(xi-xcm)  sum_i(yi-ycm)(yi-ycm) sum_i(yi-ycm)(zi-zcm)],
                      [sum_i(zi-zcm)(xi-xcm)  sum_i(zi-zcm)(yi-ycm) sum_i(zi-zcm)(zi-zcm)]]
        
        the three non-negative eigenvalues of gyr are used to define shape parameters like radius of gyration, asphericity, etc

        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.  
                       
        Returns:
            :class:`numpy.ndarray` (dim: Tx3):
                Returns the sorted eigenvalues of the Gyration Tensor.
        """
        rcm=np.mean(xyz, axis=1,keepdims=True)
        sorted_eigenvals=[]
        for frame in xyz-rcm:
            gyr=np.matmul(np.transpose(frame),frame)/xyz.shape[1]
            sorted_eigenvals.append(np.sort(np.linalg.eig(gyr)[0]))
        return np.array(sorted_eigenvals)


    def compute_MSD(self,xyz):
        R"""
        Calculates the Mean-Squared Displacement using Fast-Fourier Transform. 
        Uses Weiner-Kinchin theorem to compute the autocorrelation, and a recursion realtion from the following reference:
        see Sec. 4.2 in Calandrini V, et al. (2011) EDP Sciences (https://doi.org.10.1051/sfn/201112010).
        Also see this stackoverflow post: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.  
                       
        Returns:
            :class:`numpy.ndarray` (dim: NxT):
                Returns the MSD of each particle over the trajectory.

        """
        
        msd=[self._msd_fft(xyz[:,mono_id,:]) for mono_id in range(xyz.shape[1])]
        return np.array(msd)
        

    def _autocorrFFT(self, x):
        R"""
        Internal function. 
        """
        N=len(x)
        F = np.fft.fft(x,n=2*N)  #2*N because of zero-padding
        res = np.fft.ifft(F * F.conjugate()) #autocorrelation using Weiner Kinchin theorem
        res = (res[:N]).real   
        return res/(N-np.arange(0,N)) #this is the normalized autocorrelation

        #r is an (T,3) ndarray: [time stamps,dof]
    def _msd_fft(self, r):
        R"""
        Internal function. 
        """
        N=len(r)
        D=np.square(r).sum(axis=1)
        D=np.append(D,0)
        S2=sum([self._autocorrFFT(r[:, i]) for i in range(r.shape[1])])
        Q=2*D.sum()
        S1=[]
        for m in range(N):
            Q=Q-D[m-1]-D[N-m]
            S1.append(Q/(N-m))
        return np.array(S1) - 2*S2

    def compute_RadNumDens(self, xyz, dr=1.0, ref='centroid',center=None):

        R"""
        Calculates the radial number density of monomers; which when integrated over 
        the volume (with the appropriate kernel: 4*pi*r^2) gives the total number of monomers.
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.  

            dr (float, required):
                mesh size of radius for calculating the radial distribution. 
                can be arbitrarily small, but leads to empty bins for small values.
                bins are computed from the maximum values of radius and dr.
            
            ref (string):
                defines reference for centering the disribution. It can take three values:
                
                'origin': radial distance is calculated from the center

                'centroid' (default value): radial distributioin is computed from the centroid of the cloud of points at each time step

                'custom': user defined center of reference. 'center' is required to be specified when 'custom' reference is chosen

            center (list of float, len 3):
                defines the reference point in custom reference. required when ref='custom'
                       
        Returns:
            num_density:class:`numpy.ndarray`:
                the number density
            
            bins:class:`numpy.ndarray`:
                bins corresponding to the number density

        """

        if ref=='origin':
            rad_vals = np.ravel(np.linalg.norm(xyz,axis=2))

        elif ref=='centroid':
            rcm=np.mean(xyz,axis=1, keepdims=True)
            rad_vals = np.ravel(np.linalg.norm(xyz-rcm,axis=2))

        elif ref == 'custom':
            try:
                if len(center)!=3: raise TypeError
                center=np.array(center,dtype=float)
                center_nd=np.tile(center,(xyz.shape[0],1,1))
                rad_vals=np.ravel(np.linalg.norm(xyz-center_nd,axis=2))

            except (TypeError,ValueError):
                print("FATAL ERROR!!\n Invalid 'center' for ref='custom'.\n\
                        Please provide a valid center: [x0,y0,z0]")
                return ([0],[0])
        else:
            print("FATAL ERROR!! Unvalid 'ref'\n\
                'ref' can take one of three values: 'origin', 'centroid', and 'custom'")
            return ([0],[0])

        rdp_hist,bin_edges=np.histogram(rad_vals, 
                                bins=np.arange(0,rad_vals.max()+1,dr),
                                density=False)

        bin_mids=0.5*(bin_edges[:-1] + bin_edges[1:])
        bin_vols = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        num_density = rdp_hist/(xyz.shape[0]*bin_vols)

        return (num_density, bin_mids)

        
    def compute_RDP(self, xyz, radius=20.0, bins=200):
        R"""
        Calculates the RDP - Radial Distribution Probability. Details can be found in the following publications: 
        
            - Oliveira Jr., A.B., Contessoto, V.G., Mello, M.F. and Onuchic, J.N., 2021. A scalable computational approach for simulating complexes of multiple chromosomes. Journal of Molecular Biology, 433(6), p.166700.
            - Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173.
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function. 
            radius (float, required):
                Radius of the sphere in units of :math:`\sigma` to be considered in the calculations. The radius value should be modified depending on your simulated chromosome length. (Default value = 20.0).
            bins (int, required):
                Number of slices to be considered as spherical shells. (Default value = 200).
                       
        Returns:
            :math:`(N, 1)` :class:`numpy.ndarray`:
                Returns the radius of each spherical shell in units of :math:`\sigma`.
            :math:`(N, 1)` :class:`numpy.ndarray`:
                Returns the RDP - Radial Distribution Probability for each spherical shell.
        """
        
        def calcDist(a,b):
            R"""
            Internal function that calculates the distance between two beads. 
            """
            return np.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2  )
        
        def calc_gr(ref, pos, R, dr):
            R"""
            Internal function that calculates the distance RDP - Radial Distribution Probability. 
            """
            g_r =  np.zeros(int(R/dr))
            dd = []
            for i in range(len(pos)):
                dd.append(calcDist(pos[i],ref))
            raddi =dr
            k = 0
            while (raddi <= R):
                for i in range(0,len(pos)):
                    if (dd[i] >= raddi and dd[i] < raddi+dr):
                        g_r[k] += 1

                g_r[k] = g_r[k]/(4*np.pi*dr*raddi**2)
                raddi += dr
                k += 1
            return g_r 
        
        R_nucleus = radius
        deltaR = R_nucleus/bins                  
   
        n_frames = 0 
        g_rdf = np.zeros(bins)
 
        for i in range(len(xyz)):
            frame = xyz[i]
            centroide = np.mean(frame, axis=0)[None,:][0]
            n_frames += 1
            g_rdf += calc_gr(centroide, frame, R_nucleus, deltaR)
        
        Rx = []   
        for i in np.arange(0, int(R_nucleus+deltaR), deltaR):
            Rx.append(i)
        return(Rx, g_rdf/n_frames) 

    def traj2HiC(self, xyz, mu=3.22, rc = 1.78):
        R"""
        Calculates the *in silico* Hi-C maps (contact probability matrix) using a chromatin dyamics trajectory.   
        
        The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        Args:

            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
        
         Returns:
            :math:`(N, N)` :class:`numpy.ndarray`:
                Returns the *in silico* Hi-C maps (contact probability matrix).
        """
        def calc_prob(data, mu, rc):
            return 0.5 * (1.0 + np.tanh(mu * (rc - distance.cdist(data, data, 'euclidean'))))
        
        size = len(xyz[0])
        P = np.zeros((size, size))
        Ntotal = 0
        
        for i in range(len(xyz)):
            data = xyz[i]
            P += calc_prob(data, mu, rc)
            Ntotal += 1
            if i % 500 == 0:
                print("Reading frame {:} of {:}".format(i, len(xyz)))
        
        return(np.divide(P , Ntotal))    
            
        
    def __repr__(self):
        return '<{0}.{1} object at {2}>\nCndb file has {3} frames, with {4} beads and {5} types '.format(
      self.__module__, type(self).__name__, hex(id(self)), self.Nframes, self.Nbeads, self.uniqueChromSeq)