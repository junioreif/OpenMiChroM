import sys
sys.path.append('/home/sb95/ActiveOpenMiChroM/OpenMiChroM/')
from CndbTools import cndbTools
import numpy as np
from scipy.spatial import distance

class cndbTools_ext(cndbTools):
    def __init__(self):
        super(cndbTools_ext, self).__init__()
     
    def vel_autocorr(self, xyz, dt):
        def autocorrFFT(x):
            N=len(x)
            F = np.fft.fft(x,n=2*N)  #2*N because of zero-padding
            res = np.fft.ifft(F * F.conjugate()) #autocorrelation using Weiner Kinchin theorem
            res = (res[:N]).real
            return res/(N-np.arange(0,N)) #this is the normalized autocorrelation

        #r is an (T,3) ndarray: [time stamps,dof]
        def msd_fft(r):
            N=len(r)
            D=np.square(r).sum(axis=1)
            D=np.append(D,0)
            S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
            Q=2*D.sum()
            S1=[]
            for m in range(N):
                Q=Q-D[m-1]-D[N-m]
                S1.append(Q/(N-m))
            return (np.array(S1), S2)

        vel_dt = (xyz[dt:,:,:] - xyz[:-dt,:,:])/dt
        Cv_dt = np.array([msd_fft(vel_dt[:,mono_id,:])[1] for mono_id in range(vel_dt.shape[1])])
        Cv_norm = Cv_dt/Cv_dt[:,0].reshape(-1,1)
        Cv_norm_avg = np.mean(Cv_norm, axis=0)

        return Cv_norm_avg

    def spatial_disp_corr(self, xyz, dr_vals, dt):
        Cv_r=[]
        #for t in range(0, int(R.shape[0]-dt), int(dt)):
        for t in np.random.choice(range(1,int(xyz.shape[0]-dt)),size=(200), replace=False):
            vel_t = (xyz[t+dt,:,:] - xyz[t,:,:])/dt
            dist_t = distance.cdist(xyz[t],xyz[t], metric='euclidean')
            vel_corr = 1.0 - distance.cdist(vel_t,vel_t, metric='cosine')
            cv_dr = []
            for dr in dr_vals:
                corr_mat_dr = vel_corr*(dist_t<=dr)
                val = np.mean(corr_mat_dr[corr_mat_dr !=0])
                cv_dr.append(val)

            Cv_r.append(cv_dr)
        Cv_r = np.mean(np.array(Cv_r), axis=0)
        return np.vstack((Cv_r, dr_vals))

     
             
