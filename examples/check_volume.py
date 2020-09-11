#!/usr/bin/env python3


#===============================================================
# Compute the volume of a particle in various ways and pray that
# you get the same results bae
#===============================================================


import numpy as np
import matplotlib.pyplot as plt

try:
    import meshless as ms
except ImportError:
    print("Didn't find 'meshless' module... be sure to add it to your pythonpath!")
    quit(2)




#---------------------------
# initialize variables
#---------------------------


# temp during rewriting
srcfile = './snapshot_uniform.hdf5'    # swift output file
ptype = 'PartType0'                 # for which particle type to look for
pcoord = [0.5, 0.5]                 # coordinates of particle to work for

my_float = np.float



#========================
def main():
#========================
    

    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)
    pind = ms.find_index(x, y, pcoord)

    npart = x.shape[0]


    kernels = ms.kernels
    
    for k, kernel in enumerate(kernels):

        # transform smoothing lengths to kernel radius
        H = ms.get_H(h, kernel)

        # compute all psi_i(x_j) for all i, j
        psi_i_at_j = np.zeros((npart, npart), dtype=my_float)

        for i in range(npart):
            for j in range(npart):
                psi_i_at_j[i,j] = ms.psi(x[j], y[j], x[i], y[i], H[j], kernel)



        omega = np.zeros(npart, dtype=my_float)

        for j in range(npart):

            # compute normalisation omega for all particles
            # needs psi_i_at_j to be computed already
            omega[j] =  np.sum(psi_i_at_j[:, j])
            # omega_i = sum_j W(x_i - x_j) = sum_j psi_j(x_i) as it is currently stored in memory


        
        # compute volumes from swift data
        V_i = ms.V(pind, m, rho)
        V_tot_array = np.sum(m/rho)

        # compute volumes from computed data
        V_comp = 1/omega[pind]
        V_tot_comp = np.sum(1.0/omega)


        # dump output
        print()
        print("Kernel:", kernel)
        print('{0:20} | {1:20} | {2:20} | {3:20}'.format(' ', '1/omega', 'm/rho', 'relative difference'))
        for i in range(20+3*20 + 3*3):
            print('-', end='')
        print()
        print("{0:20} | {1:20.10f} | {2:20.10f} | {3:20.3f} {4:1}".format("Single particle:", V_comp, V_i, (V_comp - V_i)/V_i*100,"%"))
        print("{0:20} | {1:20.10f} | {2:20.10f} | {3:20.3f} {4:1}".format("Total Volume:", V_tot_comp, V_tot_array, (V_tot_comp-V_tot_array)/V_tot_array*100, "%"))


    print('[Note: Gaussian kernel doesn\'t satisfy compactness condition, it\'s allowed to be wrong]')




if __name__ == '__main__':
    main()

