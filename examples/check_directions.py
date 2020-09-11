#!/usr/bin/env python3


#===============================================================
# Compute the effective surface for Hopkins and Ivanova on a
# perturbed uniform field of particles, draw the arrows
# representing this surface in the same colour as the neighbours
# and also draw straight lines from the chosen particle to the
# neighbour so you can check whether the direction is correct
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
srcfile = './snapshot_perturbed.hdf5'    # swift output file
ptype = 'PartType0'                 # for which particle type to look for
pcoords = [ [0.5, 0.5],
            [0.7, 0.7]]             # coordinates of particle to work for




fullcolorlist=['red', 
        'green', 
        'blue', 
        'gold', 
        'magenta', 
        'cyan',
        'lime',
        'saddlebrown',
        'darkolivegreen',
        'cornflowerblue',
        'orange',
        'dimgrey',
        'navajowhite',
        'darkslategray',
        'mediumpurple',
        'lightpink',
        'mediumseagreen',
        'maroon',
        'midnightblue',
        'silver']

ncolrs = len(fullcolorlist)





#========================
def main():
#========================
    

    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)

    # convert H to h
    H = ms.get_H(h)

    # prepare figure
    nrows = len(pcoords)
    fig = plt.figure(figsize=(10, 5*nrows+0.5))



    count = 0
    for row, pcoord in enumerate(pcoords):
        
        print("Working for particle at", pcoord)

        pind = ms.find_index(x, y, pcoord, tolerance=0.05)
        nbors = ms.find_neighbours(pind, x, y, H)

        print("Computing effective surfaces")


        A_ij_Hopkins = ms.Aij_Hopkins(pind, x, y, H, m, rho)
        A_ij_Ivanova = ms.Aij_Ivanova(pind, x, y, H, m, rho)

        x_ij = ms.x_ij(pind, x, y, H, nbors=nbors)

        print("Sum Hopkins:", np.sum(A_ij_Hopkins, axis=0)) 
        print("Sum Ivanova:", np.sum(A_ij_Ivanova, axis=0)) 


        print("Plotting")

        ax1 = fig.add_subplot(nrows, 2, count+1)
        ax2 = fig.add_subplot(nrows, 2, count+2)
        count +=2

        pointsize = 100
        xmin = pcoord[0]-0.25
        xmax = pcoord[0]+0.25
        ymin = pcoord[1]-0.25
        ymax = pcoord[1]+0.25

        # plot particles in order of distance:
        # closer ones first, so that you still can see the short arrows

        dist = np.zeros(len(nbors))
        for i, n in enumerate(nbors):
            dist[i] = (x[n]-pcoord[0])**2 + (y[n]-pcoord[1])**2

        args = np.argsort(dist)


        for ax in [ax1, ax2]:
            ax.set_facecolor('lavender')
            ax.scatter(x[pind], y[pind], c='k', s=pointsize*2)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_xlabel('x')
            ax.set_ylabel('y')


            for i in range(len(nbors)):

                ii = args[i]
                n = nbors[ii]

                cc = i
                while cc > ncolrs-1:
                    cc -= ncolrs
                col = fullcolorlist[cc]

                arrwidth = 2

                def extrapolate():
                    
                    dx = x[pind] - x[n]
                    dy = y[pind] - y[n]

                    m = dy / dx

                    if m == 0:
                        x0 = 0
                        y0 = y[pind]
                        x1 = 1
                        y1 = y[pind]
                        return [x0, x1], [y0, y1]

                    if dx < 0 :
                        xn = 1
                        yn = y[pind] + m * (xn - x[pind])
                        return [x[pind], xn], [y[pind], yn]
                    else:
                        xn = 0
                        yn = y[pind] + m * (xn - x[pind])
                        return [x[pind], xn], [y[pind], yn]


                # straight line
                xx, yy = extrapolate()
                ax.plot(xx, yy, c=col, zorder=0, lw=1)
                # plot points
                ax.scatter(x[n], y[n], c=col, s=pointsize, zorder=1, lw=1, edgecolor='k')



        for i in range(len(nbors)):

            ii = args[i]
            n = nbors[ii]

            cc = i
            while cc > ncolrs-1:
                cc -= ncolrs
            col = fullcolorlist[cc]

            arrwidth = 2

            ax1.arrow(  x_ij[ii][0], x_ij[ii][1], A_ij_Hopkins[ii][0], A_ij_Hopkins[ii][1], 
                        color=col, lw=arrwidth, zorder=10+i)

            ax2.arrow(  x_ij[ii][0], x_ij[ii][1], A_ij_Ivanova[ii][0], A_ij_Ivanova[ii][1], 
                        color=col, lw=arrwidth, zorder=10+i)



        ax1.set_title(r'Hopkins $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$', fontsize=12, pad=12)
        ax2.set_title(r'Ivanova $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$', fontsize=12, pad=12)


    plt.tight_layout()
    #  plt.savefig('check_directions.png', dpi=200)
    plt.show()






if __name__ == '__main__':
    main()

