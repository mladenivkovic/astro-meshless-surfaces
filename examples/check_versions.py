#!/usr/bin/env python3


#===============================================================
# Check by computing the effective surface a la Hopkins using
# two different ways and pray that it gives identical results,
# and two Ivanova versions (one computes Aij for all particles,
# one for only one particle)
#===============================================================


import numpy as np
import matplotlib.pyplot as plt


import meshless as ms



#---------------------------
# initialize variables
#---------------------------


# temp during rewriting
srcfile = './snapshot_perturbed.hdf5'    # swift output file
ptype = 'PartType0'                 # for which particle type to look for
pcoords = [ [0.5, 0.5],
            [0.7, 0.7]]             # coordinates of particle to work for

print_by_particle = False           # whether to print differences for each particle separately


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

arrwidth = 2




#========================
def main():
#========================


    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)

    # convert H to h
    H = ms.get_H(h)

    # prepare figure
    nrows = len(pcoords)
    fig = plt.figure(figsize=(10, 5*nrows+0.5))

    # compute full ivanova only once
    Aij_Ivanova_v2_full, nbors_all = ms.Aij_Ivanova_all(x, y, H, m, rho)


    count = 0
    for row, pcoord in enumerate(pcoords):

        print("Working for particle at", pcoord)

        pind = ms.find_index(x, y, pcoord, tolerance=0.05)
        nbors = nbors_all[pind]

        print("Computing effective surfaces")


        Aij_Hopkins = ms.Aij_Hopkins(pind, x, y, H, m, rho)
        Aij_Hopkins_v2 = ms.Aij_Hopkins_v2(pind, x, y, H, m, rho)
        Aij_Ivanova = ms.Aij_Ivanova(pind, x, y, H, m, rho)
        Aij_Ivanova_v2 = Aij_Ivanova_v2_full[pind][:len(nbors)]

        x_ij = ms.x_ij(pind, x, y, H, nbors=nbors)


        #--------------------------------------------------------------------------------------------------------
        print("Comparing Hopkins:")
        print("Sum Hopkins:", np.sum(Aij_Hopkins, axis=0))
        print("Sum Hopkins_v2:", np.sum(Aij_Hopkins_v2, axis=0))

        print("Max difference x:", np.max((Aij_Hopkins[:,0] - Aij_Hopkins_v2[:,0])/Aij_Hopkins[:,0]))
        print("Max difference y:", np.max((Aij_Hopkins[:,1] - Aij_Hopkins_v2[:,1])/Aij_Hopkins[:,1]))
        abs1 = np.sqrt(Aij_Hopkins[:,0]**2 + Aij_Hopkins[:,1]**2)
        abs2 = np.sqrt(Aij_Hopkins_v2[:,0]**2 + Aij_Hopkins_v2[:,1]**2)
        print("Max difference norm:", np.max((abs1 - abs2)/abs1))
        print()

        print("Comparing Ivanova:")
        print("Sum Ivanova:", np.sum(Aij_Ivanova, axis=0))
        print("Sum Ivanova_v2:", np.sum(Aij_Ivanova_v2, axis=0))
        print("Max difference x:", np.max((Aij_Ivanova[:,0] - Aij_Ivanova_v2[:,0])/Aij_Ivanova[:,0]))
        print("Max difference y:", np.max((Aij_Ivanova[:,1] - Aij_Ivanova_v2[:,1])/Aij_Ivanova[:,1]))
        abs1 = np.sqrt(Aij_Ivanova[:,0]**2 + Aij_Ivanova[:,1]**2)
        abs2 = np.sqrt(Aij_Ivanova_v2[:,0]**2 + Aij_Ivanova_v2[:,1]**2)
        print("Max difference norm:", np.max((abs1 - abs2)/abs1))
        print("===================================================")
        print()
        #--------------------------------------------------------------------------------------------------------



        print("Plotting")

        ax1 = fig.add_subplot(nrows, 4, count+1)
        ax2 = fig.add_subplot(nrows, 4, count+2)
        ax3 = fig.add_subplot(nrows, 4, count+3)
        ax4 = fig.add_subplot(nrows, 4, count+4)
        count +=4

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

        axes = [ax1, ax2, ax3, ax4]
        Aijs = [Aij_Hopkins, Aij_Hopkins_v2, Aij_Ivanova, Aij_Ivanova_v2]
        titles = ['Aij_Hopkins', 'Aij_Hopkins_v2', 'Aij_Ivanova', 'Aij_Ivanova_v2']

        for ax, Aij, title in zip(axes, Aijs, titles):
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


                #  if print_by_particle:
                #      print("Particle colour", col)
                #      print("Max difference x:", (Aij_Hopkins[ii,0] - Aij_Hopkins_v2[ii,0])/Aij_Hopkins[ii,0])
                #      print("Max difference y:", (Aij_Hopkins[ii,1] - Aij_Hopkins_v2[ii,1])/Aij_Hopkins[ii,1])
                #      print("x v1:", Aij_Hopkins[ii,0], "v2:", Aij_Hopkins_v2[ii, 0], Aij_Hopkins[ii,0] - Aij_Hopkins_v2[ii,0])
                #      print("y v1:", Aij_Hopkins[ii,1], "v2:", Aij_Hopkins_v2[ii, 1], Aij_Hopkins[ii,1] - Aij_Hopkins_v2[ii,1])
                #      abs1 = np.sqrt(Aij_Hopkins[ii,0]**2 + Aij_Hopkins[ii,1]**2)
                #      abs2 = np.sqrt(Aij_Hopkins_v2[ii,0]**2 + Aij_Hopkins_v2[ii,1]**2)
                #      print("Max difference norm:", (abs1 - abs2)/abs1)
                #      print()


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




                ax.arrow(  x_ij[ii][0], x_ij[ii][1], Aij[ii][0], Aij[ii][1],
                            color=col, lw=arrwidth, zorder=10+i)

                ax.set_title(title+r' $\mathbf{A}_{ij}$', fontsize=12, pad=12)



    plt.tight_layout()
    #  plt.savefig('check_versions.png', dpi=200)
    plt.show()






if __name__ == '__main__':
    main()

