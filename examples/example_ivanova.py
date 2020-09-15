#!/usr/bin/env python3


# ===============================================================
# Visualize the effective area from a uniform distribution
# where the smoothing lengths have been computed properly.
#
# For a chosen particle, for each neighbour within 2h the
# effective surface is plotted as vectors in the plane
#
# Written by Mladen Ivkovic 2019
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt
import astro_meshless_surfaces as ml


# ---------------------------
# initialize variables
# ---------------------------


# temp during rewriting
srcfile = "./snapshot_uniform.hdf5"  # swift output file
ptype = "PartType0"  # for which particle type to look for
pcoord = np.array([0.5, 0.5])  # coordinates of particle to work for
pind = None  # index of particle you chose with pcoord
npart = 0

nbors = []  # indices of all relevant neighbour particles


fullcolorlist = [
    "red",
    "green",
    "blue",
    "gold",
    "magenta",
    "cyan",
    "lime",
    "saddlebrown",
    "darkolivegreen",
    "cornflowerblue",
    "orange",
    "dimgrey",
    "navajowhite",
    "darkslategray",
    "mediumpurple",
    "lightpink",
    "mediumseagreen",
    "maroon",
    "midnightblue",
    "silver",
]

ncolrs = len(fullcolorlist)


def main():

    # read data from snapshot
    x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)

    # get kernel support radius instead of smoothing length
    H = ml.get_H(h)

    # find index in the read in arrays that we want to work with
    pind = ml.find_index(x, y, pcoord)
    # find that particle's neighbours
    tree, nbors = ml.find_neighbours(pind, x, y, H)

    print("Computing effective surfaces")

    A_ij = ml.Aij_Ivanova(pind, x, y, H, tree=tree)
    x_ij = ml.x_ij(pind, x, y, H, nbors=nbors)

    print("Plotting")

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111)

    pointsize = 200
    # plot central particle
    ax1.scatter(x[pind], y[pind], c="k", s=pointsize * 2)

    # plot other particles and arrows
    for i, n in enumerate(nbors):
        cc = i
        while cc > ncolrs:
            cc -= ncolrs
        col = fullcolorlist[cc]

        ax1.scatter(x[n], y[n], c=col, s=pointsize, zorder=0, lw=1, edgecolor="k")
        arrind = 2
        arrwidth = arrind * 2
        ax1.arrow(
            x_ij[i][0],
            x_ij[i][1],
            A_ij[i][0],
            A_ij[i][1],
            color=col,
            lw=arrwidth,
            zorder=100 - arrind,
        )

    ax1.set_facecolor("lavender")
    ax1.set_xlim((0.25, 0.75))
    ax1.set_ylim((0.25, 0.75))

    ax1.set_title(
        r"Ivanova $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$",
        fontsize=18,
        pad=18,
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    plt.show()
    #  plt.savefig('effective_area_all_neighbours.png', dpi=200)


if __name__ == "__main__":
    main()
