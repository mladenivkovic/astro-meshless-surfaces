#!/usr/bin/env python3

###########################################################################################
#  package:   astro-meshless-surfaces
#  file:      deprecated.py
#  copyright: GPLv3
#             Copyright (C) 2019 EPFL (Ecole Polytechnique Federale de Lausanne)
#             LASTRO - Laboratory of Astrophysics of EPFL
#  author:    Mladen Ivkovic <mladen.ivkovic@hotmail.com>
#
# This file is part of astro-meshless-surfaces.
###########################################################################################

"""
This file contains deprecated functions that I
still have a look into every now and then and am
too lazy to look in the git history every time.

None of these functions works as-is.
"""


def Aij_Ivanova_approximate_gradients(
    pind, x, y, h, m, rho, kernel="cubic_spline", fact=1, L=1, periodic=True
):
    """
    Compute A_ij as defined by Ivanova 2013
    pind:           particle index for which to work for (The i in A_ij)
    x, y, m, rho:   full data arrays as read in from hdf5 file
    h:              kernel support radius array
    kernel:         which kernel to use
    fact:           factor for h for limit of neighbour search; neighbours are closer than fact*h
    L:              boxsize
    periodic:       whether to assume periodic boundaries

    returns:
        A_ij: array of A_ij, containing x and y component for every neighbour j of particle i
    """

    print(
        "Warning: This method hasn't been checked thoroughly in a while. Results might not be right."
    )

    npart = x.shape[0]

    neighbours = [[] for i in x]

    for l in range(npart):
        # find and store all neighbours;
        neighbours[l] = find_neighbours(l, x, y, h, fact=fact, L=L, periodic=periodic)

    # compute all psi_k(x_l) for all l, k
    # first index: index k of psi: psi_k(x)
    # second index: index of x_l: psi(x_l)
    psi_k_at_l = np.zeros((npart, npart), dtype=my_float)

    for k in range(npart):
        for l in neighbours[k]:
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_k_at_l[k, l] = psi(
                x[l],
                y[l],
                x[k],
                y[k],
                h[l],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )

        # self contribution part: k = l +> h[k] = h[l], so use h[k] here
        psi_k_at_l[k, k] = psi(
            0, 0, 0, 0, h[k], kernel=kernel, fact=fact, L=L, periodic=periodic
        )

    omega = np.zeros(npart, dtype=my_float)

    for l in range(npart):
        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[neighbours[l], l]) + psi_k_at_l[l, l]
        # omega_k = sum_l W(x_k - x_l) = sum_l psi_l(x_k) as it is currently stored in memory

    # normalize psi's and convert to float for linalg module
    for k in range(npart):
        psi_k_at_l[:, k] /= omega[k]
    if my_float != np.float:
        psi_k_at_l = psi_k_at_l.astype(np.float)

    # compute all matrices B_k
    B_k = np.zeros((npart), dtype=np.matrix)
    for k in range(npart):
        nbors = neighbours[k]
        # nbors now contains all neighbours l
        B_k[k] = get_matrix(
            x[k], y[k], x[nbors], y[nbors], psi_k_at_l[nbors, k], L=L, periodic=periodic
        )

    # compute all psi_tilde_k at every l
    psi_tilde_k_at_l = np.zeros((npart, npart, 2))
    for k in range(npart):
        # can't just go over neighbours here!
        for l in range(npart):

            dx = np.array([x[k] - x[l], y[k] - y[l]])
            psi_tilde_k_at_l[k, l] = np.dot(B_k[l], dx) * psi_k_at_l[k, l]

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]

    A_ij = np.zeros((len(nbors), 2), dtype=np.float)

    for i, j in enumerate(nbors):

        A = np.array([0.0, 0.0])
        for k in range(npart):
            psi_i_xk = psi_k_at_l[pind, k]
            psi_j_xk = psi_k_at_l[j, k]
            Vk = V(k, m, rho)
            temp = np.array([0.0, 0.0])
            for l in range(npart):
                psi_i_xl = psi_k_at_l[pind, l]
                psi_j_xl = psi_k_at_l[j, l]
                psi_tilde_l = psi_tilde_k_at_l[l, k]

                temp += (psi_j_xk * psi_i_xl - psi_i_xk * psi_j_xl) * psi_tilde_l

            temp *= Vk
            A += temp

        A_ij[i] = A

    # return -A_ij: You will actually use A_ji . F in the formula
    # for the hydrodynamics, not A_ij . F
    return -A_ij


def Aij_Ivanova_analytical_gradients(
    pind, x, y, h, m, rho, kernel="cubic_spline", fact=1, L=1, periodic=True
):
    """
    Compute A_ij as defined by Ivanova 2013. Use analytical expressions for the
    gradient of the kernels instead of the matrix representation.
    Not the recommended way to do it, needs extra computation.

    pind:           particle index for which to work for (The i in A_ij)
    x, y, m, rho:   full data arrays as read in from hdf5 file
    h:              kernel support radius array
    kernel:         which kernel to use
    fact:           factor for h for limit of neighbour search; neighbours are closer than fact*h
    L:              boxsize
    periodic:       whether to assume periodic boundaries

    returns:
        A_ij: array of A_ij, containing x and y component for every neighbour j of particle i
    """

    print(
        "Warning: This method hasn't been checked thoroughly in a while. Results might not be right."
    )

    npart = x.shape[0]

    neighbours = [[] for i in x]

    for l in range(npart):
        # find and store all neighbours;
        neighbours[l] = find_neighbours(l, x, y, h, fact=fact, L=L, periodic=periodic)

    # compute all psi_k(x_l) for all l, k
    # first index: index k of psi: psi_k(x)
    # second index: index of x_l: psi(x_l)
    psi_k_at_l = np.zeros((npart, npart), dtype=my_float)

    for k in range(npart):
        for l in neighbours[k]:
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_k_at_l[k, l] = psi(
                x[l],
                y[l],
                x[k],
                y[k],
                h[l],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )

        # self contribution part: k = l +> h[k] = h[l], so use h[k] here
        psi_k_at_l[k, k] = psi(
            0, 0, 0, 0, h[k], kernel=kernel, fact=fact, L=L, periodic=periodic
        )

    omega = np.zeros(npart, dtype=my_float)

    for l in range(npart):
        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[neighbours[l], l]) + psi_k_at_l[l, l]
        # omega_k = sum_l W(x_k - x_l) = sum_l psi_l(x_k) as it is currently stored in memory

    grad_psi_k_at_l = get_grad_psi_j_at_i_analytical(
        x, y, h, omega, psi_k_at_l, neighbours, kernel=kernel, fact=fact
    )

    # normalize psi's and convert to float for linalg module
    for k in range(npart):
        psi_k_at_l[:, k] /= omega[k]
    if my_float != np.float:
        psi_k_at_l = psi_k_at_l.astype(np.float)

    # now compute A_ij for all neighbours j of i
    nbors = neighbours[pind]

    A_ij = np.zeros((len(nbors), 2), dtype=np.float)

    for i, j in enumerate(nbors):

        A = np.array([0.0, 0.0], dtype=np.float)
        for k in range(npart):
            #  for
            psi_i_xk = psi_k_at_l[pind, k]
            psi_j_xk = psi_k_at_l[j, k]
            grad_psi_i_xk = grad_psi_k_at_l[pind, k]
            grad_psi_j_xk = grad_psi_k_at_l[j, k]
            V_k = 1 / omega[k]

            A += (psi_j_xk * grad_psi_i_xk - psi_i_xk * grad_psi_j_xk) * V_k

        A_ij[i] = A

    # return -A_ij: You will actually use A_ji . F in the formula
    # for the hydrodynamics, not A_ij . F
    return -A_ij


def Integrand_Aij_Ivanova(
    iind,
    jind,
    xx,
    yy,
    hh,
    x,
    y,
    h,
    m,
    rho,
    kernel="cubic_spline",
    fact=1,
    L=1,
    periodic=True,
):
    """
    Compute the effective area integrand for the particles iind jind at
    the positions xx, yy

    (Note that this should be integrated to get the proper effective surface)

    integrand A_ij  = psi_j(x) \nabla psi_i(x) - psi_i (x) \nabla psi_j(x)
                    = sum_k [ psi_j(x_k) psi_i(x) - psi_i(x_k) psi_j(x) ] * psi_tilde_k(x)
                    = psi_i(x) * sum_k psi_j(x_k) * psi_tilde_k(x) - psi_j(x) * sum_k psi_i(x_k) * psi_tilde_k(x)

    The last line is what is actually computed here, with the expression for the gradient
    inserted.


    iind, jind:     particle index for which to work for (The i and j in A_ij)
    xx, yy:         position at which to evaluate
    hh:             kernel support radius at xx, yy
    x, y, m, rho:   full data arrays as read in from hdf5 file
    h:              kernel support radius array
    kernel:         which kernel to use
    fact:           factor for h for limit of neighbour search; neighbours are closer than fact*h
    L:              boxsize
    periodic:       whether to assume periodic boundaries

    returns:
        A_ij: array of integrands A_ij, containing x and y component for every neighbour j of particle i



    """

    print(
        "Warning: This method hasn't been checked thoroughly in a while. Results might not be right."
    )

    nbors = find_neighbours_arbitrary_x(
        xx, yy, x, y, h, fact=fact, L=L, periodic=periodic
    )

    xk = x[nbors]
    yk = y[nbors]
    hk = h[nbors]

    # ----------------------
    # compute psi_i/j(x)
    # ----------------------

    # compute all psi(x)
    psi_x = compute_psi(
        xx, yy, xk, yk, hh, kernel=kernel, fact=fact, L=L, periodic=periodic
    )

    # normalize psis
    omega = np.sum(psi_x)
    psi_x /= omega
    if my_float != np.float:
        psi_x = psi_x.astype(np.float)

    # find where psi_i and psi_j are in that array
    try:
        inb = nbors.index(iind)
        psi_i_of_x = psi_x[inb]  # psi_i(xx, yy)
    except ValueError:
        psi_i_of_x = 0  # can happen for too small smoothing lengths
        print("Exception in psi_i_of_x: iind not found in neighbour list")

    try:
        jnb = nbors.index(jind)
        psi_j_of_x = psi_x[jnb]  # psi_j(xx, yy)
    except ValueError:
        psi_j_of_x = 0  # can happen for too small smoothing lengths
        print("Exception in psi_j_of_x: jind not found in neighbour list")

    # ------------------------------------------------
    # Compute psi_i/j(x_k) at neighbouring positions
    # ------------------------------------------------

    psi_i_xk = [None for n in nbors]
    psi_j_xk = [None for n in nbors]

    omegas = [0, 0]

    for i, n in enumerate([iind, jind]):
        # first compute all psi(xl) from neighbour's neighbours to get weights omega
        nneigh = find_neighbours(n, x, y, h, fact=fact, L=L, periodic=periodic)

        xl = x[nneigh]
        yl = y[nneigh]

        for j, nn in enumerate(nneigh):
            psi_l = compute_psi(
                x[n],
                y[n],
                xl,
                yl,
                h[n],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )

        omegas[i] = np.sum(psi_l) + psi(
            0, 0, 0, 0, h[iind], kernel=kernel, fact=fact, L=L, periodic=periodic
        )

    # now compute psi_i/j(x_k)
    for i, n in enumerate(nbors):
        psi_i_xk[i] = (
            psi(
                xk[i],
                yk[i],
                x[iind],
                y[iind],
                h[iind],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )
            / omegas[0]
        )
        psi_j_xk[i] = (
            psi(
                xk[i],
                yk[i],
                x[jind],
                y[jind],
                h[jind],
                kernel=kernel,
                fact=fact,
                L=L,
                periodic=periodic,
            )
            / omegas[1]
        )

    # ---------------------------------------------
    # Compute psi_tilde_k(x)
    # ---------------------------------------------

    # compute matrix B
    B = get_matrix(xx, yy, xk, yk, psi_x, L=L, periodic=periodic)

    # compute psi_tilde_k(xx)
    psi_tilde_k = np.empty((2, xk.shape[0]))
    for i in range(xk.shape[0]):
        dx = np.array([xk[i] - xx, yk[i] - yy])
        psi_tilde_k[:, i] = np.multiply(np.dot(B, dx), psi_x[i])

    # ----------------------------------
    # Compute A_ij
    # ----------------------------------

    sum_i = np.sum(np.multiply(psi_tilde_k, psi_i_xk), axis=1)
    sum_j = np.sum(np.multiply(psi_tilde_k, psi_j_xk), axis=1)

    A_ij = psi_i_of_x * sum_j - psi_j_of_x * sum_i

    return A_ij


def get_grad_psi_j_at_i_analytical_old(
    x,
    y,
    h,
    omega,
    W_j_at_i,
    neighbour_data,
    kernel="cubic_spline",
    L: np.ndarray = np.ones(2),
    periodic=True,
):
    """
    Compute \nabla \psi_k (x_l) for all particles k and l
    x, y, h:        arrays of positions and compact support radius of all particles
    omega:          weights; sum_j W(x - xj) for all particles x=x_k
    W_j_at_i:       W_k(x_l) npart x npart array
    neighbour_data: neighbour_data object. See function get_neighbour_data_for_all
    kernel:         which kernel to use
    L:              boxsize
    periodic:       whether to assume periodic boundaries

    returns:

        grad_psi_j_at_i: npart x max_neigh x 2 array; grad psi_j (x_i) for all i,j for both x and y direction
    """

    npart = x.shape[0]
    maxneigh = neighbour_data.maxneigh
    neighbours = neighbour_data.neighbours
    nneigh = neighbour_data.nneigh
    iinds = neighbour_data.iinds

    # gradient of psi_j at neighbour i's position
    grad_psi_j_at_i = np.zeros((npart, maxneigh, 2), dtype=my_float)
    # gradient of W_j at neighbour i's position
    grad_W_j_at_i = np.zeros((npart, maxneigh, 2), dtype=my_float)
    # gradient sum for the same h_i
    sum_grad_W = np.zeros((npart, 2), dtype=my_float)

    #  old version
    #  for i in range(npart):
    #      for j, jind in enumerate(neighbours[i]):
    #          dx, dy = get_dx(x[i], x[jind], y[i], y[jind], L=L, periodic=periodic)
    #          r = np.sqrt(dx**2 + dy**2)
    #
    #          dwdr = dWdr(r/h[i], h[i], kernel)
    #          iind = iinds[i, j]
    #
    #          grad_W_j_at_i[jind, iind, 0] = dwdr * dx / r
    #          grad_W_j_at_i[jind, iind, 1] = dwdr * dy / r
    #
    #          sum_grad_W[i] += grad_W_j_at_i[jind, iind]

    for j in range(npart):
        for i, ind_n in enumerate(neighbours[j]):
            dx, dy = get_dx(x[ind_n], x[j], y[ind_n], y[j], L=L, periodic=periodic)
            r = np.sqrt(dx ** 2 + dy ** 2)

            dwdr = dWdr(r / h[ind_n], h[ind_n], kernel)

            grad_W_j_at_i[j, i, 0] = dwdr * dx / r
            grad_W_j_at_i[j, i, 1] = dwdr * dy / r

            # now compute the term needed for the gradient sum need to do it separately: if i is neighbour of j,
            # but j is not neighbour of i, then j's contribution will be missed
            # minus: inverse dx, dy
            dwdr = dWdr(r / h[j], h[j], kernel)
            sum_grad_W[j] -= dwdr * np.array([dx, dy]) / r

    # finish computing the gradients: Need W(r, h), which is currently stored as psi
    for j in range(npart):
        for i, ind_n in enumerate(neighbours[j]):
            grad_psi_j_at_i[j, i, 0] = (
                grad_W_j_at_i[j, i, 0] / omega[ind_n]
                - W_j_at_i[j, i] * sum_grad_W[ind_n, 0] / omega[ind_n] ** 2
            )
            grad_psi_j_at_i[j, i, 1] = (
                grad_W_j_at_i[j, i, 1] / omega[ind_n]
                - W_j_at_i[j, i] * sum_grad_W[ind_n, 1] / omega[ind_n] ** 2
            )

    del grad_W_j_at_i, sum_grad_W

    return grad_psi_j_at_i


def get_neighbour_data_for_all(x, y, h, L: np.ndarray = np.ones(2), periodic=True):
    """
    Gets all the neighbour data for all particles ready.
    Assumes domain is a rectangle with boxsize L[0], L[1].
    x, y, h:    arrays of positions/h of all particles
    L:          boxsize. List/Array or scalar.
    periodic:   Whether you assume periodic boundary conditions

    returns neighbour_data object:
        self.neighbours :   List of lists of every neighbour of every particle
        self.maxneigh :     Highest number of neighbours
        self.nneigh:        integer array of number of neighbours for every particle
        self.iinds:         iinds[i, j] = which index does particle i have in the neighbour
                            list of particle j, where j is the j-th neighbour of i
                            Due to different smoothing lengths, particle j can be the
                            neighbour of i, but i not the neighbour of j.
                            In that case, the particles will be assigned indices j > nneigh[i]

    """

    import copy

    # if it isn't in a list already, create one
    # do this before function/class definition
    if not hasattr(L, "__len__"):
        L = [L, L]

    class neighbour_data:
        def __init__(self, neighbours=None, maxneigh=None, nneigh=None, iinds=None):
            self.neighbours = neighbours
            self.maxneigh = maxneigh
            self.nneigh = nneigh
            self.iinds = iinds

        # -----------------------------------------------

    class cell:
        """
        A cell object to store particles in.
        Stores particle indexes, positions, compact support radii
        """

        def __init__(self):
            self.npart = 0
            self.size = 100
            self.parts = np.zeros(self.size, dtype=np.int)
            self.x = np.zeros(self.size, dtype=np.float)
            self.y = np.zeros(self.size, dtype=np.float)
            self.h = np.zeros(self.size, dtype=np.float)
            self.xmin = 1e300
            self.xmax = -1e300
            self.ymin = 1e300
            self.ymax = -1e300
            self.hmax = -1e300
            return

        def add_particle(self, ind, xp, yp, hp):
            """
            Add a particle, store the index, positions and h
            """
            if self.npart == self.size:
                self.parts = np.append(self.parts, np.zeros(self.size, dtype=np.int))
                self.x = np.append(self.x, np.zeros(self.size, dtype=np.float))
                self.y = np.append(self.y, np.zeros(self.size, dtype=np.float))
                self.h = np.append(self.h, np.zeros(self.size, dtype=np.float))
                self.size *= 2

            self.parts[self.npart] = ind
            self.x[self.npart] = xp
            self.y[self.npart] = yp
            self.h[self.npart] = hp
            self.npart += 1

            if self.xmax < xp:
                self.xmax = xp
            if self.xmin > xp:
                self.xmin = xp
            if self.ymax < yp:
                self.ymax = yp
            if self.ymin > yp:
                self.ymin = yp
            if self.hmax < hp:
                self.hmax = hp

            return

        def is_within_h(self, xp, yp, hp):
            """
            Check whether any particle of this cell is within
            compact support of particle with x, y, h = xp, yp, hp
            """
            dx1, dy1 = get_dx(xp, self.xmax, yp, self.ymax, L=L, periodic=periodic)
            dx2, dy2 = get_dx(xp, self.xmin, yp, self.ymin, L=L, periodic=periodic)
            dxsq = min(dx1 * dx1, dx2 * dx2)
            dysq = min(dy1 * dy1, dy2 * dy2)
            if dxsq / hp ** 2 <= 1 or dysq / hp ** 2 <= 1:
                return True
            else:
                return False

        # -----------------------------------------------

    def find_neighbours_in_cell(i, j, p, xx, yy, hh, is_self):
        """
        Find neighbours of a particle in the cell with indices i,j
        of the grid
        p:      global particle index to work with
        xx, yy: position of particle x
        hh:     compact support radius for p
        is_self: whether this is the cell where p is in
        """
        n = 0
        neigh = [0 for i in range(1000)]
        ncell = grid[i][j]  # neighbour cell we're checking for

        if not is_self:
            if not ncell.is_within_h(xx, yy, hh):
                return []

        N = ncell.npart

        fhsq = hh ** 2

        for c, cp in enumerate(ncell.parts[:N]):
            if cp == p:
                # skip yourself
                continue

            dx, dy = get_dx(xx, ncell.x[c], yy, ncell.y[c], L=L, periodic=periodic)

            dist = dx ** 2 + dy ** 2

            if dist <= fhsq:
                try:
                    neigh[n] = cp
                except IndexError:
                    nneigh += [0 for i in range(1000)]
                    nneigh[n] = cp
                n += 1

        return neigh[:n]
        # -----------------------------------------------

    npart = x.shape[0]

    # first find cell size
    ncells_x = int(L[0] / h.max()) + 1
    ncells_y = int(L[1] / h.max()) + 1
    cell_size_x = L[0] / ncells_x
    cell_size_y = L[1] / ncells_y

    # create grid
    grid = [[cell() for j in range(ncells_y)] for i in range(ncells_x)]

    # sort out particles
    for p in range(npart):
        i = int(x[p] / cell_size_x)
        j = int(y[p] / cell_size_y)
        grid[i][j].add_particle(p, x[p], y[p], h[p])

    neighbours = [[] for i in x]
    nneigh = np.zeros(npart, dtype=np.int)

    # main loop: find and store all neighbours;
    # go cell by cell
    for row in range(ncells_y):
        for col in range(ncells_x):

            cell = grid[col][row]
            N = cell.npart
            parts = cell.parts
            if N == 0:
                continue

            hmax = cell.h[:N].max()

            # find over how many cells to loop in every direction
            maxdistx = int(cell_size_x / hmax + 0.5) + 1
            maxdisty = int(cell_size_y / hmax + 0.5) + 1

            xstart = -maxdistx
            xstop = maxdistx + 1
            ystart = -maxdisty
            ystop = maxdisty + 1

            # exception handling: if ncells < 4, just loop over
            # all of them so that you don't add neighbours multiple
            # times
            if ncells_x < 4:
                xstart = 0
                xstop = ncells_x
            if ncells_y < 4:
                ystart = 0
                ystop = ncells_y

            checked_cells = [
                (None, None) for i in range((2 * maxdistx + 1) * (2 * maxdisty + 1))
            ]
            it = 0

            # loop over all neighbours
            # need to loop over entire square. You need to consider
            # the maximal distance from the edges/corners, not from
            # the center of the cell!
            for i in range(xstart, xstop):
                for j in range(ystart, ystop):

                    if ncells_x < 4:
                        iind = i
                    else:
                        iind = col + i

                    if ncells_y < 4:
                        jind = j
                    else:
                        jind = row + j

                    if periodic:
                        while iind < 0:
                            iind += ncells_x
                        while iind >= ncells_x:
                            iind -= ncells_x
                        while jind < 0:
                            jind += ncells_y
                        while jind >= ncells_y:
                            jind -= ncells_y
                    else:
                        if iind < 0 or iind >= ncells_x:
                            continue
                        if jind < 0 or jind >= ncells_y:
                            continue

                    it += 1
                    if (iind, jind) in checked_cells[: it - 1]:
                        continue
                    else:
                        checked_cells[it - 1] = (iind, jind)

                    # loop over all particles in THIS cell
                    for pc, pg in enumerate(cell.parts[:N]):

                        xp = cell.x[pc]
                        yp = cell.y[pc]
                        hp = cell.h[pc]

                        neighbours[pg] += find_neighbours_in_cell(
                            iind, jind, pg, xp, yp, hp, iind == col and jind == row
                        )

    # sort neighbours by index
    for p in range(npart):
        neighbours[p].sort()
        nneigh[p] = len(neighbours[p])

    # max number of neighbours; needed for array allocation
    maxneigh = nneigh.max()

    # store the index of particle i when required as the neighbour of particle j in arrays[npart, maxneigh]
    # i.e. find index 0 <= i < maxneigh for ever j
    iinds = np.zeros((npart, 2 * maxneigh), dtype=np.int)
    current_count = copy.copy(nneigh)

    for i in range(npart):
        for jc, j in enumerate(neighbours[i]):

            try:
                iinds[i, jc] = (neighbours[j]).index(i)
            except ValueError:
                # it is possible that j is a neighbour for i, but i is not a neighbour
                # for j depending on their respective smoothing lengths
                dx, dy = get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)
                r = np.sqrt(dx ** 2 + dy ** 2)
                if r / h[j] < 1:
                    print(
                        "something went wrong when computing iinds in get_neighbour_data_for_all."
                    )
                    print("i=", i, "j=", j, "r=", r, "H=", h[j], "r/H=", r / h[j])
                    print("neighbours i:", neighbours[i])
                    print("neighbours j:", neighbours[j])
                    print("couldn't find i as neighbour of j")
                    print("exiting")
                    quit()
                else:
                    # append after nneigh[j]
                    iinds[i, jc] = current_count[j]
                    current_count[j] += 1

    nd = neighbour_data(
        neighbours=neighbours, maxneigh=maxneigh, nneigh=nneigh, iinds=iinds
    )

    return nd


def get_neighbour_data_for_all_naive(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    L: np.ndarray = np.ones(2),
    periodic: bool = True,
):
    """
    Gets all the neighbour data for all particles ready.
    Naive way: Loop over all particles for each particle
    x, y, h:    arrays of positions/h of all particles
    L:          boxsize
    periodic:   Whether you assume periodic boundary conditions

    returns neighbour_data object:
        self.neighbours :   List of lists of every neighbour of every particle
        self.maxneigh :     Highest number of neighbours
        self.nneigh:        integer array of number of neighbours for every particle
        self.iinds:         iinds[i, j] = which index does particle i have in the neighbour
                            list of particle j, where j is the j-th neighbour of i
                            Due to different smoothing lengths, particle j can be the
                            neighbour of i, but i not the neighbour of j.
                            In that case, the particles will be assigned indices j > nneigh[i]

    """

    import copy

    npart = x.shape[0]

    # find and store all neighbours;
    neighbours = [[] for i in x]
    for i in range(npart):
        neighbours[i] = find_neighbours(i, x, y, h, L=L, periodic=periodic)

    # get neighbour counts array
    nneigh = np.zeros((npart), dtype=np.int)
    for i in range(npart):
        nneigh[i] = len(neighbours[i])

    # max number of neighbours; needed for array allocation
    maxneigh = nneigh.max()

    # store the index of particle i when required as the neighbour of particle j in arrays[npart, maxneigh]
    # i.e. find index 0 <= i < maxneigh for ever j
    iinds = np.zeros((npart, 2 * maxneigh), dtype=np.int)
    current_count = copy.copy(nneigh)

    for i in range(npart):
        for jc, j in enumerate(neighbours[i]):

            try:
                iinds[i, jc] = (neighbours[j]).index(i)
            except ValueError:
                # it is possible that j is a neighbour for i, but i is not a neighbour
                # for j depending on their respective smoothing lengths
                dx, dy = get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)
                r = np.sqrt(dx ** 2 + dy ** 2)
                if r / h[j] < 1:
                    print("something went wrong when computing iinds.")
                    print("i=", i, "j=", j, "r=", r, "H=", h[j], "r/H=", r / h[j])
                    print("neighbours i:", neighbours[i])
                    print("neighbours j:", neighbours[j])
                    print("couldn't find i as neighbour of j")
                    print("exiting")
                    quit()
                else:
                    # append after nneigh[j]
                    iinds[i, jc] = current_count[j]
                    current_count[j] += 1

    class neighbour_data:
        def __init__(self, neighbours=None, maxneigh=None, nneigh=None, iinds=None):
            self.neighbours = neighbours
            self.maxneigh = maxneigh
            self.nneigh = nneigh
            self.iinds = iinds

    nd = neighbour_data(
        neighbours=neighbours, maxneigh=maxneigh, nneigh=nneigh, iinds=iinds
    )

    return nd
