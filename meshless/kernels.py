#!/usr/bin/env python3

###########################################################################################
#  package:   astro-meshless-surfaces
#  file:      kernels.py
#  brief:     contains kernel related stuff
#  copyright: GPLv3
#             Copyright (C) 2019 EPFL (Ecole Polytechnique Federale de Lausanne)
#             LASTRO - Laboratory of Astrophysics of EPFL
#  author:    Mladen Ivkovic <mladen.ivkovic@hotmail.com>
#
# This file is part of astro-meshless-surfaces.
###########################################################################################

import numpy as np
from .optional_packages import jit
from typing import Union

# Names of all available kernels
kernels = [
    "cubic_spline",
    "quartic_spline",
    "quintic_spline",
    "wendland_C2",
    "wendland_C4",
    "wendland_C6",
]

kernel_derivatives = [
    "cubic_spline",
    "quartic_spline",
    "quintic_spline",
    "wendland_C2",
    "wendland_C4",
    "wendland_C6",
]

kernels_with_gaussian = [
    "cubic_spline",
    "quartic_spline",
    "quintic_spline",
    "wendland_C2",
    "wendland_C4",
    "wendland_C6",
    "gaussian",
]

kernel_pretty_names = [
    "cubic spline kernel",
    "quartic spline kernel",
    "quintic spline kernel",
    "wendland C2 kernel",
    "wendland C4 kernel",
    "wendland C6 kernel",
]

# factors of all kernels for which fact*H = 0
kernelfacts = [1, 1, 1, 1, 1, 1, None]

kernel_H_over_h = [1.778002, 1.977173, 2.158131, 1.897367, 2.171239, 2.415230, 1000]

kernel_H_over_h_dict = {}
for i, kernel in enumerate(kernels_with_gaussian):
    kernel_H_over_h_dict[kernel] = kernel_H_over_h[i]


@jit(nopython=True)
def W(q: float, h: float, kernel: str = "cubic_spline"):
    """
    Evaluates various kernels.
    The kernels are scaled such that W(q > 1) = 0.
    Currently implemented:
        cubic_spline,
        quintic_spline,
        wendland_C2,
        wendland_C4,
        wendland_C6,
        gaussian (no compact support)


    Parameters
    ----------

    q: float
        dx / h, where dx is particle distance

    h: float
        compact support radius at particle position.
        compact support radius, not smoothing length!

    kernel: str
        which kernel to use


    Returns
    -------

    W: float
        evaluated kernel

    """
    #  https://pysph.readthedocs.io/en/latest/reference/kernels.html#liu2010

    if kernel == "cubic_spline":
        if q < 0.5:
            res = 3 * q ** 2 * (q - 1) + 0.5
        elif q < 1:
            #  res =  q * (q * (3 - q) - 3) + 1
            res = -(q ** 3) + 3 * q ** 2 - 3 * q + 1
        else:
            return 0

        #  sigma = 80./(7*pi*h**2)
        sigma = 3.63782727067189 / h ** 2
        return sigma * res

    elif kernel == "quartic_spline":

        if q < 0.2:
            res = 6 * q ** 4 - 2.4 * q ** 2 + 46 / 125
        elif q < 0.6:
            res = -4 * q ** 4 + 8 * q ** 3 - 4.8 * q ** 2 + 8 / 25 * q + 44.0 / 125
        elif q < 1:
            res = q ** 4 - 4 * q ** 3 + 6 * q ** 2 - 4 * q + 1
        else:
            return 0

        sigma = 5 ** 6 * 3 / (2398 * np.pi * h ** 2)
        return sigma * res

    elif kernel == "quintic_spline":
        if q < 0.333333333333:
            res = 10 * (q ** 4 * (1 - q) - 0.2222222222 * q ** 2) + 0.2682926829268
        elif q < 0.666666666666:
            qsq = q ** 2
            q4 = qsq ** 2
            res = (
                5
                * (
                    q4 * (q - 3)
                    + qsq * (3.333333333333333 * q - 1.5555555555555)
                    + 0.18518518518518517 * q
                )
                + 0.20987654320987653
            )
        elif q < 1:
            qsq = q ** 2
            res = qsq * (qsq * (5 - q) + 10 * (1 - q)) - 5 * q + 1
        else:
            return 0

        #  sigma = 3**7*7/(478*pi)/h**2
        sigma = 10.1945733213130 / h ** 2
        return res * sigma

    elif kernel == "wendland_C2":

        if q < 1:
            #  sigma = 7/(np.pi * h**2)
            sigma = 2.228169203286535 / h ** 2
            qsq = q ** 2
            return sigma * (qsq * (qsq * (4 * q - 15) + 10 * (2 * q - 1)) + 1)
        else:
            return 0

    elif kernel == "wendland_C4":

        if q < 1:
            #  sigma = 9/(np.pi*h**2)
            sigma = 2.864788975654116 / h ** 2
            qsq = q ** 2
            q4 = qsq ** 2

            return sigma * (
                11.666666666666666 * q4 * q4
                - 64 * q4 * qsq * q
                + 140 * qsq * q4
                - 149.3333333333333 * q4 * q
                + 70 * q4
                - 9.33333333333333 * qsq
                + 1
            )
        else:
            return 0

    elif kernel == "wendland_C6":

        if q < 1:
            #  sigma = 78/(7*np.pi*h**2)
            sigma = 3.546881588905096 / h ** 2
            return sigma * (
                32 * q ** 11
                - 231 * q ** 10
                + 704 * q ** 9
                - 1155 * q ** 8
                + 1056 * q ** 7
                - 462 * q ** 6
                + 66 * q ** 4
                - 11 * q ** 2
                + 1
            )
        else:
            return 0

    elif kernel == "gaussian":
        # gaussian without compact support
        return 1.0 / np.sqrt(np.pi) ** 3 / h ** 2 * np.exp(-(q ** 2))

    else:
        raise ValueError("Didn't find kernel")


@jit(nopython=True)
def dWdr(q: float, h: float, kernel: str = "cubic_spline"):
    """
    Evaluates kernel derivatives for various kernels:
    returns dW/dr = dW/dq dq/dr = 1/h * dW/dq

    The kernels are scaled such that W(q > 1) = 0.
    Currently implemented:
        cubic_spline,
        quintic_spline,
        wendland_C2,
        wendland_C4,
        wendland_C6,
        gaussian (no compact support)


    Parameters
    ----------

    q: float
        dx / h, where dx is particle distance

    h: float
        compact support radius at particle position.
        compact support radius, not smoothing length!

    kernel: str
        which kernel to use


    Returns
    -------

    dWdr: float
        evaluated kernel derivative

    """
    #  https://pysph.readthedocs.io/en/latest/reference/kernels.html#liu2010

    if kernel == "cubic_spline":
        if q < 0.5:
            res = 9 * q ** 2 - 6 * q
        elif q < 1:
            res = 6 * q - 3 * q ** 2 - 3
        else:
            return 0

        #  sigma = 80./(7*pi*h**2) * 1 / h
        sigma = 3.63782727067189 / h ** 3
        return sigma * res

    elif kernel == "quartic_spline":

        if q < 0.2:
            res = 24 * q ** 3 - 4.8 * q
        elif q < 0.6:
            res = -16 * q ** 3 + 24 * q ** 2 - 9.6 * q + 8 / 25
        elif q < 1:
            res = 4 * q ** 3 - 12 * q ** 2 + 12 * q - 4
        else:
            return 0

        sigma = 5 ** 6 * 3 / (2398 * np.pi * h ** 3)
        return sigma * res

    elif kernel == "quintic_spline":
        if q < 0.333333333333:
            res = 40 * q ** 3 - 50 * q ** 4 - 4.44444444444 * q
        elif q < 0.666666666666:
            res = (
                25 * q ** 4
                - 60 * q ** 3
                + 50 * q ** 2
                - 15.555555555555 * q
                + 0.9259259259259259
            )
        elif q < 1:
            res = 20 * q ** 3 - 5 * q ** 4 + 20 * q - 30 * q ** 2 - 5
        else:
            return 0

        #  sigma = 3**7*7/(478*pi)/h**2
        sigma = 10.1945733213130 / h ** 3
        return res * sigma

    elif kernel == "wendland_C2":

        if q < 1:
            #  sigma = 7/(np.pi * h**2)
            sigma = 2.228169203286535 / h ** 3
            return sigma * (20 * q ** 4 - 60 * q ** 3 + 60 * q ** 2 - 20 * q)
        else:
            return 0

    elif kernel == "wendland_C4":

        if q < 1:
            #  sigma = 9/(np.pi*h**2)
            sigma = 2.864788975654116 / h ** 3

            return sigma * (
                93.3333333333 * q ** 7
                - 448 * q ** 6
                + 840 * q ** 5
                - 746.6666666666 * q ** 4
                + 280 * q ** 3
                - 18.66666666666666 * q
            )
        else:
            return 0

    elif kernel == "wendland_C6":

        if q < 1:
            #  sigma = 78/(7*np.pi*h**2)
            sigma = 3.546881588905096 / h ** 3
            return sigma * (
                352 * q ** 10
                - 2310 * q ** 9
                + 6336 * q ** 8
                - 9240 * q ** 7
                + 7392 * q ** 6
                - 2772 * q ** 5
                + 264 * q ** 3
                - 22 * q
            )
        else:
            return 0

    return


def get_H(h: Union[float, np.ndarray], kernel: str = "cubic_spline"):
    """
    Compute the smoothing length in terms of the compact support length
    of a given kernel.
    The kernels defined above are defined and scaled to support a region <= H.


    Parameters
    ----------

    h: float or np.ndarray
        smoothing length

    kernel: str
        which kernel to use.

    
    Returns
    -------

    H: float or np.dnarray
        compact support radii.

    """

    return h * kernel_H_over_h_dict[kernel]
