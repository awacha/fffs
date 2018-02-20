# cython: embedsignature=True, cdivision=True, boundscheck=False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, M_PI, sin, cos, sqrt, log
from libc.stdlib cimport rand, RAND_MAX, malloc, free
from libc.string cimport memset

np.import_array()

#include <stdlib.h>
#include <math.h>

cdef inline double gaussrand():
    cdef:
        double x,y
        int done
    while not done:
        x = log(rand()/<double>RAND_MAX)
        y=exp(-0.5*(x-1)*(x-1))
        if (rand()/<double>RAND_MAX < y):
            if rand()/<double>RAND_MAX < 0.5:
                x = -x
            done = 1
    return x

cdef inline double FGaussLayer(double q, double R0, double sigma) nogil:
    """Calculates the scattering amplitude of a spherical gaussian electron
    density profile positioned at r=R0 and its HWHM being sigma.
    """
    return 4*M_PI*sqrt(2*M_PI*sigma**2)* \
       exp(-sigma**2*q**2/2)*(R0/q*sin(q*R0)+sigma**2*cos(q*R0))

cpdef np.ndarray[np.double_t, ndim=1] ISSVasymm(double[:] q, double R0, double dR,
                          double rhoGuestIn, double zGuestIn, double sigmaGuestIn,
                          double rhoHeadIn, double zHeadIn, double sigmaHeadIn,
                          double rhoTail, double sigmaTail,
                          double rhoHeadOut, double zHeadOut, double sigmaHeadOut,
                          double rhoGuestOut, double zGuestOut, double sigmaGuestOut,
                          double dbilayer, double ddbilayer=0,
                          Py_ssize_t Nbilayers=1, Py_ssize_t Ndistrib=1):
    cdef:
        Py_ssize_t ibilayer, idistrib, iq
        double r, w, weight
        double *F
        np.ndarray[np.double_t, ndim=1] I
    F=<double*>malloc(sizeof(double)*q.size)
    I=np.zeros(q.size, dtype=np.double)
    weight = 0
    for idistrib in range(Ndistrib):
        memset(F, 0, sizeof(double)*q.size)
        if Ndistrib>1:
            r=R0 -3*dR + 6*dR*(idistrib)/(Ndistrib-1)
            w = exp(-(r-R0)**2/(2*dR**2))
        else:
            r=R0
            w=1
        weight += w
        for ibilayer in range(Nbilayers):
            for iq in range(q.size):
                F[iq] += rhoGuestIn*FGaussLayer(q[iq], r-zGuestIn, sigmaGuestIn) +\
                         rhoHeadIn*FGaussLayer(q[iq], r-zHeadIn, sigmaHeadIn) +\
                         rhoTail*FGaussLayer(q[iq], r, sigmaTail) + \
                         rhoHeadOut*FGaussLayer(q[iq], r+zHeadOut, sigmaHeadOut) + \
                         rhoGuestOut*FGaussLayer(q[iq], r+zGuestOut, sigmaGuestOut)
            if ibilayer<Nbilayers-1:
                r += ddbilayer*gaussrand()+dbilayer
        for iq in range(q.size):
            I[iq]+=F[iq]**2

    free(F)
    return I/weight

cpdef np.ndarray[np.double_t, ndim=1] ISSVasymmGuest(double[:] q, double R0, double dR,
                               double rhoGuestIn, double zGuestIn, double sigmaGuestIn,
                               double rhoHead, double zHead, double sigmaHead,
                               double rhoTail, double sigmaTail,
                               double rhoGuestOut, double zGuestOut, double sigmaGuestOut,
                               double dbilayer, double ddbilayer=0,
                               Py_ssize_t Nbilayers=1, Py_ssize_t Ndistrib=1
                               ):
    return ISSVasymm(q, R0, dR, rhoGuestIn, zGuestIn, sigmaGuestIn,
                     rhoHead, zHead, sigmaHead,
                     rhoTail, sigmaTail,
                     rhoHead, zHead, sigmaHead,
                     rhoGuestOut, zGuestOut, sigmaGuestOut,
                     dbilayer, ddbilayer, Nbilayers, Ndistrib)

cpdef np.ndarray[np.double_t, ndim=1] ISSVsymm(double[:] q, double R0, double dR,
                         double rhoGuest, double zGuest, double sigmaGuest,
                         double rhoHead, double zHead, double sigmaHead,
                         double rhoTail, double sigmaTail,
                         double dbilayer, double ddbilayer=0,
                         Py_ssize_t Nbilayers=1, Py_ssize_t Ndistrib=1):
    return ISSVasymm(q, R0, dR, rhoGuest, zGuest, sigmaGuest, rhoHead, zHead, sigmaHead, rhoTail, sigmaTail,
                     rhoHead, zHead, sigmaHead, rhoGuest, zGuest, sigmaGuest,
                     dbilayer, ddbilayer, Nbilayers, Ndistrib)
