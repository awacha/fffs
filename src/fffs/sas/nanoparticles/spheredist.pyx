# cython: embedsignature=True, cdivision=True, boundscheck=False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport M_PI, sin, cos, exp

np.import_array()

cdef inline double Fsphere(double q, double R) nogil:
    return 4*M_PI/q**3*(sin(q*R)-q*R*cos(q*R))

cdef inline double Fsphere_unit(double q, double R) nogil:
    return 3/(q*R)**3*(sin(q*R)-q*R*cos(q*R))

cpdef np.ndarray[np.double_t, ndim=1] GaussianSizeDistributionOfSpheres(
        double[:] q, double R0, double dR, Py_ssize_t Ndistrib=100):
    cdef:
        Py_ssize_t idistrib, iq
        double r, w, weight
        np.ndarray[np.double_t, ndim=1] I
    I=np.zeros(q.size, dtype=np.double)
    weight = 0
    for idistrib in range(Ndistrib):
        if Ndistrib>1:
            r=R0 -3*dR + 6*dR*(idistrib)/<double>(Ndistrib-1.0)
            w = exp(-(r-R0)**2/(2*dR**2))
        else:
            r=R0
            w=1
        weight += w
        for iq in range(q.size):
            I[iq]+=w*Fsphere(q[iq],r)**2
    return I/weight
