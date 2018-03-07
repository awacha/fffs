# cython: embedsignature=True, cdivision=True, boundscheck=False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport M_PI, sin, cos, exp

np.import_array()

cdef inline double _HardSphere(double q, double Rhs, double fp) nogil:
    cdef:
        double alpha = (1+2*fp)**2/(1-fp)**4
        double beta = -6*fp*(1+fp/2)**2/(1-fp)**4
        double gamma = fp*alpha/2
        double A =2*Rhs*q
        double s = sin(A)
        double c = cos(A)
        double G =alpha*(s-A*c)/A**2+beta*(2*A*s+(2-A**2)*c-2)/A**3+gamma*(-A**4*c+4*((3*A**2-6)*c+(A**3-6*A)*s+6))/A**5
    return 1/(1+24*fp*G/Rhs/q)

cpdef np.ndarray[np.double_t, ndim=1] HardSphere(double[:] q, double Rhs, double fp):
    """Hard sphere structure factor.
    
    Inputs:
        q (array of double): scattering variable
        Rhs (double): hard sphere radius
        fp (double): volume proportion
        
    Outputs:
        (np.ndarray of dtype np.double, ndim=1): scattering structure factor
    """
    cdef:
        np.ndarray[np.double_t, ndim=1] I
        Py_ssize_t iq
    I=np.zeros(q.size, dtype=np.double)
    for iq in range(q.size):
        I[iq]=_HardSphere(q[iq], Rhs, fp)
    return I
