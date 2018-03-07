from ...core import ParameterDefinition, ModelFunction
from typing import Union
from .spheredist import GaussianSizeDistributionOfSpheres
from .structurefactors import HardSphere
import numpy as np

class GaussSpheres(ModelFunction):
    category = 'sas'
    subcategory = 'nanoparticles'
    name = 'GaussSpheres'
    description = 'Spherical nanoparicles with Gaussian size distribution'
    parameters = [ParameterDefinition('I0', 'intensity at q=0', 1, lbound=0),
                  ParameterDefinition('R', 'mean radius', 1, lbound=0),
                  ParameterDefinition('dR', 'HWHM of the radius distribution', 3, lbound=1, ubound=3),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ParameterDefinition('Ndistrib', 'Number of steps for numeric integration', 100, lbound=1, fittable=False),
                  ]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        I0, R, dR, bg, Ndistrib = args
        return I0*GaussianSizeDistributionOfSpheres(x, R, dR)+bg

class GaussHardSpheres(ModelFunction):
    category = 'sas'
    subcategory = 'nanoparticles'
    name = 'GaussHardSpheres'
    description = 'Spherical nanoparicles with Gaussian size distribution and hard sphere structure factor'
    parameters = [ParameterDefinition('I0', 'intensity at q=0', 1, lbound=0),
                  ParameterDefinition('R', 'mean radius', 1, lbound=0),
                  ParameterDefinition('dR', 'HWHM of the radius distribution', 3, lbound=1, ubound=3),
                  ParameterDefinition('Rhs', 'hard sphere radius', 1, lbound=0),
                  ParameterDefinition('fp', 'volume fraction', 0.1, lbound=0, ubound=1),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ParameterDefinition('Ndistrib', 'Number of steps for numeric integration', 100, lbound=1, fittable=False),
                  ]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        I0, R, dR, Rhs, fp, bg, Ndistrib = args
        return I0*GaussianSizeDistributionOfSpheres(x, R, dR)*HardSphere(x, Rhs, fp)+bg
