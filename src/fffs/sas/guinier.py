from ..core import ModelFunction, ParameterDefinition
from typing import Union
import numpy as np


class Guinier(ModelFunction):
    category = 'sas' # 'basic'
    subcategory = 'guinier'
    name = 'Guinier'
    description = 'Simple Guinier function'
    parameters = [ParameterDefinition('I0', 'intensity at q=0', 1, lbound=0),
                  ParameterDefinition('Rg', 'radius of gyration', 1, lbound=0),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        I0, Rg, bg = args
        return I0*np.exp(-x**2*Rg**2/3) + bg

class GeneralGuinier(ModelFunction):
    category = 'sas'
    subcategory = 'guinier'
    name = 'GeneralGuinier'
    description = 'Generalized Guinier function'
    parameters = [ParameterDefinition('I0', 'intensity at q=0', 1, lbound=0),
                  ParameterDefinition('Rg', 'radius of gyration', 1, lbound=0),
                  ParameterDefinition('d', 'dimensionality', 1, lbound=1, ubound=3),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ]

    def fitfunction(self, x: Union[np.ndarray, float], *args, **kwargs):
        I0, Rg, d, bg = args
        return I0*x**(-(3-d))*np.exp(-x**2*Rg**2/d) + bg

class GuinierPorod(ModelFunction):
    category = 'sas'
    subcategory = 'guinier'
    name = 'GuinierPorod'
    description = 'Empirical Guinier-Porod function'
    parameters = [ParameterDefinition('I0', 'intensity at q=0', 1, lbound=0),
                  ParameterDefinition('Rg', 'radius of gyration', 1, lbound=0),
                  ParameterDefinition('d', 'dimensionality', 3, lbound=1, ubound=3),
                  ParameterDefinition('alpha', 'power-law exponent', -4, ubound=0),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        I0, Rg, d, alpha, bg = args
        qsep = (-alpha*d-3*d+d**2)**0.5*0.5**0.5/Rg
        A = I0*qsep**(d-3-alpha)*np.exp(-qsep**2*Rg**2/d)
        return np.piecewise(x, [x<qsep, x>=qsep],
                            [lambda x, I0=I0, Rg=Rg, d=d:I0*x**(d-3)*np.exp(-x**2*Rg**2/d),
                             lambda x, A=A, alpha=alpha: A*x**alpha]) + bg

