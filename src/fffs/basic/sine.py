from ..core import ModelFunction, ParameterDefinition
from typing import Union
import numpy as np


class Sine(ModelFunction):
    category = 'basic'
    subcategory = 'trigonometric'
    name = 'sine'
    description = 'Sine function'
    parameters = [ParameterDefinition('a', 'amplitude', 1),
                  ParameterDefinition('c', 'constant baseline', 0),
                  ParameterDefinition('omega', 'circular frequency', 1, lbound=0),
                  ParameterDefinition('x0', 'starting phase', 1)]
    #siblings = ['sine']


    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        a, c, omega, x0 = args
        return a*np.sin(omega*x-x0)+c

#    def parameters_from_sibling(self, sibling:str, *args, **kwargs):
#        if sibling == 'gaussian_2':
#            i, c, sigma, x0 = args
#            return (i/(2*np.pi*sigma**2)**0.5, c, sigma, x0)
#        else:
#            raise ValueError('Unknown sibling "{}"'.format(sibling))
