from ..core import ModelFunction, ParameterDefinition
from typing import Union
import numpy as np


class Lorentzian1(ModelFunction):
    category = 'basic'
    subcategory = 'peak'
    name = 'lorentzian_1'
    description = 'Lorentzian peak'
    parameters = [ParameterDefinition('a', 'amplitude', 1),
                  ParameterDefinition('c', 'constant baseline', 0),
                  ParameterDefinition('sigma', 'half width at half maximum', 1, lbound=0),
                  ParameterDefinition('x0', 'position', 1)]
    siblings = ['lorentzian_2']

    def fitfunction(self, x: Union[np.ndarray, float], *args, **kwargs):
        a, c, gamma, x0 = args
        return a*(0.5*gamma)**2/((x-x0)**2+(0.5*gamma)**2)+c

    def parameters_from_sibling(self, sibling: str, *args, **kwargs):
        if sibling == 'lorentzian_2':
            i, c, gamma, x0 = args
            return (2* i / (np.pi*gamma), c, gamma, x0)
        else:
            raise ValueError('Unknown sibling "{}"'.format(sibling))


class Lorentzian2(ModelFunction):
    category = 'basic'
    subcategory = 'peak'
    name = 'lorentzian_2'
    description = 'Lorentzian peak'
    parameters = [ParameterDefinition('i', 'integral', 1),
                  ParameterDefinition('c', 'constant baseline', 0),
                  ParameterDefinition('gamma', 'full width at half maximum', 1, lbound=0),
                  ParameterDefinition('x0', 'position', 1)]
    siblings = ['lorentzian_1']

    def fitfunction(self, x: Union[np.ndarray, float], *args, **kwargs):
        i, c, gamma, x0 = args
        return i * 0.5 * gamma /np.pi / ((x-x0)**2+(0.5*gamma)**2) + c

    def parameters_from_sibling(self, sibling: str, *args, **kwargs):
        if sibling == 'lorentzian_1':
            a, c, gamma, x0 = args
            return (a*np.pi*gamma*0.5, c, gamma, x0)
        else:
            raise ValueError('Unknown sibling "{}"'.format(sibling))

