from ..core import ModelFunction, ParameterDefinition
from typing import Union
import numpy as np


class Gaussian1(ModelFunction):
    category = 'basic'
    subcategory = 'peak'
    name = 'gaussian_1'
    description = 'Gaussian peak'
    parameters = [ParameterDefinition('a', 'amplitude', 1),
                  ParameterDefinition('c', 'constant baseline', 0),
                  ParameterDefinition('sigma', 'half width at half maximum', 1, lbound=0),
                  ParameterDefinition('x0', 'position', 1)]
    siblings = ['gaussian_2']


    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        a, c, sigma, x0 = args
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+c

    def parameters_from_sibling(self, sibling:str, *args, **kwargs):
        if sibling == 'gaussian_2':
            i, c, sigma, x0 = args
            return (i/(2*np.pi*sigma**2)**0.5, c, sigma, x0)
        else:
            raise ValueError('Unknown sibling "{}"'.format(sibling))


class Gaussian2(ModelFunction):
    category = 'basic'
    subcategory = 'peak'
    name = 'gaussian_2'
    description = 'Gaussian peak'
    parameters = [ParameterDefinition('i', 'integral', 1),
                  ParameterDefinition('c', 'constant baseline', 0),
                  ParameterDefinition('sigma', 'half width at half maximum', 1, lbound=0),
                  ParameterDefinition('x0', 'position', 1)]
    siblings = ['gaussian_1']

    def fitfunction(self, x: Union[np.ndarray, float], *args, **kwargs):
        i, c, sigma, x0 = args
        return i/(2*np.pi*sigma**2)**0.5 * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c

    def parameters_from_sibling(self, sibling: str, *args, **kwargs):
        if sibling == 'gaussian_1':
            a, c, sigma, x0 = args
            return (a*(2*np.pi*sigma**2)**0.5, c, sigma, x0)
        else:
            raise ValueError('Unknown sibling "{}"'.format(sibling))

