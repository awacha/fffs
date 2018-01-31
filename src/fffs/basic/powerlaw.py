from ..core import ModelFunction, ParameterDefinition
import numpy as np
from typing import Union

class PowerLaw(ModelFunction):
    category = 'basic'
    subcategory = 'power-law'
    name = 'power-law'
    description = 'Power-law function'
    parameters = [ParameterDefinition('a', 'Scaling factor', 1),
                  ParameterDefinition('alpha', 'Exponent', 1)]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        a,alpha=args
        return a*x**alpha