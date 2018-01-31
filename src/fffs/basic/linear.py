from ..core import ModelFunction, ParameterDefinition
from typing import Union
import numpy as np


class Linear(ModelFunction):
    category = 'basic'
    subcategory = 'polynomial'
    name = 'linear'
    description = 'Linear polynomial'
    parameters = [ParameterDefinition('a', 'slope', 1),
                  ParameterDefinition('b', 'offset', 0)]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        a,b = args
        return a*x+b