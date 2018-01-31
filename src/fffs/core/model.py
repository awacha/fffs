import abc
from typing import Optional, Tuple, Any, List, Union, Iterable, Type
from matplotlib.figure import Figure
import numpy as np

class ParameterDefinition:
    def __init__(self, name:str, description:str, defaultvalue:Any,
                 fittable:bool=True, lbound=-np.inf, ubound=np.inf, coerce_type = float):
        self.name = name
        self.description = description
        self.defaultvalue = coerce_type(defaultvalue)
        self.coerce_type = coerce_type
        self.fittable = fittable
        self.lbound = lbound
        self.ubound = ubound


class ModelFunction(object):
    category:Optional[str] = None
    subcategory:Optional[str] = None
    name:Optional[str] = None
    description:Optional[str] = None
    parameters:List[ParameterDefinition] = []

    @abc.abstractmethod
    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs) -> np.ndarray:
        pass

    def visualize(self, fig:Figure, x:Union[np.ndarray, float], *args, **kwargs):
        pass

    def parameters_from_sibling(self, sibling:str, *args, **kwargs) -> List[Union[float, int]]:
        pass

    @classmethod
    def categories(cls) -> Iterable[str]:
        return set([c.category for c in cls._iter_subclasses()
                    if c.category is not None])

    @classmethod
    def subcategories(cls, category:str) -> Iterable[str]:
        return set([c.subcategory for c in cls._iter_subclasses()
                    if c.category==category and c.subcategory is not None])

    @classmethod
    def models(cls, category:str, subcategory:str) -> Iterable[str]:
        return set([c.name for c in cls._iter_subclasses()
                    if c.category==category and c.subcategory==subcategory and c.name is not None])

    @classmethod
    def _iter_subclasses(cls):
        yield cls
        for c in cls.__subclasses__():
            for sc in c._iter_subclasses():
                yield sc

    @classmethod
    def model(cls, category:str, subcategory:str, name:str) -> Type['ModelFunction']:
        return [c for c in cls._iter_subclasses()
                if c.category==category and c.subcategory==subcategory and c.name==name][0]


    @classmethod
    def all_models(cls) -> List[Type['ModelFunction']]:
        for c in cls._iter_subclasses():
            if c.category is not None and c.subcategory is not None and c.name is not None:
                yield c