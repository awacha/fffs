from ...core import ParameterDefinition, ModelFunction
from typing import Union
from .c_ellipsoid import AsymmetricEllipsoidalShell, EllipsoidalShellWithSizeDistribution
import numpy as np

class EllipsoidalCoreShell(ModelFunction):
    category = 'sas'
    subcategory = 'coreshell'
    name = 'EllipticCoreShell'
    description = 'Elliptic core-shell particle'
    parameters = [ParameterDefinition('I0', 'intensity at q=0', 1, lbound=0),
                  ParameterDefinition('Rg', 'radius of gyration', 1, lbound=0),
                  ParameterDefinition('a', 'semi-axis of revolution', 1, lbound=0),
                  ParameterDefinition('bdiva_mean', 'mean value of the ratio of the equatorial and the revolutional semi-axes', 3, lbound=1, ubound=3),
                  ParameterDefinition('bdiva_hwhm', 'hwhm of the Gaussian distribution of b/a', 0.1,lbound=0),
                  ParameterDefinition('ta', 'shell thickness along the axis of revolution', 1, lbound=0),
                  ParameterDefinition('tbdivta_mean', 'mean value of the ratio of the equatorial and the revolutional shell thicknesses', 1, lbound=0),
                  ParameterDefinition('tbdivta_hwhm', 'hwhmo of the Gaussian distribution of tb/ta', 0.1, lbound=0),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ParameterDefinition('Nintorientation', 'Number of equidistant points in averaging over orientation', 10, lbound=3, fittable=False),
                  ParameterDefinition('Nintanisometrycore', 'Number of equidistant points in the integration of the Gaussian distribution on b/a', 10, lbound=1, fittable=False),
                  ParameterDefinition('Nintanisometryshell', 'Number of equidistant points in the integration of the Gaussian distribution on tb/ta', 10, lbound=1, fittable=False),
                  ]

    def _get_rhos(self, I0,Rg,a,b,ta,tb):
        btb=b+tb
        ata=a+ta
        btb2ata = btb**2*ata
        b2a = b**2*a
        rhocoredivrhoshell = (ata**3*btb**2+2*ata*btb**4-5*Rg**2*ata*btb**2-a**3*b**2-2*a*b**4+5*a*b**2*Rg**2)/(5*a*b**2*Rg**2-a**3*b**2-2*a*b**4)
        eta_shell = 3*I0**0.5/4/np.pi/(rhocoredivrhoshell*b2a+btb2ata-b2a)
        eta_core = rhocoredivrhoshell*eta_shell
        return eta_core, eta_shell

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        I0, Rg, a, bdiva_mean, bdiva_hwhm, ta, tbdivta_mean, tbdivta_hwhm, bg, Nintorientation, Nintanisometrycore, Nintanisometryshell = args
        rhocore, rhoshell = self._get_rhos(I0, Rg, a, bdiva_mean*a, ta, tbdivta_mean*ta)
        return EllipsoidalShellWithSizeDistribution(
            x,
            rhocore, rhoshell,
            a, bdiva_mean, bdiva_hwhm,
            ta, tbdivta_mean, tbdivta_hwhm,
            int(Nintorientation), int(Nintanisometrycore), int(Nintanisometryshell))+bg


class EllipsoidalCoreShellRelative(ModelFunction):
    category = 'sas'
    subcategory = 'coreshell'
    name = 'EllipticCoreShell_rel'
    description = 'Elliptic core-shell particle'
    parameters = [ParameterDefinition('I0', 'intensity at q=0', 1, lbound=0),
                  ParameterDefinition('Rg', 'radius of gyration', 1, lbound=0),
                  ParameterDefinition('a', 'semi-axis of revolution', 1, lbound=0),
                  ParameterDefinition('bdiva_mean', 'mean value of the ratio of the equatorial and the revolutional semi-axes', 3, lbound=1, ubound=3),
                  ParameterDefinition('bdiva_hwhm', 'hwhm of the Gaussian distribution of b/a', 0.1,lbound=0),
                  ParameterDefinition('tadiva', 'ta/a', 1, lbound=0),
                  ParameterDefinition('tbdivta_mean', 'mean value of the ratio of the equatorial and the revolutional shell thicknesses', 1, lbound=0),
                  ParameterDefinition('tbdivta_hwhm', 'hwhmo of the Gaussian distribution of tb/ta', 0.1, lbound=0),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ParameterDefinition('Nintorientation', 'Number of equidistant points in averaging over orientation', 10, lbound=3, fittable=False),
                  ParameterDefinition('Nintanisometrycore', 'Number of equidistant points in the integration of the Gaussian distribution on b/a', 10, lbound=1, fittable=False),
                  ParameterDefinition('Nintanisometryshell', 'Number of equidistant points in the integration of the Gaussian distribution on tb/ta', 10, lbound=1, fittable=False),
                  ]

    def _get_rhos(self, I0,Rg,a,b,ta,tb):
        btb=b+tb
        ata=a+ta
        btb2ata = btb**2*ata
        b2a = b**2*a
        rhocoredivrhoshell = (ata**3*btb**2+2*ata*btb**4-5*Rg**2*ata*btb**2-a**3*b**2-2*a*b**4+5*a*b**2*Rg**2)/(5*a*b**2*Rg**2-a**3*b**2-2*a*b**4)
        eta_shell = 3*I0**0.5/4/np.pi/(rhocoredivrhoshell*b2a+btb2ata-b2a)
        eta_core = rhocoredivrhoshell*eta_shell
        return eta_core, eta_shell

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        I0, Rg, a, bdiva_mean, bdiva_hwhm, tadiva, tbdivta_mean, tbdivta_hwhm, bg, Nintorientation, Nintanisometrycore, Nintanisometryshell = args
        rhocore, rhoshell = self._get_rhos(I0, Rg, a, bdiva_mean*a, tadiva*a, tbdivta_mean*tadiva*a)
        return EllipsoidalShellWithSizeDistribution(
            x,
            rhocore, rhoshell,
            a, bdiva_mean, bdiva_hwhm,
            tadiva*a, tbdivta_mean, tbdivta_hwhm,
            int(Nintorientation), int(Nintanisometrycore), int(Nintanisometryshell))+bg
