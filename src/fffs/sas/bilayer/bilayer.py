from ...core import ModelFunction, ParameterDefinition
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Union, List, Tuple
import numpy as np

from .gauss_bilayer import ISSVasymm

class GaussianBilayerAsymm(ModelFunction):
    category = 'sas'
    subcategory = 'bilayer'
    name = 'gaussian_bilayer_asymm'
    description = 'Fully asymmetric gaussian bilayer'
    parameters = [ParameterDefinition('A', 'outer intensity scaling factor', 1, lbound=0),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ParameterDefinition('R0', 'radius of the innermost bilayer', 40, lbound=0),
                  ParameterDefinition('dR', 'hwhm of the radius of the innermost bilayer', 5, lbound=0),
                  ParameterDefinition('rhoGuestIn', 'relative electron density of the inner guest molecule layer (tail is -1)', 0),
                  ParameterDefinition('zGuestIn', 'distance of the inner guest molecule layer from the bilayer center',
                                      10, lbound=0),
                  ParameterDefinition('sigmaGuestIn', 'HWHM of the inner guest molecule layer', 5, lbound=0),
                  ParameterDefinition('rhoHeadIn', 'relative electron density of the inner headgroup layer (tail is -1)', 1),
                  ParameterDefinition('zHeadIn', 'distance of the inner headgroup layer from the bilayer center',
                                      2.5, lbound=0),
                  ParameterDefinition('sigmaHeadIn', 'HWHM of the inner headgroup layer', 5, lbound=0),
                  ParameterDefinition('sigmaTail', 'HWHM of the tail layer', 1, lbound=0),
                  ParameterDefinition('rhoHeadOut', 'relative electron density of the outer headgroup layer (tail is -1)', 1),
                  ParameterDefinition('zHeadOut', 'distance of the outer headgroup layer from the bilayer center',
                                      2.5, lbound=0),
                  ParameterDefinition('sigmaHeadOut', 'HWHM of the outer headgroup layer', 5, lbound=0),
                  ParameterDefinition('rhoGuestOut',
                                      'relative electron density of the outer guest molecule layer (tail is -1)', 0),
                  ParameterDefinition('zGuestOut', 'distance of the outer guest molecule layer from the bilayer center',
                                      10, lbound=0),
                  ParameterDefinition('sigmaGuestOut', 'HWHM of the outer guest molecule layer', 5, lbound=0),
                  ParameterDefinition('x_oligolam', 'Proportion of oligolamellarity',0.5,lbound=0, ubound=1),
                  ParameterDefinition('dbilayer', 'Periodic repeat distance of the bilayers', 6.4, lbound=0),
                  ParameterDefinition('ddbilayer', 'HWHM of the periodic repeat distance of the bilayers', 0.1, lbound=0),
                  ParameterDefinition('Nbilayer', 'Number of bilayers', 2, lbound=1, fittable=False, coerce_type=int),
                  ParameterDefinition('Ndistrib', 'Size distribution integration count', 1000, lbound=1, fittable=False, coerce_type=int),
                  ]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        (A, bg, R0, dR,
         rhoGuestIn, zGuestIn, sigmaGuestIn,
         rhoHeadIn, zHeadIn, sigmaHeadIn,
         sigmaTail,
         rhoHeadOut, zHeadOut, sigmaHeadOut,
         rhoGuestOut, zGuestOut, sigmaGuestOut,
         x_oligolam,
         dbilayer, ddbilayer,
         Nbilayer, Ndistrib
         ) = args
        return A*x_oligolam*ISSVasymm(x, R0, dR, rhoGuestIn, zGuestIn, sigmaGuestIn,
                                    rhoHeadIn, zHeadIn, sigmaHeadIn,
                                    -1, sigmaTail,
                                    rhoHeadOut, zHeadOut, sigmaHeadOut,
                                    rhoGuestOut, zGuestOut, sigmaGuestOut,
                                    dbilayer, ddbilayer,
                                    int(Nbilayer), int(Ndistrib)) +\
               A*(1-x_oligolam)* ISSVasymm(x, R0, dR, rhoGuestIn, zGuestIn, sigmaGuestIn,
                                         rhoHeadIn, zHeadIn, sigmaHeadIn,
                                         -1, sigmaTail,
                                         rhoHeadOut, zHeadOut, sigmaHeadOut,
                                         rhoGuestOut, zGuestOut, sigmaGuestOut,
                                         dbilayer, ddbilayer,
                                         1, int(Ndistrib)) + \
               bg


    def visualize(self, fig:Figure, x:Union[np.ndarray, float], *args, **kwargs):
        ax=fig.add_subplot(1,1,1)
        (A, bg, R0, dR,
         rhoGuestIn, zGuestIn, sigmaGuestIn,
         rhoHeadIn, zHeadIn, sigmaHeadIn,
         sigmaTail,
         rhoHeadOut, zHeadOut, sigmaHeadOut,
         rhoGuestOut, zGuestOut, sigmaGuestOut,
         x_oligolam,
         dbilayer, ddbilayer,
         Nbilayer, Ndistrib
         ) = args
        self._plotgaussians(ax, R0, dbilayer, Nbilayer, [
            (rhoGuestIn, -zGuestIn, sigmaGuestIn, 'Inner guest'),
            (rhoHeadIn, -zHeadIn, sigmaHeadIn, 'Inner head'),
            (-1, 0, sigmaTail, 'Carbon chain'),
            (rhoHeadOut, zHeadOut, sigmaHeadOut, 'Outer head'),
            (rhoGuestOut, zGuestOut, sigmaGuestOut, 'Outer guest')
        ])
        fig.canvas.draw()

    @staticmethod
    def _gaussian(x, A, x0, sigma, R0_is_zero=False):
        # the area under the peak must be 4*pi*sqrt(2*pi*sigma**2)*(x0**2+sigma**2)
        if sigma==0:
            return np.zeros_like(x)
        if not R0_is_zero:
            return A*4*np.pi*(x0**2+sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2))
        else:
            return A*4*np.pi*(sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2))

    def _plotgaussians(self, axes:Axes, R0:float, d:float, Nbilayers:int, values:List[Tuple[float,float,float,str]]):
        zmin = min([(z0-3*sigma) for rho, z0, sigma, label in values])+R0
        zmax = max([z0+3*sigma for rho, z0, sigma, label in values])+R0+(Nbilayers-1)*d
        z=np.linspace(zmin,zmax,1000)
        total=0
        for rho, z0, sigma, label in values:
            y=0
            for i in range(Nbilayers):
                y += self._gaussian(z, rho, R0+z0+i*d, sigma, R0==0)
            axes.plot(z,y,'-',label=label)
            total += y
        axes.plot(z, total, 'k-', label='Total')
        if R0 == 0:
            axes.set_xlabel('Distance from the bilayer center (nm)')
        else:
            axes.set_xlabel('Radius (nm)')
        axes.set_ylabel('Relative electron density')
        axes.grid(True, which='both')
        axes.legend(loc='best')

class GaussianBilayerAsymmGuest(GaussianBilayerAsymm):
    name = 'gaussian_bilayer_asymm_Guest'
    description = 'Symmetric gaussian bilayer with asymmetric guest layers'
    parameters = [ParameterDefinition('A', 'outer intensity scaling factor', 9.1e-12, lbound=0),
                  ParameterDefinition('bg', 'constant background', 0.00272, lbound=0),
                  ParameterDefinition('R0', 'radius of the innermost bilayer', 37.1, lbound=0),
                  ParameterDefinition('dR', 'hwhm of the radius of the innermost bilayer', 0.025, lbound=0),
                  ParameterDefinition('rhoGuestIn', 'relative electron density of the inner guest molecule layer (tail is -1)', 0),
                  ParameterDefinition('zGuestIn', 'distance of the inner guest molecule layer from the bilayer center',
                                      1.6903, lbound=0),
                  ParameterDefinition('sigmaGuestIn', 'HWHM of the inner guest molecule layer', 0.659, lbound=0),
                  ParameterDefinition('rhoHead', 'relative electron density of the headgroup layers (tail is -1)', 0.21787),
                  ParameterDefinition('zHead', 'distance of the headgroup layers from the bilayer center',
                                      1.6903, lbound=0),
                  ParameterDefinition('sigmaHead', 'HWHM of the headgroup layers', 0.132, lbound=0),
                  ParameterDefinition('sigmaTail', 'HWHM of the tail layer', 0.80258, lbound=0),
                  ParameterDefinition('rhoGuestOut',
                                      'relative electron density of the outer guest molecule layer (tail is -1)', 0),
                  ParameterDefinition('zGuestOut', 'distance of the outer guest molecule layer from the bilayer center',
                                      1.6903, lbound=0),
                  ParameterDefinition('sigmaGuestOut', 'HWHM of the outer guest molecule layer', 0.2799, lbound=0),
                  ParameterDefinition('x_oligolam', 'Proportion of oligolamellarity',0.127,lbound=0, ubound=1),
                  ParameterDefinition('dbilayer', 'Periodic repeat distance of the bilayers', 7.299164, lbound=0),
                  ParameterDefinition('ddbilayer', 'HWHM of the periodic repeat distance of the bilayers', 0.14581464, lbound=0),
                  ParameterDefinition('Nbilayer', 'Number of bilayers', 2, lbound=1, fittable=False, coerce_type=int),
                  ParameterDefinition('Ndistrib', 'Size distribution integration count', 1000, lbound=1, fittable=False, coerce_type=int),
                  ]

    def fitfunction(self, x:Union[np.ndarray, float], *args, **kwargs):
        (A, bg, R0, dR,
         rhoGuestIn, zGuestIn, sigmaGuestIn,
         rhoHead, zHead, sigmaHead,
         sigmaTail,
         rhoGuestOut, zGuestOut, sigmaGuestOut,
         x_oligolam,
         dbilayer, ddbilayer,
         Nbilayer, Ndistrib
         ) = args
        return super().fitfunction(x, A, bg, R0, dR, rhoGuestIn, zGuestIn, sigmaGuestIn,
                                   rhoHead, zHead, sigmaHead,
                                   sigmaTail,
                                   rhoHead, zHead, sigmaHead,
                                   rhoGuestOut, zGuestOut, sigmaGuestOut,
                                   x_oligolam, dbilayer, ddbilayer, Nbilayer, Ndistrib)

    def visualize(self, fig:Figure, x:Union[np.ndarray, float], *args, **kwargs):
        (A, bg, R0, dR,
         rhoGuestIn, zGuestIn, sigmaGuestIn,
         rhoHead, zHead, sigmaHead,
         sigmaTail,
         rhoGuestOut, zGuestOut, sigmaGuestOut,
         x_oligolam,
         dbilayer, ddbilayer,
         Nbilayer, Ndistrib
         ) = args
        return super().visualize(fig, x, A, bg, R0, dR, rhoGuestIn, zGuestIn, sigmaGuestIn,
                                   rhoHead, zHead, sigmaHead,
                                   sigmaTail,
                                   rhoHead, zHead, sigmaHead,
                                   rhoGuestOut, zGuestOut, sigmaGuestOut,
                                   x_oligolam, dbilayer, ddbilayer, Nbilayer, Ndistrib)


class GaussianBilayerSymm(GaussianBilayerAsymm):
    name = 'gaussian_bilayer_symm'
    description = 'Symmetric gaussian bilayer with asymmetric guest layers'
    parameters = [ParameterDefinition('A', 'outer intensity scaling factor', 1, lbound=0),
                  ParameterDefinition('bg', 'constant background', 0, lbound=0),
                  ParameterDefinition('R0', 'radius of the innermost bilayer', 40, lbound=0),
                  ParameterDefinition('dR', 'hwhm of the radius of the innermost bilayer', 5, lbound=0),
                  ParameterDefinition('rhoGuest',
                                      'relative electron density of the guest molecule layers (tail is -1)', 0),
                  ParameterDefinition('zGuest', 'distance of the guest molecule layers from the bilayer center',
                                      10, lbound=0),
                  ParameterDefinition('sigmaGuest', 'HWHM of the guest molecule layers', 5, lbound=0),
                  ParameterDefinition('rhoHead', 'relative electron density of the headgroup layers (tail is -1)', 1),
                  ParameterDefinition('zHead', 'distance of the headgroup layers from the bilayer center',
                                      2.5, lbound=0),
                  ParameterDefinition('sigmaHead', 'HWHM of the headgroup layers', 5, lbound=0),
                  ParameterDefinition('sigmaTail', 'HWHM of the tail layer', 1, lbound=0),
                  ParameterDefinition('x_oligolam', 'Proportion of oligolamellarity', 0.5, lbound=0, ubound=1),
                  ParameterDefinition('dbilayer', 'Periodic repeat distance of the bilayers', 6.4, lbound=0),
                  ParameterDefinition('ddbilayer', 'HWHM of the periodic repeat distance of the bilayers', 0.1,
                                      lbound=0),
                  ParameterDefinition('Nbilayer', 'Number of bilayers', 2, lbound=1, fittable=False, coerce_type=int),
                  ParameterDefinition('Ndistrib', 'Size distribution integration count', 1000, lbound=1, fittable=False,
                                      coerce_type=int),
                  ]

    def fitfunction(self, x: Union[np.ndarray, float], *args, **kwargs):
        (A, bg, R0, dR,
         rhoGuest, zGuest, sigmaGuest,
         rhoHead, zHead, sigmaHead,
         sigmaTail,
         x_oligolam,
         dbilayer, ddbilayer,
         Nbilayer, Ndistrib
         ) = args
        return super().fitfunction(x, A, bg, R0, dR, rhoGuest, zGuest, sigmaGuest,
                                   rhoHead, zHead, sigmaHead,
                                   sigmaTail,
                                   rhoHead, zHead, sigmaHead,
                                   rhoGuest, zGuest, sigmaGuest,
                                   x_oligolam, dbilayer, ddbilayer, Nbilayer, Ndistrib)

    def visualize(self, fig: Figure, x: Union[np.ndarray, float], *args, **kwargs):
        (A, bg, R0, dR,
         rhoGuest, zGuest, sigmaGuest,
         rhoHead, zHead, sigmaHead,
         sigmaTail,
         x_oligolam,
         dbilayer, ddbilayer,
         Nbilayer, Ndistrib
         ) = args
        return super().visualize(fig, x, A, bg, R0, dR, rhoGuest, zGuest, sigmaGuest,
                                 rhoHead, zHead, sigmaHead,
                                 sigmaTail,
                                 rhoHead, zHead, sigmaHead,
                                 rhoGuest, zGuest, sigmaGuest,
                                 x_oligolam, dbilayer, ddbilayer, Nbilayer, Ndistrib)
