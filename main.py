# implementacion del paper theo2007
# mutual impedanci 2 coils arbitrary position

from this import d
import pandas as pd
from pyrsistent import l
from scipy import integrate
from scipy import special
import numpy as np
# Examples

# Compute the double integral of x * y**2 over the box x ranging from 0 to 2 and y ranging from 0 to 1.

# from scipy import integrate
# f = lambda y, x: x*y**2
# integrate.dblquad(f, 0, 2, lambda x: 0, lambda x: 1)
#     (0.6666666666666667, 7.401486830834377e-15)

f=1000
lim=1000
w=2*np.pi*f
mu0=1.26e-6

## tabla1
#driver
dr1=7.04e-3
dr2=12.2e-3
dl=5.04e-3
dN=544
dl0=5.55e-6
#pickup
pr1=7.04e-3
pr2=12.40e-3
pl=5.04e-3
pN=556
pl0=5.84e-6
#placa
sigma1=29.4e6
mu1=1
c1=2.47e-3

d= l0 + r2*np.sin(tita) + (l/2)*np.cos(tita)

psi=u*np.sin(tita) + 1j*a(u,v)*np.cos(tita)


def I(x1,x2):
    return integrate.quad(lambda x: x*special.iv(1,x), x1, x2)


def H(u,v):
    h=1j*n*I(psi*r1,psi*r2)*np.exp(-a(u,v)*d)*np.sin(psi*l/2)/psi**3
    return h


def R(u,v):
    a1=a(u,v)
    b1=np.sqrt(a1**2 + 1j*w*mu1*mu0*sigma1)/mu1 
    r=(a1-b1)/(a1+b1)  
    return r

def a(u,v):
    return np.sqrt(u**2+ v**2)

def mutual(x,y,d,t):
    f = lambda v, u: (a(u,v)**-1)*H(u,v)*H(-u,-v)*R(u,v)
    dz=2*w*mu0*integrate.dblquad(f, -lim, lim, lambda u: -lim, lambda u: lim)
    return dz

