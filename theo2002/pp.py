import numpy as np
import matplotlib.pyplot as plt
from numpy import cos
from numpy import sin
from numpy import sinh
from numpy import cosh
from numpy import exp
from numpy import pi
from numpy.fft import fftshift,ifftshift,fftfreq
from numpy.fft import ifft2


b_pp={  
'xo' : 1e-3,
'yo' : 2e-3,
'zo' : 2e-3,
'c'  : 2e-3,
'zc' : 4.5e-3,
'N'  : 400,
}
# bob perpendisular
xo=1e-3
yo=2e-3
zo=2e-3
c=2e-3
zc=4.5e-3
N=400

#propiedades de la probeta
mur=1
sigma=35.4e6
mu0=4*np.pi*10e-7


# constantes del ensayo
I=1e-3; #corriente

k=lambda a,b : np.sqrt(a**2+b**2)
l=lambda a,b : np.sqrt(a**2+b**2 + 1j*w*mur*mu0*sigma)
fl=lambda a,b : (2*k(a,b)*mur)/(k(a,b)*mur+l(a,b))
Ds=lambda a,b : (1j*mu0*N*I*A(a,b)*sin(a*xo)*exp(-k(a,b)*zc))/(pi*pi*c*2*xo*b*(k(a,b))**3)
A=lambda a,b : (-b*cos(b*(yo+c))*sinh(k(a,b)*(zo+c)) +  b*cos(b*yo)*sinh(k(a,b)*zo) +   k(a,b)*sin(b*(yo+c))*cosh(k(a,b)*(zo+c)) - k(a,b)*sin(b*yo)*cosh(k(a,b)*zo)      )/(a**2+2*(b**2))

def fxab(a,b):    
    return 2*w*sigma*Ds(a,b)*fl(a,b)*exp(l(a,b)*z)*b

def fyab(a,b):
    return -2*w*sigma*Ds(a,b)*fl(a,b)*exp(l(a,b)*z)*a

def getk(t):
    N=len(t)
    dt=t[1]-t[0]
    k = fftfreq(N,d=dt)
    return fftshift(k[1:])

def getk2(t):
    N=len(t)
    dt=t[1]-t[0]
    k = fftfreq(N,d=dt)
    return fftshift(k)  

def jxjy(av,bv):
    ifx = fxab(av[:,None], bv[None,:])
    ify = fyab(av[:,None], bv[None,:])
    Jx,Jy=fftshift(ifft2(ifftshift(np.nan_to_num(ifx.T)))).real,fftshift(ifft2(ifftshift(np.nan_to_num(ify.T)))).real
    return(Jx,Jy)


def plot(xv,yv,Jx,Jy,titulo=''):
    plt.figure(figsize=[10,10])
    plt.quiver(xv/(1e-3),yv/(1e-3),Jx,Jy)
    plt.title(titulo)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.grid(True)
    return
