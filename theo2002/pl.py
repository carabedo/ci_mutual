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


#bob paralela


b_pl={
'x0' : 3e-3,
'y0' : 2e-3,
'c'  : 2e-3,
'l1' : 0.5e-3,
'l2' : 2.5e-3,
'z1' : 0.5e-3,
'z2' : 2.5e-3,
'N'  : 400,
'L0' : 980e-6
}


x0=b_pl['x0']
y0=b_pl['y0']
c=b_pl['c']
l1=b_pl['l1']
l2=b_pl['l2']
z1=b_pl['z1']
z2=b_pl['z2']
N=b_pl['N']
L0=b_pl['L0']

#propiedades de la probeta
mur=1
sigma=35.4e6
mu0=4*np.pi*10e-7

# constantes del ensayo
I=1e-3 #corriente

# fava

k=lambda u,v : np.sqrt(u**2+v**2)
l=lambda u,v : np.sqrt(u**2+v**2 + 1j*w*mur*mu0*sigma)

# ec 8
fl=lambda u,v : (2*k(u,v)*mur*np.exp(l(u,v)*z))/(k(u,v)*mur+l(u,v))

# 

def K(u,v):
      return (mu0*I*Aii(u,v)*(np.exp(-k(u,v)*z1) -np.exp(-k(u,v)*z2) ))/(2*np.pi**2*(z2-z1)*c*u*v*k(u,v)**2)

def getk(t):
    N=len(t)
    dt=t[1]-t[0]
    k = fftfreq(N,d=dt)
    return fftshift(k)

def fxab(a,b):
    return 2*w*sigma*Dsii(a,b)*fl(a,b)*exp(l(a,b)*z)*b

def fyab(a,b):
    return -2*w*sigma*Dsii(a,b)*fl(a,b)*exp(l(a,b)*z)*a

c0=(mu0*N*I) /  (2*pi**2*c*(l2-l1))

Dsii=lambda u,v : (c0 * Aii(u,v)*( exp(-k(u,v)*(l2))-exp(-k(u,v)*(l1)))  )  /  (u*v*(-k(u,v))**2);

# ec 16:
def Aii(u,v):
    if u==v:
        res=c*cos(u*(x0 -y0))/2 - sin(u*(2*c+x0+y0))/(4*u) - sin(u*(x0+y0))/(4*u)
    else:
        r0=  (sin(c*(u-v)+ u*x0 - v*y0) - sin(u*x0 - v*y0) )/(2*(u-v))    

        r1=  (sin(c*(u+v)+ u*x0 + v*y0) - sin(u*x0 + v*y0))/(2*(u+v)) 

        res= r0 - r1
    return res

def jxjy(av,bv):
    ifx = fxab(av[:,None], bv[None,:])
    ify = fyab(av[:,None], bv[None,:])
    Jx,Jy=fftshift(ifft2(ifftshift(np.nan_to_num(ifx.T)))).real,
    fftshift(ifft2(ifftshift(np.nan_to_num(ify.T)))).real
    return(Jx,Jy)

def plot(av,bv,Jx,Jy,titulo=''):
    xv,yv=getk2(av),getk2(bv)
    plt.figure(figsize=[10,10])
    plt.quiver(xv/(1e-3),yv/(1e-3),Jx,Jy)
    plt.title(titulo)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.grid(True)
    return
