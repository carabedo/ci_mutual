

import pandas as pd
from scipy import special
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import cos
from numpy import sin
from numpy import sinh
from numpy import cosh
from numpy import exp
from numpy import pi
from numpy.fft import fftshift,ifftshift,fftfreq
from numpy.fft import ifft2



frec=1000
w=2*np.pi*frec
mu0=1.26e-6

## tabla1

r1=2e-3
r2=4e-3
l=2e-3
N=400
l0=1e-3
n=N/((r2 -r1)*l)
lo=0.000741


sigma1=18.72e6
mur1=1


tita=math.radians(90)
x0=0
y0=0
z=-1e-3
coil=[r1,r2,l,n,tita,x0,y0,l0]
i0=1e-3


w=2*np.pi*frec
cj=w*sigma1*mu0*mur1*i0/np.pi

## ec 8
#paralela
def h1s(u,v,coil):
    r1=coil[0]
    r2=coil[1]
    l=coil[2]
    n=coil[3]
    l0=coil[7]
    a=np.sqrt(u**2+ v**2)
    h=n*J(a*r1,a*r2)*(np.exp(-a*(l+l0))-np.exp(-a*l0))/(2*a**3)
    return(h)

def aexp(w,z,u,v):
    a0=a(u,v)
    a1=np.sqrt(a0**2 + 1j*w*mur1*mu0*sigma1)
    num=np.exp(a1*z)
    dem=a0*(a0*mur1 + a1)
    return num/dem



## aux
def a(u,v):
    return np.sqrt(u**2+ v**2)

def J(x1,x2):
    y= (x1*(special.jv(0,x1)*special.struve(1, x1)-special.jv(1,x1)*special.struve(0, x1))
    -x2*(special.jv(0,x2)*special.struve(1, x2)-special.jv(1,x2)*special.struve(0, x2)))/(np.pi/2)
    return y


## ec 4

fkx= lambda v, u: h1s(u,v,coil)*v*aexp(w,z,u,v)
fky= lambda v, u: h1s(u,v,coil)*u*aexp(w,z,u,v)


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


xm=6e-3
xv=np.linspace(-xm,xm,200)
yv=np.linspace(-xm,xm,200)
av,bv=getk(xv),getk(yv)


ifx = fkx(av[:,None], bv[None,:])
ify = fky(av[:,None], bv[None,:])

Jx,Jy=fftshift(ifft2(ifftshift(np.nan_to_num(ifx.T)))).real,fftshift(ifft2(ifftshift(np.nan_to_num(ify.T)))).real

xv,yv=getk2(av),getk2(bv)

plt.figure(figsize=[10,10])
plt.quiver(xv/(1e-3),yv/(1e-3),Jx,Jy)
plt.title('Corrientes Inducidas en placa semi-infinita por una bobina orientada de manera perpendicular')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.grid(True)
