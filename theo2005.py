#python
# theo 2005 
# implementacion del paper theo2005
# anlytical model for tilted coils


import pandas as pd
from scipy import integrate
from scipy import special
import numpy as np
import math
import matplotlib.pyplot as plt




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
sigma1=18.72e6
mur1=1
coil=[r1,r2,l,n,l0]
i0=1e-3


# ec. 3

def dz(coil,frec,sigma,mur,lmax=1000):
    w=2*np.pi*frec
    c=1j*8*w*mu0
    
    f_re= lambda v, u: np.real(np.abs(hs(u,v,coil))**2*g(u,v,w,sigma,mur))
    f_im= lambda v, u: np.imag(np.abs(hs(u,v,coil))**2*g(u,v,w,sigma,mur))

    dz_re=c*integrate.dblquad(f_re,0,lmax, lambda u: 0, lambda u: lmax)[0]
    dz_im=c*integrate.dblquad(f_im,0,lmax, lambda u: 0, lambda u: lmax)[0]

    return(dz_re + dz_im)


def g(u,v,w,sigma,mur):
    a0=np.sqrt(u**2+ v**2)
    a1=np.sqrt(a0**2 + 1j*w*mur*mu0*sigma)
    num=mur*a0-a1
    dem=a0*(a0*mur + a1)
    return num/dem


## ec 8

def hs(u,v,coil):
    '''8'''
    r1=coil[0]
    r2=coil[1]
    l=coil[2]
    n=coil[3]
    l0=coil[4]
    a=np.sqrt(u**2+ v**2)
    h=n*J(a*r1,a*r2)*(np.exp(-a*(l+l0))-np.exp(-a*l0))/(2*a**3)
    return(h)




## ec. 4
def Jx(x,y,z,coil,frec,lim=1000):
    w=2*np.pi*frec
    cj=w*sigma1*mu0*mur1*i0/np.pi
    fx_re= lambda v, u: np.real(hs(u,v,coil)*v*aexp(x,y,z,u,v,w))
    fx_im=  lambda v, u: np.imag(hs(u,v,coil)*v*aexp(x,y,z,u,v,w))
    # real -inf,+inf
    jx_re_n=cj*integrate.dblquad(fx_re,-lim, -0.01, lambda u: -lim, lambda u: -0.01)[0]
    jx_re_p=cj*integrate.dblquad(fx_re,0.01, lim, lambda u: 0.01, lambda u: lim)[0]
    #imag -inf,+inf
    jx_im_n=cj*integrate.dblquad(fx_im,-lim, -0.01, lambda u: -lim, lambda u: -0.01)[0]
    jx_im_p=cj*integrate.dblquad(fx_im,0.01, lim, lambda u: 0.01, lambda u: lim)[0]
    return jx_re_n+jx_re_p+1j*jx_im_n+1j*jx_im_p


def Jy(x,y,z,coil,frec,lim=1000):
    w=2*np.pi*frec
    cj=w*sigma1*mu0*mur1*i0/np.pi
    fy_re= lambda v, u: -np.real(hs(u,v,coil)*u*aexp(x,y,z,u,v,w))
    fy_im=  lambda v, u: -np.imag(hs(u,v,coil)*u*aexp(x,y,z,u,v,w))
    jy_re_n=cj*integrate.dblquad(fy_re,-lim, -0.01, lambda u: -lim, lambda u: -0.01)[0]
    jy_re_p=cj*integrate.dblquad(fy_re,0.01, lim, lambda u: 0.01, lambda u: lim)[0]
    jy_im_n=cj*integrate.dblquad(fy_im,-lim, -0.01, lambda u: -lim, lambda u: -0.01)[0]
    jy_im_p=cj*integrate.dblquad(fy_im,0.01, lim, lambda u: 0.01, lambda u: lim)[0]
    return jy_re_n+jy_re_p+1j*jy_im_n+1j*jy_im_p

def aexp(x,y,z,u,v,w):
    a0=np.sqrt(u**2+ v**2)
    a1=np.sqrt(a0**2 + 1j*w*mur1*mu0*sigma1)
    num=np.exp(a1*z)*np.exp(1j*u*x)*np.exp(1j*v*y)
    dem=a0*(a0*mur1 + a1)
    return num/dem


    
def J(x1,x2):
    y= (x1*(special.jv(0,x1)*special.struve(1, x1)-special.jv(1,x1)*special.struve(0, x1))
    -x2*(special.jv(0,x2)*special.struve(1, x2)-special.jv(1,x2)*special.struve(0, x2)))/(np.pi/2)
    return y



xv=np.linspace(-10e-3,10e-3,20)
yv=np.linspace(-10e-3,10e-3,20)



Dx=np.zeros([20,20],dtype='complex')
Dy=np.zeros([20,20],dtype='complex')


for i,x in enumerate(xv):
    for j,y in enumerate(yv):
        Dx[i,j]=Jx(x,y,-1e-3,coil,100)
        Dy[i,j]=Jy(x,y,-1e-3,coil,100)
        


## plots

plt.quiver(xv,yv,Dx,Dy)