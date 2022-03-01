# implementacion del paper theo2005
# anlytical model for tilted coils


from scipy import integrate
from scipy import special
import numpy as np
import math
import matplotlib.pyplot as plt
# Examples

# Compute the double integral of x * y**2 over the box x ranging from 
# 0 to 2 and y ranging from 0 to 1.

# from scipy import integrate
# f = lambda y, x: x*y**2
# integrate.dblquad(f, 0, 2, lambda x: 0, lambda x: 1)
#     (0.6666666666666667, 7.401486830834377e-15)



frec=100
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
tita=math.radians(90)
x0=0
y0=0
coil=[r1,r2,l,n,tita,x0,y0,l0]
i0=1e-3
d=r2
dlim=500
lmin=1




## ec. 4
def Jx(x,y,z,coil,frec,lim=dlim):
    w=2*np.pi*frec
    cj=w*sigma1*mu0*mur1*i0/np.pi
    fx_re= lambda v, u: np.real(hs(u,v,coil)*v*aexp(x,y,z,u,v,w))
    fx_im=  lambda v, u: np.imag(hs(u,v,coil)*v*aexp(x,y,z,u,v,w))
    # real -inf,+inf
    jx_re_n=cj*integrate.dblquad(fx_re,-lim, -lmin, lambda u: -lim, lambda u: -lmin)[0]
    jx_re_p=cj*integrate.dblquad(fx_re,lmin, lim, lambda u: lmin, lambda u: lim)[0]
    #imag -inf,+inf
    jx_im_n=cj*integrate.dblquad(fx_im,-lim, -lmin, lambda u: -lim, lambda u: -lmin)[0]
    jx_im_p=cj*integrate.dblquad(fx_im,lmin, lim, lambda u: lmin, lambda u: lim)[0]
    return jx_re_n+jx_re_p+1j*jx_im_n+1j*jx_im_p


def Jy(x,y,z,coil,frec,lim=dlim):
    w=2*np.pi*frec
    cj=w*sigma1*mu0*mur1*i0/np.pi
    fy_re= lambda v, u: -np.real(hs(u,v,coil)*u*aexp(x,y,z,u,v,w))
    fy_im=  lambda v, u: -np.imag(hs(u,v,coil)*u*aexp(x,y,z,u,v,w))
    jy_re_n=cj*integrate.dblquad(fy_re,-lim, -lmin, lambda u: -lim, lambda u: -lmin)[0]
    jy_re_p=cj*integrate.dblquad(fy_re,lmin, lim, lambda u: lmin, lambda u: lim)[0]
    jy_im_n=cj*integrate.dblquad(fy_im,-lim, -lmin, lambda u: -lim, lambda u: -lmin)[0]
    jy_im_p=cj*integrate.dblquad(fy_im,lmin, lim, lambda u: lmin, lambda u: lim)[0]
    return jy_re_n+jy_re_p+1j*jy_im_n+1j*jy_im_p

def aexp(x,y,z,u,v,w):
    a0=np.sqrt(u**2+ v**2)
    a1=np.sqrt(a0**2 + 1j*w*mur1*mu0*sigma1)
    num=np.exp(a1*z)*np.exp(1j*u*x)*np.exp(1j*v*y)
    dem=a0*(a0*mur1 + a1)
    return num/dem

## ec 10
def hs(u,v,coil):
    '''8'''
    r1=coil[0]
    r2=coil[1]
    l=coil[2]
    n=coil[3]
    a=np.sqrt(u**2+ v**2)
    h=1j*n*np.exp(-a*d)*np.sin(u*l/2)*M(u*r1,u*r2)/(u**3)
    return(h)
    
def M(x1,x2):
    y= (x1*(special.iv(0,x1)*special.modstruve(1, x1)-special.iv(1,x1)*special.modstruve(0, x1))
    -x2*(special.iv(0,x2)*special.modstruve(1, x2)-special.iv(1,x2)*special.modstruve(0, x2)))/(np.pi/2)
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