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

import plotly.figure_factory as ff

class bpp():
    def __init__(self,xo=1e-3, yo=2e-3 ,zo=2e-3, c=2e-3, zc=4.5e-3, N=400):
        self.xo,self.yo,self.zo,self.c,self.zc,self.N=xo,yo,zo,c,zc,N  
                  
    #propiedades de la probeta
    mur=1
    sigma=35.4e6
    mu0=4*np.pi*10e-7
    I=30e-3
    z=0

    def w(self,f):
        self.w=2*np.pi*f
        return

    def jxjy(self):
        
        xo,yo,zo,c,zc,N=self.xo,self.yo,self.zo,self.c,self.zc,self.N
        mur,sigma,mu0,I = self.mur,self.sigma,self.mu0,self.I
        w=self.w    
        z=self.z
        av=self.av
        bv=self.bv

        def k(a,b): 
            return np.sqrt(a**2+b**2)
        def l(a,b) :
            return np.sqrt(a**2+b**2 + 1j*w*mur*mu0*sigma)
        def fl(a,b) :
            return (2*k(a,b)*mur)/(k(a,b)*mur+l(a,b))
        def Ds(a,b) :
            return (1j*mu0*N*I*A(a,b)*sin(a*xo)*exp(-k(a,b)*zc))/(pi*pi*c*2*xo*b*(k(a,b))**3)
        def A(a,b): 
            return (-b*cos(b*(yo+c))*sinh(k(a,b)*(zo+c)) +  b*cos(b*yo)*sinh(k(a,b)*zo) +   k(a,b)*sin(b*(yo+c))*cosh(k(a,b)*(zo+c)) - k(a,b)*sin(b*yo)*cosh(k(a,b)*zo)      )/(a**2+2*(b**2))
            
        def fxab(a,b):    
            return 2*self.w*sigma*Ds(a,b)*fl(a,b)*exp(l(a,b)*z)*b
        def fyab(a,b):
            return -2*self.w*sigma*Ds(a,b)*fl(a,b)*exp(l(a,b)*z)*a

        ifx = fxab(av[:,None], bv[None,:])
        ify = fyab(av[:,None], bv[None,:])
        Jx,Jy=fftshift(ifft2(ifftshift(np.nan_to_num(ifx.T)))).real,fftshift(ifft2(ifftshift(np.nan_to_num(ify.T)))).real

        self.jx=Jx
        self.jy=Jy
        return


    def quiver(self,lim=1,titulo=''):
        
        Jx,Jy=self.jx,self.jy

        self.getx2()
        xv,yv=self.xv,self.yv



        plt.figure(figsize=[10,10])

        mask=np.abs(xv/1e-3)< lim
        plt.quiver(xv[mask]/1e-3,yv[mask]/1e-3,Jx[mask][:,mask],Jy[mask][:,mask])

        plt.title(titulo)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.grid(True)
        return

    def stream(self,lim=1,titulo=''):
        
        Jx,Jy=self.jx,self.jy

        self.getx2()
        xv,yv=self.xv,self.yv



        plt.figure(figsize=[10,10])

        mask=np.abs(xv/1e-3)< lim
        plt.streamplot(xv[mask]/1e-3,yv[mask]/1e-3,Jx[mask][:,mask],Jy[mask][:,mask])

        plt.title(titulo)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.grid(True)
        return

    def stream2(self,lim=1,titulo=''):
        
        Jx,Jy=self.jx,self.jy

        self.getx2()
        xv,yv=self.xv,self.yv



        fig=plt.figure(figsize=[10,10])
        lim=1
        mask=np.abs(xv/1e-3)< lim
        ax0 = fig.add_subplot()
        ax0.streamplot(xv[mask]/1e-3,yv[mask]/1e-3,Jx[mask][:,mask],Jy[mask][:,mask])
        ax0.grid(True)
        return fig   
        
    def getxv(self,xm):
        self.xv=np.linspace(-xm,xm,200)
        return

    def pquiver(self,lim=1):      
        Jx,Jy=self.jx,self.jy
        self.getx2()
        xv,yv=self.xv,self.yv
        mask=np.abs(xv/1e-3)< lim
        x,y = np.meshgrid(xv[mask]/1e-3,yv[mask]/1e-3)
        fig = ff.create_quiver(x, y,Jx[mask][:,mask],Jy[mask][:,mask])
        fig.show()
        return fig

    def getk(self,t):
        N=len(t)
        dt=t[1]-t[0]
        k = fftfreq(N,d=dt)
        self.av=fftshift(k[1:])
        self.bv=fftshift(k[1:])
        return 

    def getk2(self,t):
        N=len(t)
        dt=t[1]-t[0]
        k = fftfreq(N,d=dt)
        self.av=fftshift(k)
        self.bv=fftshift(k)
        return

    def getx2(self):
        t=self.av
        N=len(t)
        dt=t[1]-t[0]
        k = fftfreq(N,d=dt)
        self.xv=fftshift(k)
        self.yv=fftshift(k)
        return