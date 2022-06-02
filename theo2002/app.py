import streamlit as st
import numpy as np
from pp import bpp

col1, col2 , col3 = st.columns(3)

with col2:
    x0 = st.slider('X0 =', 1, 10, 2)
    y0 = st.slider('y0 =', 1, 10, 2)
    f = st.slider('f [kHz] =', 1, 1000, 10)
    z = st.slider('z =', 0, 10, 0)


with col1:
    st.image('https://github.com/carabedo/labs/raw/master/img/2.png')   
    
    
    
xm=6e-3
xv=np.linspace(-xm,xm,200)
b=bpp()
b.getk(xv)
b.w(f*1e3)
b.z=z*1e-3
b.yo=y0*1e-3
b.xo=x0*1e-3
b.jxjy()

with col3:

    fig=b.stream2()
    st.pyplot(fig)







