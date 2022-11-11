#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[2]:


from science import *


# ![image.png](attachment:190016cf-1241-45d3-a8b5-4a744d5418ad.png)

# In[3]:


def λ(x): ## unit gaussian
    return exp(-pi*x**2)

def Λ(u): ## fourier transform is the same form
    return exp(-pi*u**2)

def λa(x,a=1): ## scaled unit gaussian
    return 1/abs(a)*λ(x/a)

def Λa(u,a=1): ## fourier transform is the same form
    return Λ(a*u)

# 2D

def λ2a(x,y,a=1): ## scaled unit gaussian
    return λa(x,a)*λa(y,a)

def Λ2a(ux,uy,a=1): ## fourier transform is the same form
    return Λa(ux,a)*Λa(uy,a)


# In[4]:


x=linspace(-5,5,100)
dx=x[1]-x[0]

y=λa(x,1)
plot(x,y)
print(sum(y*dx))

y=λa(x,2)
plot(x,y)
print(sum(y*dx))


# In[5]:


x,y=meshgrid(linspace(-5,5,100),linspace(-6,6,120))


# In[8]:


z=λ2a(x,y,4)
grid(False)
pcolor(x,y,z)
grid(True)
axis('equal');
colorbar()


# In[17]:


dx=x[0,1]-x[0,0]
dy=y[1,0]-y[0,0]
sum(z*dx*dy)


# In[16]:





# In[23]:


v=13.66
s=0.025
rs=4.98
rv=0.65

x,y=meshgrid(linspace(-.05,.05,100),linspace(-.06,.06,120))

z1=v*λ2a(x,y,s)
z2=-v*rv*λ2a(x,y,s*rs)
z=z1+z2

subplot(2,2,1)
pcolor(x,y,z1)
axis('equal');
colorbar()    

subplot(2,2,2)
pcolor(x,y,z2)
axis('equal');
colorbar()   

subplot(2,1,2)
pcolor(x,y,z)
axis('equal');
colorbar()    


# In[30]:


v=13.66
s=0.025
rs=4.98
rv=0.65

x,y=meshgrid(linspace(-5,5,100),linspace(-6,6,120))
z=v*(Λ2a(x,y,s) - rv*Λ2a(x,y,rs))

grid(False)
pcolor(x,y,z)
grid(True)
axis('equal');
colorbar()    


# In[33]:


x=linspace(-2,2,100)
y=zeros(len(x))
z=v*(Λ2a(x,y,s) - rv*Λ2a(x,y,rs))

plot(x,log(z))


# ![image.png](attachment:c992ec2a-b390-43f2-93d6-68ef99c2b61c.png)

# ![image.png](attachment:d8234243-de66-493b-97f8-875204d09bf3.png)

# ![image.png](attachment:30ac77c8-d821-4f47-afcc-8b8787ad3a9c.png)

# In[29]:


f=linspace(.1,40,200)
kc=15.03
rc=0.015
ksrs2_kcrc2=0.89
rs=0.072

ks=ksrs2_kcrc2*kc*rc**2/rs**2

Cf=kc*pi*rc**2*exp(-(pi*f*rc)**2)
Sf=ks*pi*rs**2*exp(-(pi*f*rc)**2*ksrs2_kcrc2)
Sf=0


# In[30]:


F=Cf-Sf
loglog(f,F)


# In[33]:


Axy=10**array([
[ -0.4371681415929203 , 0.25981055480378845 ],
[ -0.25132743362831866 , 0.34100135317997254 ],
[ -0.12920353982300892 , 0.25981055480378845 ],
[ 0.038053097345132514 , 0.4438430311231392 ],
[ 0.1707964601769909 , 0.43301759133964807 ],
[ 0.3486725663716812 , 0.649526387009472 ],
[ 0.4681415929203536 , 0.8335588633288226 ],
[ 0.6486725663716812 , 0.990527740189445 ],
[ 0.7707964601769908 , 1.047361299052774 ],
[ 0.9433628318584066 , 1.1312584573748308 ],
[ 1.0707964601769913 , 1.017591339648173 ],
[ 1.2646017699115037 , 0.8362652232746952 ],
[ 1.3840707964601768 , 0.6062246278755072 ],
])


# In[37]:


x=Axy[:,0]
y=Axy[:,1]
plot(x,y,'o')

f=linspace(.1,40,200)
kc=15.03
rc=0.015
ksrs2_kcrc2=0.89
rs=0.072

ks=ksrs2_kcrc2*kc*rc**2/rs**2

Cf=kc*pi*rc**2*exp(-(pi*f*rc)**2)
Sf=ks*pi*rs**2*exp(-(pi*f*rc)**2*ksrs2_kcrc2)

plot(f,Cf-Sf)


# In[42]:


x=Axy[:,0]
y=Axy[:,1]
plot(x,y,'o')

f=linspace(.1,40,200)
kc=15.03
rc=0.015
ksrs2_kcrc2=0.89
rs=0.072

C=15
S=12

ks=ksrs2_kcrc2*kc*rc**2/rs**2

Cf=C*exp(-(pi*f*rc)**2)
Sf=S*exp(-(pi*f*rc)**2*ksrs2_kcrc2)

plot(f,Cf-Sf)


# In[74]:


x=linspace(-30,30,1000)
y=λa(x,4)-λa(x,8)
plot(x,y)


# In[75]:


def myfft(x,y):
    import numpy as np
    yf=np.fft.fft(y)
    
    dx=x[1]-x[0]
    N=len(yf)
    
    k=np.fft.fftfreq(N, d=dx)
    
    yf=np.fft.fftshift(yf)  # unscramble the fft
    k=np.fft.fftshift(k)    # unscramble the frequencies

    return k,yf


# In[76]:


k,yf=myfft(x,y)


# In[77]:


plot(k,abs(yf))
xlim([-1,1])


# In[110]:


x=linspace(-30,30,2000)
a=4
y=λa(x,a)

dx=x[1]-x[0]
print(sum(y*dx))



subplot(2,1,1)
plot(x,y)


k,yf=myfft(x,y)
subplot(2,1,2)

plot(k,abs(yf)*dx)
dk=k[1]-k[0]
print(sum(abs(yf)*dk))


yf=Λa(k,a)
N=len(yf)
plot(k,abs(yf),'r--')
xlim([-1,1])


# In[119]:


x=linspace(-30,30,2000)
y=λa(x,4)-.8*λa(x,6)

dx=x[1]-x[0]
print(sum(y*dx))

subplot(2,1,1)
plot(x,y)


k,yf=myfft(x,y)
subplot(2,1,2)

plot(k,abs(yf)*dx)
dk=k[1]-k[0]
print(sum(abs(yf)*dk))


yf=Λa(k,4)-.8*Λa(k,6)
N=len(yf)
plot(k,abs(yf),'r--') 
xlim([-1,1])

figure(figsize=(10,4))
loglog(k,abs(yf),'r--')


# ![image.png](attachment:a711765e-2714-475c-a51e-e68fe3cfdf80.png)
# 
# ![image.png](attachment:626e9dfd-2132-413c-9de5-bafa79c77710.png)

# In[139]:


# average from watson
v=13.66
s=0.025
rs=4.98
rv=0.65

# cell A from table page 225 Derrington and Lennie
kc=15.03
rc=0.015
ksrs2_kcrc2=0.89
rs=0.072
ks=ksrs2_kcrc2*kc*rc**2/rs**2

# convert this to watson numbers
v=kc
rv=ks/v
s=rc
rs=ks/s


x=linspace(-20,20,5000)
y=v*(λa(x,s)-rv*λa(x,rs*s))

dx=x[1]-x[0]
print(sum(y*dx))

subplot(2,1,1)
plot(x,y)


k,yf=myfft(x,y)

yf=yf[k>0]
k=k[k>0]

subplot(2,1,2)

plot(k,abs(yf)*dx)
dk=k[1]-k[0]
print(sum(abs(yf)*dk))

xd=Axy[:,0]
yd=Axy[:,1]

plot(xd,yd,'ko')


yf=v*(Λa(k,s)-rv*Λa(k,rs*s))
N=len(yf)
plot(k,abs(yf),'r--') 

figure(figsize=(10,4))

yf=yf[k>0]
k=k[k>0]

plot(log(k),log(abs(yf)),'r--')
ylim([0,3])



# In[ ]:




