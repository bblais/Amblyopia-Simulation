#!/usr/bin/env python
# coding: utf-8

# In[1]:


#| output: false
get_ipython().run_line_magic('matplotlib', 'inline')
from input_environment_defs import *


# In[2]:


import pylab as plt


# ## response of ganglion cells to gratings
# 
# ![image.png](attachment:1a686632-7b39-4ee6-a77b-7aef784b5a14.png)
# 
# ![image.png](attachment:0fd51a5f-751c-4e67-b8cb-a7331578692e.png)

# ## reproduce the photoreceptor data first
# 
# ![Screenshot 2023-01-23 at 08.30.58.png](attachment:9e06ed0d-7010-41b7-a462-546c729ad053.png)
# 
# 
# $$
# R=\gamma \frac{I^n}{\sigma^n+I_m^n+I^n}
# $$
# 
# with $n=1$

# In[3]:





# In[10]:





# In[14]:


I=plt.rand(500)*100
Im=I.mean()
R=I/(Im+I)
R=R/R.max()*100

subplot(2,1,1)
plt.hist(I,rwidth=0.95,edgecolor='k');

subplot(2,1,2)
plt.hist(R,rwidth=0.95,edgecolor='k');


# In[19]:


I=plt.randn(500)*100
I=I-I.min()
Im=I.mean()
R=I/(Im+I)
R=R/R.max()*100

subplot(2,1,1)
plt.hist(I,rwidth=0.95,edgecolor='k');

subplot(2,1,2)
plt.hist(R,rwidth=0.95,edgecolor='k');


# In[25]:


def rande(N):
    from pylab import rand,log,zeros
    y=2.0*rand(N)-1.0
    r=zeros(N)
    r[y<0.0]=log(-y[y<0.0])
    r[y>0.0]=-log(y[y>0.0])

    return r


# In[30]:


I=rande(5000)*100
I=I-I.min()
Im=I.mean()
R=I/(Im+I)
R=R/R.max()*100

subplot(2,1,1)
plt.hist(I,50,rwidth=0.95,edgecolor='k');

subplot(2,1,2)
plt.hist(R,50,rwidth=0.95,edgecolor='k');

I2=(I-I.mean())/I.std()
I2=I2*R.std()+R.mean()
plt.hist(I2,50,rwidth=0.85,edgecolor='k',alpha=0.7);



# I=rande(5000)*10
# I=I-I.min()
# Im=I.mean()
# R=I/(Im+I)
# R=R/R.max()*100

# subplot(2,1,1)
# plt.hist(I,50,rwidth=0.9,edgecolor='k');

# subplot(2,1,2)
# plt.hist(R,50,rwidth=0.9,edgecolor='k');

# I2=(I-I.mean())/I.std()
# I2=I2*R.std()+R.mean()
# plt.hist(I2,50,rwidth=0.85,edgecolor='k',alpha=0.7);


# about the same distribution, just insensitive to the scale and mean level

# In[ ]:




