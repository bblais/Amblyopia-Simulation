#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[2]:


from science import *


# In[13]:


exponent=linspace(-4,2,500)
x=10**exponent


# ![image.png](attachment:a0b188df-c4fb-44fd-94eb-a4c2c37d5213.png)

# In[14]:


def u(x,kc,rc,ks,rs):
    return kc*rc*sqrt(pi)*exp(-(x/rc)**2)-ks*rs*sqrt(pi)*exp(-(x/rs)**2)


# In[17]:


plot(x,u(x,
         kc=6,
         rc=1,
         ks=.1,
         rs=18))
xscale('log')
yscale('log')


# In[ ]:




