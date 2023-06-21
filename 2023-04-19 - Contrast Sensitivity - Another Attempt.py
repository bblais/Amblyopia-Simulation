#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from science import *


# ![image.png](attachment:cebe54b6-9a9d-4e69-a565-82a7bda61c69.png)

# In[62]:


def S(ν,rc=1,kc=1,rs=1,ks=1):
    return kc*rc**2*pi*exp(-(pi*rc*ν)**2) - ks*rs**2*pi*exp(-(pi*rs*ν)**2)


# In[63]:


data="""
2.1173461092214785, 2.612404112378969
1.648900957267581, 5.437834790854295
1.1937766417144369, 18.200766836091212
0.8376776400682924, 43.77796859597804
0.612816311822054, 52.178163493270844
0.4390701741343199, 43.77796859597804
0.2986160333041584, 39.48321930173606
0.20736816348639417, 26.945935975134923
0.1501310728908175, 20.815442622705
0.10534754864987898, 13.91546610623254
0.07869145977415551, 10.861115683522995
0.05464577141043569, 10.000000000000005
0.03874675120456138, 8.218606859349654
0.008642742694895159, 3.113676884923481
2.1173461092214785, 2.3079622420963393
1.648900957267581, 4.804125562572591
1.1814033283238543, 12.945165982558997
0.8289952386887821, 26.124041123789663
0.612816311822054, 37.885701776588846
0.4390701741343199, 31.459936923255878
0.308097222245024, 26.945935975134923
0.21173461092214793, 21.693162522934387
0.15170345614893657, 17.46435160688786
0.10425563732507492, 15.113748716993474
0.0557964200174266, 8.92632398567921
0.03956262211548858, 6.895488244573798
0.008553161980897835, 3.4881961375352915
"""


arr=array([[float(_) for _ in line.split(",")] for line in data.strip().split("\n")])
arr

N=14
x_data1=arr[:N,0]
y_data1=arr[:N,1]

x_data2=arr[N:,0]
y_data2=arr[N:,1]

x_data=arr[:,0]
y_data=arr[:,1]

plot(x_data1,y_data1,'o')
plot(x_data2,y_data2,'o')
xscale('log')
yscale('log')


# ![image.png](attachment:cad5bdbc-e96a-4412-a496-b6f55b795291.png)

# In[71]:


ν=linspace(.01,3,500)
rc=.24
rs=.96
kc=350
ks=0.96*kc*rc**2/rs**2
CS=S(ν,rc=rc,rs=rs,kc=kc,ks=ks)

plot(x_data1,y_data1,'o')
plot(x_data2,y_data2,'o')
plot(ν,CS)

xscale('log')
yscale('log')

ylim([1,100])

title("as close as I can make it...but not great")


# In[41]:


from scipy.optimize import curve_fit


# In[42]:


p0=(rc,kc,rs,ks)

popt, pcov =curve_fit(S, x_data, y_data, p0=p0)
popt, pcov 


# In[44]:


rc,kc,rs,ks=popt
plot(x_data1,y_data1,'o')
plot(x_data2,y_data2,'o')
plot(ν,CS)

xscale('log')
yscale('log')


# In[ ]:





# In[ ]:





# ![image.png](attachment:5c069c92-e2b7-46b5-a977-3153111160f5.png)

# In[ ]:




