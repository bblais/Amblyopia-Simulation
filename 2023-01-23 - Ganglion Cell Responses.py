#!/usr/bin/env python
# coding: utf-8

# In[1]:


#| output: false
get_ipython().run_line_magic('matplotlib', 'inline')
from input_environment_defs import *


# In[2]:


def grid(val=True):
    plt.rcParams['axes.grid']=val
    
from Memory import Storage


# In[3]:


#| output: false

# Make the original image files
make_original_image_files()


# In[4]:


#| label: fig-orig
#| fig-cap: A Small Subset of the Original Natural Images
fname='asdf/bbsk081604_all.asdf'
image_data=pi5.asdf_load_images(fname)
im=[arr.astype(float)*image_data['im_scale_shift'][0]+
        image_data['im_scale_shift'][1] for arr in image_data['im']]
del image_data
plt.figure(figsize=(16,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(im[i],cmap=plt.cm.gray)
    plt.axis('off')
orig_im=im


# In[5]:


im=orig_im[5]
var={'im':[im],'im_scale_shift':[1.0,0.0]}

figure(figsize=(10,5))

subplot(2,3,1)
imshow(im,cmap=plt.cm.gray)
axis('off')
title(im.shape)

subplot(2,3,2)
var_dog=filters.make_dog(var)

im2=var_dog['im'][0]
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)


subplot(2,3,3)
var_blur=pi5.make_blur(var_dog,5)
im3=var_blur['im'][0]
imshow(im3,cmap=plt.cm.gray)
axis('off');
title(im3.shape)


var_blur1=var_blur



subplot(2,3,4)
imshow(im,cmap=plt.cm.gray)
axis('off')
title(im.shape)

subplot(2,3,5)
var_blur=pi5.make_blur(var,5)
im3=var_blur['im'][0]
imshow(im3,cmap=plt.cm.gray)
axis('off');
title(im3.shape)



subplot(2,3,6)
var_dog=filters.make_dog(var_blur)
im2=var_dog['im'][0]
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)

var_blur2=var_dog


# In[7]:


im1=var_blur1['im'][0]
im2=var_blur2['im'][0]

idx=np.s_[10:100,10:100]

figure(figsize=(10,5))
imd=im1-im2

for i,imm in enumerate([im1,im2,imd]):
    subplot(1,3,i+1)
    imshow(imm[idx])
    plt.colorbar()




# In[8]:



def dog(sd1,sd2,size):
    import numpy 
    v1=numpy.floor((size-1.0)/2.0)
    v2=size-1-v1
    
    y,x=numpy.mgrid[-v1:v2,-v1:v2]
    
    pi=numpy.pi
    
    # raise divide by zero error if sd1=0 and sd2=0
    
    if sd1>0:
        g=1./(2*pi*sd1*sd1)*numpy.exp(-x**2/2/sd1**2 -y**2/2/sd1**2)
        if sd2>0:
            g=g- 1./(2*pi*sd2*sd2)*numpy.exp(-x**2/2/sd2**2 -y**2/2/sd2**2)
    else:
        g=- 1./(2*pi*sd2*sd2)*numpy.exp(-x**2/2/sd2**2 -y**2/2/sd2**2)
    
    return g

def dog_filter(A,sd1=1,sd2=3,size=None,shape='valid',surround_weight=1):
    import scipy.signal as sig

    if not size:
        size=2.0*numpy.ceil(2.0*max([sd1,sd2]))+1.0
        
    if sd1==0 and sd2==0:
        B=copy.copy(A)
        return B
    
    g=dog(sd1,sd2,size)

    B=sig.convolve2d(A,g,mode=shape)
    
    return B



# In[9]:


figure()
im2=im
im2=dog_filter(im2,1,3,32,'valid')
im2=dog_filter(im2,5,0,5*3,'same')
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)

im3=im2


figure()
im2=im
im2=dog_filter(im2,5,0,5*3,'same')
im2=dog_filter(im2,1,3,32,'valid')
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)

im4=im2


# In[10]:


idx=np.s_[10:100,10:100]

figure(figsize=(10,5))
imd=im3-im4
for i,imm in enumerate([im3,im4,imd]):
    subplot(1,3,i+1)
    imshow(imm[idx])
    plt.colorbar()


# In[11]:


im2=im
x1=im2/im2.mean()-1
imshow(x1)
plt.colorbar()


# In[12]:


im2=im
x2=(im2-im2.mean())/im2.std()
imshow(x2)
plt.colorbar()


# In[13]:


imshow(x1/x2)
plt.colorbar()


# In[14]:


x1/x2


# ## response of ganglion cells to gratings
# 
# ![image.png](attachment:1a686632-7b39-4ee6-a77b-7aef784b5a14.png)
# 
# ![image.png](attachment:0fd51a5f-751c-4e67-b8cb-a7331578692e.png)

# In[15]:


background=100
C=25

λ=5
N=50
R=5
y,x=np.mgrid[:N,:N]


grid(False)
im2=C*np.sin(x/λ*2*np.pi)*((x-N/2)**2+(y-N/2)**2<R**2)+background
figure(figsize=(5,3))
imshow(im2,vmin=0,vmax=200,cmap=plt.cm.gray)
plt.colorbar()

im2=dog_filter(im2,1,3,32,'valid')
im2-=im2.mean()
im2/=im2.std()
figure(figsize=(5,3))
plt.grid(False)
imshow(im2,cmap=plt.cm.gray)
plt.colorbar()


# In[16]:


for R in tqdm(np.linspace(1,50,5)):
    S=Storage()
    for C in tqdm(np.linspace(0,100,20)):
        background=100
        #C=25

        λ=5
        N=50
        R=5
        y,x=np.mgrid[:N,:N]


        im2=C*np.sin(x/λ*2*np.pi)*((x-N/2)**2+(y-N/2)**2<R**2)+background

        im2=dog_filter(im2,1,3,32,'valid')
        im2-=im2.mean()
        im2/=im2.std()

        S+=C,im2.max()


    C,resp=S.arrays()

    plot(C,resp,'-o')



# In[17]:


R


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


def RR(I,γ,σ,Im):
    n=1
    return γ*I**n/(σ**n + Im**n + I**n)
    
    


# In[4]:


xy_str="""
-5.666666666666666, 3.6414565826330545
-5.0606060606060606, 7.843137254901961
-4.636363636363637, 14.005602240896366
-4.05050505050505, 36.134453781512605
-3.7676767676767673, 50.420168067226896
-3.242424242424242, 76.19047619047619
-2.6363636363636367, 88.51540616246498
-2.1111111111111107, 92.99719887955182
-4.656565656565656, 1.400560224089645
-4.07070707070707, 10.644257703081237
-3.7272727272727266, 18.767507002801125
-3.242424242424242, 36.414565826330545
-2.6363636363636367, 69.74789915966387
-2.1313131313131306, 81.23249299719888
-1.1010101010101012, 90.75630252100841
-3.7474747474747474, -5.3221288515406115
-3.2020202020202024, 7.002801120448183
-2.6565656565656566, 26.890756302521012
-2.1515151515151514, 46.77871148459384
-1.6060606060606064, 75.63025210084034
-1.121212121212122, 85.15406162464987
-0.5555555555555554, 90.75630252100841
-0.07070707070707094, 94.11764705882354
-3.7272727272727266, -9.523809523809504
-3.2020202020202024, -7.282913165266109
-2.6767676767676765, -1.1204481792717047
-2.1313131313131306, 11.20448179271709
-1.5858585858585856, 27.450980392156865
-1.0808080808080813, 48.73949579831933
-0.595959595959596, 71.70868347338936
-0.05050505050505194, 86.5546218487395
"""


# In[5]:


x=10**(np.array([[float(__) for __ in _.split(',')] for _ in xy_str.strip().split('\n')])[:,0])
y=np.array([[float(__) for __ in _.split(',')] for _ in xy_str.strip().split('\n')])[:,1]

x=[x[:8],x[8:15],x[15:23],x[23:]]
y=[y[:8],y[8:15],y[15:23],y[23:]]


# In[6]:


x


# In[7]:


from scipy.optimize import curve_fit

γ_mat,Im_mat=[],[]

for xx,yy in zip(x,y):
    plot(xx,yy,'o')
    popt, pcov = curve_fit(RR, xx, yy,
                                 bounds=([0,0,0], [100, 1e-14, 1]))
    
    γ_mat.append(popt[0])
    Im_mat.append(popt[-1])
    xx=10**(np.linspace(-6,0,40))
    yy=RR(xx,*popt)
    plot(xx,yy)
    print(popt)

    
plt.xscale('log')


# In[24]:


from scipy.optimize import curve_fit

γ_mat,Im_mat=[],[]

for xx,yy in zip(x,y):
    plot(xx,yy,'o')
    popt, pcov = curve_fit(RR, xx, yy,
                                 bounds=([0,0,0], [100, 1e-14, 1]))
    
    γ_mat.append(popt[0])
    Im_mat.append(popt[-1])
    xx=10**(np.linspace(-6,0,40))
    yy=RR(xx,*popt)
    plot(xx,yy)
    print(popt)

    


# the linear approximation looks pretty terrible

# In[8]:


for xx,yy,γ,Im in zip(x,y,γ_mat,Im_mat):
    plot(xx,yy,'o')
    xx=10**(np.linspace(-6,0,40))
    yy=RR(xx,*[γ,0,Im])
    plot(xx,yy)

    p=np.log10(Im)
    xx=10**(np.linspace(p-1,p+1,40))
    C=(xx-Im)/Im
    plot(xx,γ/2+γ/4*C)
    
plt.xscale('log')


# how does it compare to normalization?
# 
# see notebook 2023-01-30 - Exploring Photoreceptor Responses

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


# apply to natural images
im2=im
x2=(im2-im2.mean())/im2
imshow(x2)
plt.colorbar()


# In[51]:


im2=im
im2=(im2-im2.mean())/im2
figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)


im2=dog_filter(im2,1,3,32,'valid')
im2=(im2-im2.mean())/im2.std()


figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


# In[52]:


im2=im
#im2=(im2-im2.mean())/im2
figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)


im2=dog_filter(im2,1,3,32,'valid')
im2=(im2-im2.mean())/im2.std()

figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


# In[59]:


im2=im
im2=(im2-im2.mean())/im2
figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()
im2a=im2

im2=dog_filter(im2a,1,3,32,'valid')
im22=dog_filter(im2a**2,4,1,32,'valid')

im2=im2/(10+np.sqrt(im22))

figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


# photoreceptor subtract mean across a smallish area rather than the entire image

# In[83]:


im=orig_im[0]
import scipy.signal as sig
im2=im

if True: # local average

    N=33
    im2_average=sig.convolve2d(im2,np.ones((N,N))/N**2,mode='valid')
    ra,ca=im2_average.shape
    r,c=im2.shape
    im2=im2[N//2:N//2+ra,N//2:N//2+ca]

    im2=(im2-im2_average)/im2
    
else:
    
    im2=(im2-im2.mean())/im2

im2=im
    
figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


im2a=im2

im2=dog_filter(im2a,1,3,32,'valid')
im2=(im2-im2.mean())/im2.std()


figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


# In[77]:


im2.max()


# In[70]:


N//2


# Why is the convolution with the normalized image different than the normalized convolution of the original image?

# In[85]:


im=orig_im[5]
im2=im

im2=(im2-im2.mean())/im2
figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


im2a=im2

im2=dog_filter(im2a,1,3,32,'valid')
im2=(im2-im2.mean())/im2.std()


figure(figsize=(5,3))
imshow(im2,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


# convolution scales no problem

# In[87]:


im2=im

im2a=dog_filter(im2,1,3,32,'valid')
im2=im2/5
im2b=dog_filter(im2,1,3,32,'valid')


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=im2a/im2b
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


# convolution of DOG with a constant is zero

# In[91]:


im2=im

im2a=dog_filter(im2,1,3,32,'valid')
im2=im2+5
im2b=dog_filter(im2,1,3,32,'valid')


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=im2a/im2b
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()


# In[93]:


im2d.ravel()[:20]


# In[98]:


im2=im

mn=im2.mean()
sd=im2.std()
print(mn,sd)

im2a=dog_filter(im2,1,3,32,'valid')
im2=(im2-mn)/sd
im2b=dog_filter(im2,1,3,32,'valid')


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=im2a/im2b
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()

im2d.ravel()[:20]


# In[125]:


im2=im

mn=im2.mean()
print(mn)

im2a=dog_filter(im2,1,3,32,'valid')
im2a/=im2a.max()

im2=im2/(mn+im2)
im2b=dog_filter(im2,1,3,32,'valid')
im2b/=im2b.max()


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=(im2a-im2b)
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()

im2d.ravel()[:20]


# In[124]:


plt.hist(im2a.ravel(),200);
plt.hist(im2b.ravel(),200,alpha=0.5);


# try the average and the std over a large window, not the entire imagescipy.ndimage.generic_filter. 
# 
# hardly any difference

# In[128]:


from scipy.ndimage import generic_filter


# In[140]:


def mean(x):
    return x.mean()

def std(x):
    return x.std()


# In[146]:


im2=im
N=33

im2=im2/(im2.mean()+im2)
im2a=dog_filter(im2,1,3,32,'valid')
im2a=im2a-im2a.mean()
im2a=im2a/im2a.std()

im2=im
Im=generic_filter(im2,mean,(N,N))
im2=im2/(Im+im2)
im2b=dog_filter(im2,1,3,32,'valid')
im2b=im2b/generic_filter(im2b,std,(N,N))
im2b=im2b-im2b.mean()

# im2b=im2b/im2b.std()


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=(im2a-im2b)
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()

im2d.ravel()[:20]


# In[156]:


im2=im

im2=im2/(im2.mean()+im2)
im2a=dog_filter(im2,1,3,32,'valid')
im2a=im2a-im2a.mean()
im2a=im2a/im2a.std()

im2=im
N=33
Im=generic_filter(im2,mean,(N,N))
im2=im2/(Im+im2)
im2b=dog_filter(im2,1,3,32,'valid')
im2b=im2b/(1+generic_filter(im2b,std,(N,N)))
im2b=im2b-im2b.mean()
im2b=im2b/im2b.std()  # replace the γ


# im2b=im2b/im2b.std()


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=(im2a-im2b)
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()

im2d.ravel()[:20]


# In[153]:


plt.hist(im2a.ravel(),200);
plt.hist(im2b.ravel(),200,alpha=0.5);


# In[151]:


im2b.std()


# ## blur and order

# approximate form with global mean and std

# In[165]:


im2=im
im2=dog_filter(im2,4,0,32,'valid')

im2=im2/(im2.mean()+im2)
im2a=dog_filter(im2,1,3,32,'valid')
im2a=im2a-im2a.mean()
im2a=im2a/im2a.std()

im2=im

im2=im2/(im2.mean()+im2)
im2b=dog_filter(im2,1,3,32,'valid')
im2b=im2b-im2b.mean()
im2b=im2b/im2b.std()

im2b=dog_filter(im2b,4,0,32,'valid')
im2b=im2b-im2b.mean()
im2b=im2b/im2b.std()



# im2b=im2b/im2b.std()


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=(im2a-im2b)
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()

im2d.ravel()[:20]


# In[ ]:





# In[157]:


im2=im
im2=dog_filter(im2,4,0,32,'valid')

im2=im2/(im2.mean()+im2)
im2a=dog_filter(im2,1,3,32,'valid')
im2a=im2a-im2a.mean()
im2a=im2a/im2a.std()

im2=im
im2=dog_filter(im,4,0,32,'valid')

N=33
Im=generic_filter(im2,mean,(N,N))
im2=im2/(Im+im2)
im2b=dog_filter(im2,1,3,32,'valid')
im2b=im2b/(1+generic_filter(im2b,std,(N,N)))
im2b=im2b-im2b.mean()
im2b=im2b/im2b.std()  # replace the γ


# im2b=im2b/im2b.std()


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=(im2a-im2b)
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()

im2d.ravel()[:20]


# In[160]:


im2=im

im2=im2/(im2.mean()+im2)
im2a=dog_filter(im2,1,3,32,'valid')
im2a=dog_filter(im2a,4,0,32,'valid')
im2a=im2a-im2a.mean()
im2a=im2a/im2a.std()

im2=im

N=33
Im=generic_filter(im2,mean,(N,N))
im2=im2/(Im+im2)
im2b=dog_filter(im2,1,3,32,'valid')
im2b=im2b/(1+generic_filter(im2b,std,(N,N)))

im2b=dog_filter(im2b,4,0,32,'valid')

im2b=im2b-im2b.mean()
im2b=im2b/im2b.std()  # replace the γ


# im2b=im2b/im2b.std()


figure(figsize=(15,3))
subplot(1,3,1)
imshow(im2a,cmap=plt.cm.gray)
axis('off');
title(im2a.shape)
plt.colorbar()

subplot(1,3,2)
imshow(im2b,cmap=plt.cm.gray)
axis('off');
title(im2b.shape)
plt.colorbar()


im2d=(im2a-im2b)
subplot(1,3,3)
imshow(im2d,cmap=plt.cm.gray)
axis('off');
title(im2.shape)
plt.colorbar()

im2d.ravel()[:20]


# In[ ]:





# ![rf.freq_-1024x771.png](attachment:c9c9eab5-7bde-43a4-903b-8112dca0e805.png)

# In[33]:


R=100
S=Storage()
for λ in np.linspace(1,10,100):
    background=100
    C=50

    #λ=5
    N=100
    R=5
    y,x=np.mgrid[:N,:N]


    im2=C*np.sin(x/λ*2*np.pi)*((x-N/2)**2+(y-N/2)**2<R**2)+background

    im2=dog_filter(im2,1,3,32,'valid')
    im2-=im2.mean()
    im2/=im2.std()

    S+=λ,im2.max()


λ,resp=S.arrays()

plot(λ,resp,'-o')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'curve_fit')


# In[ ]:





# In[ ]:




