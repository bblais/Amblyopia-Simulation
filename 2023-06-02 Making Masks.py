#!/usr/bin/env python
# coding: utf-8

# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[42]:


from PIL import Image
from glob import glob
from random import choice


# In[43]:


def gaussian2d(x,y,x0,y0,sigma):
    r2=(x-x0)**2+(y-y0)**2
    return exp(-r2/2/sigma**2)

def circle(x,y,x0,y0,r):
    r2=(x-x0)**2+(y-y0)**2
    return r2<r**2


from scipy.signal import convolve2d

def randbetween(low,high):
    return rand()*(high-low)+low


# In[44]:


def deg2pixel(D):
    # bbsk081604_all_scale2.asdf size if 600x800
    # each pixel is 0.25 deg from properties of LGN cells
    
    P=D/.25
    return P

def pixel2deg(P):
    D=P*0.25
    return D


# In[45]:


def make_mask(fsig=35,g=None):
    # bbsk081604_all_scale2.asdf size if 600x800
    # each pixel is 0.25 deg from properties of LGN cells
    
    mx,my=800,600
    x,y=meshgrid(arange(mx),arange(my))
    
    if g is None:
        g=0


        # to match 2021-06-15\ Making\ Masks.ipynb the pixels are (33.0, 154.0)
        blob_deg_min=pixel2deg(33)
        blob_deg_max=pixel2deg(154)
        # blob_deg_min=3
        # blob_deg_max=14
        for i in range(15):
            g=g+circle(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))

        #    g=g+gaussian2d(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))>.5
        g=g>0


    f=gaussian2d(x,y,mx//2,my//2,fsig)
    f=f[(my//2-200):(my//2+200),(mx//2-200):(mx//2+200)]

    #f=f[(my//2-3*fsig):(my//2+3*fsig),(mx//2-3*fsig):(mx//2+3*fsig)]
    
    res=convolve2d(g,f,mode='same')
    res=res/res.max()
    
    return res


# In[46]:


import process_images_hdf5 as pi5


# In[47]:


base_image_file=fname='asdf/bbsk081604_all_scale2.asdf'
print("Base Image File:",base_image_file)

image_data=pi5.asdf_load_images(fname)

maskA_fname=pi5.filtered_images(base_image_file,
                            {'type':'mask',
                             'name':'bblais-masks-20230602/2023-06-02-*-A-fsig%d.png'% 10, 
                            'seed':101},
                            {'type':'blur','size':4},
                            {'type':'dog','sd1':1,'sd2':3},
                            {'type':'norm'},
                                verbose=True,
                          )


# In[48]:


image_data['im'][0].shape


# In[49]:


for im in image_data['im']:
    print(im.shape,end=" ")


# In[50]:


base_image_file


# In[51]:



#mx,my=1200,700
mx,my=800,600
x,y=meshgrid(arange(mx),arange(my))
g=0

# to match 2021-06-15\ Making\ Masks.ipynb the pixels are (33.0, 154.0)
blob_deg_min=pixel2deg(33)
blob_deg_max=pixel2deg(154)
for i in range(15):
    g=g+circle(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))
    
#    g=g+gaussian2d(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))>.5
g=g>0

deg2pixel(blob_deg_min),deg2pixel(blob_deg_max),blob_deg_min,blob_deg_max


# In[52]:


imshow(g,extent=[0,pixel2deg(mx),0,pixel2deg(my)])


# In[53]:


fsig=20 # pixels
f=gaussian2d(x,y,mx//2,my//2,fsig)
#f=f[(my//2-3*fsig):(my//2+3*fsig),(mx//2-3*fsig):(mx//2+3*fsig)]
f=f[(my//2-200):(my//2+200),(mx//2-200):(mx//2+200)]


# In[54]:


imshow(f,extent=[0,pixel2deg(f.shape[1]),0,pixel2deg(f.shape[0])])
colorbar()


# In[19]:


get_ipython().run_cell_magic('time', '', "res=convolve2d(g,f,mode='same')\nres=res/res.max()")


# In[20]:


pixel2deg(90)


# In[21]:


imshow(res,extent=[0,pixel2deg(res.shape[1]),0,pixel2deg(res.shape[0])])
colorbar()


# In[23]:


res=res[:600,:800]
figure(figsize=(12,8))
subplot(1,2,1)

AA=np.uint8(cm.viridis(res)*255)
AA[:,:,3]=res*255
im=Image.fromarray(AA)  
imshow(im)
title(AA.shape)

subplot(1,2,2)

FF=np.uint8(cm.viridis(1-res)*255)
FF[:,:,3]=(1-res)*255
im=Image.fromarray(FF)  
imshow(im)
title(FF.shape)


# In[24]:


import process_images_hdf5 as pi5


# In[34]:


base_image_file='asdf/bbsk081604_all_scale2.asdf'
print("Base Image File:",base_image_file)


imfname=pi5.filtered_images(
                            base_image_file,
                            )

image_data=pi5.asdf_load_images(imfname)


# In[35]:


im1=image_data['im'][5]*image_data['im_scale_shift'][0]+image_data['im_scale_shift'][1]
imshow(im1,cmap=cm.gray)


# In[39]:


figure()

A=AA
r,c=0,0
alpha_A=A[(0+r):(im1.shape[0]+r),(0+c):(im1.shape[1]+c),3]/255
im2=im1*alpha_A
imshow(im2,cmap=cm.gray)
axis('off')

figure()
F=FF
r,c=0,0
alpha_F=F[(0+r):(im1.shape[0]+r),(0+c):(im1.shape[1]+c),3]/255
im2=im1*alpha_F
imshow(im2,cmap=cm.gray)
axis('off')

figure(figsize=(12,8))
imshow(res,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
ylabel('degrees')
ax=colorbar()
rmin,rmax=res.min(),res.max()
ax.set_ticks([rmin,(rmin+rmax)/2,rmax])
ax.set_ticklabels(['Right','Both','Left'])


# In[28]:


from bigfonts import *
from IPython.display import HTML


# In[55]:


figure(figsize=(15,12))
subplot(2,3,2)
imshow(f,extent=[0,pixel2deg(f.shape[1]),0,pixel2deg(f.shape[0])])
axis('off')
subplot(2,2,3)
imshow(g,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
xlabel('degrees')
ylabel('degrees')

subplot(2,2,4)

imshow(res,extent=[0,pixel2deg(res.shape[1]),0,pixel2deg(res.shape[0])])
xlabel('degrees')
ylabel('degrees')

savefig('blob_convolution_example_fsig_%d.pdf' % fsig)


# In[32]:


figure(figsize=(15,12))
subplot(2,1,1)
imshow(res,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
ylabel('degrees')
ax=colorbar()
rmin,rmax=res.min(),res.max()
ax.set_ticks([rmin,(rmin+rmax)/2,rmax])
ax.set_ticklabels(['Right','Both','Left'])

subplot(2,2,3)
im2=im1*alpha_A
imshow(im2,cmap=cm.gray,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
xlabel('degrees')
ylabel('degrees')
title('Left')

subplot(2,2,4)
im2=im1*alpha_F
imshow(im2,cmap=cm.gray,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
xlabel('degrees')
ylabel('degrees')
title('Right')

savefig('mask_filter_example_fsig_%d.pdf' % fsig)


# In[31]:


masks={}
for f,fsig in enumerate([10,30,50,70,90,110]):
    res=make_mask(fsig,g)
    masks[fsig]=res


# In[ ]:


fig, axs = plt.subplots(3, 2,figsize=(18,15))

for f,fsig in enumerate([10,30,50,70,90,110]):
    ax = axs.ravel()[f]

    res=masks[fsig]
    im=ax.imshow(res,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
    ax.set_ylabel('degrees')
    ax.set_title(r"$\sigma_f=%g$" % fsig)
    
    if fsig in [10,30,50,70]:
        ax.set_xticklabels([])
    
h=fig.colorbar(im, ax=axs)
rmin,rmax=res.min(),res.max()
h.set_ticks([rmin,(rmin+rmax)/2,rmax])
h.set_ticklabels(['Right','Both','Left'])
savefig('mask_filter_examples_fsigs.pdf')


# In[ ]:


raise ValueError


# In[ ]:


gs=[]
for k in range(5):
    g=0


    blob_deg_min=pixel2deg(33)
    blob_deg_max=pixel2deg(154)
    # blob_deg_min=3
    # blob_deg_max=14
    percentage_covered=0
    while percentage_covered<50:
        g=g+circle(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))
        g=g>0
        percentage_covered=g.sum()/prod(g.shape)*100
    #    g=g+gaussian2d(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))>.5

    gs.append(g)
    
    subplot(1,5,k+1)
    imshow(g,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
    xlabel('degrees')
    ylabel('degrees')


# In[ ]:


g.sum()/prod(g.shape)*100


# In[ ]:


count=1
for f,fsig in enumerate([10,30,50,70,90]):
    for k in range(5):
        res=make_mask(fsig,gs[k])

        AA=np.uint8(cm.viridis(res)*255)
        AA[:,:,3]=res*255

        FF=np.uint8(cm.viridis(1-res)*255)
        FF[:,:,3]=(1-res)*255

        im=Image.fromarray(AA)    
        fname='bblais-masks-20230602/2023-06-02-%c-A-fsig%2d.png' % (k+65,fsig)
        im.save(fname)
        print(fname)

        figure(1)
        subplot(5,5,count)
        imshow(im)

        im=Image.fromarray(FF)    
        fname='bblais-masks-20230602/2023-06-02-%c-F-fsig%2d.png' % (k+65,fsig)
        im.save(fname)
        print(fname)

        figure(2)
        subplot(5,5,count)
        imshow(im)
    
        count+=1


# In[ ]:




