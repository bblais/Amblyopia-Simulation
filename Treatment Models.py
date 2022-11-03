#!/usr/bin/env python
# coding: utf-8

# In[1]:


#| output: false

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import plasticnet as pn
import process_images_hdf5 as pi5
from deficit_defs import patch_treatment
from input_environment_defs import get_input_patch_examples
from matplotlib.pyplot import figure,xlabel,ylabel,legend,gca,plot,subplot,imshow,axis
from tqdm.notebook import tqdm

def savefig(base):
    import matplotlib.pyplot as plt
    for fname in [f'Manuscript/resources/{base}.png',f'Manuscript/resources/{base}.svg']:
        print(fname)
        plt.savefig(fname)


# In[2]:


for i in tqdm(range(100)):
    pass


# ## Models of Treatments for Amblyopia {#sec-models-of-treatments}
# 
# To model the fix to the refractive imbalance we follow the deficit simulation with an input environment that is rebalanced, both eyes receiving nearly identical input patches (@fig-normal-inputs).   This process is a model of the application of refractive correction.  Although both eyes receive nearly identical input patches, we add independent Gaussian noise to each input channel to represent the natural variation in the activity in each eye.  In addition, in those cases where use employ strabismic amblyopia, the inter-eye jitter is not corrected with the refractive correction.  
# 

# In[2]:


#| output: false

def inputs_to_images1(X,buffer=5):
    ims=[]
    
    rf_size=int(np.sqrt(X.shape[1]/2))
    vmax=np.concatenate([xx[:rf_size*rf_size] for xx in X]).max()
    
    for xx in X:
        xx1=xx[:rf_size*rf_size].reshape(rf_size,rf_size)
        im=xx1
        ims.append(im)
        
    return ims

def inputs_to_images2(X,buffer=5):
    ims=[]
    
    rf_size=int(np.sqrt(X.shape[1]/2))
    vmax=np.concatenate([xx[rf_size*rf_size:] for xx in X]).max()
    
    for xx in X:
        xx2=xx[rf_size*rf_size:].reshape(rf_size,rf_size)
        im=xx2
        ims.append(im)
        
    return ims

def inputs_to_images(X,buffer=5):
    ims=[]
    vmin=X.min()
    vmax=X.max()
    
    rf_size=int(np.sqrt(X.shape[1]/2))
        
    for xx in X:
        xx1=xx[:rf_size*rf_size].reshape(rf_size,rf_size)
        xx2=xx[rf_size*rf_size:].reshape(rf_size,rf_size)
        im=np.concatenate((xx1,np.ones((rf_size,buffer))*vmax,xx2),axis=1)   
        ims.append(im)
        
    return ims

def get_input_patch_examples_treatment():
    
    seq=pn.Sequence()    
    seq+=patch_treatment(patch_noise=0.5,
               total_time=1000,number_of_neurons=1,
               eta=1e-6,
               save_interval=1)
    sim=seq.sims[0]
    pre=seq[0][1][0]
    sim.monitor(pre,['output'],1)

    seq.run(display_hash=False,print_time=False)
    m=sim.monitors['output_1']
    t,X=m.arrays()    
    
    return seq,X


# 
# 
# ## Patch treatment
# 
# The typical patch treatment is done by depriving the strong-eye of input with an eye-patch.  In the model this is equivalent to presenting the strong-eye with random noise instead of the natural image input.  Competition between the left- and right-channels drives the recovery, and is produced from the difference between *structured* input into the weak-eye and the *unstructured* (i.e. noise) input into the strong eye.  It is not driven by a reduction in input activity.  @fig-patch-inputs shows sample simulation input patterns from the patched eye.  Compare this to @fig-normal-inputs to see that the simulated patch has far less structure than the normal inputs.
# 

# In[3]:


#| echo: false
#| label: fig-patch-inputs
#| fig-cap: A sample of 24 input patches from a patched visual environment. 
#| 
sim,X=get_input_patch_examples_treatment()
ims=inputs_to_images2(X)
figure(figsize=(10,6))
for i in range(24):
    im=ims[i]
    subplot(4,6,i+1)
    imshow(im,cmap=plt.cm.gray)
    axis('off')
    
plt.savefig('Manuscript/resources/fig-patch-inputs.png')
plt.savefig('Manuscript/resources/fig-patch-inputs.svg')        
    


# 
# 
# 
# ## Contrast modification
# 
# A binocular approach to treatment can be produced with contrast reduction of the non-deprived channel relative to the deprived channel. Experimentally this can be accomplished with VR headsets[@xiao2020improved]. In the model we implement this by down-scaling the normal, unblurred channel with a simple scalar multiplier applied to each pixel (Figure [4](#fig:input) D). The contrast difference sets up competition between the two channels with the advantage given to the weak-eye channel.
# 

# In[4]:


#| label: fig-contrast-modified-inputs
#| fig-cap: A sample of 24 input patches from a normal visual environment with the right-channel down-scaled relative to the left.
#| 
sim,X=get_input_patch_examples(blur=-1,contrast=0.3)
ims=inputs_to_images(X,buffer=2)
figure(figsize=(20,6))
for i in range(24):
    im=ims[i]
    subplot(4,6,i+1)
    imshow(im,cmap=plt.cm.gray)
    axis('off')
    
plt.savefig('Manuscript/resources/fig-contrast-modified-inputs.png')
plt.savefig('Manuscript/resources/fig-contrast-modified-inputs.svg')        
    


# ## Dichoptic Mask
# 
# On top of the contrast modification, we can include the application of the dichoptic mask (Figure @fig:input E).  In this method, each eye receives a version of the input images filtered through independent masks in each channel, resulting in a mostly-independent pattern in each channel.  
# It has been observed that contrast modification combined with dichoptic masks can be an effective treatment for amblyopia[@Li:2015aa,@xiao2021randomized].  The motivation behind the application of the mask filter is that the neural system must use both channels to reconstruct the full image and thus may lead to enhanced recovery.  
# 
# The dichoptic masks are constructed with the following procedure.  A blank image (i.e. all zeros) is made to which is added 15 randomly sized circles with values equal to 1 (Figure @fig:dichopic_blob).   These images are then smoothed with a Gaussian filter of a given width, $f$.  This width is a parameter we can vary to change the overlap between the left- and right-eye images.  A high value of $f$ compared with the size of the receptive field, e.g. $f=90$, yields a high overlap between the patterns in the weak- and strong-eye inputs (Figure @fig:dichopic_filter_size).  Likewise, a small value of $f$, e.g. $f=10$, the eye inputs are nearly independent -- the patterned activity falling mostly on one of the eyes and not much to both.  Finally, the smoothed images are scaled to have values from a minimum of 0 to a maximum of 1.  This image-mask we will call $A$, and is the left-eye mask whereas the right-eye mask, $F$, is the inverse of the left-eye mask, $F\equiv 1-A$.  The mask is applied to an image by multiplying the left- and right-eye images by the left- and right-eye masks, respectively, resulting in a pair of images which have no overlap at the peaks of each mask, and nearly equal overlap in the areas of the images where the masks are near 0.5 (Figure @fig:dichopic_filter_image).   
# 

# In[3]:


from PIL import Image
from glob import glob
from random import choice
from numpy import meshgrid,arange


# In[4]:


def gaussian2d(x,y,x0,y0,sigma):
    from numpy import exp
    r2=(x-x0)**2+(y-y0)**2
    return exp(-r2/2/sigma**2)

def circle(x,y,x0,y0,r):
    r2=(x-x0)**2+(y-y0)**2
    return r2<r**2


from scipy.signal import convolve2d
from numpy.random import rand

def randbetween(low,high):
    from numpy.random import rand
    return rand()*(high-low)+low

def deg2pixel(D):
    # the bbsk images are angular size 60 degrees x 40 degrees
    # the raw images are resized so that 
    #   5.5 pixels ~ 0.5 degrees (cat retina)  - default

    P=D/.5*5.5
    return P

def pixel2deg(P):
    D=P/5.5*0.5
    return D


# In[5]:


def make_mask(fsig=35,g=None):
    # logged dog is 495 x 660 
    # the bbsk images are angular size 60 degrees x 40 degrees
    # the raw images are resized so that 
    #   5.5 pixels ~ 0.5 degrees (cat retina)  - default
    
    mx,my=660,495
    x,y=meshgrid(arange(mx),arange(my))
    
    if g is None:
        g=0


        blob_deg_min=3
        blob_deg_max=14
        for i in range(10):
            g=g+circle(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))

        #    g=g+gaussian2d(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))>.5
        g=g>0


    f=gaussian2d(x,y,mx//2,my//2,fsig)
    f=f[(my//2-200):(my//2+200),(mx//2-200):(mx//2+200)]

    #f=f[(my//2-3*fsig):(my//2+3*fsig),(mx//2-3*fsig):(mx//2+3*fsig)]
    
    res=convolve2d(g,f,mode='same')
    res=res/res.max()
    
    return res


# In[22]:


mx,my=660,495
x,y=meshgrid(arange(mx),arange(my))
g=0

blob_deg_min=3
blob_deg_max=14
for i in range(10):
    g=g+circle(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))
    
#    g=g+gaussian2d(x,y,rand()*mx,rand()*my,randbetween(deg2pixel(blob_deg_min),deg2pixel(blob_deg_max)))>.5
g=g>0
imshow(g,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
savefig('fig-dichopic_blob')


# In[23]:


fsig=20 # pixels
f=gaussian2d(x,y,mx//2,my//2,fsig)
#f=f[(my//2-3*fsig):(my//2+3*fsig),(mx//2-3*fsig):(mx//2+3*fsig)]
f=f[(my//2-200):(my//2+200),(mx//2-200):(mx//2+200)]

plt.grid(False)
imshow(f,extent=[0,pixel2deg(f.shape[1]),0,pixel2deg(f.shape[0])])
plt.colorbar()


# In[31]:


get_ipython().run_cell_magic('time', '', "res=convolve2d(g,f,mode='same')\nres=res/res.max()")


# In[32]:


plt.imshow(res,extent=[0,pixel2deg(res.shape[1]),0,pixel2deg(res.shape[0])])
plt.colorbar()


# In[33]:


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


plt.text(.35, 0.88, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.1, 0.5, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')
plt.text(.5, 0.5, "C", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('blob_convolution_example_fsig_%d' % fsig)


# In[34]:


res=res[:495,:660]
figure(figsize=(12,8))
subplot(1,2,1)

AA=np.uint8(plt.cm.viridis(res)*255)
AA[:,:,3]=res*255
im=Image.fromarray(AA)  
imshow(im)
plt.title(AA.shape)

subplot(1,2,2)

FF=np.uint8(plt.cm.viridis(1-res)*255)
FF[:,:,3]=(1-res)*255
im=Image.fromarray(FF)  
imshow(im)
plt.title(FF.shape)


# In[35]:


fname='asdf/bbsk081604_all_log2dog.asdf'
image_data=pi5.asdf_load_images(fname)
im1=image_data['im'][5]*image_data['im_scale_shift'][0]+image_data['im_scale_shift'][1]
imshow(im1,cmap=plt.cm.gray)


# In[36]:


figure(figsize=(12,8))
subplot(1,2,1)

A=AA
r,c=0,0
alpha_A=A[(0+r):(im1.shape[0]+r),(0+c):(im1.shape[1]+c),3]/255
im2=im1*alpha_A
imshow(im2,cmap=plt.cm.gray)


subplot(1,2,2)

F=FF
r,c=0,0
alpha_F=F[(0+r):(im1.shape[0]+r),(0+c):(im1.shape[1]+c),3]/255
im2=im1*alpha_F
imshow(im2,cmap=plt.cm.gray)


# In[37]:


figure(figsize=(15,12))
subplot(2,1,1)
imshow(res,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
ylabel('degrees')
ax=plt.colorbar()
rmin,rmax=res.min(),res.max()
ax.set_ticks([rmin,(rmin+rmax)/2,rmax])
ax.set_ticklabels(['Right','Both','Left'])

subplot(2,2,3)
im2=im1*alpha_A
imshow(im2,cmap=plt.cm.gray,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
xlabel('degrees')
ylabel('degrees')
plt.title('Left')

subplot(2,2,4)
im2=im1*alpha_F
imshow(im2,cmap=plt.cm.gray,extent=[0,pixel2deg(mx),0,pixel2deg(my)])
xlabel('degrees')
ylabel('degrees')
plt.title('Right')

plt.text(.33, 0.9, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.1, 0.5, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')
plt.text(.5, 0.5, "C", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('mask_filter_example_fsig_%d' % fsig)


# In[38]:


masks={}
for f,fsig in tqdm(enumerate([10,30,50,70,90,110])):
    res=make_mask(fsig,g)
    masks[fsig]=res


# In[40]:


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
savefig('mask_filter_examples_fsigs')


# from input_environment_defs import *
# 
# ## Atropine treatment
# 
# In the atropine treatment for amblyopia[@glaser2002randomized], eye-drops of atropine are applied to the strong-eye resulting in blurred vision in that eye.  Here we use the same blurred filter used to obtain the deficit (possibly with a different width) applied to the strong eye (Figure @fig:input F).  The difference in sharpness between the strong-eye inputs and the weak-eye inputs sets up competition between the two channels with the advantage given to the weak-eye.
# 
# 

# In[3]:


from input_environment_defs import *


# In[4]:


#| label: fig-atropine-inputs
#| fig-cap: A sample of 24 input patches from an  environment with atropine applied to the strong eye. The amblyopic (blurred) input is the square on the left-hand side of each pair.
sim,X=get_input_patch_examples(blur=2.5,blurred_eye='right')
ims=inputs_to_images(X,buffer=2)
figure(figsize=(20,6))
for i in range(24):
    im=ims[i]
    subplot(4,6,i+1)
    imshow(im,cmap=plt.cm.gray)
    axis('off')
    
savefig('fig-atropine-inputs')

    


# In[ ]:




