#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[ ]:


import process_images_hdf5 as pi5
import filters


# In[ ]:


from skimage import io, color
from skimage.transform import rescale,resize
import glob
import os
from tqdm.notebook import tqdm


# ## Make the scale2 image set

# In[ ]:


files=glob.glob('/Users/bblais/python/work/natural_images/images/original/all_bbsk081604/*.jpg')
files.sort()
print( "Image files: %d" % len(files))


# In[ ]:


sz=1536, 2048

sz[0]/1536*600,sz[1]/1536*600


# In[ ]:


for i,fname in (pbar := tqdm(enumerate(files),total=len(files))):
    rgb = io.imread(fname)
    if i<10:
        print(fname,rgb.shape)
    else:
        print(".",end="")
        
print()



# In[ ]:


len(files)


# In[ ]:


import filters
import process_images_hdf5 as pi5


# In[ ]:


asdf_fname='asdf/bbsk081604_all_scale2.asdf'
max_pic=1000
overwrite=True

if overwrite or not os.path.exists(asdf_fname):

    print(asdf_fname)
    
    im=[]
    for i,fname in (pbar := tqdm(enumerate(files),total=len(files))):
        if i<max_pic:
            rgb = io.imread(fname)
            lab = color.rgb2lab(rgb)
            lab=resize(lab,(600,800), anti_aliasing=True)  # turns out not all of the bbsk images are the same size!
            #lab=rescale(lab, 1/2, anti_aliasing=True)
            L=lab[:,:,0]  # luminance
            im.append(L.astype(float))
        
    var_R={'im':im,'im_scale_shift':[1.0,0.0]}
    filters.set_resolution(var_R,'uint16')
    pi5.asdf_save_images(var_R,asdf_fname) 


# In[ ]:


asdf_fname='asdf/bbsk081604_all_scale1.asdf'
max_pic=1000
if not os.path.exists(asdf_fname):

    print(asdf_fname)
    
    im=[]
    for i,fname in (pbar := tqdm(enumerate(files),total=len(files))):
        if i<max_pic:
            rgb = io.imread(fname)
            lab = color.rgb2lab(rgb)
            lab=resize(lab,(1200,1600), anti_aliasing=True)  # turns out not all of the bbsk images are the same size!
            #lab=rescale(lab, 1/2, anti_aliasing=True)
            L=lab[:,:,0]  # luminance
            im.append(L.astype(float))
        
    var_R={'im':im,'im_scale_shift':[1.0,0.0]}
    filters.set_resolution(var_R,'uint16')
    pi5.asdf_save_images(var_R,asdf_fname) 


# In[ ]:


base_image_file=asdf_fname='asdf/bbsk081604_all_scale2.asdf'
print("Base Image File:",base_image_file)


imfname=pi5.filtered_images(
                            base_image_file,
                            )


# In[ ]:


image_data=pi5.asdf_load_images(imfname)
im=[arr.astype(float)*image_data['im_scale_shift'][0]+
        image_data['im_scale_shift'][1] for arr in image_data['im']]

del image_data
figure(figsize=(16,8))
for i in range(12):
    subplot(3,4,i+1)
    imshow(im[i],cmap=plt.cm.gray)
    axis('off')
    
suptitle(imfname)


# In[ ]:


len(im)


# In[ ]:


im[0].shape


# In[ ]:


base_image_file=asdf_fname
print("Base Image File:",base_image_file)


imfname=pi5.filtered_images(
                            base_image_file,
                            {'type':'dog','sd1pro1,'sd2':3},
                            )

image_data=pi5.asdf_load_images(imfname)
im=[arr.astype(float)*image_data['im_scale_shift'][0]+
        image_data['im_scale_shift'][1] for arr in image_data['im']]

del image_data
figure(figsize=(16,8))
for i in range(12):
    subplot(3,4,i+1)
    imshow(im[i],cmap=plt.cm.gray)
    axis('off')


# ## questions
# 
# - is blur + dog same as dog+blur?

# In[ ]:




