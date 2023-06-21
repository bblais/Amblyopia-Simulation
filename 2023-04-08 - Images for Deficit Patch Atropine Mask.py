#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[2]:


from treatment_sims_2023_02_21 import *


# In[17]:


def save_image(im,fname,plot=True):
    image_data=pi5.asdf_load_images(im)
    im1=image_data['im'][5]*image_data['im_scale_shift'][0]+image_data['im_scale_shift'][1]
    if plot:
        imshow(im1,cmap=plt.cm.gray)
        axis('off')
    imsave(fname,im1,cmap=plt.cm.gray)    
    
    return im1


# In[12]:


base_image_file='asdf/bbsk081604_all.asdf'

im=pi5.filtered_images(
                    base_image_file,
                    {'type':'dog','sd1':1,'sd2':3},
                    {'type':'norm'},
                    )

save_image(im,'/Users/bblais/Downloads/im.png')


# In[18]:


base_image_file='asdf/bbsk081604_all.asdf'

bv=10


im=pi5.filtered_images(
                        base_image_file,
                        {'type':'blur','size':bv},
                        )

figure()
save_image(im,'/Users/bblais/Downloads/im_blur0.png')


im=pi5.filtered_images(
                        base_image_file,
                        {'type':'blur','size':bv},
                        {'type':'dog','sd1':1,'sd2':3},
                        {'type':'norm'},
                        )
figure()
im_blur1=save_image(im,'/Users/bblais/Downloads/im_blur1.png')


# In[19]:


im=pi5.filtered_images(
                        base_image_file,
                        {'type':'dog','sd1':1,'sd2':3},
                        {'type':'norm'},
                        {'type':'blur','size':bv},
                        )
figure()
im_blur2=save_image(im,'/Users/bblais/Downloads/im_blur2.png')


# In[28]:


subplot(2,1,1)
grid(False)
imshow(im_blur2,cmap=plt.cm.gray)
title('old')
axis('off')
colorbar(orientation='horizontal')

subplot(2,1,2)
grid(False)
imshow(im_blur1,cmap=plt.cm.gray)
title('new')
axis('off')
colorbar(orientation='horizontal')


# In[29]:


im1.shape


# In[36]:


shape=[int(_*.2) for _ in im1.shape]


# In[37]:


imsave('/Users/bblais/Downloads/noise.png',randn(*shape),cmap=plt.cm.gray)


# In[41]:


f=10
maskA_fname=pi5.filtered_images(base_image_file,
                            {'type':'mask',
                             'name':'bblais-masks-20210615/2021-06-15-*-A-fsig%d.png' % f,
                            'seed':101},
                            {'type':'dog','sd1':1,'sd2':3},
                            {'type':'norm'},                                            
                                verbose=False,
                          )
maskF_fname=pi5.filtered_images(base_image_file,
                            {'type':'mask',
                             'name':'bblais-masks-20210615/2021-06-15-*-F-fsig%d.png' % f,
                            'seed':101},
                            {'type':'dog','sd1':1,'sd2':3},
                            {'type':'norm'},                                            
                                verbose=False,
                          )

figure()
ima=save_image(maskA_fname,'/Users/bblais/Downloads/im_maskA_dog.png')
figure()
imf=save_image(maskF_fname,'/Users/bblais/Downloads/im_maskF_dog.png')



f=10
maskA_fname=pi5.filtered_images(base_image_file,
                            {'type':'mask',
                             'name':'bblais-masks-20210615/2021-06-15-*-A-fsig%d.png' % f,
                            'seed':101},
                                verbose=False,
                          )
maskF_fname=pi5.filtered_images(base_image_file,
                            {'type':'mask',
                             'name':'bblais-masks-20210615/2021-06-15-*-F-fsig%d.png' % f,
                            'seed':101},
                                verbose=False,
                          )

figure()
save_image(maskA_fname,'/Users/bblais/Downloads/im_maskA.png')
figure()
save_image(maskF_fname,'/Users/bblais/Downloads/im_maskF.png')


# In[45]:


imsave('/Users/bblais/Downloads/im_maskF_dog_contrast.png',imf,cmap=plt.cm.gray,
      vmin=-30,vmax=30)


# In[42]:


imf.min()


# In[43]:


imf.max()


# In[44]:


get_ipython().run_line_magic('pinfo', 'imsave')


# In[ ]:




