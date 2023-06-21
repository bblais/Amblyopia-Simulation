#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from PIL import Image
from tqdm import tqdm

import plasticnet as pn
import process_images_hdf5 as pi5
import filters

from plasticnet import day,ms,minute,hour,second


from matplotlib.pyplot import figure,xlabel,ylabel,legend,gca,plot,subplot,imshow,axis,title


# In[1]:


#| output: false

def inputs_to_images(X,buffer=5,scale_each_patch=False):
    ims=[]
    vmin=X.min()
    vmax=X.max()
    
    rf_size=int(np.sqrt(X.shape[1]/2))
    
    for xx in X:
        xx1=xx[:rf_size*rf_size].reshape(rf_size,rf_size)
        xx2=xx[rf_size*rf_size:].reshape(rf_size,rf_size)
        if scale_each_patch:
            vmax=max([xx1.max(),xx2.max()])
            vmin=max([xx1.min(),xx2.min()])
        
        im=np.concatenate((xx1,np.ones((rf_size,buffer))*vmax,xx2),axis=1)   
        ims.append(im)
        
    return ims

def get_input_examples(im1,im2,
                       noise1=0.1,noise2=0.1,
                       scale1=1,scale2=1,
                     mu_c=0,sigma_c=0,    
                     mu_r=0,sigma_r=0,    
                     base_image_file='asdf/bbsk081604_all_scale2.asdf'
                ):
    
    rf_size=19
    eta=2e-6
    pre1=pn.neurons.natural_images_with_jitter(im1,
                                   rf_size=rf_size,
                                    sigma_r=1,
                                   sigma_c=1,
                                   verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(im2,
                                               rf_size=rf_size,
                                other_channel=pre1,
                               mu_r=mu_r,mu_c=mu_c,
                               sigma_r=sigma_r,sigma_c=sigma_c,
                                verbose=False)

    pre1+=pn.neurons.process.scale_shift(scale1,0)
    pre2+=pn.neurons.process.scale_shift(scale2,0)

    pre1+=pn.neurons.process.add_noise_normal(0,noise1)
    pre2+=pn.neurons.process.add_noise_normal(0,noise2)

    pre=pre1+pre2

    sim=pn.simulation(99)
    sim.monitor(pre,['output'],1)
    sim.monitor(pre1,['pattern','p','c','r','pa','ca','ra'],1)
    sim.monitor(pre2,['pattern','p','c','r','pa','ca','ra'],1)

    pn.run_sim(sim,[pre],[],display_hash=False,print_time=False)

    m=sim.monitors['output']
    
    t,X=m.arrays()
    
    
    X=X[1:,:]

    
    
    return sim,X
    

def get_input_patch_examples_with_jitter(blur=2.5,noise=0.1,
                                         mu_c=10,sigma_c=2,    
                                           mu_r=0,sigma_r=1,
                                         base_image_file='asdf/bbsk081604_all_log2dog.asdf'
                                        ):
    
    rf_size=19
    eta=2e-6

    

    number_of_neurons=1,

    if blur<0:
        blur_fname=Lnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    verbose=False)
    else:
        blur_fname=Lnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'blur','size':blur},
                                    {'type':'norm'},
                                    verbose=False)

    Rnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    verbose=False)

    pre1=pn.neurons.natural_images_with_jitter(Lnorm_fname,
                                   rf_size=rf_size,
                                    sigma_r=1,
                                   sigma_c=1,
                                   verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(Rnorm_fname,rf_size=rf_size,
                                other_channel=pre1,
                               mu_r=mu_r,mu_c=mu_c,
                               sigma_r=1,sigma_c=sigma_c,
                                verbose=False)

    sigma=noise
    pre1+=pn.neurons.process.add_noise_normal(0,sigma)

    sigma=noise
    pre2+=pn.neurons.process.add_noise_normal(0,sigma)

    pre=pre1+pre2

    sim=pn.simulation(99)
    sim.monitor(pre,['output'],1)
    sim.monitor(pre1,['pattern','p','c','r','pa','ca','ra'],1)
    sim.monitor(pre2,['pattern','p','c','r','pa','ca','ra'],1)

    pn.run_sim(sim,[pre],[],display_hash=False,print_time=False)

    m=sim.monitors['output']
    
    t,X=m.arrays()
    
    
    X=X[1:,:]

    
    
    return sim,X
    


def get_input_patch_examples(blur=2.5,noise=0.1,contrast=1,blurred_eye='left',
                            base_image_file='asdf/bbsk081604_all_log2dog.asdf'):
    
    rf_size=19
    eta=2e-6

    

    number_of_neurons=1,

    
    if blur<0:
        Lnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    verbose=False)
        Rnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    verbose=False)
    elif blurred_eye=='left':
        Lnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'blur','size':blur},
                                    {'type':'norm'},
                                    verbose=False)
        Rnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    verbose=False)
    elif blurred_eye=='right':
        Lnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    verbose=False)
        Rnorm_fname=pi5.filtered_images(base_image_file,
                                    {'type':'blur','size':blur},
                                    {'type':'norm'},
                                    verbose=False)

    else:
        raise ValurError("You can't get there from here.")
        
        

    pre1=pn.neurons.natural_images(Lnorm_fname,
                                   rf_size=rf_size,verbose=False)

    pre2=pn.neurons.natural_images(Rnorm_fname,rf_size=rf_size,
                                other_channel=pre1,
                                verbose=False)

    
    sigma=noise
    pre1+=pn.neurons.process.add_noise_normal(0,sigma)

    sigma=noise
    pre2+=pn.neurons.process.scale_shift(contrast,0)
    pre2+=pn.neurons.process.add_noise_normal(0,sigma)
    pre=pre1+pre2

    sim=pn.simulation(100)
    sim.monitor(pre,['output'],1)

    pn.run_sim(sim,[pre],[],display_hash=False,print_time=False)

    m=sim.monitors['output']
    t,X=m.arrays()
    
    X=X[1:,:]
    
    
    return sim,X


# In[ ]:


def get_input_patch_examples_treatment():
    from deficit_defs import patch_treatment
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
    X=X[1:,:]
    return seq,X


# In[ ]:


def savefig(origfname):
    base,ext=os.path.splitext(origfname)
    import matplotlib.pyplot as plt
    
    print_fnames=[f'Manuscript/resources/{base}.png',f'Manuscript/resources/{base}.svg']
    if ext:
        if ext!='.png' and ext!='.svg':
            print_fnames+=[f'Manuscript/resources/{origfname}']
    
    for fname in print_fnames:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


# In[ ]:


def make_original_image_files():

    if not os.path.exists('asdf/bbsk081604_all.asdf'):

        subset=False
        animal='cat'
        show=False

        if subset:
            files=glob.glob('/Users/bblais/python/work/natural_images/images/original/subset_bbsk081604/*.jpg')
        else:
            files=glob.glob('/Users/bblais/python/work/natural_images/images/original/all_bbsk081604/*.jpg')

        files.sort()

        im_list=[]

        print( "Image files: %d" % len(files))
        print( "Animal:", animal)
        file_count=len(files)
        for count,fname in tqdm(enumerate(files)):

            im=Image.open(fname)
            orig_size=im.size

            # the bbsk images are angular size 60 degrees x 40 degrees
            # the raw images are resized so that 
            #   5.5 pixels ~ 0.5 degrees (cat retina)  - default
            # 
            #            ...or...
            #
            #   13 pixels ~ 7 degrees (mouse retina)
            #
            # see the contained iccns.pdf
            #

            if animal=='cat':
                new_size=[int(o*60./0.5*5.5/orig_size[0]) for o in orig_size]
            elif animal=='mouse':
                new_size=[int(o*60./7.*13/orig_size[0]) for o in orig_size]
            else:
                raise ValueError

            if fname==files[0]:
                print( "Resize: %dx%d --> %dx%d" % (orig_size[0],orig_size[1],
                                                                new_size[0],new_size[1]))
            im=im.resize(new_size)

            a=np.array(im)
            grayscale=.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]

            if show:
                plt.imshow(im)

            im_list.append(grayscale)


        var={'im':im_list,'im_scale_shift':[1.0,0.0]}

        pi5.asdf_save_images(var,'asdf/bbsk081604_all.asdf')

