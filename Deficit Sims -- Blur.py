#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


from deficit_defs import *


# In[16]:


_debug = False
if _debug:
    print("Debugging")


# In[3]:


base='sims/2022-11-03'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# In[4]:


rf_size=19
#eta_mat=linspace(1e-7,5e-6,11)
eta=2e-6
blur_mat=linspace(0,8,17)
print(blur_mat)
number_of_neurons=20
number_of_processes=4


# In[5]:


base_image_file='asdf/bbsk081604_all.asdf'
print("Base Image File:",base_image_file)

blur=2.5
Lfname=pi5.filtered_images(
                            base_image_file,
                            {'type':'blur','size':blur},
                            {'type':'log2dog','sd1':1,'sd2':3},
                            )
Rfname=pi5.filtered_images(
                            base_image_file,
                            {'type':'log2dog','sd1':1,'sd2':3},
                            )



# In[6]:


image_data=pi5.asdf_load_images(Lfname)
im=[arr.astype(float)*image_data['im_scale_shift'][0]+
        image_data['im_scale_shift'][1] for arr in image_data['im']]

del image_data
plt.figure(figsize=(16,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(im[i],cmap=plt.cm.gray)
    plt.axis('off')


# In[7]:


image_data=pi5.asdf_load_images(Rfname)
im=[arr.astype(float)*image_data['im_scale_shift'][0]+
        image_data['im_scale_shift'][1] for arr in image_data['im']]

del image_data
plt.figure(figsize=(16,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(im[i],cmap=plt.cm.gray)
    plt.axis('off')


# ## Premake the image files

# In[8]:


blur_mat=linspace(0,8,17)


# In[9]:


base_image_file='asdf/bbsk081604_all.asdf'
print("Base Image File:",base_image_file)

normal_image=pi5.filtered_images(
                                base_image_file,
                                {'type':'log2dog','sd1':1,'sd2':3},
                                )

for blur in blur_mat:

    Lfname=pi5.filtered_images(
                                base_image_file,
                                {'type':'blur','size':blur},
                                {'type':'log2dog','sd1':1,'sd2':3},
                                )



# In[18]:


def blur_deficit(blur=[2.5,-1],
            noise=[0.1,.1],
                 rf_size=19,eta=2e-6,
           number_of_neurons=10,
           total_time=8*day,
           save_interval=1*hour):

    
    if _debug:
        total_time=1*minute
        save_interval=1*second
        
    images=[]
    
    for bv in blur:
        if bv<0:
            im=pi5.filtered_images(
                                base_image_file,
                                {'type':'log2dog','sd1':1,'sd2':3},
                                )
        else:
            im=pi5.filtered_images(
                                    base_image_file,
                                    {'type':'blur','size':bv},
                                    {'type':'log2dog','sd1':1,'sd2':3},
                                    )
        images.append(im)
        
    pre1=pn.neurons.natural_images(images[0],
                                   rf_size=rf_size,verbose=False)

    pre2=pn.neurons.natural_images(images[1],rf_size=rf_size,
                                other_channel=pre1,
                                verbose=False)

    sigma=noise
    pre1+=pn.neurons.process.add_noise_normal(0,sigma)

    sigma=noise
    pre2+=pn.neurons.process.add_noise_normal(0,sigma)

    pre=pre1+pre2

    post=default_post(number_of_neurons)
    c=default_bcm(pre,post)
    c.eta=eta

    sim=pn.simulation(total_time)
    sim.dt=200*ms

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]


# In[23]:


def run_one_left_blur(params,overwrite=False):
    import plasticnet as pn
    count,eta,noise,blur,number_of_neurons,sfname=(params.count,params.eta,params.noise,params.blur,
                                        params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day
    ts=1*hour

    # DEBUG
    if _debug:
        t=1*minute
        ts=1*second
    
    seq+=blur_deficit(blur=[blur,-1],total_time=t,
                 noise=noise,eta=eta,number_of_neurons=number_of_neurons,
               save_interval=ts)

    
    seq.run(display_hash=False)
    pn.save(sfname,seq) 
    
    return sfname
    


# In[24]:


total_time=8*day
real_time=5*60+ 55


# In[25]:


from collections import namedtuple
params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname'])
all_params=[]
count=0
eta_count=0
eta=2e-6
noise_count=0
open_eye_noise=0.0

for blur_count,blur in enumerate(blur_mat):
    all_params.append(params(count=count,
                     eta=eta,
                         blur=blur,
                     noise=open_eye_noise,
                     number_of_neurons=number_of_neurons,
         sfname=f'{base}/deficit %d neurons logdog %d eta %d noise %d blur.asdf' % 
                                         (number_of_neurons,eta_count,noise_count,blur_count)))
        
for a in all_params[:10]:
    print(a)

print(len(all_params))

print(time2str(real_time*len(all_params)/number_of_processes))


# In[26]:


get_ipython().run_cell_magic('time', '', 'run_one_left_blur(all_params[0],overwrite=True)')


# In[28]:


pool = Pool(processes=number_of_processes)
result = pool.map_async(run_one_left_blur, all_params)
print(result.get())


# In[ ]:




