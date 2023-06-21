#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


# redoing it with eta=1e-6 instead of 2e-6

# In[2]:


from treatment_sims_2023_02_21 import *


# In[3]:


def savefig(base):
    import matplotlib.pyplot as plt
    for fname in [f'Manuscript/resources/{base}.png',f'Manuscript/resources/{base}.svg']:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


# In[ ]:





# In[4]:


base='sims/2023-02-22'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# In[5]:


rf_size=19
eta=1e-6
blur=6
number_of_neurons=20
number_of_processes=4
mu_c_mat=[0,7.5]
sigma_c_mat=[0,2]


# ## Make one deficit sim to base the others on

# In[6]:


from collections import namedtuple
params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0
eta_count=0
noise_count=0
open_eye_noise=0.1

for mu_count,mu_c in enumerate(mu_c_mat):
    for sigma_count,sigma_c in enumerate(sigma_c_mat):
        all_params.append(params(count=count,
                         eta=eta,
                         noise=open_eye_noise,
                         blur=blur,
                         number_of_neurons=number_of_neurons,
         sfname=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c.asdf',
                                mu_c=mu_c,sigma_c=sigma_c))

        count+=1
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[7]:


real_time=11*60+ 30


# In[8]:


do_params=make_do_params(all_params)
print(len(do_params))
print(time2str(real_time*len(do_params)/number_of_processes))


# ### premake the images

# In[8]:


def run_one_deficit(params,overwrite=False):
    import plasticnet as pn
    
    count,eta,noise,blur,number_of_neurons,sfname,mu_c,sigma_c=(params.count,params.eta,params.noise,params.blur,
                                        params.number_of_neurons,params.sfname,params.mu_c,params.sigma_c)
    
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day
    ts=1*hour

    seq+=blur_jitter_deficit(blur=[blur,-1],
                                total_time=t,
                                noise=noise,eta=eta,number_of_neurons=number_of_neurons,
                                mu_c=mu_c,sigma_c=sigma_c,
                                save_interval=ts)

    seq.run(display_hash=False)
    pn.save(sfname,seq) 
    
    return sfname
    
    


# In[18]:


for params in tqdm(all_params):
    run_one_deficit(params,overwrite=False)


# ### run the sims

# In[10]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(run_one_deficit, do_params)
    print(result.get())


# ## Optical fix
# 
# This work is in this notebook: 2023-02-19 - Optical Fix.ipynb - except I think I used the wrong images. :-(

# In[9]:


def run_one_continuous_fix_jitter(params,
                                  overwrite=False):
    import plasticnet as pn
    count,eta,noise,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.noise,
                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    
    deficit_base_sim=f'sims/2023-02-22/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c.asdf'

    seq=pn.Sequence()
    
    seq+=fix_jitter(total_time=100*hour,
             save_interval=20*minute,number_of_neurons=params.number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
             eta=eta,noise=noise)
    seq_load(seq,deficit_base_sim)    

    seq.run(display_hash=False)
    pn.save(sfname,seq) 
    
    return sfname
    


# In[10]:


func=run_one_continuous_fix_jitter

noise_mat=linspace(0,1,11)

all_params=[]
for n,noise in enumerate(noise_mat):
    sfname=base+f'/continuous fix {number_of_neurons} neurons noise {noise:.1f}.asdf'
    
    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname
    
    p.noise=noise
    p.mu_c=7.5
    p.sigma_c=2
    
    all_params+=[p]

all_params=to_named_tuple(all_params)  
do_params=make_do_params(all_params,verbose=True)


# In[13]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(func, do_params)
    print(result.get())


# ### Optical fix for the 4 different mu_c and sigma_c

# In[11]:


noise_mat=linspace(0,1,11)
all_params=[]
for mu_count,mu_c in enumerate(mu_c_mat):
    for sigma_count,sigma_c in enumerate(sigma_c_mat):
        func=run_one_continuous_fix_jitter


        for n,noise in enumerate(noise_mat):
            sfname=base+f'/continuous fix {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c noise {noise:.1f}.asdf'

            p=Struct()
            p.eta=eta
            p.number_of_neurons=number_of_neurons
            p.sfname=sfname

            p.noise=noise
            p.mu_c=mu_c
            p.sigma_c=sigma_c

            all_params+=[p]

all_params=to_named_tuple(all_params)  
do_params=make_do_params(all_params,verbose=True)


# In[12]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(func, do_params)
    print(result.get())


# In[ ]:





# In[ ]:





# ## Patch

# In[14]:


def run_one_continuous_patch_jitter(params,
                                  deficit_base_sim='sims/2023-02-21/deficit 20 neurons 7.5 mu_c 2 sigma_c.asdf',
                                  overwrite=False):
    import plasticnet as pn
    count,eta,noise,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.noise,
                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname

    seq=pn.Sequence()

    seq+=patch_treatment_jitter(patch_noise=noise,
               total_time=100*hour,number_of_neurons=params.number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
               eta=eta,
               save_interval=20*minute)

    seq_load(seq,deficit_base_sim)    

    seq.run(display_hash=False)
    pn.save(sfname,seq) 
    
    return sfname
            
    


# In[15]:


func=run_one_continuous_patch_jitter

closed_eye_noise_mat=linspace(0,1,21)

all_params=[]
for n,noise in enumerate(closed_eye_noise_mat):
    sfname=base+f'/continuous patch {number_of_neurons} neurons noise {noise:.1f}.asdf'
    
    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname
    
    p.noise=noise
    p.mu_c=7.5
    p.sigma_c=2
    
    all_params+=[p]

all_params=to_named_tuple(all_params)  
do_params=make_do_params(all_params,verbose=True)


# In[16]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(func, do_params)
    print(result.get())


# ## Atropine

# In[17]:


def run_one_continuous_blur_jitter(params,
                                  deficit_base_sim='sims/2023-02-21/deficit 20 neurons 7.5 mu_c 2 sigma_c.asdf',
                                   overwrite=False):
    import plasticnet as pn
    count,blur,eta,noise,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.blur,params.eta,params.noise,
                                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    
    seq=pn.Sequence()
    seq+=treatment_jitter(blur=blur,
                   noise=0.1,
                   noise2=noise,  # treated (strong-eye) noise
                   total_time=100*hour,number_of_neurons=params.number_of_neurons,
                    mu_c=mu_c,sigma_c=sigma_c,
                   eta=eta,
                   save_interval=20*minute)
    
    seq_load(seq,deficit_base_sim)    

    seq.run(display_hash=False)
    pn.save(sfname,seq) 
    
    return sfname
    


# In[18]:


func=run_one_continuous_blur_jitter


atropine_blur_mat=linspace(0,6,21)
closed_eye_noise_mat=linspace(0,1,11)

all_params=[]
for b,blur in enumerate(atropine_blur_mat):
    for n,noise in enumerate(closed_eye_noise_mat):
        sfname=base+f'/continuous atropine {number_of_neurons} neurons noise {noise:.1f} blur {blur:0.1f}.asdf'

        p=Struct()
        p.eta=eta
        p.number_of_neurons=number_of_neurons
        p.sfname=sfname

        p.noise=noise
        p.blur=blur
        p.mu_c=7.5
        p.sigma_c=2

        all_params+=[p]

all_params=to_named_tuple(all_params)  
do_params=make_do_params(all_params,verbose=True)


# In[ ]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(func, do_params)
    print(result.get())


# ## Contrast

# In[19]:


def run_one_continuous_mask_jitter(params,
                                  deficit_base_sim='sims/2023-02-21/deficit 20 neurons 7.5 mu_c 2 sigma_c.asdf',
                                   overwrite=False):
    import plasticnet as pn
    count,eta,contrast,mask,f,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.contrast,params.mask,params.f,
                                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname

    
    seq=pn.Sequence()

    seq+=treatment_jitter(f=f,
                   mask=mask,
                   contrast=contrast,
                   total_time=100*hour,
                   eta=eta,
                    mu_c=mu_c,sigma_c=sigma_c,
                   save_interval=20*minute)
    seq_load(seq,deficit_base_sim)    

    seq.run(display_hash=False)
    pn.save(sfname,seq) 

    
    return sfname
    
    


# In[20]:


func=run_one_continuous_mask_jitter


contrast_mat=linspace(0,1,11)
mask_mat=array([0,1])
f_mat=array([10,30,50,70,90])


all_params=[]
for c,contrast in enumerate(contrast_mat):
    sfname=base+f'/continuous contrast {number_of_neurons} neurons contrast {contrast:.1f}.asdf'

    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname

    p.contrast=contrast
    p.mask=0
    p.f=10. # not used when mask=0
    p.mu_c=7.5
    p.sigma_c=2

    all_params+=[p]

all_params=to_named_tuple(all_params)  
do_params=make_do_params(all_params,verbose=True)


# In[21]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(func, do_params)
    print(result.get())


# ## Contrast with Mask

# In[ ]:


func=run_one_continuous_mask_jitter


contrast_mat=linspace(0,1,11)
mask_mat=array([0,1])
f_mat=array([10,30,50,70,90])


all_params=[]
for c,contrast in enumerate(contrast_mat):
    for fi,f in enumerate(f_mat):
        sfname=base+f'/continuous contrast {number_of_neurons} neurons contrast {contrast:.1f} mask f {f}.asdf'

        p=Struct()
        p.eta=eta
        p.number_of_neurons=number_of_neurons
        p.sfname=sfname

        p.contrast=contrast
        p.mask=1
        p.f=f # not used when mask=0
        p.mu_c=7.5
        p.sigma_c=2

        all_params+=[p]

all_params=to_named_tuple(all_params)  
do_params=make_do_params(all_params,verbose=True)


# In[ ]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(func, do_params)
    print(result.get())


# In[ ]:




