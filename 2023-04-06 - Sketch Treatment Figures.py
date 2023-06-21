#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[2]:


from treatment_sims_2023_02_21 import *


# In[3]:


def savefig(base):
    import matplotlib.pyplot as plt
    for fname in [f'Manuscript/resources/{base}.png',f'Manuscript/resources/{base}.svg']:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


# In[4]:


base='sims/2023-04-06'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# In[5]:


rf_size=19
eta=1e-6
number_of_neurons=5
number_of_processes=4
mu_c_mat=[0,7.5]
sigma_c_mat=[0,2]
blur_mat=[0,2,4,6,8,10,12]


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
        for blur_count,blur in enumerate(blur_mat):
        
            all_params.append(params(count=count,
                         eta=eta,
                         noise=open_eye_noise,
                         blur=blur,
                         number_of_neurons=number_of_neurons,
         sfname=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur.asdf',
                                mu_c=mu_c,sigma_c=sigma_c))

        count+=1
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[7]:


do_params=make_do_params(all_params)
print(len(do_params))


# In[8]:


def run_one_deficit_jitter(params,overwrite=False,run=True):
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

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    
    


# In[9]:


func=run_one_deficit_jitter


# In[10]:


#premake the images
for p in tqdm(all_params):
    run_one_deficit_jitter(p,overwrite=True,run=False)


# In[11]:


# %%time
print(func.__name__)
func(do_params[0],overwrite=True)


# In[11]:


real_time=1*60+ 52


# In[12]:


print(time2str(real_time*len(do_params)/number_of_processes))


# In[13]:


if do_params:
    pool = Pool(processes=number_of_processes)
    async_results = [pool.apply_async(run_one_deficit_jitter, args=(p,),kwds={'overwrite':False,'run':True}) 
                             for p in do_params]
    results =[_.get() for _ in async_results]
    
results


# In[14]:


RR={}
for p in all_params:
    
    sfname=p.sfname
    RR[sfname]=R=Results(sfname)    
    total_time=R.t.max()
    
    figure()
    plot_max_response(sfname)
    xlabel(sfname)
    plot_mini_rfs(sfname,
                  total_time/10,.15,.85,
                  2*total_time/3,.32,.85,
                  2*2*total_time/3,.55,.85,
                  2*3*total_time/3,.78,.85,
                 )
    

    
    


# ## Optical Fix

# In[6]:


def run_one_continuous_fix_jitter(params,
                                  overwrite=False,
                                 run=True):
    import plasticnet as pn
    count,eta,noise,blur,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.noise,params.blur,
                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    
    deficit_base_sim=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur.asdf'
    
    seq=pn.Sequence()
    
    seq+=fix_jitter(total_time=100*hour,
             save_interval=20*minute,number_of_neurons=params.number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
             eta=eta,noise=noise)
    seq_load(seq,deficit_base_sim)    

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname


def run_one_continuous_patch_jitter(params,
                                    overwrite=False,
                                 run=True):
    import plasticnet as pn
    count,eta,noise,blur,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.noise,
                        params.blur,
                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname

    seq=pn.Sequence()
    deficit_base_sim=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur.asdf'
    

    seq+=patch_treatment_jitter(patch_noise=noise,
               total_time=100*hour,number_of_neurons=params.number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
               eta=eta,
               save_interval=20*minute)

    seq_load(seq,deficit_base_sim)    

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
            
    
def run_one_continuous_blur_jitter(params,
                                    overwrite=False,
                                 run=True):
    import plasticnet as pn
    count,eta,noise,blur,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.noise,
                        params.blur,
                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    deficit_base_sim=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur.asdf'
    
    seq=pn.Sequence()
    seq+=treatment_jitter(blur=blur,
                   noise=0.1,
                   noise2=noise,  # treated (strong-eye) noise
                   total_time=100*hour,number_of_neurons=params.number_of_neurons,
                    mu_c=mu_c,sigma_c=sigma_c,
                   eta=eta,
                   save_interval=20*minute)
    
    seq_load(seq,deficit_base_sim)    

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 

    return sfname
        
def run_one_continuous_mask_jitter(params,
                                    overwrite=False,
                                 run=True):
    import plasticnet as pn
    count,eta,blur,contrast,mask,f,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.blur,params.contrast,params.mask,params.f,
                                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname

    
    seq=pn.Sequence()
    deficit_base_sim=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur.asdf'

    seq+=treatment_jitter(f=f,
                   mask=mask,
                   contrast=contrast,
                   total_time=100*hour,
                   eta=eta,
                          number_of_neurons=number_of_neurons,
                    mu_c=mu_c,sigma_c=sigma_c,
                   save_interval=20*minute)
    seq_load(seq,deficit_base_sim)    

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 

    
    return sfname
    
    


# ## Fix

# In[33]:


func=run_one_continuous_fix_jitter

from collections import namedtuple

noise_mat=linspace(0,1,11)
blur=6

params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0


for mu_count,mu_c in enumerate(mu_c_mat):
    for sigma_count,sigma_c in enumerate(sigma_c_mat):
        for blur_count,blur in enumerate(blur_mat):

            for noise_count,open_eye_noise in enumerate(noise_mat):
                all_params.append(params(count=count,
                             eta=eta,
                             noise=open_eye_noise,
                                 blur=blur,
                             number_of_neurons=number_of_neurons,
                 sfname=f'{base}/optical_fix {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {open_eye_noise:.1f} noise.asdf',
                            mu_c=mu_c,sigma_c=sigma_c))

                count+=1

for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[30]:


do_params=make_do_params(all_params)
print(len(do_params))
func=run_one_continuous_fix_jitter


# In[20]:


# %%time
print(func.__name__)
func(do_params[0],overwrite=True)


# In[27]:


real_time=0*60+ 22
print(time2str(real_time*len(do_params)/number_of_processes))


# In[ ]:





# In[ ]:


if do_params:
    pool = Pool(processes=number_of_processes)
    async_results = [pool.apply_async(func, args=(p,),kwds={'overwrite':False,'run':True}) 
                             for p in do_params]
    results =[_.get() for _ in async_results]
    
results


# ## Patch

# In[34]:


func=run_one_continuous_patch_jitter

noise_mat=linspace(0,1,11)

from collections import namedtuple

params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0


for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    for blur_count,blur in enumerate(blur_mat):
        for noise_count,closed_eye_noise in enumerate(noise_mat):
            all_params.append(params(count=count,
                         eta=eta,
                         noise=closed_eye_noise,
                             blur=blur,
                         number_of_neurons=number_of_neurons,
             sfname=f'{base}/patch {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {closed_eye_noise:.1f} noise.asdf',
                        mu_c=mu_c,sigma_c=sigma_c))

            count+=1

for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[35]:


do_params=make_do_params(all_params)
print(len(do_params))


# In[37]:


get_ipython().run_cell_magic('time', '', 'print(func.__name__)\nfunc(do_params[0],overwrite=True)')


# In[39]:


real_time=1*60+ 5
print(time2str(real_time*len(do_params)/number_of_processes))


# In[ ]:


if do_params:
    pool = Pool(processes=number_of_processes)
    async_results = [pool.apply_async(func, args=(p,),kwds={'overwrite':False,'run':True}) 
                             for p in do_params]
    results =[_.get() for _ in async_results]
    
results


# ## Atropine
# 
# there should be a separate blur value for the atropine, but that would be too much.  rerun this later with the specified blur for the deficit.

# In[44]:


func=run_one_continuous_blur_jitter

noise_mat=linspace(0,1,11)

from collections import namedtuple

params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0


for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    for blur_count,blur in enumerate(blur_mat):
        for noise_count,closed_eye_noise in enumerate(noise_mat):
            all_params.append(params(count=count,
                         eta=eta,
                         noise=closed_eye_noise,
                             blur=blur,
                         number_of_neurons=number_of_neurons,
             sfname=f'{base}/atropine {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {closed_eye_noise:.1f} noise.asdf',
                        mu_c=mu_c,sigma_c=sigma_c))

            count+=1

for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[45]:


do_params=make_do_params(all_params)
print(len(do_params))


# In[46]:


get_ipython().run_cell_magic('time', '', 'print(func.__name__)\nfunc(do_params[0],overwrite=True)')


# In[47]:


real_time=1*60+ 5
print(time2str(real_time*len(do_params)/number_of_processes))


# In[ ]:


if do_params:
    pool = Pool(processes=number_of_processes)
    async_results = [pool.apply_async(func, args=(p,),kwds={'overwrite':False,'run':True}) 
                             for p in do_params]
    results =[_.get() for _ in async_results]
    
results


# ## Contrast and Mask

# In[7]:


func=run_one_continuous_mask_jitter

contrast_mat=linspace(0,1,6)  # linspace(0,1,11)
mask_mat=array([0,1])
f_mat=array([10,30,50,70,90])

from collections import namedtuple


params = namedtuple('params', ['count', 'eta','blur','contrast','f','mask','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0


for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    for blur_count,blur in enumerate(blur_mat):  # only the deficit
        for contrast_count,contrast in enumerate(contrast_mat):
            for mask in [0,1]:
                if mask:
                    for fc,f in enumerate(f_mat):
                        all_params.append(params(count=count,
                                     eta=eta,
                                         blur=blur,
                                                 contrast=contrast,
                                                 f=f,
                                                 mask=mask,
                                     number_of_neurons=number_of_neurons,
                         sfname=f'{base}/contrast mask {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {contrast:.1f} contrast {mask} mask {f} f.asdf',
                                    mu_c=mu_c,sigma_c=sigma_c))

                else:
                    f=10
                    all_params.append(params(count=count,
                                 eta=eta,
                                     blur=blur,
                                             contrast=contrast,
                                             f=f,
                                             mask=mask,
                                 number_of_neurons=number_of_neurons,
                     sfname=f'{base}/contrast mask {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {contrast:.1f} contrast {mask} mask {f} f.asdf',
                                mu_c=mu_c,sigma_c=sigma_c))
                    
                        
                count+=1
                    

for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[8]:


do_params=make_do_params(all_params)
print(len(do_params))


# In[9]:


get_ipython().run_cell_magic('time', '', 'print(func.__name__)\nfunc(do_params[0],overwrite=True)')


# In[10]:


real_time=1*60+ 5
print(time2str(real_time*len(do_params)/number_of_processes))


# In[11]:


get_ipython().run_cell_magic('time', '', "if do_params:\n    pool = Pool(processes=number_of_processes)\n    async_results = [pool.apply_async(func, args=(p,),kwds={'overwrite':False,'run':True}) \n                             for p in do_params]\n    results =[_.get() for _ in async_results]\n    \nresults")


# In[ ]:




