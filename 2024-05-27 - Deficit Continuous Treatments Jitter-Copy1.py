#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[ ]:


from defs_2024_05_11 import *


# In[ ]:


base_sim_dir="sims-2024-05-27"
if not os.path.exists(base_sim_dir):
    os.mkdir(base_sim_dir)


# In[ ]:


def default_post(number_of_neurons):
    post=pn.neurons.linear_neuron(number_of_neurons)
    post+=pn.neurons.process.sigmoid(0,50)
    return post

def default_bcm(pre,post,orthogonalization=True):
    c=pn.connections.BCM(pre,post,[-.01,.01],[.1,.2])
    
    if orthogonalization:
        c+=pn.connections.process.orthogonalization(10*minute)

    c.eta=2e-6
    c.tau=15*pn.minute   

    return c


# ## Deficit

# ![image.png](attachment:2623db32-1940-4b69-8670-93335a982eed.png)

# In[ ]:


def deficit(blur=[2.5,-1],noise=[0.1,0.1],rf_size=19,
            eta=2e-6,
           number_of_neurons=10,
            mu_c=0,sigma_c=0,    
            mu_r=0,sigma_r=0,
           total_time=8*day,
           save_interval=1*hour):

    
    
    im=[]
    for b in blur:
        if b<0:
            im+=[pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
                ]
        else:
            im+=[pi5.filtered_images(base_image_file,
                                    {'type':'blur','size':b},
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
                ]
    pre1=pn.neurons.natural_images_with_jitter(im[0],
                                   rf_size=rf_size,
                                                sigma_r=0,
                                                sigma_c=0,
                                       verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(im[1],rf_size=rf_size,
                                other_channel=pre1,
                                    mu_r=mu_r,mu_c=mu_c,
                                    sigma_r=sigma_r,sigma_c=sigma_c,
                                verbose=False)

    pre1+=pn.neurons.process.add_noise_normal(0,noise[0])

    sigma=noise
    pre2+=pn.neurons.process.add_noise_normal(0,noise[1])

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


# ## Continuous Fix
# 
# ![image.png](attachment:2555cc63-5869-483d-b046-dbd2645f50a6.png)

# In[ ]:


def fix(noise=[0.1,0.1],rf_size=19,
           number_of_neurons=10,
           total_time=8*day,
           save_interval=1*hour,
            mu_c=0,sigma_c=0,    
            mu_r=0,sigma_r=0,        
           eta=2e-6):
    
    
    im=[]
    im+=[pi5.filtered_images(base_image_file,
                        {'type':'norm'},
                        {'type':'dog','sd1':1,'sd2':3},   
                        verbose=False,
                        )
        ]
    im+=[pi5.filtered_images(base_image_file,
                        {'type':'norm'},
                        {'type':'dog','sd1':1,'sd2':3},
                        verbose=False,
                        )
        ]
                     
    
    
    pre1=pn.neurons.natural_images_with_jitter(im[0],
                                        rf_size=rf_size,
                                        sigma_r=0,
                                        sigma_c=0,
                                        verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(im[1],
                                       rf_size=rf_size,
                                       other_channel=pre1,
                                       mu_r=mu_r,mu_c=mu_c,
                                       sigma_r=sigma_r,sigma_c=sigma_c,
                                       verbose=False)
    
    


    pre1+=pn.neurons.process.add_noise_normal(0,noise[0])
    pre2+=pn.neurons.process.add_noise_normal(0,noise[1])

    pre=pre1+pre2

    post=default_post(number_of_neurons)
    c=default_bcm(pre,post)
    c.eta=eta

    save_interval=save_interval

    sim=pn.simulation(total_time)

    sim.dt=200*ms

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]

@ray.remote
def run_one_continuous_fix(params,run=True,overwrite=False):
    import plasticnet as pn
    count,eta,noise,mu_c,mu_r,sigma_c,sigma_r,number_of_neurons,sfname=(
                                params.count,
                                params.eta,
                                params.noise,
                                params.mu_c,
                                params.mu_r,
                                params.sigma_c,
                                params.sigma_r,
                                params.number_of_neurons,
                                params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()
    # deliberately use a standard deficit, with it's own eta and noise
    seq+=deficit(number_of_neurons=params.number_of_neurons,
                 mu_c=mu_c,sigma_c=sigma_c,
                mu_r=mu_r,sigma_r=sigma_r,) 

    seq+=fix(total_time=100*hour,
             save_interval=20*minute,number_of_neurons=params.number_of_neurons,
             eta=eta,noise=noise,mu_c=mu_c,sigma_c=sigma_c,
             mu_r=mu_r,sigma_r=sigma_r)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:


number_of_neurons=20
eta=1e-6
number_of_processes=8
ray.init(num_cpus=number_of_processes)


# In[ ]:


func=run_one_continuous_fix

noise_mat=linspace(0,1,11)
mu_c=0
sigma_c=0
mu_r=0
sigma_r=0

all_params=[]
for n,noise in enumerate(noise_mat):
    sfname=f'{base_sim_dir}/continuous fix {number_of_neurons} neurons noise {noise:.1f}.asdf'
    
    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname
    p.mu_c=mu_c
    p.sigma_c=sigma_c
    p.mu_r=mu_r
    p.sigma_r=sigma_r
    
    p.noise=(noise,noise)
    
    all_params+=[p]

all_params=to_named_tuple(all_params)  


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)
    print(sfname)


# In[ ]:


do_params=make_do_params(all_params,verbose=True)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# In[ ]:


assert func==run_one_continuous_fix
S=Storage()
for params in tqdm(all_params):
    sfname=params.sfname
    noise=params.noise[1]
    
    R=Results(sfname)
    idx1,idx2=[_[1] for _ in R.sequence_index]
    t=R.t/day
    recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[idx2,:]-R.ODI[idx1,:])/(t[idx2]-t[idx1]))  
    
    S+=noise,recovery_rate_μ,recovery_rate_σ
    
noise,recovery_rate_μ,recovery_rate_σ=S.arrays()    


glasses_result=noise,recovery_rate_μ,recovery_rate_σ
savevars(f'{base_sim_dir}/glasses_results.asdf','glasses_result')    


# In[ ]:


params.noise


# ## Patch treatment
# 
# ![image.png](attachment:912e3e3b-6934-48cc-857e-68bd4ed34f18.png)

# In[ ]:


def patch_treatment(noise=[0.1,0.1],rf_size=19,
                   number_of_neurons=20,
                   total_time=8*day,
                   save_interval=1*hour,
                   eta=2e-6,
                    mu_c=0,sigma_c=0,    
                    mu_r=0,sigma_r=0,        
                   ):
    

    im=[]
    im+=[pi5.filtered_images(base_image_file,
                        {'type':'norm'},
                        {'type':'dog','sd1':1,'sd2':3},   
                        verbose=False,
                        )
        ]
    im+=[pi5.filtered_images(base_image_file,
                        {'type':'norm'},
                        {'type':'dog','sd1':1,'sd2':3},   
                        verbose=False,
                        )
        ]
    
    
    pre1=pn.neurons.natural_images_with_jitter(im[0],
                                        rf_size=rf_size,
                                        sigma_r=0,
                                        sigma_c=0,
                                        verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(im[1],
                                       rf_size=rf_size,
                                       other_channel=pre1,
                                       mu_r=mu_r,mu_c=mu_c,
                                       sigma_r=sigma_r,sigma_c=sigma_c,
                                       verbose=False)


    pre1+=pn.neurons.process.add_noise_normal(0,noise[0])

    pre2+=pn.neurons.process.scale_shift(0.0,0) # zero out signal
    pre2+=pn.neurons.process.add_noise_normal(0,noise[1])

    pre=pre1+pre2

    post=default_post(number_of_neurons)
    c=default_bcm(pre,post)
    c.eta=eta

    save_interval=save_interval

    sim=pn.simulation(total_time)

    sim.dt=200*ms

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]

@ray.remote
def run_one_continuous_patch(params,run=True,overwrite=False):
    import plasticnet as pn
    count,eta,noise,mu_c,sigma_c,mu_r,sigma_r,number_of_neurons,sfname=(
        params.count,
        params.eta,
        params.noise,
        params.mu_c,
        params.sigma_c,
        params.mu_r,
        params.sigma_r,
        params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    
    seq=pn.Sequence()
    # deliberately use a standard deficit, with it's own eta and noise
    seq+=deficit(number_of_neurons=params.number_of_neurons,
                mu_c=mu_c,sigma_c=sigma_c,
                mu_r=mu_r,sigma_r=sigma_r,) 
    
    seq+=patch_treatment(noise=noise,
               total_time=100*hour,number_of_neurons=params.number_of_neurons,
               eta=eta,
               mu_c=mu_c,sigma_c=sigma_c,
               mu_r=mu_r,sigma_r=sigma_r,
               save_interval=20*minute)

    if run:
        seq.run(display_hash=False,print_time=True)
        pn.save(sfname,seq) 
    
    return sfname
        


# In[ ]:


func=run_one_continuous_patch

closed_eye_noise_mat=linspace(0,1,21)
mu_c=0
sigma_c=0
mu_r=0
sigma_r=0

all_params=[]
for n,noise in enumerate(closed_eye_noise_mat):
    sfname=f'{base_sim_dir}/continuous patch {number_of_neurons} neurons noise {noise:.1f}.asdf'
    
    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname
    p.mu_c=mu_c
    p.sigma_c=sigma_c
    p.mu_r=mu_r
    p.sigma_r=sigma_r
    
    p.noise=(0.1,noise)
    
    all_params+=[p]

all_params=to_named_tuple(all_params)  


# In[ ]:


do_params=make_do_params(all_params,verbose=True)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# In[ ]:


assert func==run_one_continuous_patch
S=Storage()
for params in tqdm(all_params):
    sfname=params.sfname
    noise=params.noise[1]
    
    R=Results(sfname)

    
    idx1,idx2=[_[1] for _ in R.sequence_index]
    t=R.t/day
    recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[idx2,:]-R.ODI[idx1,:])/(t[idx2]-t[idx1]))  
    
    S+=noise,recovery_rate_μ,recovery_rate_σ    
        
noise,recovery_rate_μ,recovery_rate_σ=S.arrays()

patch_result=noise,recovery_rate_μ,recovery_rate_σ
savevars(f'{base_sim_dir}/patch_results.asdf','patch_result')


# In[ ]:


params.noise


# ## Atropine Treatment
# 
# ![image.png](attachment:3fdad06d-8960-4207-8f47-13d9e4f38a2f.png)

# In[ ]:


def atropine_treatment(noise=[0.1,0.1],
                       blur=[-1,-1],
                       rf_size=19,
           number_of_neurons=10,
           total_time=8*day,
           save_interval=1*hour,
           eta=2e-6,
                    mu_c=0,sigma_c=0,    
                    mu_r=0,sigma_r=0,        
                      ):
    
    
    im=[]
    for b in blur:
        if b<0:
            im+=[pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
                ]
        else:
            im+=[pi5.filtered_images(base_image_file,
                                    {'type':'blur','size':b},
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
                ]
            
    pre1=pn.neurons.natural_images_with_jitter(im[0],
                                        rf_size=rf_size,
                                        sigma_r=0,
                                        sigma_c=0,
                                        verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(im[1],
                                       rf_size=rf_size,
                                       other_channel=pre1,
                                       mu_r=mu_r,mu_c=mu_c,
                                       sigma_r=sigma_r,sigma_c=sigma_c,
                                       verbose=False)

    pre1+=pn.neurons.process.add_noise_normal(0,noise[0])

    sigma=noise
    pre2+=pn.neurons.process.add_noise_normal(0,noise[1])

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




@ray.remote
def run_one_continuous_atropine(params,run=True,overwrite=False):
    import plasticnet as pn
    count,blur,eta,noise,mu_c,sigma_c,mu_r,sigma_r,number_of_neurons,sfname=(
        params.count,
            params.blur,params.eta,params.noise,
        params.mu_c,params.sigma_c,params.mu_r,params.sigma_r,
                params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    
    seq=pn.Sequence()
    # deliberately use a standard deficit, with it's own eta and noise
    seq+=deficit(number_of_neurons=params.number_of_neurons,
                mu_c=mu_c,sigma_c=sigma_c,
                mu_r=mu_r,sigma_r=sigma_r,) 

    seq+=atropine_treatment(blur=(-1,blur),
                   noise=noise,
                   total_time=100*hour,
                    number_of_neurons=params.number_of_neurons,
                   eta=eta,
                   mu_c=mu_c,sigma_c=sigma_c,
                   mu_r=mu_r,sigma_r=sigma_r,
                   save_interval=20*minute)
    

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:


func=run_one_continuous_atropine


atropine_blur_mat=linspace(0,6,21)
closed_eye_noise_mat=linspace(0,1,11)
mu_c=0
sigma_c=0
mu_r=0
sigma_r=0

all_params=[]
for b,blur in enumerate(atropine_blur_mat):
    for n,noise in enumerate(closed_eye_noise_mat):
        sfname=f'{base_sim_dir}/continuous atropine {number_of_neurons} neurons noise {noise:.1f} blur {blur:0.1f}.asdf'

        p=Struct()
        p.eta=eta
        p.number_of_neurons=number_of_neurons
        p.sfname=sfname
        p.mu_c=mu_c
        p.sigma_c=sigma_c
        p.mu_r=mu_r
        p.sigma_r=sigma_r

        p.noise=(0.1,noise)
        p.blur=blur

        all_params+=[p]

all_params=to_named_tuple(all_params)  


# In[ ]:


do_params=make_do_params(all_params,verbose=True)


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# In[ ]:





# In[ ]:


assert func==run_one_continuous_atropine
S=Storage()
for params in tqdm(all_params):
    sfname=params.sfname
    noise=params.noise[1]
    blur=params.blur
    
    R=Results(sfname)

    
    idx1,idx2=[_[1] for _ in R.sequence_index]
    t=R.t/day
    recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[idx2,:]-R.ODI[idx1,:])/(t[idx2]-t[idx1]))  
    
    S+=noise,blur,recovery_rate_μ,recovery_rate_σ    
        
noise,blur,recovery_rate_μ,recovery_rate_σ=S.arrays()



# In[ ]:





# In[ ]:


blur


# In[ ]:


blur_orig=blur
noise_orig=noise


# In[ ]:


noise_N=len(closed_eye_noise_mat)
blur_N=len(atropine_blur_mat)

noise=noise.reshape(blur_N,noise_N)
noise,blur,recovery_rate_μ,recovery_rate_σ=[_.reshape(blur_N,noise_N).T for _ in (noise,blur,recovery_rate_μ,recovery_rate_σ)]



# In[ ]:


blur


# In[ ]:


atropine_result=noise,blur,recovery_rate_μ,recovery_rate_σ

savevars(f'{base_sim_dir}/atropine_results.asdf','atropine_result')


# ## Contrast, then Contrast with Mask
# 
# ![image.png](attachment:56a6380e-987a-4fa9-95ab-c4e6e7cedf05.png)

# In[ ]:


def mask_contrast_treatment(contrast=[1,1],noise=[0.1,0.1],
              rf_size=19,eta=5e-6,
                    mu_c=0,sigma_c=0,    
                    mu_r=0,sigma_r=0,        
              f=30,  # size of the blur for mask, which is a measure of overlap
           number_of_neurons=20,
           total_time=8*day,
           save_interval=1*hour,
             mask=None,
             blur=(-1,-1)):
    
    
    if not f in [10,30,50,70,90]:
        raise ValueError("Unknown f %s" % str(f))

        
        
    im=[]
    if not mask:
        for (b,c) in zip(blur,contrast):
            if b<0:
                im+=[pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':'', 
                                        'seed':101,'apply_to_average':True},
                                        {'type':'norm'},
                                        {'type':'dog','sd1':1,'sd2':3},   
                                        verbose=False,
                                        )
                    ]
            else:
                im+=[pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':'', 
                                        'seed':101,'apply_to_average':True},
                                        {'type':'blur','size':b},
                                        {'type':'norm'},
                                        {'type':'dog','sd1':1,'sd2':3},   
                                        verbose=False,
                                        )
                    ]
    else:
        for [b,mask_name,c] in zip(blur,['A','F'],contrast):
            if b<0:
                im+=[pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':f'bblais-masks-20240511/2024-05-11-*-{mask_name}-fsig{int(f)}.png', 
                                        'seed':101,'apply_to_average':True},
                                        {'type':'norm'},
                                        {'type':'dog','sd1':1,'sd2':3},   
                                        verbose=False,
                                         )
                    ]
            else:
                im+=[pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':f'bblais-masks-20240511/2024-05-11-*-{mask_name}-fsig{int(f)}.png', 
                                        'seed':101,'apply_to_average':True},
                                        {'type':'blur','size':b},
                                        {'type':'norm'},
                                        {'type':'dog','sd1':1,'sd2':3},   
                                        verbose=False,
                                        )
                     ]
        
        
        
    pre1=pn.neurons.natural_images_with_jitter(im[0],
                                        rf_size=rf_size,
                                        sigma_r=0,
                                        sigma_c=0,
                                        verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(im[1],
                                       rf_size=rf_size,
                                       other_channel=pre1,
                                       mu_r=mu_r,mu_c=mu_c,
                                       sigma_r=sigma_r,sigma_c=sigma_c,
                                       verbose=False)

    if contrast[0]!=1:
        pre1+=pn.neurons.process.scale_shift(contrast[0],0)

    pre1+=pn.neurons.process.add_noise_normal(0,noise[0])

    
    if contrast[1]!=1:
        pre2+=pn.neurons.process.scale_shift(contrast[1],0)
    pre2+=pn.neurons.process.add_noise_normal(0,noise[1])

    pre=pre1+pre2

    post=pn.neurons.linear_neuron(number_of_neurons)
    post+=pn.neurons.process.sigmoid(-1,50)

    c=pn.connections.BCM(pre,post,[-.1,.1],[.1,.2])
    c.eta=eta
    c.tau=100*second

    save_interval=save_interval

    sim=pn.simulation(total_time)

    sim.dt=200*ms

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]


    

@ray.remote
def run_one_continuous_mask(params,run=True,overwrite=False):
    import plasticnet as pn
    count,eta,contrast,mask,f,mu_c,sigma_c,mu_r,sigma_r,number_of_neurons,sfname=(
        params.count,params.eta,params.contrast,params.mask,params.f,
        params.mu_c,params.sigma_c,params.mu_r,params.sigma_r,
        params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname

    
    seq=pn.Sequence()
    # deliberately use a standard deficit, with it's own eta and noise
    seq+=deficit(number_of_neurons=params.number_of_neurons,
                mu_c=mu_c,sigma_c=sigma_c,
                mu_r=mu_r,sigma_r=sigma_r,
                )

    seq+=mask_contrast_treatment(f=f,
                   mask=mask,
                   contrast=contrast,
                   total_time=100*hour,
                   eta=eta,
                mu_c=mu_c,sigma_c=sigma_c,
                mu_r=mu_r,sigma_r=sigma_r,
                   save_interval=20*minute)

    if run:
        seq.run(display_hash=False,print_time=True)
        pn.save(sfname,seq) 

    
    return sfname
    
    


# In[ ]:


func=run_one_continuous_mask


contrast_mat=linspace(0,1,21)
f_mat=array([10,30,50,70,90])

mu_c=0
sigma_c=0
mu_r=0
sigma_r=0

all_params=[]
for c,contrast in enumerate(contrast_mat):
    sfname=f'{base_sim_dir}/continuous contrast {number_of_neurons} neurons contrast {contrast:.2f}.asdf'

    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname

    p.contrast=(1,contrast)
    p.mask=0
    p.f=10. # not used when mask=0
    p.mu_c=mu_c
    p.sigma_c=sigma_c
    p.mu_r=mu_r
    p.sigma_r=sigma_r

    all_params+=[p]

all_params=to_named_tuple(all_params)  


# In[ ]:


do_params=make_do_params(all_params,verbose=True)


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# In[ ]:


assert func==run_one_continuous_mask
S=Storage()
for params in tqdm(all_params):
    sfname=params.sfname
    contrast=params.contrast[1]
    
    R=Results(sfname)

    
    idx1,idx2=[_[1] for _ in R.sequence_index]
    t=R.t/day
    recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[idx2,:]-R.ODI[idx1,:])/(t[idx2]-t[idx1]))  
    
    S+=contrast,recovery_rate_μ,recovery_rate_σ    
        
contrast,recovery_rate_μ,recovery_rate_σ=S.arrays()

contrast_result=contrast,recovery_rate_μ,recovery_rate_σ

savevars(f'{base_sim_dir}/contrast_results.asdf','contrast_result')


# ## Contrast with Mask

# In[ ]:


func=run_one_continuous_mask


contrast_mat=linspace(0,1,21)
f_mat=array([10,30,50,70,90])

mu_c=0
sigma_c=0
mu_r=0
sigma_r=0

all_params=[]
for c,contrast in enumerate(contrast_mat):
    for fi,f in enumerate(f_mat):
    
        sfname=f'{base_sim_dir}/continuous contrast {number_of_neurons} neurons contrast {contrast:.2f} mask f {f}.asdf'

        p=Struct()
        p.eta=eta
        p.number_of_neurons=number_of_neurons
        p.sfname=sfname

        p.contrast=(1,contrast)
        p.mask=1
        p.f=f
        p.mu_c=mu_c
        p.sigma_c=sigma_c
        p.mu_r=mu_r
        p.sigma_r=sigma_r

        all_params+=[p]

all_params=to_named_tuple(all_params)  



# In[ ]:


do_params=make_do_params(all_params,verbose=True)


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# In[ ]:





# In[ ]:


assert func==run_one_continuous_mask
S=Storage()
for params in tqdm(all_params):
    sfname=params.sfname
    contrast=params.contrast[1]
    f=params.f
    R=Results(sfname)

    
    idx1,idx2=[_[1] for _ in R.sequence_index]
    t=R.t/day
    recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[idx2,:]-R.ODI[idx1,:])/(t[idx2]-t[idx1]))  
    
    S+=f,contrast,recovery_rate_μ,recovery_rate_σ    
        
f,contrast,recovery_rate_μ,recovery_rate_σ=S.arrays()


# In[ ]:


f_N=len(f_mat)
contrast_N=len(contrast_mat)

contrast.reshape(contrast_N,f_N)


# In[ ]:


f_N=len(f_mat)
contrast_N=len(contrast_mat)


f,contrast,recovery_rate_μ,recovery_rate_σ=[_.reshape(contrast_N,f_N).T for _ in (f,contrast,recovery_rate_μ,recovery_rate_σ)]


# In[ ]:


mask_result=f,contrast,recovery_rate_μ,recovery_rate_σ

savevars(f'{base_sim_dir}/mask_results.asdf','mask_result')


# In[ ]:





# In[ ]:




