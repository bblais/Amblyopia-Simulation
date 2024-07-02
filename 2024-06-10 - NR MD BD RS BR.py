#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[ ]:


from deficit_defs import *


# In[ ]:


import ray 
number_of_processes=8
ray.init(num_cpus=number_of_processes)


# In[ ]:


base='sims/2024-06-10'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# In[ ]:


def deprivation_jitter(blur=[-1,-1],
            noise=[0.1,0.1],
              scale=[1,1],
              rf_size=19,
           number_of_neurons=10,
            mu_c=0,sigma_c=0,    
            mu_r=0,sigma_r=0,
           total_time=8*day,
           save_interval=1*hour,_debug=False):

    if _debug:
        total_time=1*minute
        save_interval=1*second

        
    images=[]
    for bv in blur:
        if bv<=0:
            im=pi5.filtered_images(
                                base_image_file,
                                {'type':'dog','sd1':1,'sd2':3},
                                {'type':'norm'},
                                )
        else:
            im=pi5.filtered_images(
                                    base_image_file,
                                    {'type':'blur','size':bv},
                                    {'type':'dog','sd1':1,'sd2':3},
                                    {'type':'norm'},
                                    )
        images.append(im)
                
        
        
        
    dt=200*ms      
    eta=2e-6
    pre1=pn.neurons.natural_images_with_jitter(images[0],
                                                rf_size=rf_size,
                                                time_between_patterns=dt,
                                                sigma_r=0,
                                                sigma_c=0,
                                                verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(images[1],
                                                rf_size=rf_size,
                                                other_channel=pre1,
                                                time_between_patterns=dt,
                                                mu_r=mu_r,mu_c=mu_c,
                                                sigma_r=sigma_r,sigma_c=sigma_c,
                                                verbose=False)

    if scale[0]!=1:
        pre1+=pn.neurons.process.scale_shift(scale[0],0)

    if scale[1]!=1:
        pre2+=pn.neurons.process.scale_shift(scale[1],0)
    
    pre1+=pn.neurons.process.add_noise_normal(0,noise[0])
    pre2+=pn.neurons.process.add_noise_normal(0,noise[1])

    
    
    
    pre=pre1+pre2

    post=default_post(number_of_neurons)
    c=default_bcm(pre,post)
    c.eta=eta

    sim=pn.simulation(total_time)
    sim.dt=dt

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]


# In[ ]:


@ray.remote
def run_one_nr(params,overwrite=False,run=True,_debug=False):
    import plasticnet as pn
    count,noise1,noise2,scale1,scale2,blur1,blur2,number_of_neurons,sfname,mu_c,sigma_c=(
        params.count,params.noise1,params.noise2,
        params.scale1,params.scale2,
        params.blur1,params.blur2,
        params.number_of_neurons,params.sfname,params.mu_c,params.sigma_c)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day*2
    ts=1*hour

    # DEBUG
    if _debug:
        t=1*minute
        ts=1*second
    
    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[scale1,scale2],
            noise=[noise1,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:


noise_mat=linspace(0,2,21)
noise_mat


# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=noise,
         noise2=noise,

         blur1=-1,
         blur2=-1,
         scale1=1,
         scale2=1,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/nr %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[ ]:


do_params=make_do_params(all_params)
len(do_params)


# In[ ]:


func=run_one_nr


# In[ ]:


# ### premake the images
# for params in tqdm(all_params):
#     result=func.remote(params,run=False,overwrite=True)
#     sfname=ray.get(result)
#     print(sfname)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'results = [func.remote(all_params[0],overwrite=True)]\nsfnames=ray.get(results)\n')


# In[ ]:


real_time=5*60+ 36
print(time2str(real_time*len(do_params)/number_of_processes))


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# # MD

# In[ ]:


@ray.remote
def run_one_md(params,overwrite=False,run=True,_debug=False):
    import plasticnet as pn
    count,noise1,noise2,scale1,scale2,blur1,blur2,number_of_neurons,sfname,mu_c,sigma_c=(
        params.count,params.noise1,params.noise2,
        params.scale1,params.scale2,
        params.blur1,params.blur2,
        params.number_of_neurons,params.sfname,params.mu_c,params.sigma_c)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day*2
    ts=1*hour

    # DEBUG
    if _debug:
        t=1*minute
        ts=1*second
    
    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[1,1],
            noise=[noise2,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[scale1,scale2],
            noise=[noise1,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=noise,
         noise2=0.1,

         blur1=-1,
         blur2=-1,
         scale1=0,
         scale2=1,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/md %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[ ]:


do_params=make_do_params(all_params)
print(len(do_params))
func=run_one_md


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)
    print(sfname)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# # BD

# In[ ]:


@ray.remote
def run_one_bd(params,overwrite=False,run=True,_debug=False):
    import plasticnet as pn
    count,noise1,noise2,scale1,scale2,blur1,blur2,number_of_neurons,sfname,mu_c,sigma_c=(
        params.count,params.noise1,params.noise2,
        params.scale1,params.scale2,
        params.blur1,params.blur2,
        params.number_of_neurons,params.sfname,params.mu_c,params.sigma_c)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day*2
    ts=1*hour

    # DEBUG
    if _debug:
        t=1*minute
        ts=1*second
    
    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[1,1],
            noise=[noise2,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[scale1,scale2],
            noise=[noise1,noise1],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=noise,
         noise2=0.1,

         blur1=-1,
         blur2=-1,
         scale1=0,
         scale2=0,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/bd %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


do_params=make_do_params(all_params)
print(len(do_params))
func=run_one_bd


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)
    print(sfname)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# # RS

# In[ ]:


@ray.remote
def run_one_rs(params,overwrite=False,run=True,_debug=False):
    import plasticnet as pn
    count,noise1,noise2,scale1,scale2,blur1,blur2,number_of_neurons,sfname,mu_c,sigma_c=(
        params.count,params.noise1,params.noise2,
        params.scale1,params.scale2,
        params.blur1,params.blur2,
        params.number_of_neurons,params.sfname,params.mu_c,params.sigma_c)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day*2
    ts=1*hour

    # DEBUG
    if _debug:
        t=1*minute
        ts=1*second
    
    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[1,1],
            noise=[noise1,noise1],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[0,1],
            noise=[noise2,noise1],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[scale1,scale2],
            noise=[noise1,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=0.1,
         noise2=noise,

         blur1=-1,
         blur2=-1,
         scale1=1,
         scale2=0,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/rs %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


do_params=make_do_params(all_params)
print(len(do_params))
func=run_one_rs


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)
    print(sfname)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# # BR

# In[ ]:


@ray.remote
def run_one_br(params,overwrite=False,run=True,_debug=False):
    import plasticnet as pn
    count,noise1,noise2,scale1,scale2,blur1,blur2,number_of_neurons,sfname,mu_c,sigma_c=(
        params.count,params.noise1,params.noise2,
        params.scale1,params.scale2,
        params.blur1,params.blur2,
        params.number_of_neurons,params.sfname,params.mu_c,params.sigma_c)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day*2
    ts=1*hour

    # DEBUG
    if _debug:
        t=1*minute
        ts=1*second
    
    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[1,1],
            noise=[noise2,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[0,1],
            noise=[noise1,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    seq+=deprivation_jitter(blur=[blur1,blur2],
            total_time=t,
            scale=[scale1,scale2],
            noise=[noise2,noise2],
            number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=noise,
         noise2=noise,

         blur1=-1,
         blur2=-1,
         scale1=1,
         scale2=1,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/br %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


do_params=make_do_params(all_params)
print(len(do_params))
func=run_one_rs


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)
    print(sfname)


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)

