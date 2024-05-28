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


base='sims/2024-05-06'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# In[ ]:


def nr_jitter(blur=[-1,-1],
            noise=[0.1,0.1],
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
def run_one(params,overwrite=False,run=True,_debug=False):
    import plasticnet as pn
    count,noise1,noise2,blur1,blur2,number_of_neurons,sfname,mu_c,sigma_c=(
        params.count,params.noise1,params.noise2,
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
    
    seq+=nr_jitter(blur=[blur1,blur2],
            total_time=t,
            noise=[noise1,noise2],number_of_neurons=number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
            save_interval=ts)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname
    


# ## start with just jitter $\mu_c$ and blur on both

# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
mu_c_mat=linspace(0,30,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for mu_count,mu_c in enumerate(mu_c_mat):
    for blur_count,blur in enumerate(blur_mat):
        all_params.append(params(count=count,
                             blur1=blur,
                             blur2=blur,
                         noise1=0.1,
                         noise2=0.1,
                         mu_c=mu_c,
                         sigma_c=sigma_c,
                         number_of_neurons=number_of_neurons,
                        sfname=f'{base}/nr %d neurons dog %d blur %d mu_c %d sigma_c.asdf' % 
                                 (number_of_neurons,blur_count,mu_c,sigma_c),
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





# In[ ]:


func=run_one


# In[ ]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)
    print(sfname)


# In[ ]:


# %%time
# results = [func.remote(all_params[0],overwrite=True)]
# sfnames=ray.get(results)


# In[ ]:


real_time=5*60+ 30
print(time2str(real_time*len(do_params)/number_of_processes))


# In[ ]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# In[ ]:





# - [ ] plot a single RF for each combination
# - [ ] plot the ORI and the responses

# In[ ]:




