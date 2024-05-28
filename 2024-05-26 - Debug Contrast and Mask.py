#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[ ]:


from defs_2024_05_11 import *


# In[ ]:


base_sim_dir="sims-2024-05-26"
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


# ## trying deficit with smaller amount of time

# In[ ]:


def deficit(blur=[2.5,-1],noise=[0.1,0.1],rf_size=19,
            eta=2e-6,
           number_of_neurons=10,
           total_time=4*day,
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
    pre1=pn.neurons.natural_images(im[0],
                                   rf_size=rf_size,verbose=False)

    pre2=pn.neurons.natural_images(im[1],rf_size=rf_size,
                                other_channel=pre1,
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


# In[ ]:


def run_one_deficit(params,run=True,overwrite=False):
    import plasticnet as pn
    count,eta,blur,number_of_neurons,sfname=(params.count,params.eta,params.blur,
                                        params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()
    # deliberately use a standard deficit, with it's own eta and noise
    seq+=deficit(blur=blur,number_of_neurons=params.number_of_neurons) 

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


func=run_one_deficit
blur=2.5
sfname=f'{base_sim_dir}/deficit {number_of_neurons} neurons blur {blur:.1f}.asdf'

p=Struct()
p.eta=eta
p.number_of_neurons=number_of_neurons
p.sfname=sfname

p.blur=(blur,-1)
params=to_named_tuple([p])[0]


# In[ ]:


params


# In[ ]:


func(params,run=True,overwrite=True)


# In[ ]:


R=Results(params.sfname)
strong_i=1
weak_i=0


# In[ ]:


plot(R.t,R.y[:,0,strong_i],'b',label='Fellow Eye')
plot(R.t,R.y[:,0,weak_i],'m',label='Amblyopic Eye')

for n in range(number_of_neurons):
    plot(R.t,R.y[:,n,0],'m')
    plot(R.t,R.y[:,n,1],'b')
    
    
ylabel('Response')
legend()
print(sfname)
reformat_time_axis()    


# In[ ]:


def contrast_treatment(contrast=[1,1],noise=[0.1,0.1],
              rf_size=19,eta=5e-6,
           number_of_neurons=20,
           total_time=8*day,
           save_interval=1*hour):
    
        
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
 
        
    pre1=pn.neurons.natural_images(im[0],
                                   rf_size=rf_size,verbose=False)

    pre2=pn.neurons.natural_images(im[1],rf_size=rf_size,
                                other_channel=pre1,
                                verbose=False)

    
    for c,n,p in zip(contrast,noise,[pre1,pre2]):
        if c!=1:
            p+=pn.neurons.process.scale_shift(c,0)
        p+=pn.neurons.process.add_noise_normal(0,n)
        

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
def run_one_contrast(params,run=True,overwrite=False):
    import plasticnet as pn
    count,eta,contrast,number_of_neurons,sfname=(params.count,params.eta,params.contrast,
                                        params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname

    
    seq=pn.Sequence()
    # deliberately use a standard deficit, with it's own eta and noise
    seq+=deficit(number_of_neurons=params.number_of_neurons) 

    seq+=contrast_treatment(number_of_neurons=number_of_neurons,
                   contrast=contrast,
                   total_time=100*hour,
                   eta=eta,
                   save_interval=20*minute)

    if run:
        seq.run(display_hash=False,print_time=True)
        pn.save(sfname,seq) 

    
    return sfname
    
    


# In[ ]:


linspace(0,1,21)


# In[ ]:


func=run_one_contrast
contrast_mat=linspace(0,1,21)

all_params=[]
for c,contrast in enumerate(contrast_mat):

    sfname=f'{base_sim_dir}/contrast {number_of_neurons} neurons contrast {contrast:.2f}.asdf'

    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname

    p.contrast=(1,contrast)
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


# In[ ]:


errorbar(contrast,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,color='k',label='No Mask') # positive = recovery

ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")

xlabel('Contrast')


# ### With 8 days of deficit
# 
# ![image.png](attachment:ce8e4dc7-6e22-4465-845a-ac05401b3b20.png)

# In[ ]:


params=all_params[10]
params


# In[ ]:


R=Results(params.sfname)

idx1,idx2=[_[1] for _ in R.sequence_index]
t=R.t/day
recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[idx2,:]-R.ODI[idx1,:])/(t[idx2]-t[idx1]))  
print(recovery_rate_μ)


plot(R.t,R.y[:,0,strong_i],'b',label='Fellow Eye')
plot(R.t,R.y[:,0,weak_i],'m',label='Amblyopic Eye')

for n in range(number_of_neurons):
    plot(R.t,R.y[:,n,0],'m')
    plot(R.t,R.y[:,n,1],'b')
    
    
ylabel('Response')
legend()
print(sfname)
reformat_time_axis()  


vlines(R.t[idx1],*ylim(),color='k')
vlines(R.t[idx2],*ylim(),color='m')


# In[ ]:


R.sequence_index


# In[ ]:


idx1


# In[ ]:


R.ODI.shape


# In[ ]:


R=Results(params.sfname)
for n in range(number_of_neurons):
    plot(R.t,R.ODI[:,n],'k',alpha=0.8)
        
        
vlines(R.t[idx1],*ylim(),color='m')
vlines(R.t[idx2],*ylim(),color='m')
        
ylabel('ODI')
print(sfname)
reformat_time_axis()    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




