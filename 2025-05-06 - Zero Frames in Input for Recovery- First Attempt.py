#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylab import *


# In[2]:


from deficit_defs_2025_02_25 import *


# In[5]:


mu_c=9
sigma_c=9
mu_r=0
sigma_r=0

# take about 3 hours to do a full set of sims

base_sim_dir=f"sims-2025-05-06 mu {mu_c} sigma {sigma_c}"
if not os.path.exists(base_sim_dir):
    print("new")
    os.mkdir(base_sim_dir)
print(base_sim_dir)


# In[6]:


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

# In[7]:


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

    sim+=pn.grating_response(print_time=False,
                        k_mat=linspace(1,20,40)/19.0*pi,
                            )

    return sim,[pre,post],[c]


# In[8]:


number_of_neurons=20
eta=1e-6
number_of_processes=8
ray.init(num_cpus=number_of_processes)


# ## Fix with Zero Fraction of Frames

# In[9]:


def fix_zero_fraction(noise=[0.1,0.1],rf_size=19,
           number_of_neurons=10,
           total_time=8*day,
           save_interval=1*hour,
            mu_c=0,sigma_c=0,    
            mu_r=0,sigma_r=0,        
           eta=2e-6,
                 fraction=[1/3,1/3]):


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


    pre1+=pn.neurons.process.zero_fraction(fraction[0])
    pre2+=pn.neurons.process.zero_fraction(fraction[1])

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

    sim+=pn.grating_response(print_time=False,
                        k_mat=linspace(1,20,40)/19.0*pi)

    return sim,[pre,post],[c]

@ray.remote
def run_one_continuous_fix_zero_fraction(params,run=True,overwrite=False):
    import plasticnet as pn
    count,eta,noise,fraction,mu_c,mu_r,sigma_c,sigma_r,number_of_neurons,sfname=(
                                params.count,
                                params.eta,
                                params.noise,
                                params.fraction,
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

    seq+=fix_zero_fraction(total_time=100*hour,
             save_interval=20*minute,number_of_neurons=params.number_of_neurons,
             eta=eta,noise=noise,fraction=fraction,mu_c=mu_c,sigma_c=sigma_c,
             mu_r=mu_r,sigma_r=sigma_r)

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 

    return sfname



# In[24]:


func=run_one_continuous_fix_zero_fraction

noise_mat=linspace(0,1,11)

all_params=[]
for n,noise in enumerate(noise_mat):
    sfname=f'{base_sim_dir}/continuous zero fraction {number_of_neurons} neurons noise {noise:.1f}.asdf'

    p=Struct()
    p.eta=eta
    p.number_of_neurons=number_of_neurons
    p.sfname=sfname
    p.mu_c=mu_c
    p.sigma_c=sigma_c
    p.mu_r=mu_r
    p.sigma_r=sigma_r

    p.noise=(noise,noise)
    p.fraction=(0,2/3)

    all_params+=[p]

all_params=to_named_tuple(all_params)  


# In[25]:


### premake the images
for params in tqdm(all_params):
    result=func.remote(params,run=False,overwrite=True)
    sfname=ray.get(result)
    print(sfname)


# In[26]:


do_params=make_do_params(all_params,verbose=True)


# In[27]:


results = [func.remote(p) for p in do_params]
sfnames=ray.get(results)


# In[28]:


RR={}
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[29]:


subResults={}
for params in tqdm(all_params):
    sfname=params.sfname
    R=RR[sfname]

    subR=Struct()
    subR.t=R.t/day
    t=subR.t

    subR.y=R.y
    subR.θ=R.θ

    idx1,idx2=[_[1] for _ in R.sequence_index]

    subR.ODI=R.ODI
    subR.ORI=R.ORI
    subR.theta_mat=R.theta_mat
    subR.k_mat=R.k_mat
    subR.SF_Var=R.SF_Var
    subR.max_SF=R.max_SF
    subR.sfname=params.sfname
    subR.params=params
    subR.sequence_index=R.sequence_index
    subR.idx=[0]+[_[-1] for _ in R.sequence_index]
    subR.W=R.W[subR.idx,::]
    subR.recovery_rate=(R.ODI[idx2,:]-R.ODI[idx1,:])/(t[idx2]-t[idx1])
    subR.SF_Var_recovery_rate=(R.SF_Var[idx2,:]-R.SF_Var[idx1,:])/(t[idx2]-t[idx1])
    subR.ODI2=R.ODI[idx2,:]
    subR.ODI1=R.ODI[idx1,:]

    subResults[params.sfname]=subR


# In[30]:


assert func==run_one_continuous_fix_zero_fraction
St=Storage()
for params in tqdm(all_params):
    sfname=params.sfname
    noise=params.noise[1]
    fraction0=params.fraction[0]
    fraction1=params.fraction[1]

    subR=subResults[sfname]

    idx0,idx1,idx2=subR.idx
    t=subR.t
    recovery_rate_μ,recovery_rate_σ=μσ(subR.recovery_rate)

    ODI_μ2=subR.ODI[idx2,:].mean()  # average across neurons, at the end of a seq, for each channel
    S=subR.ODI[idx2,:].std()
    N=subR.ODI.shape[1]
    K=1+20/N**2
    ODI_σ2=K*S/np.sqrt(N)

    ODI_μ1=subR.ODI[idx1,:].mean()  # average across neurons, at the end of a seq, for each channel
    S=subR.ODI[idx1,:].std()
    N=subR.ODI.shape[1]
    K=1+20/N**2
    ODI_σ1=K*S/np.sqrt(N)

    SF_Var_recovery_rate_μ,SF_Var_recovery_rate_σ=μσ(subR.SF_Var_recovery_rate)


    St+=noise,fraction0,fraction1,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2,SF_Var_recovery_rate_μ,SF_Var_recovery_rate_σ 

noise,fraction0,fraction1,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2,SF_Var_recovery_rate_μ,SF_Var_recovery_rate_σ=St.arrays()

glasses_result=noise,fraction0,fraction1,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2,SF_Var_recovery_rate_μ,SF_Var_recovery_rate_σ

savevars(f'{base_sim_dir}/full_zero_fraction_results.asdf',
         'subResults','glasses_result','noise_mat',
          'noise','fraction0','fraction1','recovery_rate_μ','recovery_rate_σ',
            'ODI_μ1','ODI_σ1','ODI_μ2','ODI_σ2',
          'SF_Var_recovery_rate_μ','SF_Var_recovery_rate_σ')
# savevars(f'{base_sim_dir}/glasses_results.asdf','glasses_result')    


# In[31]:


loadvars('sims-2025-04-20 mu 9 sigma 9'+'/full_glasses_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.viridis
color = colormap(np.linspace(0, 1,n))
glasses_plot_color=color[2]

glasses_μ=-recovery_rate_μ
glasses_σ=2*recovery_rate_σ

# best case
idx=argmax(glasses_μ)
max_glasses=glasses_μ[idx]+glasses_σ[idx]
min_glasses=glasses_μ[idx]-glasses_σ[idx]
print(min_glasses,max_glasses)


errorbar(noise,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color=glasses_plot_color,
        label='Glasses') # positive = recovery


loadvars(base_sim_dir+'/full_zero_fraction_results.asdf')

colormap=cm.Oranges
color = colormap(np.linspace(0, 1,n))
zero_fraction_plot_color=color[2]

#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2,,SF_Var_recovery_rate_μ,SF_Var_recovery_rate_σ =glasses_result        

zero_fraction_μ=-recovery_rate_μ
zero_fraction_σ=2*recovery_rate_σ

# best case
idx=argmax(glasses_μ)
max_zero_fraction=zero_fraction_μ[idx]+zero_fraction_σ[idx]
min_zero_fraction=zero_fraction_μ[idx]-zero_fraction_σ[idx]
print(min_zero_fraction,max_zero_fraction)

errorbar(noise,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color=zero_fraction_plot_color,
        label='Zero Fraction') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
xlabel('Open-Eye Noise Level')
legend()



# ## Fraction 1/3,1/3
# 
# ![image.png](attachment:5c8a9753-0662-42ef-a64e-3d4703fe661f.png)

# ## Fraction 0,2/3
# 
# ![image.png](attachment:e9c86576-7e6c-4df7-8dc0-163a6ae9e154.png)
# 
# - Not as high as patching.

# In[22]:


loadvars(base_sim_dir+'/full_zero_fraction_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.Oranges
color = colormap(np.linspace(0, 1,n))
zero_fraction_plot_color=color[2]

#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2,,SF_Var_recovery_rate_μ,SF_Var_recovery_rate_σ =glasses_result        

zero_fraction_μ=-recovery_rate_μ
zero_fraction_σ=2*recovery_rate_σ

# best case
idx=argmax(glasses_μ)
max_zero_fraction=zero_fraction_μ[idx]+zero_fraction_σ[idx]
min_zero_fraction=zero_fraction_μ[idx]-zero_fraction_σ[idx]
print(min_zero_fraction,max_zero_fraction)


errorbar(noise,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color=zero_fraction_plot_color) # positive = recovery
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
xlabel('Open-Eye Noise Level')
title('Zero Fraction Treatment')



# In[ ]:


matplotlib.cm.

