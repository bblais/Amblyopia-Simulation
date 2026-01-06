#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylab import *


# In[2]:


from deficit_defs_2025_02_25 import *


# In[3]:


old_base_sim_dir=f"sims-2025-05-31 mu_c sigma_c blur"
base_sim_dir=f"sims-2025-06-03 mu_c"
if not os.path.exists(base_sim_dir):
    print("new")
    os.mkdir(base_sim_dir)
print(base_sim_dir)


# In[4]:


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


# In[5]:


def deficit(blur=[2.5,-1],noise=[0.1,0.1],rf_size=19,
            eta=2e-6,
           number_of_neurons=10,
            mu_c=0,sigma_c=0,    
            mu_r=0,sigma_r=0,
           total_time=8*day,
           save_interval=1*hour):


    print("sigma_c",sigma_c)

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


@ray.remote
def run_one_deficit(params,run=True,overwrite=False):
    import plasticnet as pn
    noise,blur,mu_c,mu_r,sigma_c,sigma_r,number_of_neurons,sfname=(
                                params.noise,
                                params.blur,
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
                 noise=noise,blur=blur,
                 mu_c=mu_c,sigma_c=sigma_c,
                mu_r=mu_r,sigma_r=sigma_r,) 


    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 

    return sfname



# In[6]:


number_of_neurons=20
eta=1e-6
number_of_processes=8
ray.init(num_cpus=number_of_processes)


# In[7]:


def doit(p2):
    p=Struct()
    p.number_of_neurons=20
    p.mu_c=0
    p.sigma_c=0
    p.mu_r=0
    p.sigma_r=0
    p.blur=[-1,-1]
    p.eta=1e-6

    p.noise=(.1,.1)

    p.update(p2)

    p.sfname=base_sim_dir+f"/mu_c={p.mu_c} sigma_c={p.sigma_c} blur={p.blur[0]}.asdf"

    results = [run_one_deficit.remote(p,overwrite=True)]
    sfnames=ray.get(results)

    sfname=sfnames[0]
    R=Results(sfname)

    subR=Struct()
    subR.t=R.t/day
    t=subR.t

    subR.y=R.y
    subR.θ=R.θ

    #idx1,idx2=[_[1] for _ in R.sequence_index]

    # subR.ODI=R.ODI
    subR.ORI=R.ORI
    subR.theta_mat=R.theta_mat
    subR.k_mat=R.k_mat
    #subR.LSFV=R.LSFV
    subR.SF_Var=R.SF_Var
    subR.max_SF=R.max_SF
    subR.sfname=sfname
    subR.sequence_index=R.sequence_index
    subR.idx=[0]+[_[-1] for _ in R.sequence_index]
    subR.W=R.W[subR.idx,::]

    figure()
    ims=R.W_image()

    for c in range(2):
        subplot(1,2,c+1)
        imshow(ims[c],cmap=cm.gray)
        grid(None)
        axis('off')

    figure()
    plot(R.t,R.θ);

    figure()
    hist(R.ODI[0,:])
    hist(R.ODI[-1,:],alpha=0.1)

    figure()
    plot(R.ODI);

    return R


# ## Running one with $\mu_c=0$ with blur -1, 0, and 1

# In[8]:


p=Struct()
p.blur=[-1,-1]
p.mu_c=0
p.sigma_c=0

R=doit(p)


# In[9]:


p=Struct()
p.blur=[0,-1]
p.mu_c=0
p.sigma_c=0

R=doit(p)


# In[10]:


p=Struct()
p.blur=[1,-1]
p.mu_c=0
p.sigma_c=0

R=doit(p)


# In[11]:


p=Struct()
p.blur=[2,-1]
p.mu_c=0
p.sigma_c=0

R=doit(p)


# ## What is the difference between blur -1 and 0?  None really.

# In[12]:


blur=-1

if blur<0:
    im=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
else:
    im=pi5.filtered_images(base_image_file,
                                    {'type':'blur','size':blur},
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
im=pi5.asdf_load_images(im)
im_1=[_*im['im_scale_shift'][0]+im['im_scale_shift'][1] for _ in im['im']]
[(_.std(),_.mean()) for _ in im_1]


# In[13]:


blur=0

if blur<0:
    im=pi5.filtered_images(base_image_file,
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
else:
    im=pi5.filtered_images(base_image_file,
                                    {'type':'blur','size':blur},
                                    {'type':'norm'},
                                    {'type':'dog','sd1':1,'sd2':3},   
                                    verbose=False,
                                    )
im=pi5.asdf_load_images(im)
im_0=[_*im['im_scale_shift'][0]+im['im_scale_shift'][1] for _ in im['im']]
[(_.std(),_.mean()) for _ in im_0]


# In[14]:


subplot(2,1,1)
imshow(im_1[0])
colorbar()
title('No Blur')

subplot(2,1,2)
imshow(im_0[0])
colorbar()
title('Blur=0')


# In[15]:


imshow(im_1[0]-im_0[0])
colorbar()


# ## Why does blur=1 cause an OD shift toward the amblyopic eye but blur=2 causes an OD shift toward the fellow eye?
# 
# No idea, but the threshold is higher in the blur=1 case

# In[16]:


p=Struct()
p.blur=[1,-1]
p.mu_c=0
p.sigma_c=0

R1=doit(p)


# In[17]:


p=Struct()
p.blur=[2.5,-1]
p.mu_c=0
p.sigma_c=0

R2=doit(p)


# ## Maybe the difference is reduced with sigma_c>0?

# In[18]:


p=Struct()
p.blur=[1,-1]
p.mu_c=0
p.sigma_c=2

R1=doit(p)


# In[19]:


p=Struct()
p.blur=[2.5,-1]
p.mu_c=0
p.sigma_c=2

R2=doit(p)


# In[20]:


p=Struct()
p.blur=[3,-1]
p.mu_c=0
p.sigma_c=2

R2=doit(p)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




