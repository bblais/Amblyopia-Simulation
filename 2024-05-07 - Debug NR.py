#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[ ]:


from deficit_defs import *


# In[ ]:


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


# In[ ]:


blur=[-1,-1]
noise=[0.1,0.1]
rf_size=19
number_of_neurons=2
mu_c=0
sigma_c=0    
mu_r=0
sigma_r=0
total_time=8*day
save_interval=1*hour
_debug=False

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
pre1=pn.neurons.natural_images(images[0],
                                            rf_size=rf_size,
                                            time_between_patterns=dt,
                                            verbose=False)

pre2=pn.neurons.natural_images(images[0],
                                            rf_size=rf_size,
                                            other_channel=pre1,
                                            time_between_patterns=dt,
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

seq=pn.Sequence()

seq+=sim,[pre,post],[c]

seq.run(display_hash=True)


# In[ ]:


base='sims/'
sfname='sims/_debug_nr.asdf'
pn.save(sfname,seq)


# In[ ]:


R=Results(sfname)


# In[ ]:


import cycler

color = cm.viridis(np.linspace(0, 1,number_of_neurons))


for n in range(number_of_neurons):
    plot(R.t,R.y[:,n,0],'-',color=color[n])
    plot(R.t,R.y[:,n,1],'--',color=color[n])
    
    
ylabel('Response')
legend()
print(sfname)
reformat_time_axis()  


# In[ ]:


R.plot_rf()


# In[ ]:


ls asdf


# In[ ]:


pre=pn.neurons.natural_images('asdf/bbsk081604_all_dog.asdf',rf_size=13,verbose=False)
post=pn.neurons.linear_neuron(1)
post+=pn.neurons.process.min_max(0,500)

c=pn.connections.BCM(pre,post,[-.05,.05])
c.eta=5e-6
c.tau=1000

sim=pn.simulation(1000*1000)
sim.monitor(c,['weights','theta'],1000)

pn.run_sim(sim,[pre,post],[c],display_hash=False)

pn.utils.plot_rfs_and_theta(sim,[pre,post],[c])


# In[ ]:


pre1=pn.neurons.natural_images('asdf/bbsk081604_all_dog.asdf',
                               rf_size=13,verbose=False)

pre2=pn.neurons.natural_images('asdf/bbsk081604_all_dog.asdf',
                               rf_size=13,
                               other_channel=pre1,
                              verbose=False)


sigma=0.1
pre1+=pn.neurons.process.add_noise_normal(0,sigma)

sigma=0.1
pre2+=pn.neurons.process.add_noise_normal(0,sigma)

pre=pre1+pre2  # make a channel


post=pn.neurons.linear_neuron(1)
post+=pn.neurons.process.min_max(0,500)

c=pn.connections.BCM(pre,post,[-.05,.05])
c.eta=5e-6
c.tau=1000

sim=pn.simulation(1000*100)
sim.monitor(c,['weights','theta'],1000)

pn.run_sim(sim,[pre,post],[c],display_hash=False)

pn.utils.plot_rfs_and_theta(sim,[pre,post],[c]);


# In[ ]:


pre1=pn.neurons.natural_images('asdf/bbsk081604_all_dog.asdf',
                               rf_size=13,verbose=True)

pre2=pn.neurons.natural_images('asdf/bbsk081604_all_dog.asdf',
                               rf_size=13,
                               other_channel=pre1,
                              verbose=True)


sigma=0.1
pre1+=pn.neurons.process.add_noise_normal(0,sigma)

sigma=0.1
pre2+=pn.neurons.process.add_noise_normal(0,sigma)

pre=pre1+pre2  # make a channel


post=pn.neurons.linear_neuron(1)
post+=pn.neurons.process.min_max(0,500)

c=pn.connections.BCM(pre,post,[-.05,.05])
c.eta=5e-6
c.tau=1000

sim=pn.simulation(26)
sim.monitor(c,['weights','theta'],1000)
sim.monitor(pre,['output'],1)


pn.run_sim(sim,[pre,post],[c],display_hash=False)

pn.utils.plot_rfs_and_theta(sim,[pre,post],[c]);


# In[ ]:


m=sim.monitors['output']
t,X=m.arrays()


# In[ ]:


pre2.other


# In[ ]:


for i in range(25):
    subplot(5,5,i+1)
    grid(False)
    gca().set_xticks([])
    gca().set_yticks([])
    imshow(im[i+1],cmap='gray')


# In[ ]:


dir(pre1)


# In[ ]:


pre2.use_other_channel


# In[ ]:




