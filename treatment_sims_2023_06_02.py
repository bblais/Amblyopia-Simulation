#!/usr/bin/env python
# coding: utf-8

# In[ ]:


_debug = False
if _debug:
    print("Debugging")


# In[ ]:


from pylab import *


# In[ ]:


from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:


weak_i=0
strong_i=1


# In[ ]:


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

Blues2 = truncate_colormap(cm.Blues, 0.3, 1.0).reversed()
Oranges2 = truncate_colormap(cm.Oranges, 0.3, 1.0).reversed()


# In[1]:


def mydisplay(t,sim,neurons,connections):
    global _fig
    from IPython.display import display, clear_output
    from pylab import figure,close,gcf,imshow,subplot,plot,ones,cm,colorbar,axis

    c=connections[0]

    weights=c.weights
    num_neurons=len(weights)
    rf_size=neurons[0][0].rf_size
    num_channels=len(neurons[0])

    W=weights.reshape((num_neurons,
                        num_channels,
                        rf_size,rf_size))

    vmin,vmax=W.min(),W.max()

    #W=(W-vmin)/(vmax-vmin)

    blocks=[]
    for row in range(num_neurons):

        block_row=[]
        for col in range(num_channels):
            rf=W[row,col,:,:]
            block_row.append(rf)
            if col<num_channels-1:
                block_row.append(vmax*ones((rf_size,1)))
        blocks.append(block_row)

        if row<num_neurons-1:
            block_row=[]
            for col in range(num_channels):
                rf=W[row,col,:,:]
                block_row.append(vmax*ones((2,rf_size)))
                if col<num_channels-1:
                    block_row.append(vmax*ones((2,1)))
            blocks.append(block_row)


    im=np.block(blocks)

    t,θ=sim.monitors['theta'].arrays()

    try:
        clear_output(wait=True)
    
        _fig=figure(1)
        
        subplot(1,2,1)
        imshow(im,cmap=cm.gray)
        axis('off')
        colorbar(location='left')
        
        subplot(2,2,2)
        plot(t,θ)
        subplot(2,2,4)
        
        try:
            pp=sim.post_process[0]
            t=pp.t
            y=pn.utils.max_channel_response(pp.responses)
            y=y.transpose([2,1,0]) 
            plot(t,y[:,:,0],'m-')
            plot(t,y[:,:,1],'b-')        
        except AttributeError:
            pass
        
        _fig.canvas.draw_idle()
        display(_fig)
        close(_fig)
    except KeyboardInterrupt:
        close(_fig)
        raise
        


# In[ ]:


def to_named_tuple(params_list):
    from collections import namedtuple
    keys=list(params_list[0].keys())
    keys+=['count']
    params=namedtuple('params',keys)

    tuples_list=[]
    for count,p in enumerate(params_list):
        p2=params(count=count,
                  **p)
        tuples_list.append(p2)


    return tuples_list


# In[ ]:


from deficit_defs import *


# In[ ]:


def blur_jitter_deficit(blur=[2.5,-1],
                        noise=[0.1,.1],
                        rf_size=19,eta=2e-6,
                        mu_c=0,sigma_c=1,    
                        mu_r=0,sigma_r=1,
                        number_of_neurons=10,
                        total_time=8*day,
                        save_interval=1*hour):

    base_image_file='asdf/bbsk081604_all_scale2.asdf'
    if _debug:
        total_time=1*minute
        save_interval=1*second
        
    images=[]
    dt=200*ms
    
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
    pre1=pn.neurons.natural_images_with_jitter(images[0],
                                                rf_size=rf_size,
                                                time_between_patterns=dt,
                                                sigma_r=1,
                                                sigma_c=1,
                                                verbose=False)

    pre2=pn.neurons.natural_images_with_jitter(images[1],
                                                rf_size=rf_size,
                                                other_channel=pre1,
                                                time_between_patterns=dt,
                                                mu_r=mu_r,mu_c=mu_c,
                                                sigma_r=sigma_r,sigma_c=sigma_c,
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
    sim.dt=dt

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]


# In[ ]:


def fix_jitter(noise=0.1,rf_size=19,
           number_of_neurons=10,
            mu_c=0,sigma_c=0,    
            mu_r=0,sigma_r=0,
           total_time=8*day,
           save_interval=1*hour,
           eta=2e-6):
    
    if _debug:
        total_time=1*minute
        save_interval=1*second

    base_image_file='asdf/bbsk081604_all_scale2.asdf'
    im=pi5.filtered_images(
                        base_image_file,
                        {'type':'dog','sd1':1,'sd2':3},
                        {'type':'norm'},
                                        verbose=False,
                        )
    
    dt=200*ms        
    pre1=pn.neurons.natural_images_with_jitter(im,
                                                rf_size=rf_size,
                                                time_between_patterns=dt,
                                                sigma_r=0,
                                                sigma_c=0,
                                                verbose=False)
    pre2=pn.neurons.natural_images_with_jitter(im,
                                                rf_size=rf_size,
                                                other_channel=pre1,
                                                time_between_patterns=dt,
                                                mu_r=mu_r,mu_c=mu_c,
                                                sigma_r=sigma_r,sigma_c=sigma_c,
                                                verbose=False)


    sigma=noise
    pre1+=pn.neurons.process.add_noise_normal(0,sigma)

    sigma=noise
    pre2+=pn.neurons.process.add_noise_normal(0,sigma)

    pre=pre1+pre2

    post=default_post(number_of_neurons)
    c=default_bcm(pre,post)
    c.eta=eta

    save_interval=save_interval

    sim=pn.simulation(total_time)

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]


# In[ ]:


def treatment_jitter(contrast=1,noise=0.1,noise2=0.1,
              rf_size=19,eta=5e-6,
              f=30,  # size of the blur for mask, which is a measure of overlap
            mu_c=0,sigma_c=0,    
            mu_r=0,sigma_r=0,
           number_of_neurons=20,
           total_time=8*day,
           save_interval=1*hour,
             mask=None,
             blur=0):
    
    if _debug:
        total_time=1*minute
        save_interval=1*second
    base_image_file='asdf/bbsk081604_all_scale2.asdf'
    
    dt=200*ms        
    
    if not f in [10,30,50,70,90]:
        raise ValueError("Unknown f %s" % str(f))

        
#     print("contrast,noise,noise2,rf_size,eta,f,mu_c,sigma_c,mu_r,sigma_r,number_of_neurons,total_time,save_interval,mask,blur",
#           contrast,noise,noise2,
#               rf_size,eta,
#               f,  # size of the blur for mask, which is a measure of overlap
#             mu_c,sigma_c,    
#             mu_r,sigma_r,
#            number_of_neurons,
#            total_time,
#            save_interval,
#              mask,
#              blur)
        
        
        
    if mask:
        if blur:
            maskA_fname=pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':'bblais-masks-20230602/2023-06-02-*-A-fsig%d.png'% f, 
                                        'seed':101},
                                        {'type':'blur','size':bv},
                                        {'type':'dog','sd1':1,'sd2':3},
                                        {'type':'norm'},
                                            verbose=False,
                                      )
            maskF_fname=pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':'bblais-masks-20230602/2023-06-02-*-F-fsig%d.png' % f, 
                                        'seed':101},
                                        {'type':'blur','size':bv},
                                        {'type':'dog','sd1':1,'sd2':3},
                                        {'type':'norm'},
                                            verbose=False,
                                      )            
        else:
            maskA_fname=pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':'bblais-masks-20230602/2023-06-02-*-A-fsig%d.png' % f,
                                        'seed':101},
                                        {'type':'dog','sd1':1,'sd2':3},
                                        {'type':'norm'},                                            
                                            verbose=False,
                                      )
            maskF_fname=pi5.filtered_images(base_image_file,
                                        {'type':'mask',
                                         'name':'bblais-masks-20230602/2023-06-02-*-F-fsig%d.png' % f,
                                        'seed':101},
                                        {'type':'dog','sd1':1,'sd2':3},
                                        {'type':'norm'},                                            
                                            verbose=False,
                                      )
        
        pre1=pn.neurons.natural_images_with_jitter(maskA_fname,
                                                   rf_size=rf_size,
                                                time_between_patterns=dt,
                                                sigma_r=0,
                                                sigma_c=0,
                                    verbose=False)
        pre2=pn.neurons.natural_images_with_jitter(maskF_fname,rf_size=rf_size,
                                    other_channel=pre1,
                                    mu_r=mu_r,mu_c=mu_c,
                                    sigma_r=sigma_r,sigma_c=sigma_c,
                                    verbose=False)
        
    else:
        
        if blur:
            blur_fname=pi5.filtered_images(base_image_file,
                                        {'type':'blur','size':blur},
                                        {'type':'dog','sd1':1,'sd2':3},
                                        {'type':'norm'},                                            
                                            verbose=False,
                                          )
        
        norm_fname=pi5.filtered_images(base_image_file,
                                        {'type':'dog','sd1':1,'sd2':3},
                                        {'type':'norm'},                                            
                                            verbose=False,
                                      )
    
        
        pre1=pn.neurons.natural_images_with_jitter(norm_fname,rf_size=rf_size,
                                                time_between_patterns=dt,
                                                sigma_r=0,
                                                sigma_c=0,
                                    verbose=False)
        
        if blur:
            pre2=pn.neurons.natural_images_with_jitter(blur_fname,rf_size=rf_size,
                                        other_channel=pre1,
                                    mu_r=mu_r,mu_c=mu_c,
                                    sigma_r=sigma_r,sigma_c=sigma_c,
                                        verbose=False)
        else:
            pre2=pn.neurons.natural_images_with_jitter(norm_fname,rf_size=rf_size,
                                        other_channel=pre1,
                                    mu_r=mu_r,mu_c=mu_c,
                                    sigma_r=sigma_r,sigma_c=sigma_c,
                                        verbose=False)
            


    sigma=noise
    pre1+=pn.neurons.process.add_noise_normal(0,sigma)

    sigma=noise2
    pre2+=pn.neurons.process.scale_shift(contrast,0)
    pre2+=pn.neurons.process.add_noise_normal(0,sigma)

    pre=pre1+pre2

    post=pn.neurons.linear_neuron(number_of_neurons)
    post+=pn.neurons.process.sigmoid(-1,50)

    c=pn.connections.BCM(pre,post,[-.1,.1],[.1,.2])
    c.eta=eta
    c.tau=100*second

    save_interval=save_interval

    sim=pn.simulation(total_time)

    sim.dt=dt

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]

def patch_treatment_jitter(noise=0.1,patch_noise=0.1,rf_size=19,
                   number_of_neurons=20,
                    mu_c=0,sigma_c=0,    
                    mu_r=0,sigma_r=0,
                   total_time=8*day,
                   save_interval=1*hour,
                   eta=2e-6,
                   ):
    
    if _debug:
        total_time=1*minute
        save_interval=1*second
    base_image_file='asdf/bbsk081604_all_scale2.asdf'

    im=pi5.filtered_images(
                        base_image_file,
                        {'type':'dog','sd1':1,'sd2':3},
                        {'type':'norm'},
                                        verbose=False,                                            
                        )
    dt=200*ms        
        
    norm_fname=pi5.filtered_images(im,
                                verbose=False
                                  )
    
        
    pre1=pn.neurons.natural_images_with_jitter(norm_fname,rf_size=rf_size,
                                                time_between_patterns=dt,
                                                sigma_r=0,
                                                sigma_c=0,
                                verbose=False)
        
    pre2=pn.neurons.natural_images_with_jitter(norm_fname,rf_size=rf_size,
                                other_channel=pre1,
                                time_between_patterns=dt,
                                mu_r=mu_r,mu_c=mu_c,
                                sigma_r=sigma_r,sigma_c=sigma_c,
                                verbose=False)
            


    sigma=noise
    pre1+=pn.neurons.process.add_noise_normal(0,sigma)

    sigma=patch_noise
    pre2+=pn.neurons.process.scale_shift(0.0,0) # zero out signal
    pre2+=pn.neurons.process.add_noise_normal(0,sigma)

    pre=pre1+pre2

    post=default_post(number_of_neurons)
    c=default_bcm(pre,post)
    c.eta=eta

    save_interval=save_interval

    sim=pn.simulation(total_time)

    sim.dt=dt

    sim.monitor(post,['output'],save_interval)
    sim.monitor(c,['weights','theta'],save_interval)

    sim+=pn.grating_response(print_time=False)

    return sim,[pre,post],[c]

