#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[2]:


from deficit_defs import *


# In[3]:


def savefig(base):
    import matplotlib.pyplot as plt
    for fname in [f'Manuscript/resources/{base}.png',f'Manuscript/resources/{base}.svg']:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


# In[4]:


_debug = False
if _debug:
    print("Debugging")


# In[5]:


base='sims/2023-02-07'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# In[6]:


rf_size=19
#eta_mat=linspace(1e-7,5e-6,11)
eta=2e-6
blur_mat=linspace(0,8,17)
mu_c_mat=[0.0,7.5,15]
sigma_c_mat=linspace(0,6,7)

print(blur_mat)
print(mu_c_mat)
print(sigma_c_mat)
number_of_neurons=20
number_of_processes=6


# In[7]:


linspace(0,6,7)


# ## Premake the image files

# In[8]:


blur_mat=linspace(0,8,17)


# In[9]:


base_image_file='asdf/bbsk081604_all.asdf'
print("Base Image File:",base_image_file)

normal_image=pi5.filtered_images(
                                base_image_file,
                                {'type':'dog','sd1':1,'sd2':3},
                                {'type':'norm'},
                                )

for blur in blur_mat:

    Lfname=pi5.filtered_images(
                                base_image_file,
                                {'type':'blur','size':blur},
                                {'type':'dog','sd1':1,'sd2':3},
                                {'type':'norm'},
                                )



# In[10]:


def blur_jitter_deficit(blur=[2.5,-1],
                        noise=[0.1,.1],
                        rf_size=19,eta=2e-6,
                        mu_c=0,sigma_c=1,    
                        mu_r=0,sigma_r=1,
                        number_of_neurons=10,
                        total_time=8*day,
                        save_interval=1*hour):

    
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


# In[11]:


def run_one(params,overwrite=False):
    import plasticnet as pn
    count,eta,noise,blur,number_of_neurons,sfname,mu_c,sigma_c=(params.count,params.eta,params.noise,params.blur,
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
    
    seq+=blur_jitter_deficit(blur=[blur,-1],
                                total_time=t,
                                noise=noise,eta=eta,number_of_neurons=number_of_neurons,
                                mu_c=mu_c,sigma_c=sigma_c,
                                save_interval=ts)

    
    seq.run(display_hash=False)
    pn.save(sfname,seq) 
    
    return sfname
    


# In[12]:


real_time=20*60+ 39


# In[13]:


from collections import namedtuple
params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0
eta_count=0
noise_count=0
open_eye_noise=0.0

for mu_count,mu_c in enumerate(mu_c_mat):
    for sigma_count,sigma_c in enumerate(sigma_c_mat):
        for blur_count,blur in enumerate(blur_mat):
            all_params.append(params(count=count,
                             eta=eta,
                                 blur=blur,
                             noise=open_eye_noise,
                             number_of_neurons=number_of_neurons,
                            sfname=f'{base}/deficit %d neurons dog %d eta %d noise %d blur %d mu_c %d sigma_c.asdf' % 
                                     (number_of_neurons,eta_count,noise_count,blur_count,mu_c,sigma_c),
                            mu_c=mu_c,
                                sigma_c=sigma_c,
                                    ))

            count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))

print(time2str(real_time*len(all_params)/number_of_processes))


# In[14]:


do_params=make_do_params(all_params)
print(time2str(real_time*len(do_params)/number_of_processes))
len(do_params)


# In[15]:


# %%time
# run_one(all_params[0],overwrite=True)


# In[16]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(run_one, do_params)
    print(result.get())


# ## View the sims

# In[18]:


sfname=all_params[0].sfname

sfname=[params for params in all_params if
    params.sigma_c==0 and
    params.mu_c==7.5 and
    params.blur==0.0
][0].sfname
sfname


# In[19]:


R=Results(sfname)


# In[20]:


R.μσ


# In[21]:


R.t[-1]/day


# In[22]:


t,y,θ,W=R[16*day]


# In[23]:


t,y,θ,W=R[16*day]
vmin=W.min()
vmax=W.max()

w_im=R.weight_image(W)
count=1
for n in range(4):
    for c in range(2):
        subplot(4,2,count)
        pcolormesh(w_im[n,c,...],cmap=py.cm.gray,
                        vmin=vmin,vmax=vmax)
        ax2=gca()
        ax2.set_aspect('equal')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.xaxis.set_ticks_position('none') 
        ax2.yaxis.set_ticks_position('none') 
        
        count+=1


# In[24]:


pwd


# In[25]:


from glob import glob


# In[26]:


# sims/2023-01-27/deficit 20 neurons dog 0 eta 0 noise 0 blur 0 mu_c 1 sigma_c.asdf'
glob("sims/2023-01-27/deficit 20 neurons dog 0 eta 0 noise 0*")


# In[27]:


RR={}
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[28]:


count=0
for mu_count,mu_c in tqdm(enumerate(mu_c_mat)):
    for sigma_count,sigma_c in enumerate(sigma_c_mat):
        s=Storage()
        for blur_count,blur in enumerate(blur_mat):
            params=all_params[count]
            count+=1
            R=RR[params.sfname]
            blur=params.blur
            μ1,μ2=R.μσ[0][0]
            σ1,σ2=R.μσ[1][0]

            s+=blur,μ1,μ2,σ1,σ2


        blur,μ1,μ2,σ1,σ2=s.arrays()

        figure()
        errorbar(blur,μ1,yerr=2*σ1,marker='o',elinewidth=1,label='Deprived')
        errorbar(blur,μ2,yerr=2*σ2,marker='s',elinewidth=1,label='Normal')
        xlabel('Blur Size [pixels]')
        ylabel('Maximum Response')
        title(f'μc={mu_c},σc={sigma_c}')
        legend() 


# In[33]:


from mpl_toolkits.axes_grid1 import make_axes_locatable

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

Blues2 = truncate_colormap(cm.Blues, 0.3, 1.0).reversed()
Oranges2 = truncate_colormap(cm.Oranges, 0.3, 1.0).reversed()

v=np.flip(linspace(0.3,1,len(sigma_c_mat)))

count=0
for mu_count,mu_c in tqdm(enumerate(mu_c_mat)):
    figure()
    for sigma_count,sigma_c in enumerate(sigma_c_mat):
        s=Storage()
        for blur_count,blur in enumerate(blur_mat):
            params=all_params[count]
            count+=1
            R=RR[params.sfname]
            blur=params.blur
            μ1,μ2=R.μσ[0][0]
            σ1,σ2=R.μσ[1][0]

            s+=blur,μ1,μ2,σ1,σ2


        blur,μ1,μ2,σ1,σ2=s.arrays()

        if sigma_count==0:
            errorbar(blur,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Deprived',color=cm.Oranges(v[sigma_count]))
            errorbar(blur,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Normal',color=cm.Blues(v[sigma_count]))
        else:
            errorbar(blur,μ1,yerr=2*σ1,marker='o',elinewidth=1,color=cm.Oranges(v[sigma_count]))
            errorbar(blur,μ2,yerr=2*σ2,marker='s',elinewidth=1,color=cm.Blues(v[sigma_count]))
        xlabel('Blur Size [pixels]')
        ylabel('Maximum Response')

    legend()
    title(r'$\mu_c=%.1f$' % mu_c)

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)   
    ax_cb.grid(False)
    ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
    ax_cb2.grid(False)
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,norm=mpl.colors.Normalize(vmin=sigma_c_mat[0], vmax=sigma_c_mat[-1]),orientation='vertical')
    cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,norm=mpl.colors.Normalize(vmin=sigma_c_mat[0], vmax=sigma_c_mat[-1]),orientation='vertical')
    plt.gcf().add_axes(ax_cb)
    ax_cb.grid(True)
    ax_cb.set_yticklabels([])
    ax_cb2.grid(True)
    title(r'$\sigma_c$')
    plt.gcf().add_axes(ax_cb2)
    title(r'$\sigma_c$')

    savefig(f'fig-deficit-sigma_c-blur-mu_c-{mu_c:.0f}')


# In[37]:


a=5.6
f'{a:.0f}'


# In[38]:


R=RR[params.sfname]
μ,σ=μσ(R.ODI[-1])
μ,σ


# In[ ]:





# In[41]:


count=0
v=np.flip(linspace(0.3,1,len(sigma_c_mat)))

count=0
for mu_count,mu_c in tqdm(enumerate(mu_c_mat)):
    figure()
    for sigma_count,sigma_c in enumerate(sigma_c_mat):
        s=Storage()
        for blur_count,blur in enumerate(blur_mat):
            params=all_params[count]
            count+=1
            R=RR[params.sfname]
            blur=params.blur
            μ,σ=μσ(R.ODI[-1])

            s+=blur,μ,σ


        blur,μ,σ=s.arrays()
        errorbar(blur,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Oranges(v[sigma_count]))    
        xlabel('Blur Size [pixels]')
        ylabel(r'$\longleftarrow$ Weak Eye              Strong Eye $\longrightarrow$'+"\nODI")
        ylim([-1,1])

    title(r'$\mu_c=%.1f$' % mu_c)
        
    divider = make_axes_locatable(plt.gca())
    ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
    ax_cb2.grid(False)
    cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,norm=mpl.colors.Normalize(vmin=sigma_c_mat[0], vmax=sigma_c_mat[-1]),orientation='vertical')
    ax_cb2.grid(True)
    plt.gcf().add_axes(ax_cb2)
    title(r'$\sigma_c$')

    savefig(f'fig-deficit-ODI-sigma_c-blur-mu_c-{mu_c:.1f}')


# In[ ]:




