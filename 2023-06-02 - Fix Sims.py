#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[2]:


from treatment_sims_2023_06_02 import *


# In[10]:


def savefig(origfname):
    base,ext=os.path.splitext(origfname)
    import matplotlib.pyplot as plt
    
    print_fnames=[f'Manuscript/resources/{base}.png',f'Manuscript/resources/{base}.svg']
    if ext:
        if ext!='.png' and ext!='.svg':
            print_fnames+=[f'Manuscript/resources/{origfname}']
    
    for fname in print_fnames:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


# In[4]:


base='sims/2023-06-02'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# ## Just do the noise with no jitter

# In[5]:


rf_size=19
eta=1e-6
number_of_neurons=25
number_of_processes=4
mu_c_mat=[0,7.5,0,7.5]
sigma_c_mat=[0,2,2,0]
blur=4
noise_mat=linspace(0,1,21)
noise_mat


# In[6]:


from collections import namedtuple
params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0
eta_count=0
noise_count=0

for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise

        all_params.append(params(count=count,
                     eta=eta,
                     noise=open_eye_noise,
                     blur=blur,
                     number_of_neurons=number_of_neurons,
             sfname=f'{base}/fix {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {noise:.2f} noise.asdf',
                            mu_c=mu_c,sigma_c=sigma_c))

        count+=1
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[7]:


blur


# ## Premake the image files

# In[8]:


base_image_file='asdf/bbsk081604_all_scale2.asdf'
print("Base Image File:",base_image_file)

normal_image=pi5.filtered_images(
                                base_image_file,
                                {'type':'dog','sd1':1,'sd2':3},
                                {'type':'norm'},
                                )

Lfname=pi5.filtered_images(
                            base_image_file,
                            {'type':'blur','size':blur},
                            {'type':'dog','sd1':1,'sd2':3},
                            {'type':'norm'},
                            )



# ## Functions for Fix

# In[8]:


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
    
    seq+=fix_jitter(total_time=8*day,
             save_interval=20*minute,number_of_neurons=params.number_of_neurons,
            mu_c=mu_c,sigma_c=sigma_c,
             eta=eta,noise=noise)
    seq_load(seq,deficit_base_sim)    

    if run:
        seq.run(display_hash=False)
        pn.save(sfname,seq) 
    
    return sfname


# In[9]:


func=run_one_continuous_fix_jitter


# In[10]:


do_params=make_do_params(all_params)
len(do_params)


# In[12]:


get_ipython().run_cell_magic('time', '', 'print(func.__name__)\nfunc(all_params[0],overwrite=True)')


# In[13]:


real_time=2*60+ 38


# In[14]:


if len(do_params)>13:
    for a in do_params[:5]:
        print(a)
    print("[....]")
    for a in do_params[-5:]:
        print(a)
else:
    for a in do_params:
        print(a)
    

print(len(do_params))

print(time2str(real_time*len(do_params)/number_of_processes))


# In[15]:


if do_params:
    pool = Pool(processes=number_of_processes)
    async_results = [pool.apply_async(func, args=(p,),kwds={'overwrite':False,'run':True}) 
                             for p in do_params]
    results =[_.get() for _ in async_results]
    
results


# ## View the sims

# In[12]:


sfname=all_params[0].sfname
R=Results(sfname)


# In[15]:


t=R.t/day
recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  


# In[16]:


recovery_rate_μ


# In[17]:


t,y,θ,W=R[-1]
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


# In[7]:


RR={}
count=0
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# ![image.png](attachment:aebd2839-95ec-49e3-9ab4-d83fb6e2d3f9.png)

# In[19]:


count=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        μ1,μ2=R.μσ[0][0]
        σ1,σ2=R.μσ[1][0]

        s+=noise,μ1,μ2,σ1,σ2


    noise,μ1,μ2,σ1,σ2=s.arrays()

    figure()
    errorbar(noise,μ1,yerr=2*σ1,marker='o',elinewidth=1,label='Amblyopic')
    errorbar(noise,μ2,yerr=2*σ2,marker='s',elinewidth=1,label='Fellow')
    xlabel('Noise')
    ylabel('Maximum Response')
    title(f'μc={mu_c} σc={sigma_c} blur={blur}')
    legend()    


# In[14]:


v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        μ1,μ2=R.μσ[0][0]
        σ1,σ2=R.μσ[1][0]

        s+=noise,μ1,μ2,σ1,σ2


    noise,μ1,μ2,σ1,σ2=s.arrays()

    if idx==2:
        errorbar(noise,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[idx]))
        errorbar(noise,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[idx]))
    else:
        errorbar(noise,μ1,yerr=2*σ1,marker='o',elinewidth=1,label='_',color=cm.Blues(v[idx]))
        errorbar(noise,μ2,yerr=2*σ2,marker='s',elinewidth=1,label='_',color=cm.Oranges(v[idx]))
        
        
    dy=0.3
    text(-.05,μ1[:1]+dy,f'μ={mu_c} σ={sigma_c}',va='center',ha='right',size=8,color=cm.Blues(v[0]))         
    text(-.05,μ2[:1]-dy,f'μ={mu_c} σ={sigma_c}',va='center',ha='right',size=8,color=cm.Oranges(v[0]))
    
    idx+=1        
    
xlabel('Open-eye Noise')
ylabel('Maximum Response')
legend()    
# gca().set_xticks(range(0,13,2))
xlim([-.2,1.105])

    


# In[18]:


v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        μ,σ=μσ(R.ODI[-1])
        
        s+=noise,μ,σ


    noise,μ,σ=s.arrays()

    errorbar(noise,μ,yerr=2*σ,marker='o',elinewidth=1,label=f'μ={mu_c} σ={sigma_c}',color=cm.viridis(1-v[idx]))
    
    idx+=1
    
xlabel('Open-eye Noise')
ylabel(r'$\longleftarrow$ Weak Eye              Strong Eye $\longrightarrow$'+"\nODI")

ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

legend()    
# gca().set_xticks(range(0,13,2))

    


# In[8]:


figure(figsize=(25,8))
subplot(1,2,1)


v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        μ1,μ2=R.μσ[0][0]
        σ1,σ2=R.μσ[1][0]

        s+=noise,μ1,μ2,σ1,σ2


    noise,μ1,μ2,σ1,σ2=s.arrays()

    if idx==2:
        errorbar(noise,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[idx]))
        errorbar(noise,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[idx]))
    else:
        errorbar(noise,μ1,yerr=2*σ1,marker='o',elinewidth=1,label='_',color=cm.Blues(v[idx]))
        errorbar(noise,μ2,yerr=2*σ2,marker='s',elinewidth=1,label='_',color=cm.Oranges(v[idx]))
        
        
    dy=0.3
    text(-.05,μ1[:1]+dy,f'μ={mu_c} σ={sigma_c}',va='center',ha='right',size=8,color=cm.Blues(v[0]))         
    text(-.05,μ2[:1]-dy,f'μ={mu_c} σ={sigma_c}',va='center',ha='right',size=8,color=cm.Oranges(v[0]))
    
    idx+=1        
    
xlabel('Open-eye Noise')
ylabel('Maximum Response')
legend()    
# gca().set_xticks(range(0,13,2))
xlim([-.2,1.105])

    
    
subplot(1,2,2)
    
v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        μ,σ=μσ(R.ODI[-1])
        
        s+=noise,μ,σ


    noise,μ,σ=s.arrays()

    errorbar(noise,μ,yerr=2*σ,marker='o',elinewidth=1,label=f'μ={mu_c} σ={sigma_c}',color=cm.viridis(1-v[idx]))
    
    idx+=1
    
xlabel('Open-eye Noise')
ylabel(r'$\longleftarrow$ Weak Eye              Strong Eye $\longrightarrow$'+"\nODI")

ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

legend()    
# gca().set_xticks(range(0,13,2))

      

plt.text(.1, 0.9, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.9, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('fig-fix-response-ODI-blur.png')    
    


# In[41]:


figure(figsize=(25,16))
subplot(2,2,1)


v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        μ1,μ2=R.μσ[0][0]
        σ1,σ2=R.μσ[1][0]

        s+=noise,μ1,μ2,σ1,σ2


    noise,μ1,μ2,σ1,σ2=s.arrays()

    if idx==2:
        errorbar(noise,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[idx]))
        errorbar(noise,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[idx]))
    else:
        errorbar(noise,μ1,yerr=2*σ1,marker='o',elinewidth=1,label='_',color=cm.Blues(v[idx]))
        errorbar(noise,μ2,yerr=2*σ2,marker='s',elinewidth=1,label='_',color=cm.Oranges(v[idx]))
        
        
    dy=0.3
    text(-.05,μ1[:1]+dy,f'μ={mu_c} σ={sigma_c}',va='center',ha='right',size=8,color=cm.Blues(v[0]))         
    text(-.05,μ2[:1]-dy,f'μ={mu_c} σ={sigma_c}',va='center',ha='right',size=8,color=cm.Oranges(v[0]))
    
    idx+=1        
    
xlabel('Open-eye Noise')
ylabel('Maximum Response')
legend()    
# gca().set_xticks(range(0,13,2))
xlim([-.2,1.105])

    
    
subplot(2,2,2)
    
v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        μ,σ=μσ(R.ODI[-1])
        
        s+=noise,μ,σ


    noise,μ,σ=s.arrays()

    errorbar(noise,μ,yerr=2*σ,marker='o',elinewidth=1,label=f'μ={mu_c} σ={sigma_c}',color=cm.viridis(1-v[idx]))
    
    idx+=1
    
xlabel('Open-eye Noise')
ylabel(r'$\longleftarrow$ Amblyopic Eye              Fellow Eye $\longrightarrow$'+"\nODI")

ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

legend()    
# gca().set_xticks(range(0,13,2))

      

ax=subplot(2,2,3)
pos=ax.get_position().bounds
ax.set_position([pos[0]+.2,pos[1],pos[2],pos[3]])


v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        t=R.t/day
        recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  
    
        s+=noise,recovery_rate_μ,recovery_rate_σ


    noise,μ,σ=s.arrays()



    errorbar(noise,-μ,yerr=2*σ,marker='o',elinewidth=1,label=f'μ={mu_c} σ={sigma_c}',color=cm.viridis(1-v[idx]))
    
    idx+=1
    
xlabel('Open-eye Noise')
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")



    
    
    
plt.text(.1, 0.9, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.9, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')


plt.text(.23, 0.46, "C", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('fig-fix-response-ODI-blur.png')    
    


# In[30]:


subplot(2,2,1)
subplot(2,2,2)

ax=subplot(2,2,3)
pos=ax.get_position().bounds
ax.set_position([pos[0]+.2,pos[1],pos[2],pos[3]])


# In[28]:


pos=ax.get_position().extents
pos


# In[24]:


pos.bounds


# In[27]:


pos.extents


# In[20]:


ax.set_position([pos[0]+.2,pos[1],pos[2],pos[3])


# In[23]:


v=np.flip(linspace(0.3,1,4))

count=0
idx=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    s=Storage()
    for noise_count,noise in enumerate(noise_mat):
        open_eye_noise=noise
        params=all_params[count]
        count+=1
        R=RR[params.sfname]
        noise=params.noise
        t=R.t/day
        recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  
    
        s+=noise,recovery_rate_μ,recovery_rate_σ


    noise,μ,σ=s.arrays()



    errorbar(noise,-μ,yerr=2*σ,marker='o',elinewidth=1,label=f'μ={mu_c} σ={sigma_c}',color=cm.viridis(1-v[idx]))
    
    idx+=1
    
xlabel('Open-eye Noise')
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")

# xl=xlim()
# plot(xl,[0,0],'k-',lw=2)

# yl=array(ylim())
# ylim([-abs(yl.max()),+abs(yl.max())])

legend()    
# gca().set_xticks(range(0,13,2))

    


# In[20]:


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


# In[ ]:




