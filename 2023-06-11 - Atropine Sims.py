#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[2]:


from treatment_sims_2023_06_02 import *


# In[3]:


base='sims/2023-06-11'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# ## Just do the atropine
# 
# ## blur 6, mu_c 3, sigma_c=1  vs mu_c (0 to 20) and sigma_c (0 to 6)
# 

# In[4]:


rf_size=19
eta=1e-6
number_of_neurons=25
number_of_processes=4

Nsim=21
mu_c_mat=array( list(linspace(0,20,Nsim))+ [3]*Nsim )
sigma_c_mat=array([1]*Nsim + list(linspace(0,6,Nsim)))
blur_mat=array([6]*Nsim*2)
blur_mat,mu_c_mat,sigma_c_mat



# In[10]:


from collections import namedtuple
params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0
eta_count=0
open_eye_noise=0.1


for mu_c,sigma_c,blur in zip(mu_c_mat,sigma_c_mat,blur_mat):
    all_params.append(params(count=count,
                 eta=eta,
                 noise=open_eye_noise,
                     blur=blur,
                 number_of_neurons=number_of_neurons,
     sfname=f'{base}/atropine {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {open_eye_noise:.1f} noise.asdf',
                mu_c=mu_c,sigma_c=sigma_c))

    count+=1

for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# In[11]:


blur


# ## Functions for atropine

# In[12]:


def run_one_continuous_blur_jitter(params,
                                    overwrite=False,
                                 run=True):
    import plasticnet as pn
    count,eta,noise,blur,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.noise,
                        params.blur,
                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    deficit_base_sim=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {4} blur.asdf'
    
    seq=pn.Sequence()
    seq+=treatment_jitter(blur=blur,
                   noise=0.1,
                   noise2=noise,  # treated (strong-eye) noise
                   total_time=8*day,number_of_neurons=params.number_of_neurons,
                    mu_c=mu_c,sigma_c=sigma_c,
                   eta=eta,
                   save_interval=20*minute)
    

    if run:
        seq_load(seq,deficit_base_sim)    
        seq.run(display_hash=False)
        pn.save(sfname,seq) 

    return sfname
        


# In[13]:


func=run_one_continuous_blur_jitter


# ## Premake the image files

# In[9]:


base_image_file='asdf/bbsk081604_all_scale2.asdf'
print("Base Image File:",base_image_file)


# somehow calling these functions breaks the multiprocessing?
# Process ForkPoolWorker-1:
#_pickle.UnpicklingError: NEWOBJ class argument isn't a type object


# so run this, then restart kernel, and then skip this cell
# until I can figure out why this does this
for params in all_params:
    func(params,overwrite=False,run=False)


# In[14]:


do_params=make_do_params(all_params)
len(do_params)


# In[22]:


get_ipython().run_cell_magic('time', '', 'print(func.__name__)\nfunc(all_params[0],overwrite=True)')


# In[16]:


real_time=3*60+ 46


# In[17]:


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


# In[18]:


if do_params:
    pool = Pool(processes=number_of_processes)
    async_results = [pool.apply_async(func, args=(p,),kwds={'overwrite':False,'run':True}) 
                             for p in do_params]
    results =[_.get() for _ in async_results]
    
results


# ## View the sims

# In[19]:


sfname=all_params[0].sfname
R=Results(sfname)


# In[20]:


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


# In[22]:


plot(R.t,R.y[:,0,strong_i],'b',label='Fellow Eye')
plot(R.t,R.y[:,0,weak_i],'m',label='Amblyopic Eye')

for n in range(number_of_neurons):
    plot(R.t,R.y[:,n,0],'m')
    plot(R.t,R.y[:,n,1],'b')
    
    
ylabel('Response')
legend()
print(sfname)
reformat_time_axis()   


# In[21]:


RR={}
count=0
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[23]:


v=np.flip(linspace(0.3,1,Nsim))

count=0
idx=0
s=Storage()
for mu_c,sigma_c in zip(mu_c_mat[:Nsim],sigma_c_mat[:Nsim]):
    params=all_params[count]
    count+=1
    R=RR[params.sfname]
    μ1,μ2=R.μσ[0][0]
    σ1,σ2=R.μσ[1][0]

    s+=mu_c,sigma_c,μ1,μ2,σ1,σ2


mu_c,sigma_c,μ1,μ2,σ1,σ2=s.arrays()

idx=2
errorbar(mu_c,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[idx]))
errorbar(mu_c,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[idx]))
    
xlabel('$\mu_c$ [pixels]')
ylabel('Maximum Response')
legend()    


# In[24]:


v=np.flip(linspace(0.3,1,Nsim))

count=0
idx=0
s=Storage()
count=Nsim
for mu_c,sigma_c in zip(mu_c_mat[Nsim:],sigma_c_mat[Nsim:]):
    params=all_params[count]
    count+=1
    R=RR[params.sfname]
    μ1,μ2=R.μσ[0][0]
    σ1,σ2=R.μσ[1][0]

    s+=mu_c,sigma_c,μ1,μ2,σ1,σ2


mu_c,sigma_c,μ1,μ2,σ1,σ2=s.arrays()

idx=2
errorbar(sigma_c,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[idx]))
errorbar(sigma_c,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[idx]))
    
xlabel('$\sigma_c$ [pixels]')
ylabel('Maximum Response')
legend()    


# In[25]:


figure(figsize=(25,16))
subplot(2,2,1)

v=np.flip(linspace(0.3,1,Nsim))

count=0
idx=0
s=Storage()
for mu_c,sigma_c in zip(mu_c_mat[:Nsim],sigma_c_mat[:Nsim]):
    params=all_params[count]
    count+=1
    R=RR[params.sfname]
    μ,σ=μσ(R.ODI[-1])

    s+=mu_c,sigma_c,μ,σ
    

mu_c,sigma_c,μ,σ=s.arrays()

idx=13
errorbar(mu_c,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.viridis(v[idx]))
    
xlabel('$\\mu_c$ [pixels]')
ylabel(r'$\longleftarrow$ Amblyopic Eye              Fellow Eye $\longrightarrow$'+"\nODI")
ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

#savefig('fig-deficit-response-blur.png')

#========================================================
subplot(2,2,2)

v=np.flip(linspace(0.3,1,Nsim))

count=0
idx=0
s=Storage()
#count=Nsim
for mu_c,sigma_c in zip(mu_c_mat[:Nsim],sigma_c_mat[:Nsim]):
    params=all_params[count]
    count+=1
    R=RR[params.sfname]
    t=R.t/day
    μ,σ=recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  

    s+=mu_c,sigma_c,μ,σ
    

mu_c,sigma_c,μ,σ=s.arrays()

idx=13
errorbar(mu_c,-μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.viridis(v[idx]))
    
xlabel(r'$\mu_c$ [pixels]')
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
#savefig('fig-deficit-response-blur.png')

#========================================================
subplot(2,2,3)

v=np.flip(linspace(0.3,1,Nsim))

count=0
idx=0
s=Storage()
count=Nsim
for mu_c,sigma_c in zip(mu_c_mat[Nsim:],sigma_c_mat[Nsim:]):
    params=all_params[count]
    count+=1
    R=RR[params.sfname]
    μ,σ=μσ(R.ODI[-1])

    s+=mu_c,sigma_c,μ,σ
    

mu_c,sigma_c,μ,σ=s.arrays()

idx=13
errorbar(sigma_c,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.plasma(v[idx]))
    
xlabel('$\\sigma_c$ [pixels]')
ylabel(r'$\longleftarrow$ Amblyopic Eye              Fellow Eye $\longrightarrow$'+"\nODI")
ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

#savefig('fig-deficit-response-blur.png')

#========================================================
subplot(2,2,4)


v=np.flip(linspace(0.3,1,Nsim))

count=0
idx=0
s=Storage()
count=Nsim
for mu_c,sigma_c in zip(mu_c_mat[Nsim:],sigma_c_mat[Nsim:]):
    params=all_params[count]
    count+=1
    R=RR[params.sfname]
    t=R.t/day
    μ,σ=recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  

    s+=mu_c,sigma_c,μ,σ
    

mu_c,sigma_c,μ,σ=s.arrays()

idx=13
errorbar(sigma_c,-μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.plasma(v[idx]))
    
xlabel(r'$\sigma_c$ [pixels]')
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
#savefig('fig-deficit-response-blur.png')

plt.text(.05, 0.9, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.9, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.05, 0.5, "C", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.5, "D", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('fig-patch-ODI-recovery-rate-mu-sigma.png')    


# In[ ]:





# In[ ]:




