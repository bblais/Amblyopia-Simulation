#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[3]:


from treatment_sims_2023_06_02 import *


# In[4]:


base='sims/2023-06-11'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# ## blur 4, mu_c 3, sigma_c=1  vs mu_c (0 to 20) and sigma_c (0 to 6)

# In[7]:


rf_size=19
eta=1e-6
number_of_neurons=25
number_of_processes=4

Nsim=21
mu_c_mat=array( list(linspace(0,20,Nsim))+ [3]*Nsim )
sigma_c_mat=array([1]*Nsim + list(linspace(0,6,Nsim)))
blur_mat=array([4]*Nsim*2)
blur_mat,mu_c_mat,sigma_c_mat


# In[10]:


from collections import namedtuple
params = namedtuple('params', ['count', 'eta','noise','blur','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0
eta_count=0
noise_count=0
open_eye_noise=0.1

for mu_c,sigma_c,blur in zip(mu_c_mat,sigma_c_mat,blur_mat):

    all_params.append(params(count=count,
                 eta=eta,
                 noise=open_eye_noise,
                 blur=blur,
                 number_of_neurons=number_of_neurons,
         sfname=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur.asdf',
                        mu_c=mu_c,sigma_c=sigma_c))

    count+=1
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)


# ## Premake the image files

# In[11]:


base_image_file='asdf/bbsk081604_all_scale2.asdf'
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



# In[12]:


def run_one(params,overwrite=False):
    import plasticnet as pn
    count,eta,noise,blur,number_of_neurons,sfname,mu_c,sigma_c=(params.count,params.eta,params.noise,params.blur,
                                        params.number_of_neurons,params.sfname,params.mu_c,params.sigma_c)
    
    if not overwrite and os.path.exists(sfname):
        return sfname
    
    seq=pn.Sequence()

    t=16*day*2
    ts=1*hour

    seq+=blur_jitter_deficit(blur=[blur,-1],
                                total_time=t,
                                noise=noise,eta=eta,number_of_neurons=number_of_neurons,
                                mu_c=mu_c,sigma_c=sigma_c,
                                save_interval=ts)

    
    seq.run(display_hash=False)
    pn.save(sfname,seq) 
    
    return sfname
    


# In[ ]:





# In[9]:


get_ipython().run_cell_magic('time', '', 'run_one(all_params[0],overwrite=True)')


# In[13]:


real_time=13*60+ 19


# In[14]:


do_params=make_do_params(all_params)
len(do_params)
#run_one(do_params[0],overwrite=True)


# In[15]:


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


# In[16]:


if do_params:
    pool = Pool(processes=number_of_processes)
    result = pool.map_async(run_one, do_params)
    print(result.get())


# ## View the sims

# In[17]:


sfname=all_params[0].sfname
R=Results(sfname)


# In[18]:


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


# In[19]:


RR={}
count=0
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[52]:


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

savefig('fig-deficit-response-mu_c.png')


# In[53]:


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

savefig('fig-deficit-response-sigma_c.png')


# In[54]:


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
savefig('fig-deficit-ODI-mu_c.png')


# In[55]:


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
    
xlabel('$\sigma_c$ [pixels]')
ylabel(r'$\longleftarrow$ Weak Eye              Strong Eye $\longrightarrow$'+"\nODI")

savefig('fig-deficit-ODI-sigma_c.png')


# In[63]:


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

#========================================================
subplot(2,2,2)

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



#========================================================
subplot(2,2,3)
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
    μ,σ=μσ(R.ODI[-1])

    s+=mu_c,sigma_c,μ,σ
    

mu_c,sigma_c,μ,σ=s.arrays()

idx=13
errorbar(sigma_c,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.plasma(v[idx]))
    
xlabel('$\sigma_c$ [pixels]')
ylabel(r'$\longleftarrow$ Weak Eye              Strong Eye $\longrightarrow$'+"\nODI")
ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)



plt.text(.05, 0.9, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.9, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.05, 0.5, "C", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.5, "D", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('fig-deficit-response-ODI-mu-sigma.png')    


# In[ ]:




