#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[2]:


from treatment_sims_2023_06_02 import *


# In[3]:


base='sims/2023-06-02'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# ## Just do the contrast with no jitter

# In[4]:


rf_size=19
eta=1e-6
number_of_neurons=25
number_of_processes=4
mu_c=0
sigma_c=0
blur=4
open_eye_noise=0.1

mu_c_mat=[0,7.5,0,7.5]
sigma_c_mat=[0,2,2,0]
contrast_mat=linspace(0,1,21)  # linspace(0,1,11)
mask_mat=array([0,1])
f_mat=array([10,30,50,70,90])


contrast_mat


# In[5]:


from collections import namedtuple


params = namedtuple('params', ['count', 'eta','blur','contrast','f','mask','number_of_neurons','sfname','mu_c','sigma_c'])
all_params=[]
count=0


for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    for mask in mask_mat:
        if mask:
            for fc,f in enumerate(f_mat):
                for contrast_count,contrast in enumerate(contrast_mat):
                    all_params.append(params(count=count,
                                 eta=eta,
                                     blur=blur,
                                             contrast=contrast,
                                             f=f,
                                             mask=mask,
                                 number_of_neurons=number_of_neurons,
                                sfname=f'{base}/contrast mask {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {contrast:.2f} contrast {mask} mask {f} f.asdf',
                                mu_c=mu_c,sigma_c=sigma_c))

                    count+=1

        else:
            f=10
            for contrast_count,contrast in enumerate(contrast_mat):
                all_params.append(params(count=count,
                             eta=eta,
                                 blur=blur,
                                         contrast=contrast,
                                         f=f,
                                         mask=mask,
                             number_of_neurons=number_of_neurons,
                            sfname=f'{base}/contrast mask {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur {contrast:.2f} contrast {mask} mask {f} f.asdf',
                            mu_c=mu_c,sigma_c=sigma_c))

                count+=1



for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[7]:


blur


# ## Functions for contrast mask

# In[8]:


def run_one_continuous_mask_jitter(params,
                                    overwrite=False,
                                 run=True):
    import plasticnet as pn
    count,eta,blur,contrast,mask,f,mu_c,sigma_c,number_of_neurons,sfname=(params.count,params.eta,params.blur,params.contrast,params.mask,params.f,
                                        params.mu_c,params.sigma_c,params.number_of_neurons,params.sfname)
    
    
    if os.path.exists(sfname):
        if not overwrite:
            return sfname
        else:
            os.remove(sfname)

    
    seq=pn.Sequence()
    deficit_base_sim=f'{base}/deficit {number_of_neurons} neurons {mu_c} mu_c {sigma_c} sigma_c {blur} blur.asdf'

    seq+=treatment_jitter(f=f,
                   mask=mask,
                   contrast=contrast,
                   total_time=8*day,
                   eta=eta,
                          number_of_neurons=number_of_neurons,
                    mu_c=mu_c,sigma_c=sigma_c,
                   save_interval=20*minute)

    if run:
        
        seq_load(seq,deficit_base_sim)    
        
        seq.run(display_hash=False)
        pn.save(sfname,seq) 

    
    return sfname
    
    


# In[9]:


func=run_one_continuous_mask_jitter


# ## Premake the image files

# In[22]:


base_image_file='asdf/bbsk081604_all_scale2.asdf'
print("Base Image File:",base_image_file)


# somehow calling these functions breaks the multiprocessing?
# Process ForkPoolWorker-1:
#_pickle.UnpicklingError: NEWOBJ class argument isn't a type object


# so run this, then restart kernel, and then skip this cell
# until I can figure out why this does this
for params in all_params:
    func(params,overwrite=False,run=False)


# In[21]:


params


# In[10]:


len(all_params)


# ## Run the sims

# In[10]:


do_params=make_do_params(all_params)
len(do_params)


# In[12]:


get_ipython().run_cell_magic('time', '', 'print(func.__name__)\nfunc(all_params[0],overwrite=True)')


# In[13]:


real_time=5*60+ 51


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


# In[ ]:


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


# In[13]:


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


# In[15]:


# RR={}
# count=0
# for params in tqdm(all_params):
#     RR[params.sfname]=Results(params.sfname)


# ![image.png](attachment:53d620ef-06e0-4995-a4a5-a90881f48619.png)

# In[11]:


count=0

v=np.flip(linspace(0.3,1,len(f_mat)))

figcount=0
for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    figure()
    for mask in mask_mat:
        if mask:
            for fc,f in tqdm(enumerate(f_mat),total=len(f_mat)):
                s=Storage()
                for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                    params=all_params[count]
                    count+=1
                    R=Results(params.sfname)
                    #R=RR[params.sfname]
                    contrast=params.contrast
                    μ1,μ2=R.μσ[0][0]
                    σ1,σ2=R.μσ[1][0]

                    s+=contrast,μ1,μ2,σ1,σ2
                    
                contrast,μ1,μ2,σ1,σ2=s.arrays()

                
                if fc==1:
                    errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[fc]))
                    errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[fc]))
                    
                else:
                    errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
                    errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,color=cm.Oranges(v[fc]))
                
        else:
            s=Storage()
            for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                params=all_params[count]
                count+=1
                R=Results(params.sfname)
                #R=RR[params.sfname]
                contrast=params.contrast
                μ1,μ2=R.μσ[0][0]
                σ1,σ2=R.μσ[1][0]

                s+=contrast,μ1,μ2,σ1,σ2

            contrast,μ1,μ2,σ1,σ2=s.arrays()

            errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,color='black')
            errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,color='gray')
            
                
        xlabel('Contrast')
        ylabel('Maximum Response')
        title(f'μc={mu_c} σc={sigma_c} blur={blur}')
        legend()                        
                    
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)   
    ax_cb.grid(False)
    ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
    ax_cb2.grid(False)
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
    cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
    plt.gcf().add_axes(ax_cb)
    ax_cb.grid(True)
    ax_cb.set_yticklabels([])
    ax_cb2.grid(True)
    title(r'$f$')
    plt.gcf().add_axes(ax_cb2)
    title(r'$f$')

    figcount+=1
    
    if figcount==2:
        break


# ![image.png](attachment:18e60ee3-5e3e-45ff-bc30-6c7d1a13aebe.png)

# In[15]:


count=0

v=np.flip(linspace(0.3,1,len(f_mat)))
figcount=0

for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    figure()
    for mask in mask_mat:
        if mask:
            for fc,f in tqdm(enumerate(f_mat),total=len(f_mat)):
                s=Storage()
                for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                    params=all_params[count]
                    count+=1
                    R=Results(params.sfname)
                    #R=RR[params.sfname]
                    contrast=params.contrast
                    μ,σ=μσ(R.ODI[-1])

                    s+=contrast,μ,σ
                    
                contrast,μ,σ=s.arrays()

                
                errorbar(contrast,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
                
        else:
            s=Storage()
            for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                params=all_params[count]
                count+=1
                R=Results(params.sfname)
                #R=RR[params.sfname]
                contrast=params.contrast
                μ,σ=μσ(R.ODI[-1])

                s+=contrast,μ,σ

            contrast,μ,σ=s.arrays()

            errorbar(contrast,μ,yerr=2*σ,marker='o',elinewidth=1,color='black')

            
        ylim([-1,1])
        xl=xlim()
        plot(xl,[0,0],'k-',lw=2)
                
        xlabel('Contrast')
        ylabel(r'$\longleftarrow$ Weak Eye              Strong Eye $\longrightarrow$'+"\nODI")
        title(f'μc={mu_c} σc={sigma_c} blur={blur}')

                    

    divider = make_axes_locatable(plt.gca())
    ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
    ax_cb2.grid(False)
    cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cm.Blues,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
    ax_cb2.grid(True)
    plt.gcf().add_axes(ax_cb2)
    title(r'$f$')



    figcount+=1
    
    if figcount==2:
        break


# In[ ]:





# In[12]:


count=0

v=np.flip(linspace(0.3,1,len(f_mat)))
figcount=0

for mu_c,sigma_c in zip(mu_c_mat,sigma_c_mat):
    figure()
    for mask in mask_mat:
        if mask:
            for fc,f in tqdm(enumerate(f_mat),total=len(f_mat)):
                s=Storage()
                for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                    params=all_params[count]
                    count+=1
                    R=Results(params.sfname)
                    #R=RR[params.sfname]
                    contrast=params.contrast
                    t=R.t/day
                    recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  

                    s+=contrast,recovery_rate_μ,recovery_rate_σ
                    
                contrast,μ,σ=s.arrays()

                
                errorbar(contrast,-μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
                
        else:
            s=Storage()
            for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                params=all_params[count]
                count+=1
                R=Results(params.sfname)
                #R=RR[params.sfname]
                contrast=params.contrast
                t=R.t/day
                recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  

                s+=contrast,recovery_rate_μ,recovery_rate_σ

            contrast,μ,σ=s.arrays()

            errorbar(contrast,-μ,yerr=2*σ,marker='o',elinewidth=1,color='black')

            
        xlabel('Contrast')
        ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
        title(f'μc={mu_c} σc={sigma_c} blur={blur}')

                    

    divider = make_axes_locatable(plt.gca())
    ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
    ax_cb2.grid(False)
    cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cm.Blues,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
    ax_cb2.grid(True)
    plt.gcf().add_axes(ax_cb2)
    title(r'$f$')



    figcount+=1
    
    if figcount==2:
        break


# In[9]:


mu_c,sigma_c=7.5,2

#[p for p in all_params if p.mu_c==mu_c and p.sigma_c==sigma_c]


# In[20]:


count=0


figcount=0
mu_c,sigma_c=0,0
plot_params=[p for p in all_params if p.mu_c==mu_c and p.sigma_c==sigma_c]

fig1_results=[]
fig2_results=[]
fig3_results=[]
for mask in mask_mat:
    if mask:
        for fc,f in tqdm(enumerate(f_mat),total=len(f_mat)):
            s=Storage()
            s2=Storage()
            s3=Storage()
            for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                params=plot_params[count]
                count+=1
                R=Results(params.sfname)
                #R=RR[params.sfname]
                contrast=params.contrast
                μ1,μ2=R.μσ[0][0]
                σ1,σ2=R.μσ[1][0]

                s+=contrast,μ1,μ2,σ1,σ2
                
                μ,σ=μσ(R.ODI[-1])
                s2+=contrast,μ,σ
                
                t=R.t/day
                recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  
                s3+=contrast,recovery_rate_μ,recovery_rate_σ
                
                

            contrast,μ1,μ2,σ1,σ2=s.arrays()
            fig1_results.append( (contrast,μ1,μ2,σ1,σ2,fc,mask) )
            
            contrast,μ,σ=s2.arrays()
            fig2_results.append( (contrast,μ,σ,fc,mask) )
            
            contrast,μ,σ=s3.arrays()
            fig3_results.append( (contrast,μ,σ,fc,mask) )
            

    else:
        s=Storage()
        s2=Storage()
        s3=Storage()
        for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

            params=plot_params[count]
            count+=1
            R=Results(params.sfname)
            #R=RR[params.sfname]
            contrast=params.contrast
            μ1,μ2=R.μσ[0][0]
            σ1,σ2=R.μσ[1][0]

            s+=contrast,μ1,μ2,σ1,σ2

            
            μ,σ=μσ(R.ODI[-1])
            s2+=contrast,μ,σ
            
            t=R.t/day
            recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  
            s3+=contrast,recovery_rate_μ,recovery_rate_σ
            
            
            
        contrast,μ1,μ2,σ1,σ2=s.arrays()
        fig1_results.append( (contrast,μ1,μ2,σ1,σ2,fc,mask) )

        contrast,μ,σ=s2.arrays()
        fig2_results.append( (contrast,μ,σ,fc,mask) )

        contrast,μ,σ=s3.arrays()
        fig3_results.append( (contrast,μ,σ,fc,mask) )



# In[21]:


figure(figsize=(32,21))
v=np.flip(linspace(0.3,1,len(f_mat)))

subplot(2,2,1)

for contrast,μ1,μ2,σ1,σ2,fc,mask in fig1_results:
    if mask:
        if fc==1:
            errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[fc]))
            errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[fc]))

        else:
            errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
            errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,color=cm.Oranges(v[fc]))
    else:
        errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,color='black')
        errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,color='gray')
        
xlabel('Contrast')
ylabel('Maximum Response')
title(f'μc={mu_c} σc={sigma_c} blur={blur}')
legend()                        
        
        
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)   
ax_cb.grid(False)
ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
ax_cb2.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
plt.gcf().add_axes(ax_cb)
ax_cb.grid(True)
ax_cb.set_yticklabels([])
ax_cb2.grid(True)
title(r'$\sigma_f$') 
plt.gcf().add_axes(ax_cb2)
title(r'$\sigma_f$') 

subplot(2,2,2)

for contrast,μ,σ,fc,mask in fig2_results:
    if mask:
        errorbar(contrast,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
    else:
        errorbar(contrast,μ,yerr=2*σ,marker='o',elinewidth=1,color='black')
        
    
            
ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

xlabel('Contrast')
ylabel(r'$\longleftarrow$ Amblyopic Eye              Fellow Eye $\longrightarrow$'+"\nODI")
title(f'μc={mu_c} σc={sigma_c} blur={blur}')

divider = make_axes_locatable(plt.gca())
ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cm.Blues,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
ax_cb2.grid(True)
plt.gcf().add_axes(ax_cb2)
title(r'$\sigma_f$') 

ax=subplot(2,2,3)
pos=ax.get_position().bounds
ax.set_position([pos[0]+.2,pos[1],pos[2],pos[3]])

for contrast,μ,σ,fc,mask in fig2_results:
    if mask:
        errorbar(contrast,-μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
    else:
        errorbar(contrast,-μ,yerr=2*σ,marker='o',elinewidth=1,color='black')
        
    
            
ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

xlabel('Contrast')
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
title(f'μc={mu_c} σc={sigma_c} blur={blur}')

divider = make_axes_locatable(plt.gca())
ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cm.Blues,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
ax_cb2.grid(True)
plt.gcf().add_axes(ax_cb2)
title(r'$\sigma_f$') 


plt.text(.1, 0.9, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.9, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')


plt.text(.23, 0.46, "C", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('fig-mask-response-ODI-contrast-mu0-sigma0.png')    
        


# In[17]:


count=0


figcount=0
mu_c,sigma_c=7.5,2
plot_params=[p for p in all_params if p.mu_c==mu_c and p.sigma_c==sigma_c]

fig1_results=[]
fig2_results=[]
fig3_results=[]
for mask in mask_mat:
    if mask:
        for fc,f in tqdm(enumerate(f_mat),total=len(f_mat)):
            s=Storage()
            s2=Storage()
            s3=Storage()
            for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

                params=plot_params[count]
                count+=1
                R=Results(params.sfname)
                #R=RR[params.sfname]
                contrast=params.contrast
                μ1,μ2=R.μσ[0][0]
                σ1,σ2=R.μσ[1][0]

                s+=contrast,μ1,μ2,σ1,σ2
                
                μ,σ=μσ(R.ODI[-1])
                s2+=contrast,μ,σ
                
                t=R.t/day
                recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  
                s3+=contrast,recovery_rate_μ,recovery_rate_σ
                
                

            contrast,μ1,μ2,σ1,σ2=s.arrays()
            fig1_results.append( (contrast,μ1,μ2,σ1,σ2,fc,mask) )
            
            contrast,μ,σ=s2.arrays()
            fig2_results.append( (contrast,μ,σ,fc,mask) )
            
            contrast,μ,σ=s3.arrays()
            fig3_results.append( (contrast,μ,σ,fc,mask) )
            

    else:
        s=Storage()
        s2=Storage()
        s3=Storage()
        for contrast_count,contrast in tqdm(enumerate(contrast_mat),total=len(contrast_mat)):

            params=plot_params[count]
            count+=1
            R=Results(params.sfname)
            #R=RR[params.sfname]
            contrast=params.contrast
            μ1,μ2=R.μσ[0][0]
            σ1,σ2=R.μσ[1][0]

            s+=contrast,μ1,μ2,σ1,σ2

            
            μ,σ=μσ(R.ODI[-1])
            s2+=contrast,μ,σ
            
            t=R.t/day
            recovery_rate_μ,recovery_rate_σ=μσ((R.ODI[-1,:]-R.ODI[0,:])/(t[-1]-t[0]))  
            s3+=contrast,recovery_rate_μ,recovery_rate_σ
            
            
            
        contrast,μ1,μ2,σ1,σ2=s.arrays()
        fig1_results.append( (contrast,μ1,μ2,σ1,σ2,fc,mask) )

        contrast,μ,σ=s2.arrays()
        fig2_results.append( (contrast,μ,σ,fc,mask) )

        contrast,μ,σ=s3.arrays()
        fig3_results.append( (contrast,μ,σ,fc,mask) )



# In[19]:


figure(figsize=(32,21))
v=np.flip(linspace(0.3,1,len(f_mat)))

subplot(2,2,1)

for contrast,μ1,μ2,σ1,σ2,fc,mask in fig1_results:
    if mask:
        if fc==1:
            errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,label=f'Amblyopic',color=cm.Blues(v[fc]))
            errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,label=f'Fellow',color=cm.Oranges(v[fc]))

        else:
            errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
            errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,color=cm.Oranges(v[fc]))
    else:
        errorbar(contrast,μ1,yerr=2*σ1,marker='o',elinewidth=1,color='black')
        errorbar(contrast,μ2,yerr=2*σ2,marker='s',elinewidth=1,color='gray')
        
xlabel('Contrast')
ylabel('Maximum Response')
title(f'μc={mu_c} σc={sigma_c} blur={blur}')
legend()                        
        
        
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)   
ax_cb.grid(False)
ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
ax_cb2.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
plt.gcf().add_axes(ax_cb)
ax_cb.grid(True)
ax_cb.set_yticklabels([])
ax_cb2.grid(True)
title(r'$\sigma_f$') 
plt.gcf().add_axes(ax_cb2)
title(r'$\sigma_f$') 

subplot(2,2,2)

for contrast,μ,σ,fc,mask in fig2_results:
    if mask:
        errorbar(contrast,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
    else:
        errorbar(contrast,μ,yerr=2*σ,marker='o',elinewidth=1,color='black')
        
    
            
ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

xlabel('Contrast')
ylabel(r'$\longleftarrow$ Amblyopic Eye              Fellow Eye $\longrightarrow$'+"\nODI")
title(f'μc={mu_c} σc={sigma_c} blur={blur}')

divider = make_axes_locatable(plt.gca())
ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cm.Blues,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
ax_cb2.grid(True)
plt.gcf().add_axes(ax_cb2)
title(r'$\sigma_f$') 

ax=subplot(2,2,3)
pos=ax.get_position().bounds
ax.set_position([pos[0]+.2,pos[1],pos[2],pos[3]])

for contrast,μ,σ,fc,mask in fig2_results:
    if mask:
        errorbar(contrast,-μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Blues(v[fc]))
    else:
        errorbar(contrast,-μ,yerr=2*σ,marker='o',elinewidth=1,color='black')
        
    
            
ylim([-1,1])
xl=xlim()
plot(xl,[0,0],'k-',lw=2)

xlabel('Contrast')
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
title(f'μc={mu_c} σc={sigma_c} blur={blur}')

divider = make_axes_locatable(plt.gca())
ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=cm.Blues,norm=mpl.colors.Normalize(vmin=f_mat[0], vmax=f_mat[-1]),orientation='vertical')
ax_cb2.grid(True)
plt.gcf().add_axes(ax_cb2)
title(r'$\sigma_f$') 


plt.text(.1, 0.9, "A", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

plt.text(.5, 0.9, "B", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')


plt.text(.23, 0.46, "C", transform=plt.gcf().transFigure,
    fontsize=26, fontweight='bold', va='top')

savefig('fig-mask-response-ODI-contrast-mu75-sigma2.png')    
        


# In[ ]:




