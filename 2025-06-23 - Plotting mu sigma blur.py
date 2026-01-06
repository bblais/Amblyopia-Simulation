#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylab import *


# In[2]:


from deficit_defs_2025_02_25 import *


# In[3]:


base_sim_dir=f"sims-2025-06-04 mu_c sigma_c blur"
if not os.path.exists(base_sim_dir):
    print("new")
    os.mkdir(base_sim_dir)
print(base_sim_dir)


# In[4]:


loadvars(f'{base_sim_dir}/deficit_results.asdf')


# In[5]:


len(subResults)


# In[6]:


fnames=list(subResults.keys())


# In[7]:


R=Struct(subResults[fnames[0]])


# In[8]:


R.y.shape


# In[9]:


μσ(R.y[-1,:,:],axis=0)


# In[10]:


(μ1,μ2),(σ1,σ2)=μσ(R.y[-1,:,:],axis=0)


# In[11]:


(μ1,μ2),(σ1,σ2)


# In[16]:


count=0
for blur_count,blur in enumerate(blur_mat):
    for mu_count,mu_c in tqdm(enumerate(mu_c_mat)):
        R=Struct(subResults[fnames[count]])
        params=Struct(R.params)
        sigma_c=params.sigma_c
        print(fnames[count])
        print("\tBlur",blur,params.blur[0])
        print("\tmu_c",mu_c,params.mu_c)
        print("\tsigma_c",sigma_c)
        count+=1

        if count>10:
            break


# In[19]:


count=0
sigma_c=2.0
number_of_neurons=20
open_eye_noise=0.1

for mu_count,mu_c in tqdm(enumerate(mu_c_mat)):
    s=Storage()
    for blur_count,blur in enumerate(blur_mat):
        sfname=f'{base_sim_dir}/deficit {number_of_neurons} neurons noise {open_eye_noise:.1f} blur {blur:.1f} mu_c {mu_c:.1f} sigma_c {sigma_c:.1f}.asdf'
        R=Struct(subResults[sfname])
        params=Struct(R.params)
        blur=params.blur[0]
        sigma_c=params.sigma_c
        (μ1,μ2),(σ1,σ2)=μσ(R.y[-1,:,:],axis=0)

        s+=blur,μ1,μ2,σ1,σ2
        count+=1


    blur,μ1,μ2,σ1,σ2=s.arrays()

    figure()
    errorbar(blur,μ1,yerr=2*σ1,marker='o',elinewidth=1,label='Deprived')
    errorbar(blur,μ2,yerr=2*σ2,marker='s',elinewidth=1,label='Normal')
    xlabel('Blur Size [pixels]')
    ylabel('Maximum Response')
    title(f'μ_c={mu_c},σ_c={sigma_c}')
    legend()    



# In[ ]:





# In[23]:


count=0
v=np.flip(linspace(0.3,1,len(mu_c_mat)))

for mu_count,mu_c in enumerate(mu_c_mat):
    s=Storage()
    for blur_count,blur in enumerate(blur_mat):
        if blur==0.5:
            continue

        sfname=f'{base_sim_dir}/deficit {number_of_neurons} neurons noise {open_eye_noise:.1f} blur {blur:.1f} mu_c {mu_c:.1f} sigma_c {sigma_c:.1f}.asdf'
        R=Struct(subResults[sfname])
        μ,σ=μσ(R.ODI[-1])

        s+=blur,μ,σ
        count+=1


    blur,μ,σ=s.arrays()
    errorbar(blur,μ,yerr=2*σ,marker='o',elinewidth=1,color=cm.Oranges(v[mu_count]))    
    xlabel('Blur Size [pixels]')
    ylabel(r'$\longleftarrow$ Weak Eye              Strong Eye $\longrightarrow$'+"\nODI")
    ylim([-1,1])

divider = make_axes_locatable(plt.gca())
ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)    
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,norm=mpl.colors.Normalize(vmin=mu_c_mat[0], vmax=mu_c_mat[-1]),orientation='vertical')
ax_cb2.grid(True)
plt.gcf().add_axes(ax_cb2)
title(r'$\mu_c$')


# In[ ]:




