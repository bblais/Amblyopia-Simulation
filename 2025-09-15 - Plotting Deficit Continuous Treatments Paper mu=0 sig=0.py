#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylab import *


# In[2]:


from deficit_defs_2025_02_25 import *


# In[11]:


mu_c=0
sigma_c=0
mu_r=0
sigma_r=0

# mu_c=9
# sigma_c=9
# mu_r=0
# sigma_r=0

base_sim_dir="sims-2025-04-18 mu 0 sigma 0"


base_sim_dir=f"sims-2025-04-18 mu {mu_c} sigma {sigma_c}"
if not os.path.exists(base_sim_dir):
    raise ValueError
print(base_sim_dir)


# In[12]:


number_of_neurons=20
eta=1e-6
number_of_processes=8
#ray.init(num_cpus=number_of_processes)


# In[13]:


#loadvars(base_sim_dir+'/full_glasses_results.asdf')
# loadvars(base_sim_dir+'/patch_results.asdf')
# loadvars(base_sim_dir+'/atropine_results.asdf')
# loadvars(base_sim_dir+'/contrast_results.asdf')
# loadvars(base_sim_dir+'/mask_results.asdf')


# In[14]:


loadvars(base_sim_dir+'/full_glasses_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.viridis
color = colormap(np.linspace(0, 1,n))

glasses_plot_color=cm.viridis(np.linspace(0, 1,5))[2]

#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2,,SF_Var_recovery_rate_μ,SF_Var_recovery_rate_σ =glasses_result        

glasses_μ=-recovery_rate_μ
glasses_σ=2*recovery_rate_σ

# best case
idx=argmax(glasses_μ)
max_glasses=glasses_μ[idx]+glasses_σ[idx]
min_glasses=glasses_μ[idx]-glasses_σ[idx]
print(min_glasses,max_glasses)


errorbar(noise,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color=color[2]) # positive = recovery
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
xlabel('Open-Eye Noise Level')
title('Glasses Treatment')

# sfname=f"{savepath}/glasses_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[15]:


loadvars(base_sim_dir+'/full_glasses_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.viridis
color = colormap(np.linspace(0, 1,n))

glasses_plot_color=cm.viridis(np.linspace(0, 1,5))[2]


errorbar(noise,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='o-',color=color[2]) # positive = recovery
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")
xlabel('Open-Eye Noise Level')
title('Glasses Treatment')


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


loadvars(base_sim_dir+'/full_patch_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.viridis
color = colormap(np.linspace(0, 1,n))


#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=patch_result        

patch_μ=-recovery_rate_μ
patch_σ=2*recovery_rate_σ

# best case
idx=argmax(patch_μ)
max_patch=patch_μ[idx]+patch_σ[idx]
min_patch=patch_μ[idx]-patch_σ[idx]
print(min_patch,max_patch)


errorbar(noise,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='rs-') # positive = recovery
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
xlabel('Closed-Eye Noise Level')
title('Patch Treatment')

# sfname=f"{savepath}/patch_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[17]:


loadvars(base_sim_dir+'/full_patch_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.viridis
color = colormap(np.linspace(0, 1,n))


#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=patch_result        

patch_μ=-recovery_rate_μ
patch_σ=2*recovery_rate_σ

# best case
idx=argmax(patch_μ)
max_patch=patch_μ[idx]+patch_σ[idx]
min_patch=patch_μ[idx]-patch_σ[idx]
print(min_patch,max_patch)


errorbar(noise,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='rs-') # positive = recovery
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")
xlabel('Closed-Eye Noise Level')
title('Patch Treatment')

# sfname=f"{savepath}/patch_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[18]:


loadvars(base_sim_dir+'/full_atropine_results.asdf')
import cycler
colormap=cm.viridis
n = 25
#color = colormap(np.linspace(1, 0,int(1.2*n)))
color = colormap(np.linspace(0, 1,n))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)



#noise,blur,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=atropine_result
blur_N=blur.shape[1]
for b in range(blur_N):

    if blur[0,b] in [0,1.5,3,4.5,6]:
        errorbar(noise[:,b],-recovery_rate_μ[:,b],yerr=2*recovery_rate_σ[:,b],elinewidth=1,
                 label=f'Blur {blur[0,b]}',fmt='o-',color=color[b]) # positive = recovery
    else:
        errorbar(noise[:,b],-recovery_rate_μ[:,b],yerr=2*recovery_rate_σ[:,b],elinewidth=1,
                 color=color[b],fmt='o-') # positive = recovery


loadvars(base_sim_dir+'/full_patch_results.asdf')
#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=patch_result        
errorbar(noise,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,color='r',fmt='s-',label='Patch') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
xlabel('Closed-Eye Noise Level')
title('Atropine Treatment')

legend()
# sfname=f"{savepath}/atropine_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[19]:


loadvars(base_sim_dir+'/full_atropine_results.asdf')
import cycler
colormap=cm.viridis
n = 25
#color = colormap(np.linspace(1, 0,int(1.2*n)))
color = colormap(np.linspace(0, 1,n))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)



#noise,blur,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=atropine_result
blur_N=blur.shape[1]
for b in range(blur_N):

    if blur[0,b] in [0,1.5,3,4.5,6]:
        errorbar(noise[:,b],-SF_Var_recovery_rate_μ[:,b],yerr=2*SF_Var_recovery_rate_σ[:,b],elinewidth=1,
                 label=f'Blur {blur[0,b]}',fmt='o-',color=color[b]) # positive = recovery
    else:
        errorbar(noise[:,b],-SF_Var_recovery_rate_μ[:,b],yerr=2*SF_Var_recovery_rate_σ[:,b],elinewidth=1,
                 color=color[b],fmt='o-') # positive = recovery


loadvars(base_sim_dir+'/full_patch_results.asdf')
#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=patch_result        
errorbar(noise,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,color='r',fmt='s-',label='Patch') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")
xlabel('Closed-Eye Noise Level')
title('Atropine Treatment')

legend()
# sfname=f"{savepath}/atropine_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[ ]:





# In[ ]:





# In[20]:


loadvars(base_sim_dir+'/full_contrast_results.asdf')

#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")

xlabel('Contrast')
#title('Contrast+Mask Treatment')

xl=gca().get_xlim()
plot(xl,[0,0],'k-',lw=1)
gca().set_xlim(xl)

yl=array(gca().get_ylim())
mx=max(abs(yl))
yl=[-mx,mx]
gca().set_ylim(yl)

text(0.25,0.015,'Recovering',ha='center',va='center',color='green')
text(0.25,-0.015,'Worsening',ha='center',va='center',color='red')
arrow(.25,-.03,0,-.03,width=0.004,color='red')
arrow(.25,.03,0,.03,width=0.004,color='green')
legend()


# In[21]:


loadvars(base_sim_dir+'/full_contrast_results.asdf')

#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")

xlabel('Contrast')
#title('Contrast+Mask Treatment')

xl=gca().get_xlim()
plot(xl,[0,0],'k-',lw=1)
gca().set_xlim(xl)

yl=array(gca().get_ylim())
mx=max(abs(yl))
yl=[-mx,mx]
gca().set_ylim(yl)

text(0.25,0.015,'Recovering',ha='center',va='center',color='green')
text(0.25,-0.015,'Worsening',ha='center',va='center',color='red')
arrow(.25,-.03,0,-.03,width=0.004,color='red')
arrow(.25,.03,0,.03,width=0.004,color='green')
legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


loadvars(base_sim_dir+'/full_mask_results.asdf')

import cycler
# f_mat=array([10,30,50,70,90])
print(f_mat)
f_N=len(f_mat)


n = len(f_mat)+1
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))
colormap=cm.viridis
color = colormap(np.linspace(0, 1,int(n)))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

#f,contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=mask_result
for fi in range(f_N):

    errorbar(contrast[fi,:],-recovery_rate_μ[fi,:],yerr=2*recovery_rate_σ[fi,:],elinewidth=1,
             label=f'Mask f {f[fi,0]}',color=color[fi],fmt='o-') # positive = recovery


loadvars(base_sim_dir+'/full_contrast_results.asdf')

#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")

xlabel('Contrast')
#title('Contrast+Mask Treatment')

xl=gca().get_xlim()
plot(xl,[0,0],'k-',lw=1)
gca().set_xlim(xl)

yl=array(gca().get_ylim())
mx=max(abs(yl))
yl=[-mx,mx]
gca().set_ylim(yl)


text(0.25,0.015,'Recovering',ha='center',va='center',color='green')
text(0.25,-0.02,'Worsening',ha='center',va='center',color='red')
arrow(.25,-.04,0,-.03,width=0.004,color='red')
arrow(.25,.04,0,.03,width=0.004,color='green')
legend(fontsize=16)

# sfname=f"{savepath}/contrast_mask_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[23]:


loadvars(base_sim_dir+'/full_mask_results.asdf')

import cycler
# f_mat=array([10,30,50,70,90])
print(f_mat)
f_N=len(f_mat)


n = len(f_mat)+1
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))
colormap=cm.viridis
color = colormap(np.linspace(0, 1,int(n)))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

#f,contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=mask_result
for fi in range(f_N):

    errorbar(contrast[fi,:],-SF_Var_recovery_rate_μ[fi,:],yerr=2*SF_Var_recovery_rate_σ[fi,:],elinewidth=1,
             label=f'Mask f {f[fi,0]}',color=color[fi],fmt='o-') # positive = recovery


loadvars(base_sim_dir+'/full_contrast_results.asdf')
#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")

xlabel('Contrast')
#title('Contrast+Mask Treatment')

xl=gca().get_xlim()
plot(xl,[0,0],'k-',lw=1)
gca().set_xlim(xl)

yl=array(gca().get_ylim())
mx=max(abs(yl))
yl=[-mx,mx]
gca().set_ylim(yl)


text(0.25,0.015,'Recovering',ha='center',va='center',color='green')
text(0.25,-0.02,'Worsening',ha='center',va='center',color='red')
arrow(.25,-.04,0,-.03,width=0.004,color='red')
arrow(.25,.04,0,.03,width=0.004,color='green')
legend(fontsize=16)

# sfname=f"{savepath}/contrast_mask_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[ ]:





# In[ ]:





# In[24]:


loadvars(base_sim_dir+'/full_mask_results.asdf')

import cycler
#f_mat=array([10,30,50,70,90])
f_N=len(f_mat)


n = len(f_mat)+1
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))
colormap=cm.viridis
color = colormap(np.linspace(0, 1,int(n)))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

#f,contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=mask_result
for fi in range(f_N):

    errorbar(contrast[fi,:],-recovery_rate_μ[fi,:],yerr=2*recovery_rate_σ[fi,:],elinewidth=1,
             label=f'Mask f {f[fi,0]}',color=color[fi],fmt='o-') # positive = recovery



loadvars(base_sim_dir+'/full_contrast_results.asdf')
#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


#ylabel(r'Rate of Ocular Dominance Change'+"\n[Ocular Dominance shift/time]")
ylabel(r'Ocular Dominance Shift per Time')

xlabel('Contrast')
#title('Contrast+Mask Treatment')

xl=gca().get_xlim()
plot(xl,[0,0],'k-',lw=1)
gca().set_xlim(xl)

yl=array(gca().get_ylim())
mx=max(abs(yl))
yl=[-mx,mx]
gca().set_ylim(yl)


text(0.25,0.015,'Recovering',ha='center',va='center',color='green')
text(0.25,-0.02,'Worsening',ha='center',va='center',color='red')
arrow(.25,-.04,0,-.03,width=0.004,color='red')
arrow(.25,.04,0,.03,width=0.004,color='green')

axhspan(min_patch,max_patch, color='red', alpha=0.1,label='Patch/Atropine')

glasses_plot_color=cm.viridis(np.linspace(0, 1,5))[2]
axhspan(min_glasses,max_glasses, color=glasses_plot_color, alpha=0.1,label='Glasses')

legend(fontsize=11,loc='lower left')
gca().set_xticks([0,.2,.4,.6,.8,1.0])
gca().set_xticklabels([f"{_}%" for _ in [0,20,40,60,80,100]])
xlim([0,1])

# sfname=f"{savepath}/contrast_mask_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


#sfname=f"{savepath}/contrast_mask_treatment.pdf"
sfname=f"/Users/bblais/Downloads/contrast_mask_treatment.pdf"
print(sfname)
plt.tight_layout()
plt.savefig(sfname, transparent=True)


# In[25]:


[f"{_}%" for _ in [0,20,40,60,80,100]]


# In[26]:


loadvars(base_sim_dir+'/full_mask_results.asdf')

import cycler
#f_mat=array([10,30,50,70,90])
f_N=len(f_mat)


n = len(f_mat)+1
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))
colormap=cm.viridis
color = colormap(np.linspace(0, 1,int(n)))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

#f,contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=mask_result
for fi in range(f_N):

    errorbar(contrast[fi,:],-recovery_rate_μ[fi,:],yerr=2*recovery_rate_σ[fi,:],elinewidth=1,
             label=f'Mask f {f[fi,0]}',color=color[fi],fmt='o-') # positive = recovery



loadvars(base_sim_dir+'/full_contrast_results.asdf')
#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")

xlabel('Fellow Eye Contrast')
#title('Contrast+Mask Treatment')

xl=gca().get_xlim()
plot(xl,[0,0],'k-',lw=1)
gca().set_xlim(xl)

yl=array(gca().get_ylim())
mx=max(abs(yl))
yl=[-mx,mx]
gca().set_ylim(yl)


text(0.25,0.015,'Recovering',ha='center',va='center',color='green')
text(0.25,-0.02,'Worsening',ha='center',va='center',color='red')
arrow(.25,-.04,0,-.03,width=0.004,color='red')
arrow(.25,.04,0,.03,width=0.004,color='green')

axhspan(min_patch,max_patch, color='red', alpha=0.1,label='Patch/Atropine')

glasses_plot_color=cm.viridis(np.linspace(0, 1,5))[2]
axhspan(min_glasses,max_glasses, color=glasses_plot_color, alpha=0.1,label='Glasses Alone')

legend(fontsize=13)


# sfname=f"{savepath}/contrast_mask_treatment.pdf"
# print(sfname)
# savefig(sfname,bbox_inches="tight")


# In[ ]:




