#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylab import *


# In[2]:


from deficit_defs_2025_02_25 import *


# In[3]:


mu_c=9
sigma_c=9
mu_r=0
sigma_r=9
blur=3

# take about 3 hours to do a full set of sims

base_sim_dir=f"sims-2025-06-04 mu_c {mu_c} sigma_c {sigma_c} mu_r {mu_r} sigma_r {sigma_r} blur {blur}"
if not os.path.exists(base_sim_dir):
    print("new")
    os.mkdir(base_sim_dir)
print(base_sim_dir)
savepath=base_sim_dir


# In[4]:


number_of_neurons=20
eta=1e-6
number_of_processes=8
#ray.init(num_cpus=number_of_processes)


# In[5]:


#loadvars(base_sim_dir+'/full_glasses_results.asdf')
# loadvars(base_sim_dir+'/patch_results.asdf')
# loadvars(base_sim_dir+'/atropine_results.asdf')
# loadvars(base_sim_dir+'/contrast_results.asdf')
# loadvars(base_sim_dir+'/mask_results.asdf')


# In[6]:


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

sfname=f"{savepath}/glasses_treatment.pdf"
print(sfname)
savefig(sfname)


# In[ ]:





# In[7]:


loadvars(base_sim_dir+'/full_glasses_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.viridis
color = colormap(np.linspace(0, 1,n))

glasses_plot_color=cm.viridis(np.linspace(0, 1,5))[2]

glasses_μ_SF_Var=-SF_Var_recovery_rate_μ
glasses_σ_SF_Var=2*SF_Var_recovery_rate_σ

# best case
idx=argmax(glasses_μ)
max_glasses_SF_Var=glasses_μ_SF_Var[idx]+glasses_σ_SF_Var[idx]
min_glasses_SF_Var=glasses_μ_SF_Var[idx]-glasses_σ_SF_Var[idx]
print("_SF_Var",min_glasses_SF_Var,max_glasses_SF_Var)


errorbar(noise,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='o-',color=color[2]) # positive = recovery
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")
xlabel('Open-Eye Noise Level')
title('Glasses Treatment')

sfname=f"{savepath}/glasses_treatment_SFVar.pdf"
print(sfname)
savefig(sfname)


# In[ ]:





# In[ ]:





# In[8]:


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

sfname=f"{savepath}/patch_treatment.pdf"
print(sfname)
savefig(sfname)


# In[9]:


loadvars(base_sim_dir+'/full_patch_results.asdf')

import cycler
colormap=cm.viridis

n = 5
#colormap=cm.Blues
#color = colormap(np.linspace(1, 0,int(1.2*n)))

colormap=cm.viridis
color = colormap(np.linspace(0, 1,n))


#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=patch_result        

patch_μ_SF_Var=-recovery_rate_μ
patch_σ_SF_Var=2*recovery_rate_σ

# best case
idx=argmax(patch_μ_SF_Var)
max_patch_SF_Var=patch_μ_SF_Var[idx]+patch_σ_SF_Var[idx]
min_patch_SF_Var=patch_μ_SF_Var[idx]-patch_σ_SF_Var[idx]
print("_SF_Var",min_patch_SF_Var,max_patch_SF_Var)


errorbar(noise,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='rs-') # positive = recovery
ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")
xlabel('Closed-Eye Noise Level')
title('Patch Treatment')

sfname=f"{savepath}/patch_treatment_SFVar.pdf"
print(sfname)
savefig(sfname)


# In[10]:


loadvars(base_sim_dir+'/full_atropine_results.asdf')
import cycler
colormap=cm.viridis
n = 25
#color = colormap(np.linspace(1, 0,int(1.2*n)))
color = colormap(np.linspace(0, 1,n))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)



#noise,blur,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=atropine_result
blur_N=blur.shape[1]
max_atropine_μ=-9e9
for b in range(blur_N):

    if blur[0,b] in [0,1.5,3,4.5,6]:
        errorbar(noise[:,b],-recovery_rate_μ[:,b],yerr=2*recovery_rate_σ[:,b],elinewidth=1,
                 label=f'Blur {blur[0,b]}',fmt='o-',color=color[b]) # positive = recovery
    else:
        errorbar(noise[:,b],-recovery_rate_μ[:,b],yerr=2*recovery_rate_σ[:,b],elinewidth=1,
                 color=color[b],fmt='o-') # positive = recovery


    atropine_μ=-recovery_rate_μ[:,b]
    atropine_σ=2*recovery_rate_σ[:,b]

    # best case

    if atropine_μ.max()>max_atropine_μ:
        max_atropine_μ=atropine_μ.max()
        idx=argmax(atropine_μ)
        max_atropine=atropine_μ[idx]+atropine_σ[idx]
        min_atropine=atropine_μ[idx]-atropine_σ[idx]


print("min max atropine",min_atropine,max_atropine)        



loadvars(base_sim_dir+'/full_patch_results.asdf')
#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=patch_result        
errorbar(noise,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,color='r',fmt='s-',label='Patch') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[ODI shift/time]")
xlabel('Closed-Eye Noise Level')
title('Atropine Treatment')

legend()
sfname=f"{savepath}/atropine_treatment.pdf"
print(sfname)
savefig(sfname)


# In[11]:


loadvars(base_sim_dir+'/full_atropine_results.asdf')
import cycler
colormap=cm.viridis
n = 25
#color = colormap(np.linspace(1, 0,int(1.2*n)))
color = colormap(np.linspace(0, 1,n))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)



#noise,blur,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=atropine_result
blur_N=blur.shape[1]
max_atropine_μ_SF_Var=-9e9
for b in range(blur_N):

    if blur[0,b] in [0,1.5,3,4.5,6]:
        errorbar(noise[:,b],-SF_Var_recovery_rate_μ[:,b],yerr=2*SF_Var_recovery_rate_σ[:,b],elinewidth=1,
                 label=f'Blur {blur[0,b]}',fmt='o-',color=color[b]) # positive = recovery
    else:
        errorbar(noise[:,b],-SF_Var_recovery_rate_μ[:,b],yerr=2*SF_Var_recovery_rate_σ[:,b],elinewidth=1,
                 color=color[b],fmt='o-') # positive = recovery


    atropine_μ_SF_Var=-recovery_rate_μ[:,b]
    atropine_σ_SF_Var=2*recovery_rate_σ[:,b]

    # best case

    if atropine_μ_SF_Var.max()>max_atropine_μ_SF_Var:
        max_atropine_μ_SF_Var=atropine_μ_SF_Var.max()
        idx=argmax(atropine_μ_SF_Var)
        max_atropine_SF_Var=atropine_μ_SF_Var[idx]+atropine_σ_SF_Var[idx]
        min_atropine_SF_Var=atropine_μ_SF_Var[idx]-atropine_σ_SF_Var[idx]

print("_SF_Var min max atropine",min_atropine_SF_Var,max_atropine_SF_Var)        


loadvars(base_sim_dir+'/full_patch_results.asdf')
#noise,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=patch_result        
errorbar(noise,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,color='r',fmt='s-',label='Patch') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")
xlabel('Closed-Eye Noise Level')
title('Atropine Treatment')

legend()
sfname=f"{savepath}/atropine_treatment_SFVar.pdf"
print(sfname)
savefig(sfname)


# In[ ]:





# In[ ]:





# In[12]:


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


# In[13]:


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


# In[14]:


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
             label=f'Mask {f[fi,0]//10}'+r'$^{\circ}$',color=color[fi],fmt='o-') # positive = recovery


loadvars(base_sim_dir+'/full_contrast_results.asdf')
#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


ylabel(r'$\longleftarrow$ Slower recovery     Faster Recovery $\longrightarrow$'+"\n[Spatial Frequency Variance shift/time]")

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
legend(fontsize=16)

sfname=f"{savepath}/contrast_mask_treatment SFVar.pdf"
print(sfname)
savefig(sfname)


# In[ ]:





# In[ ]:





# In[15]:


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
             label=f'Mask {f[fi,0]//10}'+r'$^{\circ}$',color=color[fi],fmt='o-') # positive = recovery



loadvars(base_sim_dir+'/full_contrast_results.asdf')
#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-recovery_rate_μ,yerr=2*recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


#ylabel(r'Rate of Ocular Dominance Change'+"\n[Ocular Dominance shift/time]")
ylabel(r'Ocular Dominance Shift per Time')

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

legend(fontsize=11,loc='lower left')
gca().set_xticks([0,.2,.4,.6,.8,1.0])
gca().set_xticklabels([f"{_}%" for _ in [0,20,40,60,80,100]])
xlim([0,1])

gca().set_facecolor('none')

sfname=f"{savepath}/contrast_mask_treatment.pdf"
print(sfname)
plt.tight_layout()
plt.savefig(sfname,transparent=True)


# ![image.png](attachment:abd5df7f-b0b4-48e4-a3db-344dc70a442a.png)

# In[16]:


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

    errorbar(contrast[fi,:],-SF_Var_recovery_rate_μ[fi,:],yerr=2*SF_Var_recovery_rate_σ[fi,:],elinewidth=1,
             label=f'Mask {f[fi,0]//10}'+r'$^{\circ}$',color=color[fi],fmt='o-') # positive = recovery



loadvars(base_sim_dir+'/full_contrast_results.asdf')
#contrast,recovery_rate_μ,recovery_rate_σ,ODI_μ1,ODI_σ1,ODI_μ2,ODI_σ2=contrast_result
errorbar(contrast,-SF_Var_recovery_rate_μ,yerr=2*SF_Var_recovery_rate_σ,elinewidth=1,fmt='o-',color='k',label='No Mask') # positive = recovery


#ylabel(r'Rate of Ocular Dominance Change'+"\n[Ocular Dominance shift/time]")
ylabel(r"Spatial Frequency Variance shift per time")

xlabel('Fellow Eye Contrast')
#title('Contrast+Mask Treatment')

xl=gca().get_xlim()
plot(xl,[0,0],'k-',lw=1)
gca().set_xlim(xl)

yl=array(gca().get_ylim())
mx=max(abs(yl))
yl=[-mx,mx]
yl=[-.2,.2]
gca().set_ylim(yl)


text(0.25,0.015,'Recovering',ha='center',va='center',color='green')
text(0.25,-0.02,'Worsening',ha='center',va='center',color='red')
arrow(.25,-.04,0,-.03,width=0.004,color='red')
arrow(.25,.04,0,.03,width=0.004,color='green')

axhspan(min_patch_SF_Var,max_patch_SF_Var, color='red', alpha=0.1,label='Patch/Atropine')

glasses_plot_color=cm.viridis(np.linspace(0, 1,5))[2]
axhspan(min_glasses_SF_Var,max_glasses_SF_Var, color=glasses_plot_color, alpha=0.1,label='Glasses Alone')

legend(fontsize=11,loc='lower left')
gca().set_xticks([0,.2,.4,.6,.8,1.0])
gca().set_xticklabels([f"{_}%" for _ in [0,20,40,60,80,100]])
xlim([0,1])

gca().set_facecolor('none')

sfname=f"{savepath}/contrast_mask_treatment_SFVar.pdf"
print(sfname)
plt.tight_layout()
plt.savefig(sfname,transparent=True)


# In[17]:


max_patch_SF_Var


# In[ ]:





# In[ ]:




