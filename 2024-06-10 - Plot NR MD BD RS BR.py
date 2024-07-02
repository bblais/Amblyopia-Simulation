#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[ ]:


from deficit_defs import *


# In[ ]:


# import ray 
# number_of_processes=8
# ray.init(num_cpus=number_of_processes)


# In[ ]:


base='sims/2024-06-10'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


# In[ ]:


noise_mat=linspace(0,2,21)
noise_mat


# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=noise,
         noise2=noise,

         blur1=-1,
         blur2=-1,
         scale1=1,
         scale2=1,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/nr %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[ ]:


RR={}
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[ ]:


R=RR[params.sfname]
strong_i=1
weak_i=0


plot(R.t,R.y[:,0,strong_i],'b',label='Fellow Eye')
plot(R.t,R.y[:,0,weak_i],'m',label='Amblyopic Eye')

for n in range(number_of_neurons):
    plot(R.t,R.y[:,n,0],'m',lw=1)
    plot(R.t,R.y[:,n,1],'b',lw=1)
    
    
ylabel('Response')
legend()
print(params.sfname)
reformat_time_axis()    


# In[ ]:


μ=R.y.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
S=R.y.std(axis=1)
N=R.y.shape[1]
K=1+20/N**2
σ=K*S/np.sqrt(N)
errorbar(R.t,μ[:,strong_i],yerr=σ[:,strong_i],fmt='b.',lw=1,ecolor='k')
errorbar(R.t,μ[:,weak_i],yerr=σ[:,weak_i],fmt='m.',lw=1,ecolor='k')
ylabel('Response')


# In[ ]:


μ=R.y.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
S=R.y.std(axis=1)
N=R.y.shape[1]
K=1+20/N**2
σ=K*S/np.sqrt(N)


plot(R.t,μ[:,strong_i],'bo-',markersize=2,label='Right Eye',lw=1)
fill_between(R.t,μ[:,strong_i]-2*σ[:,weak_i],μ[:,strong_i]+2*σ[:,weak_i],color='b',alpha=.6)

plot(R.t,μ[:,weak_i],'mo-',markersize=2,label='Left Eye',lw=1)
fill_between(R.t,μ[:,weak_i]-2*σ[:,weak_i],μ[:,weak_i]+2*σ[:,weak_i],color='m',alpha=.6)

legend(fontsize=15)
ylabel('Response')


# In[ ]:


params


# In[ ]:


R.ODI.shape


# In[ ]:


μ=R.ODI.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
S=R.ODI.std(axis=1)
N=R.ODI.shape[1]
K=1+20/N**2
σ=K*S/np.sqrt(N)
errorbar(R.t,μ,yerr=σ,fmt='.',lw=1,ecolor='k')


# In[ ]:


v=np.flip(linspace(0.3,1,len(noise_mat)))

for params in tqdm(all_params):
    idx=where(noise_mat==params.noise1)[0][0]
    R=RR[params.sfname]
    μ=R.ODI.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.ODI.std(axis=1)
    N=R.ODI.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    errorbar(R.t,μ,yerr=σ,fmt='o',markersize=1,color=Blues2(v[idx]),
             lw=1,ecolor='k',label=f"{params.noise1:.1f}")
    

reformat_time_axis()

ylabel('ODI')
ylim([-1,1])

reformat_time_axis()
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
plt.gcf().add_axes(ax_cb)



# # MD

# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=noise,
         noise2=0.1,

         blur1=-1,
         blur2=-1,
         scale1=0,
         scale2=1,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/md %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[ ]:


RR={}
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise1)[0][0]
    R=RR[params.sfname]
    μ=R.y.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.y.std(axis=1)
    N=R.y.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    
    fill_between(R.t,μ[:,strong_i]-2*σ[:,strong_i],μ[:,strong_i]+2*σ[:,strong_i],color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ[:,strong_i],'o',markersize=2,color=Blues2(v[idx]))
    
    fill_between(R.t,μ[:,weak_i]-2*σ[:,weak_i],μ[:,weak_i]+2*σ[:,weak_i],color=Oranges2(v[idx]),alpha=.6)
    plot(R.t,μ[:,weak_i],'o',markersize=2,color=Oranges2(v[idx]))
    
    
    # errorbar(R.t,μ[:,strong_i],yerr=σ[:,strong_i],fmt='o',markersize=1,color=Blues2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Right {params.noise1:.1f}")
    # errorbar(R.t,μ[:,weak_i],yerr=σ[:,weak_i],fmt='o',markersize=1,color=Oranges2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Left {params.noise1:.1f}")
    
    
ylabel('Response')
reformat_time_axis()


divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
ax_cb.set_yticklabels([])

ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb2.tick_params(axis='y', which='major', labelsize=15)
ax_cb2.grid(True)


plt.gcf().add_axes(ax_cb)
plt.gcf().add_axes(ax_cb2)



title(r'$\sigma_{\mathrm{noise}}$',fontsize=15)



# In[ ]:


v=linspace(0.3,1,len(noise_mat))

for params in tqdm(all_params):
    idx=where(noise_mat==params.noise1)[0][0]
    
    R=RR[params.sfname]
    μ=R.ODI.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.ODI.std(axis=1)
    N=R.ODI.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    fill_between(R.t,μ-2*σ,μ+2*σ,color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ,'o',markersize=2,color=Blues2(v[idx]))
    
    # errorbar(R.t,μ,yerr=σ,fmt='.',color=Blues2(v[idx]),
    #          lw=1,ecolor='k',label=f"{params.noise1:.1f}")

reformat_time_axis()

ylabel('ODI')
ylim([-1,1])



divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
plt.gcf().add_axes(ax_cb)
ax_cb.tick_params(axis='y', which='major', labelsize=15)

ax_cb.grid(True)
title(r'$\sigma_{\mathrm{noise}}$',fontsize=15)


# # BD

# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0

for noise_count,noise in enumerate(noise_mat):
    
    all_params.append(params(count=count,

         noise1=noise,
         noise2=0.1,

         blur1=-1,
         blur2=-1,
         scale1=0,
         scale2=0,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/bd %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[ ]:


params.noise1


# In[ ]:


RR={}
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[ ]:


params.sfname


# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise1)[0][0]
    R=RR[params.sfname]
    μ=R.y.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.y.std(axis=1)
    N=R.y.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    
    fill_between(R.t,μ[:,strong_i]-2*σ[:,strong_i],μ[:,strong_i]+2*σ[:,strong_i],color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ[:,strong_i],'o',markersize=2,color=Blues2(v[idx]))
    
    fill_between(R.t,μ[:,weak_i]-2*σ[:,weak_i],μ[:,weak_i]+2*σ[:,weak_i],color=Oranges2(v[idx]),alpha=.6)
    plot(R.t,μ[:,weak_i],'o',markersize=2,color=Oranges2(v[idx]))
    
    
    # errorbar(R.t,μ[:,strong_i],yerr=σ[:,strong_i],fmt='o',markersize=1,color=Blues2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Right {params.noise1:.1f}")
    # errorbar(R.t,μ[:,weak_i],yerr=σ[:,weak_i],fmt='o',markersize=1,color=Oranges2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Left {params.noise1:.1f}")
    
    
ylabel('Response')
reformat_time_axis()


divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
ax_cb.set_yticklabels([])

ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb2.tick_params(axis='y', which='major', labelsize=15)
ax_cb2.grid(True)


plt.gcf().add_axes(ax_cb)
plt.gcf().add_axes(ax_cb2)



title(r'$\sigma_{\mathrm{noise}}$',fontsize=15)



# In[ ]:


for noise in noise_mat:
    idx=where(noise_mat==noise)[0][0]
    
    plot(range(100),randn(100)*.01+noise,'o',markersize=5,color=Blues2(v[idx]))


# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise1)[0][0]
    
    R=RR[params.sfname]
    μ=R.ODI.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.ODI.std(axis=1)
    N=R.ODI.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    fill_between(R.t,μ-2*σ,μ+2*σ,color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ,'o',markersize=2,color=Blues2(v[idx]))
    
    # errorbar(R.t,μ,yerr=σ,fmt='.',color=Blues2(v[idx]),
    #          lw=1,ecolor='k',label=f"{params.noise1:.1f}")
    
ylabel('ODI')
ylim([-1,1])

reformat_time_axis()
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
plt.gcf().add_axes(ax_cb)



# # RS

# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=0.1,
         noise2=noise,

         blur1=-1,
         blur2=-1,
         scale1=1,
         scale2=0,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/rs %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[ ]:


RR={}
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise2)[0][0]
    R=RR[params.sfname]
    μ=R.y.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.y.std(axis=1)
    N=R.y.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    
    fill_between(R.t,μ[:,strong_i]-2*σ[:,strong_i],μ[:,strong_i]+2*σ[:,strong_i],color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ[:,strong_i],'o',markersize=2,color=Blues2(v[idx]))
    
    fill_between(R.t,μ[:,weak_i]-2*σ[:,weak_i],μ[:,weak_i]+2*σ[:,weak_i],color=Oranges2(v[idx]),alpha=.6)
    plot(R.t,μ[:,weak_i],'o',markersize=2,color=Oranges2(v[idx]))
    
    
    # errorbar(R.t,μ[:,strong_i],yerr=σ[:,strong_i],fmt='o',markersize=1,color=Blues2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Right {params.noise1:.1f}")
    # errorbar(R.t,μ[:,weak_i],yerr=σ[:,weak_i],fmt='o',markersize=1,color=Oranges2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Left {params.noise1:.1f}")
    
    
ylabel('Response')
reformat_time_axis()


divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
ax_cb.set_yticklabels([])

ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb2.tick_params(axis='y', which='major', labelsize=15)
ax_cb2.grid(True)


plt.gcf().add_axes(ax_cb)
plt.gcf().add_axes(ax_cb2)



title(r'$\sigma_{\mathrm{noise}}$',fontsize=15)



# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise2)[0][0]
    
    R=RR[params.sfname]
    μ=R.ODI.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.ODI.std(axis=1)
    N=R.ODI.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    fill_between(R.t,μ-2*σ,μ+2*σ,color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ,'o',markersize=2,color=Blues2(v[idx]))
    
#    errorbar(R.t,μ,yerr=σ,fmt='o',markersize=5,color=Blues2(v[idx]),lw=0.1,ecolor='k',label=f"{params.noise2:.1f}")

reformat_time_axis()


ylabel('ODI')
ylim([-1,1])

reformat_time_axis()
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
plt.gcf().add_axes(ax_cb)




# In[ ]:


params


# # BR

# In[ ]:


from collections import namedtuple
params = namedtuple('params', 
                    ['count', 'noise1','noise2',
                     'blur1','blur2','number_of_neurons',
                     'scale1','scale2',
                     'sfname','mu_c','sigma_c'])
all_params=[]
count=0

number_of_neurons=20
noise_mat=linspace(0,2,11)
sigma_c=0
blur_mat=linspace(-1,13,15)

for noise_count,noise in enumerate(noise_mat):
    all_params.append(params(count=count,

         noise1=0.1,
         noise2=noise,

         blur1=-1,
         blur2=-1,
         scale1=1,
         scale2=1,
         mu_c=0,
         sigma_c=0,
         number_of_neurons=number_of_neurons,

        sfname=f'{base}/br %d neurons dog %d.asdf' % 
                 (number_of_neurons,noise_count),
                ))
        
    count+=1
        
for a in all_params[:5]:
    print(a)
print("[....]")
for a in all_params[-5:]:
    print(a)

print(len(all_params))


# In[ ]:


RR={}
for params in tqdm(all_params):
    RR[params.sfname]=Results(params.sfname)


# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise2)[0][0]
    R=RR[params.sfname]
    μ=R.y.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.y.std(axis=1)
    N=R.y.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    
    fill_between(R.t,μ[:,strong_i]-2*σ[:,strong_i],μ[:,strong_i]+2*σ[:,strong_i],color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ[:,strong_i],'o',markersize=2,color=Blues2(v[idx]))
    
    fill_between(R.t,μ[:,weak_i]-2*σ[:,weak_i],μ[:,weak_i]+2*σ[:,weak_i],color=Oranges2(v[idx]),alpha=.6)
    plot(R.t,μ[:,weak_i],'o',markersize=2,color=Oranges2(v[idx]))
    
    
    # errorbar(R.t,μ[:,strong_i],yerr=σ[:,strong_i],fmt='o',markersize=1,color=Blues2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Right {params.noise1:.1f}")
    # errorbar(R.t,μ[:,weak_i],yerr=σ[:,weak_i],fmt='o',markersize=1,color=Oranges2(v[idx]),
    #          lw=0.2,ecolor='k',label=f"Left {params.noise1:.1f}")
    
    
ylabel('Response')
reformat_time_axis()


divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
ax_cb.set_yticklabels([])

ax_cb2 = divider.new_horizontal(size="5%", pad=0.05)
ax_cb2.grid(False)
cb2 = mpl.colorbar.ColorbarBase(ax_cb2, cmap=Oranges2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb2.tick_params(axis='y', which='major', labelsize=15)
ax_cb2.grid(True)


plt.gcf().add_axes(ax_cb)
plt.gcf().add_axes(ax_cb2)



title(r'$\sigma_{\mathrm{noise}}$',fontsize=15)



# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise2)[0][0]
    
    
    
    R=RR[params.sfname]
    μ=R.ODI.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.ODI.std(axis=1)
    N=R.ODI.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    errorbar(R.t,μ,yerr=σ,fmt='o',markersize=2,color=Blues2(v[idx]),lw=0.2,ecolor='k',label=f"{params.noise2:.1f}")

reformat_time_axis()


ylabel('ODI')
ylim([-1,1])

reformat_time_axis()
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
plt.gcf().add_axes(ax_cb)



# In[ ]:


for params in tqdm(all_params):
    idx=where(noise_mat==params.noise2)[0][0]
    
    R=RR[params.sfname]
    μ=R.ODI.mean(axis=1)  # average across neurons, at the end of a seq, for each channel
    S=R.ODI.std(axis=1)
    N=R.ODI.shape[1]
    K=1+20/N**2
    σ=K*S/np.sqrt(N)
    fill_between(R.t,μ-2*σ,μ+2*σ,color=Blues2(v[idx]),alpha=.6)
    plot(R.t,μ,'o',markersize=2,color=Blues2(v[idx]))
    
#    errorbar(R.t,μ,yerr=σ,fmt='o',markersize=5,color=Blues2(v[idx]),lw=0.1,ecolor='k',label=f"{params.noise2:.1f}")

reformat_time_axis()


ylabel('ODI')
ylim([-1,1])

reformat_time_axis()
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.grid(False)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=Blues2,
                                norm=mpl.colors.Normalize(vmin=noise_mat[0], vmax=noise_mat[-1]),
                                orientation='vertical')
ax_cb.tick_params(axis='y', which='major', labelsize=15)
ax_cb.grid(True)
plt.gcf().add_axes(ax_cb)




# In[ ]:




