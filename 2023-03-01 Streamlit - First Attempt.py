import streamlit as st
from pylab import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from deficit_defs import *
from treatment_sims_2023_02_21 import *

def savefig(base):
    import matplotlib.pyplot as plt
    for fname in [f'Manuscript/resources/{base}.png',f'Manuscript/resources/{base}.svg']:
        print(fname)
        plt.savefig(fname, bbox_inches='tight')


def mydisplay(t,sim,neurons,connections):

    c=connections[0]

    weights=c.weights
    num_neurons=len(weights)
    rf_size=neurons[0][0].rf_size
    num_channels=len(neurons[0])

    W=weights.reshape((num_neurons,
                        num_channels,
                        rf_size,rf_size))

    vmin,vmax=W.min(),W.max()

    W=(W-vmin)/(vmax-vmin)

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

    st.empty()
    st.image(im)
    # "θ",θ.shape,t.shape
    # st.line_chart(pd.DataFrame({'x':t,'y':θ}))

placeholder = st.empty()


base='sims/2023-03-01'
if not os.path.exists(base):
    print(f"mkdir {base}")
    os.mkdir(base)


rf_size=19
eta=1e-6
blur=6
number_of_neurons=3
number_of_processes=4
mu_c=7.5
sigma_c=2
mu_r=0
sigma_r=0
noise=1

seq=pn.Sequence()

total_time=16*day
save_interval=1*hour

base_image_file='asdf/bbsk081604_all.asdf'
    
images=[]
dt=200*ms

for bv in [blur,-1]:
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



seq+=sim,[pre,post],[c]


seq.run(display=mydisplay,display_hash=False,
               time_between_display=4*save_interval)




