from pylab import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print=st.write

# comment

'''
# heading

## subheading

$$
E=mc^2
$$
'''

st.title('Uber pickups in NYC')

num_neurons=3
rf_size=19
num_channels=2

fig=figure(1)
st.pyplot(fig,clear_figure=False)

for i in range(1000):

    W=randn(num_neurons,
                        num_channels,
                        rf_size,rf_size)

    for row in range(num_neurons):
        W[row,:,:,:]+=3*row



    vmin,vmax=W.min(),W.max()

    W=(W-vmin)/(vmax-vmin)
    vmin,vmax=W.min(),W.max()

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

    # rf=W[0,0,:,:]

    # im=rf

    imshow(im,cmap=cm.gray)
    
    clf()
