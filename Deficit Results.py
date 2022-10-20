#!/usr/bin/env python
# coding: utf-8

# ## Deficit and Measuring the Effectiveness of a Treatment
# 
# Figure @fig:y_vs_t_fix_n0 shows the maximum response to oriented stimuli for $n=20$ neurons versus time.  The first 8 days of simulated time define the *deficit* period, where the neurons start in a naïve state with random synaptic weights, and are presented with natural image input blurred in the weak-eye channel as in Section @sec:methods.  Following the deficit, the simulation proceeds into the *fix* period, where the balanced natural image input is restored.  This transition is marked with a red line in Figure @fig:y_vs_t_fix_n0.  We can see only a modest improvement in the deprived-eye responses to the *fix* treatment.  This treatment depends on the noise level presented to the open eye.  In Figure @fig:y_vs_t_fix_n0, that noise level is $\sigma_n = 0$ whereas in Figure @fig:y_vs_t_fix_n1 the noise level is $\sigma_n=1$.  Increasing the open-eye noise results in an improved recovery from the deficit.  
# 
# Figure @fig:ODI_vs_t_fix_n1 shows a measurement of this recovery, using the oculur dominance index described in Section @sec:ocular-dominance-index.  Balance responses result in an $\text{ODI}=0$.  As the deficit is increased, so the ODI increases toward 1.  After the fix, with high levels of open-eye noise, the neurons nearly all recover from much of their initial deficit -- the ODI nearly returns to $\text{ODI}=0$.  A simple measure of the effectiveness of the treatment is the *rate* of the recovery of the ODI:
# 
# $$
# \text{recovery rate}=\frac{ODI_{\text{deficit}}-ODI_{\text{treatment}}}{\text{duration of treatment}}
# $$
# 

# In[ ]:




