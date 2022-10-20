#!/usr/bin/env python
# coding: utf-8

# # Conclusions
# 
# #todo 
# - [ ] discuss conclusions
# >  This section actually seems a bit superfluous right now, I wonder if we don’t need to try and link the ocular dominance measures with visual acuity
# >  
# >  Instead, we could focus the conclusion on linking the directional conclusions with ODI/day to the existing experimental literature, and making recommendations for future amblyopia treatment studies. Thoughts?
# 
# 
# Now that we have a system of simulation environments to explore, we can compare to experimentally observed rates of recovery.  From [@glaser2002randomized] we have results from several visual protocols.
# 
# 1. Only those patients are included if they had their *refractive error corrected for at least 4 weeks*
# 2. In the patching group most patients received *no more than 6-8 hours of patching per day*
# 3. The resulting improvement in the visual acuity (measured in lines) is given here:
# 
# | Time     | Patch [lines]     | Atropine [lines]         |
# | -------- | --------------- | --------------- |
# | 5 weeks  | $+2.22 \pm 0.2$ | $+1.37 \pm 0.2$ |
# | 16 weeks | $+2.94 \pm 0.2$ | $+2.42 \pm 0.2$ |
# | 24 weeks | $+3.16 \pm 0.2$ | $+2.84 \pm 0.2$ |
# 
# This small amount of data lets us estimate the relative rates of improvement from the treatments.  Since the patch treatment is only about 1/3 day, the total time for treatment would be $19 \text{weeks}\times \frac{7 \text{day}}{1 \text{week}}\times 1/3=44 \text{day}$ For patch treatment with the above data we have a rate of about $0.94 \text{lines} / 44 \text{day}=0.021 \text{lines}/\text{day}$.    Likewise, for atropine, we have a rate of about $1.47\text{lines} / 133 \text{day}=0.011 \text{lines}/\text{day}$.  So the patch treatment is approximately twice as fast as the atropine.  Looking at Figure @fig:dODI_atropine_vs_blur we see that this can put a rough constraint on the parameters.  For a closed-eye noise for the patch treatment of $\sigma_n=0.8$ (recovery rate ODI/day $\sim 0.2$), the atropine treatment must have a lower noise level -- we can look at the atropine parameters which yield recover rates ODI/day $\sim 0.1$).  For little blur, we need a noise level of around $\sigma_n=0.6$, but if the atropine produces a significant blur, then the noise level of those inputs must be much lower -- well below $\sigma_n=0.3$ for blur filter size 6.0, for example.  
# 
# This noise level for atropine is entirely consistent with the same open-eye noise level with the glasses "fix" discussed earlier.  Here we have an independent line of argument to suggest that atropine may blur the natural input, but doesn't change the overall spontaneous activity of neurons.  Further, it suggest that there is a significant physiological different in the activity distributions between unstructure input (e.g. patch) and degraded input (e.g. atropine).  
# 
# In this way we may hope to constrain other parameters of the model by comparing to experimental rates of recovery.  
# 
# 

# In[ ]:




