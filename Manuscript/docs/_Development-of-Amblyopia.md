---
title: The Effect of Strabismus and Anisometropia on the Development of Amblyopia in Synaptic Plasticity Models
bibliography: /Users/bblais/tex/bib/My_Library.bib
geometry:
  - height=11in
  - width=8.5in
  - top=3cm
  - bottom=3cm
  - left=2cm
  - right=2cm
  - headsep=10pt
  - letterpaper
lineno: "True"
autoSectionLabels: true
---

**The Effect of Strabismus and Anisometropia on the Development of Amblyopia in Synaptic Plasticity Models**

Brian S. Blais$^{1*}$ and Eric Gaier$^{2,3}$

$^{1}$Department of Biological and Biomedical Sciences, Bryant University

$^{2}$Picower Institute for Learning and Memory, Massachusetts Institute of Technology

$^{3}$Department of Ophthalmology, Boston Children's Hospital, Harvard Medical School

$^{*}$Brian Blais corresponding author

**Email**:  bblais@bryant.edu

Author Contributions**: Paste the author contributions here.

**Competing Interest Statement**: No competing interests.

**Classification**: Biological Sciences - Neuroscience

**Keywords**: amblyopia, synaptic plasticity, vision

**This PDF file includes**:
- Main Text
- Figures 1 to X
- Tables 1 to X

### Abstract

   - [x] Purpose: Investigate how strabismus and anisometropia affect amblyopia development via BCM-based synaptic plasticity under naturalistic input.
   - [x] Methods: Feedforward model using natural image patches and the BCM rule, implemented in PlasticNet (Python).
   - [x] Results: Strabismus leads to monocular dominance; anisometropia causes synaptic weakening; combined leads to more pronounced suppression.
   - [x] Comparison: Model predictions aligned with observed patterns in human amblyopia.
   - [x] Implications: Insights into cortical development and intervention timing.

---


In this work we explore the development of amblyopia from the combination of strabismus (eye misalignment) and anisometropia (unequal refractive. differences)  with the process of synaptic plasticity.  We introduce a natural-image binocular environment with asymmetry between the channels produced by refractive blurring, eye misalignment and eye jitter where we can compare normal visual development to deprived development.  The cortical responses are simulated using a computational model of neural plasticity, the Bienenstock, Cooper, and Munro (BCM) model[@BCM82].  Simulations are performed to explore ocular dominance changes in normal development, development with strabismic and anisometropic deficits as well as treatment with corrective optics. The results show a remarkable robustness of ocular dominance to strabismic deficits where anisometropia produces strong ocular dominance shifts.

**Significance Statement**

> Paste your significance statement here. Please note that it should not exceed 120 words, but should be at least 50 words in length. It should not include any references.



# Main Text

## Introduction


- [x] Define amblyopia, its causes, and forms (strabismic, anisometropic, combined).
- [x] Emphasize the role of **binocular rivalry** and input decorrelation.
- [x] Present synaptic plasticity as a computational tool for understanding cortical development.
- [x] Highlight why BCM is biologically motivated (homeostatic sliding threshold).
- [x] Gap: Few models apply BCM to naturalistic input under these specific deficits.
- [x] Objective: Use a biologically inspired model to replicate amblyopia development patterns and compare with human data.


Amblyopia is the most common cause of vision loss in children, caused by refractive errors (anisometropic) or misalignment of the eyes (strabismus)[@de2007current,thompsonNeuralPlasticityAmblyopia2021].  Since the unequal visual input to the brain can cause alterations in the synaptic pathways leading to a disparity in ocular dominance[@birch2013amblyopia], it is important to understand the possible synaptic effects amblyopia can produce.  The primary driver of synaptic changes occurs to this mismatch of inputs leading to the later visual areas of the brain.  This mismatch is not due to intensity differences but is the result of different correlations in the input patterns presented to each eye [@blakemoreConditionsRequiredMaintenance1976].

 In this paper we use a specific model of neural plasticity, the BCM model[@BCM82], to describe the development of amblyopia, exploring both the properties of refractive errors and eye misalignment.  We explore the response of the visual system to various input pattern statistics, including natural images, and how these patterns affect the synaptic plasticity of the visual cortex.  The BCM model is particularly well-suited for this task as it incorporates a homeostatic sliding threshold that adapts to the postsynaptic activity, allowing for a biologically motivated understanding of synaptic changes in response to visual input.  Our model does not include a direct mechanism for binocular rivalry, but has the competition between the eyes driven by the temporal competition between patterns as is typical of the BCM model [@BCM82,@blaisRolePresynapticActivity1999,@cooperTheoryCorticalPlasticity2004].  We compare the results of simulation to human clinical data.  This work will form the basis for further study allowing us to explore the treatments for amblyopia, such as patching or corrective optics, and how these treatments can be modeled within the BCM framework.

## Results

   - [x] Control model: Balanced OD, well-formed receptive fields.
   - [x] Strabismic input: Emergence of monocular neurons, OD histogram shift. wrt jitter and blur
   - [x] Anisometropic input: Bilateral input retained but eye with blur shows weak synaptic weight.
   - [x] Combined: Strong suppression of impaired eye; early onset of dominance patterns.
   - [x] Visualizations:
	   - [x] Weight evolution over time
	   - [x] OD histogram shifts
	   - [x] Sample receptive fields
	   - [x] Quantitative comparisons to known amblyopia severity scores

In simulations of normal development using the BCM learning rule within a natural image environment both eyes see identical patches from the scene differing only in some additive random noise.   The neuron develops normal, oriented, binocular receptive fields in this balanced environment as shown in @fig:normal_RF.

### Strabismic and Anisometropic Input

According to the deficit model (@fig:deficit_model), ocular dominance is affected by the strabismic parameters (i.e. the mean receptive field offset, $\mu_c$, and the standard deviation of that offset, $\sigma_c$) and the anisometropic parameter (i.e. the input blur).  The BCM model predicts

@fig:y_vs_t_fix_n0 shows the maximum response to oriented stimuli for $n=20$ neurons versus time.  The neurons start in a naive state with random synaptic weights, and are presented with natural image input, blurred in the amblyopic eye and normal in the fellow eye.  The ocular dominance index changes over time, shifting toward the fellow eye.  Shown in @fig:ODI_blur that while the input blur from the model of anisometry results in significant ocular dominance shifts, the inter-eye shift ($\mu_c$) has little effect on the ocular dominance except at extreme values.  Extreme is defined in terms of the receptive field size, which is 19 for all simulations in this work.    For any given blur, even for a long simulation the ODI does not converge to a value of 1.  The neuron retains binocularity with a bias toward the fellow eye.

Note that for large shift compared to the receptive field size, the cells become nearly monocular -- responding to either the amblyopic or (more often) the fellow eye with little in between.  This is a direct prediction from the BCM learning rule[@clothiaux1991synaptic,@IntratorCooper92]. All of these results are markedly robust to variation in the inter-eye jitter and across wide ranges of the anisometropic blur, as shown in @fig:ODI_jitter_sigma.

### Recovery using optical correction

The "fix" treatment with optical correction reverses the deficit (@fig:optical_correction_responses) bringing the ODI back to zero.  The dynamics of the reversal depends on the noise level in the open eye.  @fig:dODI_fix_vs_noise shows the rate of recovery as a function of this noise.  For low-noise, there is very little improvement.  For large noise, $\sigma_n=1$, the rate achieves 0.14 [ODI/day].  This measure lets us compare different treatments, and determine which are the most effective under the model assumptions.  The experimental observation is that glasses alone are only able to fully restore vision in 27% of amblyopia cases[@wallace2006treatment], where the model would predict that eventually any deficit could be recovered eventually.  This points to something in the physiology not accounted for in the model, but we can assume for practical reasons to use a small open-eye noise and compare rates of recovery rather than the magnitude of recovery.

---

## Discussion

6. Discussion
   - [x] Interpretation:
	   - [x] Strabismus induces competition through decorrelation; anisometropia affects gain.
	   - [x] BCM captures competitive and homeostatic dynamics effectively.
   - [x] Human data alignment:
		   - [ ] Discuss match/mismatch to key findings (e.g., sensitive periods, monocular deprivation effects).
   - [x] Significance:
	   - [x] Model supports hypothesis that distinct deficits have different plasticity signatures.
   - [x] Limitations:
	   - [x] No recurrent, lateral, or bottom-up input
	   - [x] Fixed V1 architecture
	   - [x] Natural images but no eye movements except in the form of random jitter
   - [x] Future work:
	   - [x] Add feedback, simulate treatment (patching), model recovery, motion
	   - [x] Explore different plasticity models in PlasticNet (e.g Hebbian learning)

   - [x] BCM model with natural images reproduces known amblyopia effects from strabismus and anisometropia.
   - [x] Differences in temporal correlation vs. signal strength produce distinct OD changes.
   - [x] Supports early intervention strategies based on synaptic dynamics.

This work extends previous work on the BCM learning rule in natural environments to include anisometropic input combined with a model of eye misalignment and jitter to model the development of amblyopic deficits.

While this model can account for many aspects of the development of amblyopia, there are some limitations that should motivate future work.  There is no model of motion or temporal correlations in the input environment, thus this model as written can't address motion deficits in amblyopic patients.  This limit could be addressed with a more elaborate model of moving natural image scenes and an additional temporal processing in the LGN or cortex.  This model could also be extended to networks, looking at the structure of ocular dominance columns and their organization in deprived environments.  Finally, a more fine-grained temporal model of plasticity such as STDP[@bushReconcilingSTDPBCM2010] or calcium dynamics[@Yeung:2004uq; @Yeung2003437] and its connection to BCM may lead to insights about the role of temporal effects in the development of amblyopia.

Despite the limitations, this model reproduces many aspects of the development of amblyopia and optical correction treatments.  One can see the different contributions of strabismus and anisometropia using the BCM model, as well as the parameter dependence of these contributions.  Future work explores other common treatments, such as eye-patching, where the model can be used to compare the effectiveness of different treatments.

## Materials and Methods

### Natural Image Input Environment

In order to approximate the visual system, we start with the following basic properties of the retina, LGN and cortex. There are approximately 1000 photoreceptors feeding into 1 ganglion cell [@JeonEtAl1998;@SterlingEtAl1988]. The retina/LGN responses show a center-surround organization, but with a center diameter less than 1$^o$ [@hubel1995eye]

We use natural scene stimuli for the simulated inputs to the visual system (@fig:orig). We start with images taken with a digital camera, with dimensions 1200 pixels by 1600 pixels and 40$^o$ by 60$^o$ real-world angular dimensions (@fig:orig). These have intensities, $I$, with mean value $I_m$ and standard deviation $I_\sigma$.  To account for the light adaptation of the photoreceptors where the responses reflect the contrast, or difference from the mean, we get the photoreceptor responses from the image intensities as[@carandini2012normalization]
$$
R=(I-I_m)/I_\sigma
$$
These responses are further processes with the ganglion responses modeled using a 32x32 pixel center-surround difference-of-Gaussians (DOG) filter to process the images, each pixel representing one photoreceptor (@fig:Rdog). The center-surround radius ratio used for the ganglion cell is 1:3, with balanced excitatory and inhibitory regions and normalized Gaussian profiles.

### Two-eye architecture

Shown in @fig:arch is the visual field, approximated here as a two-dimensional projection, to left and right retinal cells. These left and right retinal cells project to the left and right LGN cells, respectively, and finally to a single cortical cell. The LGN is assumed to be a simple relay, and does not modify the incoming retinal activity.  It is important to understand that the model we are pursuing here is a *single cortical cell* which receives input from both eyes.

In the model, normal development is simulated with identical image patches presented to both eyes combined with small independent noise in each eye (@fig:deficit_model).  The random noise is generated from a zero-mean normal distribution of a particular variance, representing the natural variation in responses of LGN neurons. Practically, the independent random noise added to each of the two-eye channels avoids the artificial situation of having mathematically identical inputs in the channels.  The development of the deficit and the subsequent optical correction treatment are modeled with added filters and jitter to these image patches, as described below in Section @sec:model-of-the-development-of-amblyopia.

For all of the simulations we use a 19x19 receptive field, which is a compromise between speed of simulation and the limits of spatial discretization.  To help gather statistics on the responses, we simulate 20 neurons at a time with an orthogonalization process applied to the weights[@hyvarinenFastRobustFixedpoint1999].  This ensures that we don't converge to the same receptive fields multiple times, biasing the counting.

### Synaptic Modification
We use a single neuron and the parabolic form of the BCM[@BCM82;@blaisRecoveryMonocularDeprivation2008] learning rule for all of the simulations, where the synaptic modification depends on the postsynaptic activity, $y$, in the following way for a single neuron

$$
y=\sigma\left(\sum_i x_i w_i \right)
$$
$$
\frac{dw_i}{dt} = \eta y(y-\theta_M) x_i
$$
$$
\frac{d\theta_M}{dt} = (y^2-\theta_M)/\tau
$$

where is $x_i$ is the $i$th  presynaptic input, $w_i$  is the $i$th synaptic weight, and $y$ is the postsynaptic output activity.  The constant, $\eta$, refers to the learning rate and the constant, $\tau$, is what we call the memory-constant and is related to the speed of the sliding threshold. The transfer function, $\sigma(\cdot)$, places minimum and maximum responses given a set of inputs and weights.

The results are extremely robust to values of $\eta$  and $\tau$ , which are generally chosen for practical, rather than theoretical, considerations.   Each of these constants is related to the time-step for the simulations, but given the phenomenological nature of the BCM theory it is beyond the scope of this paper to make detailed comparisons between simulation time and real-time.  Further, the fact that $\tau$ can be changed within a factor of 100 with no noticeable effect, the experiments presented here cannot be used address the time-scales of the molecular mechanisms underlying synaptic modification.  Whenever we refer to real-time units for a simulation, we approximate a single simulation iteration as 1 iteration = 0.2 seconds[@blaisRoleEnvironmentSynaptic1998].

In the BCM learning rule, weights decrease if $y$ is less than the modification threshold,$\theta_M$  , and increase if $y$  is greater than the modification threshold.  To stabilize learning, the modification threshold "slides" as a super-linear function of the output.  The output, $y$ , is related to the product of the inputs and the weights via a sigmoidal function, $\sigma(\cdot)$, which places constraints on the values of the output, keeping it in the range -1 and 50.  The interpretation of negative values is consistent with previous work[@blaisReceptiveFieldFormation1998], where the activity values are measured relative to spontaneous activity.  Thus, negative values are interpreted as activity below spontaneous.  We continue this usage, in order to more easily compare with previous simulations.  The role of the spontaneous level for the simulations in the natural image environment is discussed elsewhere[@blaisReceptiveFieldFormation1998].

The synaptic weights, and the modification threshold, are set to small random initial values at the beginning of a simulation.  At each iteration, an input patch is generated as described above depending on the procedure being simulated and then presented to the neuron.  After each input patch is presented, the weights are modified using the output of the neuron, the input values and the current value of the modification threshold.   In an input environment composed of patches taken from natural images, with equal patches presented to the left- and right-eyes, this process orientation selective and fully binocular cells[@blaisReceptiveFieldFormation1998].  We then present test stimulus made from sine-gratings with 24 orientations, 20 spatial frequencies, and optimized over phase.  Applying any of the blur filters to the sine gratings does not quantitatively change the result.


### Ocular Dominance Index and Recovery

Simulations are ended when selectivity has been achieved and the responses are stable. From the maximal responses of each eye, $R_{\text{left}}$ and $R_{\text{right}}$, individually, we can calculate the ocular dominance index as
$$
\text{ODI} \equiv \frac{R_{\text{right}}-R_{\text{left}}}{R_{\text{right}}+R_{\text{left}}}
$$
The ocular dominance index (ODI) has a value of $\text{ODI} \approx 1$ when stimulus to the right-eye (typically the strong eye in the simulations, by convention) yields a maximum neuronal response with little or no contribution from the left-eye.  Likewise, an ocular dominance index (ODI) has a value of $\text{ODI} \approx -1$ when stimulus to the left-eye (typically the weak eye, by convention) yields a maximum neuronal response with little or no contribution from the right-eye.  A value of $\text{ODI} \approx 0$ represents a purely binocular cell, responding equally to stimulus in either eye.

A simple measure of the effectiveness of the treatment is the *rate* of the recovery of the ODI:

$$
\text{recovery rate}=\frac{ODI_{\text{deficit}}-ODI_{\text{treatment}}}{\text{duration of treatment}}
$$


### Model of the Development of Amblyopia

The model of the vision deficit is shown in @fig:deficit_model.  Amblyopia is achieved by an imbalance in right and left inputs, and treated with a re-balance or a counter-balance of those inputs (e.g. optical correction).  In this work, we model the initial deficit as resulting from an asymmetric *blurring* of the visual inputs, as would be produced by a refractive difference between the eyes.  The amblyopic eye is presented with image patches that have been *blurred* with a normalized Gaussian filter applied to the images with a specified width.  The larger the width the blurrier the resulting filtered image (@fig:blur_filter) which results in a larger ODI shift (@fig:ODI_blur).  Using a blur filter size of 2.5 pixels produces a robust deficit in the simulations, and thus we use this deficit as the starting point for all of the treatment simulations.

We explore the effect of eye misalignment and jitter by randomly shifting the overlap between the eyes each input patch, specified with a mean shift $\mu$ and standard deviation $\sigma$.

To model the fix to the refractive imbalance we follow the deficit simulation with an input environment that is rebalanced, both eyes receiving nearly identical input patches.   This process is a model of the application of the optical correction using glasses.  Although both eyes receive nearly identical input patches, we add independent Gaussian noise to each input channel to represent the natural variation in the activity in each eye.

#todo

- [ ] Figures
	- [x] fig:deficit
	- [x] fig:y_vs_t_fix_n0
	- [x] ODI_blur
	- [x] ODI_jitter_sigma
	- [x] fig:optical_correction_responses
	- [x] fig:dODI_fix_vs_noise
	- [x] blur filter applied to images

   - [x] BCM Theory:
   - [x] Postsynaptic activity-dependent learning.
   - [x] Threshold $\theta_M$ adapts over time, supporting selectivity and stability.
   - [x] Strabismus Effects:
	   - [x] Temporal decorrelation between eyes --> reduced correlated activity --> interocular competition.
   - [x] Anisometropia Effects:
	   - [x] Image degradation (blur, contrast) -->  lower activity --> synaptic weakening.
   - [x] Natural Image Inputs:
   - [x] Better approximates realistic developmental stimuli than synthetic gratings.

---

4. Methods
   - [x] PlasticNet architecture:
   - [x] Feedforward input from two "eyes" (image patches from stereo pairs or simulated disparity).
   - [x] BCM learning rule applied to V1-like units.
   - [x] Input manipulations:
	   - [x] Strabismus: Introduce temporal desynchronization or uncorrelated patches across eyes.
	   - [x] Anisometropia: Blur or reduce contrast in one eye.
	   - [x] Combined: Both manipulations simultaneously.
   - [x] Training protocol:
	   - [x] Exposure to sequences of natural images with/without deficits.
	   - [x] Sliding threshold dynamics calibrated to developmental timescales.
   - [x] Evaluation metrics:
	   - [x] Ocular dominance index (ODI)
	   - [ ] Output activity levels and selectivity?
   - [ ] Comparison:
	   - [ ] Human data on monocular dominance histograms, acuity loss, and contrast sensitivity?


## Acknowledgments

Paste your acknowledgments here.


## References


8. References

   - [x] Classic amblyopia clinical papers
   - [x] BCM theory literature
   - [x] Computational models of V1
   - [x] Human developmental vision studies


::: {#refs}
:::

## Figures and Tables



![Receptive fields formed in a normal, balanced, natural image environment.  Every simulation produces oriented, binocular receptive fields.  Shown are the synaptic weights where white denotes strong weights and black denotes weak weights.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250609133701.png){#fig:normal_RF}






![Binocular vision model.  In the case of developing amblyopia, one channel (the amblyopic eye) has a blur filter before the retinal processing and the two channels have a relative offset and jitter, specified by a mean offset $\mu_c$ and a standard deviation of the offset $\sigma_c$.   The model for optical correction (i.e. with glasses) the model is modified by removing the blur filter but the relative offset and jitter remains unchanged.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250530110214.png){#fig:deficit_model}


![Response to oriented stimuli in a an environment where the amblyopic eye is given blurred natural image inputs and the fellow eye is presented with normal visual input.  The response of the two eye is comparable for some time and then diverge, with a growing disparity between the two eyes.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250624114812.png){#fig:y_vs_t_fix_n0}




![Effect of eye-jitter and blur on ocular dominance.  While the input blur from the model of anisometry results in significant ocular dominance shifts, the eye-jitter has little effect on the ocular dominance except at extreme values.  Note that $\sigma_c=2$ for these simulations.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250624115352.png){#fig:ODI_blur}


![Effect of eye-jitter and blur on ocular dominance.  While the input blur from the model of anisometry results in significant ocular dominance shifts, the eye-jitter has little effect on the ocular dominance except at extreme values.  Note:  $\mu_c=7.5$](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250530114201.png){#fig:ODI_jitter_sigma}

![Response to oriented stimuli in a an environment where the amblyopic eye is given the deficit (blurred natural image inputs and the fellow eye is presented with normal visual input) followed by normal visual input to both eyes representing optical correction.  The response of the two eye is comparable for some time and then diverge, with a growing disparity between the two eyes then the optical correction brings the responses to both eyes closer together.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250625084345.png){#fig:optical_correction_responses}


![Rate of recovery from the optical correction depending in the open-eye noise.  The BCM rule predicts an increase in the rate of recovery with larger channel noise because the synaptic modification is driven partly by presynaptic activity.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250625090120.png){#fig:dODI_fix_vs_noise}



![Original images](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/original_images.png){#fig:orig}


![Retinal responses represented in greyscale, with white and black representing high and low responses, respectively.  The retinal responses are the result of a 1:3 center-surround filter to model the ganglion response properties.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-Rdog.png){#fig:Rdog}





![Two-eye architecture](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250530110113.png){#fig:arch}



![The effect of blurring on the natural images.  Shown are the original (upper left) and progressively larger blur filters of 3, 6, and 9-pixel filters.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250624150519.png){#fig:blur_filter}


