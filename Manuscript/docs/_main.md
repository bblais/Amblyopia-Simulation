---
title: "How to Apply BCM Theory: A Practical Guide Using Amblyopia Treatment as an Example"
shorttitle: BCM and Amblyopia
numbersections: true
bibliography: /Users/bblais/tex/bib/Amblyopia.bib
abstract: This document should serve as an introduction to the BCM Theory [@BCM82] of cortical plasticity, using the development and treatment for amblyopia as the scientific problem to be explored.  We will explore both low-dimensional abstract environments as well as visual environments based on natural-image input to understand the dynamics of synaptic plasticity.
keywords:
  - bcm
  - amblyopia
  - synaptic plasticity
graphics: true
header-includes: \input{config/defs.tex}
toc: true
colorlinks: true
autoSectionLabels: true
---

# Introduction

In this introduction, I want to be clear about as many of the underlying assumptions being made and to highlight where the various results come from in the simulations.   By comparing low-dimension and high-dimension environments, it's my hope that it will make clearer what sorts of changes to the input environment will lead to particular dynamics of synaptic plasticity.  Building up one's intuition this way should help in proposing alternate treatments for amblyopia and understanding how the parameters of those treatments affect their outcomes. 

## The Neuron

The BCM theory[@BCM82] of synaptic plasticity starts with a simplified model of a neuron shown in [Figure @fig:simple_neuron].  In the figure, the model of the neuron has 4 inputs denoted as $x_1, x_2, x_3,$ and $x_4$.  These inputs are connected to the cell via 4 synaptic weights, denoted as $w_1, w_2, w_3,$ and $w_4$.  The output of the cell, $y$, is given as a sum of the input values each multiplied by the synaptic weight connecting it to the neuron passed through an output function, often a sigmoid, $\sigma$:
$$
y=\sigma\left(\sum_i x_i w_i \right)
$$
In this simple model, the value of neural activity $x_i=0$ or $y=0$ refers to *spontaneous activity*, thus one can have negative values for this activity representing *below spontaneous* activity.  The sigmoid function is chosen to have the effect of limiting the spontaneous level of activity, i.e. the neuron has a larger range of activity above spontaneous than below it.

The synapse values, $w_i$, may represent the combination of several synapses -- including inhibitory synapses -- which implies that the weights can take on both positive and negative values.   The number of inputs (4 in the case of  [Figure @fig:simple_neuron]) is arbitrary and will be different from one environment to another.  What is called a "low-dimensional" environment is one with few inputs, likewise a "high-dimensional" environment will have many (possibly hundreds).

![Simple model of a neuron with 4 inputs ($x_1, x_2, x_3,$ and $x_4$), connecting to the cell via 4 synaptic weights ($w_1, w_2, w_3,$ and $w_4$), yielding an output ($y$).](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Simple Neuron.pdf){#fig:simple_neuron}

## Plasticity

The BCM theory states that the synaptic weights change depending on the input to those weights, the output of the neuron, and a sliding threshold according to the following equations,
$$
\frac{dw_i}{dt} = \eta y(y-\theta_M) x_i
$${#eq:bcm1}


$$
\frac{d\theta_M}{dt} = (y^2-\theta_M)/\tau
$${#eq:bcm2}

where is $x_i$ is the $i$th  presynaptic input, $w_i$  is the $i$th synaptic weight, and $y$ is the postsynaptic output activity.  In [Equation @eq:bcm1], the weights increase if the output, $y$, is above the threshold, $\theta_M$, and decrease if the output is below the threshold. The sliding threshold, $\theta_M$, follows postsynaptic activity in a non-linear way ([Equation @eq:bcm2]) and serves to stabilize any runaway weights and the combination of the two equations enforces the selectivity property of the BCM neuron[@phd:Blais98].  The constant, $\eta$, refers to the learning rate and the constant, $\tau$, is what we call the memory constant and is related to the speed of the sliding threshold.  These parameters influence the overall rate of synaptic plasticity and are generally held constant for any series of simulations so that relative dynamics can be compared.


The form of the BCM equations is what is called the *quadratic form* and represents the simplest way of writing the minimum requirements of the BCM theory, but other forms have also been explored[@IntratorCooper92; @LawCooper94].   

![The BCM synaptic modification function.  Units are arbitrary.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-bcm-phi.svg){#fig:fig-bcm-phi.svg}


## Selectivity

One of the properties of the BCM learning rule is that in most environments the neuron becomes *selective* -- it responds to a small subset of the environment and has low or zero responses to the rest of the environment.   We can see this by looking at the response of the neuron, $y$, across the entire environment.  Imagine that there are two possible patterns:

1. input #1 is *strongly* active and input #2 is weakly active
2. input #2 is *strongly* active and input #1 is weakly active

At the start, the BCM neuron responds strongly to both patterns about the same amount of time.  After learning, the BCM becomes selective -- it responds to only one of the patterns strongly most of the time, and the other pattern yields a weak response as shown in [Figure @fig:output_dist_2d].  This distribution is *bimodal* -- there is a cluster of responses above zero and one below zero, separated by a gap of no responses.  When we think of selective responses we often think in terms of bimodal output distributions.  We will see later that this is not a general property of selective responses, however it is an intuitive way of thinking about them.

![The output distribution for a BCM neuron.  The initial distribution (above) shows that the BCM neuron responds strongly to both patterns about the same amount of time.  The final distribution (below) shows that the BCM neuron responds to only one of the patterns strongly most of the time, and the other pattern yields a weak response.  This distribution is *bimodal* -- there is a cluster of responses above zero and one below zero, separated by a gap of no responses.  Also notice that the modification threshold, $\theta_M$, settles on the value of the neural output for the selected pattern.   ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240123131156.png){#fig:output_dist_2d}

Geometrically we can plot the input patterns by plotting the activity of input #1 on the x-axis and the activity of input #2 on the y-axis -- this shows 2 primary clusters, as shown in [Figure @fig:2d_inputs].  

We can also show the values of the weights as a "dot" on this plot, with the weight for input #1 on the x-axis and the weight for input #2 on the y-axis.  Visually, it is easier to see it by drawing a line from the origin to this dot.  The output distribution is formed by projecting a perpendicular from the inputs to this line, as shown in [Figure @fig:2d_initial_weights_outputs] for the initial weights and [Figure @fig:2d_final_weights_outputs] for the final weights.  Geometrically, for the weights to have weak responses to an input pattern then the weights must be *perpendicular to that input pattern*.  One can see this for the final weights and pattern #2 in these figures.

![ 2D input environment.  .](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240123130523.png){#fig:2d_inputs}

![ Geometric interpretation of the weights and output distributions for the *initial weights* in a 2D input environment.  .](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240123130539.png){#fig:2d_initial_weights_outputs}

![ Geometric interpretation of the weights and output distributions for the *final weights* in a 2D input environment.  ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240123130559.png){#fig:2d_final_weights_outputs}

## Natural Images


![ Weights (A) and Output distributions (B) for 12 neurons trained with natural image input patterns.  The weights are shown in grayscale, with black representing low weights and white representing high weights.  The output distribution is shown on a log scale to highlight the few patterns that yield the strongest responses.   ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/weights and output distribution natural images.drawio.png){#fig:nat_image_weights_and_output_dist}

Shown in [Figure @fig:nat_image_weights_and_output_dist] are the weights and output distributions for 12 neurons trained with natural image input patterns.  We easily see that the neurons are selective to orientation -- the final weights are organized in a line at a particular orientation.  Thus, any input pattern that has that same orientation will elicit stronger responses because the pattern will align with the strong weights.  Unlike the low dimensional case earlier, the output distributions for each neuron is *not bimodal*.  The output distribution does not show a cluster above zero and a cluster at zero with a gap in between, it is smeared out over the entire range of the responses.  What makes the neuron selective is the fact that the vast bulk of the responses are near zero, and while there is no gap between low and high responses, the number of patterns yielding strong responses is very small.  Note that the output distributions are being shown on a log scale, otherwise those strong responses would not even be visible.  

This observation means that what BCM neurons select in an environment is not always *clusters* but a small number of patterns in a sea of similar patterns.  Statistically, this means that BCM is sensitive to statistics in the input environment beyond separability and 2-point correlations[@brito2024learning].  In order to understand the dynamics of BCM neurons we can explore environments with patterns that don't fall into neat clusters but instead are spread out with higher order statistics.

## Structure vs Noise

The basic intuition behind the dynamics of the neuron, as it experiences the changing input environment, comes from an understanding of the competition between *structure* and *noise*.  These terms refer to the statistics of the input environment and the mathematical form of the plasticity equations -- different sets of equations will respond to different forms of *structure* and have different dynamics under *noise*.  This can be used to compare different theories of synaptic plasticity[@BlaisEtAl99] but is beyond what we want to discuss here.

In the case of the BCM theory, the *structure* comes in the form of an environment with a minority of patterns yielding high responses and the majority yielding low responses -- the BCM neuron can become selective in this environment.  *Noise* by definition does not have this property.  It is either random variation on top of the structure or random input without the patterns that can lead to high responses.   Gaussian (i.e. normal) noise is an example of such variation.  Examples of structure and noise can be visually seen in [Figure @fig:structure_vs_noise].

![ Structure (left) and noise (right) for a BCM neuron. Shown on the axes are the input values for a two-input neuron, the value of input $x_1$ shown on the x-axis and input $x_2$ on the y-axis.  Each tiny blue dot is a snapshot of the activity of $(x_1,x_2)$ at some time.  Over time, as the neuron experiences the environment, the activity plot fills in the cloud shown.  In the case of structure can see a minority input values that are high, extending out in the x- and y-directions compared to the noise (right figure) where all directions are equal.   In the structured environment, the weights for the BCM neuron converge to those directions such that the postsynaptic response is selective.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240117155651.png){#fig:structure_vs_noise}


The *structure* in this case is generated using a distribution known as the Laplace distribution, or double exponential
$$
P(x) \sim e^{-|x|}
$$

The *noise* is generated using a Gaussian or Normal distribution,
$$
P(x) \sim e^{-x^2}
$$
Mathematically the difference leads to the Laplace environment having more values near zero and a significant number of patterns at very high values (compared to the Gaussian) -- rare patterns.  This property of the Laplace distribution (and some other distributions) is sometimes referred to as having "heavy tails".   This mathematical property is exactly the property the BCM neuron selects for when it learns, so we expect the BCM to become selective in an environment produced by a Laplace distribution vs a Gaussian one.  We can see the property of rare input patterns by looking at the distribution itself as in [Figure @ @fig:laplace_dist], shown on a semi-log plot do highlight those high-value patterns.   Compared to a Gaussian, the Laplace distribution has many more patterns with significantly higher values.  The natural image environment also has a similar structure to the Laplace environment, even for individual pixels as also shown in [Figure @ @fig:laplace_dist].  


![ Single input (pixel) statistics for the Laplace environment and the natural image environment compared to Gaussian (i.e. Normal) noise. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240228124006.png){#fig:laplace_dist}

As the BCM neuron learns, it starts with an output distribution more similar to a Gaussian and learns to respond more to the rare patterns with high activity, shown in [Figure @fig:2D_BCM_Laplace] for a 2D Laplace environment and [Figure @fig:BCM_Natural_Image] for a natural image environment.  The rare patterns are the ones above the BCM modification threshold, $\theta_M$. 

![ Output distributions for a BCM neuron in a 2D Laplace environment.  Shown are (top) the distribution of responses before any learning and (bottom) after learning compared to a Gaussian distribution of responses with the same standard deviation.  In both cases, the responses are smeared across the entire range, with fewer patterns yielding high responses.  Because these plots are on a log scale, Gaussian responses appear curved down as activity increases while Laplace responses appear as a straight decline.  The BCM neuron finds directions in the input space such that its responses are more Laplacian -- they respond more to a smaller group of patterns.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240228124702.png){#fig:2D_BCM_Laplace}


![ Output distributions for a BCM neuron in a natural image environment.  Shown are (top) the distribution of responses before any learning and (bottom) after learning compared to a Gaussian distribution of responses with the same standard deviation.  In both cases, the responses are smeared across the entire range, with fewer patterns yielding high responses.  Because these plots are on a log scale, Gaussian responses appear curved down as activity increases while Laplace responses appear as a straight decline.  The BCM neuron finds directions in the input space such that its responses are more Laplacian -- they respond more to a smaller group of patterns.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240228124932.png){#fig:BCM_Natural_Image}


## Two-eye input environments

Another form of structure occurs with two-eye inputs, where the input values can be nearly identical -- especially in what we call the *normal rearing* environment.  Examples of this are shown in [Figure @fig:2D_NR].  In the artificial case of completely identical (i.e. no input noise, [Figure @fig:2D_NR]A), the changes to the left- and right-eye weights are also identical so the weights can only move along a parallel 45$^{\circ}$ line from their starting position.  Any difference between the left- and right-eye weights is preserved as a result.  The added noise ([Figure @fig:2D_NR]B) sets up a pattern competition, where the structure along the input direction dominates over the noise in the perpendicular direction, and the weights are driven toward the diagonal. Although the neuron becomes selective in both cases, the added noise allows for better selectivity and equality between the left- and right-eye inputs.   As is typical with BCM, the larger the noise the faster the weights are driven toward the structure. 


![ 2D Normal Rearing inputs where the left- and right-eye inputs are identical (A) or nearly identical, but differ with normally distributed noise (B) .](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240601134351.png){#fig:2D_NR_inputs}


![ BCM in a 2D Normal Rearing environment where the left- and right-eye inputs are identical (A) or nearly identical, but differ with normally distributed noise (B).  .  In the case with identical inputs (A), the changes to the left- and right-eye weights are also identical so the weights can only move along a parallel 45$^{\circ}$ line from their starting position. The added noise (B) sets up a pattern competition, where the structure along the input direction dominates over the perpendicular direction, and the weights are driven toward the diagonal. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240602100107.png){#fig:2D_NR_weights}

The structure vs noise competition occurs even when the noise magnitude is *larger* than the noise magnitude, [Figure @fig:2D_NR_high_noise].

![  The same scenario as shown in [Figure @fig:2D_NR_weights]B but with a noise higher in magnitude than the structure, to the point that by eye one can't even see the structure but the BCM neuron still converges to the same direction as maximum structure.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240603065355.png){#fig:2D_NR_high_noise}

### Deprivation

In various deprivation situations we have a sequence where the neuron is allowed to train in a normal environment and then the environment switches such that the structure is changed in one or both eye-inputs.  Common deprivation scenarios include the following:

- Monocular deprivation: (structure,structure) $\rightarrow$ (noise,structure)
- Binocular deprivation: (structure,structure) $\rightarrow$ (noise,noise)
- Contrast deprivation:  (structure,structure) $\rightarrow$ (structure $\times$ contrast factor,structure)
- Reverse occlusion: (structure,structure) $\rightarrow$ (noise,structure) $\rightarrow$ (structure,noise)
- Binocular recovery: (structure,structure) $\rightarrow$ (noise,structure) $\rightarrow$ (structure,structure)

The first three (monocular deprivation,binocular deprivation, and contrast deprivation) are shown in [Figure @fig:NR_MD_BD_CD].

![ BCM under conditions of normal rearing (A), monocular deprivation (B), binocular deprivation (C), and contrast deprivation (D) in a 2D environment.  In each case the weights are driven toward the direction of largest structure.  ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240602095046.png){#fig:Pasted_image_20240602095046.png}




![Pasted image 20240602095113.png](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240602095113.png){#figref:Pasted image 20240602095113.png}




#todo 
- [ ] talk about MD, BD, RS, and BR



# Normal and Deprived Visual Environments

In order to approximate the visual system, we start with the following basic properties of the retina, LGN and cortex. There are approximately 1000 photoreceptors feeding into 1 ganglion cell [@JeonEtAl1998;@SterlingEtAl1988]. The retina/LGN responses show a center-surround organization, but with a center diameter less than 1$^o$ [@hubel1995eye]

We use natural scene stimuli for the simulated inputs to the visual system. We start with images taken with a digital camera, with dimensions 1200 pixels by 1600 pixels and 40$^o$ by 60$^o$ real-world angular dimensions ([Figure @fig:natural_images]). We model the light adaption of the photoreceptors[@bonin2005suppressive;@carandini2012normalization] where the responses reflect the contrast (i.e. difference from the mean, $I_m$) and normalized to the standard deviation of the pixel values, $I_\sigma$, 

$$
R=\frac{I-I_m}{I_\sigma}
$$
Finally, we model the contrast normalization and the suppressive field of the ganglion responses using a 32x32 pixel center-surround difference-of-Gaussians (DOG) filter to process the images ([Figure @fig:normal_vision_model]). The center-surround radius ratio used for the ganglion cell is 1:3, with balanced excitatory and inhibitory regions and normalized Gaussian profiles.   Even in the normal visual environment we explore the role of eye-jitter, where the locations of the left- and right-receptive fields are chosen to deviate from a common center by an average shift, $\mu_c$, and variation, $\sigma_c$ (see [Figure @fig:eye_jitter]). 


![ Natural Image Environment. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091111.png){#fig:natural_images}


![ Normal Vision Model. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091213.png){#fig:normal_vision_model}

![ Model of Eye Jitter. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091231.png){#fig:eye_jitter}


### Two-eye architecture

Shown in [Figure @fig:arch] is the visual field, approximated here as a two-dimensional projection, to left and right retinal cells. These left and right retinal cells project to the left and right LGN cells, respectively, and finally to a single cortical cell. The LGN is assumed to be a simple relay, and does not modify the incoming retinal activity.  It is important to understand that the model we are pursuing here is a *single cortical cell* which receives input from both eyes.  We will encounter some limitations to this model which may necessitate exploring multi-neuron systems.  

In the model, normal development is simulated with identical image patches presented to both eyes combined with small independent noise in each eye.  The random noise is generated from a zero-mean normal distribution of a particular variance, representing the natural variation in responses of LGN neurons. Practically, the independent random noise added to each of the two-eye channels avoids the artificial situation of having mathematically identical inputs in the channels.  The development of the deficit and the subsequent treatment protocols are modeled with added preprocessing to these image patches, described below in Section @sec:models-of-development-and-treatment-of-amblyopia.

For all of the simulations we use a 19x19 receptive field, which is a compromise between speed of simulation and the limits of spatial discretization.  We perform at least 20 independent simulations for each condition to address variation in the results.



![ Architecture for binocular neurons in natural image environment. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091247.png){#fig:arch}


## Ocular Dominance Index

Simulations are ended when selectivity has been achieved and the responses are stable. From the maximal responses of each eye, $R_{\text{left}}$ and $R_{\text{right}}$, individually, we can calculate the ocular dominance index as
$$
\text{ODI} \equiv \frac{R_{\text{right}}-R_{\text{left}}}{R_{\text{right}}+R_{\text{left}}}
$$
The ocular dominance index (ODI) has a value of $\text{ODI} \approx 1$ when stimulus to the right-eye (typically the fellow-eye in the simulations, by convention) yields a maximum neuronal response with little or no contribution from the left-eye.  Likewise, an ocular dominance index (ODI) has a value of $\text{ODI} \approx -1$ when stimulus to the left-eye (typically the amblyopic-eye, by convention) yields a maximum neuronal response with little or no contribution from the right-eye.  A value of $\text{ODI} \approx 0$ represents a purely binocular cell, responding equally to stimulus in either eye.

Recovery and deprivation speed is measured in ODI shift over the time of simulation, with the units of time are arbitrary.  This allows us to compare different treatment protocols and examine how the rates of deprivation vary based on the simulation and environment parameters. 

## Models of Deprivation





## Models of Development and Treatment of Amblyopia

Here we explore the model environments for the initial deficit and the treatments.  Amblyopia is achieved by an imbalance in right and left inputs, and treated with a re-balance or a counter-balance of those inputs (e.g. patching the dominant eye).  In this work, we model the initial deficit as resulting from an asymmetric *blurring* of the visual inputs, as would be produced by a refractive difference between the eyes ([Figure fig:vision_deficit_model]).  The amblyopic eye is presented with image patches that have been *blurred* with a normalized Gaussian filter applied to the images with a specified width.  The larger the width the blurrier the resulting filtered image.  Using a blur filter size of 2.5 pixels produces a robust deficit in the simulations, and thus we use this deficit as the starting point for all of the treatment simulations.   Although both eyes receive nearly identical input patches, we add independent Gaussian noise to each input channel to represent the natural variation in the activity in each eye.  

In addition to the refractive deficit we also model a strabismic jitter in the input pattern, represented as a pixel-shift of the input column and row with means $\mu_c$ and $\mu_r$, respectively, with standard deviations $\sigma_c$ and $\sigma_r$, respectively.  For example, in the simple case of $\mu_c=\mu_r=0$ and $\sigma_c=\sigma_r=0$ the eye inputs are identical (up to intrinsic noise).   If $\mu_c=5$ then the input images for the two channels are shift left-right by a constant 5 pixels.  If $\mu_c>19$ (with 19 being the receptive field size), then the eyes will see completely non-overlapping areas of the image.  

To model the optical fix to the refractive imbalance we follow the deficit simulation with an input environment that is rebalanced, both eyes receiving nearly identical input patches ([Figure @fig:optical_fix_mode]).   This process is a model of the application of glasses.  The optical fix will remove the refractive deficit but will not treat any strabismic jitter. 


### Patch treatment

The typical patch treatment is done by depriving the fellow-eye of input with an eye-patch.  In the model this is equivalent to presenting the fellow-eye with random noise instead of the natural image input ([Figure @fig:patch_model]).  Competition between the left- and right-channels drives the recovery, and is produced from the difference between *structured* input into the amblyopic-eye and the *unstructured* (i.e. noise) input into the fellow-eye.  It is not driven by a reduction in input activity.  As shown in @sec:results, increased *unstructured* input into the previously dominant eye increases the rate of recovery.  This is a general property of the BCM learning rule and has been explored in [Section @sec:structure-vs-noise].


### Atropine treatment

In the atropine treatment for amblyopia[@glaser2002randomized], eye-drops of atropine are applied to the fellow-eye resulting in blurred vision in that eye.  Here we use the same blurred filter used to obtain the deficit (possibly with a different width) applied to the fellow eye ([Figure @fig:atropine_model]) along with an additional amount of noise.  The difference in sharpness between the fellow-eye inputs and the amblyopic-eye inputs sets up competition between the two channels with the advantage given to the amblyopic-eye.

### Contrast modification

A binocular approach to treatment can be produced with contrast reduction of the non-deprived channel relative to the deprived channel. Experimentally this can be accomplished with VR headsets[@xiao2020improved]. In the model we implement this by transforming the image toward the average proportional to a simple scalar contrast value,
$$
I_{\text{new}} = I_{\text{orig}} \cdot \text{constrast} + I_m\cdot(1-\text{contrast})
$$
where $I_m$ is an image consisting of a single uniform gray value of the average of the original image, $I_{\text{orig}}$.  When $\text{constrast}=1$ there is no transformation, whereas a value of $\text{constrast}=0$ results in the image replaced with a uniform gray value of the mean image, $I_m$ ([Figure @fig:contrast_mask_model]). The contrast difference sets up competition between the two channels with the advantage given to the amblyopic-eye channel.

### Dichoptic Mask

On top of the contrast modification, we can include the application of the dichoptic mask ([Figure @fig:contrast_mask_model]).  In this method, each eye receives a version of the input images filtered through independent masks in each channel, resulting in a mostly-independent pattern in each channel.   It has been observed that contrast modification combined with dichoptic masks can be an effective treatment for amblyopia[@Li:2015aa,@xiao2022randomized].  The motivation behind the application of the mask filter is that the neural system must use both channels to reconstruct the full image and thus may lead to enhanced recovery.  

The dichoptic masks are constructed with the following procedure.  A blank image (i.e. all zeros) is made to which is added 15 randomly sized circles with values equal to 1 ([Figure @fig:dichopic_blob]).   These images are then smoothed with a Gaussian filter of a given width, $f$.  This width is a parameter we can vary to change the overlap between the left- and right-eye images.  A high value of $f$ compared with the size of the receptive field, e.g. $f=90$, yields a high overlap between the patterns in the amblyopic- and fellow-eye inputs ([Figure @fig:dichopic_filter_size]).  Likewise, a small value of $f$, e.g. $f=10$, the eye inputs are nearly independent -- the patterned activity falling mostly on one of the eyes and not much to both.  Finally, the smoothed images are scaled to have values from a minimum of 0 to a maximum of 1.  This image-mask we will call $A$, and is applied to left-eye whereas the right-eye mask, $F$, is the inverse of the left-eye mask, $F\equiv 1-A$.  The mask is applied to an image by weighting original image and the uniform gray value average, $I_m$, by the strength of the mask. ,

$$
\begin{aligned}
I_{\text{left}} &= I_{\text{orig}} \cdot A + I_m\cdot(1-A) \\
I_{\text{right}} &= I_{\text{orig}} \cdot F + I_m\cdot(1-F)
\end{aligned}
$$
resulting in a pair of images which have no overlap at the peaks of each mask, and nearly equal overlap in the areas of the images where the masks are near 0.5 ([Figure @fig:dichopic_filter_image]).   



![ Vision Deficit Model. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091305.png){#fig:vision_deficit_model}

![ Optical Fix Model.  ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091503.png){#fig:optical_fix_model}

![ Patch Model. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091523.png){#fig:patch_model}

![ Atropine Model. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091538.png){#fig:atropine_model}

![ Contrast/Mask Model. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091600.png){#fig:contrast_mask_model}



![ Mask from blob.  ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/blob_convolution_example_fsig_20.png){#fig:dichopic_blob}

![ Mask filter size. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/mask_filter_examples_fsigs.png){#fig:dichopic_filter_size}

![mask_filter_example_fsig_20.png](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/mask_filter_example_fsig_20.png){#figref:mask_filter_example_fsig_20.png}

# Results

## Results on Modeling Visual Deprivation

### Normal Rearing (NR)

![ Normal Rearing - Responses](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702201941.png){#fig:Pasted_image_20240702201941.png}

![ Normal Rearing - ODO vs time vs open-eye noise.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702202409.png){#fig:Pasted_image_20240702202409.png}



### Monocular Deprivation (MD)

![ Monocular Deprivation - responses](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702205444.png){#fig:Pasted_image_20240702205444.png}


![ Monocular Deprivation - ODI](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702203436.png){#fig:Pasted_image_20240702203436.png}

### Binocular Deprivation (BD)

![ Binocular Deprivation - responses](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702203836.png){#fig:Pasted_image_20240702203836.png}

![ Binocular Deprivation - ODI](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702203933.png){#fig:Pasted_image_20240702203933.png}


## Results on Modeling Recovery from Visual Deprivation
### Reverse Suture (RS)

![ Reverse Suture - Responses](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702205625.png){#fig:Pasted_image_20240702205625.png}

![ Reverse Suture - ODI](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702205702.png){#fig:Pasted_image_20240702205702.png}

### Binocular Recovery (BR)

![ Binocular Recovery - responses](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702210848.png){#fig:Pasted_image_20240702210848.png}

![ Binocular Recovery - ODI](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240702210907.png){#fig:Pasted_image_20240702210907.png}


## Results on Modeling Treatments for Amblyopia

As expected, the recovery from the deficit using the glasses treatment depends on the open-eye noise, with greater recovery occurring for larger noise level (Figure [@fig:glasses_treatment]).  This is due to the fact that larger presynaptic activity drives larger weight changes, and the lack of structure in the noise drives the neuron to find the structure in the now-identical eye inputs.  The patch treatment (Figure [@fig:patch_treatment]) has a larger effect on recovery, again with larger noise resulting in faster recovery.   The reason for the enhanced recovery speed is that patch treatment sets up a direct competition between the structure presented to the amblyopic eye and the noise presented to the fellow eye.  

Like patch, applying an atropine-like blur the fellow eye results in a recovery (Figure [@fig:atropine_treatment]) that depends on the noise level but also on the magnitude of the blur.  For a sufficiently larger blur, the patch and atropine treatments are roughly comparable.  


![ Glasses Treatment. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240529074233.png){#fig:glasses_treatment}

![ Patch Treatment. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240529074250.png){#fig:patch_treatment}


![ Patch Treatment. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240529074320.png){#fig:atropine_treatment}

Using a treatment constructed from a contrast reduction in the fellow eye along with dichoptic masks can produce a recovery more pronounced than any of the monocular treatments (Figure [@fig:contrast_mask_treatment]).  This recovery depends on the level of contrast and the size of the mask, with smaller masks and a contrast level around 0.3 being optimum.   If the contrast difference is not set low enough, the smaller mask enhances the competition between the fellow eye and the amblyopic eye resulting in a *worsening* of the amblyopia.  However contrast treatment alone doesn't provide an optimal recovery -- the combination of both contrast reduction and the dichoptic mask is needed.  The result is surprisingly robust to eye jitter (Figure [@fig:contrast_mask_treatment_mu9_sigma_9]) with a mean jitter of half of the receptive field and a standard deviation of the jitter equal to the mean giving the same effect, with only added variation as the primary difference.  There does seem to be in decrease in range of contrasts which result in a worsening amblyopia, which may be the result of the jitter giving a more variable deficit in the first place and thus more options for recovery.



![ Contrast Treatment with Dichoptic Mask ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240529092844.png){#fig:contrast_mask_treatment}


![ Contrast Treatment with Dichoptic Mask with jitter $\mu_c=9$, $\sigma_c=9$.  ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240529094751.png){#fig:contrast_mask_treatment_mu9_sigma_9}


#todo 
- [x] do the same sims with the min activity at 0.5





# References