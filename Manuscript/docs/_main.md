---
title: Comparing Treatments for Amblyopia with a Synaptic Plasticity Model
subtitle: 
author: Brian S. Blais
tags: [bcm, amblyopia,synaptic plasticity]
toc: true
classoption: onecolumn
colorlinks: true
linestretch: 1.5
secnumdepth: 2
lineno: false
implicit_figures: true

csl: /Users/bblais/tex/bib/apalike.csl
bibliography: /Users/bblais/tex/bib/Amblyopia.bib
---

# Preface {.unnumbered}

These notes are produced with a combination of Obsidian ([https://obsidian.md](https://obsidian.md)), pandoc ([https://pandoc.org](https://pandoc.org)), and some self-styled python scripts ([https://github.com/bblais/Amblyopia-Simulation/tree/main/Manuscript](https://github.com/bblais/Amblyopia-Simulation/tree/main/Manuscript))

## Software Installation {.unnumbered}

The software is Python-based with parts written in Cython.  

- Download the Anaconda Distribution of Python: 

[https://www.anaconda.com/products/individual#downloads](https://www.anaconda.com/products/individual#downloads)  

- Download and extract the *PlasticNet* package at: 

[https://github.com/bblais/Plasticnet/archive/refs/heads/master.zip](https://github.com/bblais/Plasticnet/archive/refs/heads/master.zip)

- Run the script `install.py`

## Printable Versions {.unnumbered}

Printable versions of this report can be found on the GitHub site for this project,

- [Microsoft Word version](https://github.com/bblais/Amblyopia-Simulation/raw/main/Manuscript/docs/Comparing-Treatments-for-Amblyopia-with-a-Synaptic-Plasticity-Model.docx)
- [PDF version](https://github.com/bblais/Amblyopia-Simulation/raw/main/Manuscript/docs/Comparing-Treatments-for-Amblyopia-with-a-Synaptic-Plasticity-Model.pdf)

# Introduction

These notes are an exploration of the problem of modeling Amblyopia and its various treatments from an approach using synaptic plasticity models. The process will involve constructing a simplified mechanism for the development of amblyopic deficits and subsequently modeling both monocular and binocular treatment protocols. The goal is to understand the dynamics of the recovery from amblyopic deficits for the different treatment protocols, to compare the effectiveness of each protocol, and to explore their limitations. Ideally we would like to use these models to inform future protocol parameters and perhaps suggest novel treatments for amblyopia.

In this part we will explore the clinical basis for amblyopia and its treatments. In the @sec:models-of-development-of-amblyopia and @sec:models-of-treatments-for-amblyopia we will explore the models that are used to describe the deficits from amblyopia and their treatment, respectively.


## What is Amblyopia?

Amblyopia is the most common cause of vision loss in children, caused by refractive errors or misalignment of the eyes [@de2007current].   


- Visual acuity
- Contrast sensitivity
- Color
- Depth (Stereopsis)
- Motion
- Visual fields 

## How is it Treated?

The current primary treatment is described in the *Amblyopia Preferred Practice Method* [@wallace2018amblyopia]. Treatments are divided into two broad categories, monocular and binocular treatments. Monocular treatments produce a competition between the two eyes by treating only the fellow eye to that the amblyopic eye recovers.  Binocular treatments seek to stimulate both eyes in such a way that binocular mechanisms can produce a recovery in the amblyopic eye.

### Monocular Treatments

The most common treatment includes 

1. the optical correction of significant refractive errors 
2. patching the dominant eye which forces the visual input to come from only the amblyopic eye. 

Although patching is the most common method of treatment, other methods are described including pharmacology and technology [@holmes2016randomized; @Kelly_2016; @Holmes_2016; @Li:2015aa;@de2007current; @Gao_2018;  @glaser2002randomized]. These include,

3. Pharmacological treatment with atropine drops in the fellow eye

Each of these treatments only directly applies to the fellow eye and the amblyopic eye is left untouched. 

### Binocular Treatments

There are some treatments which are administered to both eyes, making them binocular treatments.  The one that we will be addressing here use virtual reality headsets[@xiao2020improved; @xiao2022randomized],

4. Virtual reality input to both eyes, with contrast modification and/or  dichoptic masks

## Mechanisms for Amblyopia

Since the unequal visual input to the brain can cause alterations in the synaptic pathways leading to a disparity in ocular dominance [@birch2013amblyopia], it is important to understand the possible synaptic effects amblyopia can produce and how potential treatments will either help or hinder the recovery.  


# Methods

In this paper we use a specific model of neural plasticity, the BCM model[@BCM82], to describe the dynamics of the recovery from amblyopia under a number of treatment protocols.  Section @sec:introduction.

## Natural Image Input Environment

In order to approximate the visual system, we start with the following basic properties of the retina, LGN and cortex. There are approximately 1000 photoreceptors feeding into 1 ganglion cell [@JeonEtAl1998;@SterlingEtAl1988]. The retina/LGN responses show a center-surround organization, but with a center diameter less than 1$^o$ [@hubel1995eye]

We use natural scene stimuli for the simulated inputs to the visual system. We start with images taken with a digital camera, with dimensions 1200 pixels by 1600 pixels and 40$^o$ by 60$^o$ real-world angular dimensions (Figure @fig:orig). Photoreceptors have a logarithmic response to the stimulus, so we apply the natural logarithm to the pixel values.  Finally, we model the ganglion responses using a 32x32 pixel center-surround difference-of-Gaussians (DOG) filter to process the images, each pixel representing one photoreceptor (Figure @fig:orig). The center-surround radius ratio used for the ganglion cell is 1:3, with balanced excitatory and inhibitory regions and normalized Gaussian profiles. 

![ Original natural images.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-orig.svg){#fig:orig}



![ A Small Subset of the Natural Images filtered with a base-2 Log function and a difference of Gaussians (DOG)](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-logdog.svg){#fig:logdog}

## Two-eye architecture

Shown in Figure @fig:arch is the visual field, approximated here as a two-dimensional projection, to left and right retinal cells. These left and right retinal cells project to the left and right LGN cells, respectively, and finally to a single cortical cell. The LGN is assumed to be a simple relay, and does not modify the incoming retinal activity.  It is important to understand that the model we are pursuing here is a *single cortical cell* which receives input from both eyes.  We will encounter some limitations to this model which may necessitate exploring multi-neuron systems.  

In the model, normal development is simulated with identical image patches presented to both eyes combined with small independent noise in each eye.  The random noise is generated from a zero-mean normal distribution of a particular variance, representing the natural variation in responses of LGN neurons. Practically, the independent random noise added to each of the two-eye channels avoids the artificial situation of having mathematically identical inputs in the channels.  The development of the deficit and the subsequent treatment protocols are modeled with added preprocessing to these image patches, described later in Sections @sec:models-of-development-of-amblyopia and @sec:models-of-treatments-for-amblyopia.

For all of the simulations we use a 19x19 receptive field, which is a compromise between speed of simulation and the limits of spatial discretization.  We perform at least 20 independent simulations for each condition to address variation in the results.

![ Two-eye architecture.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/arch.png){#fig:arch}

![ A sample of 24 input patches from a normal visual environment. The left- and right-eye inputs are shown in pairs.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-normal_patches.svg){#fig:normal-inputs}


## Synaptic Modification: The BCM Learning Rule

We use a single neuron and the parabolic form of the BCM[@BCM82;@Blais:2008kx] learning rule for all of the simulations, where the synaptic modification depends on the postsynaptic activity, $y$, in the following way for a single neuron

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

![The BCM synaptic modification function.  Units are arbitrary.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-bcm-phi.svg){#fig:bcm-phi}


The results are extremely robust to values of $\eta$  and $\tau$ , which are generally chosen for practical, rather than theoretical, considerations.   Each of these constants is related to the time-step for the simulations, but given the phenomenological nature of the BCM theory it is beyond the scope of this paper to make detailed comparisons between simulation time and real-time.  Further, the fact that $\tau$ can be changed within a factor of 100 with no noticeable effect, the experiments presented here cannot be used address the time-scales of the molecular mechanisms underlying synaptic modification.  Whenever we refer to real-time units for a simulation, we approximate a single simulation iteration as 1 iteration = 0.2 seconds[@phd:Blais98].

In the BCM learning rule, weights decrease if $y$ is less than the modification threshold,$\theta_M$  , and increase if $y$  is greater than the modification threshold.  To stabilize learning, the modification threshold "slides" as a super-linear function of the output.  The output, $y$ , is related to the product of the inputs and the weights via a sigmoidal function, $\sigma(\cdot)$, which places constraints on the values of the output, keeping it in the range -1 and 50.  The interpretation of negative values is consistent with previous work[@BlaisEtAl98], where the activity values are measured relative to spontaneous activity.  Thus, negative values are interpreted as activity below spontaneous.  We continue this usage, in order to more easily compare with previous simulations.  The role of the spontaneous level for the simulations in the natural image environment is discussed elsewhere[@BlaisEtAl98].


## Simulation

The synaptic weights, and the modification threshold, are set to small random initial values at the beginning of a simulation.  At each iteration, an input patch is generated as described above depending on the procedure being simulated and then presented to the neuron.  After each input patch is presented, the weights are modified using the output of the neuron, the input values and the current value of the modification threshold.   In an input environment composed of patches taken from natural images, with equal patches presented to the left- and right-eyes as shown in Figure @fig:normal-inputs, this process orientation selective and fully binocular cells[@BlaisEtAl98].  We then present test stimulus made from sine-gratings with 24 orientations, 20 spatial frequencies, and optimized over phase.  Applying any of the blur filters to the sine gratings does not quantitatively change the result. 

![ (A) Synaptic weights where black denotes weak weights and white denotes strone weights. A clear preference for oriented stimuli can be seen. (B) BCM modification threshold over time.  The value converges to nearly the same level for all neurons. (C) Responses to Oriented Stimuli after training.  Each neuron develops orientation selectivity to a range of optimum angles.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-rf-theta-tuning-curve.svg){#fig:rf-theta-tuning-curve}



## Models of Development of Amblyopia

Amblyopia is a reduction of the best-corrected visual acuity (BCVA) with an otherwise normal eye and has many causes[@wallace2018amblyopia].  Two of the most common forms of amblyopia are strabismic and anisometropic amblyiopia.  Strabismic amblyopia occurs when the inputs from each eye do not converge and the fixating eye becomes dominant over a non-fixating eye.  Refractive amblyopia occurs with untreated unilateral refractive errors, one kind being anisometropic amblyopia where unequal refractive power in each eye leads the retinal image from the amblyopic eye to be blurred relative to the fellow eye.    Both of these processes lead to synaptic plasticity adjustments and interocular competition, enhancing the initial deficit.  

In this work we use a model of the amblyopic deficit caused by two mechanisms.  The first is a blurring of the amblyopic eye inputs, representing refractive amblyopia.  The second is eye-jitter, representing one source of strabismic amblyopia.  We can explore these mechanisms independently and in conjunction to see how they respond differentially to the various treatments.  

### Refractive amblyopia

The amblyopic eye is presented with image patches that have been *blurred* with a normalized Gaussian filter applied to the images with a specified width.  The larger the width the blurrier the resulting filtered image.  Some examples are shown in Figure @fig:blurred-inputs

![ A sample of 24 input patches from a refractive amblyopic environment. The amblyopic (blurred) input is the square on the left-hand side of each pair.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-blurred-inputs.svg){#fig:blurred-inputs}

### Strabismic amblyopia

Strabismic inputs are modeled by changing the center of the left- and right-input patches in a systematic way, with a set mean offset and a standard deviation per input patch generated.  In this way we can model completely overlapping (i.e. normal) inputs, completely non-overlapping (i.e. extreme strabismus), and any amount of overlap in between.  Some examples are shown in Figure @fig:jitter-inputs with the offset locations shown in Figure @fig:jitter-input-locations.

![ A sample of 24 input patches from a strabismic visual environment achieved through random jitter of the amblyopic (left) eye.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-jitter-inputs.svg){#fig:jitter-inputs}

![ Locations of the center of the left- and right-field of view receptive fields, jittered randomly with set mean and standard deviation.  The average receptive fields are shown as gray squares.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-jitter-locations.svg){#fig:jitter-input-locations}


## Models of Treatments for Amblyopia

To model the fix to the refractive imbalance we follow the deficit simulation with an input environment that is rebalanced, both eyes receiving nearly identical input patches (@fig:normal-inputs).   This process is a model of the application of refractive correction.  Although both eyes receive nearly identical input patches, we add independent Gaussian noise to each input channel to represent the natural variation in the activity in each eye.  In addition, in those cases where use employ strabismic amblyopia, the inter-eye jitter is not corrected with the refractive correction.  


### Patch treatment

The typical patch treatment is done by depriving the strong-eye of input with an eye-patch.  In the model this is equivalent to presenting the strong-eye with random noise instead of the natural image input.  Competition between the left- and right-channels drives the recovery, and is produced from the difference between *structured* input into the weak-eye and the *unstructured* (i.e. noise) input into the strong eye.  It is not driven by a reduction in input activity.  @fig:patch-inputs shows sample simulation input patterns from the patched eye.  Compare this to @fig:normal-inputs to see that the simulated patch has far less structure than the normal inputs.

![ A sample of 24 input patches from a patched visual environment. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-patch-inputs.svg){#fig:patch-inputs}


### Contrast modification

A binocular approach to treatment can be produced with contrast reduction of the non-deprived channel relative to the deprived channel. Experimentally this can be accomplished with VR headsets[@xiao2020improved]. In the model we implement this by down-scaling the fellow-eye channel with a simple scalar multiplier applied to each pixel. The contrast difference sets up competition between the two channels with the advantage given to the amblyopic channel.

![ A sample of 24 input patches from a normal visual environment with the right-channel down-scaled relative to the left.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-contrast-modified-inputs.svg){#fig:contrast-modified-inputs}


### Dichoptic Masks

On top of the contrast modification, we can include the application of the dichoptic mask.  In this method, each eye receives a version of the input images filtered through independent masks in each channel, resulting in a mostly-independent pattern in each channel.   It has been observed that contrast modification combined with dichoptic masks can be an effective treatment for amblyopia[@Li:2015aa,@xiao2022randomized].  The motivation behind the application of the mask filter is that the neural system must use both channels to reconstruct the full image and thus may lead to enhanced recovery.  

The dichoptic masks are constructed with the following procedure.  A blank image (i.e. all zeros) is made to which is added 15 randomly sized circles with values equal to 1 (Figure @fig:dichopic_blob A).   These images are then smoothed with a Gaussian filter of a given width, $f$ (Figure @fig:dichopic_blob B).  This width is a parameter we can vary to change the overlap between the left- and right-eye images.  A high value of $f$ compared with the size of the receptive field, e.g. $f=90$, yields a high overlap between the patterns in the weak- and strong-eye inputs (Figure @fig:dichopic_filter_size).  Likewise, a small value of $f$, e.g. $f=10$, the eye inputs are nearly independent -- the patterned activity falling mostly on one of the eyes and not much to both.  Finally, the smoothed images are scaled to have values from a minimum of 0 to a maximum of 1.  This image-mask we will call $A$, and is the left-eye mask whereas the right-eye mask, $F$, is the inverse of the left-eye mask, $F\equiv 1-A$.  The mask is applied to an image by multiplying the left- and right-eye images by the left- and right-eye masks, respectively, resulting in a pair of images which have no overlap at the peaks of each mask, and nearly equal overlap in the areas of the images where the masks are near 0.5 (Figure @fig:dichopic_filter_image).   


![ The dichoptic masks are produced by taking random circular blobs (A), convolving them a Gaussian filter of a specified size (B), resulting in the circular blobs blending into the background smoothly at the edges on the scale of the filter (C)](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/blob_convolution_example_fsig_20.svg){#fig:dichopic_blob}


![ The dichoptic masks for several different filter sizes. The larger the filter, the larger the overlap in the patterns presented to the two eyes. For a very sharp mask (upper left) patterns are nearly all to either the left or the right eye. For a wide mask (lower right) most patterns are presented to both eyes.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/mask_filter_examples_fsigs.svg){#fig:dichopic_filter_size}

![ An example of a dichoptic mask, $\sigma = 20$, applied to one of the images. The mask (A) shows how much of the input goes to each of the left and right channels.  The resulting left- and right-images (B and C, respectively) show the results of the partial independence of the two channels.  One can see areas where there is some overlap as well as areas where the pattern is only present in one of the eyes due to the application of the mask.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/mask_filter_example_fsig_20.svg){#fig:dichopic_filter_image}


### Atropine treatment

In the atropine treatment for amblyopia[@glaser2002randomized], eye-drops of atropine are applied to the strong-eye resulting in blurred vision in that eye.  Here we use the same blurred filter used to obtain the deficit (possibly with a different width) applied to the strong eye (Figure @fig:atropine-inputs).  The difference in sharpness between the strong-eye inputs and the weak-eye inputs sets up competition between the two channels with the advantage given to the weak-eye.


![ A sample of 24 input patches from an environment with atropine applied to the right eye.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-atropine-inputs.svg){#fig:atropine-inputs}

## Quantifying responses

### Ocular Dominance Index

Simulations are ended when selectivity has been achieved and the responses are stable. From the maximal responses of each eye, $R_{\text{left}}$ and $R_{\text{right}}$, individually, we can calculate the ocular dominance index as
$$
\text{ODI} \equiv \frac{R_{\text{right}}-R_{\text{left}}}{R_{\text{right}}+R_{\text{left}}}
$$
The ocular dominance index (ODI) has a value of $\text{ODI} \approx 1$ when stimulus to the right-eye (typically the strong eye in the simulations, by convention) yields a maximum neuronal response with little or no contribution from the left-eye.  Likewise, an ocular dominance index (ODI) has a value of $\text{ODI} \approx -1$ when stimulus to the left-eye (typically the weak eye, by convention) yields a maximum neuronal response with little or no contribution from the right-eye.  A value of $\text{ODI} \approx 0$ represents a purely binocular cell, responding equally to stimulus in either eye.


# Results

## Refractory and Strabismic Amblyopia

Figure @fig:deficit-mu_c-blur shows the production of a deficit effect using both refractory blurring and inter-eye jitter.  Interestingly the larger jitter offset, for small amounts of blur, reduces both the deficit.  While both variables increase the ODI shift to the stronger eye, the jitter has a more modest effect (Figure @fig:deficit-ODI-mu_c-blur).


![ Maximum response for the deprived- and normal-input channels as function of the deficit blur size (in pixels) and the mean jitter offset ($\mu_c$).  Interestingly the larger  jitter offset, for small amounts of blur, reduces both the deficit. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-deficit-mu_c-blur.svg){#fig:deficit-mu_c-blur}

![ Ocular dominance index (ODI) as function of the deficit blur size (in pixels) and the mean jitter offset ($\mu_c$).  Both variables increase the ODI shift to the stronger eye, but the jitter has a more modest effect. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-deficit-ODI-mu_c-blur.svg){#fig:deficit-ODI-mu_c-blur}





# Conclusions and Discussion



# References




