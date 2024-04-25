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





# Normal and Deprived Visual Environments

![ Natural Image Environment. ](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091111.png){#fig:natural_images}


![ Normal Vision Model.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091213.png){#fig:Pasted_image_20240301091213.png}

![ Model of Eye Jitter.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091231.png){#fig:Pasted_image_20240301091231.png}


![ Architecture for binocular neurons in natural image environment.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091247.png){#fig:Pasted_image_20240301091247.png}

![ Vision Deficit Model.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091305.png){#fig:Pasted_image_20240301091305.png}

![ Optical Fix Model.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091503.png){#fig:Pasted_image_20240301091503.png}

![ Patch Model.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091523.png){#fig:Pasted_image_20240301091523.png}

![ Atropine Model.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091538.png){#fig:Pasted_image_20240301091538.png}

![ Contrast/Mask Model.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20240301091600.png){#fig:Pasted_image_20240301091600.png}






# References