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

These notes are produced with a combination of Obsidian ([https://obsidian.md](https://obsidian.md)), pandoc ([https://pandoc.org](https://pandoc.org)), and some self-styled python scripts ()

## Software Installation {.unnumbered}

The software is Python-based with parts written in Cython.  

- Download the Anaconda Distribution of Python: 

[https://www.anaconda.com/products/individual#downloads](https://www.anaconda.com/products/individual#downloads)  

- Download and extract the *PlasticNet* package at: 

[https://github.com/bblais/Plasticnet/archive/refs/heads/master.zip](https://github.com/bblais/Plasticnet/archive/refs/heads/master.zip)

- Run the script `install.py`

## Printable Versions {.unnumbered}

Printable versions of this report can be found on the GitHub site for this project,

- [Microsoft Word version]()
- [PDF version]()

# Introduction

These notes are an exploration of the problem of modeling Amblyopia and its various treatments from an approach using synaptic plasticity models. The process will involve constructing a simplified mechanism for the development of amblyopic deficits and subsequently modeling both monocular and binocular treatment protocols. The goal is to understand the dynamics of the recovery from amblyopic deficits for the different treatment protocols, to compare the effectiveness of each protocol, and to explore their limitations. Ideally we would like to use these models to inform future protocol parameters and perhaps suggest novel treatments for amblyopia.

In this part we will explore the clinical basis for amblyopia and its treatments. In the @sec-models-of-development and @sec-models-of-treatments we will explore the models that are used to describe the deficits from amblyopia and their treatment, respectively.

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



![fig-logdog.svg](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/fig-logdog.svg){#fig:logdog}


# References




