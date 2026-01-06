---
title: Synaptic Plasticity Model Predicts Optimum Treatment Parameters for Amblyopia
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
# Main Manuscript for

**Synaptic Plasticity Model Predicts Optimum Treatment Parameters for Amblyopia**

Brian S. Blais$^{1*}$ and Eric Gaier$^{2,3}$

$^{1}$Department of Biological and Biomedical Sciences, Bryant University

$^{2}$Picower Institute for Learning and Memory, Massachusetts Institute of Technology

$^{3}$Department of Ophthalmology, Boston Children’s Hospital, Harvard Medical School

$^{*}$Brian Blais corresponding author

**Email**:  bblais@bryant.edu

**Author Contributions**: Paste the author contributions here.

**Competing Interest Statement**: No competing interests.

**Classification**: Biological Sciences - Neuroscience

**Keywords**: amblyopia, synaptic plasticity, vision

**This PDF file includes**:
- Main Text
- Figures 1 to X
- Tables 1 to X


## Abstract

- [ ] Purpose: Extend a biologically motivated synaptic plasticity model (BCM with natural image input) to simulate common amblyopia treatments.
- [ ] Methods: Use feedforward-only architecture in PlasticNet to implement patching, atropine penalization, dichoptic training, and contrast balancing protocols.
- [ ] Findings: Predicts treatment-specific recovery trajectories and identifies optimal stimulus parameters (e.g., contrast levels, patch durations).
- [ ] Implications: Offers quantitative insight into the timing and dosing of interventions, with predictions aligning with clinical results.


Amblyopia is a common cause of visual impairment that results from unequal visual inputs during development, known to manifest through synaptic alterations in the visual cortex. What is not known is the detailed mechanisms of these synaptic changes and how these mechanisms impact the dynamics of recovery. Here we use a computational model of neural plasticity, the Bienenstock, Cooper, and Munro (BCM) model (Bienenstock et al 1982), to compare the dynamics of amblyopia recovery at the neuronal level under several treatment protocols, including optical correction, patching, atropine penalization, and binocular therapies.  We use this model to determine optimal parameters for the treatment of amblyopia, showing that recovery achieved with dichoptic masks combined with an interocular contrast disparity exceeded that of patch and atropine treatments. Further, patch and atropine treatment models produced faster recovery compared to a contrast disparity alone, highlighting the importance of the dichoptic masks.  The rate of recovery depended on treatment features such as the size of the dichoptic masks and the magnitude of the contrast disparity, both experimentally accessible.  In this way, the model suggests optimal values for these modifications.

**Significance Statement**

> Paste your significance statement here. Please note that it should not exceed 120 words, but should be at least 50 words in length. It should not include any references.


# Main Text

## Introduction

- [ ] Brief review of amblyopia development and your prior findings (cite previous paper).
- [ ] Rationale: Most treatments aim to rebalance interocular competition, but empirical optimization is challenging.
- [ ] Common treatments:
	- [ ] Patching: Occlusion of dominant eye.
	- [ ] Atropine: Blurring the dominant eye pharmacologically.
	- [ ] Dichoptic training: Presenting stimuli separately to each eye to promote binocular integration.
	- [ ] Contrast balancing (e.g., Luminopia): Reduce contrast to dominant eye in a movie/game format.
- [ ] Goal: Use a synaptic plasticity model to simulate these treatments and explore parameter sensitivities.
- [ ] Hypothesis: Optimal treatments balance synaptic competition without destabilizing cortical representations.

- [ ] BCM learning and recovery:
	- [ ] Homeostatic and competitive nature is ideal for simulating rebalancing.
	- [ ] Prior modeling of amblyopia treatment: Summary of efforts in literature.
- [ ] PlasticNet advantages:
	- [ ] Realistic inputs (natural images)
	- [ ] Parameter flexibility for fine-grained treatment modeling
- [ ] Clinical motivation: Need for tailored treatments by severity, age, and amblyopia type.
## Results

- [ ] Baseline (no treatment): Persistent suppression of amblyopic eye.
- [ ] Patching:
	- [ ] Effective in early periods; risk of reverse amblyopia with excessive patching.
- [ ] Atropine:
	- [ ] Mimics patching effect with softer recovery dynamics.
- [ ] Dichoptic training:
	- [ ] Strongest recovery in binocular metrics when balanced properly.
	- [ ] Sensitive to contrast parameters.
	- [ ] Contrast reduction (Luminopia-like):
	- [ ] Continuous contrast adaptation of dominant eye yields gradual OD recovery.
- [ ] Optimal parameters:
	- [ ] Identified sweet spots for contrast ratio and patching time per treatment type.
	- [ ] Early intervention window (plasticity-dependent) strongly shapes outcome.
- [ ] Include:
	- [ ] OD histogram evolution over training.
	- [ ] Time-course plots of synaptic weight recovery.
	- [ ] Heatmaps of treatment efficacy by parameter.

## Discussion


- [ ] Key insight: Model confirms and refines clinical understanding of treatment dynamics.
- [ ] Reaffirms sensitive period effects.
- [ ] Shows contrast balancing is more forgiving than patching in timing.
- [ ] Comparison with human data:
	- [ ] Atropine vs. patching trials (e.g., PEDIG).
	- [ ] Recent dichoptic training and Luminopia results.
- [ ] Treatment optimization:
	- [ ] Parameter tuning can guide personalized treatment plans.
- [ ] Limitations:
	- [ ] Feedforward only: no attentional or reward mechanisms modeled.
	- [ ] No behavioral feedback loop.
- [ ] Future work:
	- [ ] Combine with reinforcement learning agent to model perceptual learning tasks.
	- [ ] Add spatial structure (e.g., orientation maps) for richer V1 modeling.
	- [ ] Apply to adult treatment paradigms (model reduced plasticity).
- [ ] BCM-based model predicts differing recovery dynamics under common amblyopia treatments.
- [ ] Early, contrast-based interventions may provide high efficacy with low risk.
- [ ] Synaptic plasticity modeling offers a promising tool for treatment planning and hypothesis generation.



## Materials and Methods

- [ ] Model framework:
- [ ] Same BCM-based PlasticNet model as prior paper.
- [ ] Same strabismus and/or anisometropia manipulations used to induce amblyopia.
- [ ] Treatment protocols modeled:
	- [ ] Patching: One eye’s input occluded (removed) for a percentage of training epochs.
	- [ ] Atropine: One eye blurred (e.g., Gaussian filter) simulating low-acuity vision.
	- [ ] Dichoptic: Present spatially distinct images to each eye with different contrast scaling.
	- [ ] Contrast balancing: Gradually or abruptly reduce contrast to the dominant eye during image input.
- [ ] Evaluation metrics:
	- [ ] Recovery time to balanced ocular dominance.
	- [ ] Receptive field restoration (sharpness, binocular selectivity).
- [ ] Dependence on treatment onset timing and duration.
- [ ] Explored parameters:
	- [ ] % patching time
	- [ ] Contrast ratios
	- [ ] Timing of treatment onset (early vs. late)

## Acknowledgments

Paste your acknowledgments here.



## References

- [ ] Include:
	- [ ] Clinical trials (PEDIG studies, Luminopia trials).
	- [ ] Models of binocular recovery.
	- [ ] Synaptic plasticity and BCM literature.


::: {#refs}
:::

## Figures and Tables


>  This section actually seems a bit superfluous right now, I wonder if we don’t need to try and link the ocular dominance measures with visual acuity
>
>  Instead, we could focus the conclusion on linking the directional conclusions with ODI/day to the existing experimental literature, and making recommendations for future amblyopia treatment studies. Thoughts?


Now that we have a system of simulation environments to explore, we can compare to experimentally observed rates of recovery.  From [@glaser2002randomized] we have results from several visual protocols.

1. Only those patients are included if they had their *refractive error corrected for at least 4 weeks*
2. In the patching group most patients received *no more than 6-8 hours of patching per day*
3. The resulting improvement in the visual acuity (measured in lines) is given here:

| Time     | Patch [lines]     | Atropine [lines]         |
| -------- | --------------- | --------------- |
| 5 weeks  | $+2.22 \pm 0.2$ | $+1.37 \pm 0.2$ |
| 16 weeks | $+2.94 \pm 0.2$ | $+2.42 \pm 0.2$ |
| 24 weeks | $+3.16 \pm 0.2$ | $+2.84 \pm 0.2$ |

This small amount of data lets us estimate the relative rates of improvement from the treatments.  Since the patch treatment is only about 1/3 day, the total time for treatment would be $19 \text{weeks}\times \frac{7 \text{day}}{1 \text{week}}\times 1/3=44 \text{day}$ For patch treatment with the above data we have a rate of about $0.94 \text{lines} / 44 \text{day}=0.021 \text{lines}/\text{day}$.    Likewise, for atropine, we have a rate of about $1.47\text{lines} / 133 \text{day}=0.011 \text{lines}/\text{day}$.  So the patch treatment is approximately twice as fast as the atropine.  Looking at Figure @fig:dODI_atropine_vs_blur we see that this can put a rough constraint on the parameters.  For a closed-eye noise for the patch treatment of $\sigma_n=0.8$ (recovery rate ODI/day $\sim 0.2$), the atropine treatment must have a lower noise level -- we can look at the atropine parameters which yield recover rates ODI/day $\sim 0.1$).  For little blur, we need a noise level of around $\sigma_n=0.6$, but if the atropine produces a significant blur, then the noise level of those inputs must be much lower -- well below $\sigma_n=0.3$ for blur filter size 6.0, for example.

This noise level for atropine is entirely consistent with the same open-eye noise level with the glasses "fix" discussed earlier.  Here we have an independent line of argument to suggest that atropine may blur the natural input, but doesn't change the overall spontaneous activity of neurons.  Further, it suggest that there is a significant physiological different in the activity distributions between unstructure input (e.g. patch) and degraded input (e.g. atropine).

In this way we may hope to constrain other parameters of the model by comparing to experimental rates of recovery.

### Figures and Tables


![Recovery rate (ocular dominance shift per time) for simulated treatments of amblyopia.  The simulations show that for a wide range of contrasts (10%-50%) and mask sizes, the binocular treatment is comparable to the patch and glasses treatment.  For a narrow range of contrasts (around 30-40%) and small overlap with the masks, the binocular treatment is superior to the patch and glasses treatments.](/Users/bblais/Documents/Git/Amblyopia-Simulation/Manuscript/resources/Pasted image 20250416110214.png){#fig:dODI_contrast}

#todo

- [ ] ORCID number attached to the PNAS profile for each author
- [ ] Classifications at [https://www.pnas.org/author-center/submitting-your-manuscript](https://www.pnas.org/author-center/submitting-your-manuscript)
	- Biological Sciences – Neuroscience
	- Biological Sciences - Biophysics and Computational Biology
- [ ]

