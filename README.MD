# ⚖️ Multicriteria Decision Making : A comparative study of the classical Dempster-Shaper Theory-based Analytic Hierarchy Process (DS/AHP) and its Modern Latent-based Machine Learning Approach using Probabilistic Principal Component Analysis (PPCA) and Variational Autoencoders (VAEs).

## Abstract
The Analytic Hierarchy Process (AHP) (Saaty, 1977) is a decision-making framework designed to help decicion makers (DMs) rank or prioritize alternatives, sort information based on multiple criteria via criteria and alternatives pairwise  comparison and decision makers consensus building. A recent advancement, the DS/AHP method, merges Dempster-Shafer theory with AHP, enabling judgments on groups of decision alternatives (DA) while incorporating a measure of uncertainty in the final results. Using the Dempster-Shaper Theory(DST), one can tap into its power of multiple evidence sources combination through the famous Dempter's rule of combination. 

Despite its strengths, DS/AHP lacks a proper and well-defined mathematical structure, making it cumbersome and difficult to manage as the number of criteria and alternatives grows. Plus, evaluating decisions based on multiple, often **conflicting**, criteria can possibly output erroneous results. This study shifts focus toward a modern latent-based approach, blending certain aspects of the classical DS/AHP method and some state-of-art Machine Learning (ML) techniques. However, We first take a helicopter view of the DS/AHP method to provide a high-level overview of DS/AHP to ensure accessibility by abstracting away the more granular and nitty-gritty details in order to keep every reader from getting overwhelmed and confused. This will help you get a better grasp of the intuition underpinning the whole process before transitioning into the Machine Learning paradigm, leveraging Probabilistic Principal Component Analysis (PPCA) and Variational Autoencoders (VAEs). The actual study aims to evaluate how these methods perform in terms of **feature extraction, decision-making accuracy, sensitivity to different scenarios and computational efficiency**, ultimately assessing how ML-based approaches offer a viable alternative to the traditional DS/AHP.

***keywords : Bayesian theory, Basic Probability assignment, Gaussian Mixture, Variational inference, Variationan Autoencoders, Dempter-Shafer Theory, Pairwise comparison, Maximum A Posteriori (MAP), Probabilistic Principal Component Analysis, CNNs, Multicriteria Decision Making, Decision Alternative or Alternative, Criteria.***

## I. Introduction
Decision making processes involving multiple conflicting criteria inevitably requires a systematic and logical information structuring; the consideration of evidence based on several criteria is crucial for handling uncertainty and imprecise information. The Analytic Hierarchy Process (AHP) is one of techniques used by decision-makers (DMs), when it comes to choosing the best alternatives (for example, choosing 25 best candidates among 2000 for scholarship awards) subject to multiple criteria, to rank alternatives through pairwise comparisons by assigning weights (usually using the 1-9 Saaty's scale) based on their relative importance. Over time, enhanced AHP methods have been introduced to improve its ability to handle uncertainty and tackle ambiguity in human decisions. One such improvement is Dempster-Shafer AHP (DS/AHP), the method proposed by Malcom Beynon, which incorporates Dempster-Shafer Theory (DST) in AHP. This combination allows decision-makers to express degrees of belief over a set of alternatives, rather than forcing them into assigning strict probability to each alternative. By using Dempster's rule of combination, DS/AHP aggregates evidence from multiple sources (DMs), offering a more flexible decision-making framework. 

While DS/AHP offers a better bayesian-extended mechanism for handling uncertainty, its mathematical rigidity hinders scalability, making it difficult for optimizations. More critically, decision-making processes often involve complex and high-dimensional data, which traditional AHP approaches, by entirely relying on expert-defined decisions, may struggle to manipulate effectively. These challenges have led researchers to explore modern, data-driven approaches, particularly in the field of Machine Learning (ML). The work of [Ibrahim Yilmaz, Aylin Adem, Metin Dağdeviren, 2023](https://www.sciencedirect.com/science/article/pii/S2352152X23013385) showcased a ML-integrated multi-criteria decision-making approach based on consensus for selection of energy storage locations.

This study investigates the synergy between the traditional **DS/AHP** and some Machine Learning principles in order to redefine multi-criteria decision-making. The main objective is to utilise the power of PPCA and VAES to capture hidden structures in high-dimensional data and learn intrinsic relationships within the data subject to criteria. By integrating Machine Learning , we aim to go past the barriers set by those traditional settings, potentially improving efficiency, scalability, and decision accuracy through sensitivity analysis. Let's have fun by making informed decisions!

## Prerequisite knowledge
Before moving forward with this study, it's crucial for every reader to kwow that prior necessary background knowledge in some aspects of Mathematiques and Machine learning is required. This section presents an overview of those key concepts (core mathematical formulations and the tools used for their implementations) everyone should wrap one's head around to ensure accessibility and clarity : 

* Tensorflow and Tensorflow probability basics;
* Probability distributions and bayesian inference;
*  Mixture Models & Gaussian Mixture Models (GMMs) : **soft clusturing**;
* Variational inference;
* The classical Principal Component Analysis (PCA) : A **dimensionality reduction** technique that transforms high-dimensional data into a lower-dimensional representation while preserving variance. An extended probabilistic version will be used here;
* Variational Autoencoders (VAEs) : a class of **generative models** that encode data into a **latent space** and reconstruct it using a **probabilistic** framework;
* Machine Learning and optimizations ;


Other background understandings are implied but non-negligeable. {Joke} There is no mention of DST or DS/AHP because the brief overview presented afterwards will be enough to stay synced.

## II. Prior Analysis
While this section may appear straightforward to some, these fundamental details provided are essential for downstream tasks.
### II.1. Estimating Decision Makers Priority Values (DMPVs)

Decision Maker Priority Values (**DMPVs**) represent the relative influence of each Decision Maker (**DM**) in the final decision. They are sort of decision makers weights which determine how high is the impact of each DM in a Decision Making process (DMP). If you're familiar with **GMMs**, drawing an analogy to them, DMPVs in **DS-AHP** serve a role similar to **mixing coefficients** in GMMs, determining the contribution of each component (DM) in the aggregation process. 

These values can be derived under the frequentist paradigm as well as updated using the bayesian approaches. Under the frequentist probability framework, we can view a Decision Making process (DMP) that a DM participates in as a random sample from some larger, potentially hypothetical population of DMPs. We can then make a probability statement i.e long-run frequency statements based on this larger population. However, Bayesian inference provides a more **adaptive and probabilistic** way of updating our beliefs about a DM’s competence based on prior knowledge and observed data.

#### II.1.1. Bayesian Inference for DMPVs

Suppose a particular decision maker, $DM_i$, has participated in $n$ decision-making processes, of which $k$ were deemed "good" decisions and $n - k$ were considered "bad." Our goal is to estimate $\theta_i$, the probability that $DM_i$ makes a good decision in any given DMP. Some may  

##### II.1.1.1. Prior Belief
We start with a **prior distribution** over $\theta_i$, which represents our initial belief about the DM’s competence before observing any decisions. We'll make a common choice, the **Beta distribution** (This choice is not random):

> $P(\theta_i) \sim \text{Beta}(\alpha, \beta)$

where:  
- $\alpha$ represents prior successes (belief in competence).  
- $\beta$ represents prior failures (belief in incompetence).  

##### II.1.1.2. Likelihood Function
Given $k$ good decisions out of $n$, we assume a **Binomial likelihood**:

> $P(D \mid \theta_i) = \text{Binomial}(k \mid n, \theta_i) = \binom{n}{k} \theta_i^k (1 - \theta_i)^{n-k}$

##### II.1.1.3. Posterior Update (Bayes' Rule)
Applying **Bayes’ theorem**, we obtain the posterior distribution of $\theta_i$:

> $P(\theta_i \mid D) \propto P(D \mid \theta_i) P(\theta_i)$

Since the **Beta distribution** is a conjugate prior for the **Binomial likelihood**, the posterior also follows a **Beta distribution**:

> $P(\theta_i \mid D) = \text{Beta}(\alpha + k, \beta + (n - k))$

##### II.1.1.3 Posterior mean
This distribution encapsulates our **updated belief** about $DM_i$'s competence based on past performance. The expected value of $\theta_i$, representing the most likely estimate of $DM_i$'s probability of making a good decision, is given by:

> $E[\theta_i] = \frac{\alpha + k}{\alpha + k + \beta + (n - k)}$

#### II.1.2. DMPV calculation
Let's refer to $E[\theta_i]$ as $\mu_i$ for simplicity. Once the expected values $\mu_i$ of $\theta_i$ for all $N$ DMs have been computed, the **Decision Maker Priority Values (DMPVs)**, denoted as **$DMPV(i)$** for the **$DM$** $i$, are then calculated as follows :

> $DMPV(i) = \pi_i = \frac{\lambda_i \times \mu_i}{\sum_{k = 1}^{N} \lambda_k \times \mu_k}$;

Where : 
* $\lambda_i \in [0, 1]$ is a hyperparameter termed the experience coefficient. The higher the number of DMPs the DM has participated in, the closer $\lambda_i$ value is from $1$. The general idea behind this trick is that the most experienced DMs are more likely to be trusted.

This normalizing trick ensures that the DMPVs over all DMs sum to 1 : $\sum_{k = 1}^N \pi_k = 1$.

#### II.1.3. Example 
Suppose $DM_i$ has participated in $n = 15$ decision-making processes, yielding $k = 7$ good decisions and $8$ bad ones. If we assume a **uniform prior** $\alpha = 1, \beta = 1$, the posterior distribution becomes:

$P(\theta_i \mid D) = \text{Beta}(1 + 7, 1 + 8) = \text{Beta}(8,9)$

$E[\theta_i] = \frac{\alpha + k}{\alpha + k + \beta + (n - k)} = \frac{8}{8+9} = \frac{8}{17} \approx 0.47$

This suggests that $DM_i$ has an updated estimated probability of **47%** of making a good decision in future DMPs.

### II.2. Estimating Criteria Priority Values (CPVs)
A profound transformation in multicriteria decision-making methodologies has been observed when the notion of consensus building was integrated. Like DMPVs, Criteria Priority Values serve  as **weighting factors** that quantify the contribution of each criterion to the final decision. As such, they possess the following properties : 

***Non-Negativity***
> $CPV(j) \geq 0, \quad \forall j$

***Normalization***: 
> $\sum_{j=1}^{m} CPV(j) = 1$

Where $m$ is the total number of criteria.

CPVs can be estimated using various methodologies, including **expert judgment (AHP models), statistical learning approaches, and probabilistic modeling**. This study, making use of the consensus-based expert judgments to derive those values through criteria pairwise comparisons, aims to keep the balance between the classical methods and their modern derivatives. 
An algorithm explaining these steps will be provided. However, it is essential to establish some key terminologies that underpin this process. 

During the DMP, each DM performs a **pairwise comparison** of all criteria using the **1-9 scale value**. This results in a $m \times m$ matrix ($m$ for the total number of criteria), commonly known as Pairwise Comparison Matrix (PCM). For the purpose of implementation in this study, the PCM will be referred to as Pairwise comparison Tensor (PCT).

#### II.2.1. The 1-9 Scale Value
The 1-9 Scale Value, also called Saaty's scale, integer-based scoring system used to compare criteria in a structured manner. When comparing a criterion $i$ to a criterion $j$, the scale is interpreted as follws :

 * A score of **1** indicates that the two criteria are equally important.  
 * A score greater than **1** (**2 to 9**) expresses the relative priority of criterion $i$ over criterion $j$.  
 * A reciprocal value (e.g., $1/a, a = 2...9$, if criterion $j$  is thought to be more important than criterion $i$) is used for **consistency** in comparisons.  
For problems with a higher number of criteria, this scale can be expanded.

#### II.2.2. Pairwise Comparison Tensor (PCT)
A **PCT** is a structured $m \times m$ **tensor** that encodes the comparisons made by a **DM**. It is mathematically defined by the following properties:

***Positivity***
> $PCT(i,j) > 0 \quad \forall i, j$;

***Identity Property***
> $PCT(i,i) = 1 \quad \forall i$;

A criterion compared to itself always has a score of **1**.

***Reciprocity***
> $PCT(i,j) =  \frac{1}{PCT(j, i)}$;

This suggests that if criterion $i$ is considered **x** times more important than $j$, then $j$ must be **( 1/x )** times as important as $i$.

#### II.2.3. Consensus-based CPV estimation algorithm
Consensus building is a weel-established concept in the realm of decision making. It refers to a collaborative process in which a group of DMs set their mind to achieve a broad acceptance on the best option by actively working together,  ensuring that all stakeholders' perspectives are considered while addressing potential disagreements through **discussion and compromise**, aiming for a decision that most can support. In some cases, individuals may choose to "stand aside," signaling that while they may not fully lean to the decision, they do not actively oppose it. Consensus is typically achieved when **all or nearly all** members of the group **assent to a decision**, even if some may not fully agree with every aspect of it. 

Over the years, various methods have been developed to build consensus in **Multi-Criteria Decision-Making**. [(You, X., Hou, F. & Lou, Z.)](https://doi.org/10.1007/s12559-021-09913-x) proposed a Consensus Building approach in Multi-criteria Group Decision-Making using **Single-Valued Neutrosophic Sets (SVNS)** to account for uncertainty in expert cognition. We, hereby, describe a **novel consensus-building framework** in which DMs agree on a consensus level **(1% to 100%)**, and the consensus is built using the **"Meet Me Halfway" tradeoff approach**.
A consensus level of **100%** implies that, by the end of the training loop, DMs' PCTs should be highly similar assuming the process went well. Since consensus depends on DMs' opinions, a lower consensus level allows for more variability in the final decision-making process. 

Before introducing the **consensus training loop**, two key concepts must be understood:  

- **Consensus step**: A single pass in the consensus training loop.  
- **Consensus operation**: corresponds to the PCTs updating process when two DMs mutually agree to adjust their evaluations.  

In the "Meet Me Half Way" tradeoff approach, during every consensus step, PCTs pairwise distance calculations are performed, and the consensus operation is conducted on the pair of PCTs with the maximum distance, given that the respective DMs agree to adjust their evaluations.

#### **Consensus Training Loop Steps** 

**(a) Inputs** : 

- $\Omega$ : a three-dimensional tensor, $N \times \text{Shape}(PCT)$ ($N$ for the total number of DMs), containing all N DMs' PCTs;
- $\Xi$ : consensus level (ranging from 0 to 1);
- epochs : maximum number of iteration;
- $\kappa$ : Maximum value in the scale range ($\kappa = 9$ in the 1-9 Saaty's scale);
- $\mathcal{S}$ : a set of all possible DM pairs;

**(b) Variables initialization** :

- Consensus index : $\xi = \frac{\Xi}{2}$;
- Maximum distance allowed : $d_{max} = \frac{(\kappa - 1)(1-\Xi)}{\kappa}$

**(b) Consensus loop**

(1) Compute the pairwise distances between all DMs;

(2) Identify the pair of DMs with the max distance $d_{pmax}$;

(3) If $d_{pmax} \leq d_{max}$  → **Exit the loop** (Consensus condition met);

(4) Ask the identified DM pair if they both agree for a consensus operation;

(5) **If Yes** → Update the pair of PCTs and proceed to step **(7)**;

(6) **If No** → Remove this pair from **$\mathcal{S}$** and proceed to step **(8)**;

(7) If the **maximum number of epochs** is reached → **Exit the loop**, else **repeat from step (1)**;

(8) If the **maximum number of epochs** is reached → **Exit the loop**, else **repeat from step (2)**; 

**(c) Output**

- $\Omega_{new}$ : An updated DMs' PCTs tensor.

The CPV of a criterion $i$ is determined as follows : 
> $CPV(i) = \frac{1}{N \times m}\sum_{k = 1}^N \pi_k \sum_{l = 1}^m \Omega_{new}(k,i,l)$

## III. Overview of the classical DS/AHP method

Investigating the general mathematical formulation of DS/AHP is paramount for the consideration of its analytic structure within the modern ML-based approach. Many papers describe the blueprint basic principles of DS/AHP including its mathematical thorough analysis. However, for the sake of brievety and consistency, the only important fundamental aspects of DST are described which are necessary for the tasks down the line. 
### III.1. Important definitions
**a. Frame of discernment**

The frame of discernment is a finite set $\Theta = {a_1,a_2,\ldots,a_n}$ of all possible alternatives embarked on the DMP.

**b. Basic Probability Assignment**

The basic probability assignment (BPA) is a key function in DST used to make evaluations, decisions based on information fusing from multiple data sources. Also known as a mass function, BPA $m : 2^{\Theta} \to [0,1]$ maps subsets of $\Theta$ to a value between 0 and 1.Since BPA doesn't need to meet all the requirements of a probability measure, it cannot be viewed this way. BPA's main features are :
> $\sum_{x \in 2^{\Theta}}m(x) = 1$ and $m(\emptyset) = 0$.

The notation $2^\Theta$ relates to the power set or the set of all subsets of $\Theta$ including $\emptyset$.

**c. Focal element**

Any subset x of the frame of discernment $\Theta$ for which $m(x)$  is non-zero is called a focal element and represents the exact belief in the proposition depicted by x.

**d. Belief and Plausibility functions**

***Belief*** in a hypothesis-set $\mathcal{A} \subseteq \Theta$ is drawn from the sum of the masses of all subsets of that hypothesis-set. It is the amount of belief that directly supports either the given hypothesis or a more specific one, thus forming a lower bound on its probability. The Belief function $Bel : 2^{\Theta} \to [0,1]$, measuring the strength of the evidence in favor of a proposition $p$, ranges from 0 (indicating no evidence) to 1 (denoting certainty). It's defined by :
> $Bel(\mathcal{A}) = \sum_{\mathcal{B} \subseteq \mathcal{A}}m(\mathcal{B}) \quad \forall \mathcal{A} \subseteq {\Theta}$;

It represents the confidence that a proposition y lies in A or any subset of A.

***Plausibility*** measure is a function $Pl : 2^\Theta \to [0,1]$ representing  the extent to which evidence in favor of $\sim p$ leaves room for belief in $p$.  It can be obtained as the sum of the masses of all sets whose intersection with the hypothesis is not empty. It is an upper bound on the possibility that the hypothesis could be true, because there is only so much evidence that contradicts that hypothesis : 
> $Pl(\mathcal{A}) = \sum_{\mathcal{A} \cap \mathcal{B} \neq \emptyset}m(\mathcal{B}) \quad \forall \mathcal{A} \subseteq {\Theta}$.
### III.2. Dempster's rule of combination
In MCDMs, different DMs express their beliefs over the frame in terms of belief constraints such as in the case of expressing preferences. The problem now is how to combine those independent sets of probability mass assignments in specific situations. Dempster's rule of combination is the appropriate fusion operator used to derive common shared belief between multiple sources and ignores all the conflicting belief through a normalization factor.

Given two sets of masses $m_1$ and $m_2$,  the combination (called the **joint mass**) $m_1 \oplus m_2 : 2^\Theta \to [0,1]$ is calculated in the following manner : 
```math
[m_1 \oplus m_2](y) = 
\begin{cases}
    0 & \text{if } y = \emptyset \\
    \frac{\sum\limits_{\substack{S_1 \cap S_2 = y}} m_1(S_1) m_2(S_2)}{1 - \sum\limits_{\substack{S_1 \cap S_2 = \emptyset}} m_1(S_1) m_2(S_2)} & \text{otherwise}
\end{cases}
```
### III.3. DS/AHP algorithm
Let's say, we have a DMP involving $N$ DMs, $n$ alternatives subject to $m$ criteria
In DS/AHP method, after deriving the DMs and criteria priority values, an analysis is then conducted on alternatives to output the final ranks based on these priority values and alternatives scores (obtained via DMs alternatives comparisons under each criterion). Unlike the pairwise criteria comparisons where criteria are compared between them, alternatives are directly compared to the frame of discernment $\Theta$, the set of all possible alternatives, using a second scale range $1,\cdots, \kappa_2$. These DMs alternatives comparisons provide $A_1, \cdots, A_N$ three-dimensional Knowledge matrices compiled in a four-dimensional matrix $A$.

#### III.3.1. Knowledge matrix
Given a DM $i$, his knowledge matrix $A_i$ is a $m \times d \times d$ decision matrix containing the scores of focal elements proposed by DM $i$ under $m$ criteria, where $d$ is the cardinality of $\Phi \subset \Theta$, the set of all focal elements with a non-zero BPA. Under a criterion $k$, the knowledge matrix $A_i^k$ for DM $i$ is defined as follows : 
```math
A_i^k = 
\begin{pmatrix}
    1 &0 &\cdots &0 &a_1 \cdot CPV(k)\\
    0 &1 &\ddots &\vdots &a_2 \cdot CPV(k)\\
    \vdots &\ddots &\ddots &0 &\vdots\\
    0 &\cdots &0 &1 &{a_{d-1} \cdot CPV(k)}\\
    \frac{1}{a_1 \cdot CPV(k)} &\frac{1}{a_2 \cdot CPV(k)} &\cdots &\frac{1}{a_{d-1} \cdot CPV(k)} &1
 \end{pmatrix}
 ```

Where $a_j = 2, \ldots, \kappa_2$ (alternatives scale range) is the amount of knowledge of the DM on a focal element $j$.

#### III.3.2. BPA calculation
The largest eigenvalue value associated with $A_i^{k}$ is $\gamma_i^{k} = 1+\sqrt{d-1}$ (Proof by Malcolm Beynon 2002).
The normalised values within the eigenvector defined by $(x_1 x_2, \ldots, x_d )$ associated with the largest eigenvalue of the matrix $A_i^{k}$ is :

$x_j = \frac{A_i^{k}(j,d)}{\sum_{l=1}^{d-1}A_i^{k}(l,d) + \sqrt{d-1}}$ and $x_d = \frac{\sqrt{d-1}}{\sum_{l=1}^{d-1}A_i^{k}(l,d) + \sqrt{d-1}}$

The values $xj = (x_1, x_2, \ldots, x_{d-1} )$ and $x_{d}$ constructed can be directly considered the BPA values for $d-1$ groups of DA and $\Theta$,respectively,when the matrix  $A_i^{k}$ is a knowledge matrix.







 


 





This structured approach allows for a mathematically sound and interpretable estimation of **Criterion Priority Values (CPVs)**, which will be further elaborated in the subsequent sections.













```python
    import tensorflow_probability as tfp

    gm = tfp.distributions.MixtureSameFamily()
```



***What's the magic?***

**Multi-Criteria Analysis:** Evaluate decisions based on multiple, often conflicting, criteria.<br>
**Dempster-Shafer Theory:** A powerful framework for handling uncertainty and imprecise information in decision making.<br>
**AI-powered Automation:** Leverage the ds_ahp.ipynb script, a treasure trove of functions, to automate calculations and analysis.

***Why choose this?***

**Data-Driven Decisions:** Make informed choices based on your data, stored conveniently in Docs and Excel files within the input folder.<br>
**Uncertainty Management:** Handle ambiguity and incomplete information inherent in real-world decisions using DS theory.<br>
**Streamlined Workflow:** The provided Jupyter notebook guides you through the analysis process, delivering the final results in the Final results file.


***Target***

Decision-makers facing complex scenarios with multiple criteria.
Data analysts seeking to integrate DS theory into their workflows.
Anyone interested in leveraging AI for more robust decision making.

## Getting Started

***Clone the repository:***

Bash
```
git clone https://github.com/cycyBell/ai-ds-decision-making.git
```

***Prepare your data:***

Place your decision criteria and corresponding data in Docs and Excel files within the input folder. Ensure the data is formatted correctly for the analysis (refer to the notebook for details).

***Run the analysis:***

Open ds_ahp.ipynb in a Jupyter Notebook environment and execute the cells. The notebook will guide you through the process and generate the final results.

***Customization is encouraged!***

The provided notebook serves as a foundation. Feel free to:

Modify the functions within ds_ahp.ipynb to tailor the analysis to your specific needs.
Experiment with different data formats within the input folder.
Extend the script to incorporate additional criteria or decision-making frameworks.
Reach for the Skies
This repository provides a launchpad for exploring the exciting intersection of AI and multi-criteria decision making. Take it further by:

Visualizing the results using libraries like Matplotlib or Seaborn.
Integrating the analysis pipeline into a web application for wider accessibility.
Exploring more advanced decision-making frameworks alongside DS theory.
Let's make informed decisions together, even in the face of uncertainty!
