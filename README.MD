# ⚖️ Multicriteria Decision Making : A comparative study of the classical Dempster-Shaper Theory-based Analytic Hierarchy Process (DS/AHP) and its Modern Latent-based Machine Learning Approach using Probabilistic Principal Component Analysis (PPCA) and Variational Autoencoders (VAEs).

## Abstract
The Analytic Hierarchy Process (AHP) (Saaty, 1977) is a decision-making framework designed to help decicion makers (DMs) rank or prioritize alternatives, sort information based on multiple criteria via criteria and alternatives pairwise  comparison and decision makers consensus building. A recent advancement, the DS/AHP method, merges Dempster-Shafer theory with AHP, enabling judgments on groups of decision alternatives (DA) while incorporating a measure of uncertainty in the final results. Using the Dempster-Shaper Theory(DST), one can tap into its power of multiple evidence sources combination through the famous Dempter's rule of combination. 

Despite its strengths, DS/AHP lacks a proper and well-defined mathematical structure, making it cumbersome and difficult to manage as the number of criteria and alternatives grows. Plus, evaluating decisions based on multiple, often **conflicting**, criteria can possibly output erroneous results. This study shifts focus toward a modern latent-based approach, blending certain aspects of the classical DS/AHP method and some state-of-art Machine Learning (ML) techniques. However, We first take a helicopter view of the DS/AHP method to provide a high-level overview of DS/AHP to ensure accessibility by abstracting away the more granular and nitty-gritty details in order to keep every reader from getting overwhelmed and confused. This will help you get a better grasp of the intuition underpinning the whole process before transitioning into the Machine Learning paradigm, leveraging Probabilistic Principal Component Analysis (PPCA) and Variational Autoencoders (VAEs). The actual study aims to evaluate how these methods perform in terms of **feature extraction, decision-making accuracy, sensitivity to different scenarios and computational efficiency**, ultimately assessing how ML-based approaches offer a viable alternative to the traditional DS/AHP.

***keywords : Bayesian theory, Basic Probability assignment, Gaussian Mixture, Variational inference, Variationan Autoencoders, Dempter-Shafer Theory, Pairwise comparison, Maximum A Posteriori (MAP), Probabilistic Principal Component Analysis, CNNs, Multicriteria Decision Making, Decision Alternative or Alternative, Criteria.***

## Introduction
Decision making processes involving multiple conflicting criteria inevitably requires a systematic and logical information structuring; the consideration of evidence based on several criteria is crucial for handling uncertainty and imprecise information. The Analytic Hierarchy Process (AHP) is one of techniques used by decision-makers (DMs), when it comes to choosing the best alternatives (for example, choosing 25 best candidates among 2000 for scholarship awards) subject to multiple criteria, to rank alternatives through pairwise comparisons by assigning weights (usually using the 1-9 Saaty's scale) based on their relative importance. Over time, enhanced AHP methods have been introduced to improve its ability to handle uncertainty and tackle ambiguity in human decisions. One such improvement is Dempster-Shafer AHP (DS/AHP), the method proposed by Malcom Beynon, which incorporates Dempster-Shafer Theory (DST) in AHP. This combination allows decision-makers to express degrees of belief over a set of alternatives, rather than forcing them into assigning strict probability to each alternative. By using Dempster's rule of combination, DS/AHP aggregates evidence from multiple sources (DMs), offering a more flexible decision-making framework.

While DS/AHP offers a better bayesian-extended mechanism for handling uncertainty, its mathematical rigidity hinders scalability, making it difficult for optimizations. More critically, decision-making processes often involve complex and high-dimensional data, which traditional AHP approaches, by entirely relying on expert-defined decisions, may struggle to manipulate effectively. These challenges have led researchers to explore modern, data-driven approaches, particularly in the field of Machine Learning (ML).

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

## Prior Analysis
While this section may appear straightforward to some, these fundamental details provided are essential for downstream tasks.
### 1. Estimating Decision Makers Priority Values (DMPVs)

Decision Maker Priority Values (**DMPVs**) represent the relative influence of each Decision Maker (**DM**) in the final decision. They are sort of decision makers weights which determine how high is the impact of each DM in a Decision Making process (DMP). If you're familiar with **GMMs**, drawing an analogy to them, DMPVs in **DS-AHP** serve a role similar to **mixing coefficients** in GMMs, determining the contribution of each component (DM) in the aggregation process. 

These values can be derived under the frequentist paradigm as well as updated using the bayesian approaches. Under the frequentist probability framework, we can view a Decision Making process (DMP) that a DM participates in as a random sample from some larger, potentially hypothetical population of DMPs. We can then make a probability statement i.e long-run frequency statements based on this larger population. However, Bayesian inference provides a more **adaptive and probabilistic** way of updating our beliefs about a DM’s competence based on prior knowledge and observed data.

### Bayesian Inference for DMPVs

Suppose a particular decision maker, **\( DM_i \)**, has participated in **\( n \)** decision-making processes, of which **\( k \)** were deemed "good" decisions and **\( n - k \)** were considered "bad." Our goal is to estimate **\( \theta_i \)**, the probability that **\( DM_i \)** makes a good decision in any given DMP. Some may 

#### 1. Prior Belief
We start with a **prior distribution** over **\( \theta_i \)**, which represents our initial belief about the DM’s competence before observing any decisions. We'll make a common choice, the **Beta distribution** (This choice is not random):

\[
P(\theta_i) \sim \text{Beta}(\alpha, \beta)
\]

where:  
- \( \alpha \) represents prior successes (belief in competence).  
- \( \beta \) represents prior failures (belief in incompetence).  

#### 2. Likelihood Function
Given **\( k \)** good decisions out of **\( n \)**, we assume a **Binomial likelihood**:

\[
P(D \mid \theta_i) = \text{Binomial}(k \mid n, \theta_i) = \binom{n}{k} \theta_i^k (1 - \theta_i)^{n-k}
\]

#### 3. Posterior Update (Bayes' Rule)
Applying **Bayes’ theorem**, we obtain the posterior distribution of **\( \theta_i \)**:

\[
P(\theta_i \mid D) \propto P(D \mid \theta_i) P(\theta_i)
\]

Since the **Beta distribution** is a conjugate prior for the **Binomial likelihood**, the posterior also follows a **Beta distribution**:

\[
P(\theta_i \mid D) = \text{Beta}(\alpha + k, \beta + (n - k))
\]

#### 4. Example Calculation
Suppose **\( DM_i \)** has participated in **\( n = 15 \)** decision-making processes, yielding **\( k = 7 \)** good decisions and **\( 8 \)** bad ones. If we assume a **uniform prior** (\( \alpha = 1, \beta = 1 \)), the posterior distribution becomes:

\[
P(\theta_i \mid D) = \text{Beta}(1 + 7, 1 + 8) = \text{Beta}(8,9)
\]

This distribution encapsulates our **updated belief** about **\( DM_i \)**'s competence based on past performance. The expected value of **\( \theta_i \)**, representing the most likely estimate of **\( DM_i \)**'s probability of making a good decision, is given by:

\[
E[\theta_i] = \frac{\alpha + k}{\alpha + k + \beta + (n - k)} = \frac{8}{8+9} = \frac{8}{17} \approx 0.47
\]

This suggests that **\( DM_i \)** has an updated estimated probability of **47%** of making a good decision in future DMPs.

### Conclusion

This Bayesian framework allows for a **dynamically updated** estimation of each DM’s influence, adjusting beliefs based on new observations while incorporating prior knowledge. Such an approach enhances the robustness of **DS-AHP**, reducing the risk of over-reliance on **limited historical data** while maintaining a probabilistic structure that adapts to new evidence.












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
