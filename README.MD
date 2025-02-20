# ⚖️ Multicriteria Decision Making : A comparative study of the classical Dempster-Shaper Theory-based Analytic Hierarchy Process (DS/AHP) and its Modern Latent-based Machine Learning Approach using Probabilistic Principal Component Analysis (PPCA) and Variational Autoencoders (VAEs).

## Abstract
The Analytic Hierarchy Process (AHP) is a decision-making framework designed to help decicion makers (DMs) rank alternatives or sort information based on multiple criteria via alternatives pairwise  comparison matrices. A recent advancement, the DS/AHP method, merges Dempster-Shafer theory with AHP, enabling judgments on groups of decision alternatives (DA) while incorporating a measure of uncertainty in the final results. Using the Dempster-Shaper Theory(DST), one can tap into its power of multiple evidence sources combination through the famous Dempter's rule of combination. 

Despite its strengths, DS/AHP lacks a proper and well-defined mathematical structure, making it cumbersome and difficult to manage as the number of criteria and alternatives grows. Plus, evaluating decisions based on multiple, often **conflicting**, criteria can possibly output erroneous results. Tyuphis study shifts focus toward a modern latent-based approach, blending certain aspects of the classical DS/AHP method and some state-of-art Machine Learning (ML) techniques. However, We first take a helicopter view of the DS/AHP method to provide a high-level overview of DS/AHP to ensure accessibility by abstracting away the more granular and nitty-gritty details in order to keep every reader from getting overwhelmed and confused. This will help you get a better grasp of the intuition underpinning the whole process before transitioning into the Machine Learning paradigm, leveraging Probabilistic Principal Component Analysis (PPCA) and Variational Autoencoders (VAEs). The actual study aims to evaluate how these methods perform in terms of **feature extraction, decision-making accuracy, sensitivity to different scenarios and computational efficiency**, ultimately assessing how ML-based approaches offer a viable alternative to the traditional DS/AHP.

***keywords : Bayesian theory, Gaussian Mixture, Dempter-Shafer Theory, Variational inference, Maximum A Posteriori (MAP), Probabilistic Principal Component Analysis, Variationan Autoencoders, CNNs, Multicriteria Decision Making, Decision Alternative or Alternative, Criteria.***

## Introduction
Decision making processes involving multiple conflicting criteria inevitably requires a systematic and logical information structuring; the consideration of evidence based on several criteria is crucial for handling uncertainty and imprecise information. The Analytic Hierarchy Process (AHP) is one of techniques used by decision-makers (DMs), when it comes to choosing the best alternatives (for example, choosing 25 best candidates among 2000 for scholarship awards) subject to multiple criteria, to rank alternatives through pairwise comparisons by assigning weights (usually using the 1-9 Saaty's scale) based on their relative importance. Over time, enhanced AHP methods have been introduced to improve its ability to handle uncertainty and tackle ambiguity in human decisions. One such improvement is Dempster-Shafer AHP (DS/AHP), the method proposed by Malcom Beynon, which incorporates Dempster-Shafer Theory (DST) in AHP. This combination allows decision-makers to express degrees of belief over a set of alternatives, rather than forcing them into assingning strict probability to each alternative. By using Dempster's rule of combination, DS/AHP aggregates evidence from multiple sources (DMs), offering a more flexible decision-making framework.

While DS/AHP offers a better bayesian-extended mechanism for handling uncertainty, its mathematical rigidity hinders scalability, making it difficult for optimizations. More critically, decision-making processes often involve complex and high-dimensional data, which traditional AHP approaches, by entirely relying on expert-defined decisions, may struggle to manipulate effectively. These challenges have led researchers to explore modern, data-driven approaches, particularly in the field of Machine Learning (ML).

This study investigates the synergy between the traditional **DS/AHP** and some Machine Learning principles in order to redefine multi-criteria decision-making. The main objective is to utilise the power PPCA and VAES to capture hidden structures in high-dimensional data and learn intrinsic relationships within the data subject to criteria. By integrating Machine Learning , we aim to go past the barriers set by those traditional settings, potentially improving efficiency, scalability, and decision accuracy through sensitivity analysis. Let's have fun by making informed decisions!


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
