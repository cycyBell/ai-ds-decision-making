# 🚀 A Comparative Study: Classical DS AHP vs. Modern Latent-Based Methods (Probabilistic PCA & VAE)

## 📌 Overview
This repository presents a comparative study between the **Dempster-Shafer Analytic Hierarchy Process (DS AHP)** and **modern latent-based approaches** such as **Probabilistic PCA (PPCA) and Variational Autoencoders (VAE)**. The study aims to evaluate how these methods perform in terms of **feature extraction, decision-making accuracy, and computational efficiency**.

## 📖 Prerequisite Knowledge
Before diving into this repository, familiarity with the following concepts is recommended:
- **Dempster-Shafer Theory (DST)**: A framework for reasoning under uncertainty, extending probability theory.
- **Analytic Hierarchy Process (AHP)**: A classical decision-making method that structures problems into hierarchies.
- **Probabilistic PCA (PPCA)**: An extension of PCA that models data probabilistically.
- **Variational Autoencoder (VAE)**: A generative model that learns latent representations by optimizing a probabilistic lower bound.
- **Machine Learning Basics**: Familiarity with probability distributions, Bayesian inference, and deep learning.

---

## 📊 Methodology
### 🔹 Classical Approach: DS AHP
- DS AHP is an extension of AHP within **Dempster-Shafer Theory**.
- It incorporates uncertainty modeling through belief functions.
- Used in **multi-criteria decision analysis (MCDA)** where alternatives are ranked based on weighted criteria.

### 🔹 Modern Latent-Based Approaches
#### **1️⃣ Probabilistic PCA (PPCA)**
- Probabilistic formulation of PCA that introduces **Gaussian priors**.
- Captures uncertainty in **low-dimensional representations**.
- Helps in dimensionality reduction while preserving probabilistic relationships.

#### **2️⃣ Variational Autoencoder (VAE)**
- Uses an **encoder-decoder framework** to model latent representations.
- Optimizes the **Evidence Lower Bound (ELBO)**, balancing **reconstruction loss** and **KL divergence**.
- Provides a generative approach to data compression and feature extraction.

---

## 🔨 Implementation Details
### ✅ **Implemented Methods**
- **DS AHP (Classical Approach)**: Structured decision-making framework.
- **Probabilistic PCA (PPCA)**: Implemented using **Scikit-learn**.
- **Variational Autoencoder (VAE)**: Implemented using **TensorFlow/Keras**.

### 🗂 **Dataset Used**
- Synthetic dataset sampled from **two distributions**:
  - **Half** from a **normal distribution** (mean = 47, range = 0-100).
  - **Half** from a **uniform distribution** (range = 0-100).
- Dataset shape: **(samples, 10, 12)**.

### 📈 **Evaluation Metrics**
- **Reconstruction Loss** (MSE/Binary Cross-Entropy for PPCA & VAE).
- **KL Divergence** (for VAEs, ensuring latent space regularization).
- **Computational Efficiency** (Training Time & Complexity).
- **Decision-Making Accuracy** (in DS AHP).

---

## 📊 Comparative Analysis
| Method  | Reconstruction Loss | Interpretability | Computation Time |
|---------|--------------------|-----------------|-----------------|
| DS AHP  | Not applicable     | High            | Moderate        |
| PPCA    | Moderate           | Medium          | Low             |
| VAE     | Low                | Low-Medium      | High            |

### 🔍 **Observations**
- **DS AHP is interpretable** but does not provide latent representations.
- **PPCA is computationally efficient** but does not capture complex features.
- **VAE provides the most flexible representations** but at a higher computational cost.

---

## 🚀 How to Run the Code
### 1️⃣ **Installation**
```bash
pip install numpy tensorflow scikit-learn matplotlib
```

### 2️⃣ **Running Experiments**
```bash
python run_ds_ahp.py    # Run DS AHP
python run_ppca.py      # Run Probabilistic PCA
python run_vae.py       # Train and evaluate VAE
```

### 3️⃣ **Interpreting Results**
- **Latent space visualization** (`latent_space_plot.png`).
- **Reconstruction errors** for PPCA & VAE.
- **Ranking results** from DS AHP.

---

## 🤝 Contributing
Contributions are welcome! Feel free to **submit pull requests** for improvements, new features, or dataset enhancements.

---

## 📩 Contact
For questions or collaborations, reach out via **GitHub Issues** or email at `your.email@example.com`.

---

## 📜 Citation
If you find this work useful, please cite it as:
```
@article{YourName2025,
  title={A Comparative Study of DS AHP and Modern Latent-Based Approaches},
  author={Your Name},
  year={2025},
  journal={GitHub Repository}
}
```

Happy coding! 🚀

