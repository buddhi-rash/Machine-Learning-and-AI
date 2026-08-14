# Denoising Diffusion Probabilistic Model (DDPM)

This project is a PyTorch implementation of a Generative Diffusion Model built from scratch. The goal of this project is to translate the complex probabilistic mathematics of Markov chains into an optimized, trainable neural network for image synthesis.

## 🧠 Architecture & Mathematical Approach
* **U-Net Backbone:** Engineered a custom U-Net architecture utilizing residual blocks and spatial downsampling to maximize the efficiency of the reverse diffusion training loop using Pytorch.
* **Attention Mechanisms:** Integrated multi-head self-attention to mitigate noise-induced artifacts and retain high-fidelity spatial details.
* **Variance Scheduling:** Tuned a cosine forward variance schedule for optimal noise injection.
* **Loss Optimization:** Developed a custom training loop that directly minimizes the variational lower bound (ELBO) loss.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch
* **Data Processing:** NumPy, OpenCV
* **Language:** Python

## 🚀 How to Run
1. Clone the repository and navigate to this directory.
2. Install dependencies: `pip install -r requirements.txt`
3. To train the model: `python train.py`
4. To generate images (inference): `python DDPM sample.py --checkpoint path/to/weights`

## 📊 Results (Optional but Recommended)
* **DDPM sample outputs** ![alt text](image.png)
* **DDIM sample outputs** ![alt text](image-1.png)
