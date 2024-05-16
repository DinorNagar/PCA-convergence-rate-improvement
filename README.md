# PCA-convergence-rate-improvement

During the course *Mathematical methods in data science and signal processing* I enrolled as a Master's degree my final project was to implement and improve the results presented in the article "A Stochastic PCA and SVD Algorithm with an Exponential Convergence Rate".

The arXiv to the original article is [here](https://arxiv.org/abs/1409.2848)

## Getting Started
1. Clone the repository
```bash
git clone https://github.com/DinorNagar/PCA-convergence-rate-improvement.git
```

2. Install the requirements
```bash
pip install -r requirements.txt
```
3. Run the main script
```bash
python main.py
```



## Overview
In the article, the authors implemented the algorithm vr_pca and compared it to power iterations and oja's algorithm and tested them on the mnist dataset. In my work I showed a major improvement in convergence compared to vr_pca and the other algorithms.

## Suggested Improvement
The better optimization can be achieved by changing the constant step size suggested in the article ($\eta$) to dynamicly decaying step size according to the following formula: $\eta_{t} = \frac{\eta}{t}$ where $\eta$ is the constant step size and $t$ is the iteration number.\\
An improvement example can be seen below for the case where k=1:

#### Add the image ####




Final grade - 94


