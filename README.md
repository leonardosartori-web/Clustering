# Unsupervised Clustering of the MNIST Dataset

This project explores and compares three different unsupervised clustering algorithms applied to the MNIST dataset of handwritten digits. It investigates the impact of dimensionality reduction through Principal Component Analysis (PCA) on clustering performance and computational efficiency.

## Overview

The goal of this study is to partition MNIST images into groups without using their labels during the training process. We compare:
1.  **Gaussian Mixture Model (GMM)**: A probabilistic approach using a custom implementation of the Expectation-Maximization (EM) algorithm with diagonal covariance matrices.
2.  **Normalized Cut (Spectral Clustering)**: A graph-based method that captures global structures by partitioning a similarity graph.
3.  **Mean Shift**: A non-parametric, mode-seeking algorithm that identifies local density modes without requiring a predefined number of clusters.

## Key Features

- **PCA Preprocessing**: Dimensionality reduction from 784 dimensions (28x28 pixels) down to as few as 2 components to analyze the trade-off between information loss and clustering reliability.
- **Custom GMM Implementation**: A from-scratch implementation of the GMM EM algorithm optimized for stability and performance.
- **Fast Rand Index**: An optimized calculation of the Rand Index using a contingency matrix (O(n) complexity) to evaluate clustering quality against ground-truth labels.
- **Comprehensive Benchmarking**: Detailed performance metrics including Rand Index (accuracy) and execution time across various PCA dimensions and hyperparameters.
- **Visualizations**: 
  - 2D PCA projections of cluster assignments.
  - Reconstruction of cluster means from latent space back to the original image space (prototypical digits).
  - Comparison plots for accuracy and computational cost.

## Project Structure

- `main.py`: The primary entry point for running the experiments and collecting results.
- `download_data.py`: Handles fetching the MNIST dataset from OpenML and managing local caching.
- `pca_visualization.py`: Generates 2D visualizations of the dataset and cluster assignments.
- `figures.py`: Script for generating analytical plots and reconstructing GMM prototypes.
- `report.pdf`: A comprehensive academic-style report detailing the methodology, results, and discussion.
- `clustering_results.csv`: Log file containing the raw data from experimental runs.
- `figures/`: Directory containing generated plots and visualizations.

## Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/mnist-clustering.git
    cd mnist-clustering
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Data & Run Experiments**:
    ```bash
    python main.py
    ```
    This will download the MNIST dataset (if not present) and run the full suite of tests across different algorithms and PCA dimensions.

4.  **Generate Visualizations**:
    ```bash
    python figures.py
    python pca_visualization.py
    ```

## Results Summary

- **Normalized Cut** achieved the highest clustering accuracy (Rand Index ~0.93) but was the most computationally expensive.
- **Gaussian Mixture Models** provided the best trade-off between speed and accuracy, proving to be robust across different PCA dimensions.
- **Mean Shift** showed high sensitivity to kernel bandwidth and suffered from scalability issues in higher dimensions, though it effectively identified local density modes.

## Author

- **Leonardo Sartori** - Università Ca' Foscari Venezia

## License

This project is for educational purposes as part of the "Foundations of Artificial Intelligence" course.
