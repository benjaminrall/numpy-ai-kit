# Numpy AI

[![PyPI Version](https://img.shields.io/pypi/v/numpy-ai-kit)](https://pypi.org/project/numpy-ai-kit/)
[![GitHub Wiki](https://img.shields.io/badge/github-wiki-blue?logo=github)](https://github.com/benjaminrall/numpy-ai-kit/wiki)

Numpy AI is a complete artificial intelligence library built from scratch using pure Python and Numpy. It aims to provide clear reference implementations of many algorithms for educational purposes.

The current documentation for the project can be accessed [here](https://github.com/benjaminrall/numpy-ai-kit/wiki).

## Key Features

- **Keras-style Neural Networks**: Build complex architectures like CNNs using intuitive sequential models. Supports advanced layers and multiple optimisers to give complete control over training.
- **Classical Machine Learning**: A growing suite of `scikit-learn` style estimators. Currently supports a full range of Naive Bayes classifiers.
- **Pure Numpy Backend**: Every algorithm is implemented from scratch using Numpy for maximum transparency and vectorised performance.

## Installation

Install the latest version from PyPI using `pip`:

```shell
pip install numpy-ai-kit
```

Then, you can import and use `numpyai`. See the [wiki](https://github.com/benjaminrall/numpy-ai-kit/wiki) for more details and example usage.

## Project Roadmap

Numpy AI is actively expanding to become a general-purpose AI toolkit. The following modules are currently under development:

### Neural Networks
- **Datasets**: Additional sample datasets including CIFAR-10, CIFAR-100, and Fashion MNIST
- **Layers**: More layer types including Batch Normalisation
- **Infrastructure**: Learning rate schedulers for improved training stability and training callbacks/logging capability

### Utilities
- **Preprocessing**: Encoders, scalers, and binarisers for preprocessing data
- **Feature Extraction**: Vectorisers, feature hasher, HOG (Histogram of Oriented Gradients)
- **Backend**: Generic graph implementation for use with search algorithms

### Classical ML
- **Supervised**: Linear/Logistic Regression, Decision Trees, Random Forests, AdaBoost, SVMs
- **Unsupervised**: Clustering (KMeans, DBSCAN), Gaussian Mixture Models, and Decomposition (PCA, t-SNE)
- **Model Selection**: KFold, ShuffleSplit, and Grid Search for hyperparameter tuning.

### Search & Pathfinding
- **Uninformed**: BFS, DFS, Iterative Deepening, and Dijkstra's
- **Informed**: A* and Greedy Best-First Search
- **Adversarial**: Minimax, Negamax, Expectiminimax, and Monte Carlo Tree Search (MCTS)
- **Local Search**: Hill Climbing, Simulated Annealing, Genetic Algorithms, and Particle Swarm Optimisation (PSO)
- **Constraint Satisfaction**: Backtracking and DLX (Dancing Links)

### Reinforcement Learning
- **Tabular Methods**: Q-Learning and SARSA implementations for discrete state spaces
- **Deep RL**: Integration with the `nn` module for Deep Q-Networks (DQN) and policy gradient methods
- **Environment API**: A standardised interface for creating native environments, plus a compatibility wrapper for Gymnasium environments

## License

This project is licensed under the **MIT License**. See the [`LICENSE`](./LICENSE) file for details.
