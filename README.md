# ML Algorithm Encyclopedia

This repository is a comprehensive, educational resource for machine learning algorithms, implemented entirely from scratch using minimal library dependencies. It serves as both a reference and a learning tool for understanding the core mechanics of ML algorithms without relying on high-level libraries like scikit-learn.

## Overview

Machine learning algorithms are categorized into several types based on their learning approach and application:

- **Supervised Learning**: Algorithms that learn from labeled training data to make predictions on new data.
- **Unsupervised Learning**: Algorithms that find patterns in data without labeled examples.
- **Reinforcement Learning**: Algorithms that learn through interaction with an environment.
- **Natural Language Processing**: Techniques for processing and understanding human language.
- **Utilities**: Helper functions for data preprocessing, model evaluation, and hyperparameter tuning.

## Repository Structure

```
ml-algorithm-encyclopedia/
│
├── README.md                       # Main index with links to all categories
├── requirements.txt                # Common dependencies (numpy, matplotlib, scipy)
│
├── supervised_learning/
│   ├── README.md
│   ├── regression/
│   │   ├── linear_regression.py
│   │   ├── polynomial_regression.py
│   │   ├── ridge_lasso_elasticnet.py
│   │   ├── decision_tree_regressor.py
│   │   ├── random_forest_regressor.py
│   │   ├── gradient_boosting_regressor.py
│   │   ├── svr.py
│   │   └── knn_regressor.py
│   │
│   └── classification/
│       ├── logistic_regression.py
│       ├── k_nearest_neighbors.py
│       ├── support_vector_machine.py
│       ├── naive_bayes.py
│       ├── decision_tree_classifier.py
│       ├── random_forest_classifier.py
│       ├── gradient_boosting_classifier.py
│       ├── perceptron.py
│       └── neural_network_mlp.py
│
├── unsupervised_learning/
│   ├── README.md
│   ├── clustering/
│   │   ├── k_means.py
│   │   ├── hierarchical_clustering.py
│   │   ├── dbscan.py
│   │   └── gaussian_mixture_models.py
│   │
│   └── dimensionality_reduction/
│       ├── pca.py
│       ├── kernel_pca.py
│       ├── t_sne.py
│       └── lda.py
│
├── reinforcement_learning/
│   ├── README.md
│   ├── q_learning.py
│   ├── deep_q_network.py
│   └── policy_gradients.py
│
├── natural_language_processing/
│   ├── README.md
│   ├── bag_of_words.py
│   ├── tf_idf.py
│   ├── word2vec.py
│   └── sentiment_analysis.py
│
└── utilities/
    ├── data_preprocessing.py       # StandardScaler, LabelEncoder, etc.
    ├── model_evaluation.py         # Functions for metrics, plots
    └── hyperparameter_tuning.py    # GridSearchCV, RandomizedSearchCV
```

## Dependencies

- **numpy**: Core numerical computations, array operations
- **matplotlib**: Data visualization and plotting
- **scipy**: Minimal use for advanced math functions (e.g., in t-SNE)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml-algorithm-encyclopedia.git
   cd ml-algorithm-encyclopedia
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the algorithms**
   - Navigate to any category folder (e.g., `cd supervised_learning/regression/`)
   - Run individual algorithm files: `python linear_regression.py`
   - Each file contains complete implementation with examples

## Algorithm Categories

### Supervised Learning

#### Regression (Predicting continuous values)
- **Linear Regression**: Finds linear relationships between features and target
- **Polynomial Regression**: Models non-linear relationships using polynomial features
- **Ridge/Lasso/ElasticNet**: Regularized regression to prevent overfitting
- **Decision Tree Regressor**: Tree-based regression with recursive splitting
- **Random Forest Regressor**: Ensemble of decision trees for improved accuracy
- **Gradient Boosting Regressor**: Sequential tree building to correct errors
- **SVR (Support Vector Regression)**: Finds a tube around predictions with minimal errors
- **KNN Regressor**: Predicts based on average of k nearest neighbors

#### Classification (Predicting categories)
- **Logistic Regression**: Probability-based binary/multiclass classification
- **K-Nearest Neighbors**: Classifies based on majority vote of neighbors
- **Support Vector Machine**: Finds optimal hyperplane for class separation
- **Naive Bayes**: Probabilistic classification using Bayes' theorem
- **Decision Tree Classifier**: Tree-based classification with information gain
- **Random Forest Classifier**: Ensemble of decision trees for robust classification
- **Gradient Boosting Classifier**: Sequential weak learners for strong classification
- **Perceptron**: Single-layer neural network for linear classification
- **Neural Network MLP**: Multi-layer perceptron for complex patterns

### Unsupervised Learning

#### Clustering (Grouping similar data)
- **K-Means**: Partitions data into k clusters based on centroids
- **Hierarchical Clustering**: Builds cluster hierarchy with agglomerative approach
- **DBSCAN**: Density-based clustering for arbitrary shapes and noise
- **Gaussian Mixture Models**: Probabilistic clustering with Gaussian distributions

#### Dimensionality Reduction (Simplifying data)
- **PCA**: Principal Component Analysis for linear dimensionality reduction
- **Kernel PCA**: Non-linear PCA using kernel trick
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for visualization
- **LDA**: Linear Discriminant Analysis for supervised dimensionality reduction

### Reinforcement Learning

- **Q-Learning**: Value-based method learning Q-table for optimal policy
- **Deep Q-Network**: Uses neural network to approximate Q-values
- **Policy Gradients**: Directly optimizes policy parameters using gradients

### Natural Language Processing

- **Bag-of-Words**: Text representation as word frequency vectors
- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting
- **Word2Vec**: Learns word embeddings capturing semantic relationships
- **Sentiment Analysis**: Classifies text sentiment using ML techniques

### Utilities

- **Data Preprocessing**: StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
- **Model Evaluation**: Accuracy, precision, recall, F1, MSE, R2, confusion matrix
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV with cross-validation

## Implementation Details

Each algorithm file follows a consistent structure:

1. **Brief Description**: What the algorithm does and its core intuition
2. **Best For**: Typical use cases and applications
3. **Pros & Cons**: Advantages and limitations
4. **Key Hyperparameters**: Important parameters to tune
5. **Clean, Commented Code**: Complete implementation from scratch
6. **Simple Example**: Usage demonstration with synthetic data
7. **Visualizations**: Plots where applicable (decision boundaries, clusters, etc.)

## Learning Objectives

This encyclopedia helps you:
- Understand the mathematical foundations of ML algorithms
- Implement algorithms without external dependencies
- Compare different approaches for the same task
- Gain intuition for hyperparameter effects
- Build custom ML solutions from basic components

## Contributing

Feel free to contribute by:
- Adding new algorithms
- Improving existing implementations
- Adding more comprehensive examples
- Creating visualizations for complex algorithms

## License

This project is open-source and available under the MIT License.
