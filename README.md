# ml-from-scratch

This repository serves to showcase my foundational understanding and practical implementation of various core machine learning algorithms. All algorithms are implemented "from scratch" in Python, emphasizing a deep understanding of their underlying mathematical principles and mechanics.

## Project Highlights & Implemented Algorithms

This project demonstrates proficiency in data preprocessing, model implementation, evaluation, and comparative analysis across a range of machine learning tasks.

### 1. Exploratory Data Analysis (EDA) on Fuel Economy Data

**Objective:** To thoroughly understand a real-world dataset by performing comprehensive exploratory data analysis, identifying key relationships, and preparing for predictive modeling.

* **Key Techniques:** Data loading and cleaning, descriptive statistics generation, extensive data visualization (e.g., histograms, scatter plots, box plots, correlation matrices).
* **Insights:** Explored the distribution categorical and continuous variables, investigated relationships, addressed considerations for excluding irrelevant or redundant features and outlined strategies for robust model evaluation to mitigate overfitting and underfitting.

### 2. Fundamental Regression & Classification Models

**Objective:** To build foundational predictive models from scratch, demonstrating an understanding of their core mechanics and evaluation.

#### A. Regression Task (Boston Housing Dataset)

* **Algorithms Implemented:**
    * **Ordinary Least Squares (OLS) Regression:** Implemented the normal equation for direct parameter estimation.
    * **Ridge Regression:** Incorporated L2 regularization to address multicollinearity and prevent overfitting.
    * **Lasso Regression:** Implemented L1 regularization for feature selection and sparsity.
* **Key Aspects:** Comprehensive data preprocessing (normalization, one-hot encoding where applicable), training and evaluation on unseen data, and a comparative analysis of model parameters and performance metrics (Mean Squared Error, R-squared) against established API implementations.

#### B. Classification Task (Breast Cancer Wisconsin Dataset)

* **Algorithms Implemented:**
    * **Gaussian Naive Bayes (GNB):** Applied probabilistic classification based on Bayes' theorem, considering both shared and class-specific covariances for Gaussian Discriminant Analysis (GDA).
    * **Logistic Regression:** Implemented using gradient descent to model the probability of a binary outcome.
    * **Perceptron:** Implemented the foundational algorithm for binary classification, focusing on its learning rule.
* **Key Aspects:** Thorough data preprocessing (normalization, splitting), training and evaluation on test sets, comparative analysis of model performance using metrics like accuracy, precision, recall, and F1 score, and discussion on linear separability of the dataset.

### 3. Advanced Models: SVM, MLP, and RNN

**Objective:** To implement more complex machine learning models, including deep learning concepts, and explore their performance on various datasets.

* **Support Vector Machines (SVM):** Implemented SVM for classification on synthetic half-circles and moons datasets, exploring different kernels (Linear, Polynomial, RBF, Sigmoid) to understand their effect on decision boundaries and model performance. Extended to multi-class and multi-label classification using strategies like One-vs-Rest or One-vs-One.
* **Multi-layer Perceptron (MLP):** Implemented a simple MLP for classification, demonstrating the principles of neural networks, forward propagation, and backpropagation.
* **Recurrent Neural Networks (RNNs):** Explored the capabilities of RNNs for sequential data by designing a time-series signal with long-term dependencies and implementing an RNN from scratch to capture these patterns. Investigated the impact of 'memory size' on performance.

### 4. Dimensionality Reduction Methods

**Objective:** To apply and compare various dimensionality reduction techniques for high-dimensional data visualization and analysis.

* **Techniques Implemented:**
    * **t-Distributed Stochastic Neighbor Embedding (t-SNE):** Used for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.
    * **Principal Component Analysis (PCA):** Applied for linear dimensionality reduction, transforming data to a new coordinate system where axes represent principal components.
    * **Auto-Encoder (Neural Network-Based):** Implemented a neural network architecture for unsupervised dimensionality reduction, learning efficient data representations.
* **Application:** Applied these techniques to the MNIST dataset of handwritten digits to reduce their dimensionality to 2D for visualization.
* **Analysis:** Compared the effectiveness of each method in terms of representation quality (how well digits are clustered and separated), computational efficiency, scalability, and provided insights into their underlying mechanisms.
* **Literature Review:** Summarized and critically analyzed the seminal paper "Visualizing Data using t-SNE" to deepen understanding of the algorithm's advantages, limitations, and real-world applications.

## Technologies Used

* Python
* NumPy (for matrix operations)
* Pandas (for data handling and analysis)
* Matplotlib (for plotting)
* Seaborn (for statistical data visualization)
* Scikit-learn (for comparison and specific data utilities where allowed)
* Jupyter Notebook (for interactive development and documentation)

## How to Run

Each project directory contains a Jupyter Notebook (`.ipynb`) file. To run these notebooks:

1.  Clone this repository: `git clone https://github.com/YourUsername/ml-from-scratch.git`
2.  Navigate to the desired project directory (e.g., `cd ml-from-scratch/eda-fuel-economy/`).
3.  Install necessary libraries `pip install requirements.txt`
4.  Launch Jupyter Notebook: `jupyter notebook`
5.  Open the relevant `.ipynb` file and run all cells.

Each notebook is designed to be self-contained and includes comments and explanations to convey the understanding and work.
