# Customers-Churn-Prediction



This project aims to predict customer churn using a machine learning model. The model is trained on a dataset containing various features that describe customer behavior and demographics. The goal is to build a predictive model that can identify customers at risk of churning, enabling proactive retention strategies.


## Project Overview

Customer churn is a critical issue for businesses as it directly impacts revenue. This project builds a binary classification model to predict whether a customer will churn (leave the company) based on various features such as customer demographics, usage data, and service subscriptions.

## Dataset

The dataset is taken from Kaggle and used for this project includes features such as customer age, tenure, service subscriptions, and usage patterns. The target variable is binary, indicating whether the customer churned (1) or not (0).

### Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook 
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy



2. Follow the instructions in the notebook to preprocess the data, build, train, and evaluate the model.

### Example:

- Data Preprocessing
- Model Definition
- Model Training
- Model Evaluation

## Model Training

The model is defined using TensorFlow and Keras. The neural network architecture includes:

- **Input Layer**: Based on the number of features in the dataset.
- **Hidden Layers**: Two layers with ReLU activation.
- **Output Layer**: A single node with sigmoid activation for binary classification.

### Optimizer and Loss Function

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Evaluation

The model's performance is evaluated using accuracy, precision, recall, and F1 score. You can visualize the results through confusion matrices and ROC curves.

## Results

After training the model, you'll find a summary of the results in the notebook, including:

- Accuracy on the test set
- Confusion matrix
- Precision, Recall, F1 Score
- ROC-AUC Score

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or submit an issue.

