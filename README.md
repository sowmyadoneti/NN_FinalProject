# BREAST CANCER CLASSIFICATION USING NEURAL NETWORKS

## Team Member: Sowmya Doneti - 700754085

## Problem Statement:
Breast cancer, a global health challenge, demands early detection for effective treatment. Current classification methods lack precision, especially in recognizing subtle tumor variations. To overcome this, the challenge is to create a robust neural network-based system. Neural networks offer nuanced pattern recognition, vital for distinguishing between benign and malignant tumors.

## Objectives:
- Utilization of neural networks for breast cancer classification.
- Implementation of a predictive system for real-time tumor classification.
- Assessment of model performance using accuracy and loss metrics.
- Exploration of the impact of data standardization on model training.
- Evaluation of the proposed framework in comparison to existing methodologies.

## Proposed Model:

### Data Collection and Processing:
- Load the Breast Cancer dataset into a Pandas DataFrame.
- Handle missing values by dropping columns with all missing values.

### Data Preprocessing:
- Encode the target variable 'diagnosis' to numerical labels using LabelEncoder.
- Scale the features using MinMaxScaler to bring them into a similar range.

### Neural Network Model Setup:
- Design a Multi-Layer Perceptron (MLP) model architecture using TensorFlow and Keras.
- Configure the model with input, hidden, and output layers, including L2 regularization and dropout layers.

### Model Training:
- Split the dataset into training and testing sets using train_test_split.
- Train the model on the training data, specifying epochs, batch size, and validation data.

### Model Evaluation:
- Predict test set results and evaluate model performance using confusion matrix visualization.
- Implement a function to make predictions for new patient details and determine the predicted diagnosis.

## Result/Conclusion:
The neural network model achieved promising results in classifying breast cancer tumors. The model demonstrated high accuracy and low loss on both the training and testing datasets. By leveraging neural networks, we were able to effectively distinguish between benign and malignant tumors, contributing to early detection and improved treatment outcomes for breast cancer patients.


## Repository Contents:
- data.csv: Breast Cancer dataset
- Breast_Cancer_Classification_MLP.ipynb: Jupyter Notebook containing code for data loading, preprocessing, model setup, training, and evaluation.
- README.md: Overview of the project and repository contents.
- Project Results
- Project Report
- Project PPT

## Instructions:
1. Clone the repository to your local machine.
2. Open the Jupyter Notebook file (Breast_Cancer_Classification_MLP.ipynb) using Jupyter Notebook or Google Colab.
3. Follow the step-by-step instructions in the notebook to run the code and analyze the results.
4. Feel free to explore the code and experiment with different parameters or architectures.


