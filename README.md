# E-commerce Product Text Classification with LSTM
This project is designed to classify e-commerce product descriptions into one of four categories: 

Electronics, Household, Books, and Clothing & Accessories. The goal is to automate product categorization to save time and resources, often spent in manual categorization.

## Dataset

The dataset used for this project can be obtained from Kaggle. It contains product descriptions and their corresponding categories.

## Project Objectives

### Data Loading and Processing: 

Load the dataset using Pandas for easy manipulation and preprocessing.

### Model Development: 

Create an LSTM-based neural network using TensorFlow that can accurately classify product descriptions.

### Target Metrics:

Accuracy > 85%

F1 Score > 0.7

### TensorBoard Visualization: 

Visualize training progress, accuracy, and loss metrics using TensorBoard.

### Model Saving: 
Save the trained model in .h5 format and the tokenizer in .json format for future use.

### Project Structure:

Ecommerce_LSTM.ipynb

The main Jupyter Notebook containing the entire workflow from data preprocessing to model training and evaluation.

Ecommerce_classifier_model.h5: The trained LSTM model.

tokenizer.json: The tokenizer used to preprocess the text data.

### Requirements

Python 3.x

TensorFlow

Pandas

Numpy

Matplotlib (for visualization if needed)

TensorBoard (for model training visualization)

TensorBoard Visualization:

After training starts, you can visualize the model’s performance by launching TensorBoard.

## Project Walkthrough

Data Preprocessing:

The dataset is loaded and cleaned using Pandas.

The text data is tokenized and converted into sequences suitable for LSTM input.

The labels are encoded for categorical classification.

Model Architecture:

An LSTM model is developed using TensorFlow's Sequential API.

The model is designed to classify the product descriptions into four categories based on the defined accuracy and F1 score criteria.

# Evaluation:

The model is evaluated to ensure it meets the accuracy > 85% and F1 score > 0.7 requirements.
Performance metrics are displayed on TensorBoard.

# Saving Artifacts:

The trained model is saved as Ecommerce_classifier_model.h5.
The tokenizer is saved as tokenizer.json.

# Results
The trained model achieved an accuracy of over 85% and an F1 score greater than 0.7 on the test dataset, satisfying the project's criteria.

# References
TensorFlow Documentation
Kaggle Dataset: E-commerce Text Classification
https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

# Screenshot 

## Model Architecture

<img width="461" alt="Model_Architecture" src="https://github.com/user-attachments/assets/97f32948-b907-46aa-978e-0b93aeb2f066">

## Model Performance 

<img width="497" alt="Model_Performance" src="https://github.com/user-attachments/assets/9afa68ed-514a-443b-b8cd-36446c6a5187">










