# From Words to Feelings, Emotion Prediction using Classical and Deep Learning Models

This project explores emotion detection in short text using machine learning and deep learning models. We compare a classical Linear SVM, a custom PyTorch CNN, and a fine tuned BERT model on the GoEmotions dataset, with all emotion labels mapped to Ekman’s emotion categories.

The repository is intended to be readable, reproducible, and educational, showing how different modeling choices affect performance, efficiency, and generalization.

## Overview

Emotion detection aims to identify the underlying emotion expressed in a piece of text. In this project, we focus on short form English text and study how well different models handle this task.

We evaluate three approaches

1. Linear SVM with TF IDF features  
2. CNN with pretrained GloVe embeddings  
3. BERT based transformer model  

Performance is measured using macro F1, precision, recall, and accuracy, with an emphasis on handling class imbalance.

## Dataset

We use the GoEmotions dataset developed by Google Research. It contains tens of thousands of Reddit comments annotated by multiple human raters.

For this project, the original fine grained emotion labels are mapped into Ekman’s emotion groups

joy, anger, sadness, fear, disgust, surprise, and neutral  

The dataset is loaded using the HuggingFace datasets library.

## Models

### Linear SVM

A strong classical baseline for text classification. This model uses both word level and character level TF IDF features and a one versus rest classification strategy. It performs well on short text with clear lexical signals.

### CNN

A one dimensional convolutional neural network implemented in PyTorch. It uses pretrained GloVe embeddings, multiple convolutional kernel sizes, global max pooling, and dropout for regularization.

### BERT

A transformer based model using a pretrained bert base architecture. The model is fine tuned on the emotion classification task and leverages contextual embeddings to capture semantic meaning.

## Evaluation

Models are evaluated using

1. Macro F1 score  
2. Precision and recall  
3. Accuracy  

Per class metrics and confusion matrices are included to analyze model behavior and common errors.

## Results summary

Linear SVM performs competitively despite its simplicity  
CNN underperforms due to limited data and class imbalance  
BERT achieves the best overall performance, especially for context dependent emotions  
Emotion frequency strongly influences prediction quality  

## Repository contents

1. Jupyter notebook containing preprocessing, training, and evaluation  
2. Final project report with methodology and analysis  
3. README describing the project  

Additional folders for saved models and figures can be added later.

## Requirements

Python 3.9 or higher

Main libraries used

numpy  
pandas  
scikitlearn  
torch  
transformers  
datasets  
matplotlib  

Exact versions can be added in a requirements file if needed.

## Running the project

1. Clone the repository  
2. Install the required dependencies  
3. Open the notebook and run all cells in order  

The notebook includes data loading, preprocessing, training, and evaluation for all models.

## Notes

Training the BERT model requires significantly more computation than the other models. If running on limited hardware, expect longer runtimes or consider skipping BERT training.

## Authors

Sohaib Bantan  
Azzam Alharthi  
