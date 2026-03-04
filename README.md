## Accident Prediction Using Decision Tree Classifier
-This project implements a Decision Tree Classification model to predict the likelihood of road accidents based on driver behavior and environmental conditions.

-The objective was to simulate a real-world risk prediction system by generating structured synthetic data and applying supervised machine learning techniques to classify accident occurrence (1 = Accident, 0 = No Accident).

## Problem Statement
Road accidents are influenced by multiple interacting factors such as:

Driver Age

Driving Experience

Vehicle Speed

Weather Conditions

Road Type

Time of Day

The goal was to build a classification model capable of identifying high-risk scenarios and understanding which factors contribute most to accident probability.

## Tools & Technologies
Python

Pandas (data manipulation)

NumPy (data generation & numerical computation)

Matplotlib & Seaborn (visualization)

Scikit-learn (DecisionTreeClassifier, model evaluation)

Jupyter Notebook

## Approach
Synthetic Data Generation
Created a structured dataset simulating realistic driving patterns and accident scenarios.

Exploratory Data Analysis (EDA)

Analyzed feature distributions

Examined correlations

Visualized accident patterns

Data Preprocessing

Encoded categorical variables

Split data into training and testing sets

Model Training

Implemented DecisionTreeClassifier from Scikit-learn

Trained model to classify accident vs non-accident cases

## Model Evaluation

![tuning hyper parameters](https://github.com/user-attachments/assets/5ea2c094-5a58-4477-967c-cc89fa182c16)

Before measuring the accuracy score, the hyperparameters were tuned and the accuracy score was graphed to get the best tree depth that would generalize well. 
However the graph shows that the tree depth isn't an issue because the imbalanced data is overfitting.
Accuracy

Confusion Matrix
![decision tree cm](https://github.com/user-attachments/assets/2355cfea-2a4f-4e88-8229-2fec102cf924)


Precision, Recall, F1-Score
![dt cf](https://github.com/user-attachments/assets/dd6382da-3694-410c-a77e-96e6336c7223)
The accuracy is 54% which is really low compared to the accuracy score of 64%. Recall is also 54% which is okay since it is better to flag an accident that might occur. The f1 score is 0.42 thus the balance between catching accidents
and avoiding false alarms is still weak.

Feature Importance Analysis
![decision tree feature](https://github.com/user-attachments/assets/1c8eee5c-36a3-4338-b5fc-57dc3c746f61)


## Why Decision Tree?
Easy to interpret and visualize

Handles nonlinear relationships

Automatically performs feature selection

Provides feature importance rankings

Useful for rule-based decision systems

This makes it practical for risk assessment systems and safety analytics applications.

## Key Insights
Vehicle speed emerged as a major predictive factor.

Poor weather conditions increased accident likelihood.

Lower driving experience correlated with higher risk.

The decision tree structure clearly showed how different features interact to influence outcomes.

## Learning Outcomes
Through this project, I strengthened my understanding of:

Classification algorithms

Tree-based models

Model interpretability

Overfitting and model depth control

Evaluating ML models beyond accuracy
