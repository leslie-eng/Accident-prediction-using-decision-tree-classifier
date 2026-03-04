## Accident Prediction Using Decision Tree Classifier
-This project implements a Decision Tree Classification model to predict the likelihood of road accidents based on driver behavior and environmental conditions.

-The objective was to simulate a real-world risk prediction system by generating structured synthetic data and applying supervised machine learning techniques to classify accident occurrence (1 = Accident, 0 = No Accident).

1)Problem Statement
Road accidents are influenced by multiple interacting factors such as:
Driver Age
Driving Experience
Vehicle Speed
Weather Conditions
Road Type
Time of Day
The goal was to build a classification model capable of identifying high-risk scenarios and understanding which factors contribute most to accident probability.

2) Tools & Technologies
Python
Pandas (data manipulation)
NumPy (data generation & numerical computation)
Matplotlib & Seaborn (visualization)
Scikit-learn (DecisionTreeClassifier, model evaluation)
Jupyter Notebook

3) Exploratory Data Analysis
- ![accident count balance](https://github.com/user-attachments/assets/ada745ed-6033-457b-ac1c-343012f7cdd6)
- Distribution of target variable (Accident vs No Accident)
- ![hist of kenyan drivers](https://github.com/user-attachments/assets/c3b446fc-5bed-47ce-a16a-5cfca674e9ff)
- Distribution of Ages of Kenyan drivers
- ![gender vs age](https://github.com/user-attachments/assets/97a4c2a2-a260-4c08-a7d6-e7f29afb0bca)
-
- ![accidents based on demographics](https://github.com/user-attachments/assets/20af9d8f-be64-42d3-8601-f49088bbd3c7)

- ![correlation](https://github.com/user-attachments/assets/bfdf9527-ce27-4925-80fe-d6ca3b1e06fa)

- ![boxplot](https://github.com/user-attachments/assets/892f4770-db2d-48c6-b65e-9539c82091e5)

- ![boxplot years driven](https://github.com/user-attachments/assets/db3700ce-cac7-46ee-9cc2-d8d6491dfd3b)
When drawing the boxplot for accidents based on years driven, the expectation was the lower the experience level the higher the accident count.

4) Data Preprocessing
- Handling categorical variables (Encoding)
- Feature scaling
- Feature removal; leaky features, high and low categorical features
- Train-Test-Split

5) Pivot table
- A pivot table is a bar plot of the sorted values of a particular feature based on the mean. Which road conditions lead to road accidents
 ![pivot table road condition](https://github.com/user-attachments/assets/318ed3de-b065-43bc-9a43-dcbf75f1e884)
Fair to bad roads have a higher chance of causing accidents

- are accidents more prone based on the weather.
![wether pivot table](https://github.com/user-attachments/assets/82f44a68-fb38-43be-be5e-bca3ea50c9a5)
The rainy season bar has a higher count thus indicating it as a major cause of accidents.
- Are accidents more prone based on the sign visibility?
![sign visibility](https://github.com/user-attachments/assets/0832f82f-1b2b-4e6e-99fa-fc64294d6763)

-This is a mapbox of the accident frequency based on the counties
![county mapbox](https://github.com/user-attachments/assets/fce7ed7a-1efa-4e7e-80b8-57d30042b41c)


6) Iterate
- Model builiding through a pipeline. SimpleImputer was used for Nan values and the method was mean. Ordinal encoder was used for categorical data

7) Model Evaluation
- Accuracy Score: the training accuracy was 70% and the test accuracy was 70% which is fair.
## Model Evaluation

![tuning hyper parameters](https://github.com/user-attachments/assets/5ea2c094-5a58-4477-967c-cc89fa182c16)

Before measuring the accuracy score, the hyperparameters were tuned and the accuracy score was graphed to get the best tree depth that would generalize well. 
However the graph shows that the tree depth isn't an issue because the imbalanced data is overfitting.
### Accuracy

### Confusion Matrix

![decision tree cm](https://github.com/user-attachments/assets/2355cfea-2a4f-4e88-8229-2fec102cf924)


### Precision, Recall, F1-Score

![dt cf](https://github.com/user-attachments/assets/dd6382da-3694-410c-a77e-96e6336c7223)

The accuracy is 54% which is really low compared to the accuracy score of 64%. Recall is also 54% which is okay since it is better to flag an accident that might occur. The f1 score is 0.42 thus the balance between catching accidents
and avoiding false alarms is still weak.

### Feature Importance Analysis

![decision tree feature](https://github.com/user-attachments/assets/1c8eee5c-36a3-4338-b5fc-57dc3c746f61)


8) Why Decision Tree?
Easy to interpret and visualize
Handles nonlinear relationships
Automatically performs feature selection
Provides feature importance rankings
Useful for rule-based decision systems
This makes it practical for risk assessment systems and safety analytics applications.

9) Key Insights
Vehicle speed emerged as a major predictive factor.
Poor weather conditions increased accident likelihood.
Lower driving experience correlated with higher risk.
The decision tree structure clearly showed how different features interact to influence outcomes.

10) Learning Outcomes
Through this project, I strengthened my understanding of:
Classification algorithms
Tree-based models
Model interpretability
Overfitting and model depth control
Evaluating ML models beyond accuracy
