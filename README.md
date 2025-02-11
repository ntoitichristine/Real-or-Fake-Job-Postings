# Real or Fake Job Postings Classification
## Overview
This project aims to develop a classification model to distinguish between real and fake job postings using a dataset containing 18,000 job descriptions, including approximately 800 fraudulent ones. The dataset includes both textual data (job descriptions) and meta-information about the jobs. The goal is to leverage Natural Language Processing (NLP), Machine Learning (ML), and predictive modeling techniques to identify fraudulent job postings and uncover key traits or features that differentiate them from legitimate ones.

## Business Understanding
### Overview
Fraudulent job postings pose a significant risk to job seekers, companies, and online job platforms. These fake postings can lead to financial losses, identity theft, and reputational damage. By building a robust classification model, businesses can automate the detection of fraudulent job postings, enhance platform credibility, and protect users from potential scams.

### Objectives
Predictive Modeling: Develop a classification model to predict whether a job posting is real or fake using textual and meta-features.

Feature Identification: Identify key traits, phrases, or patterns in job descriptions that are indicative of fraudulent postings.

Exploratory Data Analysis (EDA): Perform EDA to uncover insights, trends, and correlations within the dataset.

Model Interpretability: Ensure the model's predictions are interpretable, enabling stakeholders to understand the reasoning behind flagged postings.


### Key Components
#### Predictive Modeling:

Use supervised learning techniques ( Random Forest, XGBoost, and Deep Learning models) to classify job postings as real or fake.

Incorporate both textual features (TF-IDF, word embeddings) and meta-features (e.g., job title, location, salary) into the model.

#### NLP Pipelines:

Preprocess textual data using tokenization, stemming, and stopword removal.

Extract features using techniques like TF-IDF, Bag of Words.

Perform entity recognition to identify key terms (e.g., company names, locations) that may indicate fraud.

#### Machine Learning Pipelines:

Use tools such as Scikit-learn and TensorFlow for model development.

#### Model Evaluation and Optimization:

Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

Address class imbalance (800 fake vs. 17K real postings) using techniques like class weighting.

#### Success Metrics:

Primary metric: F1-score (to balance precision and recall, given the imbalanced dataset).

Secondary metrics: Accuracy, Precision, Recall, and AUC-ROC.

Business metric: Reduction in fraudulent job postings on the platform.


### Data Preparation
#### Import Libraries

from zipfile import ZipFile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from wordcloud import WordCloud

#### Load the Dataset

##### Extract the zip file
with ZipFile("Real or Fake Job Postings.zip", 'r') as f:
    f.extractall()

##### Read the CSV file as a single column
fakejobs_postings_df = pd.read_csv("fake_job_postings.csv")
fakejobs_postings_df.head(2)
Initial Dataset Exploration

##### Check the shape of the dataset
print(f'The shape of the dataset is {fakejobs_postings_df.shape}')

print(fakejobs_postings_df.columns)

print(f'The Target Variable is Fraudulent')

#### Handle Missing Values

##### Determine the number of rows with missing data
(fakejobs_postings_df.isnull().sum()[fakejobs_postings_df.isnull().sum() > 0] / len(fakejobs_postings_df) * 100).round(2)

<img width="549" alt="percentagemissingvalues" src="https://github.com/user-attachments/assets/92a2e271-b504-4a69-959d-c5bff89e4e30" />

### Data Cleaning and Preprocessing

##### Drop columns that are not necessary in the analysis
columns = ['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']
fakejobs_postings_df.drop(columns=columns, inplace=True)

##### Fill missing values in categorical columns with ' ' and numerical columns with mode
for column in fakejobs_postings_df.columns:
    if fakejobs_postings_df[column].dtype == 'object':  # Categorical column
        fakejobs_postings_df[column] = fakejobs_postings_df[column].fillna(' ')  # Fill ' '
    else:  # Numerical column
        fakejobs_postings_df[column] = fakejobs_postings_df[column].fillna(fakejobs_postings_df[column].mode()[0])  # Fill with mode

##### Ensure there are no rows with missing target variable
fakejobs_postings_df.dropna(subset=['fraudulent'], inplace=True)

##### Check for any remaining missing values and handle
fakejobs_postings_df.isnull().sum()

fakejobs_postings_df.head(2)

#### Categorical and Numerical Columns

##### Identify categorical columns (dtype = object or category)
categorical_columns = fakejobs_postings_df.select_dtypes(include=['object', 'category']).columns

print("Categorical columns:", categorical_columns)

##### Identify categorical columns (dtype = object or category)
numerical_columns = fakejobs_postings_df.select_dtypes(include=['float64', 'int64']).columns

print("numerical columns:", numerical_columns)
Exploratory Data Analysis (EDA)
Visualizing the Distribution of the Target Variable ('fraudulent')

##### Calculate the percentage of missing values for each column
missing_data_percentage = fakejobs_postings_df.isnull().mean() * 100

##### Create a barplot to visualize the percentage of missing values
plt.figure(figsize=(8,4))
sns.barplot(x=missing_data_percentage.index, y=missing_data_percentage.values, color='purple')
plt.xticks(rotation=90)
plt.xlabel('Columns')
plt.ylabel('Percentage of Missing Values')
plt.title('Percentage of Missing Values per Column')
##### Set y-axis limit from 0 to 100
plt.ylim(0, 100)
plt.show()

<img width="549" alt="percentagemissingvalues" src="https://github.com/user-attachments/assets/efb0434f-e852-4bb1-aa97-f078cd7a8cbf" />

### Exploratory Data Analysis (EDA)
#### Target Variable Distribution
•	Visualization: Countplot shows imbalance; majority are legitimate postings.

#### Industry and Location Insights
<img width="544" alt="postingsbyindustry" src="https://github.com/user-attachments/assets/48bbc17f-79a5-4eb1-b3d7-c5d97aefa117" />

<img width="479" alt="countrieswithmostpostings" src="https://github.com/user-attachments/assets/63f40e17-6d34-47e8-8e6b-e624995cd232" />


•	Industry Analysis: Highlights the top 10 sectors most frequently listed.
•	Country Analysis: The USA and Britain lead in the number of postings.

### Model Development
#### Text Processing
•	TF-IDF Transformation: Text features vectorized into numerical data for analysis.

### Machine Learning Techniques
#### 1.	K-Nearest Neighbors (KNN)
o	Performance: Accuracy of 98%, yet struggles with minority class.
<img width="445" alt="KNNconfusionmaatrix" src="https://github.com/user-attachments/assets/c1bf6df7-e612-48c7-85ca-4182afd7e0a5" />


#### 2.	Random Forest with GridSearchCV
o	Performance: Accuracy of 98%, precision for fake class is high but low recall.
<img width="401" alt="randomforestconfusionmatrix" src="https://github.com/user-attachments/assets/173fb583-ac4f-437a-b015-a299f6ce77c1" />


#### 3.	XGBoost
o	Performance: Best overall with accuracy of 99%, high precision and recall.
<img width="410" alt="XGBoostconfusionmatrix" src="https://github.com/user-attachments/assets/c644ba01-495b-419e-85ad-6f97e342000b" />


#### 4.	Gradient Boosting	
o	Performance: Accuracy of 98%, similar to Random Forest but slightly better recall.
<img width="415" alt="Gradient boosting confusion matrix" src="https://github.com/user-attachments/assets/183ae20d-bf4f-4053-a16b-446b3070f134" />


##### Key Insights
•	Best Performer: XGBoost provides optimal trade-off in precision and recall.
•	Model Decision: Favoring precision-heavy models like Random Forest if minimizing false positives is vital.

### Unsupervised Learning with PCA
#### Dimensionality Reduction
•	PCA Projection: Data reduced to 2 dimensions helps visualize and simplify the model.
<img width="583" alt="pcamodel" src="https://github.com/user-attachments/assets/3866f496-6892-42b7-87ae-ee594b56bb9c" />



### Final Report
#### Summary
•	Data Analysis: Robust feature evaluation, cleaning, and transformation led to a high-performing model.
•	Business Value: Enhances platform credibility, protects job seekers from fraud, and provides actionable insight.

#### Conclusion
This project involves a comprehensive approach to classify job postings as real or fake using machine learning techniques. The steps include data preparation, exploratory data analysis, and model development. The final model aims to provide actionable insights and help in reducing fraudulent job postings on the platform.
