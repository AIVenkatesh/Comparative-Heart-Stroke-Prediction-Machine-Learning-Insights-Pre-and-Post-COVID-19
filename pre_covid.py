import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import io
df = pd.read_csv(io.BytesIO(uploaded['healthcare-dataset-stroke-data.csv']))
df.head()
df.shape
df.describe().T
df.isnull().sum()
df.interpolate(inplace=True)
df.fillna(method='ffill',inplace=True)
df.fillna(method='bfill',inplace=True)
df.isnull().sum()
sns.countplot(df["smoking_status"])
Df=pd.get_dummies(df,columns=["gender","work_type","smoking_status"])
Df.head()
Df.drop('gender_Female',axis=1,inplace=True)
Df.drop('work_type_children',axis=1,inplace=True)
Df.drop('smoking_status_smokes',axis=1,inplace=True)
Df.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

model=DecisionTreeClassifier()
X=Df[["age","hypertension","heart_disease","avg_glucose_level","bmi","gender_Male","gender_Other","work_type_Govt_job", "work_type_Never_worked","work_type_Private","work_type_Self-employed", "smoking_status_formerly smoked","smoking_status_never smoked"]]
y=Df["stroke"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("accuracy score is ",accuracy_score(y_pred,y_test))
print("precision score is ",precision_score(y_pred,y_test))
print("recall score is ",recall_score(y_pred,y_test))
print("f1 score is ",f1_score(y_pred,y_test))
sns.countplot(df["smoking_status"])
sns.scatterplot(data=df, x='avg_glucose_level', y='bmi', hue='gender')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
