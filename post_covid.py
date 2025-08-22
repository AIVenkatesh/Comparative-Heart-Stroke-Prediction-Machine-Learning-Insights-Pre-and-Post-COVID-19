!pip install pandas scikit-learn matplotlib seaborn
!pip install pandas scikit-learn
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# Step 1: Load the dataset
  # Update this with your actual file path
data = pd.read_csv(io.BytesIO(uploaded['dataset.csv']))

# Step 2: Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Step 3: Preprocess the data
# Convert categorical variables if necessary (gender, cp, restecg, slope, thal)
data['sex'] = data['sex'].replace({0: 'Female', 1: 'Male'})
# data['sex'] = data['sex'].replace({0: 'Female', 1: 'Male'}).astype('category')
data['cp'] = data['cp'].astype('category')
data['restecg'] = data['restecg'].astype('category')
data['slope'] = data['slope'].astype('category')
data['thal'] = data['thal'].astype('category')
print(data.head())
# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)
# Step 4: Split the data into features and target
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 6: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Step 10: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

data.hist(figsize=(10,10))
plt.suptitle('Health Data Distribution: Histogram Analysis', fontsize=16)
plt.show()

import matplotlib.pyplot as plt
for column in data.columns:
plt.figure(figsize=(10, 6))
data[column].value_counts().plot(kind='bar', color='blue')
plt.title(f'Value Counts for {column}')
plt.xlabel(column)
plt.ylabel('Count')
plt.show()
