import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import warnings

# Suppressing warnings
warnings.filterwarnings('ignore')

# Data Loading
data = pd.read_csv("C:/Users/Admin/Desktop/P2/nba2021.csv")

# Displaying features and position counts
print('Features:')
print(data.columns.tolist())
position_counts = data['Pos'].value_counts()
print('\nPosition counts:')
for position, count in position_counts.items():
    print(f'{position}: {count}')
print('\nShape of data: {}'.format(data.shape))
print()

# Label encoding for categorical variables
enc = LabelEncoder()
for column in data.select_dtypes(include=[np.object]):
    data[column] = enc.fit_transform(data[column])

# Correlation matrix
correlation_matrix = data.corr()
target_correlations = correlation_matrix['Pos'].abs().sort_values(ascending=False)

# Selecting features with correlation above 0.1
selected_features = ['3P', '3PA', 'ORB', 'DRB', 'TRB', 'AST', 'BLK']
X = data[selected_features]
y = data["Pos"]
print(X)
print()
print(y)
print()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Linear SVM model
lin_svm = LinearSVC().fit(X_train, y_train)
y_train_pred_sv = lin_svm.predict(X_train)
y_test_pred_sv = lin_svm.predict(X_test)

# Model evaluation
train_accuracy = accuracy_score(y_train, y_train_pred_sv)
test_accuracy = accuracy_score(y_test, y_test_pred_sv)

print("Training set accuracy:", train_accuracy)
print("Testing set accuracy:", test_accuracy)
print()

# Confusion matrix
print("Confusion matrix:")
conf_matrix = pd.crosstab(y_test, y_test_pred_sv, rownames=['True'], colnames=['Predicted'], margins=True)
print(conf_matrix)
print()

# Stratified 10-fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(lin_svm, X, y, cv=stratified_kfold)

# Print accuracy of each fold
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i} accuracy: {score}")

# Print average accuracy across all folds
print("Average cross-validation score: {:.2f}".format(cv_scores.mean()))
