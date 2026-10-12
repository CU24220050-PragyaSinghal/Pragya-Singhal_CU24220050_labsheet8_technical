import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Load the dataset

df = pd.read_csv('Bank Customer Churn Prediction.csv')
print('Dataset Shape:', df.shape)
print(df.info())
print('\nMissing Values:\n', df.isnull().sum())

columns_to_drop = [col for col in ['customer_id', 'id', 'CustomerId', 'Surname'] if col in df.columns]
df = df.drop(columns=columns_to_drop, errors='ignore')

df = pd.get_dummies(df, drop_first=True)

target_column = 'churn' if 'churn' in df.columns else 'Exited'
if target_column not in df.columns:
    for col in df.columns:
        if 'churn' in col.lower() or 'exited' in col.lower():
            target_column = col
            break

X = df.drop(columns=[target_column])
y = df[target_column]
print(f'Target column identified as: {target_column}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f'\nAccuracy:  {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1-Score:  {f1:.4f}\n')

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['Retained (0)', 'Churned (1)'], filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree for Bank Customer Churn Prediction')
plt.savefig('decision_tree.png', dpi=150)

feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.head(10).plot(kind='barh', color='teal')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.savefig('feature_importance.png', dpi=150)

print('Top Influential Features:')
print(feature_importances.head(10))
