import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from google.colab import files

print("Please upload Training.csv and Testing.csv")
uploaded = files.upload()

try:
    df = pd.read_csv("Training.csv")
    tr = pd.read_csv("Testing.csv")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise

l1 = list(df.columns[:-1])
disease = df['prognosis'].unique().tolist()

df.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)
tr.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)

X = df[l1]
y = df[['prognosis']]
X_test = tr[l1]
y_test = tr[['prognosis']]

def train_and_evaluate(model, name):
    model.fit(X, np.ravel(y))
    y_train_pred = model.predict(X)
    y_test_pred = model.predict(X_test)
    print(f"{name} Training Accuracy: {accuracy_score(y, y_train_pred)}")
    print(f"{name} Testing Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"{name} Classification Report (Test Data):\n{classification_report(y_test, y_test_pred, target_names=disease)}")
    return model, y_train_pred, y_test_pred

rf_model, y_train_pred, y_test_pred = train_and_evaluate(RandomForestClassifier(), "Random Forest")

fig, axes = plt.subplots(1, 2, figsize=(30, 11))
sns.heatmap(confusion_matrix(y, y_train_pred), annot=True, fmt='d', cmap='Oranges', xticklabels=disease, yticklabels=disease, ax=axes[0])
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_title('Training Confusion Matrix')
axes[0].set_xticklabels(disease, rotation=45, ha='right')
axes[0].tick_params(axis='y', rotation=0)

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Oranges', xticklabels=disease, yticklabels=disease, ax=axes[1])
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].set_title('Testing Confusion Matrix')
axes[1].set_xticklabels(disease, rotation=45, ha='right')
axes[1].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()

y_prob = rf_model.predict_proba(X_test)
plt.figure(figsize=(10, 6))
for i in range(len(disease)):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{disease[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.show()

def predict_disease(model):
    psymptoms = input("Enter symptoms separated by commas: ").split(',')
    l2 = [1 if symptom.strip() in psymptoms else 0 for symptom in l1]
    predict = model.predict([l2])
    return disease[predict[0]]

predicted_disease = predict_disease(rf_model)
print(f"Predicted Disease: {predicted_disease}")
