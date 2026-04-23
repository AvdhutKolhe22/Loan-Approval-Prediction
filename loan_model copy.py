import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()


df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


df['education'] = df['education'].apply(lambda x: 1 if x == 'Graduate' else 0)
df['self_employed'] = df['self_employed'].apply(lambda x: 1 if x == 'Yes' else 0)
df['loan_status'] = df['loan_status'].astype(str).str.strip().str.lower()

df['loan_status'] = df['loan_status'].map({
    'approved': 1,
    'rejected': 0
})
print("\nLoan Status Distribution:\n", df['loan_status'].value_counts())


df.fillna(df.mean(numeric_only=True), inplace=True)


print("Loan Status Distribution:\n", df['loan_status'].value_counts())
print("\nNaNs:\n", df.isnull().sum())


X = df.drop('loan_status', axis=1)
y = df['loan_status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

accuracies = []


for name, model in models.items():
    print(f"\n===== {name} =====")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)
    print("\nBest Model:", list(models.keys())[accuracies.index(max(accuracies))])
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))



plt.figure()
plt.bar(models.keys(), accuracies)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)


for i, v in enumerate(accuracies):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.show()