import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# 1 Load Dataset
# -----------------------

df = pd.read_csv("data/churn.csv")

print(df.head())

# -----------------------
# 2 Data Cleaning
# -----------------------

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.dropna(inplace=True)

# -----------------------
# 3 Encode Categorical Data
# -----------------------

le = LabelEncoder()

for column in df.select_dtypes(include="object").columns:
    df[column] = le.fit_transform(df[column])

# -----------------------
# 4 Feature / Target Split
# -----------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------
# 5 Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 6 Train Model
# -----------------------

model = RandomForestClassifier()

model.fit(X_train, y_train)

# -----------------------
# 7 Predictions
# -----------------------

predictions = model.predict(X_test)

# -----------------------
# 8 Evaluation
# -----------------------

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print(classification_report(y_test, predictions))

# -----------------------
# 9 Feature Importance Plot
# -----------------------

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()

# -----------------------
# 10 Save Model
# -----------------------

pickle.dump(model, open("model/churn_model.pkl", "wb"))

print("Model saved successfully")