# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 🔹 Example training data (you can expand this or use your own)
data = {
    "Description": [
        "Uber ride to airport", "Dinner at restaurant", "Movie ticket",
        "Flight booking", "Electricity bill", "Online course payment",
        "Groceries from supermarket", "Hotel booking", "Office internet bill",
        "Coffee with friends", "Train ticket", "Laptop charger", "Pizza order",
        "Fuel for car", "Bus fare", "New shoes", "Netflix subscription"
    ],
    "Category": [
        "Travel", "Food", "Entertainment", "Travel", "Utilities", "Education",
        "Food", "Travel", "Utilities", "Food", "Travel", "Shopping",
        "Food", "Travel", "Travel", "Shopping", "Entertainment"
    ]
}

df = pd.DataFrame(data)

# 🔹 Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Description"])
y = df["Category"]

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# 🔹 Evaluate accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 Save model + vectorizer
joblib.dump(model, "model/expense_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model training complete! Saved in 'model/' folder.")
