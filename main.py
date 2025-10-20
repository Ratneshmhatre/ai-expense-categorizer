# main.py
import pandas as pd
import joblib

# 🔹 Load model and vectorizer
model = joblib.load("model/expense_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# 🔹 Read Excel file
input_path = "data/sample_expenses.xlsx"
df = pd.read_excel(input_path)

# 🔹 Predict categories
X_new = vectorizer.transform(df["Description"])
df["Predicted Category"] = model.predict(X_new)

# 🔹 Save results
output_path = "data/categorized_expenses.xlsx"
df.to_excel(output_path, index=False)

print("✅ Categorization complete! File saved as:", output_path)
print(df)
