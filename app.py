import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load("model/expense_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Expense Categorizer", page_icon="💰", layout="centered")

st.title("💰 AI Expense Categorizer")
st.markdown("Upload your expense Excel file and let AI automatically categorize your expenses.")

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    if "Description" not in df.columns:
        st.error("❌ Excel must contain a column named 'Description'")
    else:
        # Predict categories
        X_new = vectorizer.transform(df["Description"])
        df["Predicted Category"] = model.predict(X_new)

        # Show results
        st.success("✅ Categorization Complete!")
        st.dataframe(df)

        # -----------------------------
        # Pie Chart Visualization
        # -----------------------------
        st.subheader("📊 Expense Distribution by Category")
        category_counts = df["Predicted Category"].value_counts()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(category_counts, labels=category_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # -----------------------------
        # Download Results Section
        # -----------------------------
        st.subheader("📥 Download Results")

        # Download as Excel
        output_excel = BytesIO()
        df.to_excel(output_excel, index=False)
        output_excel.seek(0)

        st.download_button(
            label="⬇️ Download Categorized Data (Excel)",
            data=output_excel,
            file_name="categorized_expenses.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Download as CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Categorized Data (CSV)",
            data=csv_data,
            file_name="categorized_expenses.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload an Excel file to begin.")


st.markdown("---")
st.caption("💼 Built by **Ratnesh Mhatre** | AI Expense Categorizer 2025")
