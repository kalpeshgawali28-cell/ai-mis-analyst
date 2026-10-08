import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai

st.set_page_config(page_title="AI Visual Business & MIS Analyst", layout="wide")
st.title("📊 AI Visual Business, Cleaning & MIS Analyst")

# ---------------- 1. SIDEBAR CONFIGURATION ----------------
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
st.sidebar.markdown("[Get free API key](https://aistudio.google.com/)")

# ---------------- 2. FILE UPLOADER & INGESTION ----------------
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                raw_df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # ---------------- 3. DATA HYGIENE & CLEANING ----------------
    st.subheader("🧹 1. Data Hygiene & Cleaning")
    
    dup_count = int(raw_df.duplicated().sum())
    null_count = int(raw_df.isnull().sum().sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Rows", len(raw_df))
    c2.metric("Duplicate Rows", dup_count)
    c3.metric("Missing Values (Nulls)", null_count)

    enable_cleaning = st.checkbox("Apply Automated Data Cleaning", value=True)

    if enable_cleaning:
        df = raw_df.copy()

        # Step A: Drop duplicate rows
        df = df.drop_duplicates()

        # Step B: Strip leading/trailing whitespaces from text
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()

        # Step C: Impute missing numerical values with median
        num_cols = df.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        st.success(f"Cleaned dataset active: Removed {dup_count} duplicates and resolved {null_count} null values.")
        
        # Download Cleaned Data
        cleaned_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
    else:
        df = raw_df.copy()
        st.info("Using raw dataset without cleaning transformations.")

    st.markdown("**Dataset Preview (First 5 Rows):**")
    st.dataframe(df.head(5), use_container_width=True)

    # ---------------- 4. VISUAL ANALYTICS (SEABORN) ----------------
    st.subheader("📈 2. Automated Visual Analytics")
    
    clean_num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    clean_cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Correlation Matrix Heatmap**")
        if len(clean_num_cols) >= 2:
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[clean_num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
            plt.close(fig_corr)
        else:
            st.info("Requires at least 2 numerical columns to display correlation.")

    with col_right:
        st.markdown("**Category vs Metric Breakdown**")
        if clean_cat_cols and clean_num_cols:
            selected_cat = st.selectbox("Categorical Dimension", clean_cat_cols)
            selected_num = st.selectbox("Numerical Metric", clean_num_cols)

            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            top_groups = df.groupby(selected_cat)[selected_num].sum().nlargest(8).reset_index()
            sns.barplot(data=top_groups, x=selected_cat, y=selected_num, ax=ax_bar, palette="mako")
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)
            plt.close(fig_bar)
        else:
            st.info("Requires categorical and numerical columns for distribution plots.")

    # ---------------- 5. AI EXECUTIVE INSIGHTS (GEMINI 3.8 FLASH) ----------------
    st.subheader("🤖 3. AI Business Insights & Q&A")
    
    role = st.selectbox("Select Target Stakeholder:", [
        "Executive / Board Member",
        "Operations & MIS Manager",
        "Financial Controller"
    ])
    
    user_query = st.text_input(
        "Ask a business question about the cleaned data:",
        value="Summarize top business drivers, flag anomalies, and give 2 clear action items."
    )

    if st.button("Generate Executive Report"):
        if not api_key:
            st.warning("Please enter your Gemini API Key in the left sidebar.")
        else:
            try:
                client = genai.Client(api_key=api_key)
                
                # Context summary fed to Gemini
                data_summary = f"""
                Columns: {list(df.columns)}
                Shape: {df.shape[0]} rows, {df.shape[1]} columns
                Descriptive Statistics:
                {df.describe(include='all').to_string()}
                Sample Cleaned Rows:
                {df.head(5).to_string()}
                """

                prompt = f"""
                You are a Senior MIS Analyst reporting to a {role}.
                Analyze this cleaned dataset context and answer the business request with concise, high-impact bullet points:
                
                Dataset Context:
                {data_summary}
                
                Question:
                {user_query}
                """

                with st.spinner("Synthesizing cleaned data and generating report..."):
                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )
                    st.markdown("### 📋 Executive Summary")
                    st.markdown(response.text)

            except Exception as e:
                st.error(f"API Error: {e}")