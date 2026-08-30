import streamlit as st
import pandas as pd
import google.generativeai as genai

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="AI Data Quality Agent")
st.title("AI Data Quality Agent")
st.write("Upload a CSV file to analyze its quality and generate AI insights.")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    if st.button("Analyze Dataset"):

        st.subheader("Dataset Summary")

        st.write("Rows :", df.shape[0])
        st.write("Columns :", df.shape[1])

        st.write("Column Names")
        st.write(df.columns.tolist())

        st.write("Data Types")
        st.write(df.dtypes)

        st.subheader("Missing Values")

        missing_values = df.isnull().sum()
        st.write(missing_values)
        st.write("Total Missing Values :", missing_values.sum())

        st.subheader("Duplicate Records")
        duplicates = df.duplicated().sum()
        st.write(duplicates)

        st.subheader("Outlier Detection")
        numeric_columns = df.select_dtypes(include="number").columns
        total_outliers = 0

        for col in numeric_columns:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower) | (df[col] > upper)]

            total_outliers += len(outliers)

            st.write(f"{col}: {len(outliers)} outliers")

        score = 100

        score -= min(missing_values.sum() // 50, 20)
        score -= duplicates * 5
        score -= min(total_outliers // 20, 20)
        score = max(score, 0)

        st.subheader("Data Quality Score")
        st.metric("Quality Score", f"{score}/100")

        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 60:
            status = "Needs Improvement"
        else:
            status = "Poor"

        st.write("Overall Status:", status)

        st.subheader("AI Business Insights")

        prompt = f"""
You are a Senior Data Analyst.

Analyze the following dataset summary.

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Missing Values:
{missing_values.to_string()}

Duplicate Records:
{duplicates}

Total Outliers:
{total_outliers}

Data Quality Score:
{score}/100

Generate:

1. Top 5 Business Insights

2. Data Cleaning Recommendations

3. Suggested Dashboard KPIs

Use headings and bullet points.
"""

        with st.spinner("Generating AI Insights..."):

            response = model.generate_content(prompt)

        st.markdown(response.text)

st.markdown("--------------------")
st.caption("Developed for AI/ML Hackathon using Streamlit + Gemini AI")