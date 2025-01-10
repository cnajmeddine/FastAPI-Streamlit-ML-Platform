import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.figure_factory as ff
from typing import List
import json

# Constants
API_URL = "http://localhost:8000"
AVAILABLE_MODELS = ["sentiment-analysis"]

def format_bytes(size):
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def main():
    st.title("ML Platform MVP")
    
    with st.sidebar:
        st.header("Model Selection")
        selected_model = st.selectbox(
            "Choose a pre-trained model",
            options=AVAILABLE_MODELS
        )
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file:
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            data_preview = response.json()
            
            # Display data preview
            st.header("Data Preview")
            preview_df = pd.DataFrame(data_preview["preview_rows"])
            st.dataframe(preview_df)
            
            # Tabs for EDA and Predictions
            tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Sentiment Analysis"])
            
            with tab1:
                st.header("Exploratory Data Analysis")
                
                # Fetch EDA results
                eda_response = requests.get(f"{API_URL}/eda/{uploaded_file.name}")
                if eda_response.status_code == 200:
                    eda_results = eda_response.json()
                    
                    # Basic Information
                    st.subheader("Basic DataFrame Information")
                    basic_info = eda_results["basic_info"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", basic_info["rows"])
                    with col2:
                        st.metric("Columns", basic_info["columns"])
                    with col3:
                        st.metric("Total Cells", basic_info["total_cells"])
                    
                    st.metric("Memory Usage", format_bytes(basic_info["memory_usage"]))
                    
                    # Data Types
                    st.write("#### Data Types Distribution")
                    for dtype, count in basic_info["dtypes"].items():
                        st.write(f"- {dtype}: {count} columns")
                    
                    # Missing Values Analysis
                    st.subheader("Missing Values Analysis")
                    missing = eda_results["null_analysis"]
                    st.metric(
                        "Total Missing Values", 
                        missing["total_missing"],
                        f"{missing['total_missing_percentage']:.2f}%"
                    )
                    
                    # Missing values by column
                    missing_df = pd.DataFrame([
                        {
                            "Column": col,
                            "Missing Count": info["null_count"],
                            "Missing Percentage": f"{info['null_percentage']:.2f}%"
                        }
                        for col, info in missing["columns"].items()
                        if info["null_count"] > 0
                    ])
                    if not missing_df.empty:
                        st.write("#### Missing Values by Column")
                        st.dataframe(missing_df)
                    
                    # Column Analysis
                    st.subheader("Column Analysis")
                    
                    # Group columns by type
                    column_analysis = eda_results["column_analysis"]
                    numeric_cols = [col for col, info in column_analysis.items() 
                                  if info["type"] == "numeric"]
                    categorical_cols = [col for col, info in column_analysis.items() 
                                     if info["type"] == "categorical"]
                    datetime_cols = [col for col, info in column_analysis.items() 
                                   if info["type"] == "datetime"]
                    
                    # Numeric Columns
                    if numeric_cols:
                        st.write("#### Numeric Columns")
                        selected_num_col = st.selectbox(
                            "Select numeric column",
                            options=numeric_cols
                        )
                        col_info = column_analysis[selected_num_col]
                        
                        # Display statistics
                        st.write("Statistics:")
                        stats_df = pd.DataFrame([col_info["stats"]])
                        st.dataframe(stats_df)
                        
                        # Display outliers
                        st.write("Outliers:")
                        outliers = col_info["outliers"]
                        st.write(f"- Count: {outliers['count']}")
                        st.write(f"- Percentage: {outliers['percentage']:.2f}%")
                        st.write(f"- Range: {outliers['lower_bound']:.2f} to {outliers['upper_bound']:.2f}")
                        
                        # Display unique values
                        if 'value_analysis' in col_info:
                            st.write(f"Unique Values: {col_info['value_analysis']['unique_count']}")
                    
                    # Categorical Columns
                    if categorical_cols:
                        st.write("#### Categorical Columns")
                        selected_cat_col = st.selectbox(
                            "Select categorical column",
                            options=categorical_cols
                        )
                        col_info = column_analysis[selected_cat_col]
                        
                        if 'value_analysis' in col_info:
                            st.write(f"Unique Values: {col_info['value_analysis']['unique_count']}")
                            st.write("Top 5 Values:")
                            for value, info in col_info["value_analysis"]["top_values"].items():
                                st.write(f"- {value}: {info['count']} ({info['percentage']:.2f}%)")
                    
                    # Datetime Columns
                    if datetime_cols:
                        st.write("#### Datetime Columns")
                        selected_dt_col = st.selectbox(
                            "Select datetime column",
                            options=datetime_cols
                        )
                        col_info = column_analysis[selected_dt_col]
                        
                        st.write(f"Range: {col_info['min']} to {col_info['max']}")
                        st.write(f"Total Days: {col_info['range_days']}")
                    
                    # Duplicate Analysis
                    st.subheader("Duplicate Analysis")
                    dupes = eda_results["duplicate_analysis"]
                    st.write("#### Duplicate Rows")
                    st.metric(
                        "Count",
                        dupes["rows"]["count"],
                        f"{dupes['rows']['percentage']:.2f}%"
                    )
                    
                    if dupes["columns"]["count"] > 0:
                        st.write("#### Duplicate Columns")
                        st.write(f"Count: {dupes['columns']['count']}")
                        st.write("Columns:", ", ".join(dupes["columns"]["names"]))
            
            with tab2:
                st.subheader("Model Predictions")
                
                if selected_model == "sentiment-analysis":
                    text_input = st.text_area(
                        "Enter text for prediction"
                    )
                    
                    if st.button("Run Prediction"):
                        if text_input:
                            texts = text_input.split('\n')
                            response = requests.post(
                                f"{API_URL}/predict/{selected_model}",
                                json=texts
                            )
                            
                            if response.status_code == 200:
                                predictions = response.json()["predictions"]
                                st.write("### Predictions")
                                for text, pred in zip(texts, predictions):
                                    st.write(f"Text: {text}")
                                    st.write(f"Sentiment: {pred['label']}")
                                    st.write(f"Confidence: {pred['score']:.2%}")
                                    st.write("---")

if __name__ == "__main__":
    main()