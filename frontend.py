import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from typing import List
import json

# Constants
API_URL = "http://localhost:8000"
AVAILABLE_MODELS = ["sentiment-analysis"]  # Add more models as needed

def main():
    st.title("ML Platform MVP")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Selection")
        selected_model = st.selectbox(
            "Choose a pre-trained model",
            options=AVAILABLE_MODELS
        )
    
    # Main content area
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file:
        # Upload dataset to backend
        files = {"file": uploaded_file}
        response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            data_preview = response.json()
            
            # Display data preview
            st.subheader("Data Preview")
            preview_df = pd.DataFrame(data_preview["preview_rows"])
            st.dataframe(preview_df)
            
            # Tabs for EDA and Predictions
            tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Predictions"])
            
            with tab1:
                st.subheader("Exploratory Data Analysis")
                
                # Fetch EDA results
                eda_response = requests.get(f"{API_URL}/eda/{uploaded_file.name}")
                if eda_response.status_code == 200:
                    eda_results = eda_response.json()
                    
                    # Display summary statistics
                    st.write("### Summary Statistics")
                    summary_df = pd.DataFrame(eda_results["summary"])
                    st.dataframe(summary_df)
                    
                    # Display missing values
                    st.write("### Missing Values")
                    missing_df = pd.DataFrame.from_dict(
                        eda_results["missing_values"], 
                        orient='index', 
                        columns=['Count']
                    )
                    st.dataframe(missing_df)
                    
                    # Create visualizations
                    numeric_cols = [
                        col for col, dtype in eda_results["column_types"].items()
                        if "float" in dtype or "int" in dtype
                    ]
                    
                    if numeric_cols:
                        selected_col = st.selectbox(
                            "Select column for distribution plot",
                            options=numeric_cols
                        )
                        
                        # Create distribution plot using plotly
                        df = pd.read_csv(uploaded_file)
                        fig = px.histogram(df, x=selected_col)
                        st.plotly_chart(fig)
            
            with tab2:
                st.subheader("Model Predictions")
                
                # For text-based models
                if selected_model == "sentiment-analysis":
                    text_input = st.text_area(
                        "Enter text for prediction (one per line)"
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