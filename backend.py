from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from transformers import pipeline
from typing import List, Dict
import json
import io
import os
from scipy import stats

app = FastAPI(title="ML Platform API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pre-trained models
models = {
    "sentiment-analysis": pipeline("sentiment-analysis"),
    # Add more models here as needed
}

def detect_outliers(data: pd.Series) -> Dict:
    """Detect outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return {
        "count": len(outliers),
        "percentage": (len(outliers) / len(data) * 100),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_indices": outliers.index.tolist()
    }

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Handle dataset upload and return preview"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        preview = {
            "columns": df.columns.tolist(),
            "preview_rows": df.head(5).to_dict('records'),
            "row_count": len(df),
        }
        
        df.to_csv(f"temp_{file.filename}", index=False)
        return preview
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/eda/{filename}")
async def perform_eda(filename: str):
    """Perform comprehensive exploratory data analysis"""
    try:
        df = pd.read_csv(f"temp_{filename}")
        
        # Basic DataFrame Information
        basic_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "total_cells": df.size,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "dtypes": df.dtypes.value_counts().to_dict()
        }
        
        # Missing Value Analysis
        total_missing = df.isnull().sum().sum()
        missing_analysis = {
            "total_missing": total_missing,
            "total_missing_percentage": (total_missing / df.size) * 100,
            "missing_by_column": {
                col: {
                    "count": missing_count,
                    "percentage": (missing_count / len(df)) * 100
                }
                for col, missing_count in df.isnull().sum().items()
            }
        }
        
        # Column-wise Analysis
        column_analysis = {}
        
        # Numeric Columns Analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            column_analysis[col] = {
                "type": "numeric",
                "stats": {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "skewness": stats.skew(df[col].dropna()),
                    "kurtosis": stats.kurtosis(df[col].dropna())
                },
                "outliers": detect_outliers(df[col])
            }
        
        # Categorical Columns Analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            column_analysis[col] = {
                "type": "categorical",
                "unique_count": df[col].nunique(),
                "top_values": {
                    str(value): {
                        "count": count,
                        "percentage": (count / len(df)) * 100
                    }
                    for value, count in value_counts.head().items()
                }
            }
        
        # Datetime Columns Analysis
        datetime_cols = []
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                continue
        
        for col in datetime_cols:
            column_analysis[col] = {
                "type": "datetime",
                "min": df[col].min().isoformat(),
                "max": df[col].max().isoformat(),
                "range_days": (df[col].max() - df[col].min()).days
            }
        
        # Duplicate Analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        duplicate_analysis = {
            "rows": {
                "count": duplicate_rows,
                "percentage": (duplicate_rows / len(df)) * 100
            },
            "columns": {
                "count": len(duplicate_columns),
                "names": duplicate_columns
            }
        }
        
        return {
            "basic_info": basic_info,
            "missing_analysis": missing_analysis,
            "column_analysis": column_analysis,
            "duplicate_analysis": duplicate_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{model_name}")
async def predict(model_name: str, texts: List[str]):
    """Perform predictions using the selected model"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    try:
        model = models[model_name]
        predictions = model(texts)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)