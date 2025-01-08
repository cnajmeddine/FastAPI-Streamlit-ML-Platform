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
}

def convert_to_python_types(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)): 
        return None
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    return obj

def get_dtype_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Convert DataFrame dtype counts to a simple dict of strings and integers"""
    dtype_counts = {}
    for dtype, count in df.dtypes.value_counts().items():
        # Convert dtype to string representation
        dtype_str = str(dtype)
        dtype_counts[dtype_str] = int(count)
    return dtype_counts

def detect_outliers(data: pd.Series) -> Dict:
    """Detect outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return {
        "count": int(len(outliers)),
        "percentage": float(len(outliers) / len(data) * 100),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
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
        return convert_to_python_types(preview)
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
            "memory_usage": float(df.memory_usage(deep=True).sum()),
            "dtypes": get_dtype_counts(df)  # Using the new helper function
        }
        
        # Missing Value Analysis
        total_missing = int(df.isnull().sum().sum())
        missing_analysis = {
            "total_missing": total_missing,
            "total_missing_percentage": float((total_missing / df.size) * 100),
            "missing_by_column": {
                str(col): {
                    "count": int(missing_count),
                    "percentage": float((missing_count / len(df)) * 100)
                }
                for col, missing_count in df.isnull().sum().items()
            }
        }
        
        # Column-wise Analysis
        column_analysis = {}
        
        # Numeric Columns Analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats_dict = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "skewness": float(stats.skew(df[col].dropna())),
                "kurtosis": float(stats.kurtosis(df[col].dropna()))
            }
            
            column_analysis[str(col)] = {
                "type": "numeric",
                "stats": stats_dict,
                "outliers": detect_outliers(df[col])
            }
        
        # Categorical Columns Analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            column_analysis[str(col)] = {
                "type": "categorical",
                "unique_count": int(df[col].nunique()),
                "top_values": {
                    str(value): {
                        "count": int(count),
                        "percentage": float((count / len(df)) * 100)
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
            column_analysis[str(col)] = {
                "type": "datetime",
                "min": df[col].min().isoformat(),
                "max": df[col].max().isoformat(),
                "range_days": int((df[col].max() - df[col].min()).days)
            }
        
        # Duplicate Analysis
        duplicate_rows = int(df.duplicated().sum())
        duplicate_columns = [str(col) for col in df.columns[df.columns.duplicated()]]
        duplicate_analysis = {
            "rows": {
                "count": duplicate_rows,
                "percentage": float((duplicate_rows / len(df)) * 100)
            },
            "columns": {
                "count": len(duplicate_columns),
                "names": duplicate_columns
            }
        }
        
        result = {
            "basic_info": basic_info,
            "missing_analysis": missing_analysis,
            "column_analysis": column_analysis,
            "duplicate_analysis": duplicate_analysis
        }
        
        # Convert all numpy types to Python native types
        return convert_to_python_types(result)
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