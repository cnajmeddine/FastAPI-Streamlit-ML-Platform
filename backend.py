from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from transformers import pipeline
from typing import List, Dict
import io
import os
from scipy import stats
from fastapi.encoders import jsonable_encoder

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

def get_dtype_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Convert DataFrame dtype counts to a simple dict of strings and integers"""
    dtype_counts = {}
    for dtype, count in df.dtypes.value_counts().items():
        dtype_str = str(dtype)
        dtype_counts[dtype_str] = int(count)
    return dtype_counts

def detect_outliers(data: pd.Series) -> Dict:
    """Detect outliers using IQR method, handling null values"""
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return {
            "count": 0,
            "percentage": 0.0,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_indices": []
        }
    
    Q1 = clean_data.quantile(0.25)
    Q3 = clean_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
    
    return {
        "count": int(len(outliers)),
        "percentage": float(len(outliers) / len(data) * 100),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_indices": outliers.index.tolist()
    }

def analyze_column_values(series: pd.Series) -> Dict:
    """Analyze value distributions including null values"""
    total_count = len(series)
    null_count = series.isna().sum()
    value_counts = series.value_counts(dropna=False)
    
    return {
        "unique_count": int(series.nunique(dropna=False)),
        "null_count": int(null_count),
        "null_percentage": float(null_count / total_count * 100),
        "top_values": {
            str(value) if not pd.isna(value) else "null": {
                "count": int(count),
                "percentage": float(count / total_count * 100)
            }
            for value, count in value_counts.head().items()
        }
    }

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Handle dataset upload and return preview"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        preview = {
            "columns": df.columns.tolist(),
            "preview_rows": df.head(5).replace({np.nan: None}).to_dict('records'),
            "row_count": len(df),
            "null_counts": df.isna().sum().to_dict()
        }
        
        df.to_csv(f"temp_{file.filename}", index=False)
        return jsonable_encoder(preview)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/eda/{filename}")
async def perform_eda(filename: str):
    """Perform comprehensive exploratory data analysis with improved null handling"""
    try:
        df = pd.read_csv(f"temp_{filename}")
        
        # Basic DataFrame Information
        basic_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "total_cells": df.size,
            "memory_usage": float(df.memory_usage(deep=True).sum()),
            "dtypes": get_dtype_counts(df)
        }
        
        # Missing Value Analysis
        # null_analysis = {
        #     col: {
        #         "null_count": int(null_count),
        #         "null_percentage": float(null_count / len(df) * 100),
        #         "non_null_count": int(len(df) - null_count),
        #         "non_null_percentage": float((len(df) - null_count) / len(df) * 100)
        #     }
        #     for col, null_count in df.isna().sum().items()
        # }
        # Calculate total missing values and percentage
        total_missing = int(df.isnull().sum().sum())
        total_missing_percentage = float((total_missing / df.size) * 100)
        
        # Update null_analysis to include totals
        null_analysis = {
            "total_missing": total_missing,
            "total_missing_percentage": total_missing_percentage,
            "columns": {
                col: {
                    "null_count": int(null_count),
                    "null_percentage": float(null_count / len(df) * 100),
                    "non_null_count": int(len(df) - null_count),
                    "non_null_percentage": float((len(df) - null_count) / len(df) * 100)
                }
                for col, null_count in df.isna().sum().items()
            }
        }
        
        # Column-wise Analysis
        column_analysis = {}
        
        # Numeric Columns Analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            clean_data = df[col].dropna()
            stats_dict = {
                "mean": float(clean_data.mean()) if len(clean_data) > 0 else None,
                "median": float(clean_data.median()) if len(clean_data) > 0 else None,
                "std": float(clean_data.std()) if len(clean_data) > 0 else None,
                "min": float(clean_data.min()) if len(clean_data) > 0 else None,
                "max": float(clean_data.max()) if len(clean_data) > 0 else None,
                "skewness": float(stats.skew(clean_data)) if len(clean_data) > 2 else None,
                "kurtosis": float(stats.kurtosis(clean_data)) if len(clean_data) > 2 else None
            }
            
            column_analysis[str(col)] = {
                "type": "numeric",
                "stats": stats_dict,
                "outliers": detect_outliers(df[col]),
                "value_analysis": analyze_column_values(df[col])
            }
        
        # Categorical Columns Analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            column_analysis[str(col)] = {
                "type": "categorical",
                "value_analysis": analyze_column_values(df[col])
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
            clean_data = df[col].dropna()
            column_analysis[str(col)] = {
                "type": "datetime",
                "min": clean_data.min().isoformat() if len(clean_data) > 0 else None,
                "max": clean_data.max().isoformat() if len(clean_data) > 0 else None,
                "range_days": int((clean_data.max() - clean_data.min()).days) if len(clean_data) > 0 else None,
                "value_analysis": analyze_column_values(df[col])
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
            "null_analysis": null_analysis,
            "column_analysis": column_analysis,
            "duplicate_analysis": duplicate_analysis
        }
        
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{model_name}")
async def predict(model_name: str, texts: List[str]):
    """Perform predictions using the selected model"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    try:
        model = models[model_name]
        # Filter out None values and empty strings
        valid_texts = [text for text in texts if text and not pd.isna(text)]
        if not valid_texts:
            return {"predictions": []}
        predictions = model(valid_texts)
        # Ensure predictions are JSON serializable
        return jsonable_encoder({"predictions": predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)