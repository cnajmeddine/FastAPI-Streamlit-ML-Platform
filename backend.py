from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from transformers import pipeline
from typing import List, Dict
import json
import io
import os

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

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Handle dataset upload and return preview"""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Generate preview data
        preview = {
            "columns": df.columns.tolist(),
            "preview_rows": df.head(5).to_dict('records'),
            "row_count": len(df),
        }
        
        # Save dataset temporarily (in production, use proper storage)
        df.to_csv(f"temp_{file.filename}", index=False)
        
        return preview
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/eda/{filename}")
async def perform_eda(filename: str):
    """Perform exploratory data analysis on the uploaded dataset"""
    try:
        df = pd.read_csv(f"temp_{filename}")
        
        # Calculate basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {
            "summary": df[numeric_cols].describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns}
        }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{model_name}")
async def predict(model_name: str, texts: List[str]):
    """Perform predictions using the selected model"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    try:
        # Get the selected model
        model = models[model_name]
        
        # Perform prediction
        predictions = model(texts)
        
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)