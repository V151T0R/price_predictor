import sys
import os
import joblib
import torch
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Add src to sys.path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_processing import clean_data, get_preprocessor
from models import CarPriceNN

app = FastAPI(title="Car Price Predictor API")

# Setup templates and static files
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global variables to hold loaded models
PREPROCESSOR = None
MODELS = {}

@app.on_event("startup")
def load_artifacts():
    global PREPROCESSOR, MODELS
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_saved')
    
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        PREPROCESSOR = joblib.load(preprocessor_path)
    else:
        print("Warning: Preprocessor not found. Please train models first.")
        
    rf_path = os.path.join(models_dir, 'random_forest.joblib')
    if os.path.exists(rf_path):
        MODELS['random_forest'] = joblib.load(rf_path)
        
    lr_path = os.path.join(models_dir, 'linear_regression.joblib')
    if os.path.exists(lr_path):
        MODELS['linear_regression'] = joblib.load(lr_path)
        
    pytorch_path = os.path.join(models_dir, 'pytorch_nn.pth')
    if os.path.exists(pytorch_path) and PREPROCESSOR is not None:
        # Get input dim from preprocessor output shape on a dummy input if needed
        # Or just hardcode based on training. Let's calculate dynamically or load.
        # It's better to instantiate PyTorch model when predicting to know the exact dim
        MODELS['pytorch_nn'] = pytorch_path 
        
def predict_price(features_dict: dict, model_type: str) -> float:
    if PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Models not trained yet.")
        
    df = pd.DataFrame([features_dict])
    
    # Preprocess the data
    df_cleaned = clean_data(df)
    
    # If there's price column added by some mistake, drop it
    if 'price' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['price'])
        
    try:
        X_processed = PREPROCESSOR.transform(df_cleaned)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
        
    if model_type not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model_type} not found or not trained.")
        
    if model_type == 'pytorch_nn':
        input_dim = X_processed.shape[1]
        model = CarPriceNN(input_dim)
        model.load_state_dict(torch.load(MODELS['pytorch_nn'], weights_only=True))
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_processed, dtype=torch.float32)
            prediction = model(X_tensor).item()
            return prediction
    else:
        model = MODELS[model_type]
        prediction = model.predict(X_processed)[0]
        return prediction

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "prediction": None})

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    return templates.TemplateResponse(request=request, name="about.html", context={"request": request})

@app.get("/algorithms", response_class=HTMLResponse)
async def read_algorithms(request: Request):
    return templates.TemplateResponse(request=request, name="algorithms.html", context={"request": request})

@app.post("/", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    brand: str = Form(...),
    model: str = Form(...),
    model_year: int = Form(...),
    milage: str = Form(...),
    fuel_type: str = Form(...),
    engine: str = Form(...),
    transmission: str = Form(...),
    ext_col: str = Form(...),
    int_col: str = Form(...),
    accident: str = Form("None reported"),
    clean_title: str = Form("Yes"),
    model_type: str = Form(...)
):
    features = {
        "brand": brand,
        "model": model,
        "model_year": model_year,
        "milage": milage,
        "fuel_type": fuel_type,
        "engine": engine,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "clean_title": clean_title
    }
    
    try:
        price = predict_price(features, model_type)
        prediction_text = f"${price:,.2f}"
        raw_price = float(price)
    except Exception as e:
        prediction_text = f"Error: {str(e)}"
        raw_price = 0
        
    return templates.TemplateResponse(request=request, name="index.html", context={
        "request": request, 
        "prediction": prediction_text,
        "raw_price": raw_price,
        "features": features,
        "model_type": model_type
    })

# API endpoint for JSON requests
from api.schemas import PredictionRequest, PredictionResponse

@app.post("/predict", response_model=PredictionResponse)
async def predict_api(request: PredictionRequest):
    features_dict = request.features.dict()
    price = predict_price(features_dict, request.model_type)
    return PredictionResponse(predicted_price=price, model_used=request.model_type)
