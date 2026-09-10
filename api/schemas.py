from pydantic import BaseModel, Field
from typing import Optional

class CarFeatures(BaseModel):
    brand: str = Field(..., description="The brand of the car, e.g., 'Ford', 'BMW'")
    model: str = Field(..., description="The specific model, e.g., 'Mustang'")
    model_year: int = Field(..., description="The year the car was manufactured, e.g., 2018")
    milage: str = Field(..., description="The mileage of the car, e.g., '51,000 mi.' or '51000'")
    fuel_type: str = Field(..., description="Fuel type, e.g., 'Gasoline', 'Hybrid'")
    engine: str = Field(..., description="Engine specifications")
    transmission: str = Field(..., description="Transmission type, e.g., 'Automatic', '6-Speed A/T'")
    ext_col: str = Field(..., description="Exterior color, e.g., 'Black'")
    int_col: str = Field(..., description="Interior color, e.g., 'Black'")
    accident: str = Field(default="None reported", description="Accident history, e.g., 'At least 1 accident or damage reported' or 'None reported'")
    clean_title: str = Field(default="Yes", description="Whether the car has a clean title ('Yes' or 'No')")

class PredictionRequest(BaseModel):
    features: CarFeatures
    model_type: str = Field(default="random_forest", description="The model to use: 'pytorch_nn', 'random_forest', 'linear_regression'")

class PredictionResponse(BaseModel):
    predicted_price: float
    model_used: str

