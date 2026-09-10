import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class CarPriceNN(nn.Module):
    def __init__(self, input_dim):
        super(CarPriceNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def get_sklearn_models():
    """Returns a dictionary of uninitialized scikit-learn models."""
    return {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression()
    }

