import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from data_processing import prepare_data
from models import CarPriceNN, get_sklearn_models

def train_pytorch_model(X_train, y_train, input_dim, output_dir='models_saved', epochs=60, batch_size=32):
    print("Training PyTorch Neural Network...")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CarPriceNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
            
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_nn.pth'))
    print("PyTorch model saved.")
    
def train_sklearn_models(X_train, y_train, output_dir='models_saved'):
    models = get_sklearn_models()
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(output_dir, f'{name}.joblib'))
        print(f"{name} model saved.")

def main():
    data_path = os.path.join('..', 'Data', 'used_cars.csv')
    if not os.path.exists(data_path):
        # Allow running from project root
        data_path = os.path.join('Data', 'used_cars.csv')
        
    print(f"Loading and preprocessing data from {data_path}...")
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    input_dim = X_train.shape[1]
    
    train_pytorch_model(X_train, y_train, input_dim)
    train_sklearn_models(X_train, y_train)
    
    print("All models trained successfully!")

if __name__ == '__main__':
    main()

