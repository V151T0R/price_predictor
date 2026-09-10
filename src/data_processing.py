import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Cleans raw used_cars dataset strings and converts them to appropriate types.
    """
    df_cleaned = df.copy()

    # Clean milage: "51,000 mi." -> 51000
    if 'milage' in df_cleaned.columns:
        df_cleaned['milage'] = df_cleaned['milage'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        
    # Clean price: "$10,300" -> 10300
    if 'price' in df_cleaned.columns:
        df_cleaned['price'] = df_cleaned['price'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        
    # Clean accident
    if 'accident' in df_cleaned.columns:
        df_cleaned['accident'] = df_cleaned['accident'].fillna('None reported')
        df_cleaned['accident'] = df_cleaned['accident'].apply(lambda x: 1.0 if 'accident' in str(x).lower() or 'damage' in str(x).lower() else 0.0)

    # Clean title
    if 'clean_title' in df_cleaned.columns:
        df_cleaned['clean_title'] = df_cleaned['clean_title'].fillna('No')
        df_cleaned['clean_title'] = df_cleaned['clean_title'].apply(lambda x: 1.0 if str(x).lower() == 'yes' else 0.0)

    # Derive age from model_year
    if 'model_year' in df_cleaned.columns:
        current_year = 2024
        df_cleaned['age'] = current_year - df_cleaned['model_year']
        df_cleaned = df_cleaned.drop(columns=['model_year'])

    # Fill any other NA with string 'missing' for categoricals or median for numericals
    for col in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        else:
            df_cleaned[col] = df_cleaned[col].fillna('missing')

    return df_cleaned

def get_preprocessor():
    """
    Returns a ColumnTransformer for preprocessing the features.
    """
    numeric_features = ['milage', 'accident', 'clean_title', 'age']
    categorical_features = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def prepare_data(csv_path, output_dir='models_saved'):
    """
    Loads data, cleans it, fits the preprocessor, and returns X_train, X_test, y_train, y_test.
    """
    df = pd.read_csv(csv_path)
    df_cleaned = clean_data(df)
    
    X = df_cleaned.drop(columns=['price'])
    y = df_cleaned['price']

    # Split first to avoid data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = get_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Save the preprocessor
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))

    return X_train, X_test, y_train, y_test
