"""
FloodGuard AI - Machine Learning Model Training & Pipeline Exporter
Academic Project: 3rd-Semester AI / CSE
Trains a Scikit-Learn Random Forest Pipeline to predict Flood Risk Category (Low, Medium, High)
and exact probability distribution based on District Flood Susceptibility Index (DFSI) + Real-time Meteorological triggers.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "DFSI.csv"
MODEL_PATH = "model.pkl"
META_PATH = "model_metadata.json"

def generate_synthetic_weather_scenarios(df, random_seed=42):
    """
    Synthesizes realistic meteorological conditions paired with district DFSI baseline:
    - Dry/Normal days: Low rain, stable rivers -> Predominantly Low Risk
    - Moderate Monsoon: 20-70mm rain -> Low/Medium Risk based on DFSI
    - Severe Storms/Cloudbursts: 80-250mm rain -> High Risk
    """
    np.random.seed(random_seed)
    records = []
    
    for _, row in df.iterrows():
        district = row['District']
        state = row['State_Name']
        dfsi = float(row['DFSI'])
        
        # Scenario 1: Dry / Fair weather (Low risk)
        for _ in range(3):
            precip = float(np.clip(np.random.exponential(scale=0.5), 0, 5.0))
            rain_24h = float(np.clip(precip * np.random.uniform(0.5, 3.0), 0, 15.0))
            humidity = float(np.random.uniform(30.0, 75.0))
            temp = float(np.random.uniform(22.0, 38.0))
            wind = float(np.random.uniform(3.0, 20.0))
            clouds = float(np.random.uniform(5.0, 60.0))
            
            # Severity index calculation
            composite_score = 0.25 * (dfsi / 20.0) + 0.15 * (rain_24h / 50.0)
            risk_label = "Low"
            records.append({
                'District': district, 'State': state, 'DFSI': dfsi,
                'precipitation_mm': round(precip, 2),
                'rain_sum_24h': round(rain_24h, 2),
                'humidity_percent': round(humidity, 1),
                'temperature_c': round(temp, 1),
                'wind_speed_kmh': round(wind, 1),
                'cloud_cover_percent': round(clouds, 1),
                'risk_category': risk_label
            })
            
        # Scenario 2: Moderate Monsoon Shower / Wet spell
        for _ in range(3):
            precip = float(np.random.uniform(2.0, 15.0))
            rain_24h = float(np.random.uniform(25.0, 75.0))
            humidity = float(np.random.uniform(70.0, 95.0))
            temp = float(np.random.uniform(20.0, 32.0))
            wind = float(np.random.uniform(10.0, 35.0))
            clouds = float(np.random.uniform(65.0, 100.0))
            
            # Thresholding based on DFSI vulnerability
            adjusted_index = (dfsi * 0.45) + (rain_24h * 0.14) + (precip * 0.3)
            if adjusted_index > 17.5:
                risk_label = "High" if dfsi >= 17.0 and rain_24h >= 60 else "Medium"
            elif adjusted_index > 12.0:
                risk_label = "Medium"
            else:
                risk_label = "Low"
                
            records.append({
                'District': district, 'State': state, 'DFSI': dfsi,
                'precipitation_mm': round(precip, 2),
                'rain_sum_24h': round(rain_24h, 2),
                'humidity_percent': round(humidity, 1),
                'temperature_c': round(temp, 1),
                'wind_speed_kmh': round(wind, 1),
                'cloud_cover_percent': round(clouds, 1),
                'risk_category': risk_label
            })

        # Scenario 3: Intense Downpour / Cloudburst / High Flood event
        for _ in range(3):
            precip = float(np.random.uniform(15.0, 60.0))
            rain_24h = float(np.random.uniform(85.0, 260.0))
            humidity = float(np.random.uniform(85.0, 100.0))
            temp = float(np.random.uniform(18.0, 28.0))
            wind = float(np.random.uniform(25.0, 65.0))
            clouds = float(np.random.uniform(85.0, 100.0))
            
            adjusted_index = (dfsi * 0.45) + (rain_24h * 0.16) + (precip * 0.35)
            if adjusted_index > 22.0 or (dfsi >= 15.0 and rain_24h >= 80):
                risk_label = "High"
            elif adjusted_index > 15.0:
                risk_label = "Medium"
            else:
                risk_label = "Low"
                
            records.append({
                'District': district, 'State': state, 'DFSI': dfsi,
                'precipitation_mm': round(precip, 2),
                'rain_sum_24h': round(rain_24h, 2),
                'humidity_percent': round(humidity, 1),
                'temperature_c': round(temp, 1),
                'wind_speed_kmh': round(wind, 1),
                'cloud_cover_percent': round(clouds, 1),
                'risk_category': risk_label
            })
            
    return pd.DataFrame(records)

def main():
    print(f"Loading raw DFSI dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Data cleaning
    df['District'] = df['District'].astype(str).str.strip().str.replace(r'[\*\r\n"]', '', regex=True)
    df['State_Name'] = df['State_Name'].astype(str).str.strip().str.replace(r'[\*\r\n"]', '', regex=True)
    
    # Impute missing DFSI with median
    median_dfsi = df['DFSI'].dropna().median()
    df['DFSI'] = pd.to_numeric(df['DFSI'], errors='coerce').fillna(median_dfsi)
    
    print(f"Parsed {len(df)} districts across {df['State_Name'].nunique()} States / UTs.")
    print(f"DFSI range: min={df['DFSI'].min():.2f}, max={df['DFSI'].max():.2f}, mean={df['DFSI'].mean():.2f}")
    
    # Generate multi-scenario dataset
    dataset = generate_synthetic_weather_scenarios(df)
    print(f"Generated {len(dataset)} hydro-meteorological scenario samples.")
    print("Class distribution:")
    print(dataset['risk_category'].value_counts())
    
    feature_cols = [
        'DFSI',
        'precipitation_mm',
        'rain_sum_24h',
        'humidity_percent',
        'temperature_c',
        'wind_speed_kmh',
        'cloud_cover_percent'
    ]
    
    X = dataset[feature_cols]
    y = dataset['risk_category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train robust Pipeline with Random Forest Classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    print("Training Random Forest Classifier Pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Trained model exported successfully to {MODEL_PATH}")
    
    # Feature importances
    rf = pipeline.named_steps['classifier']
    importances = dict(zip(feature_cols, [round(float(v), 4) for v in rf.feature_importances_]))
    
    metadata = {
        'model_name': 'FloodGuard Random Forest Hydrometeorological Classifier',
        'classes': list(pipeline.classes_),
        'feature_columns': feature_cols,
        'accuracy': round(acc, 4),
        'feature_importances': importances,
        'district_count': len(df),
        'state_count': int(df['State_Name'].nunique())
    }
    
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata exported to {META_PATH}")

if __name__ == '__main__':
    main()
