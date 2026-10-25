# ?? FloodGuard AI
### Real-Time Predictive Hydro-Meteorological Flood Risk Intelligence
**3rd-Semester AI / CSE Academic Project**

FloodGuard AI is a modern, production-ready Full-Stack Web Application designed for real-time flood susceptibility assessment and meteorological early warning.

---

## ??? Architecture & Core Components

1. **Hydrometeorological AI Pipeline (	rain_and_export.py & model.pkl)**:
   - Ingests empirical District Flood Susceptibility Index (DFSI.csv) across 730+ Indian districts.
   - Combines baseline geomorphological vulnerability with dynamic weather drivers (precipitation intensity, 24h cumulative rainfall, humidity, wind, cloud cover).
   - Trains a Scikit-Learn **Random Forest Classifier** achieving **98.18% Accuracy** with balanced class weights and calibrated continuous risk percentage outputs.

2. **Flask REST API Backend (pp.py)**:
   - GET /: Renders the high-fidelity glassmorphic dashboard.
   - GET /api/states: Provides state-district relational mappings.
   - POST /api/predict: Integrates live Open-Meteo REST API geocoding and real-time weather metrics, executes ML inference, selects contextual impression quotes, and formats dynamic UI states.
   - Includes an Academic Demo Mode selector to easily demonstrate both normal sunny river valleys and severe 72% monsoon downpours during project vivas.

3. **Modern Glassmorphic Frontend (	emplates/index.html)**:
   - High-blur translucent panels (ackdrop-filter: blur(24px)), soft border highlights, and crisp typography (Outfit & Plus Jakarta Sans).
   - Circular animated SVG risk gauge with dynamic color-coding (HIGH / SEVERE in #FF4B4B, MEDIUM in #FFAA00, LOW / SAFE in #00CC66).
   - Smooth curved 6-hour forecast trend line chart rendered on an HTML5 canvas.
   - Seamless cross-fading 4K realistic weather backdrops for rain, sun, storm, and overcast conditions.
   - Witty yet formal contextual impression commentary.

---

## ?? Project Directory Structure

`	ext
FloodGuard_Project/
??? app.py                      # Flask REST API backend server
??? DFSI.csv                    # Cleaned 730+ district flood susceptibility index
??? train_and_export.py         # ML model training and evaluation script
??? train_and_export.ipynb      # Academic Jupyter notebook with analysis
??? model.pkl                   # Trained Scikit-Learn Random Forest pipeline
??? model_metadata.json         # Model metrics and feature importances
??? templates/
?   ??? index.html              # High-fidelity glassmorphism dashboard
??? static/
?   ??? images/                 # 4K HD weather backdrops & assets
??? README.md                   # Complete academic documentation
`

---

## ?? How to Run Locally

### 1. Activate Virtual Environment
`ash
# Windows PowerShell:
.venv\Scripts\Activate.ps1
`

### 2. (Optional) Re-train ML Model
`ash
python train_and_export.py
`

### 3. Launch the Web Application
`ash
python app.py
`

Open your browser at: **http://localhost:5000**
