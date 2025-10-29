## Quick Start

This repo contains a minimal runnable scaffold for the Rockfall Risk Prediction Platform.

### Prerequisites

- Python 3.9+
- pip

### Install

```
pip install -r requirements.txt
```

### Dataset
 Link for download - https://www.kaggle.com/datasets/lukhilaksh/rockfall-dataset?resource=download
- Paste this in data folder

### Run API (FastAPI)

```
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: http://localhost:8000/health
- Risk (JSON): http://localhost:8000/risk?rainfall_mm=10

### Run Dashboard (Streamlit)

In a separate terminal after the API is running:

```
streamlit run dashboard/streamlit_app.py --server.port 8501
```

- Dashboard: http://localhost:8501

### What you get

- Baseline risk heatmap over a mock grid using slope + rainfall heuristic
- Simple what-if control for rainfall or live rainfall (Open-Meteo)
- Alerts list, email notifications (optional), and events persistence
- DEM upload with local slope-based risk

### Environment (.env)

```
# API
API_URL=http://127.0.0.1:8000

# SMTP for email alerts (optional)
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USER=your_user
SMTP_PASS=your_password
SMTP_FROM=alerts@yourdomain.com

# SQLite path (optional)
DB_PATH=alerts.db
```

### Persistence (SQLite)

- Click "Save current alerts" in the dashboard to store a snapshot.
- View recent events in the Events panel. DB path defaults to `alerts.db` (override via `DB_PATH`).

### Live rainfall (Open-Meteo)

- Toggle "Use live rainfall" in the sidebar.
- Set latitude/longitude and hour window; the effective rainfall will use recent hourly precipitation.

### Next steps

- Replace mock grid with real DEM tiling and sensor/weather fusion
- Add calibration, severity classes, and SHAP explanations
- Prepare deployment (Docker) and role-based access
