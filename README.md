---

# 🪨 Rockfall Risk Prediction Platform (LithoSense)

An AI-powered system for predicting **rockfall risks** in mining regions using **DEM (Digital Elevation Models)**, **sensor data**, and **weather conditions**.
Built using **FastAPI**, **Streamlit**, and **Machine Learning**, the system generates **real-time risk maps** and triggers **alerts** to prevent mining accidents.

---

## 🚀 Quick Start

### 📋 Prerequisites

* Python 3.9+
* pip

---

### ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

### 📂 Dataset

Download the dataset from Kaggle:
🔗 [Rockfall Dataset](https://www.kaggle.com/datasets/lukhilaksh/rockfall-dataset?resource=download)
Paste the downloaded files into the `/data` folder.

---

### 🧠 Run API (FastAPI)

```bash
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints:

* Health → [http://localhost:8000/health](http://localhost:8000/health)
* Risk (JSON) → [http://localhost:8000/risk?rainfall_mm=10](http://localhost:8000/risk?rainfall_mm=10)

---

### 📊 Run Dashboard (Streamlit)

In a separate terminal, after starting the API:

```bash
streamlit run dashboard/streamlit_app.py --server.port 8501
```

Access the dashboard at:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧩 What You Get

✅ Baseline **risk heatmap** using slope + rainfall heuristic
✅ “What-if” controls for rainfall or live data via Open-Meteo
✅ Alerts list and optional **email notifications**
✅ DEM upload support for local slope-based risk prediction

---

## 🌎 Environment (.env)

```bash
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

---

## 💾 Persistence (SQLite)

* Click **“Save current alerts”** in the dashboard to store a snapshot.
* View recent alerts in the **Events panel**.
* Database defaults to `alerts.db` (customizable via `DB_PATH`).

---

## 🌦️ Live Rainfall (Open-Meteo Integration)

* Toggle **“Use live rainfall”** in the sidebar.
* Set latitude, longitude, and hour window to fetch recent precipitation.

---

## 🧭 Next Steps

* Replace mock grid with real **DEM tiling + sensor fusion**
* Add **calibration**, **severity classes**, and **SHAP explanations**
* Prepare **Docker deployment** and **role-based access**

---

# 💡 Smart India Hackathon (SIH) — LithoSense Overview

## 🧰 Feasibility

* Open-source tools ensure **easy and affordable implementation**
* Works on **cloud, local servers, or Raspberry Pi devices**
* Integrates **weather, sensor, and geological data** for accuracy
* Lightweight ML models optimized for **real-time predictions**
* Modular architecture allows **seamless future upgrades**

---

## 💰 Viability

* Cost-effective and **industry-friendly** solution
* Scales from **small to large mining sites**
* Prevents **mine shutdowns** and **operational losses**
* Ensures **worker safety** and **reduces accidents**
* Creates **strong market demand** in global mining tech

---

## 🔍 Detailed Solution Explanation

* AI model processes **DEM, drone imagery, sensor, and weather data**
* Generates **real-time risk maps** with **probability-based forecasts**
* Dashboard with **SMS / Email alerts** for immediate action

---

## ⚙️ Innovation and Uniqueness

* **Multi-source data fusion** in one predictive platform
* “What-if” simulation for risk forecasting
* **Low-cost, open-source**, and **scalable** across mining sites

---

## 🧠 How It Solves the Problem

* Shifts from **manual/reactive** checks to **predictive monitoring**
* Reduces **accidents, downtime, and financial losses**
* Enables **continuous, data-driven insights** for mine safety

---

## 🖥️ Dashboard UI Overview

* **Modules:** Dashboard | Predictions | Alerts | Results
* **Example Values:**

  * Risk Level: **HIGH**
  * Latitude: `23.2599`
  * Longitude: `77.4126`
  * Rainfall (forecast): `10.0 mm`

---

## 👨‍💻 Author

**Nikunj Jain**
🔗 [LinkedIn](https://www.linkedin.com/in/nikunjjain29/)
📧 [Email](mailto:nikunjjain294@gmail.com)

---

