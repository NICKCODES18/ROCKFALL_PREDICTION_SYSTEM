import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os
from io import BytesIO
from PIL import Image
import tifffile as tiff
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=Fa  lse)
import pandas as pd
import plotly.graph_objects as go

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Rockfall Risk Dashboard", layout="wide")

st.title("Rockfall Risk Platform")

# -------------------- Top Nav --------------------
nav_cols = st.columns([1,2,1])
with nav_cols[1]:
	page = st.radio("", ["Dashboard", "Predictions", "Alerts", "Results"], horizontal=True, label_visibility="collapsed")

# -------------------- Sidebar --------------------
with st.sidebar:
	st.header("Controls")
	# Data Source
	with st.expander("Data Source", expanded=True):
		use_live_rain = st.checkbox("Use live rainfall (Open-Meteo)", value=False)
		col_geo = st.container()
		with col_geo:
			lat = st.number_input("Latitude", value=23.2599, format="%.4f")
			lon = st.number_input("Longitude", value=77.4126, format="%.4f")
			hours_window = st.slider("Rainfall window (hours)", 1, 24, 6)
		rainfall_slider = st.slider("Manual rainfall (mm)", 0.0, 150.0, 10.0, 1.0)
	# Scenario
	with st.expander("Scenario", expanded=True):
		rain_delta = st.slider("Add rainfall (mm)", -20.0, 100.0, 0.0, 1.0)
		blast_effect = st.selectbox("Blast effect", ["None", "Local +0.1 risk", "Global +0.05 risk"]) 
	# Grid & Model
	with st.expander("Grid & Model", expanded=True):
		width = st.slider("Grid width (mock)", 10, 120, 40, 5)
		height = st.slider("Grid height (mock)", 10, 80, 25, 5)
		use_ml = st.checkbox("Use ML model (if available)", value=False)
		refresh = st.button("Refresh")
	# Alerts & Notifications
	with st.expander("Alerts & Notifications", expanded=True):
		threshold = st.slider("Alert threshold", 0.0, 1.0, 0.75, 0.01)
		max_results = st.slider("Max alerts", 10, 500, 100, 10)
		recipient = st.text_input("Email recipient", value="")
		send_email = st.button("Send email alerts")
		save_alerts_btn = st.button("Save current alerts")
	# Data Upload (used under Predictions > DEM tab)
	with st.expander("DEM Upload", expanded=False):
		dem_file = st.file_uploader("Upload DEM (GeoTIFF/TIFF/PNG)", type=["tif", "tiff", "png"]) 

# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def fetch_risk(width: int, height: int, rainfall_mm: float, use_ml: bool):
	endpoint = "ml_risk" if use_ml else "risk"
	params = {"width": width, "height": height, "rainfall_mm": rainfall_mm}
	r = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=15)
	r.raise_for_status()
	data = r.json()
	grid = np.zeros((data["height"], data["width"]))
	for c in data["cells"]:
		grid[c["y"], c["x"]] = c["risk"]
	return grid

@st.cache_data(show_spinner=False)
def fetch_alerts(width: int, height: int, rainfall_mm: float, threshold: float, max_results: int, use_ml: bool):
	endpoint = "ml_alerts" if use_ml else "alerts"
	params = {"width": width, "height": height, "rainfall_mm": rainfall_mm, "threshold": threshold, "max_results": max_results}
	r = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=20)
	r.raise_for_status()
	return r.json()

@st.cache_data(show_spinner=False)
def list_events(limit: int = 20):
	r = requests.get(f"{API_URL}/events", params={"limit": limit}, timeout=15)
	r.raise_for_status()
	return r.json()

@st.cache_data(show_spinner=False)
def fetch_health():
	r = requests.get(f"{API_URL}/health", timeout=10)
	r.raise_for_status()
	return r.json()

@st.cache_data(show_spinner=True)
def fetch_live_rainfall_mm(latitude: float, longitude: float, hours: int) -> float:
	try:
		url = (
			"https://api.open-meteo.com/v1/forecast"
			f"?latitude={latitude}&longitude={longitude}&hourly=precipitation&past_days=1&forecast_days=1&timezone=UTC"
		)
		r = requests.get(url, timeout=20)
		r.raise_for_status()
		data = r.json()
		times = data.get("hourly", {}).get("time", [])
		precip = data.get("hourly", {}).get("precipitation", [])
		if not times or not precip:
			return 0.0
		df = pd.DataFrame({"time": pd.to_datetime(times), "precip": precip})
		df = df.sort_values("time")
		total = float(df.tail(hours)["precip"].sum())
		return max(total, 0.0)
	except Exception:
		return 0.0


def read_dem(file) -> np.ndarray:
	name = file.name.lower()
	if name.endswith((".tif", ".tiff")):
		with tiff.TiffFile(file) as tf:
			arr = tf.asarray()
	else:
		img = Image.open(file)
		arr = np.array(img.convert("L"))
	arr = np.nan_to_num(arr.astype(np.float32))
	if arr.max() > 10000:
		arr = arr / arr.max() * 1000.0
	return arr


def compute_slope(dem: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
	gy, gx = np.gradient(dem, pixel_size)
	slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
	slope_deg = np.degrees(slope_rad)
	return np.clip(slope_deg, 0.0, 90.0)


def baseline_risk_from_slope(slope: np.ndarray, rainfall_mm: float) -> np.ndarray:
	slope_factor = np.interp(slope, [0, 45, 90], [0.05, 0.6, 1.0])
	rain_factor = np.interp(rainfall_mm, [0, 20, 100], [0.1, 0.6, 1.0])
	risk = slope_factor * (0.4 + 0.6 * rain_factor)
	return np.clip(risk, 0.0, 1.0)

# --------------- Common computed values ---------------
if use_live_rain:
	live_mm = fetch_live_rainfall_mm(lat, lon, hours_window)
	effective_rainfall = live_mm
else:
	effective_rainfall = rainfall_slider
scenario_rainfall = max(effective_rainfall + rain_delta, 0.0)

# --------------- Pages ---------------
if page == "Dashboard":
	st.markdown("### Overview")
	health = {}
	try:
		health = fetch_health()
	except Exception:
		health = {"status": "unknown", "model_loaded": False}
	# Compute quick grid and alerts
	try:
		grid = fetch_risk(width, height, scenario_rainfall, use_ml)
		if blast_effect == "Global +0.05 risk":
			grid = np.clip(grid + 0.05, 0.0, 1.0)
		mean_risk = float(np.mean(grid))
		max_risk = float(np.max(grid))
		p95_risk = float(np.percentile(grid, 95))
		alerts = fetch_alerts(width, height, scenario_rainfall, threshold, max_results, use_ml)
		active_count = alerts.get("count", 0)
	except Exception:
		mean_risk = max_risk = p95_risk = 0.0
		active_count = 0

	m1, m2, m3, m4 = st.columns(4)
	m1.metric("Overall Risk (avg)", f"{mean_risk:.2f}")
	m2.metric("Peak Risk (max)", f"{max_risk:.2f}")
	m3.metric("P95 Risk", f"{p95_risk:.2f}")
	m4.metric("Active Alerts", f"{active_count}")

	st.markdown("### System Status")
	s1, s2 = st.columns(2)
	with s1:
		st.write(f"API Status: {health.get('status','unknown')}")
		st.write(f"Model Loaded: {health.get('model_loaded', False)}")
	with s2:
		st.write(f"Data Source: {'Live' if use_live_rain else 'Manual'} rainfall")
		st.write(f"Scenario Rainfall: {scenario_rainfall:.1f} mm")

	st.markdown("### Recent Activity")
	try:
		ev = list_events()
		df_ev = pd.DataFrame(ev.get("events", []))
		if not df_ev.empty:
			st.dataframe(df_ev.head(5), use_container_width=True, hide_index=True)
		else:
			st.info("No events yet.")
	except Exception as e:
		st.error(f"Failed to load events: {e}")

elif page == "Predictions":
	(tab_pred, tab_dem) = st.tabs(["Risk & Alerts", "DEM & 3D"])
	with tab_pred:
		st.markdown("### Weather")
		c1, c2, c3, c4 = st.columns(4)
		c1.metric("Latitude", f"{lat:.4f}")
		c2.metric("Longitude", f"{lon:.4f}")
		c3.metric("Window (h)", f"{hours_window}")
		source = "Live" if use_live_rain else "Manual"
		c4.metric(f"Rain ({source})", f"{effective_rainfall:.1f} mm")
		st.divider()

		st.markdown("### Rock Prediction")
		st.caption(f"Scenario rainfall = {scenario_rainfall:.1f} mm | Model: {'ML' if use_ml else 'Heuristic'} | Blast: {blast_effect}")
		col1, col2 = st.columns([3,2])
		with col1:
			if refresh or True:
				try:
					grid = fetch_risk(width, height, scenario_rainfall, use_ml)
					if blast_effect == "Global +0.05 risk":
						grid = np.clip(grid + 0.05, 0.0, 1.0)
					fig, ax = plt.subplots(figsize=(7, 5))
					im = ax.imshow(grid, cmap="turbo", vmin=0, vmax=1, origin="lower")
					ax.set_title("Risk Heatmap")
					ax.set_xlabel("X (grid)")
					ax.set_ylabel("Y (grid)")
					cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
					cbar.set_label("Risk (0-1)")
					st.pyplot(fig, use_container_width=True)
				except Exception as e:
					st.error(f"Failed to fetch risk: {e}")
					st.info("Ensure the API is running at http://localhost:8000/health")
		with col2:
			st.subheader("Alerts")
			try:
				alerts = fetch_alerts(width, height, scenario_rainfall, threshold, max_results, use_ml)
				cells = alerts.get("cells", [])
				if blast_effect == "Local +0.1 risk" and cells:
					cx, cy = width // 2, height // 2
					for c in cells:
						if abs(c["x"] - cx) <= 3 and abs(c["y"] - cy) <= 3:
							c["risk"] = float(np.clip(c["risk"] + 0.1, 0.0, 1.0))
				st.write(f"Count (>= {alerts['threshold']:.2f}): {alerts['count']}")
				if cells:
					df = pd.DataFrame(cells)
					df["risk"] = df["risk"].round(3)
					if "slope_factor" in df.columns:
						df["slope_factor"] = df["slope_factor"].round(2)
					if "rain_factor" in df.columns:
						df["rain_factor"] = df["rain_factor"].round(2)
					st.dataframe(df[[c for c in ["x","y","risk","severity","slope_factor","rain_factor"] if c in df.columns]], use_container_width=True, hide_index=True)
					csv = df.to_csv(index=False).encode("utf-8")
					st.download_button("Download alerts CSV", data=csv, file_name="alerts.csv", mime="text/csv")
				else:
					st.info("No alerts at current threshold.")
			except Exception as e:
				st.error(f"Failed to fetch alerts: {e}")

	with tab_dem:
		st.subheader("DEM-based Risk (Local)")
		if dem_file is not None:
			try:
				dem = read_dem(dem_file)
				slope = compute_slope(dem)
				risk = baseline_risk_from_slope(slope, scenario_rainfall)
				fig2, ax2 = plt.subplots(figsize=(6, 4))
				im2 = ax2.imshow(risk, cmap="turbo", vmin=0, vmax=1, origin="lower")
				ax2.set_title("DEM Risk Heatmap")
				cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
				cbar2.set_label("Risk (0-1)")
				st.pyplot(fig2, use_container_width=True)

				st.subheader("3D Terrain (Prototype)")
				try:
					z = dem.astype(float)
					fig3 = go.Figure(data=[go.Surface(z=z, colorscale="YlOrRd", showscale=True)])
					fig3.update_layout(height=400, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Elev"))
					st.plotly_chart(fig3, use_container_width=True)
				except Exception as e3:
					st.info(f"3D view not rendered: {e3}")
			except Exception as e:
				st.error(f"Failed to process DEM: {e}")
				st.info("Try a smaller TIFF/PNG or ensure it contains elevation values.")
		else:
			st.info("Upload a DEM to see slope-based risk.")

elif page == "Alerts":
	st.markdown("### Active Alerts")
	try:
		alerts = fetch_alerts(width, height, scenario_rainfall, threshold, max_results, use_ml)
		cells = alerts.get("cells", [])
		st.write(f"Count (>= {alerts['threshold']:.2f}): {alerts['count']}")
		if cells:
			df = pd.DataFrame(cells)
			df["risk"] = df["risk"].round(3)
			if "slope_factor" in df.columns:
				df["slope_factor"] = df["slope_factor"].round(2)
			if "rain_factor" in df.columns:
				df["rain_factor"] = df["rain_factor"].round(2)
			st.dataframe(df, use_container_width=True, hide_index=True)
			csv = df.to_csv(index=False).encode("utf-8")
			st.download_button("Download alerts CSV", data=csv, file_name="alerts.csv", mime="text/csv")
		else:
			st.info("No alerts at current threshold.")
	except Exception as e:
		st.error(f"Failed to fetch alerts: {e}")

	if send_email:
		if not recipient:
			st.warning("Please enter an email recipient.")
		else:
			try:
				payload = {
					"recipients": [recipient],
					"subject": "Rockfall Alerts",
					"width": width,
					"height": height,
					"rainfall_mm": scenario_rainfall,
					"threshold": threshold,
					"max_results": max_results,
				}
				r = requests.post(f"{API_URL}/notify", json=payload, timeout=20)
				r.raise_for_status()
				resp = r.json()
				if resp.get("sent"):
					st.success(f"Email sent to {recipient}")
				else:
					st.info(f"Email not sent: {resp.get('reason')}")
					if resp.get("preview"):
						st.code(resp["preview"], language="text")
			except Exception as e:
				st.error(f"Failed to send email: {e}")

elif page == "Results":
	st.markdown("### Saved Events")
	try:
		ev = list_events()
		df_ev = pd.DataFrame(ev.get("events", []))
		if not df_ev.empty:
			st.dataframe(df_ev, use_container_width=True, hide_index=True)
			csv = df_ev.to_csv(index=False).encode("utf-8")
			st.download_button("Download events CSV", data=csv, file_name="events.csv", mime="text/csv")
		else:
			st.info("No events saved yet.")
	except Exception as e:
		st.error(f"Failed to load events: {e}")
