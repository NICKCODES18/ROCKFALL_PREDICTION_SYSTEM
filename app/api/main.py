from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import numpy as np
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)
import os
import smtplib
from email.message import EmailMessage
import sqlite3
from datetime import datetime
import joblib

app = FastAPI(title="Rockfall Risk API", version="0.4.1")

DB_PATH = os.environ.get("DB_PATH", "alerts.db")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "baseline.joblib"))

# Try load model
try:
	ML_MODEL = joblib.load(MODEL_PATH)
except Exception:
	ML_MODEL = None


def init_db():
	conn = sqlite3.connect(DB_PATH)
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			rainfall_mm REAL NOT NULL,
			threshold REAL NOT NULL,
			count INTEGER NOT NULL,
			cells TEXT NOT NULL
		);
		"""
	)
	conn.commit()
	conn.close()

init_db()


class CellRisk(BaseModel):
	x: int
	y: int
	risk: float


class RiskGrid(BaseModel):
	width: int
	height: int
	cells: List[CellRisk]


def generate_mock_slope(width: int, height: int, seed: int = 42) -> np.ndarray:
	rng = np.random.default_rng(seed)
	base = rng.uniform(10, 60, size=(height, width))
	gradient = np.linspace(0, 15, width)
	slope = base + gradient
	return np.clip(slope, 0, 90)


def baseline_risk(slope: np.ndarray, rainfall_mm: float) -> np.ndarray:
	slope_factor = np.interp(slope, [0, 45, 90], [0.05, 0.6, 1.0])
	rain_factor = np.interp(rainfall_mm, [0, 20, 100], [0.1, 0.6, 1.0])
	risk = slope_factor * (0.4 + 0.6 * rain_factor)
	return np.clip(risk, 0.0, 1.0)


def compute_factors(slope: np.ndarray, rainfall_mm: float):
	slope_factor = np.interp(slope, [0, 45, 90], [0.05, 0.6, 1.0])
	rain_factor = float(np.interp(rainfall_mm, [0, 20, 100], [0.1, 0.6, 1.0]))
	risk = np.clip(slope_factor * (0.4 + 0.6 * rain_factor), 0.0, 1.0)
	return slope_factor, rain_factor, risk


def severity_from_risk(r: float) -> str:
	if r >= 0.90:
		return "Critical"
	elif r >= 0.75:
		return "High"
	elif r >= 0.50:
		return "Moderate"
	else:
		return "Low"


@app.get("/health")
def health():
	return {"status": "ok", "model_loaded": ML_MODEL is not None}


@app.get("/risk", response_model=RiskGrid)
def get_risk(width: int = Query(40, ge=5, le=200), height: int = Query(25, ge=5, le=200), rainfall_mm: float = Query(10.0, ge=0.0, le=200.0)):
	slope = generate_mock_slope(width, height)
	risk = baseline_risk(slope, rainfall_mm)
	cells = [CellRisk(x=int(x), y=int(y), risk=float(risk[y, x])) for y in range(height) for x in range(width)]
	return RiskGrid(width=width, height=height, cells=cells)


class AlertCell(BaseModel):
	x: int
	y: int
	risk: float
	severity: str
	slope_factor: float
	rain_factor: float


class AlertsResponse(BaseModel):
	count: int
	threshold: float
	cells: List[AlertCell]


@app.get("/alerts", response_model=AlertsResponse)
def get_alerts(width: int = Query(40, ge=5, le=200), height: int = Query(25, ge=5, le=200), rainfall_mm: float = Query(10.0, ge=0.0, le=200.0), threshold: float = Query(0.75, ge=0.0, le=1.0), max_results: int = Query(100, ge=1, le=1000)):
	slope = generate_mock_slope(width, height)
	slope_f, rain_f, risk = compute_factors(slope, rainfall_mm)
	ys, xs = np.where(risk >= threshold)
	risks = risk[ys, xs]
	order = np.argsort(-risks)
	cells = []
	for i in order[:max_results]:
		x = int(xs[i]); y = int(ys[i])
		r = float(risks[i])
		cells.append(AlertCell(x=x, y=y, risk=r, severity=severity_from_risk(r), slope_factor=float(slope_f[y, x]), rain_factor=float(rain_f)))
	return AlertsResponse(count=int(len(xs)), threshold=threshold, cells=cells)


# ML endpoints
@app.get("/ml_risk", response_model=RiskGrid)
def ml_risk(width: int = Query(40, ge=5, le=200), height: int = Query(25, ge=5, le=200), rainfall_mm: float = Query(10.0, ge=0.0, le=200.0)):
	slope = generate_mock_slope(width, height)
	if ML_MODEL is None:
		# fallback
		risk = baseline_risk(slope, rainfall_mm)
	else:
		slope_factor = np.interp(slope, [0, 45, 90], [0.05, 0.6, 1.0])
		rain_factor = np.interp(rainfall_mm, [0, 20, 100], [0.1, 0.6, 1.0])
		X = np.column_stack([
			slope,
			np.full_like(slope, rainfall_mm),
			slope_factor,
			np.full_like(slope, rain_factor),
			slope * rainfall_mm / 100.0,
		]).reshape(-1, 5)
		probs = ML_MODEL.predict_proba(X)[:, 1].reshape(slope.shape)
		risk = np.clip(probs, 0.0, 1.0)
		risk = _smooth(risk) # Apply smoothing
	cells = [CellRisk(x=int(x), y=int(y), risk=float(risk[y, x])) for y in range(height) for x in range(width)]
	return RiskGrid(width=width, height=height, cells=cells)


@app.get("/ml_alerts", response_model=AlertsResponse)
def ml_alerts(width: int = Query(40, ge=5, le=200), height: int = Query(25, ge=5, le=200), rainfall_mm: float = Query(10.0, ge=0.0, le=200.0), threshold: float = Query(0.75, ge=0.0, le=1.0), max_results: int = Query(100, ge=1, le=1000)):
	slope = generate_mock_slope(width, height)
	if ML_MODEL is None:
		slope_f, rain_f, risk = compute_factors(slope, rainfall_mm)
	else:
		slope_f = np.interp(slope, [0, 45, 90], [0.05, 0.6, 1.0])
		rain_f = float(np.interp(rainfall_mm, [0, 20, 100], [0.1, 0.6, 1.0]))
		X = np.column_stack([
			slope,
			np.full_like(slope, rainfall_mm),
			slope_f,
			np.full_like(slope, rain_f),
			slope * rainfall_mm / 100.0,
		]).reshape(-1, 5)
		probs = ML_MODEL.predict_proba(X)[:, 1].reshape(slope.shape)
		risk = np.clip(probs, 0.0, 1.0)
		risk = _smooth(risk) # Apply smoothing
		rain_f = float(rain_f)
	# common assemble
	ys, xs = np.where(risk >= threshold)
	risks = risk[ys, xs]
	order = np.argsort(-risks)
	cells = []
	for i in order[:max_results]:
		x = int(xs[i]); y = int(ys[i])
		r = float(risks[i])
		cells.append(AlertCell(x=x, y=y, risk=r, severity=severity_from_risk(r), slope_factor=float(slope_f[y, x]), rain_factor=float(rain_f)))
	return AlertsResponse(count=int(len(xs)), threshold=threshold, cells=cells)


class NotifyRequest(BaseModel):
	recipients: List[str]
	subject: str | None = None
	width: int = 40
	height: int = 25
	rainfall_mm: float = 10.0
	threshold: float = 0.75
	max_results: int = 50


class NotifyResponse(BaseModel):
	sent: bool
	recipient_count: int
	reason: str | None = None
	preview: str | None = None


def _build_alert_email_body(alerts: List[AlertCell], threshold: float, rainfall_mm: float) -> str:
	lines = [
		f"Rockfall Alerts (threshold >= {threshold:.2f}, rainfall={rainfall_mm:.1f} mm)",
		"",
	]
	if not alerts:
		lines.append("No alerts at current threshold.")
	else:
		lines.append("Top alerts:")
		for i, c in enumerate(alerts[:20], 1):
			lines.append(f"{i:02d}. cell=({c.x},{c.y}) risk={c.risk:.3f} severity={c.severity} (slope={c.slope_factor:.2f}, rain={c.rain_factor:.2f})")
	return "\n".join(lines)


@app.post("/notify", response_model=NotifyResponse)
def send_notifications(req: NotifyRequest):
	slope = generate_mock_slope(req.width, req.height)
	slope_f, rain_f, risk = compute_factors(slope, req.rainfall_mm)
	ys, xs = np.where(risk >= req.threshold)
	risks = risk[ys, xs]
	order = np.argsort(-risks)
	alerts = []
	for i in order[:req.max_results]:
		x = int(xs[i]); y = int(ys[i])
		r = float(risks[i])
		alerts.append(AlertCell(x=x, y=y, risk=r, severity=severity_from_risk(r), slope_factor=float(slope_f[y, x]), rain_factor=float(rain_f)))

	body = _build_alert_email_body(alerts, req.threshold, req.rainfall_mm)
	subject = req.subject or "Rockfall Alerts"

	host = os.environ.get("SMTP_HOST")
	port = int(os.environ.get("SMTP_PORT", "587"))
	user = os.environ.get("SMTP_USER")
	password = os.environ.get("SMTP_PASS")
	from_addr = os.environ.get("SMTP_FROM", user or "alerts@example.com")

	missing = [k for k in ["SMTP_HOST", "SMTP_USER", "SMTP_PASS"] if not os.environ.get(k)]
	if missing:
		return NotifyResponse(sent=False, recipient_count=len(req.recipients), reason=f"Missing SMTP env vars: {', '.join(missing)}", preview=body)

	try:
		msg = EmailMessage()
		msg["Subject"] = subject
		msg["From"] = from_addr
		msg["To"] = ", ".join(req.recipients)
		msg.set_content(body)

		with smtplib.SMTP(host, port) as server:
			server.starttls()
			server.login(user, password)
			server.send_message(msg)
		sent = True
		reason = None
	except Exception as e:
		sent = False
		reason = str(e)

	return NotifyResponse(sent=sent, recipient_count=len(req.recipients), reason=reason, preview=None if sent else body)


class SaveEventsRequest(BaseModel):
	width: int = 40
	height: int = 25
	rainfall_mm: float = 10.0
	threshold: float = 0.75
	max_results: int = 100


class SaveEventsResponse(BaseModel):
	saved: int
	event_id: int | None


@app.post("/events", response_model=SaveEventsResponse)
def save_events(req: SaveEventsRequest):
	slope = generate_mock_slope(req.width, req.height)
	risk = baseline_risk(slope, req.rainfall_mm)
	ys, xs = np.where(risk >= req.threshold)
	risks = risk[ys, xs]
	order = np.argsort(-risks)
	alerts = [{"x": int(xs[i]), "y": int(ys[i]), "risk": float(risks[i])} for i in order[:req.max_results]]

	created_at = datetime.utcnow().isoformat()
	cells_str = str(alerts)

	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()
	cur.execute(
		"INSERT INTO events (created_at, rainfall_mm, threshold, count, cells) VALUES (?, ?, ?, ?, ?)",
		(created_at, req.rainfall_mm, req.threshold, len(alerts), cells_str),
	)
	event_id = cur.lastrowid
	conn.commit()
	conn.close()
	return SaveEventsResponse(saved=len(alerts), event_id=event_id)


class ListEventsResponse(BaseModel):
	events: List[dict]


@app.get("/events", response_model=ListEventsResponse)
def list_events(limit: int = Query(20, ge=1, le=200)):
	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()
	cur.execute("SELECT id, created_at, rainfall_mm, threshold, count FROM events ORDER BY id DESC LIMIT ?", (limit,))
	rows = cur.fetchall()
	conn.close()
	events = [
		{"id": r[0], "created_at": r[1], "rainfall_mm": r[2], "threshold": r[3], "count": r[4]}
		for r in rows
	]
	return ListEventsResponse(events=events)


def _smooth(grid: np.ndarray) -> np.ndarray:
	# Simple 3x3 mean filter without external deps
	kernel = np.array([[1,1,1],[1,2,1],[1,1,1]], dtype=float)
	kernel = kernel / kernel.sum()
	h, w = grid.shape
	p = 1
	padded = np.pad(grid, ((p,p),(p,p)), mode='edge')
	out = np.zeros_like(grid)
	for y in range(h):
		for x in range(w):
			patch = padded[y:y+3, x:x+3]
			out[y, x] = (patch * kernel).sum()
	return out
