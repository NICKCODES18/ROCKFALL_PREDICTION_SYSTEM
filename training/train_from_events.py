import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss
import joblib

# Heuristic alias sets for column inference (case-insensitive)
ALIASES = {
    "id": ["tile_id", "segment_id", "id", "cell_id"],
    "time": ["timestamp", "time", "datetime", "date"],
    # labels: explicit or derived
    "label": ["label", "event", "y", "is_event", "rockfall", "rockfall_count", "probability_occurred"],
    # core drivers
    "slope": ["slope", "slope_deg", "slope_degree", "slope_deg"],
    "rain": ["rainfall_mm", "rain", "precip", "precip_mm"],
    # common numeric features from user schema
    "aspect": ["aspect", "aspect_deg"],
    "elev": ["elevation", "elevation_m"],
    "fault_dist": ["distance_to_fault_km"],
    "hardness": ["rock_hardness"],
    "fracture": ["fracture_density"],
    "snow": ["snow_mm"],
    "temp": ["temperature_c"],
    "wind": ["wind_speed_kmh"],
    "humidity": ["humidity_percent"],
    "prior_events": ["prior_events"],
    "rock_size": ["rock_size_cm"],
    "rock_vol": ["rock_volume_m3"],
    "road_dist": ["distance_to_road_m"],
    "village_dist": ["distance_to_village_m"],
    "pop": ["population_nearby"],
    "slope_len": ["slope_length_m"],
    "catchment": ["catchment_area_m2"],
    "suscept": ["susceptibility_score"],
    "impact": ["impact_level_score"],
    "velocity": ["rock_velocity_mps"],
    "ndvi": ["ndvi"],
    "moisture": ["ground_moisture"],
    "seismic": ["seismic_activity"],
    "density": ["rock_density"],
    "energy": ["energy_released"],
    "stability": ["slope_stability_index"],
}

# Categorical columns we can factorize (if present)
CATEGORICAL_CANDIDATES = [
    "rock_type", "soil_type", "lithology", "vegetation", "land_cover",
    "region", "subregion", "event_type", "rock_classification", "monitoring_station"
]

# Preferred numeric feature order
DEFAULT_FEATURE_ORDER = [
    "slope", "rain", "aspect", "elev", "fault_dist", "hardness", "fracture", "snow",
    "temp", "wind", "humidity", "prior_events", "rock_size", "rock_vol", "road_dist",
    "village_dist", "pop", "slope_len", "catchment", "suscept", "impact", "velocity",
    "ndvi", "moisture", "seismic", "density", "energy", "stability"
]


def infer_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in columns}
    mapping: Dict[str, Optional[str]] = {k: None for k in (list(ALIASES.keys()) + ["id", "time", "label"])}
    for key, aliases in ALIASES.items():
        for alias in aliases:
            if alias in cols_lower:
                mapping[key] = cols_lower[alias]
                break
    return mapping


def select_feature_columns(mapping: Dict[str, Optional[str]]) -> List[str]:
    features = []
    for key in DEFAULT_FEATURE_ORDER:
        col = mapping.get(key)
        if col is not None:
            features.append(col)
    return features


def read_sample(csv_path: str, nrows: int = 20000) -> pd.DataFrame:
    return pd.read_csv(csv_path, nrows=nrows)


def ensure_label_column(df: pd.DataFrame, mapping: Dict[str, Optional[str]], prob_threshold: float) -> pd.DataFrame:
    # If explicit label exists, normalize to {0,1}
    label_col = mapping.get("label")
    if label_col and label_col in df.columns:
        lc = label_col.lower()
        if lc == "rockfall_count":
            df[label_col] = (pd.to_numeric(df[label_col], errors='coerce').fillna(0) > 0).astype(int)
        elif lc == "probability_occurred":
            df[label_col] = (pd.to_numeric(df[label_col], errors='coerce').fillna(0) >= prob_threshold).astype(int)
        else:
            df[label_col] = (pd.to_numeric(df[label_col], errors='coerce').fillna(0) > 0).astype(int)
        return df
    # Fallbacks
    if "Rockfall_Count" in df.columns:
        df["Rockfall_Count"] = (pd.to_numeric(df["Rockfall_Count"], errors='coerce').fillna(0) > 0).astype(int)
        mapping["label"] = "Rockfall_Count"
    elif "Probability_Occurred" in df.columns:
        df["Probability_Occurred"] = (pd.to_numeric(df["Probability_Occurred"], errors='coerce').fillna(0) >= prob_threshold).astype(int)
        mapping["label"] = "Probability_Occurred"
    else:
        raise ValueError("No label column found. Provide mapping with --map_json for 'label'.")
    return df


def factorize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_CANDIDATES:
        for variant in [col, col.title().replace("_", "_"), col.replace("_", " ")]:
            if variant in df.columns:
                codes, _ = pd.factorize(df[variant].astype(str).fillna(""))
                df[variant + "__code"] = codes.astype(np.int32)
                break
    return df


def build_dataset(csv_path: str, mapping: Dict[str, Optional[str]], sample_rows: int = 300000, class_balance: float = 3.0, prob_threshold: float = 0.5) -> pd.DataFrame:
    feature_cols = select_feature_columns(mapping)
    # Add factorized categorical codes if present later
    usecols = [c for c in {*(feature_cols), *(CATEGORICAL_CANDIDATES), mapping.get("label") or "Rockfall_Count", mapping.get("slope") or "Slope_deg"} if c is not None]

    pos_rows: List[pd.DataFrame] = []
    neg_rows: List[pd.DataFrame] = []

    chunksize = 200000
    total = 0
    for chunk in pd.read_csv(csv_path, usecols=lambda c: True, chunksize=chunksize):
        # Ensure label exists/derived in this chunk
        chunk = ensure_label_column(chunk, mapping, prob_threshold)
        # Keep only necessary columns if available
        keep_cols = list({*(feature_cols), *(CATEGORICAL_CANDIDATES), mapping["label"]} & set(chunk.columns))
        df = chunk[keep_cols].copy()
        # Numeric coercion
        for c in feature_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=[mapping["label"], *(set(feature_cols) & set(df.columns))])
        # Categorical factorization into codes
        df = factorize_categoricals(df)
        # Assemble final feature set: numeric + any __code
        extra_cats = [c for c in df.columns if c.endswith("__code")]
        final_features = [c for c in feature_cols if c in df.columns] + extra_cats
        df = df[final_features + [mapping["label"]]].copy()
        # Split pos/neg
        pos = df[df[mapping["label"]] == 1]
        neg = df[df[mapping["label"]] == 0]
        # Subsample to balance
        if len(pos) > sample_rows // (1 + class_balance):
            pos = pos.sample(sample_rows // int(1 + class_balance), random_state=42)
        if len(neg) > int(class_balance * max(1, len(pos))):
            neg = neg.sample(int(class_balance * max(1, len(pos))), random_state=42)
        pos_rows.append(pos)
        neg_rows.append(neg)
        total += len(pos) + len(neg)
        if total >= sample_rows:
            break

    data = pd.concat(pos_rows + neg_rows, axis=0, ignore_index=True)
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return data


def train_and_save(df: pd.DataFrame, label_col: str, out_path: str):
    feature_names = [c for c in df.columns if c != label_col]
    X = df[feature_names].values
    y = df[label_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = make_pipeline(
        StandardScaler(with_mean=False),  # robust for sparse-like feature sets
        LogisticRegression(max_iter=600, C=0.7, class_weight="balanced", n_jobs=1)
    )
    model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    aps = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    print(f"AP: {aps:.3f} | Brier: {brier:.3f} | n={len(df)} | d={len(feature_names)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": feature_names
    }, out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("data", "rockfall.csv"))
    parser.add_argument("--out", default=os.path.join("models", "site_baseline.joblib"))
    parser.add_argument("--map_json", default=None, help="Optional column mapping JSON file")
    parser.add_argument("--sample_rows", type=int, default=300000)
    parser.add_argument("--class_balance", type=float, default=3.0)
    parser.add_argument("--prob_label_threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Infer mapping
    sample = read_sample(args.csv, nrows=20000)
    mapping = infer_columns(list(sample.columns))

    if args.map_json and os.path.exists(args.map_json):
        with open(args.map_json, 'r') as f:
            user_map = json.load(f)
        for k, v in user_map.items():
            if v:
                mapping[k] = v
    print("Column mapping:")
    print(json.dumps(mapping, indent=2))

    df = build_dataset(args.csv, mapping, sample_rows=args.sample_rows, class_balance=args.class_balance, prob_threshold=args.prob_label_threshold)
    label_col = mapping["label"] if mapping["label"] else ("Rockfall_Count" if "Rockfall_Count" in df.columns else "Probability_Occurred")
    train_and_save(df, label_col, args.out)


if __name__ == "__main__":
    main()
