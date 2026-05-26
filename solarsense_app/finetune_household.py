"""
finetune_household.py

Fine-tunes appliance usage prediction models on household-specific data.

Pipeline:
    1. Load disaggregated per-appliance binary hourly usage from the household
    2. Fetch matching weather from Open-Meteo for the household's location
    3. For appliances with a base model:
           warm-start XGBoost from the base, add new trees on household data
    4. For appliances with no base model:
           train a fresh XGBoost from scratch on household data only
    5. Save one .pkl per household per appliance

Design decisions:
    - We ALWAYS retrain from the base model, not from the previous
      household-tuned model. This prevents drift accumulating over
      successive retraining runs.
    - The base model's knowledge (e.g. cooling responds to temperature)
      is preserved because warm-starting adds new trees on top of the
      base booster without modifying existing trees.
    - New household trees use a lower learning rate than base training
      so they refine rather than overwrite existing patterns.
    - Minimum one week of data required before fine-tuning.

Requires: xgboost, scikit-learn, pandas, numpy, requests
"""

import os
import pickle
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIGURATION
# ============================================================

# Path to the base models trained on ResNet / Plegma / REFIT
BASE_MODELS_DIR = r"C:\Users\moham\Documents\490 project new\final_models"

# Where to save per-household fine-tuned models
# Each household gets its own subfolder: HOUSEHOLDS_DIR/<household_id>/
HOUSEHOLDS_DIR  = r"C:\Users\moham\Documents\490 project new\household_models"

# Minimum hours of data required before fine-tuning
# 168 = one week of hourly data
MIN_HOURS = 168

# Feature columns — must match base model training exactly
FEATURE_COLS = [
    'weather_drybulb_temp_c',
    'weather_relative_humidity_pct',
    'hour',
    'day_of_week',
    'is_weekend',
    'month',
]

TV_FEATURE_COLS = [
    'hour',
    'day_of_week',
    'is_weekend',
    'is_evening',
]

def get_features_for(appliance, df, model=None):
    """
    Returns the correct feature DataFrame for a given appliance.
    If a model is provided, uses its stored feature_names_in_ to
    guarantee the column order matches exactly what it was trained on.
    """
    if model is not None and hasattr(model, 'feature_names_in_'):
        cols = [str(c) for c in model.feature_names_in_]
        return df[cols]
    if 'television' in appliance.lower():
        return df[TV_FEATURE_COLS]
    return df[FEATURE_COLS]

# Target column naming convention — matches base model filenames
# e.g. "elec_cooling_on" -> base model "elec_cooling_on.pkl"
TARGET_PREFIX = "elec_"
TARGET_SUFFIX = "_on"

# XGBoost settings for the household fine-tuning trees
# More trees + slightly higher LR = stronger household adaptation
# Base knowledge is preserved because warm-start freezes base trees
FINETUNE_N_ESTIMATORS  = 150   # was 50
FINETUNE_LEARNING_RATE = 0.05  # was 0.01 — refines faster
FINETUNE_PASSES        = 2     # repeat household data N times per run

# Scale pos weight cap — same as base training
SCALE_POS_WEIGHT_CAP = 5.0

# Open-Meteo archive API
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# 1. WEATHER FETCHER
# ============================================================
def fetch_weather(lat, lon, tz, start_date, end_date):
    """
    Fetches hourly temperature and humidity from Open-Meteo
    for the household's location and the date range covered
    by the household data.

    Returns a DataFrame with columns:
        timestamp, weather_drybulb_temp_c, weather_relative_humidity_pct
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     "temperature_2m,relative_humidity_2m",
        "timezone":   tz,
    }

    for attempt in range(3):
        try:
            r = requests.get(OPENMETEO_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                raise ConnectionError(f"Weather fetch failed after 3 attempts: {e}")

    df = pd.DataFrame({
        "timestamp":                     pd.to_datetime(data["hourly"]["time"]),
        "weather_drybulb_temp_c":        data["hourly"]["temperature_2m"],
        "weather_relative_humidity_pct": data["hourly"]["relative_humidity_2m"],
    })
    return df


# ============================================================
# 2. DATA LOADER
# ============================================================
def load_household_data(usage_path, lat, lon, tz, prefetched_weather=None):
    """
    Loads the household's disaggregated usage CSV and attaches
    weather data from Open-Meteo.

    Expected CSV format (produced by the disaggregation layer):
        timestamp, <appliance_1>, <appliance_2>, ...
        2024-03-01 00:00:00, 0, 1, ...
        2024-03-01 01:00:00, 1, 0, ...

    Each appliance column contains binary 0/1 hourly values.
    Column names do NOT need a prefix — the script auto-detects
    appliance columns as anything that isn't 'timestamp'.

    Returns a DataFrame with features + all appliance target columns.
    """
    df = pd.read_csv(usage_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    n_hours = len(df)
    print(f"  Loaded {n_hours:,} hourly rows")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    if n_hours < MIN_HOURS:
        raise ValueError(
            f"Only {n_hours} hours of data — need at least {MIN_HOURS} "
            f"({MIN_HOURS // 24} days). Collect more data before fine-tuning."
        )

    # Add time features
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['month']       = df['timestamp'].dt.month
    df['is_evening']  = df['hour'].between(18, 23).astype(int)

    start_date = df['timestamp'].min().strftime('%Y-%m-%d')
    end_date   = df['timestamp'].max().strftime('%Y-%m-%d')
    # Use pre-fetched weather if provided
    print(f"  Fetching weather for {start_date} → {end_date} ...")

    if prefetched_weather is not None:
        print(f"  Using pre-fetched weather ({len(prefetched_weather)} rows)")
        df_weather = prefetched_weather.copy()
    else:
        df_weather = fetch_weather(lat, lon, tz, start_date, end_date)

    # Merge on hourly timestamp
    # Drop time columns from weather that are already in df to avoid _x/_y suffixes
    weather_drop = ['timestamp'] + [c for c in df_weather.columns
                                    if c in df.columns and c != 'ts_h']
    df['ts_h']         = df['timestamp'].dt.floor('h')
    df_weather['ts_h'] = df_weather['timestamp'].dt.floor('h')
    df = df.merge(
        df_weather.drop(columns=weather_drop, errors='ignore'),
        on='ts_h', how='left'
    ).drop(columns=['ts_h'])

    missing_weather = df['weather_drybulb_temp_c'].isna().sum()
    if missing_weather > 0:
        print(f"  [!] {missing_weather} rows missing weather — forward filling")
        df[['weather_drybulb_temp_c', 'weather_relative_humidity_pct']] = (
            df[['weather_drybulb_temp_c', 'weather_relative_humidity_pct']]
            .ffill().bfill()
        )

    df = df.dropna(subset=['weather_drybulb_temp_c', 'weather_relative_humidity_pct'])
    print(f"  Final rows after weather merge: {len(df):,}")
    return df


# ============================================================
# 3. APPLIANCE DETECTOR
# ============================================================
def detect_appliance_columns(df):
    """
    Identifies appliance target columns in the household DataFrame.
    Returns a list of column names that are binary (0/1) and not
    feature or metadata columns.
    """
    non_target = set(FEATURE_COLS) | set(TV_FEATURE_COLS) | {'timestamp'}
    candidates = [c for c in df.columns if c not in non_target]

    appliance_cols = []
    for col in candidates:
        unique = df[col].dropna().unique()
        if set(unique).issubset({0, 1, 0.0, 1.0}):
            appliance_cols.append(col)

    return appliance_cols


# ============================================================
# 4. BASE MODEL NAME RESOLVER
# ============================================================
def find_base_model(appliance_col, base_models_dir):
    """
    Tries to find a matching base model .pkl for an appliance column.

    Matching strategy (in order):
      1. Exact filename match:     appliance_col + ".pkl"
      2. With standard prefix:     "elec_" + appliance_col + "_on.pkl"
      3. Partial name match:       any .pkl whose stem is contained in
                                   appliance_col or vice versa

    Returns the full path if found, None otherwise.
    """
    # exact
    exact = os.path.join(base_models_dir, f"{appliance_col}.pkl")
    if os.path.exists(exact):
        return exact

    # with prefix/suffix
    standard = os.path.join(
        base_models_dir,
        f"{TARGET_PREFIX}{appliance_col}{TARGET_SUFFIX}.pkl"
    )
    if os.path.exists(standard):
        return standard

    # partial match
    col_lower = appliance_col.lower()
    for fname in os.listdir(base_models_dir):
        if not fname.endswith('.pkl'):
            continue
        stem = fname[:-4].lower()
        if stem in col_lower or col_lower in stem:
            return os.path.join(base_models_dir, fname)

    return None


# ============================================================
# 5. FINE-TUNER
# ============================================================
def finetune_appliance(
    X, y,
    appliance_col,
    base_model_path,
    household_id,
    output_dir,
):
    """
    Fine-tunes or trains from scratch for one appliance.

    If base_model_path is provided:
        Warm-starts XGBoost from the base booster and adds
        FINETUNE_N_ESTIMATORS new trees trained on household data.
        Base trees are preserved — new trees refine predictions.

    If base_model_path is None:
        Trains a fresh XGBoost from scratch on household data only.
        Model will be weak initially but improves as data accumulates.

    Saves result to: output_dir/<appliance_col>.pkl
    """
    on_count  = int(y.sum())
    off_count = int((y == 0).sum())
    on_pct    = on_count / max(1, len(y)) * 100

    print(f"\n  Appliance: {appliance_col}")
    print(f"    ON: {on_count:,} ({on_pct:.1f}%)  OFF: {off_count:,}")

    if on_count == 0:
        print(f"    [SKIPPED] No ON events — cannot train")
        return None

    scale_pos_weight = min(off_count / on_count, SCALE_POS_WEIGHT_CAP)
    print(f"    scale_pos_weight: {scale_pos_weight:.2f}")

    if base_model_path is not None:
        # ── WARM START: load base booster, add household-specific trees ──────
        print(f"    Base model: {os.path.basename(base_model_path)}")
        with open(base_model_path, 'rb') as f:
            base_model = pickle.load(f)

        # Multiple passes: each pass appends FINETUNE_N_ESTIMATORS trees
        # on top of the previous booster, strengthening household adaptation
        # while keeping base knowledge intact
        booster = base_model.get_booster()
        for pass_num in range(FINETUNE_PASSES):
            ft = XGBClassifier(
                n_estimators=FINETUNE_N_ESTIMATORS,
                learning_rate=FINETUNE_LEARNING_RATE,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=42,
            )
            ft.fit(X, y, xgb_model=booster)
            booster = ft.get_booster()
        finetuned = ft
        mode = f"warm-start fine-tune ({FINETUNE_PASSES} passes x {FINETUNE_N_ESTIMATORS} trees)"

    else:
        # ── SCRATCH: no base model exists for this appliance ─────────────────
        print(f"    Base model: none — training from scratch")
        finetuned = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
        )
        finetuned.fit(X, y)
        mode = "scratch"

    # ── Evaluate on the same household data ───────────────────────────────────
    # Note: train == test here because household data is small.
    # These metrics reflect how well the model fits this household's patterns,
    # not generalisation. They will improve as more data accumulates.
    y_pred  = finetuned.predict(X)
    y_proba = finetuned.predict_proba(X)[:, 1]

    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = float('nan')

    print(f"    Mode: {mode}")
    print(f"    Household F1: {f1:.4f}  AUC: {auc:.4f}")
    print(f"    (train == test — metrics show household fit, not generalisation)")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{appliance_col}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(finetuned, f)
    print(f"    Saved → {save_path}")

    return {
        'appliance':    appliance_col,
        'mode':         mode,
        'on_hours':     on_count,
        'household_f1': round(f1, 4),
        'household_auc': round(auc, 4),
        'model_path':   save_path,
    }


# ============================================================
# 6. MAIN PIPELINE
# ============================================================
def run_finetuning(
    household_id,
    usage_csv_path,
    lat,
    lon,
    tz,
    base_models_dir=BASE_MODELS_DIR,
    households_dir=HOUSEHOLDS_DIR,
    prefetched_weather=None,
):
    """
    Full fine-tuning pipeline for one household.

    Parameters
    ----------
    household_id   : unique string identifier for this household
                     (used for the output folder name)
    usage_csv_path : path to the disaggregated usage CSV from the
                     Raspberry Pi — see load_household_data() for format
    lat, lon       : household location for weather fetching
    tz             : timezone string e.g. "Asia/Beirut"
    """
    print(f"\n{'='*55}")
    print(f"FINE-TUNING HOUSEHOLD: {household_id}")
    print(f"{'='*55}")

    # Output folder for this household's models
    output_dir = os.path.join(households_dir, household_id)

    # ── Load and enrich household data ────────────────────────────────────────
    df = load_household_data(usage_csv_path, lat, lon, tz,
                               prefetched_weather=prefetched_weather)

    # ── Detect appliance columns ──────────────────────────────────────────────
    appliance_cols = detect_appliance_columns(df)
    print(f"\n  Detected appliances: {appliance_cols}")

    # ── Fine-tune one model per appliance ─────────────────────────────────────
    results = []

    for col in appliance_cols:
        y = df[col].astype(int)

        # Look for a matching base model
        base_path = find_base_model(col, base_models_dir)

        # Load base model to get its feature names for get_features_for
        base_model_tmp = None
        if base_path:
            with open(base_path, 'rb') as _f:
                base_model_tmp = pickle.load(_f)

        # Use the right feature set for this appliance
        X = get_features_for(col, df, base_model_tmp)

        result = finetune_appliance(
            X=X,
            y=y,
            appliance_col=col,
            base_model_path=base_path,
            household_id=household_id,
            output_dir=output_dir,
        )
        if result:
            results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"SUMMARY — {household_id}")
    print(f"{'='*55}")
    for r in results:
        base_tag = "base+HH" if r['mode'] != 'scratch' else "scratch"
        print(f"  {r['appliance']:<30} [{base_tag}]  "
              f"F1={r['household_f1']:.3f}  AUC={r['household_auc']:.3f}  "
              f"({r['on_hours']} ON hours)")

    print(f"\nModels saved to: {output_dir}")
    return results


# ============================================================
# 7. EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # This example creates a synthetic household CSV and runs the pipeline.
    # Replace with your actual Raspberry Pi data path.

    import tempfile

    # Create a synthetic 4-week household CSV
    np.random.seed(42)
    n = 24 * 28  # 28 days

    timestamps = pd.date_range("2024-06-01 00:00", periods=n, freq="h")
    hours      = timestamps.hour

    # Simulate realistic usage patterns
    cooling_on = ((hours >= 13) & (hours <= 21)).astype(int)
    washer_on  = np.zeros(n, dtype=int)
    washer_on[np.random.choice(np.where((hours >= 9) & (hours <= 11))[0],
                               size=8, replace=False)] = 1

    df_fake = pd.DataFrame({
        'timestamp':             timestamps,
        'elec_cooling_on':       cooling_on,
        'elec_clothes_washer_on': washer_on,
        'elec_pool_pump_on':     ((hours >= 8) & (hours <= 17)).astype(int),
    })

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        df_fake.to_csv(f, index=False)
        tmp_path = f.name

    print(f"Synthetic CSV written to: {tmp_path}")

    run_finetuning(
        household_id   = "household_001",
        usage_csv_path = tmp_path,
        lat            = 33.89,
        lon            = 35.50,
        tz             = "Asia/Beirut",
    )