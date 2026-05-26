"""
scheduler.py
Runs XGBoost prediction + LP optimization for next-week schedule recommendation.
"""
import os, sys, pickle, json, tempfile
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# Add parent to path so we can import finetune_household and solar_lp_optimizer
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'base_models')

FEATURE_COLS = [
    'weather_drybulb_temp_c',
    'weather_relative_humidity_pct',
    'hour',
    'day_of_week',
    'is_weekend',
    'month',
]

TV_FEATURE_COLS = ['hour', 'day_of_week', 'is_weekend', 'is_evening']

APPLIANCE_LABELS = {
    'elec_cooling_on':         'Cooling (AC)',
    'elec_clothes_washer_on':  'Washing Machine',
    'elec_hot_water_on':       'Hot Water',
    'elec_television_on':      'Television',
    'elec_heating_on':         'Heating',
}


def get_features_for(appliance: str, df: pd.DataFrame, model=None) -> pd.DataFrame:
    if model is not None and hasattr(model, 'feature_names_in_'):
        cols = [str(c) for c in model.feature_names_in_]
        available = [c for c in cols if c in df.columns]
        if len(available) == len(cols):
            return df[cols]
    if 'television' in appliance.lower():
        available = [c for c in TV_FEATURE_COLS if c in df.columns]
        return df[available]
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available]


def fetch_weather_week(lat: float, lon: float, tz: str, start_date: str) -> pd.DataFrame:
    """Fetch 7 days of weather starting from start_date (forecast if future, archive if past)."""
    end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=6)).strftime('%Y-%m-%d')

    # Try archive first (for past/current week), then forecast
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date, 'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m',
        'timezone': tz,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame({
            'timestamp':                     pd.to_datetime(data['hourly']['time']),
            'weather_drybulb_temp_c':        data['hourly']['temperature_2m'],
            'weather_relative_humidity_pct': data['hourly']['relative_humidity_2m'],
        })
    except Exception:
        # Fallback: forecast API
        try:
            url2 = 'https://api.open-meteo.com/v1/forecast'
            p2 = {
                'latitude': lat, 'longitude': lon,
                'hourly': 'temperature_2m,relative_humidity_2m',
                'forecast_days': 7, 'timezone': tz,
            }
            r2 = requests.get(url2, params=p2, timeout=15)
            r2.raise_for_status()
            data = r2.json()
            df = pd.DataFrame({
                'timestamp':                     pd.to_datetime(data['hourly']['time']),
                'weather_drybulb_temp_c':        data['hourly']['temperature_2m'],
                'weather_relative_humidity_pct': data['hourly']['relative_humidity_2m'],
            })
        except Exception as e:
            raise ConnectionError(f'Could not fetch weather: {e}')

    df['hour']       = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month']      = df['timestamp'].dt.month
    df['is_evening'] = df['hour'].between(18, 23).astype(int)
    return df.head(168).reset_index(drop=True)


def load_user_models(user_models_dir: str) -> dict:
    """Load fine-tuned models from user's directory, fall back to base models."""
    models = {}
    for app in APPLIANCE_LABELS:
        # Try user fine-tuned first
        user_path = os.path.join(user_models_dir, f'{app}.pkl')
        base_path = os.path.join(BASE_MODELS_DIR, f'{app}.pkl')
        for path in [user_path, base_path]:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[app] = pickle.load(f)
                break
    return models


def predict_week(df_weather: pd.DataFrame, models: dict,
                  always_on: list = None) -> dict:
    """Run XGBoost models to predict next week's schedule."""
    always_on = always_on or ['fridge', 'freezer']
    schedule = {}
    for app, model in models.items():
        X = get_features_for(app, df_weather, model)
        preds = model.predict(X).astype(int)
        schedule[app] = preds.tolist()
    for app in always_on:
        schedule[app] = [1] * 168
    return schedule


def run_lp_optimization(schedule: dict, appliance_power: dict,
                          lat: float, lon: float, tz: str,
                          battery_capacity: float, battery_start_soc: float,
                          battery_min_soc: float,
                          always_on: set = None, fixed: set = None,
                          semi_fixed: set = None,
                          acceptable_hours: dict = None,
                          date: str = None) -> dict:
    """Wrapper around solar_lp_optimizer.optimize_schedule."""
    try:
        from solar_lp_optimizer import optimize_schedule, compute_solar_generation
    except ImportError:
        return {'feasible': False, 'message': 'LP optimizer not found.'}

    if date is None:
        date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Get panel specs from appliance_power meta if present
    system_kw = appliance_power.pop('__system_kw__', 5.0)
    tilt      = appliance_power.pop('__tilt__',      30)
    azimuth   = appliance_power.pop('__azimuth__',   180)

    solar = compute_solar_generation(date, lat, lon, tz, system_kw, tilt, azimuth)

    result = optimize_schedule(
        original_schedule=schedule,
        appliance_power=appliance_power,
        solar_kwh=solar,
        battery_capacity=battery_capacity,
        battery_start_soc=battery_start_soc,
        battery_min_soc=battery_min_soc,
        always_on=always_on,
        fixed=fixed,
        semi_fixed=semi_fixed,
        acceptable_hours=acceptable_hours,
    )

    # Attach solar generation for display
    result['solar_kwh'] = solar.tolist()
    return result


def _rule_based_predictions(df_weather: pd.DataFrame) -> dict:
    """
    Simple rule-based schedule when no trained models are available.
    Cooling: on when temp > 28 and hour 12-23.
    Hot water: on 06-08 and 20-21.
    Washing machine: on Mon and Thu 09-11.
    Television: on 19-23.
    Heating: off (summer).
    """
    n = len(df_weather)
    hours   = df_weather['hour'].values
    dow     = df_weather['day_of_week'].values
    temps   = df_weather['weather_drybulb_temp_c'].values

    cooling = ((hours >= 12) & (hours < 23) & (temps > 27)).astype(int)
    hot_water = (((hours >= 6) & (hours < 8)) | ((hours >= 20) & (hours < 21))).astype(int)
    washer  = (((dow == 0) | (dow == 3)) & (hours >= 9) & (hours < 11)).astype(int)
    tv      = ((hours >= 19) & (hours < 23)).astype(int)
    heating = np.zeros(n, dtype=int)

    return {
        'elec_cooling_on':        cooling.tolist(),
        'elec_hot_water_on':      hot_water.tolist(),
        'elec_clothes_washer_on': washer.tolist(),
        'elec_television_on':     tv.tolist(),
        'elec_heating_on':        heating.tolist(),
    }


def _synthetic_weather_week(start_date: str, tz: str = None) -> pd.DataFrame:
    """
    Build a neutral 168-hour weather frame when Open-Meteo is unreachable.
    Temperatures/humidity are placeholders chosen so the rule-based
    predictions in _rule_based_predictions still produce sensible output
    (e.g. cooling triggers above 27°C — we leave it just under so the
    rules fall back to time-of-day patterns rather than always-on cooling).
    """
    start = pd.Timestamp(start_date)
    timestamps = pd.date_range(start, periods=168, freq='h')
    df = pd.DataFrame({
        'timestamp':                     timestamps,
        'weather_drybulb_temp_c':        25.0,
        'weather_relative_humidity_pct': 60.0,
    })
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['month']       = df['timestamp'].dt.month
    df['is_evening']  = df['hour'].between(18, 23).astype(int)
    return df


def run_pipeline(user_id: str, user_models_dir: str,
                  lat: float, lon: float, tz: str,
                  appliance_config: dict,
                  battery_capacity: float, battery_start_soc: float,
                  battery_min_soc: float) -> dict:
    """
    Full pipeline:
    1. Fetch next week's weather
    2. Predict schedule using base models (keyed by elec_*_on)
    3. Map predictions back to user's appliance names for the LP
    4. Run LP optimization using user's appliance names
    5. Return schedule + optimization result

    The schedule returned uses USER appliance names so the frontend
    can match predictions to the user's registered appliances.
    """
    try:
        next_monday = (datetime.now() + timedelta(
            days=(7 - datetime.now().weekday()) % 7 or 7
        )).strftime('%Y-%m-%d')
    except Exception:
        next_monday = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

    weather_fallback = False
    weather_error    = None
    try:
        df_weather = fetch_weather_week(lat, lon, tz, next_monday)
    except Exception as e:
        # Open-Meteo unreachable — fall back to synthetic weather + rule-based
        # predictions rather than failing the whole endpoint.
        weather_fallback = True
        weather_error    = str(e)
        df_weather       = _synthetic_weather_week(next_monday, tz)

    # ── Load models (keyed by elec_*_on) ─────────────────────────────────────
    models = load_user_models(user_models_dir)
    # Note: if models is empty we fall back to rule-based predictions below

    # ── Predict using base model keys ─────────────────────────────────────────
    # Returns {elec_cooling_on: [...], elec_clothes_washer_on: [...], ...}
    # If no models available OR weather is synthetic (flat inputs would give
    # the XGBoost models garbage), use rule-based predictions.
    if models and not weather_fallback:
        base_predictions = predict_week(df_weather, models, always_on=[])
    else:
        base_predictions = _rule_based_predictions(df_weather)

    # ── Build a mapping: user appliance name → base model prediction ──────────
    # The NILM column map tells us which base model column each appliance type uses
    # Import APPLIANCE_COLUMN_MAP — try multiple paths since the working
    # directory varies depending on how Flask is started
    try:
        from ml.nilm import APPLIANCE_COLUMN_MAP
    except ImportError:
        try:
            from nilm import APPLIANCE_COLUMN_MAP
        except ImportError:
            # Hardcode the map as fallback — matches nilm.py definition exactly
            APPLIANCE_COLUMN_MAP = {
                'fridge':          'elec_refrigerator_on',
                'freezer':         'elec_freezer_on',
                'television':      'elec_television_on',
                'washing_machine': 'elec_clothes_washer_on',
                'heater':          'elec_heating_on',
                'cooling':         'elec_cooling_on',
                'hot_water':       'elec_hot_water_on',
            }

    # Invert: 'elec_cooling_on' -> ['living_room_ac', 'bedroom_ac', ...]
    col_to_users: dict = {}
    for user_name, cfg in appliance_config.items():
        if user_name.startswith('__'):
            continue
        atype   = cfg.get('appliance_type', 'other')
        col     = APPLIANCE_COLUMN_MAP.get(atype)  # e.g. 'elec_cooling_on'
        if col and col in base_predictions:
            col_to_users.setdefault(col, []).append(user_name)

    # Build user-keyed schedule for the LP
    # Multiple appliances of the same type share the same base prediction
    lp_schedule: dict = {}
    for col, user_names in col_to_users.items():
        pred = base_predictions[col]
        for uname in user_names:
            lp_schedule[uname] = pred

    # An appliance is "always-on" if its flexibility tier is always_on OR its
    # category is one that runs continuously (fridge/freezer). Such appliances
    # never need signature calibration — they run every hour.
    ALWAYS_ON_TYPES = {'fridge', 'freezer'}

    def _is_always_on(cfg: dict) -> bool:
        return (cfg.get('tier') == 'always_on'
                or (cfg.get('appliance_type') or '').lower() in ALWAYS_ON_TYPES)

    # Always-on appliances: run every hour
    for uname, cfg in appliance_config.items():
        if uname.startswith('__'):
            continue
        if _is_always_on(cfg):
            lp_schedule[uname] = [1] * 168

    # Appliances with no base model (type='other'): assume off
    for uname in appliance_config:
        if uname.startswith('__'):
            continue
        if uname not in lp_schedule:
            lp_schedule[uname] = [0] * 168

    # ── LP inputs ─────────────────────────────────────────────────────────────
    # Only include appliances with a known (>0) power draw
    appliance_power = {}
    for uname, cfg in appliance_config.items():
        if uname.startswith('__'):
            continue
        pkw = float(cfg.get('power_kw', 0))
        if pkw > 0 and uname in lp_schedule:
            appliance_power[uname] = pkw

    if not appliance_power:
        return {
            'week_start':        next_monday,
            'base_schedule':     lp_schedule,
            'optimized':         {'feasible': False,
                                  'message': 'No calibrated appliances found. '
                                             'Upload your load.csv to complete calibration.'},
            'battery_min_soc_pct': int(battery_min_soc * 100),
            'weather': df_weather[['timestamp','weather_drybulb_temp_c',
                                   'weather_relative_humidity_pct']].to_dict('records'),
            'weather_fallback':  weather_fallback,
            'weather_error':     weather_error,
        }

    always_on_set  = {a for a, c in appliance_config.items()
                      if not a.startswith('__') and _is_always_on(c)}
    fixed_set      = {a for a, c in appliance_config.items()
                      if not a.startswith('__') and c.get('tier') == 'fixed'}
    semi_fixed_set = {a for a, c in appliance_config.items()
                      if not a.startswith('__') and c.get('tier') == 'semi_fixed'}
    hours_map      = {a: tuple(c['hours']) for a, c in appliance_config.items()
                      if not a.startswith('__') and 'hours' in c and len(c['hours']) == 2}

    # Only LP over appliances we actually have in the schedule
    lp_sched_filtered = {k: v for k, v in lp_schedule.items() if k in appliance_power}

    system_kw = appliance_config.get('__solar__', {}).get('system_kw', 5.0)
    tilt      = appliance_config.get('__solar__', {}).get('tilt', 30)
    azimuth   = appliance_config.get('__solar__', {}).get('azimuth', 180)
    appliance_power['__system_kw__'] = system_kw
    appliance_power['__tilt__']      = tilt
    appliance_power['__azimuth__']   = azimuth

    opt_result = run_lp_optimization(
        schedule=lp_sched_filtered,
        appliance_power=appliance_power,
        lat=lat, lon=lon, tz=tz,
        battery_capacity=battery_capacity,
        battery_start_soc=battery_start_soc,
        battery_min_soc=battery_min_soc,
        always_on=always_on_set or set(),
        fixed=fixed_set or set(),
        semi_fixed=semi_fixed_set or set(),
        acceptable_hours=hours_map or None,
        date=next_monday,
    )

    # Build display name map for the frontend
    display_names = {
        k: v.get('display', k.replace('_',' ').title())
        for k, v in appliance_config.items()
        if not k.startswith('__')
    }

    # Tell the frontend which appliance keys are always-on (fridge/freezer etc.)
    # so it can present them separately rather than as a 24-hour heatmap row,
    # regardless of how the user named them.
    always_on_names = sorted(always_on_set)

    return {
        'week_start':          next_monday,
        'base_schedule':       lp_schedule,
        'optimized':           opt_result,
        'battery_min_soc_pct': int(battery_min_soc * 100),
        'display_names':       display_names,
        'always_on':           always_on_names,
        'weather':             df_weather[['timestamp', 'weather_drybulb_temp_c',
                                           'weather_relative_humidity_pct']].to_dict('records'),
        'weather_fallback':    weather_fallback,
        'weather_error':       weather_error,
    }
