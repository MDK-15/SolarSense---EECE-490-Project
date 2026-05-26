"""
nilm.py

Appliance commissioning and detection using physical power signatures.
Based on CommissioningManager approach: extracts peak power, steady-state,
settle time, and variance from a user-tagged calibration session.

Flow:
1. User adds an appliance and selects its category
2. User clicks "I'm turning it ON" → server records on_time
3. User clicks "I'm turning it OFF" → server records off_time
4. When user uploads their load CSV, calibrate_from_upload() finds that
   time window, extracts the full power readings, and runs calibrate_appliance()
5. The resulting ApplianceProfile is stored as JSON in the DB
6. detect_events() scans future uploads for matching profiles
7. build_hourly_binary() converts events to XGBoost-ready hourly binary columns
"""

import numpy as np
import pandas as pd
from enum import Enum, auto
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ── Appliance categories ───────────────────────────────────────────────────────
class ApplianceType(Enum):
    FRIDGE         = auto()
    FREEZER        = auto()
    TELEVISION     = auto()
    WASHING_MACHINE = auto()
    HEATER         = auto()
    COOLING        = auto()
    HOT_WATER      = auto()
    OTHER          = auto()


# Maps each category to the XGBoost base model filename
APPLIANCE_MODEL_MAP = {
    ApplianceType.FRIDGE:          None,                        # always-on, no model
    ApplianceType.FREEZER:         None,                        # always-on, no model
    ApplianceType.TELEVISION:      'elec_television_on.pkl',
    ApplianceType.WASHING_MACHINE: 'elec_clothes_washer_on.pkl',
    ApplianceType.HEATER:          'elec_heating_on.pkl',
    ApplianceType.COOLING:         'elec_cooling_on.pkl',
    ApplianceType.HOT_WATER:       'elec_hot_water_on.pkl',
    ApplianceType.OTHER:           None,                        # new model trained from scratch
}

# Human-readable labels shown in the UI
APPLIANCE_TYPE_LABELS = {
    'fridge':          'Fridge',
    'freezer':         'Freezer',
    'television':      'Television',
    'washing_machine': 'Washing Machine',
    'heater':          'Heater',
    'cooling':         'Cooling (AC)',
    'hot_water':       'Hot Water',
    'other':           'Other',
}

# Internal column name used in fine-tuning CSV
APPLIANCE_COLUMN_MAP = {
    'fridge':          'elec_refrigerator_on',
    'freezer':         'elec_freezer_on',
    'television':      'elec_television_on',
    'washing_machine': 'elec_clothes_washer_on',
    'heater':          'elec_heating_on',
    'cooling':         'elec_cooling_on',
    'hot_water':       'elec_hot_water_on',
}


@dataclass
class ApplianceProfile:
    """
    Physical fingerprint of an appliance extracted during commissioning.
    All power values are the appliance's NET draw (baseline already subtracted).
    """
    appliance_type:      str    # string key e.g. 'washing_machine'
    display_name:        str    # user-provided label e.g. "My Bosch Washer"
    peak_power_w:        float  # inrush peak (W)
    steady_state_power_w: float # normal operating draw (W)
    settle_time_seconds: int    # seconds from peak to steady state
    variance_w:          float  # variance of steady-state draw (W²)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> 'ApplianceProfile':
        return ApplianceProfile(**d)


class CommissioningManager:
    def __init__(self, sampling_rate_hz: int = 1):
        self.sampling_rate_hz = sampling_rate_hz

    def calibrate_appliance(self,
                             appliance_type: str,
                             display_name: str,
                             recorded_power: List[float],
                             baseline_power: float) -> ApplianceProfile:
        """
        Processes the raw power readings from the user's calibration session.

        recorded_power: list of aggregate watt readings from on_time to off_time
        baseline_power: the aggregate reading just before the user pressed ON
                        (background load of the house at that moment)

        Returns an ApplianceProfile with the extracted physical characteristics.
        """
        power_array = np.array(recorded_power, dtype=float)

        # 1. Isolate appliance's actual draw by subtracting house baseline
        isolated = power_array - baseline_power

        # Guard against degenerate inputs
        if len(isolated) == 0:
            raise ValueError("No power readings in calibration window.")
        if isolated.max() < 5:
            raise ValueError(
                f"No meaningful power step detected (max delta = {isolated.max():.1f}W). "
                "Make sure the appliance was running during the calibration window."
            )

        # 2. Peak power (inrush)
        peak_idx   = int(np.argmax(isolated))
        peak_power = float(isolated[peak_idx])

        # 3. Steady-state: mean of the final 20% of the session
        tail_length  = max(1, int(len(isolated) * 0.2))
        tail_data    = isolated[-tail_length:]
        steady_state = float(np.mean(tail_data))
        variance     = float(np.var(tail_data))

        # 4. Settle time: first index after peak where power ≤ steady_state * 1.05
        settle_idx = peak_idx
        threshold  = steady_state * 1.05
        for i in range(peak_idx, len(isolated)):
            if isolated[i] <= threshold:
                settle_idx = i
                break
        settle_time = int((settle_idx - peak_idx) / self.sampling_rate_hz)

        return ApplianceProfile(
            appliance_type=appliance_type,
            display_name=display_name,
            peak_power_w=round(peak_power, 2),
            steady_state_power_w=round(steady_state, 2),
            settle_time_seconds=settle_time,
            variance_w=round(variance, 2),
        )


# Singleton manager used by the Flask app
_manager = CommissioningManager(sampling_rate_hz=1)


# ── Calibration from upload ────────────────────────────────────────────────────
def calibrate_from_upload(load_df: pd.DataFrame,
                           on_time,
                           off_time,
                           appliance_type: str,
                           display_name: str,
                           tolerance_seconds: int = 120) -> Optional[ApplianceProfile]:
    """
    Given the user's aggregate load CSV and the on/off timestamps they tapped,
    extract the calibration window and run CommissioningManager.calibrate_appliance().

    Returns an ApplianceProfile, or None if extraction fails.
    """
    try:
        df = load_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        col = _find_power_column(df)
        if col is None:
            return None

        on_time  = pd.Timestamp(on_time)
        off_time = pd.Timestamp(off_time)
        tol      = pd.Timedelta(seconds=tolerance_seconds)

        # Baseline: median of 5 minutes before the user pressed ON
        mask_pre = (df['timestamp'] >= on_time - pd.Timedelta(minutes=5)) & \
                   (df['timestamp'] < on_time)
        baseline = float(df.loc[mask_pre, col].median()) if mask_pre.sum() > 0 else 0.0
        if np.isnan(baseline):
            baseline = float(df[col].median())

        # Calibration window
        mask_on = (df['timestamp'] >= on_time - tol) & \
                  (df['timestamp'] <= off_time + tol)
        df_win  = df.loc[mask_on, col]

        if len(df_win) < 3:
            return None

        return _manager.calibrate_appliance(
            appliance_type=appliance_type,
            display_name=display_name,
            recorded_power=df_win.tolist(),
            baseline_power=baseline,
        )

    except Exception as e:
        print(f"[NILM] calibrate_from_upload error: {e}")
        return None


# ── Event detection ────────────────────────────────────────────────────────────
def detect_events(load_df: pd.DataFrame,
                   profiles: Dict[str, ApplianceProfile],
                   tolerance_pct: float = 0.25) -> pd.DataFrame:
    """
    Scans the aggregate load for ON/OFF events matching stored appliance profiles.

    Matching uses steady_state_power_w as the reference step size, with tolerance_pct
    determining how close the measured step needs to be (default ±25%).

    Returns DataFrame: timestamp | appliance_name | event (ON/OFF) | confidence | power_w
    """
    results = []
    try:
        df = load_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        col = _find_power_column(df)
        if col is None:
            return pd.DataFrame(results)

        # Resample to 1-minute intervals to smooth noise
        df_1min = (df.set_index('timestamp')[[col]]
                     .resample('1min').mean()
                     .interpolate()
                     .reset_index())
        power = df_1min[col].values
        steps = np.diff(power)

        for i, step in enumerate(steps):
            for app_name, profile in profiles.items():
                if profile is None:
                    continue

                # Use steady-state as the expected step size for ON events
                target = profile.steady_state_power_w
                lo     = target * (1 - tolerance_pct)
                hi     = target * (1 + tolerance_pct)

                ts = df_1min['timestamp'].iloc[i + 1]

                if lo < step < hi:
                    # ON event — positive step matching steady-state draw
                    confidence = 1.0 - abs(step - target) / target
                    results.append({
                        'timestamp':  ts,
                        'appliance':  app_name,
                        'event':      'ON',
                        'confidence': round(float(confidence), 2),
                        'power_w':    round(float(step), 1),
                    })

                elif -hi < step < -lo:
                    # OFF event — negative step
                    confidence = 1.0 - abs(abs(step) - target) / target
                    results.append({
                        'timestamp':  ts,
                        'appliance':  app_name,
                        'event':      'OFF',
                        'confidence': round(float(confidence), 2),
                        'power_w':    round(float(abs(step)), 1),
                    })

    except Exception as e:
        print(f"[NILM] detect_events error: {e}")

    return pd.DataFrame(results)


# ── Hourly binary conversion ───────────────────────────────────────────────────
def build_hourly_binary(load_df: pd.DataFrame,
                         profiles: Dict[str, ApplianceProfile],
                         tolerance_pct: float = 0.25) -> pd.DataFrame:
    """
    Converts detected ON/OFF events into hourly binary columns ready for
    XGBoost fine-tuning.

    Column names use APPLIANCE_COLUMN_MAP for known categories,
    or 'elec_<app_name>_on' for OTHER appliances.

    Returns DataFrame: timestamp (hourly) | elec_<appliance>_on ...
    """
    df = load_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    hours = pd.date_range(
        df['timestamp'].min().floor('h'),
        df['timestamp'].max().ceil('h'),
        freq='h'
    )
    result = pd.DataFrame({'timestamp': hours})

    if not profiles:
        return result

    events = detect_events(load_df, profiles, tolerance_pct)
    if events.empty:
        for name, profile in profiles.items():
            col = _get_column_name(name, profile)
            result[col] = 0
        return result

    for app_name, profile in profiles.items():
        col        = _get_column_name(app_name, profile)
        app_events = events[events['appliance'] == app_name].sort_values('timestamp')

        # Track state transitions to fill hourly ON/OFF
        hourly_state = {}
        state = 0
        for _, ev in app_events.iterrows():
            hour = ev['timestamp'].floor('h')
            state = 1 if ev['event'] == 'ON' else 0
            hourly_state[hour] = state

        result[col] = result['timestamp'].map(hourly_state).fillna(0).astype(int)

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────
def _find_power_column(df: pd.DataFrame) -> Optional[str]:
    """Find the best aggregate power column in the dataframe."""
    preferred = [
        'ac_output_active_power',
        'ac_output_apparent_power',
        'pv_input_power',
        'active_power',
        'power',
    ]
    for c in preferred:
        if c in df.columns and df[c].notna().sum() > 0:
            return c
    # Fallback: any numeric column with 'power' in the name
    for c in df.select_dtypes(include='number').columns:
        if 'power' in c.lower() and df[c].median() > 1:
            return c
    return None


def _get_column_name(app_name: str, profile: ApplianceProfile) -> str:
    """
    Returns the XGBoost training column name for this appliance.
    Known categories use the standard naming convention.
    OTHER appliances get a custom column name based on the user's label.
    """
    known = APPLIANCE_COLUMN_MAP.get(profile.appliance_type)
    if known:
        return known
    # OTHER: sanitise the display name into a valid column name
    safe = profile.display_name.lower().replace(' ', '_').replace('-', '_')
    safe = ''.join(c for c in safe if c.isalnum() or c == '_')
    return f'elec_{safe}_on'


def get_base_model_filename(appliance_type: str) -> Optional[str]:
    """
    Returns the base model .pkl filename for a given appliance type string,
    or None if the appliance uses a scratch model (OTHER) or is always-on.
    """
    try:
        enum_val = ApplianceType[appliance_type.upper()]
        return APPLIANCE_MODEL_MAP.get(enum_val)
    except KeyError:
        return None