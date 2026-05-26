"""
fault_detection.py
Runs solar and battery fault detection using pre-trained autoencoders.
Models must be placed at the paths defined in BASE_SOLAR_MODEL / BASE_BATTERY_MODEL.
"""
import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample

# ── Model paths — place your downloaded .pt files here ────────────────────────
BASE_DIR           = os.path.join(os.path.dirname(__file__), '..', 'base_models')
SOLAR_MODEL_PATH   = os.path.join(BASE_DIR, 'solar_autoencoder.pt')
BATTERY_MODEL_PATH = os.path.join(BASE_DIR, 'battery_autoencoder.pt')
SOLAR_THRESHOLD_PATH   = os.path.join(BASE_DIR, 'solar_threshold.pkl')
BATTERY_THRESHOLD_PATH = os.path.join(BASE_DIR, 'battery_threshold.pkl')

SOLAR_WINDOW   = 30   # minutes per window (matches notebook)
BATTERY_CYCLE_LEN = 128

device = 'cpu'  # Cloud Run CPU inference is fine for these small models


# ── Solar autoencoder (Conv1d) ─────────────────────────────────────────────────
class SolarAutoencoder(nn.Module):
    def __init__(self, window_size=30, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=5, padding=2), nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512),        nn.ReLU(),
            nn.Unflatten(1, (64, 8)),
            nn.Upsample(size=window_size, mode='linear', align_corners=False),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 1,  kernel_size=5, padding=2), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Battery LSTM autoencoder ───────────────────────────────────────────────────
class BatteryLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.encoder_fc   = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc   = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.output_fc    = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        return self.encoder_fc(h_n[-1])

    def decode(self, z, seq_len):
        h = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(h)
        return torch.sigmoid(self.output_fc(out))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z, x.shape[1]), z


def _load_solar_model():
    m = SolarAutoencoder(window_size=SOLAR_WINDOW, latent_dim=32).to(device)
    if os.path.exists(SOLAR_MODEL_PATH):
        m.load_state_dict(torch.load(SOLAR_MODEL_PATH, map_location=device, weights_only=True))
    m.eval()
    threshold = 0.01  # fallback if no threshold file
    if os.path.exists(SOLAR_THRESHOLD_PATH):
        with open(SOLAR_THRESHOLD_PATH, 'rb') as f:
            t = pickle.load(f)
        threshold = float(t['threshold']) if isinstance(t, dict) else float(t)
    return m, threshold


def _load_battery_model():
    m = BatteryLSTMAutoencoder(input_dim=3, hidden_dim=64, latent_dim=32, num_layers=2).to(device)
    if os.path.exists(BATTERY_MODEL_PATH):
        m.load_state_dict(torch.load(BATTERY_MODEL_PATH, map_location=device, weights_only=True))
    m.eval()
    threshold = 0.01
    if os.path.exists(BATTERY_THRESHOLD_PATH):
        with open(BATTERY_THRESHOLD_PATH, 'rb') as f:
            t = pickle.load(f)
        threshold = float(t['threshold']) if isinstance(t, dict) else float(t)
    return m, threshold


# ── Solar fault detection ─────────────────────────────────────────────────────
def analyze_solar(df: pd.DataFrame) -> dict:
    """
    Input: solar CSV dataframe with columns:
        timestamp, pv_input_voltage, pv_input_current_for_battery,
        pv_input_power, battery_voltage_from_scc, is_scc_charging_on
    Returns dict with anomaly flags, scores, and human-readable messages.
    """
    result = {'status': 'ok', 'anomalies': [], 'score': 0.0, 'details': [], 'timeline': []}

    try:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').dropna(subset=['pv_input_power'])

        if len(df) < SOLAR_WINDOW:
            result['status'] = 'insufficient_data'
            result['details'].append(f'Need at least {SOLAR_WINDOW} readings. Got {len(df)}.')
            return result

        # Normalize power to 0-1 range for the autoencoder
        p = df['pv_input_power'].values.astype(np.float32)
        p_max = p.max()
        if p_max < 1e-6:
            result['anomalies'].append('No Solar Output')
            result['details'].append('Solar panel is producing no power. Check panel connections.')
            result['status'] = 'fault'
            return result
        p_norm = p / p_max

        # Slide windows
        windows = []
        window_times = []
        step = max(1, SOLAR_WINDOW // 3)
        for i in range(0, len(p_norm) - SOLAR_WINDOW + 1, step):
            windows.append(p_norm[i:i+SOLAR_WINDOW])
            window_times.append(df['timestamp'].iloc[i])
        if not windows:
            result['status'] = 'insufficient_data'
            return result

        model, threshold = _load_solar_model()
        X = torch.tensor(np.array(windows)).unsqueeze(1)
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                batch = X[i:i+64].to(device)
                recon, _ = model(batch)
                mse = ((recon - batch)**2).mean(dim=[1,2])
                errors.extend(mse.cpu().numpy())
        errors = np.array(errors)

        # Per-window anomaly flags
        anomaly_mask = errors > threshold
        anomaly_rate = float(anomaly_mask.mean())
        result['score'] = round(float(errors.mean()), 6)

        # Timeline: list of {time, error, is_anomaly}
        result['timeline'] = [
            {'time': str(t)[:16], 'error': round(float(e), 6), 'anomaly': bool(a)}
            for t, e, a in zip(window_times, errors, anomaly_mask)
        ]

        # Rule-based checks on top of autoencoder
        rule_faults = []

        # Check for clipping (power hits ceiling too often)
        if (p_norm > 0.97).mean() > 0.15:
            rule_faults.append(('Clipping Detected',
                'Solar output is hitting the panel maximum too frequently. '
                'The inverter may be undersized for your panel capacity.'))

        # Check for shading (sudden drops during the day)
        daytime = df[(df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour <= 17)]
        if len(daytime) > 10:
            drops = np.diff(daytime['pv_input_power'].values)
            if (drops < -p_max * 0.3).any():
                rule_faults.append(('Intermittent Shading',
                    'Sudden large drops in solar output detected during daylight hours. '
                    'Check for shadows cast by nearby objects or panel soiling.'))

        # Check for soiling (low output all day despite daylight)
        noon_power = df[(df['timestamp'].dt.hour.between(11, 13))]['pv_input_power']
        if len(noon_power) > 0 and noon_power.mean() < p_max * 0.3:
            rule_faults.append(('Possible Soiling / Dust',
                'Solar output at midday is unusually low (below 30% of peak). '
                'Clean the panels and check for obstructions.'))

        if anomaly_rate > 0.3 or rule_faults:
            result['status'] = 'fault'
            if anomaly_rate > 0.3:
                result['anomalies'].append('Irregular Output Pattern')
                result['details'].append(
                    f'{anomaly_rate*100:.0f}% of time windows show abnormal power curves. '
                    'This may indicate panel degradation or inverter issues.')
            for name, msg in rule_faults:
                result['anomalies'].append(name)
                result['details'].append(msg)
        else:
            result['status'] = 'ok'
            result['details'].append('Solar panels are operating normally.')

    except Exception as e:
        result['status'] = 'error'
        result['details'].append(f'Analysis error: {str(e)}')

    return result


# ── Battery fault detection ────────────────────────────────────────────────────
def analyze_battery(df: pd.DataFrame) -> dict:
    """
    Input: battery CSV with columns:
        timestamp, battery_voltage, battery_charging_current,
        battery_discharge_current, battery_capacity, is_charging_on
    Returns dict with anomaly flags and messages.
    """
    result = {'status': 'ok', 'anomalies': [], 'score': 0.0, 'details': [], 'timeline': []}

    try:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').dropna(subset=['battery_voltage', 'battery_charging_current'])

        if len(df) < 20:
            result['status'] = 'insufficient_data'
            result['details'].append('Need at least 20 battery readings.')
            return result

        # Extract charge cycles
        # A cycle starts when is_charging_on goes from 0 → 1
        charge_cycles = []
        in_cycle = False
        cycle_rows = []
        for _, row in df.iterrows():
            charging = row.get('is_charging_on', 0)
            if charging == 1:
                in_cycle = True
                cycle_rows.append(row)
            elif in_cycle and charging == 0:
                if len(cycle_rows) >= 20:
                    charge_cycles.append(pd.DataFrame(cycle_rows))
                cycle_rows = []
                in_cycle = False
        if in_cycle and len(cycle_rows) >= 20:
            charge_cycles.append(pd.DataFrame(cycle_rows))

        if not charge_cycles:
            # No complete cycles — run rule-based only
            result['details'].append('No complete charging cycles found — running rule-based checks only.')
            _battery_rule_checks(df, result)
            return result

        # Normalize each cycle and run autoencoder
        model, threshold = _load_battery_model()
        cycle_tensors = []
        for cyc in charge_cycles:
            arr = _normalize_battery_cycle(cyc)
            if arr is not None:
                cycle_tensors.append(arr)

        if cycle_tensors:
            X = torch.tensor(np.array(cycle_tensors).transpose(0, 2, 1), dtype=torch.float32)
            errors = []
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    batch = X[i:i+32].to(device)
                    recon, _ = model(batch)
                    mse = ((recon - batch)**2).mean(dim=[1,2])
                    errors.extend(mse.cpu().numpy())
            errors = np.array(errors)
            result['score'] = round(float(errors.mean()), 6)
            anomaly_rate = float((errors > threshold).mean())

            result['timeline'] = [
                {'cycle': i+1, 'error': round(float(e), 6), 'anomaly': bool(e > threshold)}
                for i, e in enumerate(errors)
            ]

            if anomaly_rate > 0.3:
                result['anomalies'].append('Abnormal Charge Profile')
                result['details'].append(
                    f'{anomaly_rate*100:.0f}% of charging cycles show abnormal patterns. '
                    'Battery may be degraded or have internal resistance issues.')

        _battery_rule_checks(df, result)

        if result['anomalies']:
            result['status'] = 'fault'
        else:
            result['details'].append('Battery is operating normally.')

    except Exception as e:
        result['status'] = 'error'
        result['details'].append(f'Analysis error: {str(e)}')

    return result


def _normalize_battery_cycle(cyc_df):
    try:
        v = cyc_df['battery_voltage'].values.astype(np.float32)
        c = cyc_df['battery_charging_current'].values.astype(np.float32)
        t = np.zeros(len(v), dtype=np.float32)  # temperature not always available
        if len(v) < 20:
            return None
        v = resample(v, BATTERY_CYCLE_LEN).astype(np.float32)
        c = resample(c, BATTERY_CYCLE_LEN).astype(np.float32)
        t = resample(t, BATTERY_CYCLE_LEN).astype(np.float32)
        v_r = v.max() - v.min()
        if v_r < 1e-6: return None
        v = (v - v.min()) / v_r
        c = np.clip(c, 0, None)
        c_m = c.max()
        if c_m < 1e-6: return None
        c = c / c_m
        return np.stack([v, c, t], axis=0)
    except:
        return None


def _battery_rule_checks(df, result):
    v = df['battery_voltage'].dropna()
    c = df['battery_charging_current'].dropna()
    d = df['battery_discharge_current'].dropna()
    cap = df['battery_capacity'].dropna()

    if len(v) > 0:
        v_max = v.max()
        v_min = v.min()
        # Overvoltage: typical LiFePO4 max ~14.4V (12V system), lead-acid ~14.7V
        if v_max > 15.5:
            result['anomalies'].append('Overvoltage')
            result['details'].append(
                f'Battery voltage reached {v_max:.1f}V, which is above safe limits. '
                'Check charge controller settings to prevent battery damage.')
        # Undervoltage: below ~11.5V for a 12V lead-acid system
        if v_min < 11.0:
            result['anomalies'].append('Deep Discharge')
            result['details'].append(
                f'Battery voltage dropped to {v_min:.1f}V. Deep discharges shorten battery life. '
                'Consider raising your minimum SoC limit in the app settings.')

    if len(d) > 0 and d.max() > 50:
        result['anomalies'].append('High Discharge Current')
        result['details'].append(
            f'Discharge current spiked to {d.max():.1f}A. '
            'This may indicate a sudden large load or a short circuit.')

    if len(cap) > 1:
        cap_trend = np.polyfit(np.arange(len(cap)), cap.values, 1)[0]
        if cap_trend < -0.5:
            result['anomalies'].append('Capacity Fade')
            result['details'].append(
                'Battery capacity has been declining over the uploaded period. '
                'This is normal aging but accelerating decline may indicate a problem.')
