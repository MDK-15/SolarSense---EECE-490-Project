"""
solar_lp_optimizer.py

Linear Programming optimizer for a solar-only household battery system.

Changes from previous version:
    - Appliance run-hours can now be REDUCED by the LP if there is not
      enough solar/battery to satisfy the full predicted schedule.
      A comfort penalty discourages unnecessary reductions.
    - The LP still tries to run each appliance for its full predicted
      hours if energy allows; it only cuts hours as a last resort.

Goal:
    Re-arrange (and if necessary reduce) a generated daily appliance
    schedule so that:
      1. Solar self-consumption is maximised
      2. The battery never drops below the user's minimum SoC
      3. The battery is discouraged from sitting at 100% during peak solar
      4. Appliances only run during acceptable hours
      5. The result stays as close as possible to the original schedule
      6. Hours are only reduced when unavoidable (comfort penalty)

Requires: pulp, pvlib, pandas, numpy
"""

import numpy as np
import pandas as pd
import pulp
import pvlib
from pvlib.location import Location


# ============================================================
# 1. SOLAR GENERATION via PVLib
# ============================================================
def compute_solar_generation(date, lat, lon, tz,
                              system_kw, tilt, azimuth,
                              altitude=0):
    """
    Returns an array of 24 hourly solar generation values in kWh.
    Uses a clear-sky (Ineichen) model with 0.85 derate.
    """
    site = Location(lat, lon, tz=tz, altitude=altitude)
    times = pd.date_range(f"{date} 00:00", f"{date} 23:00",
                          freq="1h", tz=tz)
    clearsky = site.get_clearsky(times, model="ineichen")
    solpos   = site.get_solarposition(times)
    poa      = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"], ghi=clearsky["ghi"], dhi=clearsky["dhi"],
    )
    DERATE = 0.85
    generation_kw = system_kw * (poa["poa_global"].fillna(0).values / 1000.0) * DERATE
    return np.maximum(generation_kw, 0.0)


# ============================================================
# 2. DEFAULTS
# ============================================================
DEFAULT_ACCEPTABLE_HOURS = {
    "washing_machine": (7, 22),
    "dishwasher":      (7, 23),
    "water_heater":    (5, 23),
    "cooling":         (0, 24),
    "cooking":         (0, 24),
}
DEFAULT_ALWAYS_ON  = {"fridge", "freezer"}
DEFAULT_FIXED      = {"cooking"}
DEFAULT_SEMI_FIXED = {"cooling"}


def build_appliance_config(appliances, always_on=None, fixed=None,
                            semi_fixed=None, acceptable_hours=None):
    resolved_always_on  = set(always_on)  if always_on  is not None else set(DEFAULT_ALWAYS_ON)
    resolved_fixed      = set(fixed)      if fixed      is not None else set(DEFAULT_FIXED)
    resolved_semi_fixed = set(semi_fixed) if semi_fixed is not None else set(DEFAULT_SEMI_FIXED)
    resolved_hours = dict(DEFAULT_ACCEPTABLE_HOURS)
    if acceptable_hours is not None:
        resolved_hours.update(acceptable_hours)
    for a in appliances:
        if a not in resolved_hours:
            resolved_hours[a] = (0, 24)
    return resolved_always_on, resolved_fixed, resolved_semi_fixed, resolved_hours


# ============================================================
# 3. THE OPTIMIZER
# ============================================================
def optimize_schedule(
    original_schedule,
    appliance_power,
    solar_kwh,
    battery_capacity,
    battery_start_soc,
    battery_min_soc,
    battery_max_soc=1.0,
    charge_efficiency=0.95,
    always_on=None,
    fixed=None,
    semi_fixed=None,
    acceptable_hours=None,
):
    """
    Builds and solves the LP.

    Key change: appliance run-hours can now be REDUCED if the energy
    budget does not support the full predicted schedule. A comfort
    penalty (W_COMFORT) makes reductions expensive so the LP only
    cuts hours when the battery constraint truly requires it.

    Returns a dict with the optimized schedule, battery SoC,
    and — if any hours were cut — a summary of what was reduced.
    """
    HOURS      = range(24)
    appliances = list(original_schedule.keys())

    always_on_set, fixed_set, semi_fixed_set, hours_map = build_appliance_config(
        appliances=appliances,
        always_on=always_on, fixed=fixed,
        semi_fixed=semi_fixed, acceptable_hours=acceptable_hours,
    )

    always_on_list  = [a for a in appliances if a in always_on_set]
    fixed_list      = [a for a in appliances if a in fixed_set]
    semi_fixed_list = [a for a in appliances if a in semi_fixed_set]
    shiftable_list  = [a for a in appliances
                       if a not in always_on_set | fixed_set | semi_fixed_set]

    always_on        = always_on_list
    fixed            = fixed_list
    semi_fixed       = semi_fixed_list
    shiftable        = shiftable_list
    acceptable_hours = hours_map

    # Original predicted hours per appliance (upper bound, not hard equality)
    predicted_hours = {
        a: int(sum(original_schedule[a])) for a in shiftable + semi_fixed
    }

    # ── LP setup ──────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("solar_schedule", pulp.LpMinimize)

    # Binary on/off per appliance per hour
    x = {
        a: {t: pulp.LpVariable(f"x_{a}_{t}", cat="Binary") for t in HOURS}
        for a in shiftable + semi_fixed
    }

    # Battery SoC (kWh) at end of each hour
    soc = {t: pulp.LpVariable(f"soc_{t}", lowBound=0) for t in HOURS}

    # Clipped solar (battery full, energy wasted)
    waste = {t: pulp.LpVariable(f"waste_{t}", lowBound=0) for t in HOURS}

    # Hours actually used per appliance (0 ≤ used ≤ predicted)
    # This is just a convenience expression; the LP controls it via x.
    # We add an explicit slack variable to track hours cut.
    hours_cut = {
        a: pulp.LpVariable(f"cut_{a}", lowBound=0, upBound=predicted_hours[a])
        for a in shiftable + semi_fixed
    }

    # ── Hour consumption ──────────────────────────────────────────────────────
    def hour_consumption(t):
        load = sum(appliance_power[a] for a in always_on)
        load += sum(appliance_power[a] * original_schedule[a][t] for a in fixed)
        load += sum(appliance_power[a] * x[a][t] for a in shiftable + semi_fixed)
        return load

    # ── Constraints ───────────────────────────────────────────────────────────

    # 1. Run AT MOST the predicted hours (≤ instead of ==)
    #    hours_cut[a] tracks how many hours were removed
    for a in shiftable + semi_fixed:
        actual = pulp.lpSum(x[a][t] for t in HOURS)
        prob += actual + hours_cut[a] == predicted_hours[a]
        # hours_cut is already bounded [0, predicted_hours[a]]

    # 2. Only run within acceptable hours
    for a in shiftable + semi_fixed:
        start, end = acceptable_hours.get(a, (0, 24))
        for t in HOURS:
            if not (start <= t < end):
                prob += x[a][t] == 0

    # 3. Battery dynamics
    cap = battery_capacity
    for t in HOURS:
        prev = battery_start_soc * cap if t == 0 else soc[t - 1]
        net  = solar_kwh[t] - hour_consumption(t)
        prob += soc[t] == prev + net - waste[t]
        prob += soc[t] >= battery_min_soc * cap
        prob += soc[t] <= battery_max_soc * cap
        prob += waste[t] <= solar_kwh[t]

    # ── Objective ─────────────────────────────────────────────────────────────
    # Weights — priority order:
    #   W_WASTE    = 100  → primary: don't waste solar
    #   W_COMFORT  = 50   → strong: don't cut appliance hours unnecessarily
    #   W_DEVIATION = 1   → weak:   stay close to original timing
    #   W_SEMI     = 10   → semi-fixed: reluctant to move cooling
    #
    # W_COMFORT < W_WASTE so the LP will cut hours if it genuinely can't
    # avoid wasting solar or violating battery limits otherwise.
    # W_COMFORT > W_DEVIATION so timing shifts are preferred over hour cuts.
    W_WASTE     = 100.0
    W_COMFORT   = 50.0
    W_DEVIATION = 1.0
    W_SEMI      = 10.0

    deviation_terms = []
    for a in shiftable:
        for t in HOURS:
            orig = original_schedule[a][t]
            deviation_terms.append(x[a][t] if orig == 0 else 1 - x[a][t])

    semi_move_terms = []
    for a in semi_fixed:
        for t in HOURS:
            orig = original_schedule[a][t]
            semi_move_terms.append(x[a][t] if orig == 0 else 1 - x[a][t])

    prob += (
        W_WASTE    * pulp.lpSum(waste[t] for t in HOURS)
        + W_COMFORT  * pulp.lpSum(hours_cut[a] for a in shiftable + semi_fixed)
        + W_DEVIATION * pulp.lpSum(deviation_terms)
        + W_SEMI     * pulp.lpSum(semi_move_terms)
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    status     = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[prob.status]

    if status_str != "Optimal":
        return {
            "status":   status_str,
            "feasible": False,
            "message":  (
                "No feasible schedule found even after allowing load reduction. "
                "Your always-on loads alone may exceed available solar and battery. "
                "Consider reducing minimum SoC or increasing battery/panel capacity."
            ),
        }

    # ── Extract solution ──────────────────────────────────────────────────────
    optimized = {}
    for a in always_on:
        optimized[a] = [1] * 24
    for a in fixed:
        optimized[a] = list(original_schedule[a])
    for a in shiftable + semi_fixed:
        optimized[a] = [int(round(pulp.value(x[a][t]))) for t in HOURS]

    soc_kwh     = [pulp.value(soc[t]) for t in HOURS]
    soc_percent = [round(s / cap * 100, 1) for s in soc_kwh]
    wasted      = [round(pulp.value(waste[t]), 3) for t in HOURS]

    # Summarise any hour reductions for display
    reductions = {}
    for a in shiftable + semi_fixed:
        cut = round(pulp.value(hours_cut[a]))
        if cut > 0:
            reductions[a] = {
                'predicted_hours': predicted_hours[a],
                'scheduled_hours': predicted_hours[a] - cut,
                'hours_cut':       cut,
            }

    return {
        "status":            status_str,
        "feasible":          True,
        "optimized_schedule": optimized,
        "battery_soc_kwh":   [round(s, 3) for s in soc_kwh],
        "battery_soc_percent": soc_percent,
        "wasted_solar_kwh":  wasted,
        "total_wasted_kwh":  round(sum(wasted), 3),
        "reductions":        reductions,   # {} if no hours were cut
    }


# ============================================================
# 4. EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    LAT, LON, TZ = 33.89, 35.50, "Asia/Beirut"
    DATE         = "2024-07-15"
    SYSTEM_KW    = 5.0
    TILT, AZIMUTH = 30, 180

    BATTERY_CAPACITY = 10.0
    BATTERY_START    = 0.5
    BATTERY_MIN      = 0.20

    appliance_power = {
        "fridge": 0.15, "freezer": 0.20,
        "washing_machine": 0.85, "dishwasher": 1.20,
        "water_heater": 2.00, "cooling": 1.30, "cooking": 2.50,
    }

    # Deliberately heavy schedule — more load than a 5kW system can support
    original_schedule = {
        "fridge":          [1]*24,
        "freezer":         [1]*24,
        "washing_machine": [0]*8 + [1]*6 + [0]*10,   # 6 hrs midday
        "dishwasher":      [0]*7 + [1]*5 + [0]*12,   # 5 hrs morning
        "water_heater":    [0]*6 + [1]*4 + [0]*14,   # 4 hrs morning
        "cooling":         [0]*9 + [1]*12 + [0]*3,   # 12 hrs daytime
        "cooking":         [0]*12 + [1] + [0]*5 + [1] + [0]*5,
    }

    solar = compute_solar_generation(DATE, LAT, LON, TZ, SYSTEM_KW, TILT, AZIMUTH)

    result = optimize_schedule(
        original_schedule=original_schedule,
        appliance_power=appliance_power,
        solar_kwh=solar,
        battery_capacity=BATTERY_CAPACITY,
        battery_start_soc=BATTERY_START,
        battery_min_soc=BATTERY_MIN,
        always_on={"fridge","freezer"},
        fixed={"cooking"},
        semi_fixed={"cooling"},
        acceptable_hours={"cooling":(0,24), "washing_machine":(7,22),
                          "dishwasher":(7,23), "water_heater":(5,23)},
    )

    if not result["feasible"]:
        print("INFEASIBLE:", result["message"])
    else:
        print(f"Status: {result['status']}")
        print(f"Total wasted solar: {result['total_wasted_kwh']} kWh\n")

        if result["reductions"]:
            print("⚠ Hours reduced due to energy constraints:")
            for a, r in result["reductions"].items():
                print(f"  {a}: {r['predicted_hours']}h → {r['scheduled_hours']}h "
                      f"(cut {r['hours_cut']}h)")
            print()

        print("Optimized schedule:")
        for a in ["washing_machine","dishwasher","water_heater","cooling"]:
            orig = [t for t in range(24) if original_schedule[a][t]]
            new  = [t for t in range(24) if result["optimized_schedule"][a][t]]
            print(f"  {a:18s} {orig} → {new}")

        print("\nBattery SoC:")
        for t in range(24):
            bar = "█" * int(result["battery_soc_percent"][t] / 5)
            print(f"  {t:02d}:00  {result['battery_soc_percent'][t]:5.1f}%  {bar}")