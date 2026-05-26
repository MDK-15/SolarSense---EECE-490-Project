#!/usr/bin/env python3

import subprocess
import csv
import time
import os
import glob
import logging
import shutil
from datetime import datetime, timedelta

# ── CONFIGURATION ──────────────────────────────────────
MPP_SOLAR_PATH    = "/home/tayeb/.pyenv/versions/3.11.9/bin/mpp-solar"
LOG_INTERVAL      = 1.0
BUFFER_DIR        = "/home/tayeb/inverter_buffer"

# ── LOGGING SETUP ───────────────────────────────────────
def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

log = logging.getLogger(__name__)

# ── WEEK FOLDER ─────────────────────────────────────────
def get_week_folder():
    today = datetime.now().date()
    monday = today - timedelta(days=today.weekday())
    return str(monday)

# ── USB STORAGE DETECTION ───────────────────────────────
def find_usb():
    candidates = (
        glob.glob("/media/tayeb/*")
        + glob.glob("/media/*")
        + ["/media/usb0", "/media/usb1", "/media/usb2", "/media/usb3",
           "/mnt/usb", "/mnt/usb0", "/mnt/usb1"]
    )
    for mount in candidates:
        if not os.path.ismount(mount):
            continue
        if not os.access(mount, os.W_OK):
            continue
        try:
            test_file = os.path.join(mount, ".ping")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return mount
        except OSError:
            continue
    return None

def get_usb_log_path(usb_root):
    return os.path.join(usb_root, "inverter_logs", "inverter.log")

def get_usb_week_dir(usb_root):
    return os.path.join(usb_root, "inverter_logs", get_week_folder())

def get_buffer_week_dir():
    return os.path.join(BUFFER_DIR, get_week_folder())

# ── INVERTER DETECTION ──────────────────────────────────
def find_inverter_device():
    """
    Returns (device_path, brand) or (None, None).
    Voltronic uses HID (hidraw), Growatt/Deye use serial (ttyUSB).
    """
    # Check for Voltronic first (HID device)
    for i in range(4):
        path = f"/dev/hidraw{i}"
        if os.path.exists(path):
            log.info(f"HID device found at {path} — assuming Voltronic")
            return path, "voltronic"

    # Check for serial device (Growatt or Deye)
    for i in range(4):
        path = f"/dev/ttyUSB{i}"
        if os.path.exists(path):
            log.info(f"Serial device found at {path} — probing brand...")
            brand = probe_serial_brand(path)
            if brand:
                log.info(f"Identified as {brand}")
                return path, brand
            else:
                log.warning(f"Device at {path} did not respond to any known protocol")

    return None, None

def probe_serial_brand(port):
    """
    Try Growatt then Deye Modbus on a serial port.
    Returns 'growatt', 'deye', or None.
    """
    # Try Growatt
    try:
        from pymodbus.client import ModbusSerialClient
        client = ModbusSerialClient(
            port=port, baudrate=9600, bytesize=8,
            parity='N', stopbits=1, timeout=2
        )
        if client.connect():
            # Growatt input register 1 = PV power (register 0x0001, unit 1)
            result = client.read_input_registers(address=1, count=1, slave=1)
            client.close()
            if not result.isError():
                return "growatt"
    except Exception:
        pass

    # Try Deye
    try:
        from pymodbus.client import ModbusSerialClient
        client = ModbusSerialClient(
            port=port, baudrate=9600, bytesize=8,
            parity='N', stopbits=1, timeout=2
        )
        if client.connect():
            # Deye holding register 0x0003 = device type
            result = client.read_holding_registers(address=3, count=1, slave=1)
            client.close()
            if not result.isError():
                return "deye"
    except Exception:
        pass

    return None

# ── VOLTRONIC POLLING ───────────────────────────────────
def poll_voltronic(device):
    for protocol in ["PI30", "PI30MAX"]:
        try:
            result = subprocess.run(
                [MPP_SOLAR_PATH, "-p", device, "-P", protocol,
                 "-c", "QPIGS", "-o", "screen"],
                capture_output=True, text=True, timeout=0.8
            )
            data = parse_voltronic_output(result.stdout)
            if data:
                return data
        except subprocess.TimeoutExpired:
            continue
        except FileNotFoundError as e:
            log.error(f"mpp-solar not found: {e}")
            return None
        except Exception as e:
            log.error(f"Voltronic poll error: {e}")
            return None
    return None

def parse_voltronic_output(text):
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("Command") or \
           line.startswith("---") or line.startswith("Parameter"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            data[parts[0]] = parts[1]
    return data if data else None

# ── GROWATT POLLING ─────────────────────────────────────
def poll_growatt(device):
    try:
        from pymodbus.client import ModbusSerialClient
        client = ModbusSerialClient(
            port=device, baudrate=9600, bytesize=8,
            parity='N', stopbits=1, timeout=1
        )
        if not client.connect():
            return None

        # Read input registers 0-60 (main status block)
        r = client.read_input_registers(address=0, count=60, slave=1)
        client.close()

        if r.isError():
            return None

        regs = r.registers
        return {
            # Solar
            "pv_input_voltage":              regs[3] * 0.1,
            "pv_input_current_for_battery":  regs[4] * 0.1,
            "pv_input_power":                regs[1],
            "battery_voltage_from_scc":      regs[40] * 0.01,
            "is_scc_charging_on":            1 if regs[1] > 0 else 0,
            # Battery
            "battery_voltage":               regs[37] * 0.01,
            "battery_charging_current":      regs[38] * 0.1,
            "battery_discharge_current":     regs[18] * 0.1,
            "battery_capacity":              regs[103] if len(regs) > 103 else "",
            "is_charging_on":                1 if regs[38] > 0 else 0,
            # Load
            "ac_output_voltage":             regs[35] * 0.1,
            "ac_output_frequency":           regs[36] * 0.01,
            "ac_output_active_power":        regs[35],
            "ac_output_apparent_power":      regs[34],
            "ac_output_load":                regs[39],
            "is_load_on":                    1,
            # Grid
            "ac_input_voltage":              regs[20] * 0.1,
            "ac_input_frequency":            regs[21] * 0.01,
            "is_ac_charging_on":             1 if regs[22] > 0 else 0,
            "bus_voltage":                   regs[26] * 0.1,
            # Status
            "inverter_heat_sink_temperature": regs[55] * 0.1,
            "is_scc_charging_on":            1 if regs[1] > 0 else 0,
        }
    except Exception as e:
        log.error(f"Growatt poll error: {e}")
        return None

# ── DEYE POLLING ────────────────────────────────────────
def poll_deye(device):
    try:
        from pymodbus.client import ModbusSerialClient
        client = ModbusSerialClient(
            port=device, baudrate=9600, bytesize=8,
            parity='N', stopbits=1, timeout=1
        )
        if not client.connect():
            return None

        # Read holding registers 0x003B-0x0078 (main status block)
        r = client.read_holding_registers(address=0x003B, count=60, slave=1)
        client.close()

        if r.isError():
            return None

        regs = r.registers
        return {
            # Solar
            "pv_input_voltage":              regs[0] * 0.1,
            "pv_input_current_for_battery":  regs[1] * 0.1,
            "pv_input_power":                regs[2],
            "battery_voltage_from_scc":      regs[0] * 0.1,
            "is_scc_charging_on":            1 if regs[2] > 0 else 0,
            # Battery
            "battery_voltage":               regs[13] * 0.01,
            "battery_charging_current":      regs[14] * 0.1,
            "battery_discharge_current":     regs[15] * 0.1,
            "battery_capacity":              regs[16],
            "is_charging_on":                1 if regs[14] > 0 else 0,
            # Load
            "ac_output_voltage":             regs[37] * 0.1,
            "ac_output_frequency":           regs[38] * 0.01,
            "ac_output_active_power":        regs[40],
            "ac_output_apparent_power":      regs[41],
            "ac_output_load":                regs[42],
            "is_load_on":                    1,
            # Grid
            "ac_input_voltage":              regs[33] * 0.1,
            "ac_input_frequency":            regs[34] * 0.01,
            "is_ac_charging_on":             1 if regs[29] > 0 else 0,
            "bus_voltage":                   regs[12] * 0.1,
            # Status
            "inverter_heat_sink_temperature": regs[54] * 0.1,
            "is_scc_charging_on":            1 if regs[2] > 0 else 0,
        }
    except Exception as e:
        log.error(f"Deye poll error: {e}")
        return None

# ── UNIFIED POLL ────────────────────────────────────────
def poll_inverter(device, brand):
    if brand == "voltronic":
        return poll_voltronic(device)
    elif brand == "growatt":
        return poll_growatt(device)
    elif brand == "deye":
        return poll_deye(device)
    return None

# ── CSV SCHEMAS ─────────────────────────────────────────
CSV_SCHEMAS = {
    "solar": [
        "timestamp",
        "pv_input_voltage",
        "pv_input_current_for_battery",
        "pv_input_power",
        "battery_voltage_from_scc",
        "is_scc_charging_on",
    ],
    "battery": [
        "timestamp",
        "battery_voltage",
        "battery_charging_current",
        "battery_discharge_current",
        "battery_capacity",
        "is_charging_on",
    ],
    "load": [
        "timestamp",
        "ac_output_voltage",
        "ac_output_frequency",
        "ac_output_active_power",
        "ac_output_apparent_power",
        "ac_output_load",
        "is_load_on",
    ],
    "grid": [
        "timestamp",
        "ac_input_voltage",
        "ac_input_frequency",
        "is_ac_charging_on",
        "bus_voltage",
    ],
    "status": [
        "timestamp",
        "inverter_heat_sink_temperature",
        "is_charging_on",
        "is_load_on",
        "is_ac_charging_on",
        "is_scc_charging_on",
    ],
}

# ── CSV WRITING ─────────────────────────────────────────
def write_row(week_dir, category, data, timestamp):
    os.makedirs(week_dir, exist_ok=True)
    fields = CSV_SCHEMAS[category]
    path = os.path.join(week_dir, f"{category}.csv")
    is_new = not os.path.exists(path)
    row = {"timestamp": timestamp}
    for field in fields:
        if field != "timestamp":
            row[field] = data.get(field, "")
    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        log.error(f"Failed to write {category}: {e}")

# ── BUFFER FLUSH ─────────────────────────────────────────
def flush_buffer(usb_root):
    if not os.path.exists(BUFFER_DIR):
        return
    week_folders = [f for f in glob.glob(os.path.join(BUFFER_DIR, "*"))
                    if os.path.isdir(f)]
    if not week_folders:
        return
    log.info("Flushing buffer to USB...")
    for week_folder in week_folders:
        week_name = os.path.basename(week_folder)
        usb_week_dir = os.path.join(usb_root, "inverter_logs", week_name)
        try:
            os.makedirs(usb_week_dir, exist_ok=True)
            for category in CSV_SCHEMAS:
                buffer_file = os.path.join(week_folder, f"{category}.csv")
                usb_file = os.path.join(usb_week_dir, f"{category}.csv")
                if not os.path.exists(buffer_file):
                    continue
                if not os.path.exists(usb_file):
                    shutil.move(buffer_file, usb_file)
                    continue
                with open(buffer_file, "r") as bf:
                    lines = bf.readlines()
                data_lines = lines[1:] if len(lines) > 1 else []
                if data_lines:
                    with open(usb_file, "a") as uf:
                        uf.writelines(data_lines)
                os.remove(buffer_file)
            try:
                os.rmdir(week_folder)
            except OSError:
                pass
        except OSError as e:
            log.error(f"Failed to flush {week_name} to USB: {e}")
            raise
    log.info("Buffer flush complete.")

# ── MAIN LOOP ───────────────────────────────────────────
def main():
    os.makedirs(BUFFER_DIR, exist_ok=True)
    setup_logging(os.path.join(BUFFER_DIR, "inverter.log"))
    log.info("=== Voltronic Logger starting ===")

    usb_root        = None
    was_usb_present = False
    device          = None
    brand           = None
    no_data_count   = 0

    while True:
        loop_start = time.monotonic()

        # ── USB storage check ──────────────────────────
        current_usb = find_usb()

        if current_usb and not was_usb_present:
            try:
                flush_buffer(current_usb)
                usb_root = current_usb
                was_usb_present = True
                usb_log = get_usb_log_path(usb_root)
                os.makedirs(os.path.dirname(usb_log), exist_ok=True)
                if os.path.isdir(usb_log):
                    shutil.rmtree(usb_log)
                setup_logging(usb_log)
                log.info(f"USB found at {usb_root} — logging to USB.")
            except OSError as e:
                log.warning(f"USB appeared but failed: {e} — staying on buffer.")
                was_usb_present = False
                usb_root = None

        elif not current_usb and was_usb_present:
            usb_root = None
            was_usb_present = False
            setup_logging(os.path.join(BUFFER_DIR, "inverter.log"))
            log.warning("USB removed — buffering to SD card.")

        # ── Inverter detection ─────────────────────────
        if device is None or no_data_count >= 10:
            device, brand = find_inverter_device()
            no_data_count = 0
            if device:
                log.info(f"Inverter: {brand} on {device}")
            else:
                log.warning("No inverter found — retrying in 30s...")
                time.sleep(30)
                continue

        # ── Poll inverter ──────────────────────────────
        week_dir  = get_usb_week_dir(current_usb) if current_usb else get_buffer_week_dir()
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        data      = poll_inverter(device, brand)

        if data:
            no_data_count = 0
            try:
                for category in CSV_SCHEMAS:
                    write_row(week_dir, category, data, timestamp)
                pv   = data.get("pv_input_power", "?")
                bat  = data.get("battery_voltage", "?")
                soc  = data.get("battery_capacity", "?")
                load = data.get("ac_output_active_power", "?")
                log.info(f"[{brand}] PV:{pv}W  Bat:{bat}V({soc}%)  Load:{load}W")
            except OSError as e:
                log.warning(f"Write failed ({e}) — switching to buffer.")
                was_usb_present = False
                usb_root = None
                week_dir = get_buffer_week_dir()
                setup_logging(os.path.join(BUFFER_DIR, "inverter.log"))
                try:
                    for category in CSV_SCHEMAS:
                        write_row(week_dir, category, data, timestamp)
                except OSError as e2:
                    log.error(f"Buffer write also failed: {e2}")
        else:
            no_data_count += 1
            log.warning(f"No data from inverter ({no_data_count}/10)")

        # ── Sleep to maintain 1Hz ──────────────────────
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, LOG_INTERVAL - elapsed))

if __name__ == "__main__":
    main()