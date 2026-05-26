"""
app.py — SolarSense Flask application
"""
import os, json, uuid, tempfile, shutil
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, flash, g)
from flask_bcrypt import Bcrypt
import sqlite3
import pandas as pd

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')
bcrypt = Bcrypt(app)

# Jinja filter: parse JSON string in templates
import json as _json
@app.template_filter('fromjson')
def fromjson_filter(s):
    try:
        return _json.loads(s) if s else {}
    except Exception:
        return {}

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE        = os.path.join(app.root_path, 'solar_sense.db')
GCS_BUCKET      = os.environ.get('GCS_BUCKET', 'solar-sense-models')
BASE_MODELS_DIR = os.path.join(app.root_path, 'base_models')

# Appliance types that are inherently always-on and do NOT need signature
# calibration. The user simply enters their rated power draw and they are
# assumed to run every hour. Treated as always-on everywhere in the app,
# regardless of whatever 'tier' value happens to be stored.
ALWAYS_ON_TYPES = {'fridge', 'freezer'}


def is_always_on(appliance) -> bool:
    """True if an appliance row should be treated as always-on / power-entry-only.

    An appliance qualifies if either its category (appliance_type) is one of the
    always-on types (fridge/freezer) OR its flexibility tier is explicitly
    'always_on'. Accepts a sqlite3.Row, a dict, or anything supporting __getitem__.
    """
    try:
        atype = (appliance['appliance_type'] or '').lower()
    except (KeyError, IndexError, TypeError):
        atype = ''
    try:
        tier = (appliance['tier'] or '').lower()
    except (KeyError, IndexError, TypeError):
        tier = ''
    return atype in ALWAYS_ON_TYPES or tier == 'always_on'

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id               TEXT PRIMARY KEY,
            username         TEXT UNIQUE NOT NULL,
            password         TEXT NOT NULL,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            lat              REAL,
            lon              REAL,
            tz               TEXT DEFAULT "Asia/Beirut",
            battery_capacity REAL DEFAULT 10.0,
            battery_min_soc  REAL DEFAULT 0.2,
            solar_system_kw  REAL DEFAULT 5.0,
            solar_tilt       REAL DEFAULT 30.0,
            solar_azimuth    REAL DEFAULT 180.0,
            setup_complete   INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS appliances (
            id             TEXT PRIMARY KEY,
            user_id        TEXT NOT NULL,
            name           TEXT NOT NULL,
            display_name   TEXT NOT NULL,
            appliance_type TEXT DEFAULT "other",
            power_kw       REAL DEFAULT 0.0,
            tier           TEXT DEFAULT "shiftable",
            hour_start     INTEGER DEFAULT 7,
            hour_end       INTEGER DEFAULT 22,
            has_signature       INTEGER DEFAULT 0,
            calibration_pending INTEGER DEFAULT 0,
            signature_json      TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS signature_events (
            id       TEXT PRIMARY KEY,
            user_id  TEXT NOT NULL,
            app_id   TEXT NOT NULL,
            on_time  TIMESTAMP,
            off_time TIMESTAMP,
            status   TEXT DEFAULT "pending",
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS upload_log (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            upload_type TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed   INTEGER DEFAULT 0
        );
    ''')
    db.commit()

    # Migrations — add columns that may be missing from older DB files
    cols = [row[1] for row in db.execute("PRAGMA table_info(appliances)").fetchall()]
    if 'appliance_type' not in cols:
        db.execute("ALTER TABLE appliances ADD COLUMN appliance_type TEXT DEFAULT 'other'")
        db.commit()
    if 'power_kw' not in cols:
        db.execute("ALTER TABLE appliances ADD COLUMN power_kw REAL DEFAULT 0.0")
        db.commit()
    if 'calibration_pending' not in cols:
        db.execute("ALTER TABLE appliances ADD COLUMN calibration_pending INTEGER DEFAULT 0")
        db.commit()

    # Repair any fridge/freezer appliances that were stored with a tier other
    # than 'always_on'. These appliances never need calibration; pinning them to
    # always_on ensures they are included in generated schedules and shown with
    # a power-entry field instead of a "needs calibration" badge.
    db.execute(
        "UPDATE appliances SET tier='always_on' "
        "WHERE LOWER(appliance_type) IN ('fridge','freezer') AND tier != 'always_on'"
    )
    db.commit()


with app.app_context():
    init_db()


# ── GCS helpers (lazy import — won't crash if SDK not installed locally) ───────
def upload_model_to_gcs(local_path: str, user_id: str, filename: str):
    try:
        from google.cloud import storage as gcs
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(f'models/{user_id}/{filename}')
        blob.upload_from_filename(local_path)
    except Exception as e:
        app.logger.warning(f'GCS upload skipped: {e}')


def download_model_from_gcs(user_id: str, filename: str, dest_path: str) -> bool:
    # Skip GCS entirely if no credentials are configured (local dev mode)
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and        not os.environ.get('GCS_BUCKET'):
        return False
    try:
        from google.cloud import storage as gcs
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(f'models/{user_id}/{filename}')
        if blob.exists():
            blob.download_to_filename(dest_path)
            return True
    except Exception as e:
        app.logger.warning(f'GCS download skipped: {e}')
    return False


def get_user_models_dir(user_id: str) -> str:
    """Download user models from GCS into a temp dir. Falls back to base models."""
    tmpdir = tempfile.mkdtemp(prefix=f'user_{user_id}_')
    try:
        from ml.scheduler import APPLIANCE_LABELS
        for app_name in APPLIANCE_LABELS:
            fname = f'{app_name}.pkl'
            dest  = os.path.join(tmpdir, fname)
            if not download_model_from_gcs(user_id, fname, dest):
                base = os.path.join(BASE_MODELS_DIR, fname)
                if os.path.exists(base):
                    shutil.copy(base, dest)
    except Exception as e:
        app.logger.warning(f'get_user_models_dir: {e}')
    return tmpdir


# ── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    if 'user_id' not in session:
        return None
    user = get_db().execute(
        'SELECT * FROM users WHERE id = ?', (session['user_id'],)
    ).fetchone()
    if user is None:
        # Session references a user that doesn't exist in the DB
        # (e.g. DB was replaced). Clear the session so login works.
        session.clear()
    return user


def require_user():
    """Returns user or aborts with JSON 401 — use in API routes."""
    user = get_current_user()
    if user is None:
        from flask import abort
        abort(401)
    return user


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('auth.html', mode='register')
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('auth.html', mode='register')
        db = get_db()
        if db.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone():
            flash('Username already taken.', 'error')
            return render_template('auth.html', mode='register')
        user_id = str(uuid.uuid4())
        pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        db.execute('INSERT INTO users (id, username, password) VALUES (?,?,?)',
                   (user_id, username, pw_hash))
        db.commit()
        session['user_id'] = user_id
        return redirect(url_for('setup'))
    return render_template('auth.html', mode='register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db   = get_db()
        user = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        if not user or not bcrypt.check_password_hash(user['password'], password):
            flash('Invalid username or password.', 'error')
            return render_template('auth.html', mode='login')
        session['user_id'] = user['id']
        if not user['setup_complete']:
            return redirect(url_for('setup'))
        return redirect(url_for('dashboard'))
    return render_template('auth.html', mode='login')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── Setup ─────────────────────────────────────────────────────────────────────
@app.route('/setup')
@login_required
def setup():
    user       = get_current_user()
    appliances = get_db().execute(
        'SELECT * FROM appliances WHERE user_id=?', (user['id'],)
    ).fetchall()
    return render_template('setup.html', user=user, appliances=appliances)


@app.route('/api/setup/step1', methods=['POST'])
@login_required
def setup_step1():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400
    user = get_current_user()
    try:
        get_db().execute(
            '''UPDATE users SET lat=?,lon=?,tz=?,
               battery_capacity=?,battery_min_soc=?,
               solar_system_kw=?,solar_tilt=?,solar_azimuth=?
               WHERE id=?''',
            (float(data['lat']), float(data['lon']),
             data.get('tz','Asia/Beirut'),
             float(data.get('battery_capacity', 10)),
             float(data.get('battery_min_soc', 20)) / 100,
             float(data.get('solar_system_kw', 5)),
             float(data.get('solar_tilt', 30)),
             float(data.get('solar_azimuth', 180)),
             user['id'])
        )
        get_db().commit()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/setup/appliance', methods=['POST'])
@login_required
def setup_add_appliance():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400
    user   = get_current_user()
    app_id = str(uuid.uuid4())
    try:
        appliance_type = data.get('appliance_type', 'other')
        tier           = data.get('tier', 'shiftable')
        # Fridges and freezers are always-on by nature: they never need signature
        # calibration, only a rated power draw. Force the always_on tier so the
        # rest of the app classifies them correctly.
        if appliance_type in ALWAYS_ON_TYPES:
            tier = 'always_on'
        get_db().execute(
            '''INSERT INTO appliances
               (id,user_id,name,display_name,appliance_type,power_kw,tier,hour_start,hour_end)
               VALUES (?,?,?,?,?,?,?,?,?)''',
            (app_id, user['id'],
             data.get('name','').lower().replace(' ','_'),
             data.get('display_name', data.get('name','')),
             appliance_type,
             0.0,
             tier,
             int(data.get('hour_start', 7)),
             int(data.get('hour_end',  22)))
        )
        get_db().commit()
        return jsonify({'ok': True, 'app_id': app_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/setup/finish', methods=['POST'])
@login_required
def setup_finish():
    get_db().execute('UPDATE users SET setup_complete=1 WHERE id=?',
                     (session['user_id'],))
    get_db().commit()
    return jsonify({'ok': True})


# ── Onboarding ────────────────────────────────────────────────────────────────
@app.route('/onboarding')
@login_required
def onboarding():
    user       = get_current_user()
    appliances = get_db().execute(
        'SELECT * FROM appliances WHERE user_id=?', (user['id'],)
    ).fetchall()
    return render_template('onboarding.html', user=user, appliances=appliances)


@app.route('/api/signature/start', methods=['POST'])
@login_required
def signature_start():
    data   = request.get_json(silent=True) or {}
    app_id = data.get('app_id')
    db     = get_db()
    appl   = db.execute('SELECT id FROM appliances WHERE id=? AND user_id=?',
                        (app_id, session['user_id'])).fetchone()
    if not appl:
        return jsonify({'error': 'Appliance not found'}), 404
    event_id = str(uuid.uuid4())
    db.execute(
        'INSERT INTO signature_events (id,user_id,app_id,on_time,status) VALUES (?,?,?,?,?)',
        (event_id, session['user_id'], app_id, datetime.utcnow(), 'waiting_off')
    )
    db.commit()
    return jsonify({'ok': True, 'event_id': event_id})


@app.route('/api/signature/stop', methods=['POST'])
@login_required
def signature_stop():
    data     = request.get_json(silent=True) or {}
    event_id = data.get('event_id')
    db       = get_db()
    event    = db.execute(
        'SELECT * FROM signature_events WHERE id=? AND user_id=?',
        (event_id, session['user_id'])
    ).fetchone()
    if not event:
        return jsonify({'error': 'Event not found'}), 404
    db.execute("UPDATE signature_events SET off_time=?,status=? WHERE id=?",
               (datetime.utcnow(), 'pending_upload', event_id))
    # Mark appliance as calibration_pending = 1 immediately
    # so the UI shows it as done without waiting for CSV upload
    db.execute("UPDATE appliances SET calibration_pending=1 WHERE id=?",
               (event['app_id'],))
    db.commit()
    return jsonify({'ok': True,
                    'message': 'Timestamps saved. Upload your power log to finish calibration.'})


# ── Geocoding proxy ───────────────────────────────────────────────────────────
@app.route('/api/geocode')
@login_required
def geocode():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'results': []})
    try:
        import requests as req
        r = req.get(
            'https://geocoding-api.open-meteo.com/v1/search',
            params={'name': q, 'count': 6, 'language': 'en', 'format': 'json'},
            timeout=8
        )
        return jsonify(r.json())
    except Exception as e:
        return jsonify({'results': [], 'error': str(e)})


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    user       = get_current_user()
    appliances = get_db().execute(
        'SELECT * FROM appliances WHERE user_id=?', (user['id'],)
    ).fetchall()
    return render_template('dashboard.html', user=user,
                           appliances=appliances, active_tab='schedule')


# ── Schedule tab ──────────────────────────────────────────────────────────────
@app.route('/schedule')
@login_required
def schedule():
    user       = get_current_user()
    appliances = get_db().execute(
        'SELECT * FROM appliances WHERE user_id=?', (user['id'],)
    ).fetchall()
    return render_template('schedule.html', user=user,
                           appliances=appliances, active_tab='schedule')


@app.route('/api/upload/load', methods=['POST'])
@login_required
def upload_load():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    user   = get_current_user()
    db     = get_db()
    tmpdir = tempfile.mkdtemp()
    try:
        # Verify finetune_household.py is available
        import importlib.util, sys
        ft_paths = [
            os.path.join(app.root_path, 'finetune_household.py'),
            os.path.join(os.path.dirname(app.root_path), 'finetune_household.py'),
        ]
        ft_path = next((p for p in ft_paths if os.path.exists(p)), None)
        if ft_path:
            spec = importlib.util.spec_from_file_location('finetune_household', ft_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['finetune_household'] = mod
        else:
            app.logger.warning('finetune_household.py not found — skipping fine-tuning')
    except Exception as e:
        app.logger.warning(f'finetune_household import warning: {e}')

    try:
        csv_path = os.path.join(tmpdir, 'load.csv')
        request.files['file'].save(csv_path)
        df_load = pd.read_csv(csv_path)
        df_load['timestamp'] = pd.to_datetime(df_load['timestamp'])

        # 1. Signature calibration for any pending events
        from ml.nilm import calibrate_from_upload, build_hourly_binary, ApplianceProfile
        for ev in db.execute(
            '''SELECT e.*,a.name as app_name FROM signature_events e
               JOIN appliances a ON e.app_id=a.id
               WHERE e.user_id=? AND e.status="pending_upload"''',
            (user['id'],)
        ).fetchall():
            on_t  = datetime.fromisoformat(str(ev['on_time']))
            off_t = datetime.fromisoformat(str(ev['off_time']))
            appl  = db.execute('SELECT * FROM appliances WHERE id=?',
                               (ev['app_id'],)).fetchone()
            profile = calibrate_from_upload(
                df_load, on_t, off_t,
                appl['appliance_type'] if appl else 'other',
                appl['display_name']   if appl else ev['app_name']
            )
            if profile:
                db.execute(
                    'UPDATE appliances SET has_signature=1,signature_json=?,power_kw=? WHERE id=?',
                    (json.dumps(profile.to_dict()),
                     round(profile.steady_state_power_w / 1000, 3),
                     ev['app_id'])
                )
                db.execute("UPDATE signature_events SET status='complete' WHERE id=?",
                           (ev['id'],))
        db.commit()

        # 2. Build hourly binary
        sigs = {}
        for a in db.execute('SELECT * FROM appliances WHERE user_id=?',
                            (user['id'],)).fetchall():
            if a['has_signature'] and a['signature_json']:
                try:
                    sigs[a['name']] = ApplianceProfile.from_dict(
                        json.loads(a['signature_json']))
                except Exception:
                    pass

        df_hourly = build_hourly_binary(df_load, sigs)
        if df_hourly.empty or len(df_hourly) < 168:
            return jsonify({'warning': 'Not enough data yet — keep collecting!'})

        # 3. Fine-tune (gracefully skips if finetune_household.py not found)
        try:
            from finetune_household import run_finetuning
            usage_path = os.path.join(tmpdir, 'usage.csv')
            df_hourly.to_csv(usage_path, index=False)
            run_finetuning(
                household_id=user['id'],
                usage_csv_path=usage_path,
                lat=user['lat'], lon=user['lon'], tz=user['tz'],
                base_models_dir=BASE_MODELS_DIR,
                households_dir=tmpdir,
            )
        except Exception as ft_err:
            app.logger.warning(f'Fine-tuning skipped: {ft_err}')

        # 4. Upload to GCS
        outdir = os.path.join(tmpdir, user['id'])
        if os.path.exists(outdir):
            for fname in os.listdir(outdir):
                if fname.endswith('.pkl'):
                    upload_model_to_gcs(
                        os.path.join(outdir, fname), user['id'], fname)

        db.execute('INSERT INTO upload_log (id,user_id,upload_type,processed) VALUES (?,?,?,1)',
                   (str(uuid.uuid4()), user['id'], 'load'))
        db.commit()
        return jsonify({'ok': True,
                        'message': 'Done! Your schedule predictions are now personalised to your home.'})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route('/api/schedule/predict', methods=['POST'])
@login_required
def predict_schedule():
    user       = get_current_user()
    appliances = get_db().execute(
        'SELECT * FROM appliances WHERE user_id=?', (user['id'],)
    ).fetchall()

    # Build appliance config — include ALL appliances that have power_kw > 0
    # Also include appliance_type so the scheduler can map to base models
    appliance_config = {}
    for a in appliances:
        pkw = float(a['power_kw']) if a['power_kw'] else 0.0
        if pkw > 0:
            appliance_config[a['name']] = {
                'power_kw':       pkw,
                'tier':           'always_on' if is_always_on(a) else a['tier'],
                'hours':          [a['hour_start'], a['hour_end']],
                'display':        a['display_name'],
                'appliance_type': a['appliance_type'] or 'other',
            }

    # For appliances with power_kw = 0 but has_signature (calibrated), or that
    # are always-on (fridge/freezer), still include them so their schedule can
    # be predicted. Always-on appliances never need a signature.
    for a in appliances:
        if a['name'] not in appliance_config and (a['has_signature'] or is_always_on(a)):
            appliance_config[a['name']] = {
                'power_kw':       float(a['power_kw']) if a['power_kw'] else 0.0,
                'tier':           'always_on' if is_always_on(a) else a['tier'],
                'hours':          [a['hour_start'], a['hour_end']],
                'display':        a['display_name'],
                'appliance_type': a['appliance_type'] or 'other',
            }

    app.logger.info(f"predict_schedule: {len(appliance_config)} appliances: {list(appliance_config.keys())}")
    appliance_config['__solar__'] = {
        'system_kw': user['solar_system_kw'],
        'tilt':      user['solar_tilt'],
        'azimuth':   user['solar_azimuth'],
    }

    user_models_dir = get_user_models_dir(user['id'])
    try:
        from ml.scheduler import run_pipeline
        result = run_pipeline(
            user_id=user['id'],
            user_models_dir=user_models_dir,
            lat=user['lat'], lon=user['lon'], tz=user['tz'],
            appliance_config=appliance_config,
            battery_capacity=user['battery_capacity'],
            battery_start_soc=0.5,
            battery_min_soc=user['battery_min_soc'],
        )
        return jsonify(result)
    finally:
        shutil.rmtree(user_models_dir, ignore_errors=True)


# ── Diagnostics tab ───────────────────────────────────────────────────────────
@app.route('/faults')
@login_required
def faults():
    user = get_current_user()
    return render_template('faults.html', user=user, active_tab='faults')


@app.route('/api/upload/solar', methods=['POST'])
@login_required
def upload_solar():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, 'solar.csv')
        request.files['file'].save(path)
        from ml.fault_detection import analyze_solar
        return jsonify(analyze_solar(pd.read_csv(path)))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route('/api/upload/battery', methods=['POST'])
@login_required
def upload_battery():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, 'battery.csv')
        request.files['file'].save(path)
        from ml.fault_detection import analyze_battery
        return jsonify(analyze_battery(pd.read_csv(path)))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



@app.route('/api/appliances/<app_id>/reset-signature', methods=['POST'])
@login_required
def reset_signature(app_id):
    db   = get_db()
    appl = db.execute('SELECT id FROM appliances WHERE id=? AND user_id=?',
                      (app_id, session['user_id'])).fetchone()
    if not appl:
        return jsonify({'error': 'Not found'}), 404
    db.execute("""UPDATE appliances
                  SET has_signature=0, calibration_pending=0,
                      signature_json=NULL, power_kw=0
                  WHERE id=?""", (app_id,))
    db.execute("DELETE FROM signature_events WHERE app_id=? AND user_id=?",
               (app_id, session['user_id']))
    db.commit()
    return jsonify({'ok': True})


# ── Appliance management page ─────────────────────────────────────────────────
@app.route('/appliances')
@login_required
def appliances_page():
    user       = get_current_user()
    appliances = get_db().execute(
        'SELECT * FROM appliances WHERE user_id=?', (user['id'],)
    ).fetchall()
    return render_template('appliances.html', user=user,
                           appliances=appliances, active_tab='appliances')


# ── Settings page ─────────────────────────────────────────────────────────────
@app.route('/settings')
@login_required
def settings():
    user = get_current_user()
    return render_template('settings.html', user=user, active_tab='settings')

# ── Appliance API ─────────────────────────────────────────────────────────────
@app.route('/api/appliances')
@login_required
def get_appliances():
    rows = get_db().execute(
        'SELECT * FROM appliances WHERE user_id=?', (session['user_id'],)
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/appliances/<app_id>', methods=['PUT', 'DELETE'])
@login_required
def manage_appliance(app_id):
    db   = get_db()
    appl = db.execute('SELECT * FROM appliances WHERE id=? AND user_id=?',
                      (app_id, session['user_id'])).fetchone()
    if not appl:
        return jsonify({'error': 'Not found'}), 404
    if request.method == 'DELETE':
        db.execute('DELETE FROM appliances WHERE id=?', (app_id,))
        db.commit()
        return jsonify({'ok': True})
    data = request.get_json(silent=True) or {}
    # Determine the tier, keeping fridges/freezers pinned to always_on so they
    # are never reclassified as needing calibration.
    new_tier = data.get('tier', appl['tier'])
    if (appl['appliance_type'] or '').lower() in ALWAYS_ON_TYPES:
        new_tier = 'always_on'
    db.execute(
        '''UPDATE appliances SET display_name=?,power_kw=?,tier=?,
           hour_start=?,hour_end=? WHERE id=?''',
        (data.get('display_name', appl['display_name']),
         float(data.get('power_kw', appl['power_kw'])),
         new_tier,
         int(data.get('hour_start', appl['hour_start'])),
         int(data.get('hour_end',   appl['hour_end'])),
         app_id)
    )
    db.commit()
    return jsonify({'ok': True})


@app.route('/api/user/settings', methods=['PUT'])
@login_required
def update_settings():
    data = request.get_json(silent=True) or {}
    get_db().execute(
        '''UPDATE users SET lat=?,lon=?,tz=?,
           battery_capacity=?,battery_min_soc=?,
           solar_system_kw=?,solar_tilt=?,solar_azimuth=? WHERE id=?''',
        (float(data.get('lat', 33.89)),
         float(data.get('lon', 35.50)),
         data.get('tz', 'Asia/Beirut'),
         float(data.get('battery_capacity', 10)),
         float(data.get('battery_min_soc',  20)) / 100,
         float(data.get('solar_system_kw',   5)),
         float(data.get('solar_tilt',        30)),
         float(data.get('solar_azimuth',    180)),
         session['user_id'])
    )
    get_db().commit()
    return jsonify({'ok': True})


@app.errorhandler(401)
def handle_401(e):
    return jsonify({'error': 'Session expired. Please log in again.',
                    'redirect': '/login'}), 401

@app.errorhandler(500)
def handle_500(e):
    app.logger.error(f'500 error: {e}')
    return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'Unhandled exception: {e}')
    return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
