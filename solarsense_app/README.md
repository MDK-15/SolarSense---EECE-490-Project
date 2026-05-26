# SolarSense

Smart energy management web app for Lebanese solar-only households.

## Project Structure

```
solar_app/
├── app.py                    # Flask application (routes, DB, GCS)
├── requirements.txt
├── Dockerfile
├── solar_sense.db            # SQLite database (auto-created on first run)
│
├── base_models/              # ← Place your trained .pkl files here
│   ├── elec_cooling_on.pkl
│   ├── elec_clothes_washer_on.pkl
│   ├── elec_hot_water_on.pkl
│   ├── elec_television_on.pkl
│   ├── elec_heating_on.pkl
│   ├── solar_autoencoder.pt      # from Solar_Fault_Detection notebook
│   ├── solar_threshold.pkl       # p95 threshold value (pickle float)
│   ├── battery_autoencoder.pt    # from Battery_Fault_Detection notebook
│   └── battery_threshold.pkl     # p95 threshold value
│
├── ml/
│   ├── fault_detection.py    # Solar + battery anomaly detection
│   ├── nilm.py               # Signature capture + detection
│   └── scheduler.py          # XGBoost predict + LP wrapper
│
├── templates/
│   ├── base.html
│   ├── auth.html             # Login / register
│   ├── setup.html            # Onboarding wizard (location, system, appliances)
│   ├── onboarding.html       # Appliance signature capture
│   ├── schedule.html         # Tab 1: schedule recommendation
│   └── faults.html           # Tab 2: solar + battery diagnostics
│
└── static/
    ├── style.css
    └── app.js
```

Also copy into the `solar_app/` root these files from your project:
- `finetune_household.py`
- `solar_lp_optimizer.py`

## Local Setup

```bash
cd solar_app
pip install -r requirements.txt
python app.py
# Visit http://localhost:8080
```

## Google Cloud Run Deployment

### 1. Create a GCS bucket for user models

```bash
gsutil mb gs://solar-sense-models
```

### 2. Set environment variables

In Cloud Run, set:
- `SECRET_KEY` — a long random string for session signing
- `GCS_BUCKET` — your bucket name (default: `solar-sense-models`)

### 3. Build and deploy

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/solar-sense
gcloud run deploy solar-sense \
  --image gcr.io/YOUR_PROJECT/solar-sense \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars SECRET_KEY=your-secret,GCS_BUCKET=solar-sense-models
```

### 4. Persistent SQLite

Cloud Run containers are stateless. For the SQLite database to persist across
deployments, mount a Cloud Filestore NFS volume or switch to Cloud SQL (PostgreSQL).
For a small number of users, the simplest approach is to use Cloud SQL with the
`flask-sqlalchemy` adapter — a straightforward migration.

## Saving Model Thresholds

After training the autoencoders in the notebooks, save the threshold values:

```python
import pickle
# Solar
THRESHOLD = np.percentile(train_errors, 95)
with open('solar_threshold.pkl', 'wb') as f:
    pickle.dump(float(THRESHOLD), f)

# Battery
THRESHOLD = np.percentile(train_errors, 90)
with open('battery_threshold.pkl', 'wb') as f:
    pickle.dump(float(THRESHOLD), f)
```

Then place both `.pkl` files in `base_models/`.

## Privacy

Raw data uploaded by users is deleted immediately after processing:
- Load CSV → disaggregated → fine-tuned model → **CSV deleted**
- Parquet intermediary → **deleted**
- Only the fine-tuned `.pkl` model is kept (in GCS)

Solar and battery CSVs are analysed in memory and never written to disk.
