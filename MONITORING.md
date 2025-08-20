# Model Loading Monitoring Guide

## Current Status: ✅ WORKING

Based on the logs, your models are successfully loading from Google Cloud Storage.

## Monitoring Commands

### 1. Real-time Model Loading Status

```bash
# Watch model loading in real-time
gcloud logging tail "resource.type=cloud_run_revision AND textPayload:model" \
  --format="table(timestamp,textPayload)"

# Watch GCS downloads
gcloud logging tail "resource.type=cloud_run_revision AND textPayload:GCS" \
  --format="table(timestamp,textPayload)"
```

### 2. Check Model Loading Success

```bash
# Verify models loaded successfully
gcloud logging read "resource.type=cloud_run_revision AND textPayload:loaded successfully" \
  --limit 5 --format="table(timestamp,textPayload)"

# Check for any loading errors
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" \
  --limit 10 --format="table(timestamp,severity,textPayload)"
```

### 3. Monitor Performance Metrics

```bash
# Check memory usage during model loading
gcloud logging read "resource.type=cloud_run_revision AND textPayload:MB RAM used" \
  --limit 10 --format="table(timestamp,textPayload)"

# Check cold start times
gcloud logging read "resource.type=cloud_run_revision AND textPayload:Starting new instance" \
  --limit 5 --format="table(timestamp,textPayload)"
```

### 4. Test Model Functionality

```bash
# Your Cloud Run Service URL
SERVICE_URL="https://kurdishtts-gcp-137160846844.europe-north2.run.app"

# Test if models are working
curl "$SERVICE_URL/speakers?dialect=sorani"

# Test TTS generation (Sorani)
curl -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "سڵاو، چۆنی؟", "dialect": "sorani", "speaker": "speaker_0000"}' \
  --output test_sorani.wav

# Test TTS generation (Kurmanji)
curl -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Silav, çawa yî?", "dialect": "kurmanji", "speaker": "speaker_0000"}' \
  --output test_kurmanji.wav
```

## Expected Log Patterns

### ✅ Successful Model Loading

```
- Attempting to download model files from Google Cloud Storage bucket: kurdishttsmodels
- Successfully downloaded [file] from bucket kurdishttsmodels to [local_path]
- Successfully downloaded all model files from GCS
- Loading models from files...
- Sorani model loaded successfully
- Kurmanji model loaded successfully
- All models loaded and cached successfully
```

### ⚠️ Warning Patterns (Normal)

```
- File [file] exists but is not valid JSON: [error]. Will re-download from GCS.
- Could not initialize NNPACK! Reason: Unsupported hardware.
```

### ❌ Error Patterns (Concerning)

```
- Failed to download [file]: [error]
- Failed to load [dialect] model: [error]
- File not found: [file]
```

## Performance Benchmarks

### Expected Timings

- **Cold Start**: 2-5 minutes (model download + loading)
- **Warm Start**: 30-60 seconds (model loading only)
- **Memory Usage**: ~2.5-3GB after model loading
- **Response Time**: 3-5 seconds for TTS generation

### Monitoring Dashboard

1. **Google Cloud Console** → **Cloud Run** → **Your Service** → **Metrics**
2. Monitor:
   - Request count
   - Request latency
   - Memory usage
   - CPU usage

## Troubleshooting

### If Models Don't Load

1. Check GCS bucket permissions
2. Verify file paths in GCS
3. Check service account permissions
4. Review error logs

### If Performance is Poor

1. Increase memory allocation (4GB → 8GB)
2. Reduce concurrency to 1
3. Monitor cold start times
4. Consider model optimization

## Quick Health Check

```bash
# One-liner to check if everything is working
curl -s "https://kurdishtts-gcp-137160846844.europe-north2.run.app/speakers?dialect=sorani" | jq '.available_speakers | length'
```

Expected output: A number > 0 (indicating speakers are loaded)
