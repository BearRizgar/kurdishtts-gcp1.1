# Kurdish TTS Google Cloud Run Deployment Guide

## Overview

This guide will help you deploy the Kurdish TTS application to Google Cloud Run with models stored in Google Cloud Storage.

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **Google Cloud CLI** installed and configured
3. **Docker** installed locally (for testing)
4. **Models uploaded to Google Cloud Storage** in the `kurdishttsmodels` bucket

## Model Files in GCS

Based on your GCS bucket, ensure these files are uploaded:

### Sorani Models (`models/sorani/`)

- `checkpoint_95000.pth` (891.2 MB)
- `yourtts_improved_config_fixed.json` (6.2 KB)
- `kurdish_text_cleaners.py` (21 KB)
- `simple_multispeaker_formatter.py` (3.9 KB)

### Kurmanji Models (`models/kurmanji/`)

- `checkpoint_80000.pth` (891 MB)
- `kurmanji_config.json` (6.2 KB)
- `kurmanji_text_cleaners.py` (8.6 KB)
- `kurmanji_multispeaker_formatter.py` (3.8 KB)

## Environment Variables

The application supports these environment variables:

- `GCS_BUCKET`: GCS bucket name (default: `kurdishttsmodels`)
- `USE_GCS`: Whether to use GCS (default: `true`)
- `ALLOWED_ORIGINS`: CORS origins (default: `http://localhost:3000,https://www.kurdishtts.com`)
- `PORT`: Port to run on (default: `8080`)

## Deployment Steps

### 1. Build and Deploy to Cloud Run

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Build and deploy
gcloud run deploy kurdishtts \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --set-env-vars "GCS_BUCKET=kurdishttsmodels,USE_GCS=true" \
  --project $PROJECT_ID
```

### 2. Alternative: Build Locally and Push

```bash
# Build the container
docker build -t gcr.io/$PROJECT_ID/kurdishtts .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/kurdishtts

# Deploy from GCR
gcloud run deploy kurdishtts \
  --image gcr.io/$PROJECT_ID/kurdishtts \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --set-env-vars "GCS_BUCKET=kurdishttsmodels,USE_GCS=true"
```

## Service Account Setup

### Option 1: Use Default Compute Service Account (Recommended)

The default service account should have the necessary permissions. If not, grant them:

```bash
# Get the default service account
export SERVICE_ACCOUNT=$(gcloud iam service-accounts list --filter="displayName:Compute Engine default service account" --format="value(email)")

# Grant Storage Object Viewer role
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.objectViewer"
```

### Option 2: Create Custom Service Account

```bash
# Create service account
gcloud iam service-accounts create kurdishtts-sa \
  --display-name="Kurdish TTS Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:kurdishtts-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# Use in deployment
gcloud run deploy kurdishtts \
  --source . \
  --platform managed \
  --region us-central1 \
  --service-account=kurdishtts-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1
```

## Resource Configuration

### Memory and CPU

- **Memory**: 4GB (required for model loading)
- **CPU**: 2 vCPUs (recommended for inference speed)
- **Concurrency**: 1 (to avoid memory issues)

### Timeout

- **Request timeout**: 3600 seconds (60 minutes) for model loading
- **Cold start**: First request may take 2-5 minutes to download models

## Testing the Deployment

### 1. Test the Health Check

```bash
curl https://your-service-url.run.app/speakers?dialect=sorani
```

### 2. Test Speech Generation

```bash
curl -X POST https://your-service-url.run.app/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "سڵاو، چۆنی؟",
    "dialect": "sorani",
    "speaker": "speaker_0000"
  }' \
  --output test_audio.wav
```

## Monitoring and Logs

### View Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=kurdishtts" --limit 50
```

### Monitor Performance

- Check Cloud Run metrics in Google Cloud Console
- Monitor memory usage during cold starts
- Watch for timeout errors during model loading

## Troubleshooting

### Common Issues

1. **Model Download Failures**

   - Check GCS bucket permissions
   - Verify file paths in GCS
   - Check service account permissions

2. **Memory Issues**

   - Increase memory allocation to 8GB if needed
   - Reduce concurrency to 1
   - Monitor memory usage in logs

3. **Timeout Issues**

   - Increase timeout to 3600 seconds
   - Check network connectivity to GCS
   - Verify model file sizes

4. **Import Errors**
   - Ensure all Python dependencies are in requirements.txt
   - Check for missing model files

### Debug Commands

```bash
# Check service status
gcloud run services describe kurdishtts --region us-central1

# View recent logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=kurdishtts" --limit 20

# Test GCS access
gcloud storage ls gs://kurdishttsmodels/models/
```

## Cost Optimization

1. **Use Cloud Run's auto-scaling** to scale to zero when not in use
2. **Monitor cold start times** and optimize model loading
3. **Consider using Cloud CDN** for frequently accessed models
4. **Use appropriate instance sizes** based on actual usage

## Security Considerations

1. **CORS Configuration**: Update `ALLOWED_ORIGINS` for production domains
2. **Service Account**: Use minimal required permissions
3. **HTTPS**: Cloud Run provides HTTPS by default
4. **Authentication**: Consider adding authentication if needed

## Updates and Maintenance

### Updating Models

1. Upload new model files to GCS
2. Redeploy the service (models are downloaded fresh each deployment)
3. Test with new models

### Updating Code

1. Push code changes
2. Redeploy with `gcloud run deploy`
3. Monitor logs for any issues

## Support

For issues related to:

- **Google Cloud Run**: Check [Cloud Run documentation](https://cloud.google.com/run/docs)
- **Google Cloud Storage**: Check [GCS documentation](https://cloud.google.com/storage/docs)
- **Application Issues**: Check application logs and this deployment guide
