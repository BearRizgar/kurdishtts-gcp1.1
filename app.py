import os
import torch
from flask import Flask, request, jsonify, send_file, redirect, after_this_request, make_response
from pathlib import Path
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
import numpy as np
import soundfile as sf
from flask_cors import CORS
import gc
import time
import psutil
from datetime import datetime, timedelta
import logging
import re
import sys
import librosa
import io
import wave
import json
from scipy import signal, ndimage
from google.cloud import storage

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
# Force deployment update - using correct model paths
logger = logging.getLogger(__name__)

# Production configuration
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

# Get allowed origins from environment variable or default to localhost and production
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,https://www.kurdishtts.com').split(',')

# Configure CORS with proper security settings
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "max_age": 3600
    }
})

# Production security headers
@app.after_request
def add_headers(response):
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; media-src 'self' blob: data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
    
    # Cache control
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000'
    else:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    
    return response

# Production performance optimizations
device = torch.device("cpu")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = True
torch.backends.mkldnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Error handling
@app.errorhandler(404)
def not_found_error(error):
    logger.error(f'Page not found: {request.url}')
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    logger.error(f'Unhandled Exception: {str(e)}', exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

def load_model_from_paths(model_path, config_path):
    """Load a TTS model from specific paths."""
    model_path = Path(model_path)
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")
    
    config = VitsConfig()
    config.load_json(str(config_path))
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # Load checkpoint to get speaker information
    checkpoint = torch.load(str(model_path), map_location=device)
    
    # Initialize speaker manager
    from TTS.tts.utils.speakers import SpeakerManager
    speaker_manager = SpeakerManager()
    
    # Get speaker count from checkpoint
    if 'model' in checkpoint and 'emb_g.weight' in checkpoint['model']:
        num_speakers = checkpoint['model']['emb_g.weight'].shape[0]
        logger.info(f"Found {num_speakers} speakers in checkpoint")
        
        # Create speaker names
        speaker_names = [f"speaker_{i:04d}" for i in range(num_speakers)]
        
        # Set up speaker manager
        speaker_manager.name_to_id = {name: i for i, name in enumerate(speaker_names)}
        speaker_manager.id_to_name = {i: name for i, name in enumerate(speaker_names)}
        
        # Update config with correct number of speakers
        config.num_speakers = num_speakers
    else:
        logger.warning("Could not determine speaker count from checkpoint, using default")
        # Fallback to default speaker setup
        speaker_names = ["speaker_0000", "speaker_0001"]  # male, female
        speaker_manager.name_to_id = {name: i for i, name in enumerate(speaker_names)}
        speaker_manager.id_to_name = {i: name for i, name in enumerate(speaker_names)}
        config.num_speakers = 2
    
    model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
    
    # Load model state
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model, ap, tokenizer



# --- GCS download helpers (currently unused, but kept for future use) ---
def download_from_gcs(bucket_name, gcs_path, local_path):
    """Download a file from Google Cloud Storage to a local path."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        # Check if blob exists
        if not blob.exists():
            raise FileNotFoundError(f"Blob {gcs_path} not found in bucket {bucket_name}")
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Successfully downloaded {gcs_path} to {local_path}")
        
    except Exception as e:
        logger.error(f"Failed to download {gcs_path} from bucket {bucket_name}: {str(e)}")
        raise e

def ensure_model_file(local_path, gcs_bucket, gcs_path):
    need_download = False
    if not os.path.exists(local_path):
        need_download = True
        logger.info(f"File {local_path} does not exist, will download from GCS")
    else:
        # Check if file is empty
        if os.path.getsize(local_path) == 0:
            logger.warning(f"File {local_path} exists but is empty. Will re-download from GCS.")
            need_download = True
        else:
            # Optionally, check if it's valid JSON
            if local_path.endswith('.json'):
                try:
                    with open(local_path, 'r') as f:
                        json.load(f)
                except Exception as e:
                    logger.warning(f"File {local_path} exists but is not valid JSON: {e}. Will re-download from GCS.")
                    need_download = True
    
    if need_download:
        try:
            logger.info(f"Downloading {gcs_path} from bucket {gcs_bucket} to {local_path}")
            download_from_gcs(gcs_bucket, gcs_path, local_path)
        except Exception as e:
            logger.error(f"Failed to download {gcs_path}: {str(e)}")
            # If this is a critical file, re-raise the exception
            if 'checkpoint' in local_path or 'config.json' in local_path:
                raise e
            else:
                logger.warning(f"Non-critical file {local_path} download failed, continuing...")
    else:
        logger.info(f"Model file already exists and is valid: {local_path}")

# --- KurdishTTSManager ---
class KurdishTTSManager:
    def __init__(self):
        self.models = {}
        self.model_cache = {}
        self.load_models()
    
    def load_models(self):
        # Clear any existing models
        self.models.clear()
        self.model_cache.clear()
        
        # Get GCS configuration from environment variables
        gcs_bucket = os.getenv('GCS_BUCKET', 'kurdishttsmodels1')
        use_gcs = os.getenv('USE_GCS', 'true').lower() == 'true'
        
        # Create models directory if it doesn't exist
        os.makedirs('models/sorani', exist_ok=True)
        os.makedirs('models/kurmanji', exist_ok=True)
        
        if use_gcs:
            try:
                logger.info(f"Attempting to download model files from Google Cloud Storage bucket: {gcs_bucket}")
                
                # Sorani model files
                ensure_model_file('models/sorani/checkpoint_250000.pth', gcs_bucket, 'models/sorani/checkpoint_250000.pth')
                ensure_model_file('models/sorani/yourtts_hifigan_enhanced_config.json', gcs_bucket, 'models/sorani/yourtts_hifigan_enhanced_config.json')
                ensure_model_file('models/sorani/kurdish_text_cleaners.py', gcs_bucket, 'models/sorani/kurdish_text_cleaners.py')
                ensure_model_file('models/sorani/simple_multispeaker_formatter.py', gcs_bucket, 'models/sorani/simple_multispeaker_formatter.py')
                
                # Kurmanji model files
                ensure_model_file('models/kurmanji/checkpoint_80000.pth', gcs_bucket, 'models/kurmanji/checkpoint_80000.pth')
                ensure_model_file('models/kurmanji/kurmanji_config.json', gcs_bucket, 'models/kurmanji/kurmanji_config.json')
                ensure_model_file('models/kurmanji/kurmanji_text_cleaners.py', gcs_bucket, 'models/kurmanji/kurmanji_text_cleaners.py')
                ensure_model_file('models/kurmanji/kurmanji_multispeaker_formatter.py', gcs_bucket, 'models/kurmanji/kurmanji_multispeaker_formatter.py')
                
                logger.info("Successfully downloaded all model files from GCS")
                
            except Exception as e:
                logger.error(f"Failed to download models from GCS: {str(e)}")
                logger.warning("Falling back to local model files (if available)")
                # Continue with local files if they exist
        
        # Load models from files (either downloaded from GCS or local)
        logger.info("Loading models from files...")
        
        try:
            self.models['sorani'] = load_model_from_paths(
                'models/sorani/checkpoint_250000.pth',
                'models/sorani/yourtts_hifigan_enhanced_config.json'
            )
            logger.info("Sorani model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Sorani model: {str(e)}")
            raise e
        
        try:
            self.models['kurmanji'] = load_model_from_paths(
                'models/kurmanji/checkpoint_80000.pth',
                'models/kurmanji/kurmanji_config.json'
            )
            logger.info("Kurmanji model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Kurmanji model: {str(e)}")
            raise e
        
        # Pre-cache all models
        self.model_cache['sorani'] = self.models['sorani']
        self.model_cache['kurmanji'] = self.models['kurmanji']
        
        logger.info("All models loaded and cached successfully")
    
    def get(self, dialect):
        # Return cached model based on dialect
        if dialect in self.model_cache:
            return self.model_cache[dialect]
        return None
    
    def clear_cache(self):
        """Clear the model cache and free memory."""
        # Don't clear the cache, just run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

tts_manager = KurdishTTSManager()

def add_prosody_markers(text, dialect='sorani'):
    # Add prosody markers for natural pauses
    # Replace punctuation with prosody markers
    text = text.replace('،', ' , ')
    text = text.replace('؛', ' ; ')
    text = text.replace('؟', ' ? ')
    text = text.replace('!', ' ! ')
    text = text.replace('.', ' . ')

    if dialect == 'kurmanji':
        # Kurmanji-specific conjunctions and natural pause points
        conjunctions = [
            'û', 'lê', 'belê', 'çimkî', 'ji ber ku', 'lewra', 'ji bo ku',
            'herwiha', 'bi taybetî', 'bi rastî', 'bi heman awayî',
            'wekî ku', 'wekî din', 'wekî din jî', 'wekî wê jî'
        ]
        # Reduce aggressive pause insertion to prevent unnatural speech patterns
        # Only add pauses for major conjunctions, not all prepositions
        major_conjunctions = ['û', 'lê', 'belê', 'çimkî', 'ji ber ku', 'lewra', 'ji bo ku']
        for conj in major_conjunctions:
            text = text.replace(f' {conj} ', f' {conj} , ')
        
        # Remove aggressive preposition and relative marker processing
        # prepositions = ['bi', 'di', 'ji', 'li', 'ser', 'bin', 'ber', 'pêş']
        # for prep in prepositions:
        #     text = text.replace(f' {prep} ', f' {prep} , ')
        # relative_markers = ['ku', 'yê', 'ya', 'yên', 'yê ku', 'ya ku']
        # for marker in relative_markers:
        #     text = text.replace(f' {marker} ', f' , {marker} ')
        # Remove emphasis markers to prevent volume fluctuations
        # emphasis_words = ['gelek', 'pir', 'zêde', 'herî', 'tam']
        # for word in emphasis_words:
        #     text = text.replace(f' {word} ', f' ! {word} ! ')

        # Reduce aggressive text manipulation to preserve natural speech patterns
        # Only handle the most common and necessary patterns
        # Handle compound words (keep only the most common ones)
        text = text.replace(' di nav ', ' dinav ')
        text = text.replace(' di ser ', ' diser ')
        
        # Remove aggressive verb prefix and suffix manipulation
        # text = text.replace(' di ', ' di-')
        # text = text.replace(' bi ', ' bi-')
        # text = text.replace(' ji ', ' ji-')
        # text = text.replace(' -a ', 'a ')
        # text = text.replace(' -ê ', 'ê ')
        # text = text.replace(' -an ', 'an ')
        # text = text.replace(' -ên ', 'ên ')
    else:
        conjunctions = ['و', 'بەڵام', 'چونکە', 'لەبەر ئەوەی', 'لەوەڕ', 'بۆ ئەوەی']
        for conj in conjunctions:
            text = text.replace(f' {conj} ', f' {conj} , ')

    # Clean up multiple spaces and ensure proper spacing around markers
    text = ' '.join(text.split())
    return text

def add_extra_commas(text, n=8):
    words = text.split()
    for i in range(n, len(words), n):
        words[i-1] += ','
    return ' '.join(words)

# Helper to extract waveform from model output
def extract_wav_from_output(outputs):
    if isinstance(outputs, dict):
        for key in ['wav', 'waveform', 'audio']:
            if key in outputs:
                return outputs[key]
        return next(iter(outputs.values()))
    elif isinstance(outputs, (tuple, list)):
        return outputs[0]
    elif isinstance(outputs, torch.Tensor):
        return outputs
    else:
        raise ValueError(f"Unexpected output type: {type(outputs)})")



def generate_single_sentence(model, ap, tokenizer, text, dialect, speaker="speaker_0000"):
    """Generate speech for a single sentence."""
    try:
        # Get speaker ID from speaker manager
        if hasattr(model, 'speaker_manager') and model.speaker_manager:
            if speaker in model.speaker_manager.name_to_id:
                speaker_id = model.speaker_manager.name_to_id[speaker]
            else:
                # Fallback to first speaker if requested speaker not found
                speaker_id = 0
                logger.warning(f"Speaker '{speaker}' not found, using first speaker")
        else:
            # Fallback for models without speaker manager
            speaker_id = 0 if speaker == "male" else 1
        
        speaker_inputs = torch.LongTensor([speaker_id])
        
        # Optimized parameters for better audio quality
        # Based on successful Gradio interface settings
        if dialect == 'sorani':
            length_scale = 1.70  # Slower for Sorani speakers
            noise_scale = 0.3    # Lower noise for more natural speech
            noise_scale_dp = 0.5  # Stable duration prediction
        else:  # kurmanji
            length_scale = 1.50  # Slower for Kurmanji speakers
            noise_scale = 0.3    # Lower noise for more natural speech
            noise_scale_dp = 0.5  # Stable duration prediction
        
        logger.info(f"Using optimized parameters: length_scale={length_scale}, noise_scale={noise_scale}, noise_scale_dp={noise_scale_dp} for text length {len(text)} with speaker {speaker} (ID: {speaker_id})")
        
        # Preprocess text based on dialect
        if dialect == 'kurmanji':
            # Import Kurmanji text cleaner
            try:
                from models.kurmanji.kurmanji_text_cleaners import kurmanji_cleaners
                cleaned_text = kurmanji_cleaners(text)
                logger.info(f"Kurmanji text cleaned: '{text}' -> '{cleaned_text}'")
            except ImportError:
                cleaned_text = text
                logger.warning("Kurmanji text cleaner not found, using original text")
        else:  # sorani
            # Import Sorani text cleaner
            try:
                from models.sorani.kurdish_text_cleaners import kurdish_cleaners
                cleaned_text = kurdish_cleaners(text)
                logger.info(f"Sorani text cleaned: '{text}' -> '{cleaned_text}'")
            except ImportError:
                cleaned_text = text
                logger.warning("Sorani text cleaner not found, using original text")
        
        # Tokenize text
        text_inputs = tokenizer.text_to_ids(cleaned_text)
        text_inputs = torch.tensor(text_inputs).unsqueeze(0)
        text_lengths = torch.LongTensor([text_inputs.size(1)])
        
        # Move to device if available
        device = next(model.parameters()).device
        text_inputs = text_inputs.to(device)
        speaker_inputs = speaker_inputs.to(device)
        text_lengths = text_lengths.to(device)
        
        # Temporarily adjust model parameters for this inference
        original_length_scale = getattr(model, 'length_scale', 1.0)
        original_noise_scale = getattr(model, 'inference_noise_scale', 0.667)
        original_noise_scale_dp = getattr(model, 'inference_noise_scale_dp', 0.8)
        
        # Set optimized model parameters
        model.length_scale = length_scale
        model.inference_noise_scale = noise_scale
        model.inference_noise_scale_dp = noise_scale_dp
        
        # Add temperature and sampling parameters for more natural speech
        if hasattr(model, 'temperature'):
            model.temperature = 0.7  # Add some randomness for naturalness
        if hasattr(model, 'top_k'):
            model.top_k = 50  # Limit vocabulary diversity
        if hasattr(model, 'top_p'):
            model.top_p = 0.8  # Nucleus sampling threshold
        
        with torch.no_grad():
            try:
                # Use the correct VITS inference method with proper parameters
                audio = model.inference(
                    x=text_inputs,
                    aux_input={
                        'speaker_ids': speaker_inputs,
                        'x_lengths': text_lengths,
                        'd_vectors': None,
                        'language_ids': None,
                        'durations': None
                    }
                )
                logger.info("Used VITS inference with aux_input")
                
            except Exception as e:
                logger.warning(f"VITS inference failed: {e}, trying fallback")
                # Fallback to simple inference
                audio = model.inference(text_inputs)
                logger.info("Used fallback simple inference")
        
        # Restore original parameters
        model.length_scale = original_length_scale
        model.inference_noise_scale = original_noise_scale
        model.inference_noise_scale_dp = original_noise_scale_dp
        
        # Extract audio from output
        if isinstance(audio, dict):
            if 'model_outputs' in audio:
                audio = audio['model_outputs']
            elif 'wav' in audio:
                audio = audio['wav']
            elif 'output' in audio:
                audio = audio['output']
            else:
                # Take the first tensor value
                for key, value in audio.items():
                    if isinstance(value, torch.Tensor):
                        audio = value
                        logger.info(f"Using audio from key: {key}")
                        break
                else:
                    raise ValueError("Could not find audio tensor in model output")
        
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Denormalize audio
        wav = ap.denormalize(audio)
        
        # Apply post-processing for better quality
        wav = post_process_audio(wav, dialect)
        
        logger.info(f"Generated audio: {wav.shape} samples")
        return wav
        
    except Exception as e:
        logger.error(f"Error in generate_single_sentence: {e}")
        raise e

def post_process_audio(audio: np.ndarray, dialect='sorani') -> np.ndarray:
    """Apply soft, round post-processing to improve audio quality and reduce distortion."""
    try:
        from scipy import signal
        
        print("🔧 Starting soft, round post-processing...")
        
        # Step 1: Remove DC offset
        audio = audio - np.mean(audio)
        
        # Step 2: Apply very gentle low-pass filtering to soften harsh frequencies
        sample_rate = 22050
        high_freq = 8000 / (sample_rate / 2)  # 8kHz cutoff to soften high frequencies
        b, a = signal.butter(3, high_freq, btype='low')
        audio = signal.filtfilt(b, a, audio)
        
        # Step 3: Apply very gentle high-pass filtering to remove low-frequency rumble
        low_freq = 60 / (sample_rate / 2)  # 60 Hz cutoff (very gentle)
        b, a = signal.butter(2, low_freq, btype='high')
        audio = signal.filtfilt(b, a, audio)
        
        # Step 4: Apply very soft compression with higher threshold and lower ratio
        threshold = 0.7  # Higher threshold - less compression
        ratio = 1.5      # Much lower ratio - softer compression
        
        audio_compressed = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
            audio
        )
        
        # Step 5: Apply gentle spectral smoothing to reduce harshness
        # Use a larger smoothing window for softer sound
        window_size = 5
        audio_smooth = signal.savgol_filter(audio_compressed, window_size, 2)
        
        # Mix original with smoothed to preserve some detail
        audio_compressed = 0.7 * audio_compressed + 0.3 * audio_smooth
        
        # Step 6: Apply longer, gentler fade in/out for softer transitions
        fade_samples = min(3000, len(audio_compressed) // 10)  # Longer fade
        if fade_samples > 0:
            # Use smoother fade curve for softer transitions
            fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
            fade_out = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, fade_samples)))
            
            audio_compressed[:fade_samples] *= fade_in
            audio_compressed[-fade_samples:] *= fade_out
        
        # Step 7: Apply gentle volume boost for better audibility
        rms = np.sqrt(np.mean(audio_compressed**2))
        if rms < 0.25:  # If volume is too low
            volume_boost = 1.2  # Gentle boost by 20%
            audio_compressed = audio_compressed * volume_boost
            print(f"🔊 Applied gentle volume boost of {volume_boost}x (RMS was {rms:.3f})")
        
        # Step 8: Final normalization with lower peak to prevent distortion
        max_val = np.max(np.abs(audio_compressed))
        if max_val > 0.8:  # Lower peak threshold for softer sound
            audio_compressed = audio_compressed * (0.8 / max_val)
        
        print("✅ Applied soft, round post-processing: gentle filtering, soft compression, spectral smoothing, and natural fading")
        return audio_compressed
        
    except Exception as e:
        print(f"⚠️ Post-processing failed: {e}")
        print("🔄 Returning original audio...")
        return audio

def trim_silence(wav, top_db=25):
    """Trim leading and trailing silence with optimized settings for better quality."""
    try:
        # Use more aggressive trimming for cleaner audio
        trimmed, _ = librosa.effects.trim(wav, top_db=top_db, frame_length=2048, hop_length=512)
        return trimmed
    except Exception as e:
        logger.warning(f"Silence trimming failed: {e}, returning original audio")
        return wav

def fade_out(wav, sample_rate, fade_ms=50):
    """Apply fade-out with optimized settings for smoother transitions."""
    fade_len = int(sample_rate * fade_ms / 1000)
    if fade_len > len(wav):
        fade_len = len(wav)
    
    # Use cosine-based fade for smoother transition
    fade_curve = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
    wav[-fade_len:] *= fade_curve
    return wav

# =====================================================================
# AUDIO ENHANCEMENT FUNCTIONS
# =====================================================================

def enhance_audio_quality(audio, sample_rate):
    """
    Enhance audio quality with soft, round processing to reduce distortion
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate (typically 22050)
    
    Returns:
        Enhanced audio array with softer, more natural sound
    """
    try:
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # VIBRATO REDUCTION: Very light pitch smoothing to reduce artificial vibrato
        enhanced = reduce_vibrato_light(audio, sample_rate)
        
        # Very soft dynamic range compression
        threshold = 0.75  # Higher threshold - less compression
        ratio = 1.3       # Much lower ratio - softer compression
        compressed = np.where(
            np.abs(enhanced) > threshold,
            np.sign(enhanced) * (threshold + (np.abs(enhanced) - threshold) / ratio),
            enhanced
        )
        
        # Gentle low-pass filtering to soften harsh frequencies
        nyquist = sample_rate / 2
        low_freq = 7000 / nyquist  # 7kHz cutoff to soften high frequencies
        b, a = signal.butter(3, low_freq, btype='low')
        low_pass = signal.filtfilt(b, a, compressed)
        
        # Mix with original to preserve some detail
        enhanced = 0.8 * compressed + 0.2 * low_pass
        
        # Spectral smoothing for softer sound
        enhanced = spectral_smoothing(enhanced, sample_rate)
        
        # Final normalization with lower peak to prevent distortion
        enhanced = enhanced / np.max(np.abs(enhanced)) * 0.8
        return enhanced
            
    except Exception as e:
        logger.warning(f"⚠️ Audio enhancement failed: {e}, returning original")
        return audio



def reduce_vibrato(audio, sample_rate):
    """
    Reduce vibrato by smoothing pitch variations in the frequency domain
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
    
    Returns:
        Audio with reduced vibrato
    """
    try:
        # STFT with smaller hop length for better time resolution
        stft = librosa.stft(audio, n_fft=1024, hop_length=128)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # VIBRATO REDUCTION TECHNIQUES:
        
        # 1. Temporal smoothing of magnitude (reduces pitch variations)
        # Use larger temporal window to smooth out vibrato
        magnitude_smooth = ndimage.median_filter(magnitude, size=(1, 5))
        
        # 2. Phase smoothing to reduce frequency modulation
        # Smooth phase differences to reduce vibrato
        phase_diff = np.diff(phase, axis=1)
        phase_diff_smooth = ndimage.median_filter(phase_diff, size=(1, 3))
        
        # Reconstruct phase from smoothed differences
        phase_smooth = np.zeros_like(phase)
        phase_smooth[:, 0] = phase[:, 0]
        for i in range(1, phase.shape[1]):
            phase_smooth[:, i] = phase_smooth[:, i-1] + phase_diff_smooth[:, i-1]
        
        # 3. Reconstruct audio with smoothed magnitude and phase
        stft_smooth = magnitude_smooth * np.exp(1j * phase_smooth)
        return librosa.istft(stft_smooth, hop_length=128)
        
    except Exception as e:
        logger.warning(f"⚠️ Vibrato reduction failed: {e}, returning original")
        return audio


def reduce_vibrato_light(audio, sample_rate):
    """
    Very light vibrato reduction using gentle temporal smoothing
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
    
    Returns:
        Audio with very lightly reduced vibrato
    """
    try:
        # Very gentle temporal smoothing approach
        # Apply a very gentle low-pass filter to reduce high-frequency pitch variations
        nyquist = sample_rate / 2
        cutoff = 0.9  # Keep almost all frequencies, just smooth very slightly
        b, a = signal.butter(2, cutoff, btype='low')
        smoothed = signal.filtfilt(b, a, audio)
        
        # Mix original with smoothed to preserve most natural variation
        return 0.85 * audio + 0.15 * smoothed
        
    except Exception as e:
        logger.warning(f"⚠️ Light vibrato reduction failed: {e}, returning original")
        return audio


def spectral_smoothing(audio, sample_rate):
    """Apply gentle spectral smoothing for softer sound"""
    try:
        # STFT
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Apply very gentle median filtering for smoothing
        magnitude_smooth = ndimage.median_filter(magnitude, size=(2, 1))  # Smaller filter for gentler smoothing
        
        # Mix original with smoothed to preserve detail
        magnitude_mixed = 0.8 * magnitude + 0.2 * magnitude_smooth
        
        # Reconstruct
        stft_smooth = magnitude_mixed * np.exp(1j * phase)
        return librosa.istft(stft_smooth, hop_length=256)
        
    except Exception:
        return audio



@app.route('/speakers', methods=['GET'])
def get_speakers():
    """Get available speakers for each dialect."""
    try:
        dialect = request.args.get('dialect', 'sorani')
        model_tuple = tts_manager.get(dialect)
        if not model_tuple:
            return jsonify({'error': f'Failed to load {dialect} model'}), 400
        model, _, _ = model_tuple
        if hasattr(model, 'speaker_manager') and model.speaker_manager:
            return jsonify({
                'dialect': dialect,
                'speakers': model.speaker_manager.name_to_id,
                'available_speakers': list(model.speaker_manager.name_to_id.keys())
            })
        else:
            return jsonify({
                'dialect': dialect,
                'speakers': {"male": 0, "female": 1},
                'available_speakers': ["male", "female"]
            })
    except Exception as e:
        logger.error(f"Error in get_speakers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/all-speakers', methods=['GET'])
def get_all_speakers():
    """Get all available speakers for both dialects."""
    try:
        all_speakers = {}
        
        # Get Sorani speakers
        try:
            sorani_tuple = tts_manager.get('sorani')
            if sorani_tuple and hasattr(sorani_tuple[0], 'speaker_manager') and sorani_tuple[0].speaker_manager:
                all_speakers['sorani'] = {
                    'speakers': sorani_tuple[0].speaker_manager.name_to_id,
                    'available_speakers': list(sorani_tuple[0].speaker_manager.name_to_id.keys()),
                    'total_count': len(sorani_tuple[0].speaker_manager.name_to_id)
                }
            else:
                all_speakers['sorani'] = {
                    'speakers': {"male": 0, "female": 1},
                    'available_speakers': ["male", "female"],
                    'total_count': 2
                }
        except Exception as e:
            logger.warning(f"Error loading Sorani speakers: {e}")
            all_speakers['sorani'] = {
                'error': f'Failed to load Sorani speakers: {str(e)}',
                'speakers': {},
                'available_speakers': [],
                'total_count': 0
            }
        
        # Get Kurmanji speakers
        try:
            kurmanji_tuple = tts_manager.get('kurmanji')
            if kurmanji_tuple and hasattr(kurmanji_tuple[0], 'speaker_manager') and kurmanji_tuple[0].speaker_manager:
                all_speakers['kurmanji'] = {
                    'speakers': kurmanji_tuple[0].speaker_manager.name_to_id,
                    'available_speakers': list(kurmanji_tuple[0].speaker_manager.name_to_id.keys()),
                    'total_count': len(kurmanji_tuple[0].speaker_manager.name_to_id)
                }
            else:
                all_speakers['kurmanji'] = {
                    'speakers': {"male": 0, "female": 1},
                    'available_speakers': ["male", "female"],
                    'total_count': 2
                }
        except Exception as e:
            logger.warning(f"Error loading Kurmanji speakers: {e}")
            all_speakers['kurmanji'] = {
                'error': f'Failed to load Kurmanji speakers: {str(e)}',
                'speakers': {},
                'available_speakers': [],
                'total_count': 0
            }
        
        # Add summary
        total_sorani = all_speakers['sorani'].get('total_count', 0)
        total_kurmanji = all_speakers['kurmanji'].get('total_count', 0)
        
        return jsonify({
            'summary': {
                'total_speakers': total_sorani + total_kurmanji,
                'sorani_speakers': total_sorani,
                'kurmanji_speakers': total_kurmanji
            },
            'dialects': all_speakers
        })
        
    except Exception as e:
        logger.error(f"Error in get_all_speakers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_speech():
    try:
        clear_model_cache()
        process = psutil.Process(os.getpid())
        logger.info(f'Before inference: {process.memory_info().rss / 1024 ** 2:.2f} MB RAM used')
        
        data = request.json
        text = data.get('text', '')
        dialect = data.get('dialect', 'sorani')
        speaker = data.get('speaker', 'speaker_0000')  # Default to first speaker
        
        # NEW: Enhancement parameter (enabled by default)
        enhance_audio = data.get('enhance_audio', True)
        
        logger.info(f"Processing text: '{text}' with dialect {dialect} and speaker {speaker}")
        if enhance_audio:
            logger.info("Audio enhancement enabled")
        
        # Get model
        model_tuple = tts_manager.get(dialect)
        if not model_tuple:
            return jsonify({'error': f'Failed to load {dialect} model'}), 400
        model, ap, tokenizer = model_tuple
        
        # Apply prosody markers for the text
        processed_text = add_prosody_markers(text, dialect)
        # Add extra commas for natural phrasing
        processed_text = add_extra_commas(processed_text, n=8)
        logger.info(f"Processing text with prosody markers: {processed_text}")
        
        # Generate audio for the entire text
        wav = generate_single_sentence(model, ap, tokenizer, processed_text, dialect, speaker)
        
        # Trim trailing silence and apply fade-out
        wav = trim_silence(wav, top_db=30)
        wav = fade_out(wav, ap.sample_rate, fade_ms=20)
        
        # NEW: Apply enhancement if requested
        if enhance_audio:
            logger.info("🎵 Applying audio enhancement...")
            start_time = time.time()
            wav = enhance_audio_quality(wav, ap.sample_rate)
            enhancement_time = time.time() - start_time
            logger.info(f"✅ Enhancement completed in {enhancement_time:.3f} seconds")
        
        # Convert to 16-bit PCM
        audio_data = (wav * 32768).astype(np.int16)
        
        # Create audio file
        audio_file = io.BytesIO()
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(ap.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        audio_file.seek(0)
        logger.info(f'After inference: {process.memory_info().rss / 1024 ** 2:.2f} MB RAM used')
        
        # Create response with enhancement info in headers
        response = make_response(send_file(audio_file, mimetype='audio/wav'))
        if enhance_audio:
            response.headers['X-Audio-Enhanced'] = 'true'
        
        return response
        
    except Exception as e:
        logger.error(f"Error in generate_speech: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhancement-info', methods=['GET'])
def get_enhancement_info():
    """Get information about available enhancement options"""
    return jsonify({
        'enhancement_available': True,
        'enhancement_description': {
            'description': 'Optimized audio enhancement for Kurdish speech with vibrato reduction',
            'processing_time': 'Fast (~0.017s additional)',
            'quality_improvement': 'Noticeable improvement in naturalness, clarity, and reduced artificial vibrato',
            'features': [
                'Vibrato reduction for more stable pitch',
                'Dynamic range compression',
                'High-frequency emphasis for Kurdish consonants',
                'Spectral smoothing for Kurdish phonemes',
                'DC offset removal and normalization'
            ]
        },
        'usage': {
            'parameter': 'enhance_audio',
            'values': [True, False],
            'default': True,
            'description': 'Enhancement is enabled by default. Set enhance_audio to false to disable'
        },
        'examples': {
            'default_enhanced': {
                'text': 'سڵاو',
                'dialect': 'sorani',
                'speaker': 'speaker_0000'
                # enhance_audio defaults to True
            },
            'disable_enhancement': {
                'text': 'سڵاو',
                'dialect': 'sorani', 
                'speaker': 'speaker_0000',
                'enhance_audio': False
            }
        }
    })

@app.after_request
def cleanup_after_request(response):
    """Clean up resources after each request."""
    clear_model_cache()
    return response

def clear_model_cache():
    # Don't clear the cache, just run garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Only run the Flask development server if this file is run directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 