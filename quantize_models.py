#!/usr/bin/env python3
"""
Script to quantize TTS models to reduce size while preserving quality.
This will convert the large models (850MB) to smaller quantized versions.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import logging
import sys
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quantize_model(model_path, config_path, output_path, quantization_type='int8'):
    """
    Quantize a TTS model to reduce its size.
    
    Args:
        model_path: Path to the original model checkpoint
        config_path: Path to the model config file
        output_path: Path to save the quantized model
        quantization_type: Type of quantization ('int8', 'fp16', or 'dynamic')
    """
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load config
        config = VitsConfig()
        config.load_json(str(config_path))
        
        # Initialize components
        ap = AudioProcessor.init_from_config(config)
        tokenizer, config = TTSTokenizer.init_from_config(config)
        
        # Initialize speaker manager
        from TTS.tts.utils.speakers import SpeakerManager
        speaker_manager = SpeakerManager()
        
        # Load checkpoint
        checkpoint = torch.load(str(model_path), map_location='cpu')
        
        # Get speaker count from checkpoint
        if 'model' in checkpoint and 'emb_g.weight' in checkpoint['model']:
            num_speakers = checkpoint['model']['emb_g.weight'].shape[0]
            logger.info(f"Found {num_speakers} speakers in checkpoint")
            
            # Create speaker names
            speaker_names = [f"speaker_{i:04d}" for i in range(num_speakers)]
            speaker_manager.name_to_id = {name: i for i, name in enumerate(speaker_names)}
            speaker_manager.id_to_name = {i: name for i, name in enumerate(speaker_names)}
            config.num_speakers = num_speakers
        else:
            logger.warning("Could not determine speaker count from checkpoint, using default")
            speaker_names = ["speaker_0000", "speaker_0001"]
            speaker_manager.name_to_id = {name: i for i, name in enumerate(speaker_names)}
            speaker_manager.id_to_name = {i: name for i, name in enumerate(speaker_names)}
            config.num_speakers = 2
        
        # Create model
        model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
        
        # Load model state
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        
        # Get original model size
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"Original model size: {original_size:.2f} MB")
        
        # Quantize the model
        logger.info(f"Applying {quantization_type} quantization...")
        
        if quantization_type == 'int8':
            # Use PyTorch's dynamic quantization for int8
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d}, 
                dtype=torch.qint8
            )
            
        elif quantization_type == 'fp16':
            # Use half precision (FP16)
            quantized_model = model.half()
            
        elif quantization_type == 'dynamic':
            # Use dynamic quantization with int8
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d},
                dtype=torch.qint8
            )
            
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        # Save quantized model
        logger.info(f"Saving quantized model to {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the quantized model
        torch.save(quantized_model.state_dict(), output_path)
        
        # Get quantized model size
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Error quantizing model: {str(e)}")
        return False

def main():
    """Main function to quantize both Sorani and Kurmanji models."""
    
    # Model configurations
    models_to_quantize = [
        {
            'name': 'Sorani',
            'model_path': 'models/sorani/checkpoint_95000.pth',
            'config_path': 'models/sorani/yourtts_improved_config_fixed.json',
            'output_path': 'models/sorani/checkpoint_95000_quantized.pth'
        },
        {
            'name': 'Kurmanji',
            'model_path': 'models/kurmanji/checkpoint_80000.pth',
            'config_path': 'models/kurmanji/kurmanji_config.json',
            'output_path': 'models/kurmanji/checkpoint_80000_quantized.pth'
        }
    ]
    
    # Quantization types to try (in order of preference)
    quantization_types = ['int8', 'fp16', 'dynamic']
    
    for model_config in models_to_quantize:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {model_config['name']} model")
        logger.info(f"{'='*50}")
        
        # Check if input files exist
        if not os.path.exists(model_config['model_path']):
            logger.error(f"Model file not found: {model_config['model_path']}")
            continue
            
        if not os.path.exists(model_config['config_path']):
            logger.error(f"Config file not found: {model_config['config_path']}")
            continue
        
        # Try different quantization types
        success = False
        for q_type in quantization_types:
            logger.info(f"\nTrying {q_type} quantization...")
            
            try:
                success = quantize_model(
                    model_config['model_path'],
                    model_config['config_path'],
                    model_config['output_path'],
                    q_type
                )
                
                if success:
                    logger.info(f"✅ Successfully quantized {model_config['name']} model with {q_type}")
                    break
                else:
                    logger.warning(f"❌ {q_type} quantization failed for {model_config['name']}")
                    
            except Exception as e:
                logger.error(f"❌ Error with {q_type} quantization: {str(e)}")
                continue
        
        if not success:
            logger.error(f"❌ Failed to quantize {model_config['name']} model with any quantization type")
        else:
            logger.info(f"✅ {model_config['name']} model quantization completed successfully!")
    
    logger.info(f"\n{'='*50}")
    logger.info("Quantization process completed!")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main() 