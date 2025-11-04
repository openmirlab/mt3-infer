"""
Inference-only model loader for YourMT3.

This module provides a simplified loader that bypasses the training infrastructure
(initialize_trainer, update_config, etc.) and directly loads PyTorch Lightning
checkpoints for inference.
"""

import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from mt3_infer.models.yourmt3.model.ymt3 import YourMT3
from mt3_infer.models.yourmt3.utils.task_manager import TaskManager

# Module aliasing for pickle unpickling compatibility
# YourMT3 checkpoints were saved with old module paths, so we create aliases
import mt3_infer.models.yourmt3.utils
import mt3_infer.models.yourmt3.model
import mt3_infer.models.yourmt3.config
import mt3_infer.models.yourmt3.config.vocabulary
import mt3_infer.models.yourmt3.utils.task_manager

# Temporarily add module aliases for checkpoint loading
sys.modules['utils'] = mt3_infer.models.yourmt3.utils
sys.modules['model'] = mt3_infer.models.yourmt3.model
sys.modules['config'] = mt3_infer.models.yourmt3.config
sys.modules['config.vocabulary'] = mt3_infer.models.yourmt3.config.vocabulary
sys.modules['utils.task_manager'] = mt3_infer.models.yourmt3.utils.task_manager


def load_model_for_inference(
    checkpoint_path: str,
    device: str = 'cpu',
    task_name: str = 'mt3_full_plus'
) -> YourMT3:
    """
    Load YourMT3 model from checkpoint for inference only.

    This is a simplified version of load_model_checkpoint() from model_helper.py
    that doesn't require the full training infrastructure.

    Args:
        checkpoint_path: Path to .ckpt checkpoint file
        device: Device to load model on ('cpu', 'cuda', or 'auto')
        task_name: Task configuration name (default: 'mt3_full_plus')

    Returns:
        YourMT3 model loaded and ready for inference
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    # Note: Module aliases are set at module level to handle pickle unpickling
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"Loaded hyperparameters from checkpoint")
    else:
        raise ValueError("Checkpoint does not contain 'hyper_parameters' field")

    # Initialize task manager
    task_manager = TaskManager(task_name=task_name)

    # Create model with hyperparameters from checkpoint
    model = YourMT3(
        audio_cfg=hparams.get('audio_cfg'),
        model_cfg=hparams.get('model_cfg'),
        shared_cfg=hparams.get('shared_cfg'),
        task_manager=task_manager,
        eval_vocab=hparams.get('eval_vocab'),
        eval_drum_vocab=hparams.get('eval_drum_vocab'),
        eval_subtask_key=hparams.get('eval_subtask_key', 'default'),
        onset_tolerance=hparams.get('onset_tolerance', 0.05),
        write_output_vocab=hparams.get('write_output_vocab'),  # For MIDI output
        # Inference-only, so we don't need optimizer/scheduler params
    )

    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model state dict")
    else:
        raise ValueError("Checkpoint does not contain 'state_dict' field")

    # Initialize MIDI output vocab if not already set (needed for inference)
    if not hasattr(model, 'midi_output_inverse_vocab'):
        from config.vocabulary import program_vocab_presets
        from utils.utils import create_inverse_vocab

        # Use default GM extended plus vocabulary
        model.midi_output_vocab = program_vocab_presets["gm_ext_plus"]
        model.midi_output_inverse_vocab = create_inverse_vocab(model.midi_output_vocab)
        print(f"Initialized MIDI output vocabulary")

    # Move to device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model
