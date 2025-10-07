"""
Audio processing utilities for MT3-Infer.

Provides functions for loading, validating, and preprocessing audio data.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from mt3_infer.exceptions import AudioError


def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file (WAV, FLAC, MP3, etc.).
        sr: Target sample rate in Hz. Default is 16000.

    Returns:
        Tuple of (audio, sample_rate):
        - audio: Float32 array with shape (n_samples,).
        - sample_rate: Actual sample rate (should match `sr` after resampling).

    Raises:
        AudioError: File not found, unsupported format, or loading failed.
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise AudioError(
            f"Audio file not found: {path}\n"
            f"Check that the path is correct and the file exists."
        )

    target_sr = sr
    torchaudio_audio = None
    original_sr = target_sr

    try:
        import torchaudio

        waveform, original_sr = torchaudio.load(str(path_obj))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if original_sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, original_sr, target_sr)
            original_sr = target_sr

        torchaudio_audio = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
    except Exception:
        torchaudio_audio = None

    if torchaudio_audio is None:
        try:
            audio, original_sr = sf.read(path, dtype="float32")
        except Exception as e:
            raise AudioError(
                f"Failed to load audio from {path}: {e}\n"
                f"Ensure the file is a valid audio format (WAV, FLAC, MP3, etc.)"
            ) from e

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1).astype(np.float32)

        if original_sr != target_sr:
            try:
                from scipy import signal
                num_samples = int(len(audio) * target_sr / original_sr)
                audio = signal.resample(audio, num_samples).astype(np.float32)
                original_sr = target_sr
            except ImportError:
                import warnings

                warnings.warn(
                    f"scipy not available for resampling. "
                    f"Audio will use original sample rate {original_sr}Hz instead of {target_sr}Hz. "
                    f"Install scipy for resampling: uv add scipy",
                    UserWarning
                )
            except Exception as e:
                raise AudioError(
                    f"Resampling failed (from {original_sr}Hz to {target_sr}Hz): {e}"
                ) from e

        audio_out = audio.astype(np.float32)
        sr_out = original_sr
    else:
        audio_out = torchaudio_audio
        sr_out = target_sr

    if audio_out.size == 0:
        raise AudioError(
            f"Loaded audio from {path} is empty (0 samples)\n"
            f"File may be corrupted or contain no audio data"
        )

    return audio_out, sr_out


def validate_audio(audio: np.ndarray) -> Tuple[bool, str]:
    """
    Validate audio array format.

    Args:
        audio: Audio array to validate.

    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if audio is valid, False otherwise.
        - error_message: Description of validation error (empty if valid).

    Example:
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> is_valid, msg = validate_audio(audio)
        >>> print(is_valid, msg)
        True ""
    """
    if not isinstance(audio, np.ndarray):
        return False, f"audio must be numpy.ndarray, got {type(audio).__name__}"

    if audio.ndim not in (1, 2):
        return False, f"audio must be 1D or 2D, got shape {audio.shape}"

    if audio.dtype not in (np.float32, np.float64):
        return False, f"audio dtype must be float32 or float64, got {audio.dtype}"

    if audio.size == 0:
        return False, "audio array is empty"

    audio_max = np.abs(audio).max()
    if audio_max > 10.0:
        return False, f"audio values suspiciously large (max={audio_max:.3f}), expected roughly [-1.0, 1.0]"

    return True, ""


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1.0, 1.0] range.

    Args:
        audio: Audio array to normalize.

    Returns:
        Normalized audio array (same dtype as input).

    Example:
        >>> audio = np.array([0.5, 1.0, -2.0], dtype=np.float32)
        >>> normalized = normalize_audio(audio)
        >>> print(normalized)
        [ 0.25  0.5  -1.  ]
    """
    if audio.size == 0:
        return audio  # Return empty array as-is

    audio_max = np.abs(audio).max()
    if audio_max == 0:
        return audio
    return audio / audio_max


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio array to resample.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Resampled audio array (float32).

    Raises:
        AudioError: Resampling failed.

    Note:
        Requires scipy for resampling. If not available, raises AudioError.

    Example:
        >>> audio_8k = np.random.randn(8000).astype(np.float32)
        >>> audio_16k = resample_audio(audio_8k, orig_sr=8000, target_sr=16000)
        >>> print(audio_16k.shape)
        (16000,)
    """
    if orig_sr == target_sr:
        return audio

    try:
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples).astype(np.float32)
        return resampled
    except ImportError as e:
        raise AudioError(
            "scipy is required for audio resampling.\n"
            "Install with: uv add scipy"
        ) from e
    except Exception as e:
        raise AudioError(
            f"Resampling failed (from {orig_sr}Hz to {target_sr}Hz): {e}"
        ) from e
