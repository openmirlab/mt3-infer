"""
MT3Base - Abstract base class for all MT3 model adapters.

Defines the standard inference lifecycle interface that all framework-specific
adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import mido
import numpy as np


class MT3Base(ABC):
    """
    Abstract base class for all MT3 model adapters.

    This class defines the standard inference lifecycle:
    1. load_model() - Load model weights from checkpoint
    2. preprocess() - Convert raw audio to model input format
    3. forward() - Run model inference
    4. decode() - Convert model outputs to MIDI
    5. transcribe() - End-to-end pipeline (DO NOT OVERRIDE)

    Adapters must implement methods 1-4; method 5 is final.
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._model_loaded = False

    @abstractmethod
    def load_model(self, checkpoint_path: str, device: str = "auto") -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint file.
            device: Target device ("cuda", "cpu", "auto").
                   "auto" will use CUDA if available, otherwise CPU.

        Raises:
            FileNotFoundError: Checkpoint file does not exist.
            RuntimeError: Model loading failed.

        Note:
            Implementations should set self._model_loaded = True on success.
        """
        pass

    @abstractmethod
    def preprocess(self, audio: np.ndarray, sr: int) -> Any:
        """
        Convert raw audio to model input format.

        Args:
            audio: Audio waveform with shape (n_samples,), dtype float32.
                  Values should be in range [-1.0, 1.0].
            sr: Sample rate in Hz (typically 16000).

        Returns:
            Model-specific input tensor/array (PyTorch Tensor, TF Tensor, JAX Array).

        Raises:
            ValueError: Invalid audio format or sample rate.

        Note:
            Implementations may resample audio if sr doesn't match model requirements.
        """
        pass

    @abstractmethod
    def forward(self, features: Any) -> Any:
        """
        Run model inference.

        Args:
            features: Preprocessed model input from preprocess().

        Returns:
            Model-specific output representation (logits, tokens, etc.).

        Raises:
            RuntimeError: Inference failed (OOM, device error, etc.).

        Note:
            Implementations should use no-gradient context (torch.inference_mode,
            tf.function, jax.jit) for performance.
        """
        pass

    @abstractmethod
    def decode(self, outputs: Any) -> mido.MidiFile:
        """
        Convert model outputs to MIDI.

        Args:
            outputs: Model output from forward().

        Returns:
            MIDI file object with transcribed notes.

        Raises:
            ValueError: Output format is invalid or cannot be decoded.

        Note:
            Implementations should ensure:
            - At least one track exists
            - Note on/off events are properly paired
            - Timestamps are monotonically increasing
            - Velocities are in [0, 127]
        """
        pass

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> mido.MidiFile:
        """
        End-to-end transcription pipeline.

        This method MUST NOT be overridden by adapters. It provides a consistent
        interface across all MT3 implementations.

        Args:
            audio: Audio waveform with shape (n_samples,) or (n_samples, n_channels).
                  dtype must be float32 or float64.
                  Values should be in range [-1.0, 1.0].
            sr: Sample rate in Hz. Default is 16000.

        Returns:
            MIDI file object with transcribed notes.

        Raises:
            ValueError: Invalid inputs (audio format, sample rate, etc.).
            RuntimeError: Transcription failed (model not loaded, inference error, etc.).

        Example:
            >>> import numpy as np
            >>> from mt3_infer import load_model
            >>> model = load_model("mr_mt3")
            >>> audio = np.random.randn(16000 * 10).astype(np.float32)
            >>> midi = model.transcribe(audio, sr=16000)
            >>> midi.save("output.mid")
        """
        # Validate model is loaded
        if not self._model_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() before transcription.\n"
                "Example: model.load_model('/path/to/checkpoint.pth')"
            )

        # Validate audio format
        if not isinstance(audio, np.ndarray):
            raise ValueError(
                f"audio must be numpy.ndarray, got {type(audio).__name__}"
            )

        if audio.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"audio dtype must be float32 or float64, got {audio.dtype}\n"
                f"Convert with: audio = audio.astype(np.float32)"
            )

        # Handle multi-channel audio (convert to mono)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1).astype(audio.dtype)
        elif audio.ndim != 1:
            raise ValueError(
                f"audio must be 1D or 2D array, got shape {audio.shape}\n"
                f"Expected shape: (n_samples,) or (n_samples, n_channels)"
            )

        # Check for empty audio
        if audio.size == 0:
            raise ValueError(
                "audio array is empty (0 samples)\n"
                "Provide valid audio data with at least 1 sample"
            )

        # Warn if audio values exceed expected range
        audio_max = np.abs(audio).max()
        if audio_max > 1.0:
            import warnings
            warnings.warn(
                f"audio values exceed [-1.0, 1.0] range (max={audio_max:.3f}). "
                f"Consider normalizing: audio = audio / np.abs(audio).max()",
                UserWarning
            )

        # Validate sample rate
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError(
                f"sample rate must be positive integer, got {sr}"
            )

        # Run inference pipeline
        try:
            features = self.preprocess(audio, sr)
            outputs = self.forward(features)
            midi = self.decode(outputs)
            return midi
        except Exception as e:
            # Wrap low-level exceptions with context
            raise RuntimeError(
                f"Transcription failed during {e.__class__.__name__}: {e}\n"
                f"Audio shape: {audio.shape}, Sample rate: {sr}Hz"
            ) from e

    def __repr__(self) -> str:
        """String representation of the adapter."""
        model_status = "loaded" if self._model_loaded else "not loaded"
        return f"{self.__class__.__name__}(model_status={model_status})"
