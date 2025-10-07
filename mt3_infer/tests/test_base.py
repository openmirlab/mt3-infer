"""
Smoke tests for MT3Base abstract class and validation logic.
"""

import numpy as np
import pytest

from mt3_infer.base import MT3Base


class DummyAdapter(MT3Base):
    """Minimal MT3Base implementation for testing."""

    def load_model(self, checkpoint_path: str, device: str = "auto") -> None:
        self._model_loaded = True

    def preprocess(self, audio: np.ndarray, sr: int):
        return audio  # Pass through

    def forward(self, features):
        return features  # Pass through

    def decode(self, outputs):
        import mido
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", note=60, velocity=64, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=100))
        midi.tracks.append(track)
        return midi


def test_mt3base_is_abstract():
    """Test that MT3Base cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MT3Base()


def test_transcribe_requires_model_loaded():
    """Test that transcribe raises if model not loaded."""
    adapter = DummyAdapter()
    audio = np.random.randn(1000).astype(np.float32)

    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.transcribe(audio, sr=16000)


def test_transcribe_validates_audio_type():
    """Test that transcribe rejects non-numpy arrays."""
    adapter = DummyAdapter()
    adapter.load_model("dummy.pth")

    with pytest.raises(ValueError, match="must be numpy.ndarray"):
        adapter.transcribe([1, 2, 3], sr=16000)


def test_transcribe_validates_audio_dtype():
    """Test that transcribe rejects invalid dtypes."""
    adapter = DummyAdapter()
    adapter.load_model("dummy.pth")

    audio_int = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        adapter.transcribe(audio_int, sr=16000)


def test_transcribe_handles_empty_audio():
    """Test that transcribe raises on empty audio array."""
    adapter = DummyAdapter()
    adapter.load_model("dummy.pth")

    audio_empty = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="audio array is empty"):
        adapter.transcribe(audio_empty, sr=16000)


def test_transcribe_converts_stereo_to_mono():
    """Test that transcribe converts stereo to mono."""
    adapter = DummyAdapter()
    adapter.load_model("dummy.pth")

    # Stereo audio (n_samples, 2)
    audio_stereo = np.random.randn(1000, 2).astype(np.float32)
    midi = adapter.transcribe(audio_stereo, sr=16000)

    assert midi is not None
    assert len(midi.tracks) > 0


def test_transcribe_warns_on_large_values():
    """Test that transcribe warns if audio exceeds [-1, 1]."""
    adapter = DummyAdapter()
    adapter.load_model("dummy.pth")

    audio_large = np.array([0.5, 2.0, -3.0], dtype=np.float32)

    with pytest.warns(UserWarning, match="exceed"):
        adapter.transcribe(audio_large, sr=16000)


def test_transcribe_validates_sample_rate():
    """Test that transcribe validates sample rate."""
    adapter = DummyAdapter()
    adapter.load_model("dummy.pth")

    audio = np.random.randn(1000).astype(np.float32)

    with pytest.raises(ValueError, match="sample rate must be positive integer"):
        adapter.transcribe(audio, sr=-16000)

    with pytest.raises(ValueError, match="sample rate must be positive integer"):
        adapter.transcribe(audio, sr=0)


def test_transcribe_success():
    """Test successful transcription pipeline."""
    adapter = DummyAdapter()
    adapter.load_model("dummy.pth")

    audio = np.random.randn(16000).astype(np.float32)
    midi = adapter.transcribe(audio, sr=16000)

    assert midi is not None
    import mido
    assert isinstance(midi, mido.MidiFile)
    assert len(midi.tracks) > 0


def test_adapter_repr():
    """Test adapter string representation."""
    adapter = DummyAdapter()

    # Before loading
    repr_before = repr(adapter)
    assert "not loaded" in repr_before

    # After loading
    adapter.load_model("dummy.pth")
    repr_after = repr(adapter)
    assert "loaded" in repr_after
