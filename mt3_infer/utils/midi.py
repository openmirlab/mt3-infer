"""
MIDI processing utilities for MT3-Infer.

Provides functions for validating and hashing MIDI files.
"""

import hashlib
from typing import Tuple

import mido


def midi_to_hash(midi: mido.MidiFile) -> str:
    """
    Generate SHA256 hash of MIDI file for regression testing.

    Args:
        midi: MIDI file object to hash.

    Returns:
        Hexadecimal SHA256 hash string.

    Note:
        This is used for regression tests to ensure adapter outputs
        remain consistent across versions.

    Example:
        >>> midi = mido.MidiFile()
        >>> midi.tracks.append(mido.MidiTrack())
        >>> hash1 = midi_to_hash(midi)
        >>> hash2 = midi_to_hash(midi)
        >>> assert hash1 == hash2  # Deterministic
    """
    # Serialize MIDI to bytes
    from io import BytesIO
    buffer = BytesIO()
    midi.save(file=buffer)
    midi_bytes = buffer.getvalue()

    # Compute SHA256 hash
    return hashlib.sha256(midi_bytes).hexdigest()


def validate_midi(midi: mido.MidiFile) -> Tuple[bool, str]:
    """
    Validate MIDI file structure and contents.

    Args:
        midi: MIDI file object to validate.

    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if MIDI is valid, False otherwise.
        - error_message: Description of validation error (empty if valid).

    Checks:
    - At least one track exists
    - Note on/off events are properly paired
    - Timestamps are monotonically increasing
    - Velocities are in [0, 127]

    Example:
        >>> midi = mido.MidiFile()
        >>> midi.tracks.append(mido.MidiTrack())
        >>> is_valid, msg = validate_midi(midi)
        >>> print(is_valid)
        True
    """
    # Check that at least one track exists
    if len(midi.tracks) == 0:
        return False, "MIDI file has no tracks"

    for track_idx, track in enumerate(midi.tracks):
        # Track note on/off pairing
        active_notes = {}  # note_number -> timestamp

        # Track absolute time for monotonicity check
        abs_time = 0

        for msg in track:
            abs_time += msg.time

            # Check time is non-negative
            if msg.time < 0:
                return False, f"Track {track_idx}: negative time delta {msg.time}"

            # Check note events
            if msg.type == "note_on":
                if msg.velocity == 0:
                    # Note on with velocity 0 is equivalent to note off
                    if msg.note in active_notes:
                        del active_notes[msg.note]
                else:
                    # Validate velocity range
                    if not (0 <= msg.velocity <= 127):
                        return False, f"Track {track_idx}: invalid velocity {msg.velocity}, must be in [0, 127]"
                    active_notes[msg.note] = abs_time

            elif msg.type == "note_off":
                if msg.note in active_notes:
                    del active_notes[msg.note]
                # Note: It's OK to have note_off without prior note_on (some MIDIs do this)

        # Warn if there are unclosed notes (but don't fail validation)
        if active_notes:
            import warnings
            warnings.warn(
                f"Track {track_idx}: {len(active_notes)} unclosed notes. "
                f"This may indicate improper MIDI generation.",
                UserWarning
            )

    return True, ""


def count_notes(midi: mido.MidiFile) -> int:
    """
    Count total number of note events in MIDI file.

    Args:
        midi: MIDI file object.

    Returns:
        Total number of note_on events (excluding note_on with velocity=0).

    Example:
        >>> midi = mido.MidiFile()
        >>> track = mido.MidiTrack()
        >>> track.append(mido.Message("note_on", note=60, velocity=64, time=0))
        >>> track.append(mido.Message("note_off", note=60, velocity=0, time=100))
        >>> midi.tracks.append(track)
        >>> count_notes(midi)
        1
    """
    note_count = 0
    for track in midi.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                note_count += 1
    return note_count


def get_midi_duration(midi: mido.MidiFile) -> float:
    """
    Get total duration of MIDI file in seconds.

    Args:
        midi: MIDI file object.

    Returns:
        Duration in seconds.

    Example:
        >>> midi = mido.MidiFile()
        >>> track = mido.MidiTrack()
        >>> track.append(mido.Message("note_on", note=60, velocity=64, time=0))
        >>> track.append(mido.Message("note_off", note=60, velocity=0, time=480))
        >>> midi.tracks.append(track)
        >>> duration = get_midi_duration(midi)
        >>> print(f"{duration:.2f}s")
    """
    return midi.length


def midi_to_audio(
    midi: mido.MidiFile,
    sr: int = 44100,
    soundfont_path: str = None
) -> "np.ndarray":
    """
    Synthesize MIDI to audio for quality checking.

    This is a simple synthesis function to help users hear the transcription output.
    Uses fluidsynth (preferred) or midi2audio (Python fallback) for synthesis.

    Args:
        midi: MIDI file object to synthesize.
        sr: Sample rate for output audio. Default is 44100 Hz.
        soundfont_path: Path to custom soundfont file (optional).
                       If None, uses default system soundfont.

    Returns:
        Audio waveform as numpy array, shape (n_samples,), dtype float32.
        Values are in range [-1.0, 1.0].

    Raises:
        ImportError: If neither fluidsynth nor midi2audio is available.
        RuntimeError: If synthesis fails.

    Note:
        This is a basic synthesis for quick quality checking. For professional
        audio rendering, use a DAW or dedicated MIDI synthesizer.

        Synthesis methods (in order of preference):
        1. fluidsynth (best quality, requires system install)
        2. midi2audio (Python fallback, still requires fluidsynth)

        Installation options:
        - Option 1 (best quality): sudo apt-get install fluidsynth fluid-soundfont-gm
        - Option 2 (Python only): uv add midi2audio (still needs fluidsynth installed)

    Example:
        >>> from mt3_infer import transcribe
        >>> from mt3_infer.utils.midi import midi_to_audio
        >>> import soundfile as sf
        >>>
        >>> # Transcribe audio
        >>> midi = transcribe(audio, model="mt3_pytorch")
        >>>
        >>> # Synthesize to audio for listening
        >>> synth_audio = midi_to_audio(midi, sr=44100)
        >>> sf.write("transcription_preview.wav", synth_audio, 44100)
    """
    import numpy as np

    # Try method 1: Direct fluidsynth (best quality)
    fluidsynth_available = _check_fluidsynth_available()

    if fluidsynth_available:
        return _synthesize_with_fluidsynth(midi, sr, soundfont_path)

    # Try method 2: Python midi2audio fallback
    try:
        return _synthesize_with_midi2audio(midi, sr, soundfont_path)
    except ImportError:
        pass

    # Neither method available
    raise ImportError(
        "MIDI synthesis requires fluidsynth.\n\n"
        "Installation options:\n"
        "  Option 1 (Recommended - best quality):\n"
        "    Ubuntu/Debian: sudo apt-get install fluidsynth fluid-soundfont-gm\n"
        "    macOS: brew install fluid-synth\n"
        "    Windows: Download from https://github.com/FluidSynth/fluidsynth/releases\n\n"
        "  Option 2 (Python wrapper):\n"
        "    uv add midi2audio\n"
        "    (Note: Still requires fluidsynth to be installed system-wide)"
    )


def _check_fluidsynth_available() -> bool:
    """Check if fluidsynth binary is available."""
    import subprocess
    try:
        subprocess.run(
            ["fluidsynth", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _synthesize_with_fluidsynth(
    midi: mido.MidiFile,
    sr: int,
    soundfont_path: str = None
) -> "np.ndarray":
    """Synthesize using direct fluidsynth command (best quality)."""
    import tempfile
    import subprocess
    import os
    import numpy as np

    # Find a soundfont if not provided
    if soundfont_path is None:
        soundfont_path = _find_soundfont()
        if soundfont_path is None:
            raise RuntimeError(
                "No soundfont found. Please provide soundfont_path.\n"
                "Install with: sudo apt-get install fluid-soundfont-gm\n"
                "Or download from: https://github.com/FluidSynth/fluidsynth/wiki/SoundFont"
            )

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_file:
        midi.save(midi_file.name)
        midi_path = midi_file.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = audio_file.name

    try:
        # Run fluidsynth
        cmd = [
            "fluidsynth",
            "-ni",  # No interactive mode
            soundfont_path,
            midi_path,
            "-F", audio_path,  # Output file
            "-r", str(sr),  # Sample rate
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"fluidsynth failed: {result.stderr}")

        # Load synthesized audio
        audio = _load_audio_file(audio_path)

        return audio

    finally:
        # Cleanup temp files
        if os.path.exists(midi_path):
            os.unlink(midi_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def _synthesize_with_midi2audio(
    midi: mido.MidiFile,
    sr: int,
    soundfont_path: str = None
) -> "np.ndarray":
    """Synthesize using midi2audio Python library (fallback)."""
    import tempfile
    import os

    try:
        from midi2audio import FluidSynth
    except ImportError:
        raise ImportError(
            "midi2audio not installed.\n"
            "Install with: uv add midi2audio"
        )

    # Find soundfont if not provided
    if soundfont_path is None:
        soundfont_path = _find_soundfont()

    # Create FluidSynth instance
    if soundfont_path:
        fs = FluidSynth(sound_font=soundfont_path, sample_rate=sr)
    else:
        fs = FluidSynth(sample_rate=sr)

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_file:
        midi.save(midi_file.name)
        midi_path = midi_file.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = audio_file.name

    try:
        # Synthesize
        fs.midi_to_audio(midi_path, audio_path)

        # Load synthesized audio
        audio = _load_audio_file(audio_path)

        return audio

    finally:
        # Cleanup
        if os.path.exists(midi_path):
            os.unlink(midi_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def _find_soundfont() -> str:
    """Find system soundfont file."""
    import os

    soundfont_paths = [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
        "/usr/share/soundfonts/default.sf2",
        "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls",
    ]

    for path in soundfont_paths:
        if os.path.exists(path):
            return path

    return None


def _load_audio_file(audio_path: str) -> "np.ndarray":
    """Load audio file and normalize."""
    import numpy as np

    try:
        import soundfile as sf
        audio, _ = sf.read(audio_path, dtype='float32')
    except ImportError:
        # Fallback to scipy if soundfile not available
        from scipy.io import wavfile
        _, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 32768.0  # Convert int16 to float32

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalize to [-1, 1]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val

    return audio
