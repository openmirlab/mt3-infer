"""
Smoke tests for MIDI utilities.
"""

import mido
import pytest

from mt3_infer.utils.midi import count_notes, get_midi_duration, midi_to_hash, validate_midi


def create_simple_midi():
    """Helper to create a simple valid MIDI file."""
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    return midi


def test_validate_midi_success():
    """Test validate_midi with valid MIDI."""
    midi = create_simple_midi()
    is_valid, msg = validate_midi(midi)

    assert is_valid is True
    assert msg == ""


def test_validate_midi_rejects_no_tracks():
    """Test validate_midi rejects MIDI with no tracks."""
    midi = mido.MidiFile()
    is_valid, msg = validate_midi(midi)

    assert is_valid is False
    assert "no tracks" in msg


def test_validate_midi_rejects_negative_time():
    """Test validate_midi rejects negative time deltas."""
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=64, time=-10))
    midi.tracks.append(track)

    is_valid, msg = validate_midi(midi)

    assert is_valid is False
    assert "negative time" in msg


def test_validate_midi_warns_on_unclosed_notes():
    """Test validate_midi warns on unclosed notes."""
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    # Note on without note off
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    midi.tracks.append(track)

    with pytest.warns(UserWarning, match="unclosed notes"):
        is_valid, msg = validate_midi(midi)
        assert is_valid is True  # Warning, not error


def test_midi_to_hash_deterministic():
    """Test midi_to_hash produces consistent hashes."""
    midi = create_simple_midi()

    hash1 = midi_to_hash(midi)
    hash2 = midi_to_hash(midi)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex length


def test_midi_to_hash_different_for_different_midi():
    """Test midi_to_hash produces different hashes for different MIDI."""
    midi1 = create_simple_midi()

    midi2 = mido.MidiFile()
    track2 = mido.MidiTrack()
    track2.append(mido.Message("note_on", note=62, velocity=64, time=0))
    track2.append(mido.Message("note_off", note=62, velocity=0, time=480))
    midi2.tracks.append(track2)

    hash1 = midi_to_hash(midi1)
    hash2 = midi_to_hash(midi2)

    assert hash1 != hash2


def test_count_notes():
    """Test count_notes counts note_on events."""
    midi = create_simple_midi()
    count = count_notes(midi)

    assert count == 1


def test_count_notes_excludes_velocity_zero():
    """Test count_notes excludes note_on with velocity=0."""
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_on", note=60, velocity=0, time=100))  # Equivalent to note_off
    midi.tracks.append(track)

    count = count_notes(midi)
    assert count == 1


def test_get_midi_duration():
    """Test get_midi_duration returns duration in seconds."""
    midi = create_simple_midi()
    duration = get_midi_duration(midi)

    assert isinstance(duration, float)
    assert duration > 0
