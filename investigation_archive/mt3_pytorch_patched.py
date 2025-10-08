#!/usr/bin/env python3
"""
Patched MT3-PyTorch adapter with instrument leakage filtering.

This adapter includes post-processing to filter out incorrect instrument
assignments that occur when transcribing pure drum tracks.
"""

import sys
import numpy as np
from pathlib import Path
import mido
import tempfile

# Add to path for imports
sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter


class MT3PyTorchPatchedAdapter(MT3PyTorchAdapter):
    """Patched MT3-PyTorch adapter with automatic instrument filtering."""

    def __init__(self, filter_mode='auto'):
        """
        Initialize patched adapter.

        Args:
            filter_mode: 'auto', 'drums_only', 'smart', or None
        """
        super().__init__()
        self.filter_mode = filter_mode

    def decode(self, outputs: np.ndarray) -> mido.MidiFile:
        """
        Decode model outputs to MIDI with optional filtering.

        Args:
            outputs: Token predictions (batch, seq_len)

        Returns:
            MIDI file with filtered instruments
        """
        # Get base MIDI from parent
        midi = super().decode(outputs)

        # Apply filtering based on mode
        if self.filter_mode == 'auto':
            midi = self._auto_filter(midi)
        elif self.filter_mode == 'drums_only':
            midi = self._filter_drums_only(midi)
        elif self.filter_mode == 'smart':
            midi = self._smart_filter(midi)

        return midi

    def _auto_filter(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Automatically detect and filter based on content.

        If >50% of notes are drums, apply drums_only filter.
        """
        # Count drum vs non-drum notes
        drum_notes = 0
        total_notes = 0

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    total_notes += 1
                    if hasattr(msg, 'channel') and msg.channel == 9:
                        drum_notes += 1

        if total_notes == 0:
            return midi

        drum_ratio = drum_notes / total_notes

        # If mostly drums, filter out non-drums
        if drum_ratio > 0.5:
            print(f"Auto-filtering: {drum_ratio:.1%} drums detected, applying drums_only filter")
            return self._filter_drums_only(midi)

        return midi

    def _filter_drums_only(self, midi: mido.MidiFile) -> mido.MidiFile:
        """Keep only MIDI channel 10 (drums)."""
        filtered_mid = mido.MidiFile()
        filtered_mid.ticks_per_beat = midi.ticks_per_beat

        for track in midi.tracks:
            new_track = mido.MidiTrack()

            # Track if we're keeping any messages
            has_messages = False

            for msg in track:
                # Keep non-channel messages (tempo, time signature, etc.)
                if not hasattr(msg, 'channel'):
                    new_track.append(msg)
                    has_messages = True
                # Keep only drum channel (9 = channel 10 in MIDI)
                elif msg.channel == 9:
                    new_track.append(msg)
                    has_messages = True
                # Skip program changes for non-drum channels
                elif msg.type == 'program_change' and msg.channel != 9:
                    continue
                # Skip notes on non-drum channels
                elif msg.type in ['note_on', 'note_off'] and msg.channel != 9:
                    continue

            # Only add track if it has messages
            if has_messages:
                filtered_mid.tracks.append(new_track)

        return filtered_mid

    def _smart_filter(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Context-aware filtering based on musical patterns.

        This is a more sophisticated filter that considers:
        - Velocity patterns
        - Temporal clustering
        - Pitch ranges
        """
        # For now, just use drums_only as placeholder
        # Full implementation would analyze patterns
        return self._filter_drums_only(midi)


def test_patched_adapter():
    """Test the patched adapter."""
    from mt3_infer.utils.audio import load_audio

    print("Testing MT3-PyTorch Patched Adapter")
    print("="*60)

    # Load model
    adapter = MT3PyTorchPatchedAdapter(filter_mode='auto')
    checkpoint_path = "/home/worzpro/Desktop/dev/patched_modules/mt3-infer/.mt3_checkpoints/mt3_pytorch"
    adapter.load_model(checkpoint_path)

    # Test with drum audio
    audio_file = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(audio_file, sr=16000)

    # Transcribe
    print(f"\nTranscribing: {audio_file}")
    midi = adapter.transcribe(audio, sr)

    # Analyze result
    drum_notes = 0
    non_drum_notes = 0
    instruments = {}

    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if hasattr(msg, 'channel'):
                    if msg.channel == 9:
                        drum_notes += 1
                    else:
                        non_drum_notes += 1
                        # Track instrument
                        instruments[msg.channel] = instruments.get(msg.channel, 0) + 1

    total_notes = drum_notes + non_drum_notes

    print(f"\nResults after filtering:")
    print(f"- Total notes: {total_notes}")
    print(f"- Drum notes: {drum_notes} ({drum_notes/total_notes*100:.1f}%)")
    print(f"- Non-drum notes: {non_drum_notes} ({non_drum_notes/total_notes*100:.1f}%)")

    if non_drum_notes > 0:
        print(f"\n⚠️ Still have {non_drum_notes} non-drum notes after filtering!")
        print(f"Instruments: {instruments}")
    else:
        print(f"\n✅ Successfully filtered all non-drum instruments!")

    # Save filtered MIDI
    output_file = "transcription_filtered_patched.mid"
    midi.save(output_file)
    print(f"\nSaved filtered MIDI to: {output_file}")

    return drum_notes, non_drum_notes


if __name__ == "__main__":
    test_patched_adapter()