# Copyright 2024 MT3-Infer Contributors
# SPDX-License-Identifier: Apache-2.0
"""
MIDI constants to replace note_seq dependency.

These are standard MIDI specification values that don't require note_seq.
"""

# MIDI pitch range (0-127)
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127

# MIDI velocity range (0-127)
MIN_MIDI_VELOCITY = 0
MAX_MIDI_VELOCITY = 127

# MIDI program (instrument) range (0-127)
MIN_MIDI_PROGRAM = 0
MAX_MIDI_PROGRAM = 127

# MIDI channel range (0-15)
MIN_MIDI_CHANNEL = 0
MAX_MIDI_CHANNEL = 15

# Drum channel (General MIDI standard)
DRUM_CHANNEL = 9
