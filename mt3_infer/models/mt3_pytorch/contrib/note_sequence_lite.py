# Copyright 2024 MT3-Infer Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight NoteSequence implementation without protobuf dependency.

This replaces note_seq.NoteSequence for inference-only use cases,
avoiding the protobuf<4.0 dependency that conflicts with modern packages.
"""

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class Note:
    """A single MIDI note."""
    start_time: float = 0.0
    end_time: float = 0.0
    pitch: int = 0
    velocity: int = 100
    program: int = 0
    is_drum: bool = False
    instrument: int = 0


class NoteList:
    """List-like container for notes that mimics protobuf repeated field."""

    def __init__(self):
        self._notes: List[Note] = []

    def add(self, start_time: float = 0.0, end_time: float = 0.0,
            pitch: int = 0, velocity: int = 100, program: int = 0,
            is_drum: bool = False, instrument: int = 0) -> Note:
        """Add a new note and return it."""
        note = Note(
            start_time=start_time,
            end_time=end_time,
            pitch=pitch,
            velocity=velocity,
            program=program,
            is_drum=is_drum,
            instrument=instrument,
        )
        self._notes.append(note)
        return note

    def extend(self, notes):
        """Extend with iterable of notes."""
        self._notes.extend(notes)

    def __iter__(self):
        return iter(self._notes)

    def __len__(self):
        return len(self._notes)

    def __getitem__(self, idx):
        return self._notes[idx]

    def __delitem__(self, idx):
        del self._notes[idx]


class NoteSequence:
    """Lightweight NoteSequence compatible with note_seq.NoteSequence API."""

    def __init__(self, ticks_per_quarter: int = 220):
        self.ticks_per_quarter = ticks_per_quarter
        self.notes = NoteList()
        self.total_time: float = 0.0

    def CopyFrom(self, other: 'NoteSequence') -> None:
        """Copy from another NoteSequence."""
        self.ticks_per_quarter = other.ticks_per_quarter
        self.total_time = other.total_time
        self.notes = NoteList()
        for note in other.notes:
            self.notes.add(
                start_time=note.start_time,
                end_time=note.end_time,
                pitch=note.pitch,
                velocity=note.velocity,
                program=note.program,
                is_drum=note.is_drum,
                instrument=note.instrument,
            )
