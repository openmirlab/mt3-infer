"""
Vocabulary and Codec utilities for MT3 MIDI decoding.

Extracted from MR-MT3 contrib/ modules to avoid TensorFlow dependencies.

Original sources:
- contrib/event_codec.py (event encoding/decoding)
- contrib/run_length_encoding.py (token sequence decoding)
- contrib/note_sequences.py (MIDI note state machine)
- contrib/vocabularies.py (vocabulary config, minimal extraction)

License: Apache 2.0 (from MT3 Authors)
"""

import dataclasses
import math
from typing import Any, Callable, List, Mapping, MutableMapping, MutableSet, Optional, Sequence, Tuple

import numpy as np


# Constants from vocabularies.py
DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 127
DEFAULT_VELOCITY = 100
DEFAULT_NOTE_DURATION = 0.01
MIN_NOTE_DURATION = 0.01

# Special token IDs (decoded)
DECODED_EOS_ID = -1
DECODED_INVALID_ID = -2


# ===== Event Codec (from event_codec.py) =====

@dataclasses.dataclass
class EventRange:
    """Range of values for an event type."""
    type: str
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    """Single music event (shift, pitch, velocity, etc.)."""
    type: str
    value: int


class Codec:
    """Encode and decode music events to/from token indices."""

    def __init__(self, max_shift_steps: int, steps_per_second: float,
                 event_ranges: List[EventRange]):
        self.steps_per_second = steps_per_second
        self._shift_range = EventRange(
            type='shift', min_value=0, max_value=max_shift_steps)
        self._event_ranges = [self._shift_range] + event_ranges

    @property
    def num_classes(self) -> int:
        return sum(er.max_value - er.min_value + 1 for er in self._event_ranges)

    @property
    def max_shift_steps(self) -> int:
        return self._shift_range.max_value

    def encode_event(self, event: Event) -> int:
        """Encode an event to a token index."""
        offset = 0
        for er in self._event_ranges:
            if event.type == er.type:
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f'Event value {event.value} not in range '
                        f'[{er.min_value}, {er.max_value}] for type {event.type}')
                return offset + event.value - er.min_value
            offset += er.max_value - er.min_value + 1
        raise ValueError(f'Unknown event type: {event.type}')

    def decode_event_index(self, index: int) -> Event:
        """Decode a token index to an Event."""
        offset = 0
        for er in self._event_ranges:
            if offset <= index <= offset + er.max_value - er.min_value:
                return Event(
                    type=er.type, value=er.min_value + index - offset)
            offset += er.max_value - er.min_value + 1
        raise ValueError(f'Unknown event index: {index}')

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Return [min_id, max_id] for an event type."""
        offset = 0
        for er in self._event_ranges:
            if event_type == er.type:
                return offset, offset + (er.max_value - er.min_value)
            offset += er.max_value - er.min_value + 1
        raise ValueError(f'Unknown event type: {event_type}')


# ===== Vocabulary (from vocabularies.py, minimal) =====

def velocity_to_bin(velocity: int, num_velocity_bins: int) -> int:
    """Convert MIDI velocity to velocity bin."""
    if velocity == 0:
        return 0
    else:
        return math.ceil(num_velocity_bins * velocity / 127)


def bin_to_velocity(velocity_bin: int, num_velocity_bins: int) -> int:
    """Convert velocity bin to MIDI velocity."""
    if velocity_bin == 0:
        return 0
    else:
        return int(127 * velocity_bin / num_velocity_bins)


def build_codec(num_velocity_bins: int = 1) -> Codec:
    """Build MT3 event codec."""
    event_ranges = [
        EventRange('pitch', 0, 127),  # MIDI pitches
        EventRange('velocity', 0, num_velocity_bins),
        EventRange('tie', 0, 0),
        EventRange('program', 0, 127),  # MIDI programs
        EventRange('drum', 0, 127),  # Drum pitches
    ]
    return Codec(
        max_shift_steps=DEFAULT_STEPS_PER_SECOND * DEFAULT_MAX_SHIFT_SECONDS,
        steps_per_second=DEFAULT_STEPS_PER_SECOND,
        event_ranges=event_ranges
    )


# ===== Run-Length Encoding (from run_length_encoding.py) =====

def decode_events(
    state: Any,
    tokens: np.ndarray,
    start_time: float,
    max_time: Optional[float],
    codec: Codec,
    decode_event_fn: Callable[[Any, float, Event, Codec], None],
) -> Tuple[int, int]:
    """Decode token sequence into events, updating state.

    Returns:
        (invalid_events, dropped_events)
    """
    invalid_events = 0
    dropped_events = 0
    cur_steps = 0
    cur_time = start_time

    for token_idx, token in enumerate(tokens):
        try:
            event = codec.decode_event_index(int(token))
        except ValueError:
            invalid_events += 1
            continue

        if event.type == 'shift':
            cur_steps += event.value
            cur_time = start_time + cur_steps / codec.steps_per_second
            if max_time and cur_time > max_time:
                dropped_events = len(tokens) - token_idx
                break
        else:
            cur_steps = 0
            try:
                decode_event_fn(state, cur_time, event, codec)
            except ValueError:
                invalid_events += 1
                continue

    return invalid_events, dropped_events


# ===== Note Sequences (from note_sequences.py) =====

# Simplified NoteSequence (avoiding protobuf dependency)
@dataclasses.dataclass
class Note:
    """MIDI note."""
    pitch: int
    start_time: float
    end_time: float
    velocity: int
    program: int = 0
    is_drum: bool = False


@dataclasses.dataclass
class NoteSequence:
    """Collection of MIDI notes."""
    notes: List[Note] = dataclasses.field(default_factory=list)
    total_time: float = 0.0
    ticks_per_quarter: int = 220


@dataclasses.dataclass
class NoteDecodingState:
    """State machine for decoding tokens to notes."""
    current_time: float = 0.0
    current_velocity: int = DEFAULT_VELOCITY
    current_program: int = 0
    active_pitches: MutableMapping[Tuple[int, int], Tuple[float, int]] = dataclasses.field(
        default_factory=dict)
    tied_pitches: MutableSet[Tuple[int, int]] = dataclasses.field(default_factory=set)
    is_tie_section: bool = False
    note_sequence: NoteSequence = dataclasses.field(default_factory=NoteSequence)


def decode_note_event(
    state: NoteDecodingState,
    time: float,
    event: Event,
    codec: Codec
) -> None:
    """Process note event and update decoding state."""
    if time < state.current_time:
        raise ValueError(f'event time < current time: {time} < {state.current_time}')

    state.current_time = time

    if event.type == 'pitch':
        pitch = event.value
        if state.is_tie_section:
            # Tied pitch (note continuation from previous segment)
            if (pitch, state.current_program) not in state.active_pitches:
                raise ValueError(f'inactive pitch/program in tie section: {pitch}/{state.current_program}')
            if (pitch, state.current_program) in state.tied_pitches:
                raise ValueError(f'pitch/program already tied: {pitch}/{state.current_program}')
            state.tied_pitches.add((pitch, state.current_program))
        elif state.current_velocity == 0:
            # Note offset (end note)
            if (pitch, state.current_program) not in state.active_pitches:
                raise ValueError(f'note-off for inactive pitch/program: {pitch}/{state.current_program}')
            onset_time, onset_velocity = state.active_pitches.pop((pitch, state.current_program))
            end_time = max(time, onset_time + MIN_NOTE_DURATION)
            state.note_sequence.notes.append(Note(
                pitch=pitch,
                start_time=onset_time,
                end_time=end_time,
                velocity=onset_velocity,
                program=state.current_program,
                is_drum=False
            ))
            state.note_sequence.total_time = max(state.note_sequence.total_time, end_time)
        else:
            # Note onset (start note)
            if (pitch, state.current_program) in state.active_pitches:
                # Pitch already active - end previous note and start new one
                onset_time, onset_velocity = state.active_pitches.pop((pitch, state.current_program))
                end_time = max(time, onset_time + MIN_NOTE_DURATION)
                state.note_sequence.notes.append(Note(
                    pitch=pitch,
                    start_time=onset_time,
                    end_time=end_time,
                    velocity=onset_velocity,
                    program=state.current_program,
                    is_drum=False
                ))
            state.active_pitches[(pitch, state.current_program)] = (time, state.current_velocity)

    elif event.type == 'velocity':
        # Set velocity for next note
        num_velocity_bins = codec.event_type_range('velocity')[1] - codec.event_type_range('velocity')[0]
        state.current_velocity = bin_to_velocity(event.value, num_velocity_bins)

    elif event.type == 'program':
        # Set program (instrument) for next note
        state.current_program = event.value

    elif event.type == 'drum':
        # Drum note (uses drum channel)
        pitch = event.value
        if state.current_velocity == 0:
            # Drum offset
            if (pitch, 9) not in state.active_pitches:  # Channel 9 is drums
                raise ValueError(f'drum note-off for inactive pitch: {pitch}')
            onset_time, onset_velocity = state.active_pitches.pop((pitch, 9))
            end_time = max(time, onset_time + MIN_NOTE_DURATION)
            state.note_sequence.notes.append(Note(
                pitch=pitch,
                start_time=onset_time,
                end_time=end_time,
                velocity=onset_velocity,
                program=0,
                is_drum=True
            ))
            state.note_sequence.total_time = max(state.note_sequence.total_time, end_time)
        else:
            # Drum onset
            if (pitch, 9) in state.active_pitches:
                onset_time, onset_velocity = state.active_pitches.pop((pitch, 9))
                end_time = max(time, onset_time + MIN_NOTE_DURATION)
                state.note_sequence.notes.append(Note(
                    pitch=pitch,
                    start_time=onset_time,
                    end_time=end_time,
                    velocity=onset_velocity,
                    program=0,
                    is_drum=True
                ))
            state.active_pitches[(pitch, 9)] = (time, state.current_velocity)

    elif event.type == 'tie':
        # End tie section and close notes that were not re-declared as tied.
        if not state.is_tie_section:
            raise ValueError('tie section end event when not in tie section')

        for (pitch, program), (onset_time, onset_velocity) in list(state.active_pitches.items()):
            if (pitch, program) in state.tied_pitches:
                continue

            end_time = max(time, onset_time + MIN_NOTE_DURATION)
            state.note_sequence.notes.append(Note(
                pitch=pitch,
                start_time=onset_time,
                end_time=end_time,
                velocity=onset_velocity,
                program=0 if program == 9 else program,
                is_drum=(program == 9)
            ))
            state.note_sequence.total_time = max(state.note_sequence.total_time, end_time)
            state.active_pitches.pop((pitch, program), None)

        state.tied_pitches.clear()
        state.is_tie_section = False

    else:
        raise ValueError(f'Unknown event type: {event.type}')


def begin_tied_pitches_section(state: NoteDecodingState) -> None:
    """Enter tie section at beginning of segment."""
    state.tied_pitches = set()
    state.is_tie_section = True


def flush_note_decoding_state(state: NoteDecodingState) -> NoteSequence:
    """Flush any remaining active notes and return NoteSequence."""
    # End any still-active notes
    for (pitch, program), (onset_time, onset_velocity) in state.active_pitches.items():
        end_time = max(state.current_time, onset_time + MIN_NOTE_DURATION)
        state.note_sequence.notes.append(Note(
            pitch=pitch,
            start_time=onset_time,
            end_time=end_time,
            velocity=onset_velocity,
            program=program,
            is_drum=(program == 9 or pitch in state.tied_pitches)
        ))
        state.note_sequence.total_time = max(state.note_sequence.total_time, end_time)

    return state.note_sequence


# ===== High-Level Decoding (from metrics_utils.py) =====

def decode_and_combine_predictions(
    predictions: Sequence[Mapping[str, Any]],
    codec: Codec,
) -> Tuple[NoteSequence, int, int]:
    """Decode predictions into a combined NoteSequence.

    Args:
        predictions: List of dicts with 'est_tokens' and 'start_time'
        codec: Event codec for decoding

    Returns:
        (note_sequence, total_invalid_events, total_dropped_events)
    """
    # Initialize decoding state
    state = NoteDecodingState()
    total_invalid_events = 0
    total_dropped_events = 0

    # Sort predictions by start time
    sorted_predictions = sorted(predictions, key=lambda p: p['start_time'])

    for pred in sorted_predictions:
        # Begin new segment
        begin_tied_pitches_section(state)

        # Decode tokens for this segment
        tokens = pred['est_tokens']
        start_time = pred['start_time']

        invalid, dropped = decode_events(
            state=state,
            tokens=tokens,
            start_time=start_time,
            max_time=None,
            codec=codec,
            decode_event_fn=decode_note_event,
        )

        total_invalid_events += invalid
        total_dropped_events += dropped

        # Exit tie section
        state.is_tie_section = False

    # Flush remaining notes
    note_sequence = flush_note_decoding_state(state)

    return note_sequence, total_invalid_events, total_dropped_events
