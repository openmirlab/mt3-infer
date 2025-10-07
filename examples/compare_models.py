#!/usr/bin/env python3
"""Compare MR-MT3 and YourMT3 on real audio."""

import time
from mt3_infer.adapters.mr_mt3 import MRMT3Adapter
from mt3_infer.adapters.yourmt3 import YourMT3Adapter
from mt3_infer.utils.audio import load_audio

print('=== Model Comparison: MR-MT3 vs YourMT3 ===')
print()

# Load test audio
print('Loading audio...')
audio, sr = load_audio('assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav', sr=16000)
print(f'✓ Audio: {len(audio)} samples ({len(audio)/sr:.1f}s) at {sr} Hz')
print()

# Test MR-MT3
print('--- Testing MR-MT3 ---')
start = time.time()
mr_adapter = MRMT3Adapter()
mr_adapter.load_model('refs/mr-mt3/pretrained/mt3.pth', device='cpu')
load_time_mr = time.time() - start

start = time.time()
mr_midi = mr_adapter.transcribe(audio, sr)
infer_time_mr = time.time() - start

# Count MR-MT3 notes
mr_notes = sum(
    1 for track in mr_midi.tracks
    for msg in track
    if msg.type == 'note_on' and msg.velocity > 0
)
mr_midi.save('comparison_mr_mt3.mid')

print(f'Load time: {load_time_mr:.1f}s')
print(f'Inference time: {infer_time_mr:.1f}s')
print(f'Notes detected: {mr_notes}')
print(f'MIDI saved: comparison_mr_mt3.mid')
print()

# Test YourMT3
print('--- Testing YourMT3 ---')
start = time.time()
yt_adapter = YourMT3Adapter(model_key='ymt3plus')
yt_adapter.load_model(device='cpu')
load_time_yt = time.time() - start

start = time.time()
yt_midi = yt_adapter.transcribe(audio, sr)
infer_time_yt = time.time() - start

# Count YourMT3 notes
yt_notes = sum(
    1 for track in yt_midi.tracks
    for msg in track
    if msg.type == 'note_on' and msg.velocity > 0
)
yt_midi.save('comparison_yourmt3.mid')

print(f'Load time: {load_time_yt:.1f}s')
print(f'Inference time: {infer_time_yt:.1f}s')
print(f'Notes detected: {yt_notes}')
print(f'MIDI saved: comparison_yourmt3.mid')
print()

# Comparison
print('=== Summary ===')
print()
print(f'{"Model":<15} {"Notes":<10} {"Inference Time":<20} {"Load Time"}')
print('-' * 60)
print(f'{"MR-MT3":<15} {mr_notes:<10} {infer_time_mr:<20.1f}s {load_time_mr:.1f}s')
print(f'{"YourMT3":<15} {yt_notes:<10} {infer_time_yt:<20.1f}s {load_time_yt:.1f}s')
print()
print(f'Note difference: {abs(mr_notes - yt_notes)} notes ({abs(mr_notes - yt_notes)/max(mr_notes, yt_notes)*100:.1f}%)')

if infer_time_mr < infer_time_yt:
    speedup = infer_time_yt / infer_time_mr
    print(f'MR-MT3 is {speedup:.1f}x faster')
else:
    speedup = infer_time_mr / infer_time_yt
    print(f'YourMT3 is {speedup:.1f}x faster')
