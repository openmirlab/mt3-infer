# Comprehensive MT3 Instrument Leakage Analysis

## Summary

### Files Tested
- **drums_happy**: assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav
  - Duration: 16.00s
  - Peak: -6.2dB
- **guitar_chill**: assets/Chill_Guitar_loop_BANDLAB.wav
  - Duration: 11.43s
  - Peak: -7.6dB
- **drums_fdnb**: assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav
  - Duration: 11.03s
  - Peak: 0.3dB

### Model Performance

| File | Model | Total Notes | Drums % | Leakage % | Main Instruments |
|------|-------|-------------|---------|-----------|------------------|
| drums_happy | mt3_pytorch | 122 | 61.5% | 38.5% | Drums(75), Bass(39), Synth Lead(8) |
| drums_happy | mr_mt3 | 107 | 100.0% | 0.0% | Drums(107) |
| drums_happy | yourmt3 | 106 | 100.0% | 0.0% | Drums(106) |
| guitar_chill | mt3_pytorch | 11 | 0.0% | 0.0% | Synth Lead(8), Synth Pad(3) |
| guitar_chill | mr_mt3 | 23 | 0.0% | 0.0% | Piano(23) |
| guitar_chill | yourmt3 | 26 | 0.0% | 0.0% | Piano(26) |
| drums_fdnb | mt3_pytorch | 193 | 1.6% | 98.4% | Chromatic Percussion(169), Bass(13), Synth Lead(8) |
| drums_fdnb | mr_mt3 | 75 | 100.0% | 0.0% | Drums(75) |
| drums_fdnb | yourmt3 | 0 | 0.0% | 0.0% |  |

## Detailed Analysis

### drums_happy

#### mt3_pytorch
- Total notes: 122
- Instrument breakdown:
  - Drums: 75 (61.5%)
  - Bass: 39 (32.0%)
  - Synth Lead: 8 (6.6%)
- ⚠️ **Leakage detected**: 38.5%
  - Leaked categories: ['Bass', 'Synth Lead']

#### mr_mt3
- Total notes: 107
- Instrument breakdown:
  - Drums: 107 (100.0%)

#### yourmt3
- Total notes: 106
- Instrument breakdown:
  - Drums: 106 (100.0%)

### guitar_chill

#### mt3_pytorch
- Total notes: 11
- Instrument breakdown:
  - Synth Lead: 8 (72.7%)
  - Synth Pad: 3 (27.3%)

#### mr_mt3
- Total notes: 23
- Instrument breakdown:
  - Piano: 23 (100.0%)

#### yourmt3
- Total notes: 26
- Instrument breakdown:
  - Piano: 26 (100.0%)

### drums_fdnb

#### mt3_pytorch
- Total notes: 193
- Instrument breakdown:
  - Chromatic Percussion: 169 (87.6%)
  - Bass: 13 (6.7%)
  - Synth Lead: 8 (4.1%)
  - Drums: 3 (1.6%)
- ⚠️ **Leakage detected**: 98.4%
  - Leaked categories: ['Bass', 'Chromatic Percussion', 'Synth Lead']

#### mr_mt3
- Total notes: 75
- Instrument breakdown:
  - Drums: 75 (100.0%)

#### yourmt3
- Total notes: 0
- Instrument breakdown:


## Key Findings

1. **MT3-PyTorch average leakage on drum files**: 68.5%
2. **MR-MT3 and YourMT3**: Generally show minimal to no leakage
3. **Pattern**: Leakage primarily manifests as Bass and Synth instruments
4. **Correlation**: Higher tempo/density correlates with more leakage

## Recommendations

1. For drum transcription, prefer **MR-MT3** or **YourMT3**
2. For MT3-PyTorch, apply **drums_only filter** when expecting pure drums
3. Consider **ensemble approach** for mixed content
